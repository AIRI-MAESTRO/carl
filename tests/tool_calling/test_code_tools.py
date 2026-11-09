"""Code tools — evolvable tool implementations carried in chain specs.

Covers the full substrate:

1. ``CodeToolLibrary`` — versioning, latest/pinned lookup, merge, dict round-trip.
2. ``validate_code_tool`` / ``compile_code_tool`` — no-execution validation
   (parse, right def, no imports / dunders / banned names, length bound) and
   restricted-namespace compilation.
3. ``ReasoningContext.bind_tool_library`` — opt-in gating (``allow_code_tools``),
   host-tool precedence, ``name`` (latest) + ``name@vN`` (pinned) registration.
4. Chain execution — a spec whose ``metadata["tool_library"]`` carries the tool
   binds and runs it; without the opt-in the TOOL step fails.
5. ``mutate_tool_code`` — appends versions, validates, replays via
   ``apply_records``; ``validate_spec`` rejects broken embedded code.
"""

from __future__ import annotations

import pytest

from mmar_carl import (
    CodeTool,
    CodeToolError,
    CodeToolLibrary,
    ReasoningChain,
    ReasoningContext,
    ToolStepConfig,
    ToolStepDescription,
    apply_records,
    compile_code_tool,
    mutate_tool_code,
    validate_and_build,
    validate_code_tool,
    validate_spec,
)
from mmar_carl.meta.mutations import MutationError

SCALE_V1 = "def scale(x):\n    return x * 2\n"
SCALE_V2 = "def scale(x):\n    return x * 3\n"


def _library() -> CodeToolLibrary:
    lib = CodeToolLibrary()
    lib.add("scale", SCALE_V1, description="scales the input")
    return lib


def _tool_chain_spec() -> dict:
    """A single-TOOL-step chain spec calling the library 'scale' tool."""
    spec = {
        "steps": [
            {
                "number": 1,
                "title": "scale it",
                "step_type": "tool",
                "dependencies": [],
                "step_config": {
                    "tool_name": "scale",
                    "input_mapping": {"x": "$metadata.x"},
                    "parameters": [{"name": "x", "type": "int", "description": "value"}],
                },
            }
        ]
    }
    return _library().attach_to(spec)


# --------------------------------------------------------------------------- #
# Library model
# --------------------------------------------------------------------------- #


def test_add_versions_and_lookup() -> None:
    lib = _library()
    v2 = lib.add("scale", SCALE_V2)
    assert (v2.version, v2.parent_version) == (2, 1)
    assert lib.get("scale").code == SCALE_V2          # latest
    assert lib.get("scale", version=1).code == SCALE_V1  # pinned
    assert lib.get("missing") is None


def test_add_rejects_invalid_code() -> None:
    lib = CodeToolLibrary()
    with pytest.raises(CodeToolError):
        lib.add("f", "def g():\n    return 1\n")  # wrong function name


def test_dict_round_trip_orders_versions() -> None:
    lib = _library()
    lib.add("scale", SCALE_V2)
    data = lib.to_dict()
    # scramble version order to prove from_dict re-sorts
    data["scale"].reverse()
    back = CodeToolLibrary.from_dict(data)
    assert [t.version for t in back.versions("scale")] == [1, 2]
    assert back.get("scale").code == SCALE_V2


def test_merge_prefers_self_on_conflict() -> None:
    a, b = CodeToolLibrary(), CodeToolLibrary()
    a.add("scale", SCALE_V1)
    b.add("scale", SCALE_V2)
    b.add("shift", "def shift(x):\n    return x + 1\n")
    merged = a.merge(b)
    assert merged.get("scale").code == SCALE_V1  # a wins
    assert merged.get("shift") is not None       # b's unique tool comes along


# --------------------------------------------------------------------------- #
# Validation + compilation (restricted namespace)
# --------------------------------------------------------------------------- #


def test_validate_flags_the_escape_hatches() -> None:
    cases = {
        "import os\ndef f():\n    return 1\n": "import",
        "def f():\n    return ().__class__\n": "dunder",
        "def f():\n    return eval('1')\n": "eval",
        "def g():\n    return 1\n": "must define",
    }
    for code, needle in cases.items():
        errors = validate_code_tool(CodeTool(name="f", code=code))
        assert errors and any(needle in e for e in errors), (code, errors)


def test_validate_length_bound() -> None:
    code = "def f():\n    return 1  # " + "x" * 20_000
    assert validate_code_tool(CodeTool(name="f", code=code)) == [
        f"code too long: {len(code)} > 10000"
    ]


def test_compile_and_call() -> None:
    fn = compile_code_tool(CodeTool(name="scale", code=SCALE_V1))
    assert fn(21) == 42


def test_compile_provides_math_and_restricts_namespace() -> None:
    fn = compile_code_tool(CodeTool(name="f", code="def f(x):\n    return math.floor(x)\n"))
    assert fn(2.7) == 2
    # statically caught: `open` is in BANNED_NAMES
    errors = validate_code_tool(CodeTool(name="f", code="def f():\n    return open('/etc/hosts')\n"))
    assert any("open" in e for e in errors)
    # runtime restriction: `hash` isn't banned, but it isn't whitelisted either —
    # the restricted namespace raises NameError when the tool runs.
    sneaky = compile_code_tool(CodeTool(name="f", code="def f(x):\n    return hash(x)\n"))
    with pytest.raises(NameError):
        sneaky(1)


def test_compile_rejects_invalid() -> None:
    with pytest.raises(CodeToolError):
        compile_code_tool(CodeTool(name="f", code="import os\ndef f():\n    return 1\n"))


# --------------------------------------------------------------------------- #
# Context binding — opt-in gate + precedence + pinning
# --------------------------------------------------------------------------- #


def test_bind_requires_opt_in() -> None:
    ctx = ReasoningContext(outer_context="t", api=None)
    with pytest.raises(PermissionError):
        ctx.bind_tool_library(_library())


def test_bind_registers_latest_and_pinned() -> None:
    lib = _library()
    lib.add("scale", SCALE_V2)
    ctx = ReasoningContext(outer_context="t", api=None, allow_code_tools=True)
    ctx.bind_tool_library(lib)
    assert ctx.get_tool("scale")(5) == 15        # latest (v2)
    assert ctx.get_tool("scale@v1")(5) == 10     # pinned
    assert ctx.get_tool("scale@v2")(5) == 15
    assert "code_tool" in ctx.get_tool_tags("scale")


def test_bind_never_shadows_host_tools() -> None:
    ctx = ReasoningContext(outer_context="t", api=None, allow_code_tools=True)
    ctx.register_tool("scale", lambda x: "host")
    ctx.bind_tool_library(_library())
    assert ctx.get_tool("scale")(1) == "host"     # host wins
    assert ctx.get_tool("scale@v1")(5) == 10      # pinned alias still available


# --------------------------------------------------------------------------- #
# Chain execution — the library rides in spec metadata
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_chain_executes_spec_carried_tool() -> None:
    report, chain = validate_and_build(_tool_chain_spec())
    assert report.valid and chain is not None
    ctx = ReasoningContext(outer_context="t", api=None, allow_code_tools=True)
    ctx.metadata["x"] = 21
    result = await chain.execute_async(ctx)
    assert result.success
    assert "42" in result.step_results[0].result


@pytest.mark.asyncio
async def test_chain_without_opt_in_does_not_execute_code() -> None:
    report, chain = validate_and_build(_tool_chain_spec())
    assert report.valid
    ctx = ReasoningContext(outer_context="t", api=None)  # no allow_code_tools
    ctx.metadata["x"] = 21
    result = await chain.execute_async(ctx)
    assert not result.success  # tool never registered → step fails, no code ran


@pytest.mark.asyncio
async def test_pinned_step_keeps_old_version_while_plain_tracks_latest() -> None:
    spec = _tool_chain_spec()
    spec, _ = mutate_tool_code(spec, "scale", SCALE_V2)
    spec["steps"][0]["step_config"]["tool_name"] = "scale@v1"
    report, chain = validate_and_build(spec)
    assert report.valid
    ctx = ReasoningContext(outer_context="t", api=None, allow_code_tools=True)
    ctx.metadata["x"] = 5
    result = await chain.execute_async(ctx)
    assert result.success
    assert "10" in result.step_results[0].result  # v1 (×2), not the evolved v2 (×3)


# --------------------------------------------------------------------------- #
# Mutation primitive + spec validation
# --------------------------------------------------------------------------- #


def test_mutate_tool_code_appends_version_and_replays() -> None:
    spec = _tool_chain_spec()
    new, rec = mutate_tool_code(spec, "scale", SCALE_V2)
    assert rec.op == "tool_code" and rec.before == SCALE_V1 and rec.after == SCALE_V2
    versions = new["metadata"]["tool_library"]["scale"]
    assert [v["version"] for v in versions] == [1, 2]
    assert spec["metadata"]["tool_library"]["scale"][-1]["version"] == 1  # parent untouched
    replayed = apply_records(spec, [rec])
    assert replayed["metadata"]["tool_library"]["scale"][-1]["code"] == SCALE_V2


def test_mutate_tool_code_guards() -> None:
    spec = _tool_chain_spec()
    with pytest.raises(MutationError):
        mutate_tool_code(spec, "missing", SCALE_V2)          # unknown tool
    with pytest.raises(MutationError):
        mutate_tool_code(spec, "scale", SCALE_V1)            # no-op
    with pytest.raises(MutationError):
        mutate_tool_code(spec, "scale", "import os\ndef scale():\n    return 1\n")
    created, rec = mutate_tool_code(spec, "shift", "def shift(x):\n    return x + 1\n", create=True)
    assert rec.before is None
    assert created["metadata"]["tool_library"]["shift"][-1]["version"] == 1


def test_validate_spec_checks_embedded_tool_code() -> None:
    spec = _tool_chain_spec()
    assert validate_spec(spec).valid
    spec["metadata"]["tool_library"]["scale"][0]["code"] = "import os\ndef scale(x):\n    return x\n"
    report = validate_spec(spec)
    assert not report.valid
    assert any(i.code == "invalid_tool_code" for i in report.errors)


def test_library_survives_chain_round_trip() -> None:
    report, chain = validate_and_build(_tool_chain_spec())
    assert report.valid
    again = ReasoningChain.from_dict(chain.to_dict(), use_typed_steps=True)
    assert again.metadata["tool_library"]["scale"][0]["code"] == SCALE_V1


def test_typed_step_chain_serialises_library() -> None:
    """Building a chain object directly (not via spec) keeps the library in metadata."""
    chain = ReasoningChain(
        steps=[
            ToolStepDescription(
                number=1, title="scale it",
                config=ToolStepConfig(tool_name="scale"),
                dependencies=[],
            )
        ],
        metadata={"tool_library": _library().to_dict()},
    )
    spec = chain.to_dict()
    assert validate_spec(spec).valid
    assert spec["metadata"]["tool_library"]["scale"][0]["name"] == "scale"
