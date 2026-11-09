"""Both halves of the ``$``-reference contract for CONDITIONAL steps.

The catalog advertises ``step_config.branches.*.condition`` as a reference
field (``reference_fields("conditional")``). Two things follow from that claim,
and each used to be broken in a way that produced no error message at all:

1. **Execution must honour it.** A condition carrying a ``$`` reference used to
   reach simpleeval verbatim, where the ``$`` is a ``SyntaxError`` that the
   executor swallowed into ``False``. Downstream measurement (an evolutionary
   search whose stop gate is ``value >= $memory.evo.target_fitness``, with
   ``best=0.95`` against ``target=0.9``): the branch did not route, the step
   still reported ``success=True``, and the result was byte-identical to a
   genuinely unmet threshold. Every run silently stopped stopping early.

2. **Loading many of them must be able to keep quiet.** The reference
   diagnostics ``ReasoningChain`` emits are for a human writing one chain; a
   program generating thousands drowned in them (~1,300 per suite run
   downstream) and reached for message-text filtering instead.
"""

from __future__ import annotations

import warnings
from typing import Any

from mmar_carl import (
    ConditionalBranch,
    ConditionalStepConfig,
    ConditionalStepDescription,
    Language,
    ReasoningChain,
    ReasoningContext,
    ToolStepConfig,
    ToolStepDescription,
)
from mmar_carl.meta.catalog import (
    quiet_reference_warnings,
    reference_fields,
    validate_and_build,
)
from mmar_carl.step_executors import ConditionalStepExecutor
from tests.mocks import MockLLMClient


def _context(**metadata: Any) -> ReasoningContext:
    ctx = ReasoningContext(outer_context="", api=MockLLMClient(), model="test", language=Language.ENGLISH)
    ctx.metadata.update(metadata)
    return ctx


def _gate(condition: str, *, on: str = "$memory.evo.best") -> ConditionalStepDescription:
    """A one-branch CONDITIONAL: step 2 when ``condition`` holds, else step 3."""
    return ConditionalStepDescription(
        number=1,
        title="Stop check",
        dependencies=[],
        config=ConditionalStepConfig(
            condition_context_key=on,
            branches=[ConditionalBranch(condition=condition, next_step=2)],
            default_step=3,
        ),
    )


async def _route(condition: str, ctx: ReasoningContext, *, on: str = "$memory.evo.best") -> Any:
    """Execute the gate alone and return its ``result_data``."""
    result = await ConditionalStepExecutor().execute(_gate(condition, on=on), ctx)
    assert result.success, result.error_message
    return result.result_data


# --------------------------------------------------------------------------- #
# 1. References inside branch conditions
# --------------------------------------------------------------------------- #


async def test_branch_condition_reads_a_memory_reference() -> None:
    """The measured downstream failure: a threshold kept in memory instead of
    baked into the condition text.

    Without reference resolution both calls below route to the default step, so
    a search that has already beaten its target keeps burning generations and
    the only symptom is a slow run that looks like a search regression.
    """
    ctx = _context()
    ctx.memory_write("target_fitness", 0.9, namespace="evo")

    ctx.memory_write("best", 0.95, namespace="evo")
    assert (await _route("value >= $memory.evo.target_fitness", ctx))["next_step"] == 2

    ctx.memory_write("best", 0.50, namespace="evo")
    assert (await _route("value >= $memory.evo.target_fitness", ctx))["next_step"] == 3


async def test_the_catalog_claim_is_true_for_every_advertised_path() -> None:
    """Ties the executor's behaviour to the catalog entry a spec generator reads.

    An LLM planner is handed ``reference_fields("conditional")`` and will emit
    ``$`` refs in whatever it lists. If the list and the executor drift apart
    again, the generated chains are wrong and nothing raises.
    """
    assert "step_config.branches.*.condition" in reference_fields("conditional")

    ctx = _context(expected="admin", threshold=3)
    ctx.memory_write("best", "admin", namespace="evo")
    assert (await _route("equals:$metadata.expected", ctx))["next_step"] == 2
    assert (await _route("len(value) > $metadata.threshold", ctx))["next_step"] == 2


async def test_a_resolved_reference_keeps_its_python_type() -> None:
    """References are bound as names, never spliced into the expression text.

    A spliced string operand would be re-parsed as code: this metadata value is
    a string that happens to read like an expression, and splicing would make
    ``value == $metadata.needle`` compare against ``"ab"`` instead of against
    the string the caller stored.
    """
    ctx = _context(needle="'a' + 'b'")
    ctx.memory_write("best", "'a' + 'b'", namespace="evo")
    assert (await _route("value == $metadata.needle", ctx))["next_step"] == 2


async def test_a_dollar_inside_a_string_literal_is_not_a_reference() -> None:
    """Quoted operands are compared, not resolved.

    ``value == '$HOME/bin'`` asks whether the value is that literal path
    template — a shell-ish string is the realistic case, since a reference
    token is exactly ``$`` followed by a name. Substituting inside the quotes
    would compare against an unrelated string instead, and the condition would
    just start being false.
    """
    ctx = _context()
    ctx.memory_write("best", "$HOME/bin", namespace="evo")
    assert (await _route("value == '$HOME/bin'", ctx))["next_step"] == 2


async def test_only_a_whole_prefix_argument_is_a_reference() -> None:
    """A prefix argument is literal text; only an argument that is *entirely*
    one reference gets resolved.

    ``equals:$HOME/bin`` compares against that path template verbatim — the
    shape a config check actually uses. Resolving embedded tokens instead would
    reduce it to ``/bin`` (no such metadata key) and the comparison would start
    coming out false with nothing logged. The regex case is the same rule seen
    from the other side: ``$`` in ``matches:`` is an end anchor, not a token.
    """
    ctx = _context()

    ctx.memory_write("best", "$HOME/bin", namespace="evo")
    assert (await _route("equals:$HOME/bin", ctx))["next_step"] == 2

    ctx.memory_write("best", "404", namespace="evo")
    assert (await _route(r"matches:^\d{3}$", ctx))["next_step"] == 2


async def test_an_unevaluatable_condition_is_reported_not_silently_false() -> None:
    """A condition that cannot be evaluated must not be indistinguishable from
    one that evaluated ``False``.

    Both route to ``default_step`` — one broken branch should not fail a chain
    whose other branches still route — but only the broken one warns and lands
    in ``result_data['condition_errors']``. That difference is the whole reason
    the original bug survived: a typo'd condition and an unmet threshold
    produced byte-identical results.
    """
    ctx = _context()
    ctx.memory_write("best", 0.95, namespace="evo")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        broken = await _route("undefined_func(value)", ctx)
    assert broken["next_step"] == 3
    assert "undefined_func(value)" in broken["condition_errors"][0]
    assert len(caught) == 1 and "could not be evaluated" in str(caught[0].message)

    unmet = await _route("value >= 2.0", ctx)
    assert unmet["next_step"] == 3
    assert "condition_errors" not in unmet


async def test_a_reference_condition_routes_through_a_real_chain() -> None:
    """End-to-end through the DAG, not just the executor: the branch step must
    actually run.

    The executor only reports ``next_step``; routing is the DAGExecutor's job
    (it skips the non-chosen branches), so an executor-level assertion alone
    would not catch routing that reports one thing and runs another.
    """
    steps = [
        _gate("value >= $memory.evo.target_fitness"),
        ToolStepDescription(
            number=2,
            title="Stop",
            dependencies=[1],
            config=ToolStepConfig(tool_name="stop", input_mapping={}),
        ),
        ToolStepDescription(
            number=3,
            title="Continue",
            dependencies=[1],
            config=ToolStepConfig(tool_name="keep_going", input_mapping={}),
        ),
    ]
    ctx = _context()
    ctx.memory_write("best", 0.95, namespace="evo")
    ctx.memory_write("target_fitness", 0.9, namespace="evo")
    ran: list[str] = []
    ctx.register_tool("stop", lambda: ran.append("stop") or "stop")
    ctx.register_tool("keep_going", lambda: ran.append("keep_going") or "keep_going")

    with quiet_reference_warnings():
        chain = ReasoningChain(steps=steps, max_workers=1)
    result = await chain.execute_async(ctx)

    assert result.success, result.get_final_output()
    assert ran == ["stop"]


# --------------------------------------------------------------------------- #
# 2. Quiet loading
# --------------------------------------------------------------------------- #

_NOISY_SPEC: dict[str, Any] = {
    "steps": [
        {
            "number": 1,
            "title": "Route",
            "step_type": "conditional",
            "dependencies": [],
            "step_config": {
                "branches": [{"condition": "nonempty", "next_step": 2}],
                "condition_context_key": "$history[-1]",
            },
        },
        {"number": 2, "title": "Answer", "step_type": "llm", "dependencies": [1], "aim": "answer"},
    ]
}


def _unrelated_warning() -> None:
    """Stands in for every *other* ``UserWarning`` a suite raises."""
    warnings.warn("an unrelated library warning", UserWarning, stacklevel=2)


def test_hand_written_chains_still_get_the_reference_warning() -> None:
    """The default must stay loud.

    The warning's whole audience is the person writing one chain by hand and
    referencing history from step 1; a quiet mode that turned itself on by
    default would take that away from them.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        validate_and_build(_NOISY_SPEC)
    assert any("$history[-1]" in str(w.message) for w in caught)


def test_quiet_loading_changes_what_a_chain_says_not_what_it_is() -> None:
    """A quiet build must produce the same chain and the same report.

    If ``quiet`` ever started skipping validation rather than just the warnings,
    a bulk loader would stop being told about genuinely invalid specs — the
    failure mode is worse than the noise it was meant to fix.
    """
    def references(seen: list[Any]) -> list[str]:
        return [str(w.message) for w in seen if "reference" in str(w.message)]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loud_report, loud_chain = validate_and_build(_NOISY_SPEC)
        loud = references(caught)
        quiet_report, quiet_chain = validate_and_build(_NOISY_SPEC, quiet=True)
        total = references(caught)

    # Count *reference* warnings, not every warning in the window. `record=True`
    # captures whatever else the interpreter happens to emit here — a GC
    # finalizer from an earlier async test ("unclosed event loop") landed
    # between these two builds and failed this on suite timing alone, saying
    # nothing about quiet mode. That unrelated traffic is not unowned: keeping
    # it at one warning is the whole subject of
    # `test_quiet_mode_does_not_make_unrelated_warnings_repeat` below, which
    # measures it under the "warn once" filter this assert cannot see anyway.
    assert loud and total == loud, f"quiet build emitted {total[len(loud):]}"
    assert loud_chain.to_dict() == quiet_chain.to_dict()
    assert loud_report.model_dump() == quiet_report.model_dump()


def test_quiet_mode_does_not_make_unrelated_warnings_repeat() -> None:
    """Pins the mechanism, because the obvious implementation is a trap.

    Wrapping each build in ``warnings.catch_warnings()`` mutates the global
    filter list, which invalidates Python's per-location "warn once" registry.
    Measured here over 200 builds: that approach makes the unrelated warning
    below fire 200 times instead of once — a caller silencing one warning would
    get every other warning in the process repeated, which is how the noise
    problem this fixes came back three-fold downstream.
    """
    fired = {"reference": 0, "unrelated": 0}

    def count(message: Any, *_args: Any, **_kwargs: Any) -> None:
        fired["reference" if "reference" in str(message) else "unrelated"] += 1

    # `simplefilter("default")` is the per-location "warn once" mode this is
    # about; pytest's own capture runs "always", which would hide the effect.
    with warnings.catch_warnings():
        warnings.simplefilter("default")
        warnings.showwarning = count  # type: ignore[assignment]
        for _ in range(200):
            with quiet_reference_warnings():
                _unrelated_warning()
                validate_and_build(_NOISY_SPEC)

    assert fired["reference"] == 0, "quiet mode let the reference warning through"
    # Bounded, not exactly 1, and the bound is the whole point: the trap this
    # pins makes the count 200 (one per build), so anything in single digits
    # proves the registry is NOT being invalidated per iteration.
    #
    # Exactly-1 is not achievable in a shared process. `warnings` keys its
    # "warn once" registry on the CALLER's location (stacklevel=2 → the loop
    # line here) and invalidates every registry whenever the global filter list
    # is mutated by anyone. A first-use lazy import inside `validate_and_build`
    # bumps that version once, so the unrelated warning fires a second time —
    # measured at 2 in a full-suite run and 1 when this file runs alone. Pinning
    # 1 made the test order-dependent, which is a worse property than the one
    # decimal place of precision it bought.
    assert fired["unrelated"] <= 5, (
        f"the unrelated warning fired {fired['unrelated']} times over 200 builds — "
        "quiet mode is invalidating the warn-once registry, so silencing one "
        "warning repeats every other warning in the process")


def test_the_loud_method_comes_back_after_nesting_and_after_an_exception() -> None:
    """Quiet mode swaps a method on ``ReasoningChain``; leaking that swap would
    silence the warning for the rest of the process — including for the human
    the warning exists for. Both exits have to restore it."""

    def warned() -> int:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            validate_and_build(_NOISY_SPEC)
        return len(caught)

    before = warned()
    assert before > 0

    with quiet_reference_warnings():
        with quiet_reference_warnings():
            assert warned() == 0
        assert warned() == 0, "the inner exit must not restore the outer context's silence"
    assert warned() == before

    try:
        with quiet_reference_warnings():
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert warned() == before
