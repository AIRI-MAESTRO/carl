"""A serializable pointer to a sub-chain, and the hook that binds it back.

``AgentHandoffStepDescription.sub_chain`` is ``exclude=True`` — a live
``ReasoningChain``, never serialized — so ``from_dict`` injects a placeholder to
make the constructor succeed. That leaves a spec unable to say *which*
sub-agent it wanted, and leaves an unbound handoff indistinguishable from a
bound one until it executes a step titled ``__placeholder__`` and returns a
wrong answer with no error anywhere.

``sub_chain_ref`` carries the pointer across a round trip;
``validate_and_build(resolve_sub_chain=...)`` turns it back into a chain; and a
ref that does not resolve is *reported* rather than left silently holding the
placeholder. The third is the one that matters — the first two are plumbing.
"""

from typing import Any, Optional

import pytest
from mmar_carl import ReasoningChain, validate_and_build
from mmar_carl.models.steps import LLMStepDescription

REF = "memory-card:9f1b6a52"


def _handoff_spec(ref: Optional[str] = REF) -> dict[str, Any]:
    step: dict[str, Any] = {
        "number": 1,
        "title": "delegate",
        "step_type": "agent_handoff",
        "dependencies": [],
        "step_config": {"input_mapping": {}, "output_memory_key": "out"},
    }
    if ref is not None:
        step["sub_chain_ref"] = ref
    return {"steps": [step]}


def _named(name: str) -> ReasoningChain:
    return ReasoningChain(steps=[LLMStepDescription(number=1, title=name, aim="a")])


def test_the_ref_survives_a_round_trip() -> None:
    """The property `sub_chain` structurally cannot have: build → to_dict →
    build again, and the spec still names the sub-agent it wanted."""
    _report, chain = validate_and_build(_handoff_spec())
    dumped = chain.to_dict()

    assert dumped["steps"][0]["sub_chain_ref"] == REF

    _report2, rebuilt = validate_and_build(dumped, resolve_sub_chain=lambda ref: _named(f"real<{ref}>"))
    assert rebuilt.steps[0].sub_chain.steps[0].title == f"real<{REF}>"


def test_the_hook_receives_the_ref_and_its_chain_is_bound() -> None:
    seen: list[str] = []

    def resolve(ref: str) -> ReasoningChain:
        seen.append(ref)
        return _named("resolved")

    report, chain = validate_and_build(_handoff_spec(), resolve_sub_chain=resolve)

    assert seen == [REF]
    assert chain.steps[0].sub_chain.steps[0].title == "resolved"
    assert not report.warnings, [w.message for w in report.warnings]


def test_an_unresolved_ref_is_reported_and_not_left_silent() -> None:
    """The whole reason this feature is worth having.

    Without the report, a spec naming a sub-agent that could not be found builds
    green, executes, and runs the placeholder — a wrong answer that looks exactly
    like a right one. The chain is still returned (inspecting a spec you cannot
    resolve is legitimate), but the caller is told, by step number.
    """
    report, chain = validate_and_build(_handoff_spec())

    assert report.valid, "an unbindable ref is a warning, not a broken spec"
    codes = [(w.code, w.step, w.field) for w in report.warnings]
    assert ("unresolved_sub_chain_ref", 1, "sub_chain_ref") in codes, codes
    assert chain.steps[0].sub_chain.steps[0].title == "__placeholder__", (
        "the placeholder must still be there — the warning is the only difference")


def test_a_hook_returning_none_is_as_loud_as_no_hook_at_all() -> None:
    """A store that simply does not have the card is the common case, and it
    must not read as success. The message distinguishes the two so an operator
    knows whether to pass a hook or to go looking for the entity."""
    no_hook, _ = validate_and_build(_handoff_spec())
    missed, _ = validate_and_build(_handoff_spec(), resolve_sub_chain=lambda _ref: None)

    assert len(missed.warnings) == len(no_hook.warnings) == 1
    assert "hook returned None" in missed.warnings[0].message
    assert "no resolve_sub_chain hook" in no_hook.warnings[0].message


def test_a_raising_hook_is_a_report_not_a_crash() -> None:
    """Resolution is a store lookup, so it can fail for reasons that have
    nothing to do with the spec. One unreachable sub-agent must not deny the
    caller a chain, or a transient outage becomes a build error."""

    def resolve(_ref: str) -> ReasoningChain:
        raise ConnectionError("store unreachable")

    report, chain = validate_and_build(_handoff_spec(), resolve_sub_chain=resolve)

    assert chain is not None and report.valid
    assert "ConnectionError" in report.warnings[0].message


def test_one_bad_ref_does_not_block_the_others() -> None:
    """Collected, not raised on the first failure: a ten-agent chain with one
    missing card should bind the nine and name the one."""
    spec: dict[str, Any] = {"steps": []}
    for number, ref in enumerate(["good-a", "missing", "good-b"], start=1):
        spec["steps"].append({
            "number": number, "title": f"s{number}", "step_type": "agent_handoff",
            "dependencies": [] if number == 1 else [number - 1],
            "sub_chain_ref": ref,
            "step_config": {"input_mapping": {}, "output_memory_key": f"out{number}"},
        })

    report, chain = validate_and_build(
        spec, resolve_sub_chain=lambda ref: None if ref == "missing" else _named(ref))

    assert [w.step for w in report.warnings] == [2]
    assert chain.steps[0].sub_chain.steps[0].title == "good-a"
    assert chain.steps[2].sub_chain.steps[0].title == "good-b"


def test_a_handoff_without_a_ref_behaves_exactly_as_before() -> None:
    """Opt-in: the field defaults to None, so every existing spec builds
    byte-identically and gains no warning. Without this the feature would
    re-diagnose every chain in every caller's suite."""
    report, chain = validate_and_build(_handoff_spec(ref=None))

    assert not report.warnings
    assert chain.steps[0].sub_chain.steps[0].title == "__placeholder__"
    assert "sub_chain_ref" not in chain.to_dict()["steps"][0]


def test_other_step_types_do_not_grow_the_key() -> None:
    """`to_dict` emits it only when set, so an LLM step's serialized form is
    unchanged — otherwise every spec digest in every downstream project moves."""
    _report, chain = validate_and_build(
        {"steps": [{"number": 1, "title": "t", "step_type": "llm", "aim": "a", "dependencies": []}]})

    assert "sub_chain_ref" not in chain.to_dict()["steps"][0]


@pytest.mark.parametrize("ref", ["", None])
def test_an_empty_ref_is_not_a_ref(ref: Optional[str]) -> None:
    """Neither an absent nor an empty pointer should trigger resolution — an
    empty string would otherwise be handed to a store as a lookup key."""
    called: list[str] = []
    report, _chain = validate_and_build(
        _handoff_spec(ref=ref), resolve_sub_chain=lambda r: called.append(r) or _named("x"))

    assert called == []
    assert not report.warnings
