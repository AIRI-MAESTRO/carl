"""A branch condition whose ``$`` reference names something nobody wrote.

X4 taught ``ConditionalStepExecutor`` to resolve ``$`` refs. This file covers
the half X4 left open: a ref that does not resolve — a typo, a renamed
namespace, a key written by a step that did not run.

Measured on the pre-fix executor (2026-07-28), one branch, ``default_step=3``:

    value >= $memory.evo.no_such_key   -> step 3, warned  (TypeError vs None)
    value == $memory.evo.no_such_key   -> step 3, silent
    $memory.evo.no_such_key is None    -> step 2, silent   (routes!)
    contains:$memory.evo.no_such_key   -> step 2, silent   (routes!)
    startswith:/endswith:/matches:     -> step 2, silent   (routes!)
    equals:$memory.evo.no_such_key     -> step 3, silent

So the failure was never uniformly "silently False": an unresolved reference
becomes ``None``, which the prefix forms stringify to ``""``, and ``""`` is
*in* every string, *starts* every string and ``re.search("")`` matches every
string. Those four branches matched unconditionally. A moderation gate written
as ``contains:$memory.policy.banned_phrase`` therefore sent **every** input down
the banned branch the day the namespace was renamed, and reported success.

The contract now: a condition that cannot be evaluated warns once, lands in
``result_data['condition_errors']``, and is treated as not matched — never as a
value. Routing is deliberately unchanged for conditions that legitimately
evaluate ``False``.
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
from mmar_carl.meta.catalog import quiet_reference_warnings
from mmar_carl.step_executors import ConditionalStepExecutor
from tests.mocks import MockLLMClient

MISSING = "$memory.evo.no_such_key"


def _context() -> ReasoningContext:
    """A context with ``$memory.evo.best`` set and nothing else in that namespace."""
    ctx = ReasoningContext(outer_context="", api=MockLLMClient(), model="test", language=Language.ENGLISH)
    ctx.memory_write("best", 0.95, namespace="evo")
    return ctx


def _gate(*conditions: str, on: str = "$memory.evo.best") -> ConditionalStepDescription:
    """One CONDITIONAL routing to 2, 4, 5... in order, defaulting to step 3."""
    return ConditionalStepDescription(
        number=1,
        title="Stop check",
        dependencies=[],
        config=ConditionalStepConfig(
            condition_context_key=on,
            branches=[
                ConditionalBranch(condition=c, next_step=2 if i == 0 else 3 + i)
                for i, c in enumerate(conditions)
            ],
            default_step=3,
        ),
    )


async def _route(
    condition: str,
    ctx: ReasoningContext,
    *,
    executor: ConditionalStepExecutor | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Execute the gate alone; return ``(result_data, warning texts)``.

    A fresh executor per call by default: the warn-once cache lives on the
    executor instance, and sharing one would make these tests order-dependent.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = await (executor or ConditionalStepExecutor()).execute(_gate(condition), ctx)
    assert result.success, result.error_message
    return result.result_data or {}, [str(w.message) for w in caught]


# --------------------------------------------------------------------------- #
# Cannot-evaluate is not False
# --------------------------------------------------------------------------- #


async def test_an_unresolvable_reference_is_reported_not_treated_as_a_value() -> None:
    """The measured downstream failure this closes: an evolutionary policy whose
    stop gate reads ``value >= $memory.evo.target_fitness`` against a *renamed*
    namespace. The gate stops firing, the run goes to budget every time, and the
    only symptom is a uniform drop in the measured *speed* component — which
    reads as a real finding about engine quality, not as a bug.

    Both the broken gate and a genuinely unmet threshold still route to
    ``default_step`` (one bad branch must not fail the chain), so the whole
    signal is that only one of them carries ``condition_errors``.
    """
    ctx = _context()

    broken, warned = await _route(f"value >= {MISSING}", ctx)
    assert broken["next_step"] == 3
    assert MISSING in broken["condition_errors"][0]
    assert len(warned) == 1 and "is not set in this context" in warned[0]

    unmet, quiet = await _route("value >= 2.0", ctx)
    assert unmet["next_step"] == 3
    assert "condition_errors" not in unmet and quiet == []


async def test_every_operator_that_swallowed_a_missing_reference_now_reports_it() -> None:
    """Each of these evaluated without complaint before, and the operator decided
    the outcome: ``==`` gave False, ``is None`` gave True, ``bool()`` gave False.

    Enumerated rather than tested through one representative condition because
    the pre-fix behaviour was *operator-dependent* — a fix that only handles the
    comparison it was reported against would leave ``is None`` still routing.
    """
    ctx = _context()
    for condition in (
        f"value == {MISSING}",
        f"{MISSING} is None",
        f"bool({MISSING})",
        f"value >= 0.1 and {MISSING} > 0",
        "value >= $metadata.nope.deeper",
    ):
        data, warned = await _route(condition, ctx)
        assert data["next_step"] == 3, condition
        assert data.get("condition_errors"), condition
        assert len(warned) == 1, condition


async def test_a_prefix_argument_that_does_not_resolve_no_longer_matches_everything() -> None:
    """The dangerous direction, and the reason this file exists at all: these
    four did not fail silently *to False*, they matched unconditionally.

    ``$memory.policy.banned_phrase`` resolving to nothing became ``""``, and
    every string contains, starts with and regex-matches the empty string. A
    filter written that way routed every input down its "matched" branch the day
    the key was renamed — a false positive on every single execution, reported
    as success.
    """
    ctx = _context()
    for condition in (
        f"contains:{MISSING}",
        f"startswith:{MISSING}",
        f"endswith:{MISSING}",
        f"matches:{MISSING}",
        f"equals:{MISSING}",
    ):
        data, warned = await _route(condition, ctx)
        assert data["next_step"] == 3, f"{condition} still matched"
        assert data.get("condition_errors"), condition
        assert len(warned) == 1, condition


async def test_a_reference_whose_lookup_raises_is_handled_like_a_missing_one() -> None:
    """``$history[-9]`` on a two-entry history raises ``IndexError`` inside
    resolution, which happens before the evaluator's own try/except: it used to
    escape as a step *failure* (``success=False``), so the same class of mistake
    — a bad index versus a bad key — either killed the chain or said nothing,
    depending on which one you made.

    Both are now the same thing: not evaluatable, reported, not matched.
    """
    ctx = _context()
    ctx.history.extend(["step one", "step two"])

    data, warned = await _route("value == $history[-9]", ctx)
    assert data["next_step"] == 3
    assert "IndexError" in data["condition_errors"][0]
    assert len(warned) == 1


async def test_a_broken_branch_does_not_stop_the_branches_after_it() -> None:
    """A chain with three routes must keep routing when one condition is typo'd.

    Raising out of the step instead would turn a single mistyped key into a dead
    chain, which is why this executor reports rather than raises — the reason
    the reported-not-raised design is worth its extra surface.
    """
    ctx = _context()
    executor = ConditionalStepExecutor()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = await executor.execute(_gate(f"value >= {MISSING}", "value >= 0.9"), ctx)

    assert result.success
    data = result.result_data or {}
    assert data["next_step"] == 4, "the second branch still had to route"
    assert data["matched_condition"] == "value >= 0.9"
    assert len(data["condition_errors"]) == 1
    assert len(caught) == 1


# --------------------------------------------------------------------------- #
# What must NOT change
# --------------------------------------------------------------------------- #


async def test_a_stored_none_is_a_value_not_a_missing_reference() -> None:
    """This is the distinction the sentinel buys, and the reason "treat a ref
    that resolves to None as unevaluatable" was not good enough.

    A step that legitimately wrote ``None`` — an optional result, a cleared flag
    — is a context the condition *can* be evaluated against. Collapsing it into
    "missing" would make ``$memory.evo.cleared is None`` unusable and warn on a
    perfectly healthy run, which is how a diagnostic becomes noise and gets
    filtered out.
    """
    ctx = _context()
    ctx.memory_write("cleared", None, namespace="evo")

    data, warned = await _route("$memory.evo.cleared is None", ctx)
    assert data["next_step"] == 2
    assert "condition_errors" not in data and warned == []


async def test_conditions_without_references_are_untouched() -> None:
    """Backwards compatibility, stated as the shapes that existed before any of
    this: no ``$`` token means no resolution, so a False condition must stay a
    plain, quiet False and a True one must stay True.

    ``equals:$HOME/bin`` is in the list because it is the one literal that *looks*
    like a reference: only an argument that is entirely one reference token is
    resolved, so this compares against the path template verbatim.
    """
    ctx = _context()
    ctx.memory_write("best", "approved", namespace="evo")

    expected = {
        "contains:appro": 2,
        "equals:approved": 2,
        "startswith:app": 2,
        "endswith:ved": 2,
        r"matches:^appr": 2,
        "nonempty": 2,
        "value == 'approved'": 2,
        "len(value) > 3": 2,
        "contains:denied": 3,
        "equals:denied": 3,
        "empty": 3,
        "value == 'denied'": 3,
        "equals:$HOME/bin": 3,
    }
    for condition, next_step in expected.items():
        data, warned = await _route(condition, ctx)
        assert data["next_step"] == next_step, condition
        assert "condition_errors" not in data, condition
        assert warned == [], condition


async def test_a_healthy_decision_keeps_its_exact_result_data_shape() -> None:
    """``condition_errors`` is added only when there are errors.

    Callers (and the run record) compare ``result_data`` of a conditional step;
    an always-present empty list would change every stored artifact for no
    information.
    """
    ctx = _context()
    data, _ = await _route("value >= $memory.evo.threshold", _seeded(ctx))
    assert data == {"next_step": 2, "matched_condition": "value >= $memory.evo.threshold"}


def _seeded(ctx: ReasoningContext) -> ReasoningContext:
    ctx.memory_write("threshold", 0.9, namespace="evo")
    return ctx


# --------------------------------------------------------------------------- #
# Volume
# --------------------------------------------------------------------------- #


async def test_the_warning_is_once_per_failure_not_once_per_execution() -> None:
    """A policy chain runs once per generation; 200 generations of one broken
    stop gate must not be 200 identical lines.

    This project has measured what happens otherwise: one warning grew to 5,983
    of 10,372 log lines and buried 4,389 real ones, at which point the log stops
    being read at all. ``condition_errors`` stays per-execution, so nothing is
    lost for a program — only the human channel is deduplicated.
    """
    ctx = _context()
    executor = ConditionalStepExecutor()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(50):
            result = await executor.execute(_gate(f"value >= {MISSING}"), ctx)
            assert (result.result_data or {}).get("condition_errors"), "data channel deduplicated"

    assert len(caught) == 1


async def test_a_different_failure_in_the_same_step_still_warns() -> None:
    """Deduplication keyed on the step alone would silence the second bug found
    in a step that already has one — the exact way a "quiet the noise" fix turns
    into the next silent failure.
    """
    ctx = _context()
    executor = ConditionalStepExecutor()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        await executor.execute(_gate(f"value >= {MISSING}"), ctx)
        await executor.execute(_gate("value >= $memory.evo.other_typo"), ctx)
        await executor.execute(_gate(f"value >= {MISSING}"), ctx)

    assert len(caught) == 2
    assert "no_such_key" in str(caught[0].message)
    assert "other_typo" in str(caught[1].message)


# --------------------------------------------------------------------------- #
# End to end
# --------------------------------------------------------------------------- #


async def test_a_broken_gate_is_visible_in_the_finished_chain_result() -> None:
    """Through the DAG, not just the executor: the evidence has to survive into
    ``ReasoningResult.step_results``.

    An executor-level assertion alone would not prove a caller can find out. The
    warning is a stream a batch run does not keep; the run record is what it
    keeps, and this is the failure whose whole cost was being invisible after
    the fact.
    """
    steps = [
        _gate(f"value >= {MISSING}"),
        ToolStepDescription(
            number=2, title="Stop", dependencies=[1],
            config=ToolStepConfig(tool_name="stop", input_mapping={}),
        ),
        ToolStepDescription(
            number=3, title="Continue", dependencies=[1],
            config=ToolStepConfig(tool_name="keep_going", input_mapping={}),
        ),
    ]
    ctx = _context()
    ran: list[str] = []
    ctx.register_tool("stop", lambda: ran.append("stop") or "stop")
    ctx.register_tool("keep_going", lambda: ran.append("keep_going") or "keep_going")

    with quiet_reference_warnings():
        chain = ReasoningChain(steps=steps, max_workers=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = await chain.execute_async(ctx)

    assert result.success, result.get_final_output()
    assert ran == ["keep_going"], "a broken gate must fall through, not route"
    gate_result = next(sr for sr in result.step_results if sr.step_number == 1)
    assert MISSING in (gate_result.result_data or {})["condition_errors"][0]
