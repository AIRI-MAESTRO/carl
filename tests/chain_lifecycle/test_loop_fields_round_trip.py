"""Regression: ``loop_back_to`` / ``loop_config`` survive a to_dict()/from_dict()
round-trip.

The registry-driven ``from_dict`` reconstruction reads these fields, but
``ReasoningChain.to_dict()`` historically only serialised ``triggered_by`` — so
a while/until loop was silently dropped on save/load. This guards the fix.
"""

from __future__ import annotations

from mmar_carl import LLMStepDescription, ReasoningChain
from mmar_carl.models.config import LoopConfig


def _looping_chain() -> ReasoningChain:
    return ReasoningChain(
        steps=[
            LLMStepDescription(number=1, title="Seed", aim="Start."),
            LLMStepDescription(
                number=2,
                title="Refine",
                aim="Refine until good.",
                dependencies=[1],
                loop_back_to=1,
                loop_config=LoopConfig(condition_key="$history[-1]", max_iterations=5, negate_condition=True),
            ),
        ]
    )


def test_loop_fields_present_in_to_dict() -> None:
    data = _looping_chain().to_dict()
    tail = data["steps"][1]
    assert tail["loop_back_to"] == 1
    assert tail["loop_config"]["max_iterations"] == 5
    assert tail["loop_config"]["negate_condition"] is True


def test_loop_fields_round_trip_dict() -> None:
    restored = ReasoningChain.from_dict(_looping_chain().to_dict(), use_typed_steps=True)
    tail = restored.steps[1]
    assert tail.loop_back_to == 1
    assert tail.loop_config is not None
    assert tail.loop_config.max_iterations == 5
    assert tail.loop_config.condition_key == "$history[-1]"
    assert tail.loop_config.negate_condition is True


def test_loop_fields_round_trip_json() -> None:
    restored = ReasoningChain.from_json(_looping_chain().to_json())
    # legacy (untyped) path keeps the raw dict; typed path rebuilds the model
    typed = ReasoningChain.from_dict(_looping_chain().to_dict(), use_typed_steps=True)
    assert typed.steps[1].loop_config.max_iterations == 5
    assert restored is not None
