"""Tests that the ``ChainBuilder.from_description`` planning prompt is rendered
from the live step-type catalog (``mmar_carl.meta``) and gives the planner the
guidance it needs to produce pydantic-valid output on the first attempt.

Live benchmark observation: the planner naturally emits ``reasoning_questions``
as a JSON array; pydantic rejected it. Mitigations: (a) list→bullet coercion in
``LLMStepDescription``, (b) retry-with-feedback, and (c) the planner prompt
telling the LLM the types up front. The prompt is now auto-generated, so it
covers *every* registered step type (not just llm/tool/memory/transform) and
updates itself when a new step type is added.
"""

from __future__ import annotations

import pytest

from mmar_carl.chain import ChainBuilder
from mmar_carl.meta import get_step_catalog
from mmar_carl.models.llm_client_base import LLMClientBase


VALID_RESPONSE = '{"steps": [{"number": 1, "step_type": "llm", "title": "t", "aim": "x"}]}'


class _SpyPlanner(LLMClientBase):
    def __init__(self, response: str = VALID_RESPONSE) -> None:
        self._response = response
        self.prompts: list[str] = []

    async def get_response(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self._response

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        return await self.get_response(prompt)


async def _prompt_for(task: str = "t", **kwargs) -> str:
    planner = _SpyPlanner()
    await ChainBuilder.from_description(task, planner, **kwargs)
    return planner.prompts[0]


# ---------------------------------------------------------------------------
# Auto-updating coverage — the whole point of generating from the catalog.
# ---------------------------------------------------------------------------


class TestAutoUpdatingCoverage:
    @pytest.mark.asyncio
    async def test_prompt_documents_every_registered_step_type(self) -> None:
        prompt = await _prompt_for()
        for name in get_step_catalog().names():
            assert name in prompt, f"step type {name!r} missing from planner prompt"

    @pytest.mark.asyncio
    async def test_prompt_covers_types_beyond_the_legacy_four(self) -> None:
        """Old hand-written prompt only documented llm/tool/memory/transform."""
        prompt = await _prompt_for()
        for name in ("supervisor", "debate", "agent_skill", "evaluation", "parallel_sampling"):
            assert name in prompt


# ---------------------------------------------------------------------------
# Cross-cutting guidance the planner needs to emit valid JSON.
# ---------------------------------------------------------------------------


class TestPlanningPromptGuidance:
    @pytest.mark.asyncio
    async def test_prompt_warns_string_fields_not_arrays(self) -> None:
        prompt = await _prompt_for()
        assert "NOT an array" in prompt
        for field in ("aim", "reasoning_questions", "stage_action", "example_reasoning"):
            assert field in prompt

    @pytest.mark.asyncio
    async def test_prompt_forbids_triggered_by(self) -> None:
        prompt = await _prompt_for()
        assert "triggered_by" in prompt
        assert "do NOT use" in prompt

    @pytest.mark.asyncio
    async def test_prompt_marks_step_config_as_object(self) -> None:
        prompt = await _prompt_for()
        assert "step_config" in prompt
        assert "NOT array" in prompt

    @pytest.mark.asyncio
    async def test_prompt_declares_reference_grammar(self) -> None:
        prompt = await _prompt_for()
        assert "$history" in prompt
        assert "$memory" in prompt

    @pytest.mark.asyncio
    async def test_prompt_respects_max_steps(self) -> None:
        prompt = await _prompt_for(max_steps=4)
        assert "at most 4 steps" in prompt

    @pytest.mark.asyncio
    async def test_prompt_includes_task_and_rules(self) -> None:
        prompt = await _prompt_for("My task here")
        assert "My task here" in prompt
        assert "Rules" in prompt
        assert "step_type" in prompt


# ---------------------------------------------------------------------------
# Available tools / skills are surfaced when provided.
# ---------------------------------------------------------------------------


class TestAvailableToolsAndSkills:
    @pytest.mark.asyncio
    async def test_available_tools_listed(self) -> None:
        prompt = await _prompt_for(available_tools=["web_search", "calculator"])
        assert "web_search" in prompt
        assert "calculator" in prompt

    @pytest.mark.asyncio
    async def test_available_skills_listed(self) -> None:
        prompt = await _prompt_for(available_skills=["pdf", "pptx"])
        assert "pdf" in prompt


# ---------------------------------------------------------------------------
# Retry path inherits the same generated prompt.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retry_prompts_inherit_generated_prompt() -> None:
    planner = _SpyPlanner(response="bad json")
    with pytest.raises(ValueError):
        await ChainBuilder.from_description("t", planner, max_retries=2)
    assert len(planner.prompts) == 3  # 3 attempts
    for p in planner.prompts:
        assert "Available step types" in p


# ---------------------------------------------------------------------------
# End-to-end: a correctly-typed response still produces a valid chain.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_planner_with_canned_typed_response_still_succeeds() -> None:
    canned = """
    {
      "steps": [
        {
          "number": 1,
          "title": "Plan",
          "step_type": "llm",
          "dependencies": [],
          "aim": "Identify the problem.",
          "reasoning_questions": "What are we solving?",
          "stage_action": "Enumerate quantities and operations.",
          "example_reasoning": "12 - 5 = 7"
        }
      ]
    }
    """
    planner = _SpyPlanner(response=canned)
    chain = await ChainBuilder.from_description("solve a problem", planner)
    assert len(chain.steps) == 1
    assert chain.steps[0].aim == "Identify the problem."
    assert chain.steps[0].reasoning_questions == "What are we solving?"
