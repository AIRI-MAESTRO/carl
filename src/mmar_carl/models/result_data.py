"""Typed views over :attr:`StepExecutionResult.result_data`.

``result_data: dict[str, Any]`` is universal but loose
— CARE's TUI renders per-step-type detail panes (AgentSkill output
files, Debate transcripts, Supervisor decisions, ParallelSampling
candidates) and would rather hold a typed view than poke at a dict.

These Pydantic models are **non-strict** by design: every field is
``Optional[...]`` with a sensible default. The shapes are documented
through field descriptions; extra keys produced by the executor are
preserved in ``extras`` (pydantic ``model_config["extra"] = "allow"``)
so an upstream change adding a key doesn't break callers.

Each typed accessor on :class:`StepExecutionResult` returns ``None``
when the step's :class:`StepType` doesn't match — so CARE can write::

    if (skill := result.as_skill_output()):
        for path in skill.output_files:
            ...

without an isinstance dance.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

# ---------------------------------------------------------------------------
# AgentSkill
# ---------------------------------------------------------------------------


class SkillOutput(BaseModel):
    """Typed view over ``StepExecutionResult.result_data`` for
    :class:`~mmar_carl.AgentSkillStepDescription`.

    Captures the canonical keys emitted by every AgentSkill execution
    mode (LLM, SCRIPT, HYBRID, SUBAGENT, LLM_AGENT). Extras (mode-
    specific keys like ``script_returncode``, ``parsed_output``,
    ``schema_warnings``) are preserved on the model via ``extra="allow"``.
    """

    model_config = ConfigDict(extra="allow")

    skill_name: Optional[str] = Field(
        default=None,
        description="The resolved skill's name (from ``SkillManifest.name``).",
    )
    execution_mode: Optional[str] = Field(
        default=None,
        description=(
            "The AgentSkill execution mode value: "
            "``llm`` / ``script`` / ``hybrid`` / ``subagent`` / ``llm_agent``."
        ),
    )
    output_files: list[dict[str, str]] = Field(
        default_factory=list,
        description=(
            "List of files the skill wrote to ``/workspace/out``. Each "
            "entry has at least ``name`` and ``path`` keys; LLM_AGENT "
            "mode adds ``content_type`` when known."
        ),
    )
    workspace_root: Optional[str] = Field(
        default=None,
        description="Host path to the workspace root (LLM_AGENT mode).",
    )
    persisted_workspace: Optional[str] = Field(
        default=None,
        description=(
            "Host path when ``persist_workspace=True`` keeps the "
            "workspace alive after the step completes (debugging aid)."
        ),
    )
    parsed_output: Optional[dict[str, Any]] = Field(
        default=None,
        description=(
            "When ``output_schema`` is set on the config, the parsed "
            "JSON object that passed schema validation."
        ),
    )
    schema_validated: Optional[bool] = Field(
        default=None,
        description=(
            "``True`` when ``output_schema`` validation succeeded, "
            "``False`` when it failed (and ``schema_warnings`` carries "
            "the errors), ``None`` when no schema was configured."
        ),
    )
    iterations: Optional[int] = Field(
        default=None,
        description="LLM_AGENT: number of tool-call iterations.",
    )
    tool_calls_made: Optional[int] = Field(
        default=None,
        description="LLM_AGENT: total tool calls dispatched.",
    )


# ---------------------------------------------------------------------------
# Debate
# ---------------------------------------------------------------------------


class DebateTurn(BaseModel):
    """One turn in a multi-role debate transcript."""

    model_config = ConfigDict(extra="allow")

    round: int
    role: str
    argument: str


class DebateTranscript(BaseModel):
    """Typed view over Debate step result_data.

    Keys produced by :class:`~mmar_carl.step_executors.DebateStepExecutor`:
    ``verdict``, ``transcript``, ``rounds_executed``, ``role_call_count``,
    ``topic``.
    """

    model_config = ConfigDict(extra="allow")

    verdict: str = Field(
        default="",
        description="The judge's synthesised verdict (final answer).",
    )
    transcript: list[DebateTurn] = Field(
        default_factory=list,
        description=(
            "Round-robin list of ``{round, role, argument}`` entries in "
            "execution order."
        ),
    )
    rounds_executed: int = Field(
        default=0,
        description="Number of debate rounds that completed.",
    )
    role_call_count: int = Field(
        default=0,
        description="Total LLM calls made across all role turns "
                    "(``len(roles) × rounds_executed`` on success).",
    )
    topic: str = Field(
        default="",
        description="The resolved task / topic the debate was about.",
    )


# ---------------------------------------------------------------------------
# Supervisor
# ---------------------------------------------------------------------------


class SupervisorDecision(BaseModel):
    """Typed view over Supervisor step result_data.

    Keys produced by
    :class:`~mmar_carl.step_executors.SupervisorStepExecutor`:
    ``agent_selected``, ``routing_reply``, ``sub_result``,
    ``sub_chain_success``, ``steps_executed``.
    """

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    agent_selected: Optional[str] = Field(
        default=None,
        description="Name of the agent the LLM routed to.",
    )
    routing_reply: Optional[str] = Field(
        default=None,
        description="Raw text the supervisor LLM emitted for the "
                    "routing decision (useful for debugging matches).",
    )
    sub_result: Optional[Any] = Field(
        default=None,
        description=(
            "Full :class:`ReasoningResult` of the chosen sub-chain. "
            "Typed as ``Any`` to avoid a circular import."
        ),
    )
    sub_chain_success: Optional[bool] = Field(
        default=None,
        description="Whether the sub-chain reported success.",
    )
    steps_executed: Optional[int] = Field(
        default=None,
        description="Number of steps the chosen sub-chain ran.",
    )


# ---------------------------------------------------------------------------
# ParallelSampling
# ---------------------------------------------------------------------------


class ParallelSamples(BaseModel):
    """Typed view over ParallelSampling step result_data.

    Keys produced by
    :class:`~mmar_carl.step_executors.ParallelSamplingStepExecutor`:
    ``n_samples``, ``n_successes``, ``aggregation``, ``candidates``.
    """

    model_config = ConfigDict(extra="allow")

    n_samples: int = Field(
        default=0,
        description="Number of independent samples requested.",
    )
    n_successes: int = Field(
        default=0,
        description="Number of samples that completed without error.",
    )
    aggregation: Optional[str] = Field(
        default=None,
        description=(
            "Aggregation strategy used: ``majority_vote`` / "
            "``best_of_n`` / ``llm_judge``."
        ),
    )
    candidates: list[str] = Field(
        default_factory=list,
        description=(
            "Raw response text from each successful sample. The winner "
            "(per ``aggregation``) is in ``StepExecutionResult.result``."
        ),
    )


# ---------------------------------------------------------------------------
# WaitStep
# ---------------------------------------------------------------------------


class WaitOutcome(BaseModel):
    """Canonical structured result emitted by a successful WaitStep."""

    trigger: Literal["after", "at", "event"]
    elapsed_seconds: float = Field(ge=0.0)
    condition_index: Optional[int] = Field(
        default=None,
        ge=0,
        description="Winning AnyOf condition index; None for a direct leaf condition.",
    )
    seconds: Optional[float] = Field(default=None, ge=0.0)
    timestamp: Optional[str] = None
    name: Optional[str] = None
    payload: Any = None


# ---------------------------------------------------------------------------
# MapStep
# ---------------------------------------------------------------------------


class MapItemStatus(StrEnum):
    """Terminal lifecycle state for one MapStep item."""

    COMPLETED = "completed"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"


class MapItemOutcome(BaseModel):
    """Canonical outcome for one input item, indexed by input order."""

    index: int = Field(ge=0)
    status: MapItemStatus
    success: bool
    output: Any = None
    error_type: str | None = None
    error_message: str | None = None
    execution_time: float = Field(ge=0.0)

    @model_validator(mode="after")
    def _validate_terminal_state(self) -> MapItemOutcome:
        completed = self.status == MapItemStatus.COMPLETED
        if self.success != completed:
            raise ValueError("success must be true exactly for completed items")
        if completed:
            if self.error_type is not None or self.error_message is not None:
                raise ValueError("completed items cannot contain error details")
        else:
            if self.output is not None:
                raise ValueError("non-completed items cannot contain output")
            if not self.error_type or not self.error_message:
                raise ValueError("non-completed items require error details")
        return self


class MapOutcome(BaseModel):
    """Ordered aggregate emitted after all MapStep items reach a terminal state."""

    items: list[MapItemOutcome]
    total_items: int = Field(ge=0)
    completed_items: int = Field(ge=0)
    failed_items: int = Field(ge=0)
    timed_out_items: int = Field(ge=0)
    cancelled_items: int = Field(ge=0)
    max_concurrency: int = Field(ge=1)
    enforcement_gaps: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_aggregate(self) -> MapOutcome:
        if self.total_items != len(self.items):
            raise ValueError("total_items must equal len(items)")
        if [item.index for item in self.items] != list(range(self.total_items)):
            raise ValueError("items must be in contiguous input order")
        expected = {
            MapItemStatus.COMPLETED: self.completed_items,
            MapItemStatus.FAILED: self.failed_items,
            MapItemStatus.TIMED_OUT: self.timed_out_items,
            MapItemStatus.CANCELLED: self.cancelled_items,
        }
        for status, count in expected.items():
            actual = sum(item.status == status for item in self.items)
            if count != actual:
                raise ValueError(f"{status.value}_items count does not match items")
        if sum(expected.values()) != self.total_items:
            raise ValueError("terminal item counts must sum to total_items")
        return self


__all__ = [
    "DebateTranscript",
    "DebateTurn",
    "MapItemOutcome",
    "MapItemStatus",
    "MapOutcome",
    "ParallelSamples",
    "SkillOutput",
    "SupervisorDecision",
    "WaitOutcome",
]
