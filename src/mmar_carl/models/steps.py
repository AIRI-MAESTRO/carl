"""
Step description classes for CARL reasoning system.
"""

from typing import Any, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .agent_skill import AgentSkillExecutionMode, AgentSkillStepConfig  # noqa: F401
from .config import (
    AgentHandoffStepConfig,
    AgentStepConfig,
    ClaudeCodeStepConfig,
    CodeStepConfig,
    CodexStepConfig,
    CommandPlanStepConfig,
    CommandStepConfig,
    ConditionalStepConfig,
    ContextQuery,
    DebateStepConfig,
    EvaluationStepConfig,
    HumanInputStepConfig,
    LLMStepConfig,
    LoopConfig,
    MapStepConfig,
    MCPResourceStepConfig,
    MCPStepConfig,
    MemoryStepConfig,
    ParallelSamplingAggregation,  # noqa: F401 — re-exported for convenience
    ParallelSamplingStepConfig,
    ShellSessionStepConfig,
    StepCache,
    StepConfig,
    StructuredOutputStepConfig,
    SupervisorStepConfig,
    ToolDiscoveryStepConfig,
    ToolStepConfig,
    TransformStepConfig,
    WaitStepConfig,
)
from .enums import StepType


class StepDescriptionBase(BaseModel):
    """
    Abstract base class for all step descriptions.

    All step types share these common fields and methods.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="before")
    @classmethod
    def reject_mismatched_wire_step_type(cls, value: Any) -> Any:
        """Keep untagged ``AnyStepDescription`` unions type-safe on raw dicts."""

        if cls is StepDescriptionBase or not isinstance(value, dict):
            return value
        raw_type = value.get("step_type")
        if raw_type is None:
            return value
        try:
            supplied = StepType(raw_type)
        except (TypeError, ValueError):
            raise ValueError(f"unknown step_type {raw_type!r}") from None
        expected = cls.model_construct().step_type
        if supplied != expected:
            raise ValueError(
                f"step_type {supplied.value!r} does not match {cls.__name__} "
                f"({expected.value!r})"
            )
        return value

    # === Core Fields (required for all step types) ===
    number: int = Field(..., description="Step number in the sequence")
    title: str = Field(..., description="Human-readable title of the step")
    dependencies: list[int] = Field(default_factory=list, description="List of step numbers this step depends on")
    triggered_by: list[str] = Field(
        default_factory=list,
        description=(
            "Optional list of event names that gate this step's execution. "
            "Used alongside (or instead of) ``dependencies`` for name-based "
            "DAG edges. A step becomes ready only when ALL listed events have "
            "been emitted via ``context.emit_event(name, payload)`` AND its "
            "numeric ``dependencies`` are satisfied. Event payloads can be "
            "read via the ``$event.<name>`` reference syntax."
        ),
    )
    checkpoint: bool = Field(default=False, description="Mark this step as a RE-PLAN rollback checkpoint")
    checkpoint_name: str | None = Field(default=None, description="Optional checkpoint name")
    replan_enabled: bool | None = Field(
        default=None,
        description="Optional per-step RE-PLAN override (None = chain policy default)",
    )
    metrics: list[Any] = Field(
        default_factory=list,
        description="List of MetricBase instances to evaluate after step execution",
        exclude=True,
    )

    # Loop-back fields (optional — only needed on the "tail" step of a loop body)
    loop_back_to: Optional[int] = Field(
        default=None,
        description=(
            "Step number to loop back to after this step completes successfully. "
            "Together with loop_config, enables cyclic iteration within the chain. "
            "The loop body is the range [loop_back_to, this_step_number] inclusive."
        ),
    )
    loop_config: Optional[LoopConfig] = Field(
        default=None,
        description=(
            "Loop configuration controlling the loop condition and budget guard. "
            "Required when loop_back_to is set."
        ),
    )

    # Step result caching (memoization)
    cache: Optional[StepCache] = Field(
        default=None,
        exclude=True,
        description=(
            "Optional memoization config. When set, the DAGExecutor checks a per-executor "
            "in-memory cache before running the step. On a hit the cached result is returned "
            "immediately. See StepCache for TTL and custom key_fn options."
        ),
    )

    # Step type is determined by the concrete class
    @property
    def step_type(self) -> StepType:
        """Get the step type. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement step_type")

    def depends_on(self, step_number: int) -> bool:
        """Check if this step depends on a given step number."""
        return step_number in self.dependencies

    def has_dependencies(self) -> bool:
        """Check if this step has any dependencies."""
        return len(self.dependencies) > 0

    def is_llm_step(self) -> bool:
        """Check if this is an LLM reasoning step."""
        return self.step_type == StepType.LLM

    def is_tool_step(self) -> bool:
        """Check if this is a tool execution step."""
        return self.step_type == StepType.TOOL

    def is_mcp_step(self) -> bool:
        """Check if this is an MCP protocol step."""
        return self.step_type == StepType.MCP

    def is_memory_step(self) -> bool:
        """Check if this is a memory operation step."""
        return self.step_type == StepType.MEMORY

    def is_transform_step(self) -> bool:
        """Check if this is a data transformation step."""
        return self.step_type == StepType.TRANSFORM

    def is_conditional_step(self) -> bool:
        """Check if this is a conditional branching step."""
        return self.step_type == StepType.CONDITIONAL

    def is_structured_output_step(self) -> bool:
        """Check if this is a structured output generation step."""
        return self.step_type == StepType.STRUCTURED_OUTPUT

    # For serialization - subclasses should provide step_config if applicable
    @property
    def step_config(self) -> Optional[StepConfig]:
        """Get step-specific configuration. Override in subclasses."""
        return None

    def get_llm_field(self, field_name: str, default: str = "") -> str:
        """
        Safely get an LLM-specific field value.

        Works for both typed LLMStepDescription and legacy StepDescription.
        Returns default for non-LLM step types.
        """
        return getattr(self, field_name, default)


class LLMStepDescription(StepDescriptionBase):
    """
    LLM reasoning step description.

    This is the default step type for chain-of-thought reasoning with LLM.

    Supports per-step LLM configuration via the llm_config field:
        ```python
        LLMStepDescription(
            number=1,
            title="Complex Analysis",
            aim="Perform deep analysis",
            llm_config=LLMStepConfig(model="anthropic/claude-3.5-sonnet")
        )
        ```
    """

    # LLM-specific fields
    aim: str = Field(default="", description="Primary objective of this step")
    reasoning_questions: str = Field(default="", description="Key questions to answer")
    step_context_queries: list[ContextQuery | str] = Field(
        default_factory=list,
        description="List of queries to extract relevant context from outer_context (RAG-like)",
    )
    stage_action: str = Field(default="", description="Specific action to perform")
    example_reasoning: str = Field(default="", description="Example of expert reasoning")

    # Per-step LLM configuration override
    llm_config: Optional[LLMStepConfig] = Field(
        default=None,
        description="Optional LLM configuration override for this step (model, temperature, etc.)",
    )

    # Per-step retry override
    retry_max: Optional[int] = Field(
        default=None,
        description="Override retry attempts for this step (None = use context default)",
    )

    # Per-step timeout override
    timeout: Optional[float] = Field(
        default=None,
        gt=0,
        description="Timeout for this step in seconds (None = use chain default)",
    )

    @property
    def step_type(self) -> StepType:
        return StepType.LLM

    @field_validator(
        "aim", "reasoning_questions", "stage_action", "example_reasoning", mode="before"
    )
    @classmethod
    def _coerce_str_or_list(cls, value: Any) -> Any:
        """Accept str | list[str|number|...] and coerce lists to a newline-joined string.

        LLM planners (e.g. ``ChainBuilder.from_description``) frequently emit
        enumerated questions / actions as JSON arrays. Reject-on-array forces
        callers to retry; instead, coerce to ``"- item1\n- item2\n..."`` so
        the chain can execute.
        """
        if isinstance(value, list):
            return "\n".join(f"- {item}" for item in value)
        return value

    @model_validator(mode="after")
    def validate_llm_fields(self) -> "LLMStepDescription":
        """Validate that LLM step has required fields."""
        if not self.aim:
            raise ValueError("LLM steps require 'aim' to be set")
        return self

    def model_dump(self, **kwargs) -> dict[str, Any]:
        """Dump model to dict including step_type property."""
        data = super().model_dump(**kwargs)
        data["step_type"] = self.step_type.value  # Convert enum to string
        return data


class AgentStepDescription(StepDescriptionBase):
    """Bounded ReAct loop over an explicit host-tool allowlist."""

    config: AgentStepConfig = Field(..., description="Agent loop configuration")
    llm_config: Optional[LLMStepConfig] = Field(
        default=None,
        description="Optional LLM configuration override for this step.",
    )

    @property
    def step_type(self) -> StepType:
        return StepType.AGENT

    @property
    def step_config(self) -> AgentStepConfig:
        return self.config

    def model_dump(self, **kwargs) -> dict[str, Any]:
        data = super().model_dump(**kwargs)
        data["step_type"] = self.step_type.value
        return data


class ClaudeCodeStepDescription(StepDescriptionBase):
    """Delegate one DAG node to a headless Claude Code CLI agent."""

    config: ClaudeCodeStepConfig = Field(..., description="Claude Code CLI configuration")

    @property
    def step_type(self) -> StepType:
        return StepType.CLAUDE_CODE

    @property
    def step_config(self) -> ClaudeCodeStepConfig:
        return self.config

    @model_validator(mode="after")
    def reject_side_effect_cache(self) -> "ClaudeCodeStepDescription":
        if self.cache is not None:
            raise ValueError(
                "ClaudeCodeStep cannot be cached because the agent may inspect or modify workspace state"
            )
        return self

    def model_dump(self, **kwargs) -> dict[str, Any]:
        data = super().model_dump(**kwargs)
        data["step_type"] = self.step_type.value
        return data


class CodexStepDescription(StepDescriptionBase):
    """Delegate one DAG node to a local Codex agent through the Codex SDK."""

    config: CodexStepConfig = Field(..., description="Codex delegation configuration")

    @property
    def step_type(self) -> StepType:
        return StepType.CODEX

    @property
    def step_config(self) -> CodexStepConfig:
        return self.config

    @model_validator(mode="after")
    def reject_agent_cache(self) -> "CodexStepDescription":
        if self.cache is not None:
            raise ValueError(
                "CodexStep cannot be cached because a Codex turn may inspect or modify workspace state"
            )
        return self

    def model_dump(self, **kwargs) -> dict[str, Any]:
        data = super().model_dump(**kwargs)
        data["step_type"] = self.step_type.value
        return data


class CodeStepDescription(StepDescriptionBase):
    """Execute exact generated Python through a host-authorized strict sandbox."""

    config: CodeStepConfig = Field(..., description="Generated-code execution configuration")

    @property
    def step_type(self) -> StepType:
        return StepType.CODE

    @property
    def step_config(self) -> CodeStepConfig:
        return self.config

    def model_dump(self, **kwargs) -> dict[str, Any]:
        data = super().model_dump(**kwargs)
        data["step_type"] = self.step_type.value
        return data


class WaitStepDescription(StepDescriptionBase):
    """Self-contained asynchronous wait over a timer or named event."""

    config: WaitStepConfig = Field(..., description="Wait condition configuration")

    @property
    def step_type(self) -> StepType:
        return StepType.WAIT

    @property
    def step_config(self) -> WaitStepConfig:
        return self.config

    def model_dump(self, **kwargs) -> dict[str, Any]:
        data = super().model_dump(**kwargs)
        data["step_type"] = self.step_type.value
        return data


class MapStepDescription(StepDescriptionBase):
    """Apply one registered tool to every item in a resolved JSON array.

    Results are collected in input order. Item failures and timeouts do not
    stop sibling calls; the step succeeds only when every item completes.
    """

    config: MapStepConfig = Field(..., description="Map step configuration.")

    @property
    def step_type(self) -> StepType:
        return StepType.MAP

    @property
    def step_config(self) -> MapStepConfig:
        return self.config

    def model_dump(self, **kwargs) -> dict[str, Any]:
        data = super().model_dump(**kwargs)
        data["step_type"] = self.step_type.value
        return data


class ToolStepDescription(StepDescriptionBase):
    """
    Tool execution step description.

    Executes external functions/tools registered in the context.
    """

    config: ToolStepConfig = Field(..., description="Tool configuration")

    @property
    def step_type(self) -> StepType:
        return StepType.TOOL

    @property
    def step_config(self) -> ToolStepConfig:
        return self.config

    def model_dump(self, **kwargs) -> dict[str, Any]:
        """Dump model to dict including step_type property."""
        data = super().model_dump(**kwargs)
        data["step_type"] = self.step_type.value  # Convert enum to string
        return data


class MCPStepDescription(StepDescriptionBase):
    """
    MCP (Model Context Protocol) step description.

    Calls tools on MCP servers.
    """

    config: MCPStepConfig = Field(..., description="MCP configuration")

    @property
    def step_type(self) -> StepType:
        return StepType.MCP

    @property
    def step_config(self) -> MCPStepConfig:
        return self.config

    def model_dump(self, **kwargs) -> dict[str, Any]:
        """Dump model to dict including step_type property."""
        data = super().model_dump(**kwargs)
        data["step_type"] = self.step_type.value  # Convert enum to string
        return data


class MCPResourceStepDescription(StepDescriptionBase):
    """
    MCP (Model Context Protocol) resource-reading step description.

    Fetches a named *resource* (read-only data) from an MCP server and
    stores its content in memory + history. Distinct from
    :class:`MCPStepDescription` which invokes *tools* on the server.

    Example::

        MCPResourceStepDescription(
            number=1,
            title="Load API reference",
            config=MCPResourceStepConfig(
                server=MCPServerConfig(server_name="docs", transport="sse",
                                       url="http://docs/sse"),
                resource_uri="docs://api/reference.md",
                output_memory_key="api_docs",
            ),
        )
    """

    config: MCPResourceStepConfig = Field(..., description="MCP resource configuration")

    @property
    def step_type(self) -> StepType:
        return StepType.MCP_RESOURCE

    @property
    def step_config(self) -> MCPResourceStepConfig:
        return self.config

    def model_dump(self, **kwargs) -> dict[str, Any]:
        data = super().model_dump(**kwargs)
        data["step_type"] = self.step_type.value
        return data


class MemoryStepDescription(StepDescriptionBase):
    """
    Memory operation step description.

    Performs read/write/append/delete/list operations on shared memory.
    """

    config: MemoryStepConfig = Field(..., description="Memory operation configuration")

    @property
    def step_type(self) -> StepType:
        return StepType.MEMORY

    @property
    def step_config(self) -> MemoryStepConfig:
        return self.config

    def model_dump(self, **kwargs) -> dict[str, Any]:
        """Dump model to dict including step_type property."""
        data = super().model_dump(**kwargs)
        data["step_type"] = self.step_type.value  # Convert enum to string
        return data


class TransformStepDescription(StepDescriptionBase):
    """
    Data transformation step description.

    Performs data transformations without LLM calls.
    """

    config: TransformStepConfig = Field(..., description="Transform configuration")

    @property
    def step_type(self) -> StepType:
        return StepType.TRANSFORM

    @property
    def step_config(self) -> TransformStepConfig:
        return self.config

    def model_dump(self, **kwargs) -> dict[str, Any]:
        """Dump model to dict including step_type property."""
        data = super().model_dump(**kwargs)
        data["step_type"] = self.step_type.value  # Convert enum to string
        return data


class CommandStepDescription(StepDescriptionBase):
    """
    Argv-based operating-system command step description.

    Executes an OS command through the sandbox ``SkillRuntime``. See
    :class:`~mmar_carl.models.config.CommandStepConfig` for the security model
    (argv commands, runtime isolation, and network-policy reporting).
    """

    config: CommandStepConfig = Field(..., description="OS command configuration")

    @property
    def step_type(self) -> StepType:
        return StepType.COMMAND

    @property
    def step_config(self) -> CommandStepConfig:
        return self.config

    @model_validator(mode="after")
    def reject_side_effect_cache(self) -> "CommandStepDescription":
        if self.cache is not None:
            raise ValueError("CommandStep cannot be cached because command execution may have side effects")
        return self

    def model_dump(self, **kwargs) -> dict[str, Any]:
        """Dump model to dict including step_type property."""
        data = super().model_dump(**kwargs)
        data["step_type"] = self.step_type.value  # Convert enum to string
        return data


class CommandPlanStepDescription(StepDescriptionBase):
    """LLM selection of one typed, host-owned command capability."""

    config: CommandPlanStepConfig = Field(..., description="Typed command planning configuration")
    llm_config: Optional[LLMStepConfig] = Field(
        default=None,
        description="Optional LLM override for the planning call.",
    )
    retry_max: Optional[int] = Field(default=None, ge=1, le=10)
    timeout: float = Field(default=30.0, gt=0, le=300.0)

    @property
    def step_type(self) -> StepType:
        return StepType.COMMAND_PLAN

    @property
    def step_config(self) -> CommandPlanStepConfig:
        return self.config

    @model_validator(mode="after")
    def reject_registry_blind_cache(self) -> "CommandPlanStepDescription":
        if self.cache is not None:
            raise ValueError(
                "CommandPlanStep cannot be cached because its host capability registry is runtime-only"
            )
        return self

    def model_dump(self, **kwargs) -> dict[str, Any]:
        data = super().model_dump(**kwargs)
        data["step_type"] = self.step_type.value
        return data


class ShellSessionStepDescription(StepDescriptionBase):
    """Static commands executed by one POSIX shell process."""

    config: ShellSessionStepConfig = Field(..., description="One-process shell session configuration")

    @property
    def step_type(self) -> StepType:
        return StepType.SHELL_SESSION

    @property
    def step_config(self) -> ShellSessionStepConfig:
        return self.config

    @model_validator(mode="after")
    def reject_side_effect_cache(self) -> "ShellSessionStepDescription":
        if self.cache is not None:
            raise ValueError(
                "ShellSessionStep cannot be cached because shell execution may have side effects"
            )
        return self

    def model_dump(self, **kwargs) -> dict[str, Any]:
        data = super().model_dump(**kwargs)
        data["step_type"] = self.step_type.value
        return data


class ConditionalStepDescription(StepDescriptionBase):
    """
    Conditional branching step description.

    Evaluates conditions and determines next step.
    """

    config: ConditionalStepConfig = Field(..., description="Conditional configuration")

    @property
    def step_type(self) -> StepType:
        return StepType.CONDITIONAL

    @property
    def step_config(self) -> ConditionalStepConfig:
        return self.config

    def model_dump(self, **kwargs) -> dict[str, Any]:
        """Dump model to dict including step_type property."""
        data = super().model_dump(**kwargs)
        data["step_type"] = self.step_type.value  # Convert enum to string
        return data


class StructuredOutputStepDescription(StepDescriptionBase):
    """Structured output generation step description."""

    config: StructuredOutputStepConfig = Field(..., description="Structured output configuration")

    @property
    def step_type(self) -> StepType:
        return StepType.STRUCTURED_OUTPUT

    @property
    def step_config(self) -> StructuredOutputStepConfig:
        return self.config

    def model_dump(self, **kwargs) -> dict[str, Any]:
        """Dump model to dict including step_type property."""
        data = super().model_dump(**kwargs)
        data["step_type"] = self.step_type.value  # Convert enum to string
        return data


class AgentHandoffStepDescription(StepDescriptionBase):
    """
    Agent handoff step description.

    Runs a complete sub-chain inside the parent chain's execution.
    The sub-chain receives an isolated ``ReasoningContext`` derived from the
    parent, with inputs resolved from parent memory/history.  The sub-chain's
    result is merged back into the parent context via ``config.output_memory_key``.

    Example::

        AgentHandoffStepDescription(
            number=3,
            title="Delegate to research agent",
            sub_chain=research_chain,
            config=AgentHandoffStepConfig(
                input_mapping={"input.topic": "$memory.input.topic"},
                output_memory_key="research_result",
            ),
        )

    Notes
    -----
    - ``sub_chain`` is a runtime-only field and is not serialized.
    - Sub-chain failures propagate to the parent step by default
      (control with ``config.propagate_failure``).
    - The full ``ReasoningResult`` of the sub-chain is always available in
      ``step_result.result_data["sub_result"]``.
    """

    model_config = {"arbitrary_types_allowed": True}

    sub_chain: Any = Field(
        ...,
        description="ReasoningChain instance to execute as the sub-agent.",
        exclude=True,
    )
    config: AgentHandoffStepConfig = Field(..., description="Handoff configuration")

    @property
    def step_type(self) -> StepType:
        return StepType.AGENT_HANDOFF

    @property
    def step_config(self) -> AgentHandoffStepConfig:
        return self.config

    def model_dump(self, **kwargs) -> dict[str, Any]:
        data = super().model_dump(**kwargs)
        data["step_type"] = self.step_type.value
        return data


class SupervisorStepDescription(StepDescriptionBase):
    """
    Supervisor / hierarchical-routing step description.

    Routes a task to one of N registered sub-chains using an LLM. The supervisor
    LLM is shown the available agent names and the resolved task; its reply is
    matched against ``agents`` and the chosen sub-chain is executed with an
    isolated context derived from the parent.

    Example::

        SupervisorStepDescription(
            number=1,
            title="Route to specialist",
            agents={"pdf": pdf_chain, "search": search_chain, "code": code_chain},
            config=SupervisorStepConfig(
                routing_prompt=(
                    "Pick ONE specialist for the task. Reply with just the name.\\n"
                    "Specialists: {agents}\\n\\nTask: {task}"
                ),
                output_memory_key="specialist_result",
            ),
        )

    Notes
    -----
    - ``agents`` is a runtime-only field (not serialized) keyed by agent name
      with :class:`ReasoningChain` values.
    - The full ``ReasoningResult`` of the chosen sub-chain is available in
      ``step_result.result_data["sub_result"]``; the selected agent name is in
      ``result_data["agent_selected"]``.
    - Sub-chain failures propagate by default (control with ``config.propagate_failure``).
    """

    model_config = {"arbitrary_types_allowed": True}

    agents: dict[str, Any] = Field(
        ...,
        description="Mapping of agent name → ReasoningChain instance.",
        exclude=True,
    )
    config: SupervisorStepConfig = Field(..., description="Supervisor configuration")

    @property
    def step_type(self) -> StepType:
        return StepType.SUPERVISOR

    @property
    def step_config(self) -> SupervisorStepConfig:
        return self.config

    def model_dump(self, **kwargs) -> dict[str, Any]:
        data = super().model_dump(**kwargs)
        data["step_type"] = self.step_type.value
        return data


class DebateStepDescription(StepDescriptionBase):
    """
    Round-robin multi-agent debate step description.

    Runs ``len(config.roles) * config.rounds`` LLM calls in strict turn-taking
    order, then a single judge synthesis call. The debate transcript and the
    judge's verdict are both surfaced in ``result_data``.

    Example::

        DebateStepDescription(
            number=3,
            title="Debate the approach",
            config=DebateStepConfig(
                roles=["proponent", "critic"],
                rounds=2,
                judge_prompt="Topic: {task}\\n\\nDebate:\\n{transcript}\\n\\nVerdict:",
                output_memory_key="verdict",
            ),
        )

    Result data
    -----------
    - ``verdict`` — string returned by the judge.
    - ``transcript`` — list of ``{"round": int, "role": str, "argument": str}`` dicts.
    - ``rounds_executed`` — int (equals ``config.rounds`` on success).
    - ``role_call_count`` — total number of role calls made (``len(roles) * rounds``).
    """

    config: DebateStepConfig = Field(..., description="Debate configuration")

    @property
    def step_type(self) -> StepType:
        return StepType.DEBATE

    @property
    def step_config(self) -> DebateStepConfig:
        return self.config

    def model_dump(self, **kwargs) -> dict[str, Any]:
        data = super().model_dump(**kwargs)
        data["step_type"] = self.step_type.value
        return data


class EvaluationStepDescription(StepDescriptionBase):
    """
    Inline quality-gate step description.

    Evaluates the output of a previously executed step against a set of criteria
    and reacts according to ``config.on_fail`` when criteria are not satisfied.

    Example::

        EvaluationStepDescription(
            number=5,
            title="Quality gate",
            dependencies=[4],
            config=EvaluationStepConfig(
                evaluates_step=4,
                criteria=["The response is at least 200 words", "All claims have sources"],
                on_fail=EvalFailAction.RETRY_WITH_FEEDBACK,
            ),
        )
    """

    config: EvaluationStepConfig = Field(..., description="Evaluation step configuration")

    @property
    def step_type(self) -> StepType:
        return StepType.EVALUATION

    @property
    def step_config(self) -> EvaluationStepConfig:
        return self.config

    def model_dump(self, **kwargs) -> dict[str, Any]:
        data = super().model_dump(**kwargs)
        data["step_type"] = self.step_type.value
        return data


class AgentSkillStepDescription(StepDescriptionBase):
    """
    AgentSkill execution step description.

    Executes an AgentSkill (https://agentskills.io) as a step in a CARL reasoning chain.
    Supports LLM-driven, script-based, and hybrid execution modes.

    Example (LLM mode):
        AgentSkillStepDescription(
            number=1,
            title="Read PDF",
            config=AgentSkillStepConfig(
                skill="pdf",
                task="Extract all text from the PDF at {pdf_path}",
                input_mapping={"pdf_path": "$memory.input.pdf_path"},
            ),
        )

    Example (script mode):
        AgentSkillStepDescription(
            number=1,
            title="Extract PDF text",
            config=AgentSkillStepConfig(
                skill="/path/to/pdf-skill",
                task="Extract text",
                execution_mode=AgentSkillExecutionMode.SCRIPT,
                script_name="scripts/extract.py",
                script_args={"output": "text"},
                input_mapping={"input": "$memory.pdf_path"},
            ),
        )
    """

    config: AgentSkillStepConfig = Field(..., description="AgentSkill configuration")

    @property
    def step_type(self) -> StepType:
        return StepType.AGENT_SKILL

    @property
    def step_config(self) -> AgentSkillStepConfig:
        return self.config

    def model_dump(self, **kwargs) -> dict[str, Any]:
        data = super().model_dump(**kwargs)
        data["step_type"] = self.step_type.value
        return data


class ParallelSamplingStepDescription(StepDescriptionBase):
    """
    Parallel sampling step — runs ``n_samples`` independent copies of *base_step*
    concurrently and aggregates their responses into a single winning result.

    Implements the strategy from "More Agents Is All You Need" (arXiv 2402.05120):
    sampling multiple independent completions and voting improves reasoning accuracy
    by ~5–10% without any additional fine-tuning.

    Aggregation strategies
    ----------------------
    - ``majority_vote`` — most common response after optional normalisation wins.
    - ``best_of_n`` / ``llm_judge`` — an LLM judge selects the best candidate.

    Example::

        ParallelSamplingStepDescription(
            number=2,
            title="Synthesize answer with voting",
            dependencies=[1],
            base_step=LLMStepDescription(
                number=2, title="_sample",
                aim="Answer the user's question accurately.",
            ),
            config=ParallelSamplingStepConfig(
                n_samples=5,
                aggregation=ParallelSamplingAggregation.MAJORITY_VOTE,
            ),
        )
    """

    base_step: "LLMStepDescription" = Field(
        ...,
        description="Template LLM step to run N times in parallel.",
        exclude=True,
    )
    config: ParallelSamplingStepConfig = Field(
        ...,
        description="Parallel sampling configuration.",
    )

    @property
    def step_type(self) -> StepType:
        return StepType.PARALLEL_SAMPLING

    @property
    def step_config(self) -> ParallelSamplingStepConfig:
        return self.config

    def model_dump(self, **kwargs) -> dict[str, Any]:
        data = super().model_dump(**kwargs)
        data["step_type"] = self.step_type.value
        return data


class ToolDiscoveryStepDescription(StepDescriptionBase):
    """
    Tool discovery step — loads tools from an external source at runtime and
    registers them in the execution context so that subsequent ToolSteps can use them.

    This enables dynamic, intent-based tool loading: instead of pre-registering all
    tools before chain execution, a ToolDiscoveryStep can load only the tools relevant
    to the current task.

    Example::

        chain = ReasoningChain(steps=[
            # Step 1: discover search tools
            ToolDiscoveryStepDescription(
                number=1,
                title="Load tools",
                config=ToolDiscoveryStepConfig(
                    source=ModuleToolSource(module="myapp.search_tools"),
                    output_memory_key="loaded_tools",
                ),
            ),
            # Step 2: use a discovered tool
            ToolStepDescription(
                number=2,
                title="Search",
                dependencies=[1],
                config=ToolStepConfig(
                    tool_name="web_search",
                    input_mapping={"query": "$history[-1]"},
                ),
            ),
        ])
    """

    config: ToolDiscoveryStepConfig = Field(
        ..., description="Tool discovery configuration."
    )

    @property
    def step_type(self) -> StepType:
        return StepType.TOOL_DISCOVERY

    @property
    def step_config(self) -> ToolDiscoveryStepConfig:
        return self.config

    def model_dump(self, **kwargs) -> dict[str, Any]:
        data = super().model_dump(**kwargs)
        data["step_type"] = self.step_type.value
        return data


class HumanInputStepDescription(StepDescriptionBase):
    """
    Human-in-the-loop step that awaits one typed, process-local text response.

    Only a validated response returned by ``on_human_input_requested`` is
    successful. Missing providers, timeout, cancellation and invalid responses
    are explicit non-success outcomes; no implicit fallback is fabricated.

    Example::

        HumanInputStepDescription(
            number=4,
            title="User review required",
            config=HumanInputStepConfig(
                prompt="Please review the draft and provide feedback:",
                timeout=300,
                output_memory_key="user_feedback",
            ),
        )

    Usage with an event loop::

        ctx = ReasoningContext(...)

        async def get_user_input(request):
            user_value = await ask_user(request.prompt)
            return HumanInputResponse(
                request_id=request.request_id,
                value=user_value,
                actor_id="local-user",
            )

        ctx.on_human_input_requested = get_user_input
        result = await chain.execute_async(ctx)

    Notes
    -----
    - The callback receives ``HumanInputRequest`` and returns a Pydantic
      ``HumanInputResponse`` (or an awaitable response).
    - Synchronous callbacks must return promptly; asynchronous callbacks should own
      any UI or transport wait they require.
    - Sensitive responses are written only to the configured memory key and are
      redacted from result text and history.
    """

    config: HumanInputStepConfig = Field(
        ..., description="Human input step configuration."
    )

    @property
    def step_type(self) -> StepType:
        return StepType.HUMAN_INPUT

    @property
    def step_config(self) -> HumanInputStepConfig:
        return self.config

    def model_dump(self, **kwargs) -> dict[str, Any]:
        data = super().model_dump(**kwargs)
        data["step_type"] = self.step_type.value
        return data


# Union type for all step descriptions
AnyStepDescription = Union[
    LLMStepDescription,
    AgentStepDescription,
    ClaudeCodeStepDescription,
    CodexStepDescription,
    CodeStepDescription,
    WaitStepDescription,
    MapStepDescription,
    ToolStepDescription,
    MCPStepDescription,
    MemoryStepDescription,
    TransformStepDescription,
    CommandPlanStepDescription,
    CommandStepDescription,
    ShellSessionStepDescription,
    ConditionalStepDescription,
    StructuredOutputStepDescription,
    AgentSkillStepDescription,
    EvaluationStepDescription,
    AgentHandoffStepDescription,
    ParallelSamplingStepDescription,
    ToolDiscoveryStepDescription,
    HumanInputStepDescription,
    SupervisorStepDescription,
    DebateStepDescription,
    MCPResourceStepDescription,
]


class StepDescription(BaseModel):
    """
    DEPRECATED: Use typed step classes instead.

    This class provides BACKWARD COMPATIBILITY with the previous unified API.
    For new code, prefer using the specific step classes:
    - LLMStepDescription - for LLM reasoning steps
    - ToolStepDescription - for external function calls
    - MCPStepDescription - for MCP protocol calls
    - MemoryStepDescription - for memory operations
    - TransformStepDescription - for data transformations
    - CommandStepDescription - for argv-based operating-system commands
    - ShellSessionStepDescription - for static one-process shell sessions
    - ConditionalStepDescription - for conditional branching
    - StructuredOutputStepDescription - for structured JSON output

    Migration example:
        # Old (deprecated):
        StepDescription(
            number=1,
            title="Analysis",
            aim="Analyze data",
            step_type=StepType.LLM
        )

        # New (recommended):
        LLMStepDescription(
            number=1,
            title="Analysis",
            aim="Analyze data"
        )

    Supports multiple step types:
    - LLM (default): Standard LLM reasoning with prompts
    - TOOL: External function/tool execution
    - MCP: Model Context Protocol server calls
    - MEMORY: Read/write operations on shared memory
    - TRANSFORM: Data transformations without LLM
    - COMMAND: Operating-system command execution through a runtime
    - SHELL_SESSION: Static commands sharing one shell process
    - CONDITIONAL: Branching logic based on conditions
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **data):
        """Initialize with deprecation warning."""
        import warnings

        warnings.warn(
            "StepDescription is deprecated. Use typed step classes instead: "
            "LLMStepDescription, ToolStepDescription, MCPStepDescription, "
            "MemoryStepDescription, TransformStepDescription, CommandStepDescription, "
            "ShellSessionStepDescription, "
            "ConditionalStepDescription. "
            "See documentation for migration guide.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(**data)

    # === Core Fields (required for all step types) ===
    number: int = Field(..., description="Step number in the sequence")
    title: str = Field(..., description="Human-readable title of the step")
    dependencies: list[int] = Field(default_factory=list, description="List of step numbers this step depends on")
    triggered_by: list[str] = Field(
        default_factory=list,
        description="Event names that must be emitted before this step becomes ready.",
    )
    loop_back_to: Optional[int] = Field(default=None)
    loop_config: Optional[LoopConfig] = Field(default=None)
    checkpoint: bool = Field(default=False, description="Mark this step as a RE-PLAN rollback checkpoint")
    checkpoint_name: str | None = Field(default=None, description="Optional checkpoint name")
    replan_enabled: bool | None = Field(
        default=None,
        description="Optional per-step RE-PLAN override (None = chain policy default)",
    )

    # === Step Type Configuration ===
    step_type: StepType = Field(default=StepType.LLM, description="Type of step execution")
    step_config: Optional[
        Union[
            AgentStepConfig,
            ClaudeCodeStepConfig,
            CodexStepConfig,
            CodeStepConfig,
            WaitStepConfig,
            ToolStepConfig,
            MCPStepConfig,
            MemoryStepConfig,
            TransformStepConfig,
            CommandPlanStepConfig,
            CommandStepConfig,
            ShellSessionStepConfig,
            ConditionalStepConfig,
            StructuredOutputStepConfig,
        ]
    ] = Field(default=None, description="Type-specific configuration (required for non-LLM steps)")

    # === LLM Step Fields (used when step_type=LLM) ===
    aim: str = Field(default="", description="Primary objective of this step")
    reasoning_questions: str = Field(default="", description="Key questions to answer")
    step_context_queries: list[ContextQuery | str] = Field(
        default_factory=list, description="List of queries to extract relevant context from outer_context (RAG-like)"
    )
    stage_action: str = Field(default="", description="Specific action to perform")
    example_reasoning: str = Field(default="", description="Example of expert reasoning")

    # Per-step LLM configuration override (used when step_type=LLM)
    llm_config: Optional[LLMStepConfig] = Field(
        default=None,
        description="Optional LLM configuration override for this step (model, temperature, etc.)",
    )

    # Per-step retry override
    retry_max: Optional[int] = Field(
        default=None,
        description="Override retry attempts for this step (None = use context default)",
    )

    # Per-step timeout override
    timeout: Optional[float] = Field(
        default=None,
        gt=0,
        description="Timeout for this step in seconds (None = use chain default)",
    )

    metrics: list[Any] = Field(
        default_factory=list,
        description="List of MetricBase instances to evaluate after step execution",
        exclude=True,
    )

    @model_validator(mode="after")
    def validate_step_config(self) -> "StepDescription":
        """Validate that step configuration matches step type."""
        if self.step_type == StepType.LLM:
            # LLM steps need aim at minimum
            if not self.aim:
                raise ValueError("LLM steps require 'aim' to be set")
        elif self.step_type == StepType.TOOL:
            if not isinstance(self.step_config, ToolStepConfig):
                raise ValueError("TOOL steps require ToolStepConfig")
        elif self.step_type == StepType.AGENT:
            if not isinstance(self.step_config, AgentStepConfig):
                raise ValueError("AGENT steps require AgentStepConfig")
        elif self.step_type == StepType.CLAUDE_CODE:
            if not isinstance(self.step_config, ClaudeCodeStepConfig):
                raise ValueError("CLAUDE_CODE steps require ClaudeCodeStepConfig")
        elif self.step_type == StepType.CODEX:
            if not isinstance(self.step_config, CodexStepConfig):
                raise ValueError("CODEX steps require CodexStepConfig")
        elif self.step_type == StepType.CODE:
            if not isinstance(self.step_config, CodeStepConfig):
                raise ValueError("CODE steps require CodeStepConfig")
        elif self.step_type == StepType.WAIT:
            if not isinstance(self.step_config, WaitStepConfig):
                raise ValueError("WAIT steps require WaitStepConfig")
        elif self.step_type == StepType.MCP:
            if not isinstance(self.step_config, MCPStepConfig):
                raise ValueError("MCP steps require MCPStepConfig")
        elif self.step_type == StepType.MEMORY:
            if not isinstance(self.step_config, MemoryStepConfig):
                raise ValueError("MEMORY steps require MemoryStepConfig")
        elif self.step_type == StepType.TRANSFORM:
            if not isinstance(self.step_config, TransformStepConfig):
                raise ValueError("TRANSFORM steps require TransformStepConfig")
        elif self.step_type == StepType.COMMAND_PLAN:
            if not isinstance(self.step_config, CommandPlanStepConfig):
                raise ValueError("COMMAND_PLAN steps require CommandPlanStepConfig")
        elif self.step_type == StepType.COMMAND:
            if not isinstance(self.step_config, CommandStepConfig):
                raise ValueError("COMMAND steps require CommandStepConfig")
        elif self.step_type == StepType.SHELL_SESSION:
            if not isinstance(self.step_config, ShellSessionStepConfig):
                raise ValueError("SHELL_SESSION steps require ShellSessionStepConfig")
        elif self.step_type == StepType.CONDITIONAL:
            if not isinstance(self.step_config, ConditionalStepConfig):
                raise ValueError("CONDITIONAL steps require ConditionalStepConfig")
        elif self.step_type == StepType.STRUCTURED_OUTPUT:
            if not isinstance(self.step_config, StructuredOutputStepConfig):
                raise ValueError("STRUCTURED_OUTPUT steps require StructuredOutputStepConfig")
        return self

    def depends_on(self, step_number: int) -> bool:
        """Check if this step depends on a given step number."""
        return step_number in self.dependencies

    def has_dependencies(self) -> bool:
        """Check if this step has any dependencies."""
        return len(self.dependencies) > 0

    def is_llm_step(self) -> bool:
        """Check if this is an LLM reasoning step."""
        return self.step_type == StepType.LLM

    def is_tool_step(self) -> bool:
        """Check if this is a tool execution step."""
        return self.step_type == StepType.TOOL

    def is_mcp_step(self) -> bool:
        """Check if this is an MCP protocol step."""
        return self.step_type == StepType.MCP

    def is_memory_step(self) -> bool:
        """Check if this is a memory operation step."""
        return self.step_type == StepType.MEMORY

    def is_transform_step(self) -> bool:
        """Check if this is a data transformation step."""
        return self.step_type == StepType.TRANSFORM

    def is_command_step(self) -> bool:
        """Check if this is an operating-system command step."""
        return self.step_type == StepType.COMMAND

    def is_command_plan_step(self) -> bool:
        """Check if this step selects a typed command capability."""
        return self.step_type == StepType.COMMAND_PLAN

    def is_shell_session_step(self) -> bool:
        """Check if this is a one-process shell session step."""
        return self.step_type == StepType.SHELL_SESSION

    def is_conditional_step(self) -> bool:
        """Check if this is a conditional branching step."""
        return self.step_type == StepType.CONDITIONAL

    def is_structured_output_step(self) -> bool:
        """Check if this is a structured output generation step."""
        return self.step_type == StepType.STRUCTURED_OUTPUT

    def to_typed_step(self) -> AnyStepDescription:
        """
        Convert this legacy StepDescription to the appropriate typed step class.

        Returns:
            The appropriate typed step description instance.
        """
        if self.step_type == StepType.LLM:
            return LLMStepDescription(
                number=self.number,
                title=self.title,
                dependencies=self.dependencies,
                triggered_by=self.triggered_by,
                loop_back_to=self.loop_back_to,
                loop_config=self.loop_config,
                checkpoint=self.checkpoint,
                checkpoint_name=self.checkpoint_name,
                replan_enabled=self.replan_enabled,
                aim=self.aim,
                reasoning_questions=self.reasoning_questions,
                step_context_queries=self.step_context_queries,
                stage_action=self.stage_action,
                example_reasoning=self.example_reasoning,
                llm_config=self.llm_config,
            )
        elif self.step_type == StepType.TOOL:
            return ToolStepDescription(
                number=self.number,
                title=self.title,
                dependencies=self.dependencies,
                triggered_by=self.triggered_by,
                loop_back_to=self.loop_back_to,
                loop_config=self.loop_config,
                checkpoint=self.checkpoint,
                checkpoint_name=self.checkpoint_name,
                replan_enabled=self.replan_enabled,
                config=self.step_config,  # type: ignore
            )
        elif self.step_type == StepType.AGENT:
            return AgentStepDescription(
                number=self.number,
                title=self.title,
                dependencies=self.dependencies,
                checkpoint=self.checkpoint,
                checkpoint_name=self.checkpoint_name,
                replan_enabled=self.replan_enabled,
                config=self.step_config,  # type: ignore
                llm_config=self.llm_config,
            )
        elif self.step_type == StepType.CLAUDE_CODE:
            return ClaudeCodeStepDescription(
                number=self.number,
                title=self.title,
                dependencies=self.dependencies,
                triggered_by=self.triggered_by,
                loop_back_to=self.loop_back_to,
                loop_config=self.loop_config,
                checkpoint=self.checkpoint,
                checkpoint_name=self.checkpoint_name,
                replan_enabled=self.replan_enabled,
                config=self.step_config,  # type: ignore
            )
        elif self.step_type == StepType.CODEX:
            return CodexStepDescription(
                number=self.number,
                title=self.title,
                dependencies=self.dependencies,
                triggered_by=self.triggered_by,
                loop_back_to=self.loop_back_to,
                loop_config=self.loop_config,
                checkpoint=self.checkpoint,
                checkpoint_name=self.checkpoint_name,
                replan_enabled=self.replan_enabled,
                config=self.step_config,  # type: ignore
            )
        elif self.step_type == StepType.CODE:
            return CodeStepDescription(
                number=self.number,
                title=self.title,
                dependencies=self.dependencies,
                checkpoint=self.checkpoint,
                checkpoint_name=self.checkpoint_name,
                replan_enabled=self.replan_enabled,
                config=self.step_config,  # type: ignore
            )
        elif self.step_type == StepType.WAIT:
            return WaitStepDescription(
                number=self.number,
                title=self.title,
                dependencies=self.dependencies,
                checkpoint=self.checkpoint,
                checkpoint_name=self.checkpoint_name,
                replan_enabled=self.replan_enabled,
                config=self.step_config,  # type: ignore
            )
        elif self.step_type == StepType.MCP:
            return MCPStepDescription(
                number=self.number,
                title=self.title,
                dependencies=self.dependencies,
                triggered_by=self.triggered_by,
                loop_back_to=self.loop_back_to,
                loop_config=self.loop_config,
                checkpoint=self.checkpoint,
                checkpoint_name=self.checkpoint_name,
                replan_enabled=self.replan_enabled,
                config=self.step_config,  # type: ignore
            )
        elif self.step_type == StepType.MEMORY:
            return MemoryStepDescription(
                number=self.number,
                title=self.title,
                dependencies=self.dependencies,
                triggered_by=self.triggered_by,
                loop_back_to=self.loop_back_to,
                loop_config=self.loop_config,
                checkpoint=self.checkpoint,
                checkpoint_name=self.checkpoint_name,
                replan_enabled=self.replan_enabled,
                config=self.step_config,  # type: ignore
            )
        elif self.step_type == StepType.TRANSFORM:
            return TransformStepDescription(
                number=self.number,
                title=self.title,
                dependencies=self.dependencies,
                triggered_by=self.triggered_by,
                loop_back_to=self.loop_back_to,
                loop_config=self.loop_config,
                checkpoint=self.checkpoint,
                checkpoint_name=self.checkpoint_name,
                replan_enabled=self.replan_enabled,
                config=self.step_config,  # type: ignore
            )
        elif self.step_type == StepType.COMMAND_PLAN:
            return CommandPlanStepDescription(
                number=self.number,
                title=self.title,
                dependencies=self.dependencies,
                triggered_by=self.triggered_by,
                loop_back_to=self.loop_back_to,
                loop_config=self.loop_config,
                checkpoint=self.checkpoint,
                checkpoint_name=self.checkpoint_name,
                replan_enabled=self.replan_enabled,
                config=self.step_config,  # type: ignore
                llm_config=self.llm_config,
                retry_max=self.retry_max,
                timeout=self.timeout or 30.0,
            )
        elif self.step_type == StepType.COMMAND:
            return CommandStepDescription(
                number=self.number,
                title=self.title,
                dependencies=self.dependencies,
                triggered_by=self.triggered_by,
                loop_back_to=self.loop_back_to,
                loop_config=self.loop_config,
                checkpoint=self.checkpoint,
                checkpoint_name=self.checkpoint_name,
                replan_enabled=self.replan_enabled,
                config=self.step_config,  # type: ignore
            )
        elif self.step_type == StepType.SHELL_SESSION:
            return ShellSessionStepDescription(
                number=self.number,
                title=self.title,
                dependencies=self.dependencies,
                triggered_by=self.triggered_by,
                loop_back_to=self.loop_back_to,
                loop_config=self.loop_config,
                checkpoint=self.checkpoint,
                checkpoint_name=self.checkpoint_name,
                replan_enabled=self.replan_enabled,
                config=self.step_config,  # type: ignore
            )
        elif self.step_type == StepType.CONDITIONAL:
            return ConditionalStepDescription(
                number=self.number,
                title=self.title,
                dependencies=self.dependencies,
                triggered_by=self.triggered_by,
                loop_back_to=self.loop_back_to,
                loop_config=self.loop_config,
                checkpoint=self.checkpoint,
                checkpoint_name=self.checkpoint_name,
                replan_enabled=self.replan_enabled,
                config=self.step_config,  # type: ignore
            )
        elif self.step_type == StepType.STRUCTURED_OUTPUT:
            return StructuredOutputStepDescription(
                number=self.number,
                title=self.title,
                dependencies=self.dependencies,
                triggered_by=self.triggered_by,
                loop_back_to=self.loop_back_to,
                loop_config=self.loop_config,
                checkpoint=self.checkpoint,
                checkpoint_name=self.checkpoint_name,
                replan_enabled=self.replan_enabled,
                config=self.step_config,  # type: ignore
            )
        else:
            raise ValueError(f"Unknown step type: {self.step_type}")


def create_step(
    number: int,
    title: str,
    step_type: StepType = StepType.LLM,
    dependencies: list[int] | None = None,
    checkpoint: bool = False,
    checkpoint_name: str | None = None,
    replan_enabled: bool | None = None,
    *,
    # LLM step fields
    aim: str = "",
    reasoning_questions: str = "",
    step_context_queries: list[ContextQuery | str] | None = None,
    stage_action: str = "",
    example_reasoning: str = "",
    llm_config: LLMStepConfig | None = None,
    # Non-LLM step config
    config: StepConfig = None,
) -> AnyStepDescription:
    """
    Factory function to create the appropriate step description type.

    This is the preferred way to create steps in new code.

    Args:
        number: Step number in the sequence
        title: Human-readable title
        step_type: Type of step (LLM, TOOL, MCP, MEMORY, TRANSFORM, CONDITIONAL, STRUCTURED_OUTPUT)
        dependencies: List of step numbers this step depends on
        aim: Primary objective (LLM steps only)
        reasoning_questions: Key questions to answer (LLM steps only)
        step_context_queries: Context queries for RAG (LLM steps only)
        stage_action: Specific action (LLM steps only)
        example_reasoning: Example reasoning (LLM steps only)
        llm_config: Per-step LLM configuration override (LLM steps only)
        config: Step configuration (non-LLM steps only)

    Returns:
        The appropriate typed step description instance.
    """
    deps = dependencies or []

    if step_type == StepType.LLM:
        return LLMStepDescription(
            number=number,
            title=title,
            dependencies=deps,
            checkpoint=checkpoint,
            checkpoint_name=checkpoint_name,
            replan_enabled=replan_enabled,
            aim=aim,
            reasoning_questions=reasoning_questions,
            step_context_queries=step_context_queries or [],
            stage_action=stage_action,
            example_reasoning=example_reasoning,
            llm_config=llm_config,
        )
    elif step_type == StepType.AGENT:
        if not isinstance(config, AgentStepConfig):
            raise ValueError("AGENT steps require AgentStepConfig")
        return AgentStepDescription(
            number=number,
            title=title,
            dependencies=deps,
            checkpoint=checkpoint,
            checkpoint_name=checkpoint_name,
            replan_enabled=replan_enabled,
            config=config,
            llm_config=llm_config,
        )
    elif step_type == StepType.CLAUDE_CODE:
        if not isinstance(config, ClaudeCodeStepConfig):
            raise ValueError("CLAUDE_CODE steps require ClaudeCodeStepConfig")
        return ClaudeCodeStepDescription(
            number=number,
            title=title,
            dependencies=deps,
            checkpoint=checkpoint,
            checkpoint_name=checkpoint_name,
            replan_enabled=replan_enabled,
            config=config,
        )
    elif step_type == StepType.CODEX:
        if not isinstance(config, CodexStepConfig):
            raise ValueError("CODEX steps require CodexStepConfig")
        return CodexStepDescription(
            number=number,
            title=title,
            dependencies=deps,
            checkpoint=checkpoint,
            checkpoint_name=checkpoint_name,
            replan_enabled=replan_enabled,
            config=config,
        )
    elif step_type == StepType.CODE:
        if not isinstance(config, CodeStepConfig):
            raise ValueError("CODE steps require CodeStepConfig")
        return CodeStepDescription(
            number=number,
            title=title,
            dependencies=deps,
            checkpoint=checkpoint,
            checkpoint_name=checkpoint_name,
            replan_enabled=replan_enabled,
            config=config,
        )
    elif step_type == StepType.WAIT:
        if not isinstance(config, WaitStepConfig):
            raise ValueError("WAIT steps require WaitStepConfig")
        return WaitStepDescription(
            number=number,
            title=title,
            dependencies=deps,
            checkpoint=checkpoint,
            checkpoint_name=checkpoint_name,
            replan_enabled=replan_enabled,
            config=config,
        )
    elif step_type == StepType.MAP:
        if not isinstance(config, MapStepConfig):
            raise ValueError("MAP steps require MapStepConfig")
        return MapStepDescription(
            number=number,
            title=title,
            dependencies=deps,
            checkpoint=checkpoint,
            checkpoint_name=checkpoint_name,
            replan_enabled=replan_enabled,
            config=config,
        )
    elif step_type == StepType.TOOL:
        if not isinstance(config, ToolStepConfig):
            raise ValueError("TOOL steps require ToolStepConfig")
        return ToolStepDescription(
            number=number,
            title=title,
            dependencies=deps,
            checkpoint=checkpoint,
            checkpoint_name=checkpoint_name,
            replan_enabled=replan_enabled,
            config=config,
        )
    elif step_type == StepType.MCP:
        if not isinstance(config, MCPStepConfig):
            raise ValueError("MCP steps require MCPStepConfig")
        return MCPStepDescription(
            number=number,
            title=title,
            dependencies=deps,
            checkpoint=checkpoint,
            checkpoint_name=checkpoint_name,
            replan_enabled=replan_enabled,
            config=config,
        )
    elif step_type == StepType.MEMORY:
        if not isinstance(config, MemoryStepConfig):
            raise ValueError("MEMORY steps require MemoryStepConfig")
        return MemoryStepDescription(
            number=number,
            title=title,
            dependencies=deps,
            checkpoint=checkpoint,
            checkpoint_name=checkpoint_name,
            replan_enabled=replan_enabled,
            config=config,
        )
    elif step_type == StepType.TRANSFORM:
        if not isinstance(config, TransformStepConfig):
            raise ValueError("TRANSFORM steps require TransformStepConfig")
        return TransformStepDescription(
            number=number,
            title=title,
            dependencies=deps,
            checkpoint=checkpoint,
            checkpoint_name=checkpoint_name,
            replan_enabled=replan_enabled,
            config=config,
        )
    elif step_type == StepType.COMMAND_PLAN:
        if not isinstance(config, CommandPlanStepConfig):
            raise ValueError("COMMAND_PLAN steps require CommandPlanStepConfig")
        return CommandPlanStepDescription(
            number=number,
            title=title,
            dependencies=deps,
            checkpoint=checkpoint,
            checkpoint_name=checkpoint_name,
            replan_enabled=replan_enabled,
            config=config,
            llm_config=llm_config,
        )
    elif step_type == StepType.COMMAND:
        if not isinstance(config, CommandStepConfig):
            raise ValueError("COMMAND steps require CommandStepConfig")
        return CommandStepDescription(
            number=number,
            title=title,
            dependencies=deps,
            checkpoint=checkpoint,
            checkpoint_name=checkpoint_name,
            replan_enabled=replan_enabled,
            config=config,
        )
    elif step_type == StepType.SHELL_SESSION:
        if not isinstance(config, ShellSessionStepConfig):
            raise ValueError("SHELL_SESSION steps require ShellSessionStepConfig")
        return ShellSessionStepDescription(
            number=number,
            title=title,
            dependencies=deps,
            checkpoint=checkpoint,
            checkpoint_name=checkpoint_name,
            replan_enabled=replan_enabled,
            config=config,
        )
    elif step_type == StepType.CONDITIONAL:
        if not isinstance(config, ConditionalStepConfig):
            raise ValueError("CONDITIONAL steps require ConditionalStepConfig")
        return ConditionalStepDescription(
            number=number,
            title=title,
            dependencies=deps,
            checkpoint=checkpoint,
            checkpoint_name=checkpoint_name,
            replan_enabled=replan_enabled,
            config=config,
        )
    elif step_type == StepType.STRUCTURED_OUTPUT:
        if not isinstance(config, StructuredOutputStepConfig):
            raise ValueError("STRUCTURED_OUTPUT steps require StructuredOutputStepConfig")
        return StructuredOutputStepDescription(
            number=number,
            title=title,
            dependencies=deps,
            checkpoint=checkpoint,
            checkpoint_name=checkpoint_name,
            replan_enabled=replan_enabled,
            config=config,
        )
    else:
        raise ValueError(f"Unknown step type: {step_type}")
