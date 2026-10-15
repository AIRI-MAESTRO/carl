"""
Step configuration models for CARL reasoning system.
"""

import keyword
import math
import warnings
from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Callable, Dict, Literal, Optional, Self, Union

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator

from ..command_policy import normalize_memory_limit
from ..network_enforcement import (
    NetworkEnforcerError,
    normalize_network_allowlist_hosts,
)
from .artifacts import ArtifactInput, ArtifactOutput
from .enums import MemoryOperation


class ExecutionMode(str, Enum):
    """Execution strategy for LLM step."""

    FAST = "fast"
    SELF_CRITIC = "self_critic"


class ToolParameter(BaseModel):
    """Parameter definition for tool calls."""

    name: str = Field(..., description="Parameter name")
    type: str = Field(default="string", description="Parameter type (string, int, float, bool, list, dict)")
    description: str = Field(default="", description="Parameter description")
    required: bool = Field(default=True, description="Whether the parameter is required")
    default: Any = Field(default=None, description="Default value if not required")


class ToolErrorRecovery(BaseModel):
    """
    Error recovery configuration for tool execution.

    Provides retry logic and fallback strategies when a tool fails or times out.

    Example::

        ToolStepConfig(
            tool_name="web_search",
            error_recovery=ToolErrorRecovery(
                retry_max=2,
                retry_delay=1.0,
                on_timeout="cached_search",       # fallback tool name
                on_exception="cached_search",     # fallback tool name
            )
        )
    """

    retry_max: int = Field(
        default=0,
        ge=0,
        description="Number of extra attempts after the first failure (0 = no retries).",
    )
    retry_delay: float = Field(
        default=0.0,
        ge=0.0,
        description="Seconds to wait between retry attempts.",
    )
    on_timeout: Optional[str] = Field(
        default=None,
        description=(
            "Name of a registered fallback tool to call when the primary tool times out. "
            "The fallback receives the same kwargs as the primary tool."
        ),
    )
    on_exception: Optional[str] = Field(
        default=None,
        description=(
            "Name of a registered fallback tool to call when the primary tool raises an exception "
            "(after all retries are exhausted). The fallback receives the same kwargs."
        ),
    )


class ToolStepConfig(BaseModel):
    """Configuration for tool/function call steps."""

    tool_name: str = Field(..., description="Name of the tool to call")
    tool_description: str = Field(default="", description="Description of what the tool does")
    parameters: list[ToolParameter] = Field(default_factory=list, description="Tool input parameters")
    input_mapping: dict[str, str] = Field(
        default_factory=dict,
        description="Maps step context keys to tool parameter names. Use '$history' for previous results.",
    )
    output_key: str = Field(default="result", description="Key to store tool output in step result")
    timeout: float = Field(default=30.0, description="Timeout in seconds for tool execution")
    retry_on_error: bool = Field(default=True, description="Whether to retry on error")
    error_recovery: Optional[ToolErrorRecovery] = Field(
        default=None,
        description="Error recovery strategy: retries, fallback tool on timeout or exception.",
    )
    allowed_tool_tags: Optional[list[str]] = Field(
        default=None,
        description=(
            "Optional whitelist of tags. When set, the executor refuses to call the "
            "configured tool unless its registered tag set intersects this list. "
            "Use to keep destructive tools out of steps that only need informational access."
        ),
    )

    # The actual callable is set at runtime, not serialized
    _tool_callable: Optional[Callable] = None

    model_config = {"arbitrary_types_allowed": True}


class AgentStepConfig(BaseModel):
    """Configuration for a bounded tool-using ``AgentStep``.

    The model must emit exactly one call per iteration. Calls to an explicitly
    allowed host tool continue the loop; the runtime-provided ``finish``
    meta-tool is the only successful exit.
    """

    goal: str = Field(
        ...,
        min_length=1,
        description="Static goal the agent should attempt to complete.",
    )
    input_mapping: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Maps input names to context references such as ``$history[-1]`` "
            "or ``$memory.input.question``. Resolved values are supplied as data."
        ),
    )
    tools: list[str] = Field(
        ...,
        min_length=1,
        description=(
            "Explicit non-empty allowlist of registered host tools. "
            "The reserved ``finish`` meta-tool must not be listed."
        ),
    )
    system_prompt: str = Field(
        default="",
        description="Optional step-specific instructions appended after host instructions.",
    )
    max_iterations: int = Field(
        default=8,
        ge=1,
        description="Maximum number of model-directed loop iterations.",
    )
    max_tool_calls: int = Field(
        default=12,
        ge=0,
        description="Maximum number of ordinary host-tool calls; ``finish`` does not count.",
    )
    timeout_seconds: float = Field(
        default=120.0,
        gt=0,
        description="Whole-step wall-clock limit in seconds.",
    )
    model_timeout_seconds: float = Field(
        default=60.0,
        gt=0,
        description="Wall-clock limit for each model call.",
    )
    tool_timeout_seconds: float = Field(
        default=30.0,
        gt=0,
        description="Wall-clock limit for each host-tool call.",
    )
    max_tokens: Optional[int] = Field(
        default=None,
        ge=1,
        description=(
            "Optional aggregate provider-token budget. Enforced when the LLM client "
            "reports token usage."
        ),
    )
    max_tool_result_chars: int = Field(
        default=16_000,
        ge=256,
        description="Maximum serialized characters retained from one tool observation.",
    )
    max_transcript_chars: int = Field(
        default=100_000,
        ge=1024,
        description="Maximum serialized conversation size before another model call.",
    )
    output_schema: Optional[dict[str, Any]] = Field(
        default=None,
        description="Optional JSON Schema subset used to validate ``finish.result``.",
    )
    output_key: Optional[str] = Field(
        default=None,
        description="Optional memory key written only after a valid ``finish``.",
    )
    output_namespace: str = Field(
        default="agent",
        min_length=1,
        description="Memory namespace used by ``output_key``.",
    )

    @field_validator("goal")
    @classmethod
    def _strip_goal(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("goal must not be blank")
        return value

    @field_validator("tools")
    @classmethod
    def _validate_tools(cls, value: list[str]) -> list[str]:
        normalized = [name.strip() for name in value]
        if any(not name for name in normalized):
            raise ValueError("tool names must not be blank")
        if "finish" in normalized:
            raise ValueError("'finish' is a reserved AgentStep meta-tool")
        if len(set(normalized)) != len(normalized):
            raise ValueError("AgentStep tool names must be unique")
        return normalized

    @field_validator("input_mapping")
    @classmethod
    def _validate_input_mapping(cls, value: dict[str, str]) -> dict[str, str]:
        normalized: dict[str, str] = {}
        for raw_name, raw_reference in value.items():
            name = raw_name.strip()
            reference = raw_reference.strip()
            if not name:
                raise ValueError("AgentStep input names must not be blank")
            if name in normalized:
                raise ValueError("AgentStep input names must be unique after trimming")
            if not reference.startswith("$"):
                raise ValueError(
                    "AgentStep input mappings must be context references starting with '$'"
                )
            normalized[name] = reference
        return normalized

    @field_validator("output_key")
    @classmethod
    def _validate_output_key(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("AgentStep output_key must not be blank")
        return value

    @field_validator("output_namespace")
    @classmethod
    def _validate_output_namespace(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("AgentStep output_namespace must not be blank")
        return value


class CodexStepConfig(BaseModel):
    """Configuration for delegating one CARL step to a local Codex agent.

    The serialized config describes the task and the maximum authority the
    step may request. Authentication and the Codex runtime remain host-owned;
    the executor uses the local ``openai-codex`` SDK and never serializes API
    keys or login state.

    ``read-only`` is the safe default. ``workspace-write`` must be selected
    explicitly for steps that are expected to edit files. Headless execution
    always denies approval requests, so the agent cannot pause a CARL chain to
    ask for additional authority.
    """

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, populate_by_name=True)

    task: str = Field(
        ...,
        min_length=1,
        max_length=16_000,
        validation_alias=AliasChoices("task", "instruction"),
        description=(
            "Task prompt for the Codex agent. When input_mapping is non-empty, "
            "treated as a str.format template — each {placeholder} must have an "
            "input_mapping entry. With an empty input_mapping the string is "
            "passed verbatim (literal braces are safe). "
            "(Accepted on the wire as 'instruction' for backward compatibility.)"
        ),
    )
    input_mapping: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Placeholder -> source mapping. Sources starting with '$' are "
            "resolved via the standard reference syntax ($history[-1], "
            "$memory.ns.key, $outer_context, ...); quoted strings are literals; "
            "anything else is passed through as-is."
        ),
    )
    cwd: Optional[str] = Field(
        default=None,
        description="Optional working directory exposed to the local Codex thread.",
    )
    sandbox: Literal["read-only", "workspace-write"] = Field(
        default="read-only",
        description="Codex filesystem sandbox preset. Full access is intentionally unsupported.",
    )
    model: Optional[str] = Field(
        default=None,
        min_length=1,
        description="Optional Codex model override; None uses the local Codex default.",
    )
    reasoning_effort: Optional[
        Literal["none", "minimal", "low", "medium", "high", "xhigh"]
    ] = Field(default=None, description="Optional Codex reasoning-effort override.")
    developer_instructions: Optional[str] = Field(
        default=None,
        max_length=16_000,
        description="Optional host-authored developer instructions for the Codex thread.",
    )
    resume_session: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("resume_session", "thread_id_source"),
        description=(
            "Codex thread id to resume. A '$'-reference is resolved at runtime — "
            "point it at a prior step's stored id, e.g. '$memory.codex.step_1', "
            "to continue one Codex thread across chain steps; a plain string is "
            "used as a literal thread id. When omitted, the step starts a new "
            "thread. (Accepted on the wire as 'thread_id_source'.)"
        ),
    )
    ephemeral: bool = Field(
        default=True,
        description="Whether a newly started Codex thread is omitted from durable thread history.",
    )
    timeout: float = Field(
        default=300.0,
        gt=0,
        le=3_600,
        validation_alias=AliasChoices("timeout", "timeout_seconds"),
        description=(
            "Whole-step wall-clock budget in seconds, including local runtime "
            "startup. (Accepted on the wire as 'timeout_seconds'.)"
        ),
    )
    max_input_bytes: int = Field(
        default=131_072,
        ge=1_024,
        le=1_048_576,
        description="Maximum UTF-8 bytes allowed for the fully rendered task prompt.",
    )
    output_memory_key: Optional[str] = Field(
        default=None,
        description=(
            "Optional memory key to write the agent's final answer to "
            "(the parsed object when output_schema is set, the answer text otherwise)."
        ),
    )
    output_namespace: str = Field(
        default="codex",
        min_length=1,
        description="Memory namespace used by output_memory_key.",
    )
    output_schema: Optional[dict[str, Any]] = Field(
        default=None,
        description=(
            "Optional JSON Schema. When set, the Codex final response must "
            "contain a JSON payload matching this schema; the parsed object is "
            "stored in result_data['structured_output'] and the step fails on "
            "mismatch."
        ),
    )
    store_session_key: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("store_session_key", "store_thread_key"),
        description=(
            "Memory key (namespace 'codex') for the thread id. Defaults to "
            "'step_<number>' so a later step can resume the same Codex thread "
            "via resume_session='$memory.codex.step_<n>'. "
            "(Accepted on the wire as 'store_thread_key'.)"
        ),
    )

    @field_validator("task")
    @classmethod
    def _validate_task(cls, value: str) -> str:
        value = value.strip()
        if not value or "\x00" in value:
            raise ValueError("CodexStep task must be non-blank and NUL-free")
        return value

    @field_validator("input_mapping")
    @classmethod
    def _validate_input_mapping(cls, value: dict[str, str]) -> dict[str, str]:
        if len(value) > 32:
            raise ValueError("CodexStep input_mapping supports at most 32 values")
        normalized: dict[str, str] = {}
        for raw_name, raw_source in value.items():
            name = raw_name.strip()
            source = raw_source.strip()
            if not name.isidentifier() or keyword.iskeyword(name):
                raise ValueError(f"CodexStep input name must be a safe identifier: {raw_name!r}")
            if not source or len(source) > 2_048 or "\x00" in source:
                raise ValueError(
                    f"CodexStep input source for {name!r} must be non-empty, "
                    "NUL-free, and at most 2048 characters"
                )
            if name in normalized:
                raise ValueError("CodexStep input names must be unique after trimming")
            normalized[name] = source
        return normalized

    @field_validator("cwd", "resume_session", "developer_instructions", "model", "store_session_key")
    @classmethod
    def _validate_optional_text(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = value.strip()
        if not value or "\x00" in value:
            raise ValueError("CodexStep optional text fields must be non-blank and NUL-free")
        return value

    @field_validator("timeout")
    @classmethod
    def _validate_timeout(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("CodexStep timeout must be finite")
        return value

    @field_validator("output_memory_key")
    @classmethod
    def _validate_output_key(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("CodexStep output_memory_key must not be blank")
        return value

    @field_validator("output_namespace")
    @classmethod
    def _validate_output_namespace(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("CodexStep output_namespace must not be blank")
        return value


class ClaudeCodeStepConfig(BaseModel):
    """Configuration for delegating one CARL step to a headless Claude Code CLI agent.

    The step runs ``claude -p <task> --output-format json`` as a subprocess and
    collects the agent's final answer, session id, token usage, and cost.
    Authentication stays host-owned (the locally installed CLI's own auth).
    """

    task: str = Field(
        ...,
        description=(
            "Task prompt for the Claude Code agent. When input_mapping is "
            "non-empty, treated as a str.format template — each {placeholder} "
            "must have an input_mapping entry. With an empty input_mapping the "
            "string is passed verbatim (literal braces are safe)."
        ),
    )
    input_mapping: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Placeholder -> source mapping. Sources starting with '$' are "
            "resolved via the standard reference syntax ($history[-1], "
            "$memory.ns.key, $outer_context, ...); quoted strings are literals; "
            "anything else is passed through as-is."
        ),
    )
    cli_path: str = Field(default="claude", description="Claude Code CLI executable (name on PATH or absolute path)")
    cwd: Optional[str] = Field(default=None, description="Working directory the agent runs in (its project root)")
    model: Optional[str] = Field(default=None, description="--model override (e.g. 'sonnet', 'opus')")
    max_turns: Optional[int] = Field(default=None, ge=1, description="--max-turns budget for agentic turns")
    allowed_tools: list[str] = Field(
        default_factory=list,
        description="--allowedTools whitelist, e.g. ['Read', 'Grep', 'Bash(git log:*)']",
    )
    disallowed_tools: list[str] = Field(default_factory=list, description="--disallowedTools blacklist")
    permission_mode: Optional[str] = Field(
        default=None,
        description="--permission-mode: 'default' | 'acceptEdits' | 'bypassPermissions' | 'plan'",
    )
    append_system_prompt: Optional[str] = Field(default=None, description="--append-system-prompt text")
    system_prompt: Optional[str] = Field(
        default=None,
        description="--system-prompt replacement text (overrides the CLI's default agent system prompt)",
    )
    stream: bool = Field(
        default=False,
        description=(
            "Stream agent progress: uses --output-format stream-json and forwards "
            "each assistant text block to context.on_llm_chunk as it arrives."
        ),
    )
    resume_session: Optional[str] = Field(
        default=None,
        description=(
            "Session id to resume (--resume). A '$'-reference is resolved at "
            "runtime — point it at a prior step's stored id, e.g. "
            "'$memory.claude_code.step_1', to continue one agent session "
            "across chain steps."
        ),
    )
    add_dirs: list[str] = Field(default_factory=list, description="Extra directories the agent may access (--add-dir)")
    extra_cli_args: list[str] = Field(default_factory=list, description="Escape hatch: raw argv appended to the command")
    env: dict[str, str] = Field(default_factory=dict, description="Extra environment variables for the CLI process")
    timeout: float = Field(default=600.0, gt=0, description="Wall-clock budget in seconds; the process is killed on expiry")
    max_output_chars: int = Field(
        default=20000,
        gt=0,
        description="Result text longer than this is truncated in result/history (full text stays in result_data['payload'])",
    )
    store_session_key: Optional[str] = Field(
        default=None,
        description=(
            "Memory key (namespace 'claude_code') for the session id. "
            "Defaults to 'step_<number>'."
        ),
    )
    output_memory_key: Optional[str] = Field(
        default=None,
        description=(
            "Optional memory key to write the agent's final answer to "
            "(the parsed object when output_schema is set, the answer text otherwise)."
        ),
    )
    output_namespace: str = Field(
        default="claude_code",
        description="Memory namespace used with output_memory_key.",
    )
    output_schema: Optional[dict[str, Any]] = Field(
        default=None,
        description=(
            "Optional JSON Schema. When set, the agent's final answer must contain "
            "a JSON payload matching this schema; the parsed object is stored in "
            "result_data['structured_output'] and the step fails on mismatch."
        ),
    )

    @field_validator("output_memory_key")
    @classmethod
    def _validate_output_memory_key(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = value.strip()
        if not value:
            raise ValueError("ClaudeCodeStep output_memory_key must not be blank")
        return value

    @field_validator("output_namespace")
    @classmethod
    def _validate_output_namespace(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("ClaudeCodeStep output_namespace must not be blank")
        return value


class MCPServerConfig(BaseModel):
    """Configuration for MCP server connection."""

    server_name: str = Field(..., description="Name of the MCP server")
    transport: Literal["stdio", "http", "sse"] = Field(
        default="stdio",
        description=(
            "Transport type: 'stdio' (subprocess pipe), 'http' (streamable HTTP — MCP 2025-03-26 spec), "
            "'sse' (HTTP + Server-Sent Events — legacy MCP transport)."
        ),
    )
    command: Optional[str] = Field(default=None, description="Command to start stdio server")
    args: list[str] = Field(default_factory=list, description="Arguments for stdio server")
    url: Optional[str] = Field(default=None, description="Base URL for HTTP/SSE server")
    headers: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Extra HTTP headers sent with every request (e.g. Authorization: Bearer …). "
            "Applied to both 'http' and 'sse' transports."
        ),
    )


class MCPStepConfig(BaseModel):
    """Configuration for MCP protocol steps."""

    server: MCPServerConfig = Field(..., description="MCP server configuration")
    tool_name: str = Field(..., description="Name of the MCP tool to call")
    arguments: dict[str, Any] = Field(default_factory=dict, description="Static arguments for the tool")
    argument_mapping: dict[str, str] = Field(
        default_factory=dict, description="Maps step context keys to MCP tool arguments"
    )
    timeout: float = Field(default=60.0, description="Timeout in seconds")


class MCPResourceStepConfig(BaseModel):
    """
    Configuration for MCP resource-reading steps.

    Resources are read-only data exposed by an MCP server (files,
    documentation, schemas, etc.) — distinct from MCP tools, which take
    arguments and perform actions. This step type fetches a named resource
    and writes it to memory (and history) without making any tool call.

    Example::

        MCPResourceStepDescription(
            number=2,
            title="Load API docs",
            config=MCPResourceStepConfig(
                server=MCPServerConfig(server_name="docs", transport="sse", url="..."),
                resource_uri="docs://api/reference.md",
                output_memory_key="api_docs",
            ),
        )
    """

    server: MCPServerConfig = Field(..., description="MCP server configuration")
    resource_uri: str = Field(
        ...,
        min_length=1,
        description="URI of the resource to read (e.g. 'docs://api/reference.md').",
    )
    output_memory_key: str = Field(
        default="",
        description=(
            "Memory key to store the resource content. When non-empty, the "
            "content is written to ``memory[output_namespace][output_memory_key]``. "
            "When empty, the content is only available via ``$history[-1]`` / "
            "``$metadata.step_N``."
        ),
    )
    output_namespace: str = Field(
        default="mcp_resource",
        description="Memory namespace for ``output_memory_key`` writes.",
    )
    timeout: float = Field(
        default=30.0,
        gt=0,
        description="Timeout in seconds for the MCP read_resource call.",
    )


class MemoryStepConfig(BaseModel):
    """Configuration for memory read/write steps."""

    operation: MemoryOperation = Field(..., description="Memory operation type")
    memory_key: str = Field(..., description="Key in memory store")
    value_source: Optional[str] = Field(
        default=None, description="Source of value for write operations (context key or '$history[-1]')"
    )
    default_value: Any = Field(default=None, description="Default value if key not found on read")
    namespace: str = Field(default="default", description="Memory namespace for isolation")


class TransformStepConfig(BaseModel):
    """Configuration for data transformation steps (no LLM call)."""

    transform_type: Literal["extract", "format", "aggregate", "filter", "map"] = Field(
        ..., description="Type of transformation"
    )
    input_key: str = Field(default="$history[-1]", description="Key to get input from")
    output_format: Optional[str] = Field(default=None, description="Output format template")
    expression: Optional[str] = Field(default=None, description="Transformation expression (for extract/filter)")
    # For 'map' operations - applies a simple template to each item
    map_template: Optional[str] = Field(default=None, description="Template for map operations")


class _RuntimeExecutionConfig(BaseModel):
    """Controls shared by argv commands and one-process shell sessions."""

    model_config = {"allow_inf_nan": False}

    input_mapping: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Maps environment-safe names to context references. Resolved values "
            "are exported as CARL_ARG_<NAME>; a concrete step may also expose "
            "them through another injection-safe channel."
        ),
    )
    env: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Explicit environment variables. The full host environment is not "
            "inherited; local execution receives only PATH/LANG/LC_ALL plus "
            "these keys and CARL-owned derived variables."
        ),
    )
    artifact_inputs: list[ArtifactInput] = Field(
        default_factory=list,
        description="Context values staged as bounded files below runtime in/.",
    )
    artifact_outputs: list[ArtifactOutput] = Field(
        default_factory=list,
        description="Required bounded files collected from runtime out/.",
    )
    working_dir: Optional[str] = Field(
        default=None,
        description=(
            "Runtime-local working directory. Defaults to the runtime's writable "
            "ephemeral out/ directory; CARL never deletes an explicitly supplied cwd."
        ),
    )
    runtime: str = Field(
        default="local",
        description="Runtime backend name: local | docker | e2b | firejail (or a registered backend).",
    )
    enforcement_mode: Literal["strict", "best_effort"] = Field(
        default="strict",
        description=(
            "strict fails before execution if a requested control is not enforced; "
            "best_effort runs only when host policy permits it and reports gaps."
        ),
    )
    network: Literal["none", "allowlist", "host"] = Field(
        default="none",
        description="Network policy. Fail-closed default none requests no egress.",
    )
    network_allowlist: list[str] = Field(
        default_factory=list,
        description="Exact egress hosts requested when network='allowlist'.",
    )
    timeout: float = Field(
        default=30.0,
        gt=0,
        description="Wall-clock command/session runtime timeout in seconds.",
    )
    artifact_io_timeout: float = Field(
        default=30.0,
        gt=0,
        description="Independent timeout for staging and collecting each artifact.",
    )
    cpu_limit: Optional[float] = Field(
        default=None, gt=0, description="CPU core limit (backends that support it)."
    )
    mem_limit: Optional[str] = Field(
        default=None, description="Memory limit, Docker-style suffix string, e.g. '512m'."
    )
    pids_limit: Optional[int] = Field(
        default=None, gt=0, description="Maximum process count (backends that support it)."
    )
    allow_nonzero_exit: bool = Field(
        default=False,
        description="When False, a non-zero process exit marks the step failed.",
    )
    max_output_bytes: int = Field(
        default=1_000_000,
        gt=0,
        description="Maximum retained stdout and stderr bytes, independently.",
    )
    output_key: str = Field(
        default="result",
        description="Key under which stdout is exposed in result_data.",
    )

    @model_validator(mode="before")
    @classmethod
    def ignore_legacy_self_authorization(cls, value: Any) -> Any:
        """Load old chains without letting chain data grant host authority."""

        if not isinstance(value, dict):
            return value
        legacy = {
            key: value[key]
            for key in ("allow_unsafe_local", "allow_unenforced_network")
            if key in value
        }
        if not legacy:
            return value
        warnings.warn(
            "allow_unsafe_local and allow_unenforced_network are deprecated and "
            "ignored; supply a host-owned CommandPolicy instead",
            DeprecationWarning,
            stacklevel=2,
        )
        cleaned = dict(value)
        for key in legacy:
            cleaned.pop(key, None)
        return cleaned

    @field_validator("input_mapping")
    @classmethod
    def validate_input_names(cls, value: dict[str, str]) -> dict[str, str]:
        import re

        invalid = [name for name in value if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) is None]
        if invalid:
            raise ValueError(f"input_mapping keys must be environment-safe identifiers: {invalid!r}")
        normalized = [name.upper() for name in value]
        if len(normalized) != len(set(normalized)):
            raise ValueError("input_mapping keys must remain unique after uppercasing")
        return value

    @field_validator("env")
    @classmethod
    def validate_environment(cls, value: dict[str, str]) -> dict[str, str]:
        invalid = [
            key
            for key, item in value.items()
            if not key
            or "=" in key
            or "\x00" in key
            or "\x00" in item
            or key.upper().startswith("CARL_ARG_")
            or key.upper().startswith("CARL_ARTIFACT_")
        ]
        if invalid:
            raise ValueError(f"invalid or CARL-reserved environment variables: {invalid!r}")
        normalized = [key.upper() for key in value]
        if len(normalized) != len(set(normalized)):
            raise ValueError("environment keys must remain unique after uppercasing")
        return value

    @field_validator("network_allowlist")
    @classmethod
    def normalize_network_hosts(cls, value: list[str]) -> list[str]:
        if not value:
            return []
        try:
            return list(normalize_network_allowlist_hosts(tuple(value)))
        except (NetworkEnforcerError, TypeError) as exc:
            raise ValueError(str(exc)) from exc

    @field_validator("working_dir")
    @classmethod
    def validate_working_dir(cls, value: str | None) -> str | None:
        if value is not None:
            try:
                value.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise ValueError("working_dir must be valid UTF-8") from exc
            if not value.strip() or "\x00" in value:
                raise ValueError("working_dir must be non-empty and contain no NUL bytes")
        return value

    @field_validator("mem_limit")
    @classmethod
    def normalize_mem_limit(cls, value: str | None) -> str | None:
        return None if value is None else normalize_memory_limit(value)

    @model_validator(mode="after")
    def validate_runtime_contract(self) -> "_RuntimeExecutionConfig":
        if self.network == "allowlist" and not self.network_allowlist:
            raise ValueError("network='allowlist' requires at least one host")
        if self.network != "allowlist" and self.network_allowlist:
            raise ValueError("network_allowlist is only valid with network='allowlist'")

        for label, artifacts in (
            ("artifact_inputs", self.artifact_inputs),
            ("artifact_outputs", self.artifact_outputs),
        ):
            names = [artifact.name.upper() for artifact in artifacts]
            paths = [artifact.path.casefold() for artifact in artifacts]
            if len(names) != len(set(names)):
                raise ValueError(f"{label} names must be unique after uppercasing")
            if len(paths) != len(set(paths)):
                raise ValueError(f"{label} paths must remain unique after case-folding")
        return self

    @field_validator("output_key")
    @classmethod
    def validate_output_key(cls, value: str) -> str:
        reserved = {
            "approval",
            "artifact_manifest",
            "artifacts",
            "command",
            "command_source",
            "command_count",
            "capability_fingerprint",
            "capability_id",
            "capability_revision",
            "enforcement_report",
            "exit_code",
            "network_enforced",
            "network_enforcement",
            "network_policy",
            "policy_decision",
            "resolved_argument_count",
            "script_sha256",
            "shell",
            "stderr",
            "stderr_truncated",
            "stdout",
            "stdout_truncated",
        }
        if value in reserved:
            raise ValueError(f"output_key {value!r} collides with command result metadata")
        return value


class CommandStepConfig(_RuntimeExecutionConfig):
    """Configuration for operating-system command steps.

    A command step runs one executable with an explicit argv list. Because
    its inputs may be LLM-generated or data-derived, this config is built to
    be **safe by default**: no shell is inserted implicitly, execution goes
    through a ``SkillRuntime``, the host environment is not inherited in
    full, and networking is denied unless explicitly opened.

    The executable and static flags in ``command`` are fixed when the chain
    is built. Resolved ``input_mapping`` values are appended as data
    arguments; ``stdin_source`` passes context through stdin. A preceding LLM
    step therefore cannot silently replace the executable. If a caller
    deliberately puts ``bash -lc`` (or an equivalent interpreter) in
    ``command``, shell semantics and their risks apply.

    ``working_dir`` is a runtime-local cwd. For ``local`` it may be an
    absolute host path. For sandbox runtimes it must be meaningful inside the
    sandbox; when omitted the executor selects that runtime's writable
    workspace. It is never treated as a workspace owned by CARL and is never
    deleted during cleanup.

    Security model (enforced by ``CommandStepExecutor``):

    * ``command`` is an argv list — there is no shell, so shell
      metacharacters in resolved values cannot inject extra commands.
    * ``input_mapping`` values are resolved via the usual ``$history`` /
      ``$memory`` / ``$outer_context`` references and passed as **discrete
      argv tokens** (appended) and as ``CARL_ARG_<NAME>`` environment
      variables — never string-interpolated into the command.
    * ``runtime`` selects the backend by name. Permission is not stored in the
      chain: the application must supply a host-owned ``CommandPolicy`` on the
      ``ReasoningContext``. In particular, ``runtime="local"`` cannot
      self-authorize through serialized step configuration.
    * ``network`` defaults to ``"none"``. ``cpu_limit`` / ``mem_limit`` /
      ``pids_limit`` request resource controls. Under ``enforcement_mode=
      "strict"`` execution fails before launch when the selected backend
      cannot enforce any requested control.
    """

    command: Optional[list[str]] = Field(
        default=None,
        min_length=1,
        description=(
            "Command as an argv list, e.g. ['grep', '-n', 'TODO']. NOT a shell "
            "string — this eliminates shell-injection. Use ['bash', '-lc', script] "
            "only when you deliberately need shell features."
        ),
    )
    plan_source: Optional[str] = Field(
        default=None,
        description=(
            "Context reference to a CommandPlanRecord produced by CommandPlanStep. "
            "Exactly one of command or plan_source must be set."
        ),
    )
    planned_capability_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Capability ids this planned CommandStep is willing to consume. This "
            "only narrows the host registry; it never grants execution authority."
        ),
    )
    stdin_source: Optional[str] = Field(
        default=None,
        description="Optional context reference whose resolved value is piped to the command's stdin.",
    )

    @field_validator("command")
    @classmethod
    def validate_command_argv(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        if not value[0]:
            raise ValueError("command executable must not be empty")
        if any("\x00" in arg for arg in value):
            raise ValueError("command arguments must not contain NUL bytes")
        return value

    @field_validator("plan_source")
    @classmethod
    def validate_plan_source(cls, value: str | None) -> str | None:
        if value is not None and (
            not value.strip()
            or "\x00" in value
            or not value.startswith(("$steps.", "$metadata."))
        ):
            raise ValueError(
                "plan_source must reference a CommandPlanStep via $steps.* or "
                "an externally supplied record via $metadata.*"
            )
        return value

    @field_validator("planned_capability_ids")
    @classmethod
    def validate_planned_capability_ids(cls, value: list[str]) -> list[str]:
        import re

        if len(value) != len(set(value)):
            raise ValueError("planned_capability_ids must be unique")
        for capability_id in value:
            if re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]{0,127}", capability_id) is None:
                raise ValueError(
                    "planned_capability_ids must match [A-Za-z][A-Za-z0-9_.-]{0,127}"
                )
        return value

    @model_validator(mode="after")
    def validate_command_source(self) -> "CommandStepConfig":
        if (self.command is None) == (self.plan_source is None):
            raise ValueError("exactly one of command or plan_source must be set")
        if self.plan_source is not None:
            if not self.planned_capability_ids:
                raise ValueError("planned CommandStep requires planned_capability_ids")
            if self.input_mapping:
                raise ValueError(
                    "planned CommandStep cannot append untyped input_mapping arguments; "
                    "put typed values in CommandPlanStep inputs instead"
                )
        elif self.planned_capability_ids:
            raise ValueError("planned_capability_ids is only valid with plan_source")
        return self


class CommandPlanStepConfig(BaseModel):
    """Choose typed arguments for one host-provided command capability.

    The serialized step may request a subset of capability ids, but the
    runtime-only registry decides which ids and argument schemas actually
    exist. The LLM never supplies an executable, argv prefix, runtime,
    network mode, or resource limits.
    """

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    instruction: str = Field(..., min_length=1, max_length=16_000)
    capability_ids: list[str] = Field(..., min_length=1, max_length=32)
    input_mapping: dict[str, str] = Field(
        default_factory=dict,
        description="Named context references included in the planner prompt.",
    )

    @field_validator("instruction")
    @classmethod
    def validate_instruction(cls, value: str) -> str:
        if "\x00" in value:
            raise ValueError("instruction must not contain NUL bytes")
        try:
            value.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ValueError("instruction must be valid UTF-8") from exc
        return value

    @field_validator("capability_ids")
    @classmethod
    def validate_capability_ids(cls, value: list[str]) -> list[str]:
        import re

        if len(value) != len(set(value)):
            raise ValueError("capability_ids must be unique")
        for capability_id in value:
            if re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]{0,127}", capability_id) is None:
                raise ValueError(
                    "capability_ids must match [A-Za-z][A-Za-z0-9_.-]{0,127}"
                )
        return value

    @field_validator("input_mapping")
    @classmethod
    def validate_planner_inputs(cls, value: dict[str, str]) -> dict[str, str]:
        import re

        if len(value) > 32:
            raise ValueError("input_mapping supports at most 32 values")
        invalid = [
            name
            for name in value
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) is None
        ]
        if invalid:
            raise ValueError(
                f"input_mapping keys must be safe identifiers: {invalid!r}"
            )
        normalized = [name.casefold() for name in value]
        if len(normalized) != len(set(normalized)):
            raise ValueError("input_mapping keys must be unique after case-folding")
        invalid_sources: list[str] = []
        for name, source in value.items():
            try:
                source.encode("utf-8")
            except UnicodeEncodeError:
                invalid_sources.append(name)
                continue
            if not source or len(source) > 2_048 or "\x00" in source:
                invalid_sources.append(name)
        if invalid_sources:
            raise ValueError(
                "input_mapping references must be non-empty, valid UTF-8, NUL-free, "
                f"and at most 2048 characters: {invalid_sources!r}"
            )
        return value


class ShellSessionStepConfig(_RuntimeExecutionConfig):
    """A static sequence executed by one POSIX shell process.

    The commands share cwd, shell variables, functions, and filesystem state
    for the lifetime of this step. They are not interpolated from context;
    dynamic values are exposed only through ``CARL_ARG_*`` environment
    variables or declared input artifacts. The whole script is supplied on
    stdin to ``<shell> -s`` and is covered by the approval fingerprint.

    Authorizing the shell authorizes every command in this static script; the
    executable allowlist cannot constrain programs launched *inside* it.
    Therefore interpreter approval remains enabled by default, and untrusted
    sessions belong in a genuinely isolated runtime rather than ``local``.
    """

    commands: list[str] = Field(..., min_length=1)
    shell: str = Field(default="/bin/sh", min_length=1)
    stop_on_error: bool = Field(
        default=True,
        description="Prefix the script with `set -e` so the shell stops on an unhandled failure.",
    )

    @field_validator("shell")
    @classmethod
    def validate_shell(cls, value: str) -> str:
        if "\x00" in value:
            raise ValueError("shell executable must not contain NUL bytes")
        return value

    @field_validator("commands")
    @classmethod
    def validate_session_commands(cls, value: list[str]) -> list[str]:
        invalid = [index for index, command in enumerate(value) if not command.strip() or "\x00" in command]
        if invalid:
            raise ValueError(f"session commands must be non-empty and NUL-free: indices {invalid!r}")
        return value


class ConditionalBranch(BaseModel):
    """A single conditional branch."""

    condition: str = Field(..., description="Condition expression (evaluated against context)")
    next_step: int = Field(..., description="Step number to execute if condition is true")


class ConditionalStepConfig(BaseModel):
    """Configuration for conditional branching steps."""

    branches: list[ConditionalBranch] = Field(..., description="List of conditional branches")
    default_step: Optional[int] = Field(default=None, description="Default step if no condition matches")
    condition_context_key: str = Field(
        default="$history[-1]", description="Context key to evaluate conditions against"
    )


class StructuredOutputStepConfig(BaseModel):
    """Configuration for schema-constrained structured output steps."""

    input_source: str = Field(
        default="$history[-1]",
        description="Input source used to build the structured output prompt",
    )
    output_schema: dict[str, Any] = Field(
        ...,
        description="JSON Schema that model output must match",
    )
    schema_name: str = Field(default="StructuredOutput", description="Human-readable schema name")
    instruction: str = Field(
        default="",
        description="Additional instruction for the model before schema conversion",
    )
    strict_json: bool = Field(
        default=True,
        description="If true, executor asks model to return only raw JSON",
    )

    @classmethod
    def from_pydantic_model(
        cls,
        model_cls: type[BaseModel],
        input_source: str = "$history[-1]",
        instruction: str = "",
        strict_json: bool = True,
    ) -> "StructuredOutputStepConfig":
        """Create structured config from a Pydantic model class."""
        return cls(
            input_source=input_source,
            output_schema=model_cls.model_json_schema(),
            schema_name=model_cls.__name__,
            instruction=instruction,
            strict_json=strict_json,
        )


class AgentHandoffStepConfig(BaseModel):
    """
    Configuration for agent handoff steps.

    An agent handoff step runs a complete sub-chain inside the parent chain's
    execution, with isolated context derived from the parent. The result is
    merged back into the parent via ``output_memory_key``.

    Example::

        AgentHandoffStepDescription(
            number=3,
            title="Delegate to research agent",
            sub_chain=research_chain,
            config=AgentHandoffStepConfig(
                input_mapping={
                    "input.topic": "$memory.input.topic",
                    "input.lang": "'english'",
                },
                output_memory_key="research_result",
            ),
        )

    Input mapping
    -------------
    Keys are dotted ``namespace.key`` paths written into the sub-chain's memory.
    Values are parent context references resolved by ``resolve_context_reference``:
    ``$memory.ns.key``, ``$history[-1]``, ``$outer_context``, or quoted literals.

    If a key has no dot the value is written to the ``"input"`` namespace.

    Output
    ------
    On success, the sub-chain's final result (last history entry) is written to
    ``context.memory_write(output_memory_key, result, namespace=output_namespace)``.
    The full ``ReasoningResult`` is always available in ``result_data["sub_result"]``.
    """

    input_mapping: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Maps resolved parent values to sub-chain memory entries. "
            "Keys are 'namespace.key' in sub-chain memory; values are parent context references."
        ),
    )
    output_memory_key: str = Field(
        default="",
        description="Memory key in the parent context to store the sub-chain's result. "
                    "Empty string disables automatic result storage.",
    )
    output_namespace: str = Field(
        default="handoff",
        description="Namespace to use when writing the output to parent memory.",
    )
    propagate_failure: bool = Field(
        default=True,
        description="If True, a failed sub-chain marks this step as failed. "
                    "If False, sub-chain failures are recorded in result_data but the parent step succeeds.",
    )
    inherit_tools: bool = Field(
        default=True,
        description="If True, the sub-chain's context receives a copy of the parent tool registry.",
    )
    timeout: Optional[float] = Field(
        default=None,
        gt=0,
        description="Maximum seconds to allow the sub-chain to run. None = no timeout.",
    )


class LoopConfig(BaseModel):
    """
    Configuration for loop-back execution in a reasoning chain.

    Attach to a step via ``loop_back_to`` / ``loop_config`` on
    ``StepDescriptionBase``.  After the step completes successfully the
    executor checks ``condition_key``; when the resolved value is truthy
    (and ``max_iterations`` is not exhausted) it resets the loop body steps
    and re-runs them.

    Example::

        # Steps 1-2 form the loop body; step 2 drives iteration.
        ToolStepDescription(
            number=2,
            title="Refine answer",
            config=ToolStepConfig(tool_name="refiner", input_mapping={}),
            loop_back_to=1,
            loop_config=LoopConfig(condition_key="$memory.loop.needs_retry", max_iterations=5),
        )
    """

    condition_key: str = Field(
        default="",
        description=(
            "Reference to the value that controls loop continuation. "
            "Accepts any context reference supported by resolve_context_reference: "
            "``$memory.namespace.key``, ``$history[-1]``, or ``$outer_context``. "
            "The resolved value is cast to bool; truthy → loop continues. "
            "Empty string (default) means 'always loop' up to max_iterations."
        ),
    )
    max_iterations: int = Field(
        default=10,
        ge=1,
        description="Maximum number of times the loop body may be re-executed (budget guard).",
    )
    negate_condition: bool = Field(
        default=False,
        description=(
            "If True, the continuation decision is negated: loop continues when the resolved "
            "value is *falsy* (until-loop semantics). Default False = while-loop semantics."
        ),
    )


class EvalFailAction(str, Enum):
    """Action to take when an EvaluationStep's criteria are not met."""

    CONTINUE = "continue"
    """Log the failure and let the chain continue normally."""

    ABORT = "abort"
    """Fail the chain immediately (step result marked as failure)."""

    RETRY_WITH_FEEDBACK = "retry_with_feedback"
    """Re-run an improved LLM response using the critique as additional context,
    then re-evaluate up to *max_retries* times before falling through to CONTINUE."""


class EvaluationStepConfig(BaseModel):
    """
    Configuration for inline quality-gate (evaluation) steps.

    An evaluation step assesses the output of a previously executed step and
    reacts according to ``on_fail`` when one or more criteria are not satisfied.

    Example::

        EvaluationStepDescription(
            number=5,
            title="Check output quality",
            dependencies=[4],
            config=EvaluationStepConfig(
                evaluates_step=4,
                criteria=[
                    "The response is at least 200 words",
                    "All numerical claims have sources",
                ],
                on_fail=EvalFailAction.RETRY_WITH_FEEDBACK,
                max_retries=2,
            ),
        )

    Criteria evaluation methods
    ---------------------------
    ``"rule"`` (default when all criteria are simpleeval expressions):
        Each criterion is a boolean expression evaluated by simpleeval over
        a ``value`` variable holding the target step's output string.
        Example: ``"len(value) >= 200"`` or ``"'source' in value"``.

        Shorthand patterns (no eval required):
        - ``"min_words:<N>"``  — at least N whitespace-separated words
        - ``"contains:<text>"`` — case-insensitive substring check
        - ``"startswith:<prefix>"`` / ``"endswith:<suffix>"``

    ``"llm"``:
        Passes all criteria to an LLM judge in a structured prompt.
        The LLM returns ``PASS`` or ``FAIL`` with a critique string.
        Criteria can be plain English sentences.
    """

    evaluates_step: int = Field(
        ...,
        description=(
            "Step number whose output is evaluated. The evaluation step must declare "
            "this step in its dependencies so the output is available in history."
        ),
    )
    criteria: list[str] = Field(
        ...,
        min_length=1,
        description="List of evaluation criteria. Interpretation depends on evaluation_method.",
    )
    on_fail: EvalFailAction = Field(
        default=EvalFailAction.CONTINUE,
        description="Action to take when one or more criteria are not satisfied.",
    )
    evaluation_method: Literal["rule", "llm"] = Field(
        default="rule",
        description="How to evaluate criteria: 'rule' uses simpleeval expressions, 'llm' uses an LLM judge.",
    )
    input_source: str = Field(
        default="$history[-1]",
        description=(
            "Context reference pointing to the evaluated step's output "
            "(e.g. '$history[-1]', '$history[-2]'). Defaults to the most recent history entry."
        ),
    )
    max_retries: int = Field(
        default=1,
        ge=0,
        description=(
            "Maximum number of LLM retry calls when on_fail=RETRY_WITH_FEEDBACK. "
            "After exhausting retries the chain continues normally."
        ),
    )


class ParallelSamplingAggregation(str, Enum):
    """Aggregation strategy for ParallelSamplingStep."""

    MAJORITY_VOTE = "majority_vote"  # Most common response wins (exact/normalized match)
    BEST_OF_N = "best_of_n"  # LLM judge picks the best candidate
    LLM_JUDGE = "llm_judge"  # Alias for best_of_n with explicit judge prompt


class ParallelSamplingStepConfig(BaseModel):
    """Configuration for ParallelSamplingStep."""

    n_samples: int = Field(
        default=5,
        ge=2,
        description="Number of independent LLM samples to draw in parallel.",
    )
    aggregation: ParallelSamplingAggregation = Field(
        default=ParallelSamplingAggregation.MAJORITY_VOTE,
        description="Strategy used to select the winning response from all samples.",
    )
    judge_prompt: str = Field(
        default="",
        description=(
            "Optional custom judge prompt for BEST_OF_N / LLM_JUDGE aggregation. "
            "Receives the candidates as a numbered list. Leave empty to use the default."
        ),
    )
    normalize_for_vote: bool = Field(
        default=True,
        description=(
            "When True, responses are lowercased and stripped before majority-vote comparison, "
            "reducing spurious mismatches from whitespace/case differences."
        ),
    )


class ModuleToolSource(BaseModel):
    """
    Discover tools by scanning a Python module for public callables.

    Example::

        ModuleToolSource(module="mypackage.tools")
        ModuleToolSource(module="mypackage.tools", name_prefix="search_")
        ModuleToolSource(module="mypackage.tools", tag="information")
    """

    module: str = Field(..., description="Fully-qualified Python module path to import and scan.")
    name_prefix: str = Field(
        default="",
        description=(
            "Only register callables whose names start with this prefix. "
            "Empty string (default) matches all public callables."
        ),
    )
    tag: str = Field(
        default="",
        description=(
            "Only register callables that have the given tag in their "
            "``__carl_tool_tags__`` attribute (set by the @carl_tool decorator). "
            "Empty string (default) disables tag filtering."
        ),
    )
    strip_prefix: bool = Field(
        default=False,
        description=(
            "When True, ``name_prefix`` is stripped from registered tool names. "
            "E.g. prefix='tool_', strip_prefix=True → 'tool_search' registered as 'search'."
        ),
    )


class CallableToolSource(BaseModel):
    """
    Discover tools by calling a factory function.

    The factory must return a ``dict[str, Callable]`` mapping tool names to callables.

    Example::

        def my_factory() -> dict:
            return {"search": search_fn, "summarize": summarize_fn}

        CallableToolSource(factory=my_factory)
    """

    model_config = {"arbitrary_types_allowed": True}

    factory: Callable[[], Dict[str, Callable]] = Field(
        ..., description="Zero-argument callable that returns a dict of {tool_name: callable}."
    )


class DictToolSource(BaseModel):
    """
    Register a static dict of tools.

    Example::

        DictToolSource(tools={"search": search_fn, "calc": calc_fn})
    """

    model_config = {"arbitrary_types_allowed": True}

    tools: Dict[str, Callable] = Field(
        ..., description="Mapping of tool names to callables to register."
    )


ToolSource = Union[ModuleToolSource, CallableToolSource, DictToolSource]


class ToolDiscoveryStepConfig(BaseModel):
    """
    Configuration for a ToolDiscoveryStep.

    Discovers tools from an external source and registers them in the context
    so that subsequent ToolSteps can use them.

    Example::

        ToolDiscoveryStepDescription(
            number=1,
            title="Load search tools",
            config=ToolDiscoveryStepConfig(
                source=ModuleToolSource(module="myapp.search_tools", name_prefix="tool_"),
                output_memory_key="discovered_tools",
            ),
        )
    """

    source: ToolSource = Field(
        ...,
        description="Source of tools to discover and register.",
    )
    output_memory_key: str = Field(
        default="",
        description=(
            "If set, writes the list of discovered tool names to "
            "``context.memory['tools'][output_memory_key]`` for downstream inspection."
        ),
    )
    tool_timeout: Optional[float] = Field(
        default=None,
        description=(
            "Optional per-call timeout applied to all discovered sync tools "
            "(wraps them with AsyncToolWrapper). Has no effect on async tools."
        ),
    )


class CodeStepConfig(BaseModel):
    """Execute one exact runtime-generated Python source string."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    source: str = Field(
        ...,
        min_length=1,
        max_length=1024,
        description="Context reference resolving to exact Python source text.",
    )
    runtime_profile: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Host-owned CodeExecutionPolicy profile id.",
    )
    input_mapping: dict[str, str] = Field(
        default_factory=dict,
        description="Input object key to context-reference mapping.",
    )
    input_schema: dict[str, Any] = Field(
        ...,
        description="Required strict-subset JSON schema for the input object.",
    )
    output_schema: dict[str, Any] = Field(
        ...,
        description="Required strict-subset JSON schema for the returned JSON value.",
    )
    timeout_seconds: float = Field(default=10.0, gt=0)
    max_source_bytes: int = Field(default=20_000, gt=0)
    max_input_bytes: int = Field(default=1_000_000, gt=0)
    max_output_bytes: int = Field(default=1_000_000, gt=0)
    output_key: str | None = Field(
        default=None,
        description="Optional memory key written only after valid completion.",
    )
    output_namespace: str = Field(default="code", min_length=1, max_length=256)

    @field_validator("source")
    @classmethod
    def _source_must_be_a_context_reference(cls, value: str) -> str:
        value = value.strip()
        if not value.startswith("$"):
            raise ValueError("CodeStep source must be a context reference starting with '$'")
        return value

    @field_validator("runtime_profile")
    @classmethod
    def _runtime_profile_must_be_an_identifier(cls, value: str) -> str:
        import re

        value = value.strip()
        if re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]{0,127}", value) is None:
            raise ValueError("runtime_profile must match [A-Za-z][A-Za-z0-9_.-]{0,127}")
        return value

    @field_validator("input_mapping")
    @classmethod
    def _input_mapping_must_use_named_references(
        cls, value: dict[str, str],
    ) -> dict[str, str]:
        import re

        normalized: dict[str, str] = {}
        for raw_name, raw_reference in value.items():
            name = raw_name.strip()
            reference = raw_reference.strip()
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) is None:
                raise ValueError(f"invalid CodeStep input name {raw_name!r}")
            if name in normalized:
                raise ValueError("CodeStep input names must be unique after trimming")
            if not reference.startswith("$"):
                raise ValueError("CodeStep input mappings must be context references")
            normalized[name] = reference
        return normalized

    @field_validator("input_schema")
    @classmethod
    def _input_schema_must_be_supported(cls, value: dict[str, Any]) -> dict[str, Any]:
        from ..code_execution import validate_code_schema

        validate_code_schema(value, require_object=True)
        return value

    @field_validator("output_schema")
    @classmethod
    def _output_schema_must_be_supported(cls, value: dict[str, Any]) -> dict[str, Any]:
        from ..code_execution import validate_code_schema

        validate_code_schema(value)
        return value

    @field_validator("output_key")
    @classmethod
    def _output_key_must_be_non_empty(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        if not value or len(value) > 256:
            raise ValueError("output_key must be non-empty and at most 256 characters")
        return value

    @field_validator("output_namespace")
    @classmethod
    def _output_namespace_must_be_non_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("output_namespace cannot be blank")
        return value

    @model_validator(mode="after")
    def _input_mapping_must_satisfy_static_schema(self) -> Self:
        mapped = set(self.input_mapping)
        properties = set(self.input_schema.get("properties", {}))
        required = set(self.input_schema.get("required", []))
        missing = required - mapped
        if missing:
            raise ValueError(
                f"input_mapping does not provide required schema keys {sorted(missing)!r}"
            )
        if self.input_schema.get("additionalProperties", True) is False:
            extra = mapped - properties
            if extra:
                raise ValueError(
                    f"input_mapping contains undeclared schema keys {sorted(extra)!r}"
                )
        return self


class HumanInputStepConfig(BaseModel):
    """Configuration for one typed, process-local text request."""

    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(
        default="Please provide input:",
        max_length=8192,
        description="Human-readable prompt displayed when asking for input.",
    )
    timeout: Optional[float] = Field(
        default=None,
        gt=0,
        description=(
            "Maximum seconds to wait for the typed host callback. Timeout is a "
            "non-success outcome. None means wait indefinitely."
        ),
    )
    min_length: int = Field(
        default=0,
        ge=0,
        description="Minimum accepted response length.",
    )
    max_length: int = Field(
        default=4096,
        ge=1,
        le=65536,
        description="Maximum accepted response length.",
    )
    sensitive: bool = Field(
        default=False,
        description=(
            "Redact the answer from StepExecutionResult and model-visible history. "
            "Sensitive answers require output_memory_key."
        ),
    )
    output_memory_key: Optional[str] = Field(
        default=None,
        description=(
            "If set, write the collected input to this key in the 'human_input' "
            "memory namespace. Useful for downstream steps to read via "
            "``$memory.human_input.<key>``."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _reject_removed_fallback(cls, value: Any) -> Any:
        if isinstance(value, dict) and "fallback_value" in value:
            raise ValueError(
                "fallback_value was removed: missing input and timeout are now "
                "explicit non-success outcomes"
            )
        return value

    @field_validator("prompt")
    @classmethod
    def _prompt_must_be_non_empty(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("prompt cannot be empty")
        return normalized

    @field_validator("timeout")
    @classmethod
    def _timeout_must_be_finite(cls, value: Optional[float]) -> Optional[float]:
        if value is not None and not math.isfinite(value):
            raise ValueError("timeout must be finite")
        return value

    @field_validator("output_memory_key")
    @classmethod
    def _output_key_must_be_non_empty(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("output_memory_key cannot be empty")
        if len(normalized) > 256:
            raise ValueError("output_memory_key cannot exceed 256 characters")
        return normalized

    @model_validator(mode="after")
    def _validate_text_contract(self) -> "HumanInputStepConfig":
        if self.min_length > self.max_length:
            raise ValueError("min_length cannot exceed max_length")
        if self.sensitive and self.output_memory_key is None:
            raise ValueError("sensitive input requires output_memory_key")
        return self


class AfterWaitCondition(BaseModel):
    """Complete after a relative process-local duration."""

    type: Literal["after"] = "after"
    seconds: float = Field(
        ...,
        ge=0.0,
        description="Finite non-negative duration in seconds.",
    )

    @field_validator("seconds")
    @classmethod
    def _seconds_must_be_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("seconds must be finite")
        return value


class AtWaitCondition(BaseModel):
    """Complete at a timezone-aware absolute timestamp."""

    type: Literal["at"] = "at"
    timestamp: datetime = Field(
        ...,
        description="Timezone-aware absolute timestamp.",
    )

    @field_validator("timestamp")
    @classmethod
    def _timestamp_must_be_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamp must be timezone-aware")
        return value


class EventWaitCondition(BaseModel):
    """Complete when a named event exists in the current ReasoningContext."""

    type: Literal["event"] = "event"
    name: str = Field(..., description="Non-empty process-local event name.")

    @field_validator("name")
    @classmethod
    def _event_name_must_be_non_empty(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("event name cannot be empty")
        return normalized


WaitLeafCondition = Annotated[
    Union[AfterWaitCondition, AtWaitCondition, EventWaitCondition],
    Field(discriminator="type"),
]


class AnyOfWaitCondition(BaseModel):
    """Complete when the first declared leaf condition completes."""

    type: Literal["any_of"] = "any_of"
    conditions: list[WaitLeafCondition] = Field(
        ...,
        min_length=2,
        description="At least two After, At, or Event conditions to race.",
    )


WaitCondition = Annotated[
    Union[
        AfterWaitCondition,
        AtWaitCondition,
        EventWaitCondition,
        AnyOfWaitCondition,
    ],
    Field(discriminator="type"),
]


class WaitStepConfig(BaseModel):
    """Configuration for a self-contained asynchronous WaitStep."""

    condition: WaitCondition = Field(
        ...,
        description="Relative timer, absolute timer, named event, or AnyOf race.",
    )
    output_memory_key: Optional[str] = Field(
        default=None,
        description=(
            "Optional key for the structured wait outcome in the 'wait' "
            "memory namespace."
        ),
    )

    @field_validator("output_memory_key")
    @classmethod
    def _output_key_must_be_non_empty(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("output_memory_key cannot be empty")
        return normalized


class MapStepConfig(BaseModel):
    """Configuration for bounded ordered fan-out over a JSON array.

    ``items_source`` and every shared ``input_mapping`` value are resolved
    once before any tool call starts. Each array element is then passed to the
    same registered tool under ``item_parameter``; ``index_parameter`` can add
    the zero-based input index. V1 deliberately has no retry, filter, reduce,
    dynamic graph mutation, or per-item memory writes.
    """

    items_source: str = Field(
        ...,
        description=(
            "Context reference resolving to a JSON array: $outer_context, "
            "$memory.*, $steps.*, $metadata.*, or $event.*."
        ),
    )
    tool_name: str = Field(..., description="Registered tool invoked once per item.")
    item_parameter: str = Field(
        default="item",
        description="Tool keyword receiving the current JSON array element.",
    )
    index_parameter: str | None = Field(
        default=None,
        description="Optional tool keyword receiving the zero-based input index.",
    )
    input_mapping: dict[str, str] = Field(
        default_factory=dict,
        description="Shared tool kwargs resolved once from the step context.",
    )
    max_items: int = Field(
        default=1000,
        ge=1,
        description="Reject larger arrays before invoking the tool.",
    )
    max_concurrency: int = Field(
        default=8,
        ge=1,
        le=256,
        description="Maximum number of active item tool calls.",
    )
    item_timeout_seconds: float = Field(
        default=30.0,
        gt=0,
        description="Per-item wall-clock timeout in seconds.",
    )
    output_memory_key: str | None = Field(
        default=None,
        description="Optional key receiving the aggregate outcome after collection.",
    )
    output_namespace: str = Field(
        default="map",
        description="Memory namespace used by output_memory_key.",
    )

    @staticmethod
    def _validate_reference(value: str, field_name: str) -> str:
        normalized = value.strip()
        allowed = normalized == "$outer_context" or any(
            normalized.startswith(prefix) and len(normalized) > len(prefix)
            for prefix in (
                "$memory.",
                "$steps.",
                "$metadata.",
                "$event.",
            )
        )
        if not allowed:
            raise ValueError(
                f"{field_name} must use $outer_context, $memory.*, $steps.*, "
                "$metadata.*, or $event.*"
            )
        return normalized

    @staticmethod
    def _validate_parameter_name(value: str, field_name: str) -> str:
        normalized = value.strip()
        if not normalized.isidentifier() or keyword.iskeyword(normalized):
            raise ValueError(f"{field_name} must be a valid Python parameter name")
        return normalized

    @field_validator("items_source")
    @classmethod
    def _items_source_must_be_a_context_reference(cls, value: str) -> str:
        return cls._validate_reference(value, "items_source")

    @field_validator("tool_name", "output_namespace")
    @classmethod
    def _non_empty_names(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("value cannot be empty")
        return normalized

    @field_validator("item_parameter")
    @classmethod
    def _item_parameter_must_be_valid(cls, value: str) -> str:
        return cls._validate_parameter_name(value, "item_parameter")

    @field_validator("index_parameter")
    @classmethod
    def _index_parameter_must_be_valid(
        cls, value: str | None
    ) -> str | None:
        if value is None:
            return None
        return cls._validate_parameter_name(value, "index_parameter")

    @field_validator("output_memory_key")
    @classmethod
    def _output_memory_key_must_be_non_empty(
        cls, value: str | None
    ) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("output_memory_key cannot be empty")
        return normalized

    @field_validator("item_timeout_seconds")
    @classmethod
    def _item_timeout_must_be_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("item_timeout_seconds must be finite")
        return value

    @field_validator("input_mapping")
    @classmethod
    def _input_mapping_must_be_valid(
        cls, value: dict[str, str]
    ) -> dict[str, str]:
        normalized: dict[str, str] = {}
        for parameter, source in value.items():
            name = cls._validate_parameter_name(parameter, "input_mapping key")
            normalized[name] = cls._validate_reference(
                source, f"input_mapping[{name!r}]"
            )
        return normalized

    @model_validator(mode="after")
    def _parameter_names_must_not_collide(self) -> "MapStepConfig":
        reserved = {self.item_parameter}
        if self.index_parameter is not None:
            if self.index_parameter == self.item_parameter:
                raise ValueError("index_parameter must differ from item_parameter")
            reserved.add(self.index_parameter)
        overlap = reserved.intersection(self.input_mapping)
        if overlap:
            raise ValueError(
                "input_mapping cannot overwrite per-item parameters: "
                + ", ".join(sorted(overlap))
            )
        return self


# Union type for all step configurations
StepConfig = Union[
    AgentStepConfig,
    CodexStepConfig,
    CodeStepConfig,
    ToolStepConfig,
    MCPStepConfig,
    MemoryStepConfig,
    TransformStepConfig,
    CommandPlanStepConfig,
    CommandStepConfig,
    ShellSessionStepConfig,
    ConditionalStepConfig,
    StructuredOutputStepConfig,
    EvaluationStepConfig,
    AgentHandoffStepConfig,
    ParallelSamplingStepConfig,
    ToolDiscoveryStepConfig,
    HumanInputStepConfig,
    WaitStepConfig,
    MapStepConfig,
    LoopConfig,
    None,
]


class StepCache(BaseModel):
    """
    Step result memoization configuration.

    When attached to a step via the ``cache`` field, the DAG executor checks a
    per-executor in-memory cache **before** running the step.  On a cache hit
    the stored result is returned immediately — no LLM or tool call is made.
    On a miss the step runs normally and the result is stored for future hits.

    The cache lives on the ``DAGExecutor`` instance, so it is shared across
    all batches within a single chain execution (useful for loops that revisit
    the same inputs) but reset on each new ``chain.execute()`` call.

    Example::

        LLMStepDescription(
            number=2,
            title="Classify intent",
            aim="Classify the user intent",
            cache=StepCache(
                ttl=300,
                key_fn=lambda ctx: ctx.outer_context[:512],
            ),
        )

    A step is considered a cache hit when:
    1. A cache entry exists for the computed key, AND
    2. The entry has not expired (``ttl=None`` → never expires).
    """

    model_config = {"arbitrary_types_allowed": True}

    ttl: Optional[int] = Field(
        default=None,
        ge=1,
        description=(
            "Cache time-to-live in seconds. After this many seconds the cached "
            "result expires and the step will be re-executed on the next call. "
            "None (default) means the cache never expires within the executor's lifetime."
        ),
    )
    key_fn: Optional[Callable[..., str]] = Field(
        default=None,
        exclude=True,
        description=(
            "Optional callable that receives the current ``ReasoningContext`` and returns "
            "a string cache key.  When None, the default key combines the step number "
            "and the first 256 characters of ``context.outer_context``."
        ),
    )


class ContextQuery(BaseModel):
    """
    Individual context query with optional search configuration override.

    Allows fine-grained control over search strategy for specific queries.
    """

    query: str = Field(..., description="The query text for context extraction")
    search_strategy: Optional[Literal["substring", "vector"]] = Field(
        default=None, description="Override search strategy for this query"
    )
    search_config: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional search configuration for this query"
    )

    def __str__(self) -> str:
        return self.query


class LLMStepConfig(BaseModel):
    """
    Configuration for per-step LLM overrides.

    Allows specifying a different model for specific LLM steps,
    overriding the default set in ReasoningContext.

    Example usage:
        ```python
        # Override model for a specific step (OpenAI-compatible APIs)
        LLMStepDescription(
            number=1,
            title="Complex Analysis",
            aim="Perform complex analysis",
            llm_config=LLMStepConfig(model="anthropic/claude-3.5-sonnet")
        )
        ```
    """

    # Model override (for OpenAI-compatible APIs)
    model: Optional[str] = Field(
        default=None,
        description="Model identifier to use for this step (e.g., 'openai/gpt-4o', 'anthropic/claude-3.5-sonnet')",
    )

    # Temperature override
    temperature: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Temperature for this step (overrides default)",
    )

    # Max tokens override
    max_tokens: Optional[int] = Field(
        default=None,
        description="Max tokens for this step (overrides default)",
    )

    # Per-step timeout override
    timeout: Optional[float] = Field(
        default=None,
        gt=0,
        description="Timeout for this step in seconds (None = use chain default)",
    )

    # Token budget warning threshold
    token_budget_warning: Optional[int] = Field(
        default=None,
        gt=0,
        description=(
            "Emit a warning when total tokens consumed by this step exceed this threshold. "
            "None = no warning. Only works with clients that return usage info (e.g. OpenAICompatibleClient)."
        ),
    )

    # Execution mode override
    execution_mode: ExecutionMode = Field(
        default=ExecutionMode.FAST,
        description="Execution strategy for this step",
    )

    # SELF_CRITIC configuration
    self_critic_evaluators: list[str] = Field(
        default_factory=lambda: ["llm"],
        min_length=1,
        description="Ordered evaluator names for SELF_CRITIC mode; all must approve",
    )
    self_critic_max_revisions: int = Field(
        default=1,
        ge=0,
        description="Maximum candidate regeneration rounds when evaluator chain disapproves",
    )
    self_critic_instruction: str = Field(
        default="",
        description="Optional extra instruction for built-in 'llm' self-critic evaluator.",
    )
    self_critic_disapprove_feedback: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Static feedback appended to regeneration notes when evaluator disapproves. "
            "Keys are evaluator names; optional '*' applies to any evaluator."
        ),
    )

    # Multi-turn message history
    use_message_history: bool = Field(
        default=False,
        description=(
            "When True, the LLM step sends a structured message list instead of a flat prompt. "
            "System prompt and outer context become the first 'system' message; "
            "previous turns from ``context.messages`` are included as-is; "
            "the current step prompt is appended as a 'user' message; "
            "the LLM response is stored back into ``context.messages`` as an 'assistant' message. "
            "Requires a client that supports :meth:`get_response_with_messages` "
            "(e.g. OpenAICompatibleClient)."
        ),
    )


class SupervisorStepConfig(BaseModel):
    """
    Configuration for supervisor / hierarchical routing steps.

    A supervisor step asks an LLM to pick *one* of N named sub-chains
    (registered as ``agents`` on the :class:`SupervisorStepDescription`) and
    runs the chosen sub-chain. The result is folded back into the parent's
    history / memory just like an :class:`AgentHandoffStepConfig`.

    Example::

        SupervisorStepDescription(
            number=1,
            title="Route to specialist",
            agents={"pdf": pdf_chain, "search": search_chain, "code": code_chain},
            config=SupervisorStepConfig(
                routing_prompt=(
                    "Choose ONE specialist for the task below. "
                    "Reply with just the specialist name.\\n\\nTask: {task}"
                ),
                task_source="$outer_context",
                fallback_agent="search",
                output_memory_key="specialist_result",
            ),
        )

    Routing prompt
    --------------
    ``{task}`` is replaced with the resolved task string before the LLM call;
    ``{agents}`` is replaced with a comma-joined list of available agent names.

    The LLM's reply is matched (case-insensitive, whitespace-trimmed, prefix-
    tolerant) against the registered agent names. If no match is found and
    ``fallback_agent`` is set, that agent is used; otherwise the step fails
    with a descriptive error.
    """

    routing_prompt: str = Field(
        ...,
        min_length=1,
        description=(
            "Prompt template the supervisor LLM sees. Supports ``{task}`` (the "
            "resolved task string) and ``{agents}`` (comma-joined agent names)."
        ),
    )
    task_source: str = Field(
        default="$outer_context",
        description=(
            "Where to resolve the task string from. Same syntax as input_mapping "
            "values: ``$outer_context``, ``$memory.ns.key``, ``$history[-1]``, "
            "or a quoted literal."
        ),
    )
    fallback_agent: Optional[str] = Field(
        default=None,
        description=(
            "Agent name to use when the LLM's reply doesn't match any registered "
            "agent. When ``None`` (default), an unparseable routing reply fails the step."
        ),
    )
    input_mapping: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Forwarded to the chosen sub-chain — same semantics as "
            ":class:`AgentHandoffStepConfig.input_mapping`."
        ),
    )
    output_memory_key: str = Field(
        default="",
        description="Memory key in the parent context to store the sub-chain's result.",
    )
    output_namespace: str = Field(
        default="supervisor",
        description="Memory namespace for ``output_memory_key`` writes.",
    )
    propagate_failure: bool = Field(
        default=True,
        description=(
            "If True, a failed sub-chain marks this step as failed. "
            "If False, sub-chain failures are recorded in result_data but the parent step succeeds."
        ),
    )
    inherit_tools: bool = Field(
        default=True,
        description="If True, the sub-chain inherits the parent context's tool registry.",
    )
    timeout: Optional[float] = Field(
        default=None,
        gt=0,
        description="Optional total timeout (seconds) for the sub-chain.",
    )
    llm_config: Optional[LLMStepConfig] = Field(
        default=None,
        description="Optional LLM configuration override for the *routing* call itself.",
    )


class DebateStepConfig(BaseModel):
    """
    Configuration for round-robin multi-agent debate steps.

    A debate step runs N round-robin turns over a list of named roles
    (e.g. ``["proponent", "critic"]``). Each role's LLM call sees the topic
    plus the running transcript and produces its next argument. After all
    rounds complete, a single judge LLM call synthesises the transcript into
    a final verdict.

    Example::

        DebateStepDescription(
            number=3,
            title="Debate the approach",
            config=DebateStepConfig(
                roles=["proponent", "critic"],
                role_prompts={
                    "proponent": "Argue IN FAVOUR of the proposal. Be concrete.",
                    "critic":    "Argue AGAINST the proposal. Flag risks.",
                },
                rounds=2,
                judge_prompt="Given the debate above, what is the best answer? Topic: {task}",
                output_memory_key="verdict",
            ),
        )

    Prompt templating
    -----------------
    Per-role prompt sees ``{task}``, ``{role}``, ``{round}``, ``{transcript}``.
    Judge prompt sees ``{task}`` and ``{transcript}``.

    LLM-call count
    --------------
    ``len(roles) * rounds + 1`` (the final ``+1`` is the judge).
    """

    roles: list[str] = Field(
        ...,
        min_length=2,
        description="Names of the debating roles (≥ 2). Used as transcript speaker labels.",
    )
    role_prompts: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Optional per-role instruction template. Missing roles fall back to a "
            "neutral 'Argue your position' instruction. Supports ``{task}``, "
            "``{role}``, ``{round}``, ``{transcript}`` placeholders."
        ),
    )
    rounds: int = Field(
        default=2,
        ge=1,
        description="Number of round-robin rounds.",
    )
    task_source: str = Field(
        default="$outer_context",
        description=(
            "Where to resolve the debate topic from — same reference syntax as "
            "``input_mapping`` values."
        ),
    )
    judge_prompt: str = Field(
        ...,
        min_length=1,
        description=(
            "Prompt for the final synthesis call. Supports ``{task}`` and "
            "``{transcript}`` placeholders. Required."
        ),
    )
    output_memory_key: str = Field(
        default="",
        description="Memory key in the parent context for the judge's verdict.",
    )
    output_namespace: str = Field(
        default="debate",
        description="Memory namespace for ``output_memory_key`` writes.",
    )
    llm_config: Optional[LLMStepConfig] = Field(
        default=None,
        description="LLM config applied to every role call AND the judge call (default).",
    )
    role_llm_configs: dict[str, LLMStepConfig] = Field(
        default_factory=dict,
        description=(
            "Optional per-role LLM config overrides. Used in place of ``llm_config`` "
            "for the specified role's turns. The judge always uses ``llm_config``."
        ),
    )


class StepGroup(BaseModel):
    """
    Apply ``llm_config`` overrides to a named subset of steps in a chain.

    Useful for separating phases — e.g. give all "analysis" steps
    ``temperature=0.0`` while leaving "creative" steps at the chain default::

        ReasoningChain(
            default_llm_config=LLMStepConfig(model="gpt-4o", temperature=0.7),
            step_groups=[
                StepGroup(
                    name="analysis",
                    steps=[2, 3, 4],
                    llm_config=LLMStepConfig(temperature=0.0),
                ),
            ],
            steps=[...],
        )

    Precedence at runtime (highest → lowest):

    1. Per-step ``llm_config`` (explicitly set fields)
    2. ``StepGroup.llm_config`` (explicitly set fields on the group)
    3. ``ReasoningChain.default_llm_config``
    4. Context default client

    Group config is "baked into" each member step's ``llm_config`` at chain
    construction time using pydantic's ``model_fields_set`` to detect which
    fields the caller actually set — defaults are not propagated, so the
    chain default still wins for unset fields.

    Validation rules (enforced by :class:`ReasoningChain`):

    - Every step number listed must exist in the chain.
    - A step may appear in at most one group.
    """

    name: str = Field(..., min_length=1, description="Human-readable group label (used in errors).")
    steps: list[int] = Field(
        ...,
        min_length=1,
        description="Step numbers that receive this group's llm_config overrides.",
    )
    llm_config: LLMStepConfig = Field(
        ...,
        description="LLM config whose explicitly-set fields are merged into each member step.",
    )
