"""
Codex-as-step: delegate a whole subtask to a local Codex agent.

Runs a local Codex agent through the optional ``openai-codex`` Python SDK as
a CARL chain step. Unlike an LLM step (single completion) or an AGENT step
(CARL-managed ReAct loop over registered tools), a Codex step hands the task
to a fully autonomous coding agent with its own tool suite, sandboxed to
``read-only`` or ``workspace-write`` (full access is intentionally
unsupported) and with all approval requests denied in headless runs.

The interface deliberately mirrors the Claude Code step
(:mod:`mmar_carl.claude_code_step`):

- ``result``            — the agent's final answer text (goes into history)
- ``result_data``       — session_id (= thread_id), turn_id, status, usage,
  structured_output, sandbox details
- ``token_usage``       — mapped into CARL's {prompt, completion, total} convention
- structured output     — with ``output_schema`` the agent's final answer must
  contain a JSON payload matching the schema; the parsed object lands in
  ``result_data["structured_output"]``
- memory output         — ``output_memory_key`` writes the final answer (parsed
  object when a schema is set, text otherwise) into memory for downstream steps
- session continuation  — the thread id is written to memory namespace
  ``codex`` so a later step can resume the same Codex thread via
  ``resume_session="$memory.codex.step_<n>"``

Transport: the ``openai-codex`` SDK (``pip install 'mmar-carl[codex]'``);
authentication stays in the host's Codex installation (``codex login``) and
serialized chains never contain credentials.

Model classes live with the other step models:
:class:`~mmar_carl.models.config.CodexStepConfig` and
:class:`~mmar_carl.models.steps.CodexStepDescription`.

Known limitation vs the Claude Code step: no streaming — the SDK returns one
completed turn, so there is no ``stream`` option and ``on_llm_chunk`` is not
fed by this executor.
"""

from __future__ import annotations

import asyncio
import importlib.metadata
import json
import shutil
import time
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from ._optional_deps import require_openai_codex
from .models.config import CodexStepConfig
from .models.context import ReasoningContext
from .models.enums import Language, StepType
from .models.prompts import PromptTemplate
from .models.results import StepExecutionResult
from .models.steps import StepDescription
from .step_executors import (
    StepExecutorBase,
    _extract_json_payload,
    _structured_output_validator,
    _validate_structured_output,
    register_executor,
    resolve_context_reference,
)

__all__ = [
    "CODEX_RUNTIME_REQUIREMENT",
    "CodexRuntimeStatus",
    "CodexStepExecutor",
    "SESSION_MEMORY_NAMESPACE",
    "check_codex_runtime",
]

SESSION_MEMORY_NAMESPACE = "codex"

# Distribution that provides the ``openai_codex`` SDK — the identifier
# ReasoningChain.preflight reports for codex steps.
CODEX_RUNTIME_REQUIREMENT = "openai-codex"

_ERROR_DETAIL_LIMIT = 500


def _truncate_error(message: str) -> str:
    """Bound provider error text — backends can return whole HTML pages."""
    if len(message) <= _ERROR_DETAIL_LIMIT:
        return message
    return message[:_ERROR_DETAIL_LIMIT] + " …[truncated]"


def _load_codex_sdk() -> Any:
    """Lazy import seam kept separate so hermetic tests can replace the SDK."""

    return require_openai_codex()


class CodexRuntimeStatus(BaseModel):
    """Result of :func:`check_codex_runtime` — is a usable Codex runtime installed?"""

    available: bool = Field(..., description="True when the openai-codex SDK imports")
    sdk_version: Optional[str] = Field(default=None, description="Installed openai-codex distribution version")
    cli_path: Optional[str] = Field(
        default=None,
        description="Informational: `codex` binary found on PATH (auth is managed via `codex login`)",
    )
    error: Optional[str] = Field(default=None, description="Actionable reason when available is False")


def check_codex_runtime() -> CodexRuntimeStatus:
    """Validate that a local Codex runtime is usable.

    Checks that the optional ``openai-codex`` SDK imports and reports its
    version plus the location of the ``codex`` binary (informational — login
    state stays host-owned). Use before building/executing chains with
    ``codex`` steps; :meth:`ReasoningChain.preflight` runs the same check for
    every chain that references a codex step.
    """
    cli_path = shutil.which("codex")
    try:
        _load_codex_sdk()
    except ImportError as exc:
        return CodexRuntimeStatus(available=False, cli_path=cli_path, error=str(exc))
    try:
        sdk_version = importlib.metadata.version(CODEX_RUNTIME_REQUIREMENT)
    except importlib.metadata.PackageNotFoundError:
        sdk_version = None
    return CodexRuntimeStatus(available=True, sdk_version=sdk_version, cli_path=cli_path)


def _enum_value(value: Any) -> str:
    raw = getattr(value, "value", value)
    return str(raw)


def _model_dump(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return value
    return {
        name: getattr(value, name)
        for name in getattr(value, "__annotations__", {})
        if hasattr(value, name)
    }


class CodexStepExecutor(StepExecutorBase):
    """Executor that runs one local Codex turn through the ``openai-codex`` SDK."""

    async def execute(
        self,
        step: StepDescription,
        context: ReasoningContext,
        prompt_template: Optional[PromptTemplate] = None,
    ) -> StepExecutionResult:
        start_time = time.time()
        config: CodexStepConfig = step.step_config  # type: ignore[assignment]

        try:
            inputs = self._resolve_inputs(config, context)
            task_text = self._render_task(config, inputs)
            cwd = self._resolve_cwd(config)
            resume_id = self._resolve_resume_session(config, context)

            try:
                session_id, turn = await asyncio.wait_for(
                    self._run_turn(
                        config=config,
                        prompt=task_text,
                        cwd=cwd,
                        resume_id=resume_id,
                    ),
                    timeout=config.timeout,
                )
            except asyncio.TimeoutError:
                raise RuntimeError(
                    f"Codex turn timed out after {config.timeout}s"
                ) from None

            resumed = resume_id is not None
            error_message = self._diagnose_failure(turn)
            if error_message is not None:
                result_data = self._turn_result_data(config, session_id, turn, resumed, cwd)
                return self._failure(step, context, error_message, start_time, result_data)

            return self._success(step, context, config, session_id, turn, resumed, cwd, start_time)

        except Exception as e:
            return self._failure(step, context, _truncate_error(str(e)), start_time, result_data=None)

    # === Input resolution ===

    def _resolve_inputs(self, config: CodexStepConfig, context: ReasoningContext) -> dict[str, str]:
        resolved: dict[str, str] = {}
        for name, source in config.input_mapping.items():
            is_reference = source.startswith("$") or (
                len(source) >= 2 and source[0] == source[-1] and source[0] in "'\""
            )
            value = resolve_context_reference(source, context) if is_reference else source
            if value is None:
                value = ""
            if not isinstance(value, str):
                value = json.dumps(value, ensure_ascii=False, default=str)
            resolved[name] = value
        return resolved

    def _render_task(self, config: CodexStepConfig, inputs: dict[str, str]) -> str:
        if not inputs:
            rendered = config.task
        else:
            try:
                rendered = config.task.format(**inputs)
            except (KeyError, IndexError) as exc:
                raise ValueError(
                    f"task template placeholder {exc} has no matching input_mapping entry"
                ) from exc
        if len(rendered.encode("utf-8")) > config.max_input_bytes:
            raise ValueError(
                f"rendered task is {len(rendered.encode('utf-8'))} bytes, "
                f"exceeding max_input_bytes ({config.max_input_bytes})"
            )
        return rendered

    def _resolve_resume_session(
        self, config: CodexStepConfig, context: ReasoningContext
    ) -> Optional[str]:
        raw = config.resume_session
        if not raw:
            return None
        if raw.startswith("$"):
            value = resolve_context_reference(raw, context)
            if not value:
                raise ValueError(f"resume_session reference {raw!r} resolved to an empty value")
            return str(value).strip()
        return raw

    @staticmethod
    def _resolve_cwd(config: CodexStepConfig) -> Optional[str]:
        if config.cwd is None:
            return None
        path = Path(config.cwd).expanduser().resolve()
        if not path.is_dir():
            raise ValueError(f"CodexStep cwd is not an existing directory: {path}")
        return str(path)

    # === SDK invocation ===

    async def _run_turn(
        self,
        *,
        config: CodexStepConfig,
        prompt: str,
        cwd: Optional[str],
        resume_id: Optional[str],
    ) -> tuple[str, Any]:
        sdk = _load_codex_sdk()
        sandbox = {
            "read-only": sdk.Sandbox.read_only,
            "workspace-write": sdk.Sandbox.workspace_write,
        }[config.sandbox]
        approval_mode = sdk.ApprovalMode.deny_all

        async with sdk.AsyncCodex() as codex:
            thread_kwargs = {
                "approval_mode": approval_mode,
                "cwd": cwd,
                "developer_instructions": config.developer_instructions,
                "model": config.model,
                "sandbox": sandbox,
            }
            if resume_id is None:
                thread = await codex.thread_start(
                    **thread_kwargs,
                    ephemeral=config.ephemeral,
                )
            else:
                thread = await codex.thread_resume(resume_id, **thread_kwargs)

            run_kwargs: dict[str, Any] = {"sandbox": sandbox}
            if config.reasoning_effort is not None:
                # The SDK's generated enum accepts its stable wire value.
                run_kwargs["effort"] = config.reasoning_effort
            turn_result = await thread.run(prompt, **run_kwargs)
            return thread.id, turn_result

    # === Output diagnosis ===

    @staticmethod
    def _diagnose_failure(turn: Any) -> Optional[str]:
        """Return an error message, or None when the turn succeeded."""
        status = _enum_value(getattr(turn, "status", "failed"))
        error_message = getattr(getattr(turn, "error", None), "message", None)
        if status == "completed" and not error_message:
            return None
        return _truncate_error(error_message or f"Codex turn ended with status {status}")

    # === Result assembly ===

    @staticmethod
    def _token_usage(turn: Any) -> tuple[dict[str, int], Any]:
        usage = getattr(turn, "usage", None)
        total = getattr(usage, "total", None)
        if total is None:
            return {}, _model_dump(usage)
        prompt = int(getattr(total, "input_tokens", 0) or 0)
        completion = int(getattr(total, "output_tokens", 0) or 0)
        aggregate = int(getattr(total, "total_tokens", prompt + completion) or 0)
        return (
            {"prompt": prompt, "completion": completion, "total": aggregate},
            _model_dump(usage),
        )

    @staticmethod
    def _structured_output(schema: dict[str, Any], response: str) -> Any:
        """Parse and validate the final response against ``output_schema``.

        Raises ``ValueError`` (surfaced as step failure) when the response has
        no JSON payload or the payload violates the schema.
        """
        try:
            parsed = _extract_json_payload(response)
        except Exception as exc:
            raise ValueError(
                f"output_schema is set but the Codex response contains no JSON payload: {exc}"
            ) from exc
        validator = _structured_output_validator(schema)
        _validate_structured_output(parsed, validator)
        return parsed

    def _turn_result_data(
        self,
        config: CodexStepConfig,
        session_id: str,
        turn: Any,
        resumed: bool,
        cwd: Optional[str],
    ) -> dict[str, Any]:
        _, raw_usage = self._token_usage(turn)
        return {
            "session_id": session_id,
            "thread_id": session_id,  # Codex-domain alias for session_id
            "turn_id": getattr(turn, "id", None),
            "status": _enum_value(getattr(turn, "status", "failed")),
            "configured_model": config.model,
            "sandbox": config.sandbox,
            "cwd": cwd,
            "ephemeral": config.ephemeral if not resumed else None,
            "resumed": resumed,
            "usage": raw_usage,
        }

    def _history_entry(self, step: StepDescription, context: ReasoningContext, text: str) -> str:
        if context.language == Language.ENGLISH:
            return f"Step {step.number}. {step.title} [CODEX]\nResult: {text}\n"
        return f"Шаг {step.number}. {step.title} [CODEX]\nРезультат: {text}\n"

    def _success(
        self,
        step: StepDescription,
        context: ReasoningContext,
        config: CodexStepConfig,
        session_id: str,
        turn: Any,
        resumed: bool,
        cwd: Optional[str],
        start_time: float,
    ) -> StepExecutionResult:
        response = getattr(turn, "final_response", None) or ""

        structured_output: Any = None
        if config.output_schema is not None:
            # Raises ValueError → caught in execute() → failure result.
            structured_output = self._structured_output(config.output_schema, response)

        if session_id:
            key = config.store_session_key or f"step_{step.number}"
            context.memory_write(key, session_id, namespace=SESSION_MEMORY_NAMESPACE)

        if config.output_memory_key:
            output_value = structured_output if structured_output is not None else response
            context.memory_write(
                config.output_memory_key, output_value, namespace=config.output_namespace
            )

        token_usage, _ = self._token_usage(turn)
        result_data = self._turn_result_data(config, session_id, turn, resumed, cwd)
        result_data["structured_output"] = structured_output

        updated_history = context.history.copy()
        updated_history.append(self._history_entry(step, context, response))

        return StepExecutionResult(
            step_number=step.number,
            step_title=step.title,
            step_type=StepType.CODEX,
            result=response,
            result_data=result_data,
            success=True,
            execution_time=time.time() - start_time,
            updated_history=updated_history,
            token_usage=token_usage,
            model=config.model,
        )

    def _failure(
        self,
        step: StepDescription,
        context: ReasoningContext,
        error_message: str,
        start_time: float,
        result_data: Optional[dict[str, Any]],
    ) -> StepExecutionResult:
        return StepExecutionResult(
            step_number=step.number,
            step_title=step.title,
            step_type=StepType.CODEX,
            result="",
            result_data=result_data,
            success=False,
            error_message=error_message,
            execution_time=time.time() - start_time,
            updated_history=context.history.copy(),
        )


# Importing ``mmar_carl`` installs the built-in executor. Users can still
# replace it through the normal register_executor extension point.
register_executor(StepType.CODEX, CodexStepExecutor())
