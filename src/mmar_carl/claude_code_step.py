"""
Claude-Code-as-step: delegate a whole subtask to a headless Claude Code agent.

Runs the ``claude`` CLI in print mode (``claude -p``) as a CARL chain step.
Unlike an LLM step (single completion) or an AGENT step (CARL-managed ReAct
loop over registered tools), a Claude Code step hands the task to a fully
autonomous coding agent with its own tool suite (file access, shell, web,
MCP) and collects the structured result:

- ``result``            — the agent's final answer text (goes into history)
- ``result_data``       — session_id, num_turns, total_cost_usd, usage,
  structured_output, raw payload
- ``token_usage``       — mapped into CARL's {prompt, completion, total} convention
- streaming             — with ``stream=True`` the executor switches to
  ``--output-format stream-json`` and forwards each assistant text block to
  ``context.on_llm_chunk`` as it arrives
- structured output     — with ``output_schema`` the agent's final answer must
  contain a JSON payload matching the schema; the parsed object lands in
  ``result_data["structured_output"]``
- memory output         — ``output_memory_key`` writes the final answer (parsed
  object when a schema is set, text otherwise) into memory for downstream steps
- session continuation  — the session id is written to memory namespace
  ``claude_code`` so a later step can resume the same agent session via
  ``resume_session="$memory.claude_code.step_<n>"``

Transport: subprocess ``claude -p <task>``. No extra Python dependency —
requires the Claude Code CLI on PATH (or ``cli_path``).

Model classes live with the other step models:
:class:`~mmar_carl.models.config.ClaudeCodeStepConfig` and
:class:`~mmar_carl.models.steps.ClaudeCodeStepDescription`.
"""

import asyncio
import json
import os
import shutil
import subprocess
import time
from collections import deque
from typing import Any, Optional

from pydantic import BaseModel, Field

from .models.config import ClaudeCodeStepConfig
from .models.context import ReasoningContext
from .models.enums import Language, StepType
from .models.prompts import PromptTemplate
from .models.results import StepExecutionResult
from .models.steps import ClaudeCodeStepDescription, StepDescription
from .step_executors import (
    StepExecutorBase,
    _dispatch_llm_chunk,
    _extract_json_payload,
    _structured_output_validator,
    _validate_structured_output,
    register_executor,
    resolve_context_reference,
)

__all__ = [
    "ClaudeCodeCliStatus",
    "ClaudeCodeStepConfig",
    "ClaudeCodeStepDescription",
    "ClaudeCodeStepExecutor",
    "SESSION_MEMORY_NAMESPACE",
    "check_claude_code_cli",
]

SESSION_MEMORY_NAMESPACE = "claude_code"


class ClaudeCodeCliStatus(BaseModel):
    """Result of :func:`check_claude_code_cli` — is a usable CLI installed?"""

    available: bool = Field(..., description="True when the CLI resolves (and the version probe passed, if run)")
    cli_path: str = Field(..., description="The requested executable (name on PATH or path)")
    resolved_path: Optional[str] = Field(default=None, description="Absolute path shutil.which resolved to")
    version: Optional[str] = Field(default=None, description="First line of `<cli> --version` output when probed")
    error: Optional[str] = Field(default=None, description="Actionable reason when available is False")


def check_claude_code_cli(
    cli_path: str = "claude",
    *,
    probe_version: bool = True,
    timeout: float = 10.0,
) -> ClaudeCodeCliStatus:
    """Validate that a local Claude Code CLI installation is usable.

    Resolves ``cli_path`` (bare name against PATH, or a direct file path) and,
    unless ``probe_version=False``, runs ``<cli> --version`` to confirm the
    executable actually responds. Use before building/executing chains with
    ``claude_code`` steps; :meth:`ReasoningChain.preflight` performs the
    cheaper existence-only check for every referenced ``cli_path``.
    """
    resolved = shutil.which(cli_path)
    if resolved is None:
        return ClaudeCodeCliStatus(
            available=False,
            cli_path=cli_path,
            error=(
                f"Claude Code CLI not found at {cli_path!r}. "
                "Install it (npm install -g @anthropic-ai/claude-code) or set cli_path."
            ),
        )
    if not probe_version:
        return ClaudeCodeCliStatus(available=True, cli_path=cli_path, resolved_path=resolved)

    try:
        probe = subprocess.run(
            [resolved, "--version"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return ClaudeCodeCliStatus(
            available=False,
            cli_path=cli_path,
            resolved_path=resolved,
            error=f"Claude Code CLI version probe failed: {exc}",
        )
    if probe.returncode != 0:
        detail = (probe.stderr or probe.stdout or "").strip()[:200]
        return ClaudeCodeCliStatus(
            available=False,
            cli_path=cli_path,
            resolved_path=resolved,
            error=f"Claude Code CLI version probe exited with code {probe.returncode}: {detail}",
        )
    stdout = (probe.stdout or "").strip()
    version = stdout.splitlines()[0] if stdout else None
    return ClaudeCodeCliStatus(
        available=True, cli_path=cli_path, resolved_path=resolved, version=version
    )


# Diagnostic context kept from a streaming run (last N raw stdout lines).
_STREAM_TAIL_LINES = 20


class ClaudeCodeStepExecutor(StepExecutorBase):
    """Executor that runs ``claude -p <task>`` as a subprocess.

    Non-streaming runs use ``--output-format json`` (one result object on
    stdout). Streaming runs use ``--output-format stream-json --verbose``
    (JSONL events) and forward assistant text to ``context.on_llm_chunk``.
    """

    async def execute(
        self,
        step: StepDescription,
        context: ReasoningContext,
        prompt_template: Optional[PromptTemplate] = None,
    ) -> StepExecutionResult:
        start_time = time.time()
        config: ClaudeCodeStepConfig = step.step_config  # type: ignore[assignment]

        try:
            inputs = self._resolve_inputs(config, context)
            task_text = self._render_task(config, inputs)
            resume_id = self._resolve_resume_session(config, context)
            argv = self._build_argv(config, task_text, resume_id)

            if config.stream:
                returncode, payload, stdout, stderr = await self._run_cli_streaming(
                    argv, config, context, step
                )
            else:
                returncode, stdout, stderr = await self._run_cli(argv, config)
                payload = self._parse_payload(stdout)

            error_message = self._diagnose_failure(returncode, payload, stdout, stderr)
            if error_message is not None:
                return self._failure(step, context, error_message, start_time, payload)

            assert payload is not None  # _diagnose_failure guarantees this
            return self._success(step, context, config, payload, start_time)

        except Exception as e:
            return self._failure(step, context, str(e), start_time, payload=None)

    # === Input resolution ===

    def _resolve_inputs(self, config: ClaudeCodeStepConfig, context: ReasoningContext) -> dict[str, str]:
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

    def _render_task(self, config: ClaudeCodeStepConfig, inputs: dict[str, str]) -> str:
        if not inputs:
            return config.task
        try:
            return config.task.format(**inputs)
        except (KeyError, IndexError) as exc:
            raise ValueError(
                f"task template placeholder {exc} has no matching input_mapping entry"
            ) from exc

    def _resolve_resume_session(
        self, config: ClaudeCodeStepConfig, context: ReasoningContext
    ) -> Optional[str]:
        raw = config.resume_session
        if not raw:
            return None
        if raw.startswith("$"):
            value = resolve_context_reference(raw, context)
            if not value:
                raise ValueError(f"resume_session reference {raw!r} resolved to an empty value")
            return str(value)
        return raw

    # === CLI invocation ===

    def _build_argv(
        self, config: ClaudeCodeStepConfig, task_text: str, resume_id: Optional[str]
    ) -> list[str]:
        argv = [config.cli_path, "-p", task_text]
        if config.stream:
            # Print mode requires --verbose for stream-json output.
            argv += ["--output-format", "stream-json", "--verbose"]
        else:
            argv += ["--output-format", "json"]
        if config.model:
            argv += ["--model", config.model]
        if config.max_turns is not None:
            argv += ["--max-turns", str(config.max_turns)]
        if config.allowed_tools:
            argv += ["--allowedTools", ",".join(config.allowed_tools)]
        if config.disallowed_tools:
            argv += ["--disallowedTools", ",".join(config.disallowed_tools)]
        if config.permission_mode:
            argv += ["--permission-mode", config.permission_mode]
        if config.system_prompt:
            argv += ["--system-prompt", config.system_prompt]
        if config.append_system_prompt:
            argv += ["--append-system-prompt", config.append_system_prompt]
        for directory in config.add_dirs:
            argv += ["--add-dir", directory]
        if resume_id:
            argv += ["--resume", resume_id]
        argv += config.extra_cli_args
        return argv

    async def _spawn(self, argv: list[str], config: ClaudeCodeStepConfig) -> asyncio.subprocess.Process:
        env = {**os.environ, **config.env}
        try:
            return await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=config.cwd,
                env=env,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Claude Code CLI not found at {config.cli_path!r}. "
                "Install it (npm install -g @anthropic-ai/claude-code) or set cli_path."
            ) from exc

    async def _run_cli(
        self, argv: list[str], config: ClaudeCodeStepConfig
    ) -> tuple[int, str, str]:
        process = await self._spawn(argv, config)
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(), timeout=config.timeout
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            raise RuntimeError(
                f"Claude Code CLI timed out after {config.timeout}s"
            ) from None

        return (
            process.returncode or 0,
            stdout_bytes.decode("utf-8", errors="replace"),
            stderr_bytes.decode("utf-8", errors="replace"),
        )

    async def _run_cli_streaming(
        self,
        argv: list[str],
        config: ClaudeCodeStepConfig,
        context: ReasoningContext,
        step: StepDescription,
    ) -> tuple[int, Optional[dict[str, Any]], str, str]:
        """Run the CLI in stream-json mode.

        Returns ``(returncode, result_payload, stdout_tail, stderr)`` where
        ``stdout_tail`` is the last few raw lines kept for failure diagnosis.
        """
        process = await self._spawn(argv, config)

        async def _consume_stdout() -> tuple[Optional[dict[str, Any]], list[str]]:
            payload: Optional[dict[str, Any]] = None
            tail: deque[str] = deque(maxlen=_STREAM_TAIL_LINES)
            assert process.stdout is not None
            while True:
                line_bytes = await process.stdout.readline()
                if not line_bytes:
                    break
                line = line_bytes.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                tail.append(line)
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(event, dict):
                    continue
                event_type = event.get("type")
                if event_type == "assistant":
                    for text in self._assistant_texts(event):
                        _dispatch_llm_chunk(
                            context.on_llm_chunk,
                            text,
                            step_number=step.number,
                            stage="claude_code",
                        )
                elif event_type == "result":
                    payload = event
            return payload, list(tail)

        async def _drain_stderr() -> bytes:
            assert process.stderr is not None
            return await process.stderr.read()

        try:
            (payload, tail), stderr_bytes = await asyncio.wait_for(
                asyncio.gather(_consume_stdout(), _drain_stderr()),
                timeout=config.timeout,
            )
            await process.wait()
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            raise RuntimeError(
                f"Claude Code CLI timed out after {config.timeout}s"
            ) from None

        return (
            process.returncode or 0,
            payload,
            "\n".join(tail),
            stderr_bytes.decode("utf-8", errors="replace"),
        )

    @staticmethod
    def _assistant_texts(event: dict[str, Any]) -> list[str]:
        """Extract text blocks from a stream-json assistant event."""
        message = event.get("message")
        if not isinstance(message, dict):
            return []
        content = message.get("content")
        if not isinstance(content, list):
            return []
        texts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str) and text:
                    texts.append(text)
        return texts

    # === Output parsing ===

    @staticmethod
    def _parse_payload(stdout: str) -> Optional[dict[str, Any]]:
        text = stdout.strip()
        if not text:
            return None
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        # Defensive: tolerate stray non-JSON lines before the result object
        for line in reversed(text.splitlines()):
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                continue
        return None

    @staticmethod
    def _diagnose_failure(
        returncode: int,
        payload: Optional[dict[str, Any]],
        stdout: str,
        stderr: str,
    ) -> Optional[str]:
        """Return an error message, or None when the run succeeded."""
        if returncode != 0:
            detail = (payload or {}).get("result") or stderr.strip() or stdout.strip()
            return f"Claude Code CLI exited with code {returncode}: {detail[:500]}"
        if payload is None:
            detail = stderr.strip() or stdout.strip()
            return f"Claude Code CLI did not return parseable JSON: {detail[:500]}"
        if payload.get("is_error"):
            subtype = payload.get("subtype", "unknown")
            detail = str(payload.get("result", ""))[:500]
            return f"Claude Code run failed ({subtype}): {detail}"
        return None

    # === Result assembly ===

    @staticmethod
    def _token_usage(payload: dict[str, Any]) -> dict[str, int]:
        usage = payload.get("usage") or {}
        prompt = (
            int(usage.get("input_tokens", 0))
            + int(usage.get("cache_creation_input_tokens", 0))
            + int(usage.get("cache_read_input_tokens", 0))
        )
        completion = int(usage.get("output_tokens", 0))
        return {"prompt": prompt, "completion": completion, "total": prompt + completion}

    @staticmethod
    def _model_name(payload: dict[str, Any]) -> Optional[str]:
        model_usage = payload.get("modelUsage")
        if isinstance(model_usage, dict) and model_usage:
            return next(iter(model_usage))
        return None

    @staticmethod
    def _structured_output(schema: dict[str, Any], result_text: str) -> Any:
        """Parse and validate the agent's answer against ``output_schema``.

        Raises ``ValueError`` (surfaced as step failure) when the answer has
        no JSON payload or the payload violates the schema.
        """
        try:
            parsed = _extract_json_payload(result_text)
        except Exception as exc:
            raise ValueError(
                f"output_schema is set but the agent's answer contains no JSON payload: {exc}"
            ) from exc
        validator = _structured_output_validator(schema)
        _validate_structured_output(parsed, validator)
        return parsed

    def _history_entry(self, step: StepDescription, context: ReasoningContext, text: str) -> str:
        if context.language == Language.ENGLISH:
            return f"Step {step.number}. {step.title} [CLAUDE CODE]\nResult: {text}\n"
        return f"Шаг {step.number}. {step.title} [CLAUDE CODE]\nРезультат: {text}\n"

    def _success(
        self,
        step: StepDescription,
        context: ReasoningContext,
        config: ClaudeCodeStepConfig,
        payload: dict[str, Any],
        start_time: float,
    ) -> StepExecutionResult:
        full_text = str(payload.get("result", ""))

        structured_output: Any = None
        if config.output_schema is not None:
            # Raises ValueError → caught in execute() → failure result.
            structured_output = self._structured_output(config.output_schema, full_text)

        result_text = full_text
        if len(result_text) > config.max_output_chars:
            result_text = result_text[: config.max_output_chars] + "\n…[truncated]"

        session_id = payload.get("session_id")
        if session_id:
            key = config.store_session_key or f"step_{step.number}"
            context.memory_write(key, session_id, namespace=SESSION_MEMORY_NAMESPACE)

        if config.output_memory_key:
            output_value = structured_output if structured_output is not None else full_text
            context.memory_write(
                config.output_memory_key, output_value, namespace=config.output_namespace
            )

        result_data = {
            "session_id": session_id,
            "num_turns": payload.get("num_turns"),
            "total_cost_usd": payload.get("total_cost_usd"),
            "duration_ms": payload.get("duration_ms"),
            "usage": payload.get("usage"),
            "structured_output": structured_output,
            "payload": payload,
        }

        updated_history = context.history.copy()
        updated_history.append(self._history_entry(step, context, result_text))

        return StepExecutionResult(
            step_number=step.number,
            step_title=step.title,
            step_type=StepType.CLAUDE_CODE,
            result=result_text,
            result_data=result_data,
            success=True,
            execution_time=time.time() - start_time,
            updated_history=updated_history,
            token_usage=self._token_usage(payload),
            model=self._model_name(payload),
        )

    def _failure(
        self,
        step: StepDescription,
        context: ReasoningContext,
        error_message: str,
        start_time: float,
        payload: Optional[dict[str, Any]],
    ) -> StepExecutionResult:
        return StepExecutionResult(
            step_number=step.number,
            step_title=step.title,
            step_type=StepType.CLAUDE_CODE,
            result="",
            result_data={"payload": payload} if payload else None,
            success=False,
            error_message=error_message,
            execution_time=time.time() - start_time,
            updated_history=context.history.copy(),
        )


register_executor(StepType.CLAUDE_CODE, ClaudeCodeStepExecutor())
