"""Runtime adapter for invoking a serialized CARL chain as a registered tool."""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import hashlib
import json
import time
import uuid
from typing import Any

from .models.chain_tool import (
    ChainToolDefinition,
    ChainToolOutcome,
    ChainToolStatus,
    validate_chain_tool_value,
)
from .models.context import ReasoningContext
from .tool_definition import ToolDefinition

_CHAIN_TOOL_STACK: contextvars.ContextVar[tuple[str, ...]] = contextvars.ContextVar(
    "carl_chain_tool_stack",
    default=(),
)
_POLL_SECONDS = 0.05


class ChainToolRuntime:
    """Callable runtime bound to one host context and one immutable definition."""

    def __init__(
        self,
        definition: ChainToolDefinition,
        parent_context: ReasoningContext,
    ) -> None:
        self.definition = definition
        self.parent_context = parent_context
        self.__carl_tool_definition__ = ToolDefinition(
            name=definition.name,
            description=definition.description,
            parameters=definition.input_schema,
        )
        self.__carl_chain_tool_sha256__ = definition.snapshot_sha256
        self.__carl_chain_tool_contract_sha256__ = definition.calculate_contract_sha256()

    async def __call__(self, **arguments: Any) -> dict[str, Any]:
        """Run the embedded chain and return a JSON-compatible typed envelope."""
        outcome = await self.invoke(**arguments)
        return outcome.model_dump(mode="json")

    async def invoke(self, **arguments: Any) -> ChainToolOutcome:
        """Run the embedded snapshot with isolated mutable state."""
        started = time.monotonic()
        invocation_id = uuid.uuid4().hex
        stack = _CHAIN_TOOL_STACK.get()
        depth = len(stack) + 1
        input_sha256: str | None = None

        if self.definition.calculate_snapshot_sha256() != self.definition.snapshot_sha256:
            return self._outcome(
                invocation_id=invocation_id,
                depth=depth,
                status=ChainToolStatus.UNAVAILABLE,
                started=started,
                error_code="snapshot_digest_mismatch",
                error_message="embedded chain snapshot changed after validation",
            )

        try:
            arguments = _json_copy(arguments)
            input_sha256 = _json_sha256(arguments)
        except (TypeError, ValueError) as exc:
            return self._outcome(
                invocation_id=invocation_id,
                depth=depth,
                status=ChainToolStatus.INVALID_INPUT,
                started=started,
                error_code="input_not_json",
                error_message=str(exc),
            )

        if _json_size_bytes(arguments) > self.definition.max_input_bytes:
            return self._outcome(
                invocation_id=invocation_id,
                depth=depth,
                status=ChainToolStatus.INVALID_INPUT,
                started=started,
                error_code="input_too_large",
                error_message=f"input exceeds {self.definition.max_input_bytes} bytes",
                input_sha256=input_sha256,
            )

        try:
            validate_chain_tool_value(arguments, self.definition.input_schema)
        except (TypeError, ValueError) as exc:
            return self._outcome(
                invocation_id=invocation_id,
                depth=depth,
                status=ChainToolStatus.INVALID_INPUT,
                started=started,
                error_code="input_schema_mismatch",
                error_message=str(exc),
                input_sha256=input_sha256,
            )

        digest = self.definition.snapshot_sha256
        if digest in stack or depth > self.definition.max_depth:
            return self._outcome(
                invocation_id=invocation_id,
                depth=depth,
                status=ChainToolStatus.RECURSION_LIMIT,
                started=started,
                error_code="recursion_limit",
                error_message=(
                    "embedded chain recursion cycle detected"
                    if digest in stack
                    else f"maximum nested-chain depth {self.definition.max_depth} exceeded"
                ),
                input_sha256=input_sha256,
            )

        if self.parent_context.is_cancelled():
            return self._outcome(
                invocation_id=invocation_id,
                depth=depth,
                status=ChainToolStatus.CANCELLED,
                started=started,
                error_code="cancelled_by_host",
                error_message="parent execution was cancelled before invocation",
                input_sha256=input_sha256,
            )

        missing = [name for name in self.definition.allowed_tools if not self.parent_context.has_tool(name)]
        if missing:
            return self._outcome(
                invocation_id=invocation_id,
                depth=depth,
                status=ChainToolStatus.UNAVAILABLE,
                started=started,
                error_code="missing_host_tools",
                error_message=f"missing host tools: {', '.join(missing)}",
                input_sha256=input_sha256,
            )

        try:
            from .chain import ReasoningChain

            child_chain = ReasoningChain.from_dict_typed(self.definition.chain_snapshot)
            child_context = self._make_child_context(arguments)
        except Exception as exc:  # noqa: BLE001 - snapshot validation is normalized
            return self._outcome(
                invocation_id=invocation_id,
                depth=depth,
                status=ChainToolStatus.UNAVAILABLE,
                started=started,
                error_code="invalid_chain_snapshot",
                error_message=f"{type(exc).__name__}: {exc}",
                input_sha256=input_sha256,
            )

        token = _CHAIN_TOOL_STACK.set((*stack, digest))
        task = asyncio.create_task(child_chain.execute_async(child_context))
        deadline = started + self.definition.timeout_seconds
        try:
            while True:
                done, _ = await asyncio.wait({task}, timeout=_POLL_SECONDS)
                if task in done:
                    result = task.result()
                    break
                if self.parent_context.is_cancelled():
                    await _cancel_owned_task(task, child_context)
                    return self._outcome(
                        invocation_id=invocation_id,
                        depth=depth,
                        status=ChainToolStatus.CANCELLED,
                        started=started,
                        error_code="cancelled_by_host",
                        error_message="parent execution cancelled the nested chain",
                        input_sha256=input_sha256,
                    )
                if time.monotonic() >= deadline:
                    await _cancel_owned_task(task, child_context)
                    return self._outcome(
                        invocation_id=invocation_id,
                        depth=depth,
                        status=ChainToolStatus.TIMED_OUT,
                        started=started,
                        error_code="chain_tool_timeout",
                        error_message=(f"nested chain exceeded {self.definition.timeout_seconds}s"),
                        input_sha256=input_sha256,
                    )
        except asyncio.CancelledError:
            await _cancel_owned_task(task, child_context)
            raise
        except Exception as exc:  # noqa: BLE001 - nested runtime failures are normalized
            if not task.done():
                await _cancel_owned_task(task, child_context)
            return self._outcome(
                invocation_id=invocation_id,
                depth=depth,
                status=ChainToolStatus.FAILED,
                started=started,
                error_code="nested_execution_exception",
                error_message=f"{type(exc).__name__}: {exc}",
                input_sha256=input_sha256,
            )
        finally:
            _CHAIN_TOOL_STACK.reset(token)

        if not result.success:
            return self._outcome(
                invocation_id=invocation_id,
                depth=depth,
                status=ChainToolStatus.FAILED,
                started=started,
                error_code="nested_chain_failed",
                error_message=result.error or "nested chain returned success=false",
                token_usage=result.token_usage,
                executed_steps=len(result.step_results),
                input_sha256=input_sha256,
            )

        try:
            from .step_executors import resolve_context_reference

            output = resolve_context_reference(
                self.definition.output_reference,
                child_context,
            )
            output = _json_copy(output)
        except (TypeError, ValueError) as exc:
            return self._outcome(
                invocation_id=invocation_id,
                depth=depth,
                status=ChainToolStatus.INVALID_OUTPUT,
                started=started,
                error_code="output_not_json",
                error_message=str(exc),
                token_usage=result.token_usage,
                executed_steps=len(result.step_results),
                input_sha256=input_sha256,
            )

        if _json_size_bytes(output) > self.definition.max_output_bytes:
            return self._outcome(
                invocation_id=invocation_id,
                depth=depth,
                status=ChainToolStatus.INVALID_OUTPUT,
                started=started,
                error_code="output_too_large",
                error_message=f"output exceeds {self.definition.max_output_bytes} bytes",
                token_usage=result.token_usage,
                executed_steps=len(result.step_results),
                input_sha256=input_sha256,
            )

        try:
            validate_chain_tool_value(output, self.definition.output_schema)
        except (TypeError, ValueError) as exc:
            return self._outcome(
                invocation_id=invocation_id,
                depth=depth,
                status=ChainToolStatus.INVALID_OUTPUT,
                started=started,
                error_code="output_schema_mismatch",
                error_message=str(exc),
                token_usage=result.token_usage,
                executed_steps=len(result.step_results),
                input_sha256=input_sha256,
            )

        return self._outcome(
            invocation_id=invocation_id,
            depth=depth,
            status=ChainToolStatus.COMPLETED,
            started=started,
            output=output,
            output_sha256=_json_sha256(output),
            token_usage=result.token_usage,
            executed_steps=len(result.step_results),
            input_sha256=input_sha256,
        )

    def _make_child_context(self, arguments: dict[str, Any]) -> ReasoningContext:
        parent = self.parent_context
        child = ReasoningContext(
            outer_context=json.dumps(
                arguments,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
            ),
            api=parent.api,
            model=parent.model,
            retry_max=parent.retry_max,
            language=parent.language,
            system_prompt=parent.system_prompt,
            memory={},
            max_history_entries=parent.max_history_entries,
            trim_strategy=parent.trim_strategy,
        )
        for name in self.definition.allowed_tools:
            tool = parent.get_tool(name)
            if tool is not None:
                child.register_tool(
                    name,
                    tool,
                    tags=sorted(parent.get_tool_tags(name)),
                )
        return child

    def _outcome(
        self,
        *,
        invocation_id: str,
        depth: int,
        status: ChainToolStatus,
        started: float,
        output: Any = None,
        input_sha256: str | None = None,
        output_sha256: str | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
        token_usage: dict[str, int] | None = None,
        executed_steps: int = 0,
    ) -> ChainToolOutcome:
        return ChainToolOutcome(
            invocation_id=invocation_id,
            tool_name=self.definition.name,
            snapshot_sha256=self.definition.snapshot_sha256,
            depth=depth,
            status=status,
            success=status is ChainToolStatus.COMPLETED,
            input_sha256=input_sha256,
            output_sha256=output_sha256,
            output=output,
            error_code=error_code,
            error_message=error_message,
            duration_seconds=max(0.0, time.monotonic() - started),
            token_usage=token_usage or {},
            executed_steps=executed_steps,
        )


async def _cancel_owned_task(
    task: asyncio.Task[Any],
    context: ReasoningContext,
) -> None:
    context.cancel()
    if not task.done():
        task.cancel()
    with contextlib.suppress(asyncio.CancelledError, Exception):
        await task


def _json_copy(value: Any) -> Any:
    encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
    return json.loads(encoded)


def _json_size_bytes(value: Any) -> int:
    return len(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = ["ChainToolRuntime"]
