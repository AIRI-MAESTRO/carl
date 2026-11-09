"""Experimental dependency-scoped execution for host-approved async tools."""

from __future__ import annotations

import asyncio
import heapq
import inspect
import math
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any

from ._dependency_graph import dependency_back_edges
from .models.context import ReasoningContext
from .models.enums import StepType
from .models.results import StepExecutionResult
from .models.steps import ToolStepDescription
from .step_executors import ToolStepExecutor, resolve_context_reference

type AsyncTool = Callable[..., Awaitable[Any]]

_STEP_REFERENCE = re.compile(r"\$steps\.(0|-?[1-9][0-9]*)\.(?:result|result_data(?:\.[^.]+)*)")
_OUTER_REFERENCE = re.compile(r"\$outer_context(?:\.[^.]+)*")
_MISSING = object()


def _json_copy(value: Any) -> Any:
    """Copy JSON data without coercing types or accepting non-finite numbers."""
    value_type = type(value)
    if value is None or value_type in (str, bool, int):
        return value
    if value_type is float and math.isfinite(value):
        return value
    if value_type is list:
        return [_json_copy(item) for item in value]
    if value_type is dict and all(type(key) is str for key in value):
        return {key: _json_copy(item) for key, item in value.items()}
    raise ValueError("Expected JSON data with string keys and finite numbers")


def _referenced_steps(step: ToolStepDescription) -> tuple[int, ...]:
    references: dict[int, None] = {}
    for source in step.config.input_mapping.values():
        match = _STEP_REFERENCE.fullmatch(source)
        if match:
            references[int(match[1])] = None
        elif not (
            _OUTER_REFERENCE.fullmatch(source) or (len(source) >= 2 and source[0] in "\"'" and source[-1] == source[0])
        ):
            raise ValueError(f"Step {step.number} has unsupported input reference {source!r}")

    return tuple(references)


def _validate_reference_scope(
    references: Mapping[int, tuple[int, ...]],
    dependencies: Mapping[int, list[int]],
    dependents: Mapping[int, list[int]],
) -> None:
    consumers: dict[int, set[int]] = {}
    for number, sources in references.items():
        if sources:
            parents = set(dependencies[number])
            for source in sources:
                if source not in parents:
                    consumers.setdefault(source, set()).add(number)

    unresolved: dict[int, list[int]] = {}
    for source, outstanding in consumers.items():
        # One traversal serves every transitive reference to this producer.
        seen = {source}
        pending = [source] if source in dependents else []
        while pending and outstanding:
            for child in dependents[pending.pop()]:
                if child not in seen:
                    seen.add(child)
                    outstanding.discard(child)
                    pending.append(child)
        for number in outstanding:
            unresolved.setdefault(number, []).append(source)

    if unresolved:
        number = next(number for number in references if number in unresolved)
        raise ValueError(f"Step {number} references non-ancestor steps: {sorted(unresolved[number])}")


def _validate_steps(
    steps: Sequence[ToolStepDescription], tools: Mapping[str, AsyncTool]
) -> tuple[dict[int, tuple[int, ...]], dict[int, list[int]]]:
    dependencies: dict[int, list[int]] = {}
    for step in steps:
        if type(step) is not ToolStepDescription:
            raise ValueError("Eager execution requires ToolStepDescription steps")
        if step.number in dependencies:
            raise ValueError(f"Duplicate step number {step.number}")
        if len(set(step.dependencies)) != len(step.dependencies):
            raise ValueError(f"Step {step.number} declares a dependency more than once")
        dependencies[step.number] = step.dependencies

        config = step.config
        if (
            step.triggered_by
            or step.checkpoint
            or step.checkpoint_name is not None
            or step.replan_enabled is not None
            or step.loop_back_to is not None
            or step.loop_config is not None
            or step.cache is not None
            or step.metrics
            or step.metadata
            or config.error_recovery is not None
            or config.allowed_tool_tags is not None
            or config.output_key != "result"
            or not config.retry_on_error
        ):
            raise ValueError(f"Step {step.number} uses unsupported eager execution options")
        if not math.isfinite(config.timeout) or config.timeout <= 0:
            raise ValueError(f"Step {step.number} requires a finite, positive tool timeout")
        tool = tools.get(config.tool_name)
        if not (inspect.isfunction(tool) and tool.__code__.co_flags & inspect.CO_COROUTINE):
            raise ValueError(f"Step {step.number} requires a host-approved async function for {config.tool_name!r}")

    dependents: dict[int, list[int]] = {number: [] for number in dependencies}
    for number, parents in dependencies.items():
        for parent in parents:
            if parent not in dependencies:
                raise ValueError(f"Step {number} has missing dependency {parent}")
            dependents[parent].append(number)
    cycle = next(dependency_back_edges(dependencies), None)
    if cycle is not None:
        raise ValueError(f"Cyclic eager tool dependencies from step {cycle[0]} through {cycle[1]}")
    references = {step.number: _referenced_steps(step) for step in steps}
    _validate_reference_scope(references, dependencies, dependents)
    return references, dependents


class _EagerToolExecutor(ToolStepExecutor):
    def __init__(self, run_deadline: float | None) -> None:
        self._run_deadline = run_deadline

    def _resolve_input_value(self, source: str, context: ReasoningContext) -> Any:
        value = resolve_context_reference(source, context, default=_MISSING)
        if value is _MISSING:
            raise ValueError(f"Missing eager tool input {source!r}")
        return value

    async def _call_tool(self, tool_callable: AsyncTool, kwargs: dict[str, Any], timeout: float) -> Any:
        inputs = _json_copy(kwargs)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        start_deadline = deadline if self._run_deadline is None else min(deadline, self._run_deadline)
        finished_at = math.inf

        async def invoke() -> Any:
            nonlocal finished_at
            try:
                if loop.time() >= start_deadline:
                    raise TimeoutError
                return await tool_callable(**inputs)
            finally:
                finished_at = loop.time()

        call = asyncio.create_task(invoke())
        try:
            done, _ = await asyncio.wait((call,), timeout=max(0, deadline - loop.time()))
            if not done or finished_at >= deadline:
                raise TimeoutError
        except (asyncio.CancelledError, TimeoutError):
            # Only this supervisor cancels the tool. Waiting via asyncio.wait
            # keeps a run-level cancellation from interrupting tool cleanup.
            call.cancel()
            cancelled_during_cleanup = False
            while not call.done():
                try:
                    await asyncio.wait((call,))
                except asyncio.CancelledError:
                    cancelled_during_cleanup = True
            if not call.cancelled():
                call.exception()
            if cancelled_during_cleanup:
                raise asyncio.CancelledError
            raise
        return call.result()


def _failed(step: ToolStepDescription, message: str, *, skipped: bool = False) -> StepExecutionResult:
    return StepExecutionResult(
        step_number=step.number,
        step_title=step.title,
        step_type=StepType.TOOL,
        result="",
        success=False,
        skipped=skipped,
        error_message=message,
    )


async def _execute_step(
    step: ToolStepDescription,
    tool: AsyncTool,
    outer_context: dict[str, Any],
    referenced_results: dict[str, dict[str, Any]],
    run_deadline: float | None,
) -> StepExecutionResult:
    context = ReasoningContext(
        api=None,
        outer_context="",
        runtime_input=outer_context,
        metadata={"step_results": referenced_results},
    )
    context.register_tool(step.config.tool_name, tool)
    result = await _EagerToolExecutor(run_deadline).execute(step, context)
    if result.success:
        try:
            result.result_data = _json_copy(result.result_data)
        except (ValueError, RecursionError):
            return _failed(step, "Tool result must be JSON data with string keys and finite numbers")
    return result


async def execute_eager_tools(
    steps: Sequence[ToolStepDescription],
    *,
    tools: Mapping[str, AsyncTool],
    outer_context: dict[str, Any] | None = None,
    max_concurrency: int = 4,
    timeout: float | None = None,
) -> list[StepExecutionResult]:
    """Run a static tool graph with bounded overlap and return results in declaration order.

    The host supplies trusted, stateless coroutine functions that cooperate with
    cancellation. Inputs can use quoted string literals, JSON outer context, or
    the result text/data of declared ancestors. Dependency failures skip their
    descendants. The optional run timeout raises TimeoutError after tools drain.
    """
    if type(max_concurrency) is not int or max_concurrency <= 0:
        raise ValueError("max_concurrency must be a positive integer")
    if timeout is not None and (not math.isfinite(timeout) or timeout <= 0):
        raise ValueError("timeout must be finite and positive")
    if outer_context is not None and type(outer_context) is not dict:
        raise ValueError("outer_context must be a JSON object")
    try:
        inputs = _json_copy(outer_context if outer_context is not None else {})
    except RecursionError as exc:
        raise ValueError("outer_context must be finite, acyclic JSON data") from exc
    tools = dict(tools)
    steps = tuple(steps)
    references, dependents = _validate_steps(steps, tools)
    nodes = {step.number: step.model_copy(deep=True) for step in steps}
    order = {number: index for index, number in enumerate(nodes)}
    pending = {number: len(step.dependencies) for number, step in nodes.items()}
    ready = [(order[number], number) for number, count in pending.items() if count == 0]
    heapq.heapify(ready)
    results: dict[int, StepExecutionResult] = {}

    def complete(number: int, result: StepExecutionResult) -> None:
        results[number] = result
        for child in dependents[number]:
            pending[child] -= 1
            if pending[child] == 0:
                heapq.heappush(ready, (order[child], child))

    loop = asyncio.get_running_loop()
    deadline = None if timeout is None else loop.time() + timeout
    running: dict[asyncio.Task[StepExecutionResult], int] = {}
    completed: asyncio.Queue[asyncio.Task[StepExecutionResult]] = asyncio.Queue()
    async with asyncio.timeout_at(deadline), asyncio.TaskGroup() as group:
        while True:
            if deadline is not None and loop.time() >= deadline:
                # The timer may be due before the loop can deliver its cancellation.
                await loop.create_future()
            if ready and len(running) < max_concurrency:
                _, number = heapq.heappop(ready)
                step = nodes[number]
                failed = [parent for parent in step.dependencies if not results[parent].success]
                if failed:
                    complete(number, _failed(step, f"Unsuccessful dependencies: {failed}", skipped=True))
                    continue

                visible = {
                    str(parent): {"result": results[parent].result, "result_data": results[parent].result_data}
                    for parent in references[number]
                }
                task = group.create_task(_execute_step(step, tools[step.config.tool_name], inputs, visible, deadline))
                running[task] = number
                task.add_done_callback(completed.put_nowait)
                continue
            if not running:
                break

            task = await completed.get()
            while True:
                number = running.pop(task)
                result = _failed(nodes[number], "Tool cancelled") if task.cancelled() else task.result()
                complete(number, result)
                if completed.empty():
                    break
                task = completed.get_nowait()
    return [results[number] for number in nodes]
