"""
DAG execution engine for CARL reasoning chains.

Handles parallel execution of reasoning steps based on their dependencies.
Supports multiple step types through pluggable executors.
"""

import asyncio
import copy
import json
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field

from ._dependency_graph import dependency_back_edges
from ._ready_frontier import ReadyFrontier
from .logging_utils import (
    log_batch_start,
    log_chain_complete,
    log_chain_start,
    log_step_complete,
    log_step_start,
    log_warning,
)
from .models import (
    AnyStepDescription,
    PromptTemplate,
    ReasoningContext,
    ReasoningResult,
    ReplanAction,
    ReplanAggregationOutcome,
    ReplanCheckerInput,
    ReplanCheckerVote,
    ReplanEvent,
    ReplanPolicy,
    ReplanRollbackTarget,
    ReplanTargetType,
    ReplanVerdict,
    StepDescription,
    StepDescriptionBase,
    StepExecutionResult,
    StepType,
)
from .cow_memory import COWMemoryStore
from .models.llm_client_base import ChatMessage
from .memory_schema import MemorySchemaError, validate_memory_write
from .execution_trace import TRACE_TEXT_LIMIT, ExecutionTrace, TraceEvent
from .replan import CheckerVote, aggregate_replan_votes, create_checker_from_spec
from .step_executors import get_executor, resolve_context_reference
from .tracing import create_chain_trace


_StepCache = dict[str, tuple[str, Any, dict, str | None, float]]


@dataclass
class _RunState:
    executor: "DAGExecutor"
    deadline: float | None
    initial_paused_duration: float
    result_cache: _StepCache = field(default_factory=dict)
    deadline_expired: bool = False

    def remaining(self, context: ReasoningContext) -> float | None:
        if self.deadline is None:
            return None
        if self.deadline_expired:
            return 0.0
        paused = context.paused_duration() - self.initial_paused_duration
        remaining = self.deadline + paused - time.monotonic()
        self.deadline_expired = remaining <= 0
        return remaining


_active_run: ContextVar[_RunState | None] = ContextVar("carl_dag_run", default=None)


class _MemorySizeProbe:
    """Reuse sizes for scalar-only memory; mutable/custom values are remeasured."""

    def __init__(self) -> None:
        self._snapshot: dict[str, dict[str, Any]] | None = None
        self._size = 0

    def measure(self, memory: dict[str, Any]) -> int:
        snapshot = self._snapshot
        scalar_only = type(memory) is dict and all(
            type(namespace) is str and type(values) is dict
            and all(type(key) is str and type(value) in (str, int, float, bool, type(None))
                    for key, value in values.items())
            for namespace, values in memory.items()
        )
        if scalar_only and snapshot is not None and memory.keys() == snapshot.keys() and all(
            values.keys() == snapshot[namespace].keys()
            and all(value is snapshot[namespace][key] for key, value in values.items())
            for namespace, values in memory.items()
        ):
            return self._size
        self._size = _estimate_memory_bytes(memory)
        self._snapshot = {namespace: values.copy() for namespace, values in memory.items()} if scalar_only else None
        return self._size


def _estimate_memory_bytes(memory: dict[str, Any]) -> int:
    """
    Estimate the serialized size of working memory in bytes.

    Uses JSON serialization with a fallback to ``str()`` for non-serializable
    objects so that the estimate is always available without raising.
    """
    try:
        return len(json.dumps(memory, default=str))
    except Exception:
        return 0


#: Per-value ceiling for captured ``TraceEvent.inputs`` entries. A trace is a
#: per-step attribution record, not a copy of the payload the step consumed —
#: an unbounded capture would make traces of data-heavy chains unusable.
_TRACE_INPUT_VALUE_LIMIT = 2000


def _utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string.

    Matches the ``ExecutionTrace.executed_at`` convention so every timestamp
    in a trace parses with a single :func:`datetime.fromisoformat` call.
    """
    return datetime.now(timezone.utc).isoformat()


def _trace_safe_value(value: Any) -> Any:
    """Return a JSON-serialisable, size-bounded view of *value*.

    Trace inputs are arbitrary resolved Python objects (tool arguments, memory
    values, sub-chain payloads). Anything that does not survive ``json.dumps``
    degrades to its string form rather than breaking trace serialisation.
    """
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value[:_TRACE_INPUT_VALUE_LIMIT]
    try:
        rendered = json.dumps(value, default=str)
    except Exception:
        rendered = str(value)
    if len(rendered) > _TRACE_INPUT_VALUE_LIMIT:
        return rendered[:_TRACE_INPUT_VALUE_LIMIT]
    try:
        return json.loads(rendered)
    except Exception:
        return rendered


def _capture_step_inputs(
    step: Any,
    context: ReasoningContext,
    redacted_refs: set[str],
) -> dict[str, Any]:
    """Resolve a step's declared ``input_mapping`` for the trace.

    Called immediately before the step runs so the recorded values are the
    ones the executor actually consumed. Best-effort by contract: a step type
    without an ``input_mapping`` records ``{}``, and a reference that fails to
    resolve records ``None`` instead of failing the run.

    References listed in *redacted_refs* (memory keys written by a
    ``sensitive=True`` HumanInputStep) are stored as ``"[redacted]"`` so the
    redaction that already protects ``result`` is not defeated by a downstream
    step reading the same key.
    """
    config = getattr(step, "step_config", None)
    mapping = getattr(config, "input_mapping", None)
    if not isinstance(mapping, dict) or not mapping:
        return {}

    captured: dict[str, Any] = {}
    for name, source in mapping.items():
        key = str(name)
        if not isinstance(source, str):
            captured[key] = _trace_safe_value(source)
            continue
        if source.strip() in redacted_refs:
            captured[key] = "[redacted]"
            continue
        try:
            captured[key] = _trace_safe_value(resolve_context_reference(source, context))
        except Exception:
            captured[key] = None
    return captured


def _result_is_redacted(result: StepExecutionResult) -> bool:
    """Whether the step deliberately withheld its own output.

    Currently set by ``HumanInputStepExecutor`` for ``sensitive=True`` steps,
    whose outcome payload carries ``redacted=True``.
    """
    data = result.result_data
    return isinstance(data, dict) and bool(data.get("redacted"))


def _redacted_memory_ref(step: Any, result: StepExecutionResult) -> Optional[str]:
    """Return the ``$memory.*`` reference a redacted step wrote its value to.

    A ``sensitive=True`` HumanInputStep must declare ``output_memory_key``, so
    the secret does land in memory even though it is stripped from the step's
    own result. Downstream steps that read that key must not re-leak it into
    ``TraceEvent.inputs``.
    """
    if step is None or not _result_is_redacted(result):
        return None
    key = getattr(getattr(step, "step_config", None), "output_memory_key", None)
    if not key:
        return None
    return f"$memory.human_input.{key}"


def _extract_sub_trace(result: StepExecutionResult) -> Optional[ExecutionTrace]:
    """Pull the nested :class:`ExecutionTrace` out of a delegating step.

    Agent-handoff and supervisor steps put the sub-chain's live
    ``ReasoningResult`` into ``result_data["sub_result"]``. Lifting its trace
    onto the parent event keeps the nested run structured; without it
    ``to_json``'s ``default=str`` would flatten the whole sub-execution into a
    Python ``repr``.
    """
    data = result.result_data
    if not isinstance(data, dict):
        return None
    sub_trace = getattr(data.get("sub_result"), "trace", None)
    return sub_trace if isinstance(sub_trace, ExecutionTrace) else None


class ExecutionCancelledError(Exception):
    """Raised when chain execution is cancelled by user.

    The partially-built :class:`~mmar_carl.models.results.ReasoningResult`
    is attached as :attr:`result` so callers can recover completed-step
    outputs after catching::

        try:
            result = await chain.execute_async(ctx)
        except ExecutionCancelledError as exc:
            partial = exc.result
            print(partial.get_partial_final_output() or "no work done")
    """

    def __init__(self, message: str, *, result: Optional["ReasoningResult"] = None) -> None:
        super().__init__(message)
        self.result = result


class ExecutionNode(BaseModel):
    """
    Represents a node in the execution DAG.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    step: Any
    dependencies: list["ExecutionNode"] = Field(default_factory=list)
    dependents: list["ExecutionNode"] = Field(default_factory=list)
    executed: bool = False
    executing: bool = False
    result: StepExecutionResult | None = None

    def can_execute(self) -> bool:
        """Check if this node can be executed (all dependencies completed)."""
        return all(dep.executed for dep in self.dependencies)

    def is_ready(self) -> bool:
        """Check if this node is ready for execution (not executing or executed)."""
        return not self.executing and not self.executed and self.can_execute()


ExecutionNode.model_rebuild()


class _ExecutionSnapshot(BaseModel):
    """Snapshot of mutable chain execution state for rollback support."""

    executed_nodes: set[int]
    completed_step_numbers: set[int]
    result_cache: _StepCache
    all_results: list[StepExecutionResult]
    history: list[str]
    messages: list[ChatMessage]
    metadata: dict[str, Any]
    memory: dict[str, dict[str, Any]]


class _CheckpointSnapshot(BaseModel):
    """Named checkpoint snapshot with ordering metadata."""

    name: str
    step_number: int
    sequence: int
    snapshot: _ExecutionSnapshot


def validate_planner_schedule(steps: Sequence[Any], replan_policy: ReplanPolicy | None, max_injections: int) -> bool:
    """Validate dynamic DAG constraints at construction and execution boundaries."""
    planners = [step for step in steps if step.step_type == StepType.PLANNER]
    if not planners:
        return False
    if replan_policy and replan_policy.enabled:
        raise ValueError("PlannerStep cannot be combined with RE-PLAN")
    if any(getattr(step, "loop_back_to", None) is not None for step in steps):
        raise ValueError("PlannerStep cannot be combined with loop-back steps")
    budget = sum(min(step.step_config.max_actions, len(step.step_config.actions)) for step in planners)
    if budget > max_injections:
        raise ValueError("PlannerStep action budgets exceed the chain's max_injections")
    return True


class DAGExecutor:
    """
    Executes reasoning chains as a Directed Acyclic Graph (DAG).

    Automatically parallelizes execution where dependencies allow.

    Note on parallel execution:
        - Each parallel step gets an isolated copy of memory (deep copy)
        - Tool registry is shared via shallow copy - tools MUST be stateless
        - Memory writes in parallel steps ARE merged back after batch completion
          but are NOT visible to other parallel steps in the same batch

    Note on conditional steps:
        - CONDITIONAL steps return next_step in result_data but DAG execution
          currently proceeds in topological order
        - For branching behavior, use separate chains or design dependencies accordingly
        - This is a known limitation that may be addressed in future versions

    Note on cancellation:
        - Call context.cancel() to request cancellation
        - Execution will stop after the current batch completes
        - Partial results are returned with success=False

    Performance Notes:
        For chains with large memory payloads, deep copying for parallel
        execution can be expensive. Consider:

        1. Using smaller, focused memory namespaces
        2. Reducing max_workers if memory is huge (default is 1)
        3. Using external state storage (Redis, etc.) for very large state
        4. Setting max_workers=1 for memory-heavy chains to avoid deep copies

    Token Usage Tracking:
        Token usage is tracked per step and aggregated in ReasoningResult.
        Note: Token counts are only available when the LLM client provides them
        (e.g., OpenAI API responses include usage data).
    """

    def __init__(
        self,
        max_workers: int | str = 1,
        prompt_template: PromptTemplate | None = None,
        enable_progress: bool = False,
        timeout: float | None = None,
        replan_policy: ReplanPolicy | None = None,
        max_injections: int = 50,
        fail_fast: bool = False,
    ):
        """
        Initialize the DAG executor.

        Args:
            max_workers: Maximum number of parallel executions, or ``"auto"``
                to match the size of each batch (no over-provisioning).  With
                ``"auto"`` a batch of 1 runs sequentially; a batch of N runs
                N coroutines concurrently — optimal for most chains.
            prompt_template: Template for generating prompts
            enable_progress: Whether to enable progress tracking
            timeout: Maximum total execution time in seconds (None = no limit)
        """
        if isinstance(max_workers, str) and max_workers != "auto":
            raise ValueError(f"max_workers must be a positive integer or 'auto', got {max_workers!r}")
        if type(fail_fast) is not bool:
            raise ValueError("fail_fast must be a boolean")
        if fail_fast and max_workers != "auto" and (type(max_workers) is not int or max_workers < 1):
            raise ValueError("max_workers must be a positive integer or 'auto' with fail_fast")
        self.fail_fast = fail_fast
        self.max_workers = max_workers
        self.prompt_template = prompt_template or PromptTemplate()
        self.enable_progress = enable_progress
        self.timeout = timeout
        self.replan_policy = replan_policy
        self.max_injections = max_injections
        self._execution_stats = {
            "total_steps": 0,
            "executed_steps": 0,
            "failed_steps": 0,
            "parallel_batches": 0,
            "total_time": 0.0,
        }

    @property
    def _result_cache(self) -> _StepCache:
        run = _active_run.get()
        return run.result_cache if run is not None and run.executor is self else {}

    def build_execution_graph(self, steps: Sequence[StepDescription | StepDescriptionBase | AnyStepDescription]) -> list[ExecutionNode]:
        """
        Build an execution graph from step descriptions.

        Args:
            steps: list of step descriptions

        Returns:
            list of execution nodes forming the DAG
        """
        if not steps:
            return []

        # Create nodes
        step_map: dict[int, ExecutionNode] = {}
        for step in steps:
            node = ExecutionNode(step=step, dependencies=[], dependents=[])
            step_map[step.number] = node

        # Build dependencies
        for step in steps:
            node = step_map[step.number]
            for dep_number in step.dependencies:
                if dep_number in step_map:
                    dependency_node = step_map[dep_number]
                    node.dependencies.append(dependency_node)
                    dependency_node.dependents.append(node)
                else:
                    raise ValueError(
                        f"Step {step.number} ('{step.title}') depends on non-existent step {dep_number}. "
                        f"Available steps: {sorted(step_map.keys())}. "
                        "Check your dependencies configuration."
                    )

        # Validate no cycles
        self._validate_no_cycles(step_map)

        return list(step_map.values())

    def _validate_no_cycles(self, nodes: dict[int, ExecutionNode]) -> None:
        """
        Validate that the execution graph has no cycles.

        Args:
            nodes: Dictionary of execution nodes

        Raises:
            ValueError: If cycles are detected
        """
        dependencies = {
            number: [dependency.step.number for dependency in node.dependencies]
            for number, node in nodes.items()
        }
        cycle = next(dependency_back_edges(dependencies), None)
        if cycle is not None:
            root, _ = cycle
            raise ValueError(f"Cycle detected in execution graph involving step {root}")

    def _process_injections(
        self,
        *,
        batch_results: list[StepExecutionResult],
        nodes: list[ExecutionNode],
        step_by_number: dict[int, Any],
        executed_nodes: set[int],
        injection_budget_used: int,
    ) -> int:
        """
        Splice ``StepExecutionResult.injected_steps`` into the running DAG.

        For each ``injected_steps`` entry returned by the batch:

        1. Validate it has a ``number`` and ``dependencies`` attribute (i.e.
           it looks like a step description).
        2. Resolve dependencies: an entry may name either an already-
           registered step number or the *original* number of a sibling
           injected earlier in the same result (tracked in a per-result
           rename table, so injectors can express intra-result DAGs in
           their own numbering). Anything else is a forward reference and
           raises ``ValueError``.
        3. Renumber a copy to ``max_existing_step + 1`` so it never collides
           with declared step numbers (the caller's original ``number`` is
           preserved in ``metadata["__original_injected_number"]`` when the
           step is a typed step; we don't mutate the original).
        4. Build an :class:`ExecutionNode` and link its dependencies, then
           append to ``nodes`` + ``step_by_number``.
        5. Enforce ``self.max_injections`` budget — raises ``ValueError`` on
           overflow with a descriptive message naming the offending step.

        Returns the updated ``injection_budget_used`` counter.
        """
        injections_this_batch = 0
        for result in batch_results:
            if not result.injected_steps:
                continue
            if injection_budget_used + injections_this_batch + len(result.injected_steps) > self.max_injections:
                raise ValueError(
                    f"Step {result.step_number} '{result.step_title}' tried to exceed "
                    f"the chain's max_injections budget ({self.max_injections})."
                )
            planned_tail = None
            planner_dependents = []
            if result.step_type == StepType.PLANNER:
                planner = next(node for node in nodes if node.step.number == result.step_number)
                planner_dependents = list(planner.dependents)
            # Per-result rename table: injector's own numbering → assigned
            # number. Lets later siblings depend on earlier ones.
            renamed: dict[int, int] = {}
            for raw_step in result.injected_steps:
                if not hasattr(raw_step, "number") or not hasattr(raw_step, "dependencies"):
                    raise ValueError(
                        f"Step {result.step_number} '{result.step_title}' returned an "
                        f"invalid injected_steps entry — expected a step-description "
                        f"object with .number and .dependencies, got {type(raw_step).__name__}"
                    )

                # Resolve each dependency against the rename table first
                # (a sibling injected earlier in this result), then against
                # the existing graph. Anything else is a forward reference.
                resolved_deps: list[int] = []
                for dep in (raw_step.dependencies or []):
                    if dep in renamed:
                        resolved_deps.append(renamed[dep])
                    elif dep in step_by_number:
                        resolved_deps.append(dep)
                    else:
                        raise ValueError(
                            f"Step {result.step_number} '{result.step_title}' injected "
                            f"a step depending on unknown step number {dep}. "
                            f"Injected steps may only reference already-registered "
                            f"step numbers."
                        )

                # Renumber a copy to avoid collisions with declared step
                # numbers, leaving the caller's object untouched.
                original_number = raw_step.number
                new_number = max(step_by_number) + 1
                if isinstance(raw_step, BaseModel):
                    update: dict[str, Any] = {"number": new_number, "dependencies": resolved_deps}
                    if "metadata" in type(raw_step).model_fields:
                        update["metadata"] = {
                            **raw_step.metadata,
                            "__original_injected_number": original_number,
                        }
                    step = raw_step.model_copy(update=update)
                else:
                    step = raw_step
                    step.number = new_number
                    step.dependencies = resolved_deps
                    if isinstance(getattr(step, "metadata", None), dict):
                        step.metadata["__original_injected_number"] = original_number
                renamed[original_number] = new_number

                # Wire dependencies (numeric → ExecutionNode link)
                deps = []
                for dep in resolved_deps:
                    dep_node = next((n for n in nodes if n.step.number == dep), None)
                    if dep_node is not None:
                        deps.append(dep_node)

                new_node = ExecutionNode(step=step, dependencies=deps)
                # Link the reverse edges so dependents know about us (harmless
                # but keeps the graph consistent for future processing).
                for dep_node in deps:
                    dep_node.dependents.append(new_node)
                nodes.append(new_node)
                step_by_number[new_number] = step
                injections_this_batch += 1
                planned_tail = new_node
            if result.step_type == StepType.PLANNER and planned_tail is not None:
                for dependent in planner_dependents:
                    dependent.dependencies.append(planned_tail)
                    planned_tail.dependents.append(dependent)
                result.result_data["step_numbers"] = list(renamed.values())
        return injection_budget_used + injections_this_batch

    async def execute_step(self, node: ExecutionNode, context: ReasoningContext) -> StepExecutionResult:
        """
        Execute a single step using the appropriate executor based on step type.

        Args:
            node: Execution node to execute
            context: Reasoning context

        Returns:
            Step execution result
        """
        step = node.step
        start_time = time.time()
        # Measured wall-clock start. Recorded here — not reconstructed from
        # batch order later — so a trace shows when each step really ran.
        started_at = _utc_now_iso()

        # Log step start
        log_step_start(step.number, step.title, str(step.step_type))

        step_timeout = getattr(step, "timeout", None) or self.timeout
        memory_schema = context.memory_schema

        # Call step_start callback if registered
        if context.on_step_start:
            try:
                context.on_step_start(step.number, step.title)
            except Exception:
                # Don't fail execution if callback fails
                pass

        # ── Step result cache check ────────────────────────────────────────────
        # Compute the cache key (if caching is configured for this step).
        # On a cache hit, return the stored result immediately — no LLM/tool call.
        step_cache_cfg = getattr(step, "cache", None)
        cache_key: str | None = None
        if step_cache_cfg is not None:
            if step_cache_cfg.key_fn is not None:
                try:
                    cache_key = step_cache_cfg.key_fn(context)
                except Exception:
                    cache_key = None  # key_fn error → treat as miss
            if cache_key is None:
                # Default key: step number + first 256 chars of outer_context
                cache_key = f"{step.number}:{context.outer_context[:256]}"

            cached = self._result_cache.get(cache_key)
            if cached is not None:
                cached_result_str, cached_result_data, cached_token_usage, cached_history_entry, stored_at = cached
                ttl = step_cache_cfg.ttl
                if ttl is None or (time.time() - stored_at) < ttl:
                    # ── Cache HIT ──────────────────────────────────────────────
                    new_history = context.history.copy()
                    if cached_history_entry is not None:
                        new_history.append(cached_history_entry)
                    cached_step_result = StepExecutionResult(
                        step_number=step.number,
                        step_title=step.title,
                        step_type=step.step_type,
                        result=cached_result_str,
                        result_data=cached_result_data,
                        token_usage=cached_token_usage,
                        success=True,
                        updated_history=new_history,
                        profiling={"cache_hit": True},
                        started_at=started_at,
                        ended_at=_utc_now_iso(),
                    )
                    if context.on_step_complete:
                        try:
                            context.on_step_complete(cached_step_result)
                        except Exception:
                            pass
                    log_step_complete(step.number, True, time.time() - start_time)
                    return cached_step_result
        # ── End cache check ────────────────────────────────────────────────────

        # Create a LangFuse span for this step if tracing is active
        trace = context.metadata.get("__langfuse_trace")
        span = None
        if trace is not None:
            span_input: dict[str, Any] = {
                "step_type": str(step.step_type),
                "number": step.number,
                "title": step.title,
                "dependencies": getattr(step, "dependencies", []),
            }

            # Add step-type-specific details to span input
            step_config = getattr(step, "step_config", None)
            if step_config is not None:
                from .models import StepType as ST

                if step.step_type == ST.TOOL:
                    span_input["tool_name"] = getattr(step_config, "tool_name", None)
                    span_input["input_mapping"] = getattr(step_config, "input_mapping", {})
                elif step.step_type == ST.MEMORY:
                    span_input["operation"] = str(getattr(step_config, "operation", ""))
                    span_input["memory_key"] = getattr(step_config, "memory_key", "")
                    span_input["namespace"] = getattr(step_config, "namespace", "default")
                elif step.step_type == ST.TRANSFORM:
                    span_input["transform_type"] = getattr(step_config, "transform_type", "")
                    span_input["input_key"] = getattr(step_config, "input_key", "")
                elif step.step_type == ST.MCP:
                    span_input["tool_name"] = getattr(step_config, "tool_name", None)
                    server = getattr(step_config, "server", None)
                    if server:
                        span_input["server_name"] = getattr(server, "server_name", "")
                elif step.step_type == ST.CONDITIONAL:
                    span_input["condition_key"] = getattr(step_config, "condition_context_key", "")

            span = trace.start_span(
                name=f"step_{step.number}: {step.title}",
                input=span_input,
            )
            context.metadata["__langfuse_span"] = span

        # Get the appropriate executor for this step type
        executor = get_executor(step.step_type)

        # Inject step-specific RE-PLAN feedback into executor-visible metadata.
        replan_feedback_map = context.metadata.get("__replan_feedback_by_step")
        step_feedback: list[str] | None = None
        if isinstance(replan_feedback_map, dict):
            raw_feedback = replan_feedback_map.get(str(step.number))
            if isinstance(raw_feedback, str):
                step_feedback = [raw_feedback]
            elif isinstance(raw_feedback, list):
                step_feedback = [str(item) for item in raw_feedback if str(item).strip()]
        if step_feedback:
            context.metadata["__replan_feedback"] = step_feedback

        run = _active_run.get()
        if run is not None and run.executor is not self:
            run = None

        # Execute the step with optional per-step timeout
        try:
            if run is not None:
                remaining = run.remaining(context)
                if remaining is not None:
                    step_timeout = min(step_timeout, remaining) if step_timeout is not None else remaining
            if step_timeout is not None:
                result = await asyncio.wait_for(
                    executor.execute(step, context, self.prompt_template), timeout=step_timeout
                )
            else:
                result = await executor.execute(step, context, self.prompt_template)
        except asyncio.TimeoutError:
            result = StepExecutionResult(
                step_number=step.number,
                step_title=step.title,
                step_type=step.step_type,
                result="",
                success=False,
                error_message=f"Step timed out after {step_timeout}s",
            )
        finally:
            context.metadata.pop("__replan_feedback", None)

        # Stamp measured start/end on the result. Executors build their own
        # ``StepExecutionResult`` (and the timeout path builds one here), so
        # the executor is the single place that can see both ends of the call.
        result = result.model_copy(
            update={"started_at": started_at, "ended_at": _utc_now_iso()}
        )

        # Validate before reporting completion or letting the next queued step start.
        if self.fail_fast and result.success and memory_schema:
            self._check_snapshot_memory(context.memory, memory_schema, result)

        # Run step-level metrics on successful results
        step_metrics = getattr(step, "metrics", [])
        if step_metrics and result.success and result.result:
            metric_scores: dict[str, float] = {}
            for metric in step_metrics:
                remaining = run.remaining(context) if run is not None else None
                if remaining is not None and remaining <= 0:
                    break
                try:
                    async with asyncio.timeout(remaining):
                        score = await metric.compute_async(result)
                    metric_scores[metric.name] = float(score)
                except Exception as e:
                    from .logging_utils import log_warning
                    log_warning(f"Metric '{getattr(metric, 'name', repr(metric))}' failed on step {step.number}: {e}")
            if metric_scores:
                result = result.model_copy(update={"metrics": metric_scores})

        # ── Store to cache (miss path) ─────────────────────────────────────────
        if step_cache_cfg is not None and cache_key is not None and result.success:
            last_history_entry = result.updated_history[-1] if result.updated_history else None
            self._result_cache[cache_key] = (
                result.result,
                result.result_data,
                result.token_usage,
                last_history_entry,
                time.time(),
            )
        # ── End cache store ───────────────────────────────────────────────────

        # End the span with step outcome
        if span is not None:
            end_output: dict[str, Any] = {"success": result.success}
            if result.result:
                end_output["result"] = result.result[:1000]
            if result.result_data is not None:
                try:
                    import json as _json

                    serialized = _json.dumps(result.result_data, default=str)
                    end_output["result_data"] = _json.loads(serialized)
                except (TypeError, ValueError):
                    end_output["result_data"] = str(result.result_data)[:1000]
            if result.error_message:
                end_output["error"] = result.error_message
            if result.execution_time:
                end_output["execution_time"] = result.execution_time
            if result.metrics:
                end_output["metrics"] = result.metrics
            span.update(output=end_output)
            span.end()
            context.metadata.pop("__langfuse_span", None)

        if context.on_step_complete:
            try:
                context.on_step_complete(result)
            except Exception:
                pass

        # Log step completion
        log_step_complete(
            step.number,
            result.success,
            time.time() - start_time,
            step_title=step.title,
            step_type=step.step_type,
            error_message=result.error_message,
        )

        return result

    @staticmethod
    def _skip_unstarted(
        node: ExecutionNode,
        context: ReasoningContext,
        *,
        failed_step_number: int | None = None,
    ) -> StepExecutionResult:
        """Report an unstarted node without emitting a misleading start event."""
        reason = (
            "Skipped because cancellation was requested before the step started."
            if failed_step_number is None else
            "Skipped because fail_fast stopped the chain after "
            f"step {failed_step_number} failed."
        )
        profiling = (
            {"cancelled": True} if failed_step_number is None else
            {"fail_fast": True, "failed_step_number": failed_step_number}
        )
        result = StepExecutionResult(
            step_number=node.step.number,
            step_title=node.step.title,
            step_type=node.step.step_type,
            result="",
            success=False,
            skipped=True,
            error_message=reason,
            profiling=profiling,
        )
        if context.on_step_complete:
            try:
                context.on_step_complete(result)
            except Exception:
                pass
        return result

    async def _execute_controlled_batch(
        self,
        ready_nodes: list[ExecutionNode],
        contexts: list[ReasoningContext],
        max_workers: int,
    ) -> list[StepExecutionResult | Exception]:
        """Gate queued work on pause/cancel, optionally stop it after failure.

        Failure state is per batch/run, never shared on the executor instance.
        The semaphore guard is checked before creating the step coroutine so
        waiting nodes cannot start after a completed failure releases a slot.
        """
        semaphore = asyncio.Semaphore(max_workers)
        failed_step_number: int | None = None
        stop_event = asyncio.Event()
        aborted = False
        run_state = _active_run.get()
        if run_state is not None and run_state.executor is not self:
            run_state = None

        def deadline_expired(context: ReasoningContext) -> bool:
            if run_state is None:
                return False
            run_state.remaining(context)
            if run_state.deadline_expired:
                stop_event.set()
            return run_state.deadline_expired

        async def wait_if_paused(context: ReasoningContext) -> None:
            # A failure in an in-flight sibling must also release paused queue
            # waiters: the failed run must not require a user Resume to finish.
            while (
                context.is_pause_requested()
                and not context.is_cancelled()
                and not stop_event.is_set()
                and not deadline_expired(context)
            ):
                waiters = [
                    asyncio.create_task(context.wait_for_resume()),
                    asyncio.create_task(stop_event.wait()),
                ]
                try:
                    await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)
                finally:
                    for waiter in waiters:
                        if not waiter.done():
                            waiter.cancel()
                    await asyncio.gather(*waiters, return_exceptions=True)

        async def run(node: ExecutionNode, context: ReasoningContext) -> StepExecutionResult | Exception:
            nonlocal failed_step_number, aborted
            async with semaphore:
                if aborted:
                    raise asyncio.CancelledError()
                try:
                    await wait_if_paused(context)
                except asyncio.CancelledError:
                    aborted = True
                    raise
                if context.is_cancelled():
                    return self._skip_unstarted(node, context)
                if failed_step_number is not None:
                    return self._skip_unstarted(
                        node, context, failed_step_number=failed_step_number
                    )
                if deadline_expired(context):
                    return TimeoutError("Chain execution timed out before step started.")
                context._mark_step_active()
                try:
                    result = await self.execute_step(node, context)
                except asyncio.CancelledError:
                    # Close the queue before releasing the slot; gather's
                    # cancellation cleanup may not run until the next turn.
                    aborted = True
                    raise
                except Exception as exc:
                    if not self.fail_fast:
                        # Match the legacy serial/parallel exception contract.
                        if max_workers == 1:
                            aborted = True
                            raise
                        return exc
                    # Keep unexpected executor exceptions a failed step, not a
                    # transport error. Cancellation (BaseException) propagates.
                    import traceback

                    result = StepExecutionResult(
                        step_number=node.step.number,
                        step_title=node.step.title,
                        step_type=node.step.step_type,
                        result="",
                        success=False,
                        error_message=str(exc),
                        error_traceback="".join(traceback.format_exception(exc)),
                    )
                    if context.on_step_complete:
                        try:
                            context.on_step_complete(result)
                        except Exception:
                            pass
                finally:
                    deadline_expired(context)
                    context._mark_step_inactive()
                if self.fail_fast and not result.success and not result.skipped and failed_step_number is None:
                    failed_step_number = result.step_number
                    stop_event.set()
                    result.profiling = {**result.profiling, "fail_fast_trigger": True}
                return result

        tasks = [
            asyncio.create_task(run(node, ctx))
            for node, ctx in zip(ready_nodes, contexts)
        ]
        try:
            return list(await asyncio.gather(*tasks))
        except BaseException:
            # gather does not cancel siblings when an individual task raises
            # CancelledError. Drain them before snapshot contexts are closed;
            # no coroutine may keep using those contexts after this exits.
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

    async def execute_batch(
        self, ready_nodes: list[ExecutionNode], context: ReasoningContext
    ) -> list[StepExecutionResult]:
        """
        Execute a batch of ready nodes in parallel.

        Args:
            ready_nodes: list of nodes ready for execution
            context: Reasoning context

        Returns:
            list of step execution results
        """
        if not ready_nodes:
            return []

        # Create independent contexts for parallel execution.
        # Memory isolation uses COWMemoryStore: each snapshot shares read access to the
        # parent memory (zero copy for unread namespaces) and only shallow-copies a
        # namespace dict on first access.  This replaces the expensive copy.deepcopy and
        # also fixes a correctness issue where the full-snapshot merge could overwrite a
        # sibling step's writes with stale data.
        context_snapshots = []
        try:
            for _ in ready_nodes:
                snapshot = ReasoningContext(
                    outer_context=context.outer_context,
                    runtime_input=context.runtime_input,
                    allow_local_agent_execution=context.allow_local_agent_execution,
                    decision_endpoints=context.decision_endpoints,
                    api=context.api,
                    model=context.model,
                    retry_max=context.retry_max,
                    history=context.history.copy(),
                    metadata=context.metadata.copy(),
                    language=context.language,
                    system_prompt=context.system_prompt,
                    memory={},  # placeholder — replaced below after model init
                    max_history_entries=context.max_history_entries,
                    trim_strategy=context.trim_strategy,
                    # Opt-in prompt capture is a chain-run setting, not per-step
                    # state — carry it onto every parallel snapshot.
                    trace_prompts=context.trace_prompts,
                    messages=list(context.messages),  # copy structured message history
                    # Long-term memory: shared reference (LTM is thread-safe by design)
                    session_id=context.session_id,
                    long_term_memory=context.long_term_memory,
                    # Preserve callbacks
                    on_step_start=context.on_step_start,
                    on_step_complete=context.on_step_complete,
                    on_chain_complete=context.on_chain_complete,
                    on_progress=context.on_progress,
                    on_llm_chunk=context.on_llm_chunk,
                    on_human_input_requested=context.on_human_input_requested,
                    command_policy=context.command_policy,
                    code_execution_policy=context.code_execution_policy,
                    command_capability_registry=context.command_capability_registry,
                    network_enforcer=context.network_enforcer,
                    on_command_approval_requested=context.on_command_approval_requested,
                    # parallel snapshots fire intra-step events
                    # to the same consumer as the parent context.
                    on_step_event=context.on_step_event,
                    memory_schema=context.memory_schema,
                )
                # Replace placeholder memory with a COW store backed by parent memory.
                # Pydantic doesn't revalidate on direct attribute set (no validate_assignment),
                # so the COWMemoryStore (a dict subclass) is accepted as-is.
                snapshot.memory = COWMemoryStore(base=context.memory)
                # Copy tool registry (shallow copy is fine for callables)
                snapshot._tool_registry = context._tool_registry.copy()
                # Copy tool tag metadata (parallel to _tool_registry) — deep copy so
                # per-snapshot mutations don't leak into the main context.
                snapshot._tool_tags = {name: set(tags) for name, tags in context._tool_tags.items()}
                # Share the process-local event bus by reference. WaitStep must
                # observe an event emitted by a sibling while the same parallel
                # batch is still running; copying and merging only after gather
                # would deadlock that batch.
                snapshot._event_bus_token = context._event_bus_token
                # Copy self-critic evaluator registry
                snapshot._self_critic_evaluator_registry = context._self_critic_evaluator_registry.copy()
                # Copy RE-PLAN checker registry
                snapshot._replan_checker_registry = context._replan_checker_registry.copy()
                # Share the cancellation token by reference so a cancel
                # requested on the parent reaches every in-flight snapshot
                # on its next poll. Previously we copied the bool, which
                # meant a user cancel only stopped the *next* batch.
                snapshot._cancel_token = context._cancel_token
                # same shared-token contract for pause/resume.
                snapshot._pause_token = context._pause_token
                context_snapshots.append(snapshot)

            workers = len(ready_nodes) if self.max_workers == "auto" else self.max_workers
            results = await self._execute_controlled_batch(ready_nodes, context_snapshots, workers)

            # Process results and handle exceptions
            processed_results = []
            import traceback as _traceback
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    error_tb = _traceback.format_exception(type(result), result, result.__traceback__)
                    processed_results.append(
                        StepExecutionResult(
                            step_number=ready_nodes[i].step.number,
                            step_title=ready_nodes[i].step.title,
                            step_type=ready_nodes[i].step.step_type,
                            result="",
                            success=False,
                            error_message=str(result),
                            error_traceback="".join(error_tb),
                        )
                    )
                else:
                    processed_results.append(result)

            # Merge memory changes back from successful parallel steps.
            # With COWMemoryStore only the keys a step actually *wrote* (or
            # mutated in place) are merged, and the keys it deleted are dropped
            # from the parent.  Namespaces a step merely read contribute
            # nothing, so a read-only step can no longer revert a sibling's
            # write.  Steps in one batch resolve write/write conflicts on the
            # same key by last-write-wins in step declaration order.
            for i, result in enumerate(processed_results):
                if result.success:
                    snap_mem = context_snapshots[i].memory
                    writes = self._snapshot_writes(snap_mem)
                    if isinstance(snap_mem, COWMemoryStore):
                        deletes = snap_mem.pending_deletes
                        dropped_namespaces = snap_mem.deleted_namespaces
                    else:
                        deletes = {}
                        dropped_namespaces = set()
                    self._commit_memory_changes(
                        context,
                        writes=writes,
                        deletes=deletes,
                        dropped_namespaces=dropped_namespaces,
                        result=result,
                    )
                    continue

                # MapStep's accepted collect-all contract commits only its one
                # configured aggregate output even when some items fail. Do not
                # merge any other failed-step snapshot mutations.
                if result.step_type == StepType.MAP:
                    config = ready_nodes[i].step.step_config
                    output_key = getattr(config, "output_memory_key", None)
                    if output_key is None:
                        continue
                    namespace = getattr(config, "output_namespace", "map")
                    snap_mem = context_snapshots[i].memory
                    if isinstance(snap_mem, COWMemoryStore):
                        namespace_data = snap_mem.pending_writes.get(namespace, {})
                    else:
                        namespace_data = snap_mem.get(namespace, {})
                    if output_key in namespace_data:
                        self._commit_memory_changes(
                            context,
                            writes={namespace: {output_key: namespace_data[output_key]}},
                            deletes={},
                            dropped_namespaces=set(),
                            result=result,
                        )

            # Merge tool registry changes from successful snapshot contexts.
            # Tool discovery steps register tools in their snapshot; those tools
            # must be visible to subsequent steps in the main context.
            for i, result in enumerate(processed_results):
                if result.success:
                    snap_registry = context_snapshots[i]._tool_registry
                    snap_tags = context_snapshots[i]._tool_tags
                    for tool_name, tool_fn in snap_registry.items():
                        if tool_name not in context._tool_registry:
                            context._tool_registry[tool_name] = tool_fn
                            # Carry over any tags the snapshot registered alongside the tool
                            context._tool_tags[tool_name] = set(snap_tags.get(tool_name, set()))

            # Merge execution mode diagnostics from snapshot contexts.
            context.metadata.setdefault("execution_mode_details", {})
            for i, result in enumerate(processed_results):
                if not result.success:
                    continue
                snapshot_mode_details = context_snapshots[i].metadata.get("execution_mode_details")
                if not isinstance(snapshot_mode_details, dict):
                    continue
                step_key = str(result.step_number)
                if step_key in snapshot_mode_details:
                    context.metadata["execution_mode_details"][step_key] = snapshot_mode_details[step_key]

            return processed_results
        finally:
            # Release only what each snapshot built for itself — the per-step
            # ``llm_config`` override clients.
            #
            # A snapshot is constructed with ``api=context.api``, i.e. it holds a
            # *reference* to the caller's client rather than one of its own. The
            # full ``close()`` would close that client, and closing an
            # OpenAICompatibleClient closes its AsyncOpenAI and with it the shared
            # httpx connection pool — aborting every request anyone else still has
            # in flight on it. With one client shared across concurrently evaluated
            # cases (the normal shape of a dataset evaluation) the first chain to
            # finish a batch killed the other chains' in-flight requests, which
            # surfaced as ``APIConnectionError("Connection error.")``: measured at
            # 23-26% of chain executions lost against vLLM, flat across client
            # concurrency 1..40. The caller's client is the caller's to close.
            for snapshot in context_snapshots:
                try:
                    await snapshot.close_derived_clients()
                except Exception:
                    pass  # Don't fail if cleanup fails

    @staticmethod
    def _snapshot_writes(memory: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        if isinstance(memory, COWMemoryStore):
            return memory.pending_writes
        # Plain-dict snapshots use the historical whole-namespace merge.
        return {namespace: dict(data) for namespace, data in memory.items()}

    @classmethod
    def _check_snapshot_memory(
        cls,
        memory: dict[str, dict[str, Any]],
        schema: dict[str, dict[str, Any]],
        result: StepExecutionResult,
    ) -> None:
        try:
            for namespace, data in cls._snapshot_writes(memory).items():
                for key, value in data.items():
                    validate_memory_write(schema, namespace, key, value)
        except MemorySchemaError as exc:
            result.success = False
            result.error_message = str(exc)

    @staticmethod
    def _commit_memory_changes(
        context: ReasoningContext,
        *,
        writes: "dict[str, dict[str, Any]]",
        deletes: "dict[str, set[str]]",
        dropped_namespaces: "set[str]",
        result: StepExecutionResult | None = None,
    ) -> None:
        """Apply one step's memory changes to the parent context.

        Deletions are applied before writes so a key a step removed cannot be
        resurrected by its own re-write ordering, and every merged value is
        routed through :func:`validate_memory_write` — the batch merge is the
        one path into parent memory that does not go through
        :meth:`ReasoningContext.memory_write`, so without this a step writing
        ``context.memory[ns][key]`` directly would bypass the declared schema
        entirely.  A violating value is not merged and its step is reported as
        failed, matching ``memory_write``'s "raise, and do not write" contract
        while keeping the failure attributable to the step that caused it.
        """
        for namespace in dropped_namespaces:
            context.memory.pop(namespace, None)

        for namespace, keys in deletes.items():
            parent_ns = context.memory.get(namespace)
            if not isinstance(parent_ns, dict):
                continue
            for key in keys:
                parent_ns.pop(key, None)

        schema = getattr(context, "memory_schema", None)
        for namespace, data in writes.items():
            if not data:
                continue
            if namespace not in context.memory:
                context.memory[namespace] = {}
            target = context.memory[namespace]
            for key, value in data.items():
                if schema is not None:
                    try:
                        validate_memory_write(schema, namespace, key, value)
                    except MemorySchemaError as exc:
                        log_warning(
                            "Discarding memory write %s.%s from step %s: %s",
                            namespace,
                            key,
                            getattr(result, "step_number", "?"),
                            exc,
                        )
                        if result is not None and result.success:
                            result.success = False
                            result.error_message = str(exc)
                        continue
                target[key] = value

    def _capture_snapshot(
        self,
        *,
        executed_nodes: set[int],
        all_results: list[StepExecutionResult],
        context: ReasoningContext,
    ) -> _ExecutionSnapshot:
        """Capture mutable state for rollback."""
        metadata = {key: value for key, value in context.metadata.items() if key != "__langfuse_trace"}
        return _ExecutionSnapshot(
            executed_nodes=executed_nodes.copy(),
            completed_step_numbers=set(context.get_executed_step_numbers()),
            result_cache=self._result_cache.copy(),
            all_results=[result.model_copy(deep=True) for result in all_results],
            history=context.history.copy(),
            messages=copy.deepcopy(context.messages),
            metadata=copy.deepcopy(metadata),
            memory=copy.deepcopy(context.memory),
        )

    def _restore_snapshot(
        self,
        snapshot: _ExecutionSnapshot,
        *,
        nodes: list[ExecutionNode],
        context: ReasoningContext,
    ) -> tuple[set[int], list[StepExecutionResult]]:
        """Restore mutable state from a snapshot."""
        executed_nodes = snapshot.executed_nodes.copy()
        all_results = [result.model_copy(deep=True) for result in snapshot.all_results]

        context.history = snapshot.history.copy()
        context.messages = copy.deepcopy(snapshot.messages)
        context.forget_completed_steps(
            set(context.get_executed_step_numbers()) - snapshot.completed_step_numbers
        )
        metadata = copy.deepcopy(snapshot.metadata)
        if "__langfuse_trace" in context.metadata:
            metadata["__langfuse_trace"] = context.metadata["__langfuse_trace"]
        context.metadata = metadata
        context.memory = copy.deepcopy(snapshot.memory)
        result_cache = self._result_cache
        result_cache.clear()
        result_cache.update(snapshot.result_cache)

        for node in nodes:
            node.executed = node.step.number in executed_nodes
            node.executing = False
            node.result = None

        return executed_nodes, all_results

    @staticmethod
    def _is_checkpoint_step(step: StepDescription | StepDescriptionBase | AnyStepDescription) -> bool:
        return bool(getattr(step, "checkpoint", False))

    @staticmethod
    def _checkpoint_name(step: StepDescription | StepDescriptionBase | AnyStepDescription) -> str:
        checkpoint_name = getattr(step, "checkpoint_name", None)
        return checkpoint_name if checkpoint_name else f"step_{step.number}"

    @staticmethod
    def _should_evaluate_replan(
        *,
        policy: ReplanPolicy,
        step: StepDescription | StepDescriptionBase | AnyStepDescription,
        result: StepExecutionResult,
    ) -> bool:
        trigger = policy.trigger

        # Step-level override can force-disable/enable checks.
        step_replan_enabled = getattr(step, "replan_enabled", None)
        if step_replan_enabled is False:
            return False

        if trigger.step_numbers and step.number not in trigger.step_numbers:
            return False
        if trigger.step_types and step.step_type not in trigger.step_types:
            return False
        if trigger.checkpoint_only and not bool(getattr(step, "checkpoint", False)):
            return False

        if result.success:
            return trigger.evaluate_after_step
        return trigger.evaluate_after_failure

    @staticmethod
    def _budget_snapshot(
        *,
        chain_replans: int,
        per_step_replans: dict[int, int],
        target_visits: dict[str, int],
        same_target_streak: int,
    ) -> dict[str, Any]:
        return {
            "chain_replans": chain_replans,
            "per_step_replans": {str(step): count for step, count in per_step_replans.items()},
            "target_visits": dict(target_visits),
            "same_target_streak": same_target_streak,
        }

    def _build_checker_input(
        self,
        *,
        result: StepExecutionResult,
        step: StepDescription | StepDescriptionBase | AnyStepDescription,
        context: ReasoningContext,
        checkpoints: list[_CheckpointSnapshot],
        budget_snapshot: dict[str, Any],
        all_results: list[StepExecutionResult],
    ) -> ReplanCheckerInput:
        recent_errors = [
            failed.error_message or "" for failed in all_results if not failed.success and failed.error_message
        ]
        checkpoint_names = [checkpoint.name for checkpoint in checkpoints]
        checkpoint_steps = [checkpoint.step_number for checkpoint in checkpoints]

        return ReplanCheckerInput(
            step_number=result.step_number,
            step_title=result.step_title,
            step_type=result.step_type,
            step_success=result.success,
            step_result=result.result,
            step_result_data=result.result_data,
            step_error=result.error_message,
            history=context.history.copy(),
            recent_errors=recent_errors,
            checkpoint_names=checkpoint_names,
            checkpoint_steps=checkpoint_steps,
            budget_snapshot=budget_snapshot,
            metadata={"step_results_total": len(all_results)},
        )

    @staticmethod
    def _normalize_feedback(verdict: ReplanVerdict) -> list[str]:
        feedback: list[str] = []
        if verdict.reason:
            feedback.append(verdict.reason)
        feedback.extend(hint for hint in verdict.regeneration_hints if hint)
        # Preserve order and remove duplicates.
        seen: set[str] = set()
        deduped: list[str] = []
        for item in feedback:
            if item not in seen:
                deduped.append(item)
                seen.add(item)
        return deduped

    def _resolve_rollback_target(
        self,
        *,
        action: ReplanAction,
        verdict: ReplanVerdict,
        policy: ReplanPolicy,
        current_step_number: int,
    ) -> ReplanRollbackTarget | None:
        if action == ReplanAction.CONTINUE or action == ReplanAction.FAIL:
            return None
        if action == ReplanAction.RETRY_CURRENT_STEP:
            return ReplanRollbackTarget(target_type=ReplanTargetType.CURRENT_STEP)
        if action == ReplanAction.RESTART_CHAIN:
            return ReplanRollbackTarget(target_type=ReplanTargetType.CHAIN_START)
        if action == ReplanAction.REPLAN_FROM_CHECKPOINT:
            if verdict.suggested_target is not None:
                return verdict.suggested_target
            return policy.default_checkpoint_target
        # Defensive fallback.
        return ReplanRollbackTarget(target_type=ReplanTargetType.CURRENT_STEP, step_number=current_step_number)

    def _resolve_target_snapshot(
        self,
        *,
        target: ReplanRollbackTarget,
        current_step_number: int,
        chain_start_snapshot: _ExecutionSnapshot | None,
        pre_step_snapshots: dict[int, _ExecutionSnapshot],
        checkpoints: list[_CheckpointSnapshot],
    ) -> tuple[_ExecutionSnapshot | None, str]:
        if target.target_type == ReplanTargetType.CHAIN_START:
            return chain_start_snapshot, "chain_start"

        if target.target_type == ReplanTargetType.CURRENT_STEP:
            snapshot = pre_step_snapshots.get(current_step_number)
            return snapshot, "current_step"

        if target.target_type == ReplanTargetType.STEP_NUMBER:
            if target.step_number is None:
                return None, "step:missing"
            snapshot = pre_step_snapshots.get(target.step_number)
            return snapshot, f"step:{target.step_number}"

        if target.target_type == ReplanTargetType.NAMED_CHECKPOINT:
            name = target.checkpoint_name or ""
            for checkpoint in reversed(checkpoints):
                if checkpoint.name == name:
                    return checkpoint.snapshot, f"named:{name}"
            return None, f"named:{name}"

        if target.target_type == ReplanTargetType.NEAREST_CHECKPOINT:
            for checkpoint in reversed(checkpoints):
                if checkpoint.step_number < current_step_number:
                    return checkpoint.snapshot, f"named:{checkpoint.name}"
            return chain_start_snapshot, "chain_start"

        return None, "unknown"

    async def execute(
        self,
        steps: Sequence[StepDescription | StepDescriptionBase | AnyStepDescription],
        context: ReasoningContext,
        *,
        resume_from_step_numbers: Optional[set[int]] = None,
    ) -> ReasoningResult:
        """
        Execute a complete reasoning chain.

        Args:
            steps: list of step descriptions
            context: Initial reasoning context
            resume_from_step_numbers: Optional set of step numbers that
                are already complete (from a prior :class:`ContextSnapshot`).
                Each named node is pre-marked ``executed=True`` so the
                DAG scheduler skips it; the chain resumes at the first
                un-done step. paired with
                :meth:`ReasoningChain.execute_async`'s ``resume_from``
                argument and :meth:`ReasoningContext.snapshot`.

        Returns:
            Complete reasoning result

        Raises:
            TimeoutError: If execution exceeds the configured timeout
            ExecutionCancelledError: If cancellation is requested via context.cancel()
        """
        deadline = None if self.timeout is None else time.monotonic() + self.timeout
        run = _RunState(self, deadline, context.paused_duration())
        token = _active_run.set(run)
        try:
            return await self._execute_run(steps, context, run, resume_from_step_numbers)
        finally:
            _active_run.reset(token)
            run.result_cache.clear()

    async def _execute_run(
        self,
        steps: Sequence[StepDescription | StepDescriptionBase | AnyStepDescription],
        context: ReasoningContext,
        run: _RunState,
        resume_from_step_numbers: set[int] | None,
    ) -> ReasoningResult:
        has_planner = validate_planner_schedule(steps, self.replan_policy, self.max_injections)
        if has_planner and resume_from_step_numbers is not None:
            raise ValueError("PlannerStep cannot resume from a saved snapshot")
        terminate_numbers = {step.number for step in steps if step.step_type == StepType.TERMINATE}
        if resume_from_step_numbers and terminate_numbers.intersection(resume_from_step_numbers):
            raise ValueError("Cannot resume a snapshot that already executed TerminateStep")
        start_time = time.time()
        context.reset_cancellation()
        injection_budget_used = 0
        memory_size = _MemorySizeProbe()

        # Create a LangFuse trace for this chain execution (no-op when disabled)
        chain_meta = context.metadata.get("chain_metadata", {})
        chain_name = chain_meta.get("trace_name") or chain_meta.get("name", "*Unnamed CARL Chain*")
        session_id = chain_meta.get("session_id")

        # Log chain start
        log_chain_start(chain_name, len(steps), self.max_workers)

        # Create structured execution trace
        exec_trace = ExecutionTrace(chain_title=chain_name)

        trace = create_chain_trace(
            chain_name=chain_name,
            context_preview=context.outer_context,
            total_steps=len(steps),
            language=str(context.language),
            session_id=session_id,
        )
        context.metadata["__langfuse_trace"] = trace

        # Build execution graph
        nodes = self.build_execution_graph(steps)
        if not nodes:
            self._execution_stats["total_steps"] = 0
            trace.update(output={"success": True, "executed_steps": 0})
            trace.end()
            result = ReasoningResult(
                success=True,
                history=[],
                step_results=[],
                total_execution_time=time.time() - start_time,
                context_metadata=dict(context.metadata),
            )
            if context.on_chain_complete:
                try:
                    context.on_chain_complete(result)
                except Exception:
                    pass
            return result

        step_by_number: dict[int, StepDescription | StepDescriptionBase | AnyStepDescription] = {step.number: step for step in steps}

        # Build RE-PLAN checker runtimes once per chain run.
        replan_policy = self.replan_policy
        replan_checkers: list[tuple[str, Any]] = []
        if replan_policy and replan_policy.enabled and replan_policy.checkers:
            for checker_spec in replan_policy.checkers:
                checker_name = getattr(checker_spec, "name", type(checker_spec).__name__)
                checker_runtime = create_checker_from_spec(checker_spec, context)
                replan_checkers.append((checker_name, checker_runtime))

        replan_enabled = bool(replan_policy and replan_policy.enabled and replan_checkers)
        replan_events: list[ReplanEvent] = []

        # Execute DAG
        executed_nodes: set[int] = set()
        # pre-mark already-completed steps as executed
        # so the scheduler skips them. The caller (chain.execute_async
        # via `resume_from=ContextSnapshot`) restores history / memory
        # / metadata first; here we only need to flip the DAG-node
        # flags so dependents see their predecessors as done.
        if resume_from_step_numbers:
            for node in nodes:
                if node.step.number in resume_from_step_numbers:
                    node.executed = True
                    executed_nodes.add(node.step.number)
        if terminate_numbers:
            # A reused context can contain outputs from an earlier run. Only
            # snapshot-completed nodes may supply earlier outputs in this run.
            step_results = dict(context.metadata.get("step_results", {}))
            for node in nodes:
                number = node.step.number
                if number not in executed_nodes:
                    step_results.pop(str(number), None)
                    context.metadata.pop(f"step_{number}", None)
            context.metadata["step_results"] = step_results
        frontier = ReadyFrontier.for_static_graph(nodes, replan_enabled=replan_enabled)
        all_results: list[StepExecutionResult] = []
        current_history = context.history.copy()
        batch_count = 0
        # ``$memory.*`` references holding values a redacted step produced.
        # Downstream steps that read them get "[redacted]" in their trace
        # inputs instead of the secret itself.
        redacted_input_refs: set[str] = set()
        cancelled = False
        replan_failed = False
        fail_fast_step_number: int | None = None
        termination: StepExecutionResult | None = None
        replan_fail_message = ""

        # RE-PLAN runtime state
        chain_start_snapshot = (
            self._capture_snapshot(executed_nodes=executed_nodes, all_results=all_results, context=context)
            if replan_enabled else None
        )
        pre_step_snapshots: dict[int, _ExecutionSnapshot] = {}
        checkpoints: list[_CheckpointSnapshot] = []
        chain_replans = 0
        per_step_replans: dict[int, int] = {}
        rollback_target_visits: dict[str, int] = {}
        last_rollback_target_key: str | None = None
        same_target_streak = 0

        # Ensure internal metadata containers exist.
        context.metadata.setdefault("__replan_feedback_by_step", {})

        # Loop iteration tracking: maps "loop_back_to-tail_step" → iteration count
        loop_iteration_counts: dict[str, int] = {}

        def check_chain_deadline() -> None:
            remaining = run.remaining(context)
            if remaining is not None and remaining <= 0:
                message = (
                    f"Chain execution timed out after {self.timeout}s. "
                    f"Completed {len(executed_nodes)}/{len(nodes)} steps."
                )
                trace.update(output={"success": False, "error": message})
                trace.end()
                raise TimeoutError(message)

        while len(executed_nodes) < len(nodes):
            # Check for cancellation
            if context.is_cancelled():
                cancelled = True
                break

            # pause check at batch boundaries. In-flight
            # steps from the previous batch have already completed by
            # the time we reach this point in the loop. Await resume
            # (the event is set whenever no pause is active, so this
            # is a no-op fast path).
            if context.is_pause_requested():
                check_chain_deadline()
                await context.wait_for_resume()
                # A cancellation may have been requested while paused;
                # re-check before scheduling the next batch.
                if context.is_cancelled():
                    cancelled = True
                    break

            check_chain_deadline()

            # Find ready nodes: numeric deps satisfied AND all triggered_by events emitted
            def _events_ready(step: Any) -> bool:
                triggered_by = getattr(step, "triggered_by", None) or []
                return all(context.has_event(name) for name in triggered_by)

            ready_nodes = frontier.ready_nodes if frontier is not None else [
                node
                for node in nodes
                if node.step.number not in executed_nodes
                and node.can_execute()
                and _events_ready(node.step)
            ]

            if not ready_nodes:
                # This should not happen in a valid DAG. List remaining steps
                # and any unfired event triggers so the user can diagnose.
                remaining = [n.step.number for n in nodes if n.step.number not in executed_nodes]
                unmet_events: dict[int, list[str]] = {}
                for n in nodes:
                    if n.step.number in executed_nodes:
                        continue
                    triggered_by = getattr(n.step, "triggered_by", None) or []
                    missing = [name for name in triggered_by if not context.has_event(name)]
                    if missing:
                        unmet_events[n.step.number] = missing
                event_info = f" Unmet events: {unmet_events}." if unmet_events else ""
                raise ValueError(
                    f"Deadlock detected: unable to execute steps {remaining}. "
                    f"Check for missing dependencies or circular references.{event_info}"
                )

            batch_count += 1
            log_batch_start(batch_count, len(ready_nodes))
            if self.enable_progress:
                print(f"Executing batch {batch_count} with {len(ready_nodes)} steps")

            # Call on_progress callback if registered
            if context.on_progress:
                try:
                    context.on_progress(len(executed_nodes), len(nodes))
                except Exception:
                    pass

            # Capture pre-execution snapshot for each step in this batch (retry target).
            if replan_enabled:
                batch_snapshot = self._capture_snapshot(
                    executed_nodes=executed_nodes,
                    all_results=all_results,
                    context=context,
                )
                for node in ready_nodes:
                    pre_step_snapshots[node.step.number] = batch_snapshot

            # Snapshot each step's resolved inputs *before* it runs — after the
            # batch, the context has already moved on and the values a step
            # actually consumed are no longer recoverable.
            batch_inputs: dict[int, dict[str, Any]] = {
                node.step.number: _capture_step_inputs(node.step, context, redacted_input_refs)
                for node in ready_nodes
            }

            # Execute batch
            check_chain_deadline()
            batch_results = await self.execute_batch(ready_nodes, context)
            all_results.extend(batch_results)
            batch_failure = next((r for r in batch_results if r.profiling.get("fail_fast_trigger")), None)
            terminal_candidate = min(
                (r for r in batch_results if r.step_type == StepType.TERMINATE and r.success and not r.skipped),
                key=lambda r: r.step_number,
                default=None,
            ) if batch_failure is None else None

            # Process dynamic step injection (steps that ask for new follow-on
            # work via ``StepExecutionResult.injected_steps``). Done before
            # trace recording so injected steps are visible to the rest of the
            # loop in the same batch iteration.
            if terminal_candidate is None:
                injection_budget_used = self._process_injections(
                    batch_results=[r for r in batch_results if r.success] if self.fail_fast else batch_results,
                    nodes=nodes,
                    step_by_number=step_by_number,
                    executed_nodes=executed_nodes,
                    injection_budget_used=injection_budget_used,
                )
                if injection_budget_used:
                    frontier = None

                for result in batch_results:
                    if result.success and result.step_type == StepType.PLANNER and result.injected_steps:
                        context.emit_step_event(
                            result.step_number,
                            "planner_actions_scheduled",
                            {
                                "actions": list(result.result_data["actions"]),
                                "step_numbers": list(result.result_data["step_numbers"]),
                            },
                        )

            # Record a TraceEvent for each step result in this batch.
            #
            # STATE vs LEDGER. ``all_results`` is *state*: it is what the DAG
            # scheduler, the success verdict, ``$steps.N`` references and the
            # final output read, so a RE-PLAN rollback (``_restore_snapshot``)
            # and a loop-back restart both rewind it — a discarded attempt must
            # not leak into the answer. ``exec_trace.events`` is the *ledger*:
            # it is append-only and is never rewound, so it keeps one entry per
            # step execution that actually happened, discarded attempts
            # included. A rollback undoes the chain's state, not its bill, so
            # anything cumulative (token spend below, execution counts) must be
            # summed from the ledger and never from the state.
            for batch_result in batch_results:
                # A step that redacted its own output must not have that value
                # reappear via its inputs or prompt — and any memory key it
                # wrote is off-limits to later steps' input capture too.
                redacted = _result_is_redacted(batch_result)
                redacted_ref = _redacted_memory_ref(
                    step_by_number.get(batch_result.step_number), batch_result
                )
                if redacted_ref:
                    redacted_input_refs.add(redacted_ref)
                exec_trace.events.append(
                    TraceEvent(
                        step_number=batch_result.step_number,
                        step_title=batch_result.step_title,
                        step_type=batch_result.step_type,
                        success=batch_result.success,
                        skipped=batch_result.skipped,
                        result=batch_result.result[:TRACE_TEXT_LIMIT] if batch_result.result else "",
                        result_data=batch_result.result_data,
                        error_message=batch_result.error_message,
                        execution_time=batch_result.execution_time,
                        token_usage=batch_result.token_usage,
                        batch_index=batch_count - 1,
                        inputs={} if redacted else batch_inputs.get(batch_result.step_number, {}),
                        model=batch_result.model,
                        prompt=(
                            batch_result.prompt[:TRACE_TEXT_LIMIT]
                            if batch_result.prompt and not redacted
                            else None
                        ),
                        started_at=batch_result.started_at,
                        ended_at=batch_result.ended_at,
                        sub_trace=_extract_sub_trace(batch_result),
                    )
                )

            # Process conditional routing decisions
            for result in batch_results:
                if terminal_candidate is None and result.success and result.step_type == StepType.CONDITIONAL:
                    frontier = None
                    next_step = result.result_data.get("next_step")
                    if next_step is not None:
                        self._skip_conditional_branches(
                            nodes, result.step_number, next_step, executed_nodes, all_results
                        )
                    else:
                        # No branch matched and no default — skip ALL downstream branches
                        self._skip_all_conditional_branches(
                            nodes, result.step_number, executed_nodes, all_results
                        )

            # Update history from successful results
            # Sort by step number to maintain deterministic order
            batch_results.sort(key=lambda r: r.step_number)
            seen_steps = set()

            for result in batch_results:
                if result.success and result.step_number not in seen_steps:
                    # Add the latest history entry from this step
                    if result.updated_history:
                        new_entry = result.updated_history[-1]
                        # FIX: Use add_to_history() to enforce max_history_entries limit
                        context.add_to_history(new_entry)
                        seen_steps.add(result.step_number)
                    # Propagate structured message history (multi-turn mode)
                    if result.updated_messages:
                        context.messages = list(result.updated_messages)

            # Sync current_history with context (may have been trimmed by max_history_entries)
            current_history = context.history.copy()

            # ----------------------------------------------------------------
            # Profiling: annotate each step result with memory / history sizes
            # measured after the batch has been merged into the live context.
            # ----------------------------------------------------------------
            memory_bytes_after = memory_size.measure(context.memory)
            history_bytes_after = sum(len(e) for e in current_history)
            for result in batch_results:
                history_bytes_added = (
                    len(result.updated_history[-1]) if result.success and result.updated_history else 0
                )
                # Merge with any pre-existing profiling data (e.g. cache_hit=True set by execute_step).
                result.profiling = {
                    **result.profiling,
                    "history_bytes_added": history_bytes_added,
                    "memory_bytes_after": memory_bytes_after,
                    "history_bytes_after": history_bytes_after,
                    "batch_index": batch_count - 1,
                }

            # Mark nodes as executed
            batch_by_number = {r.step_number: r for r in batch_results}
            for node in ready_nodes:
                node.result = batch_by_number.get(node.step.number)
                if node.result is not None and node.result.skipped and (
                    self.fail_fast or node.result.profiling.get("cancelled")
                ):
                    # An unstarted node is not resumable as completed work.
                    continue
                node.executed = True
                executed_nodes.add(node.step.number)
                # record on the context too so a later
                # ``ctx.snapshot()`` captures the completed step numbers
                # for resume.
                if not self.fail_fast or (node.result is not None and node.result.success):
                    context.record_step_executed(node.step.number)
            if frontier is not None:
                frontier.advance()

            # Update step-level metadata immediately after batch execution so that
            # loop condition checks (below) can read $metadata.step_N references.
            # This must happen BEFORE the loop check to avoid always reading None
            # on the first iteration.
            context.metadata.setdefault("step_results", {})
            for result in batch_results:
                if result.success:
                    context.metadata["step_results"][str(result.step_number)] = {
                        "step_number": result.step_number,
                        "title": result.step_title,
                        "step_type": str(result.step_type),
                        "result": result.result,
                        "result_data": result.result_data,
                        "tool_name": result.tool_name,
                    }
                    context.metadata[f"step_{result.step_number}"] = (
                        result.result_data if result.result_data is not None else result.result
                    )
                elif terminate_numbers:
                    # A failed loop iteration must not expose the output of
                    # an earlier successful iteration to a terminating step.
                    context.metadata["step_results"].pop(str(result.step_number), None)
                    context.metadata.pop(f"step_{result.step_number}", None)

            # A final step failure takes precedence over loops and RE-PLAN.
            # Preserve real results from siblings already running, then expose
            # all remaining nodes as skipped rather than leaving the UI pending.
            if context.is_cancelled() and (
                terminal_candidate is not None or self.fail_fast
                or any(r.profiling.get("cancelled") and r.skipped for r in batch_results)
            ):
                cancelled = True
                break
            if batch_failure is not None:
                fail_fast_step_number = batch_failure.step_number
                reported_steps = {r.step_number for r in all_results}
                for node in nodes:
                    if node.step.number in executed_nodes or node.step.number in reported_steps:
                        continue
                    skipped_result = self._skip_unstarted(
                        node, context, failed_step_number=fail_fast_step_number
                    )
                    node.result = skipped_result
                    all_results.append(skipped_result)
                    exec_trace.events.append(
                        TraceEvent(
                            step_number=skipped_result.step_number,
                            step_title=skipped_result.step_title,
                            step_type=skipped_result.step_type,
                            success=False,
                            skipped=True,
                            result="",
                            error_message=skipped_result.error_message,
                            execution_time=0.0,
                            batch_index=batch_count - 1,
                        )
                    )
                break

            if terminal_candidate is not None:
                termination = terminal_candidate
                for node in nodes:
                    if node.step.number in executed_nodes:
                        continue
                    skipped_result = StepExecutionResult(
                        step_number=node.step.number,
                        step_title=node.step.title,
                        step_type=node.step.step_type,
                        result="",
                        success=True,
                        skipped=True,
                        profiling={"terminated_by": termination.step_number},
                    )
                    node.result = skipped_result
                    all_results.append(skipped_result)
                    if context.on_step_complete:
                        try:
                            context.on_step_complete(skipped_result)
                        except Exception:
                            pass
                break

            # Check for loop-back directives on successfully completed steps.
            # When a step declares loop_back_to, we re-queue the loop body if the
            # condition is truthy and the iteration budget has not been exhausted.
            loop_restarted = False
            for result in batch_results:
                if not result.success:
                    continue
                step = step_by_number[result.step_number]
                loop_back_to_num = getattr(step, "loop_back_to", None)
                if loop_back_to_num is None:
                    continue

                loop_cfg = getattr(step, "loop_config", None)
                max_iters = loop_cfg.max_iterations if loop_cfg else 10
                loop_key = f"{loop_back_to_num}-{result.step_number}"
                current_iter = loop_iteration_counts.get(loop_key, 0)

                if current_iter >= max_iters:
                    # Budget exhausted — stop looping, continue chain normally.
                    continue

                # Evaluate continuation condition.
                if loop_cfg and loop_cfg.condition_key:
                    try:
                        cond_value = resolve_context_reference(loop_cfg.condition_key, context)
                    except Exception:
                        cond_value = False
                    should_continue = bool(cond_value)
                    if loop_cfg.negate_condition:
                        should_continue = not should_continue  # until-loop: loop while NOT condition
                    if not should_continue:
                        continue  # Condition says stop — exit loop.

                # --- Loop body reset ---
                loop_iteration_counts[loop_key] = current_iter + 1

                # Determine which step numbers are in the loop body.
                loop_body: set[int] = {
                    n.step.number
                    for n in nodes
                    if loop_back_to_num <= n.step.number <= result.step_number
                }

                # Reset node state for loop body steps.
                for node in nodes:
                    if node.step.number in loop_body:
                        node.executed = False
                        node.executing = False
                        node.result = None
                        executed_nodes.discard(node.step.number)

                if self.fail_fast:
                    context.forget_completed_steps(loop_body)

                # Drop old results for loop body steps; keep them for tracing in metadata.
                old_loop_results = [r for r in all_results if r.step_number in loop_body]
                all_results = [r for r in all_results if r.step_number not in loop_body]
                context.metadata.setdefault("loop_iteration_history", {}).setdefault(
                    loop_key, []
                ).append({
                    "iteration": current_iter,
                    "results": [r.model_dump(exclude={"updated_history"}) for r in old_loop_results],
                })

                loop_restarted = True
                break  # Only one loop restart per batch.

            if loop_restarted:
                continue  # Skip replan checks; restart the while loop.

            # Consume step-level RE-PLAN feedback once a step has executed.
            feedback_map = context.metadata.get("__replan_feedback_by_step")
            if isinstance(feedback_map, dict):
                for result in batch_results:
                    feedback_map.pop(str(result.step_number), None)

            # Capture post-step checkpoint snapshots after state updates are merged.
            if replan_enabled:
                for result in batch_results:
                    if not result.success:
                        continue
                    step = step_by_number[result.step_number]
                    if not self._is_checkpoint_step(step):
                        continue
                    checkpoint_snapshot = self._capture_snapshot(
                        executed_nodes=executed_nodes,
                        all_results=all_results,
                        context=context,
                    )
                    checkpoints.append(
                        _CheckpointSnapshot(
                            name=self._checkpoint_name(step),
                            step_number=step.number,
                            sequence=len(checkpoints) + 1,
                            snapshot=checkpoint_snapshot,
                        )
                    )

            # Run chain-level RE-PLAN policy checks.
            rollback_applied = False
            if replan_enabled and replan_policy is not None:
                for result in batch_results:
                    step = step_by_number[result.step_number]
                    if not self._should_evaluate_replan(policy=replan_policy, step=step, result=result):
                        continue

                    budget_before = self._budget_snapshot(
                        chain_replans=chain_replans,
                        per_step_replans=per_step_replans,
                        target_visits=rollback_target_visits,
                        same_target_streak=same_target_streak,
                    )
                    checker_input = self._build_checker_input(
                        result=result,
                        step=step,
                        context=context,
                        checkpoints=checkpoints,
                        budget_snapshot=budget_before,
                        all_results=all_results,
                    )

                    votes: list[CheckerVote] = []
                    for checker_name, checker_runtime in replan_checkers:
                        check_chain_deadline()
                        try:
                            async with asyncio.timeout(run.remaining(context)):
                                verdict = await checker_runtime.evaluate(checker_input, context)
                            if not isinstance(verdict, ReplanVerdict):
                                raise TypeError(
                                    f"Checker '{checker_name}' returned {type(verdict).__name__}, expected ReplanVerdict"
                                )
                        except Exception as exc:
                            verdict = ReplanVerdict(
                                action=ReplanAction.FAIL,
                                reason=f"RE-PLAN checker '{checker_name}' failed: {exc}",
                                confidence=0.0,
                                metadata={"checker_exception": str(exc)},
                            )
                        votes.append(CheckerVote(checker_name=checker_name, verdict=verdict))

                    check_chain_deadline()
                    aggregate = aggregate_replan_votes(votes, replan_policy.aggregation)
                    final_action = aggregate.selected_verdict.action
                    rollback_target = self._resolve_rollback_target(
                        action=final_action,
                        verdict=aggregate.selected_verdict,
                        policy=replan_policy,
                        current_step_number=result.step_number,
                    )
                    feedback = self._normalize_feedback(aggregate.selected_verdict)
                    budget_exhausted = False
                    event_note = ""

                    if aggregate.triggered and final_action != ReplanAction.CONTINUE:
                        projected_chain_replans = chain_replans + 1
                        projected_step_replans = per_step_replans.get(result.step_number, 0) + 1

                        rollback_target_key = rollback_target.to_key() if rollback_target else "none"
                        projected_target_visits = rollback_target_visits.get(rollback_target_key, 0) + 1
                        projected_streak = (
                            same_target_streak + 1
                            if rollback_target_key == last_rollback_target_key
                            else 1
                        )

                        budgets = replan_policy.budgets
                        budget_limits: list[tuple[bool, str]] = [
                            (
                                budgets.max_replans_per_chain > 0
                                and projected_chain_replans > budgets.max_replans_per_chain,
                                f"max_replans_per_chain={budgets.max_replans_per_chain} exceeded",
                            ),
                            (
                                budgets.max_replans_per_step > 0
                                and projected_step_replans > budgets.max_replans_per_step,
                                (
                                    f"max_replans_per_step={budgets.max_replans_per_step} exceeded for "
                                    f"step {result.step_number}"
                                ),
                            ),
                            (
                                budgets.max_visits_per_checkpoint > 0
                                and projected_target_visits > budgets.max_visits_per_checkpoint,
                                (
                                    f"max_visits_per_checkpoint={budgets.max_visits_per_checkpoint} exceeded for "
                                    f"target {rollback_target_key}"
                                ),
                            ),
                            (
                                budgets.max_same_rollback_target_repeats > 0
                                and projected_streak > budgets.max_same_rollback_target_repeats,
                                (
                                    "max_same_rollback_target_repeats="
                                    f"{budgets.max_same_rollback_target_repeats} exceeded for "
                                    f"target {rollback_target_key}"
                                ),
                            ),
                        ]
                        exhausted_limits = [reason for is_exhausted, reason in budget_limits if is_exhausted]
                        if exhausted_limits:
                            budget_exhausted = True
                            event_note = "; ".join(exhausted_limits)
                            if budgets.fail_on_budget_exhaustion:
                                final_action = ReplanAction.FAIL
                            else:
                                final_action = ReplanAction.CONTINUE

                        if final_action != ReplanAction.CONTINUE:
                            chain_replans = projected_chain_replans
                            per_step_replans[result.step_number] = projected_step_replans
                            rollback_target_visits[rollback_target_key] = projected_target_visits
                            last_rollback_target_key = rollback_target_key
                            same_target_streak = projected_streak

                        if final_action not in {ReplanAction.CONTINUE, ReplanAction.FAIL}:
                            target_snapshot, resolved_target_key = self._resolve_target_snapshot(
                                target=rollback_target or ReplanRollbackTarget(
                                    target_type=ReplanTargetType.CURRENT_STEP
                                ),
                                current_step_number=result.step_number,
                                chain_start_snapshot=chain_start_snapshot,
                                pre_step_snapshots=pre_step_snapshots,
                                checkpoints=checkpoints,
                            )
                            if target_snapshot is None:
                                final_action = ReplanAction.FAIL
                                event_note = f"Unable to resolve rollback target: {resolved_target_key}"
                            else:
                                executed_nodes, all_results = self._restore_snapshot(
                                    target_snapshot,
                                    nodes=nodes,
                                    context=context,
                                )
                                current_history = context.history.copy()
                                checkpoints = [
                                    checkpoint
                                    for checkpoint in checkpoints
                                    if checkpoint.step_number in executed_nodes
                                ]
                                pre_step_snapshots = {
                                    step_number: snapshot
                                    for step_number, snapshot in pre_step_snapshots.items()
                                    if step_number in executed_nodes
                                }
                                rollback_applied = True
                                if feedback:
                                    feedback_map = context.metadata.setdefault("__replan_feedback_by_step", {})
                                    if isinstance(feedback_map, dict):
                                        existing = feedback_map.get(str(result.step_number), [])
                                        if isinstance(existing, str):
                                            existing_items = [existing]
                                        elif isinstance(existing, list):
                                            existing_items = [str(item) for item in existing if str(item).strip()]
                                        else:
                                            existing_items = []
                                        merged: list[str] = []
                                        for item in existing_items + feedback:
                                            if item and item not in merged:
                                                merged.append(item)
                                        feedback_map[str(result.step_number)] = merged
                                event_note = (
                                    event_note or f"Rollback applied to target '{resolved_target_key}'."
                                )

                    aggregation_outcome = ReplanAggregationOutcome(
                        strategy=replan_policy.aggregation.strategy,
                        triggered=aggregate.triggered,
                        trigger_count=aggregate.trigger_count,
                        total_count=aggregate.total_count,
                        mandatory_satisfied=aggregate.mandatory_satisfied,
                        selected_checker=aggregate.selected_checker,
                        selected_action=final_action,
                    )
                    replan_events.append(
                        ReplanEvent(
                            sequence=len(replan_events) + 1,
                            step_number=result.step_number,
                            step_title=result.step_title,
                            checker_votes=[
                                ReplanCheckerVote(
                                    checker_name=vote.checker_name,
                                    action=vote.verdict.action,
                                    reason=vote.verdict.reason,
                                    confidence=vote.verdict.confidence,
                                    suggested_target=vote.verdict.suggested_target,
                                    regeneration_hints=vote.verdict.regeneration_hints,
                                    metadata=vote.verdict.metadata,
                                )
                                for vote in votes
                            ],
                            aggregation=aggregation_outcome,
                            final_action=final_action,
                            rollback_target=rollback_target if final_action != ReplanAction.CONTINUE else None,
                            feedback_passed=feedback if rollback_applied else [],
                            triggering_checkers=aggregate.triggering_checkers,
                            budget_usage=self._budget_snapshot(
                                chain_replans=chain_replans,
                                per_step_replans=per_step_replans,
                                target_visits=rollback_target_visits,
                                same_target_streak=same_target_streak,
                            ),
                            budget_exhausted=budget_exhausted,
                            note=event_note,
                        )
                    )

                    if final_action == ReplanAction.FAIL:
                        replan_failed = True
                        replan_fail_message = (
                            event_note
                            or aggregate.selected_verdict.reason
                            or f"RE-PLAN policy requested fail at step {result.step_number}"
                        )
                        all_results.append(
                            StepExecutionResult(
                                step_number=result.step_number,
                                step_title=result.step_title,
                                step_type=result.step_type,
                                result="",
                                success=False,
                                error_message=f"RE-PLAN failure: {replan_fail_message}",
                                execution_time=0.0,
                                updated_history=context.history.copy(),
                            )
                        )
                        break

                    if rollback_applied:
                        break

            if replan_failed:
                break
            if rollback_applied:
                continue

        if not cancelled and fail_fast_step_number is None:
            check_chain_deadline()

        # Calculate final stats
        total_time = time.time() - start_time
        exclude_skipped = self.fail_fast or termination is not None
        successful_steps = [r for r in all_results if r.success and (not exclude_skipped or not r.skipped)]
        failed_steps = [r for r in all_results if not r.success and (not exclude_skipped or not r.skipped)]
        fail_fast_summary = {
            "enabled": self.fail_fast,
            "triggered": fail_fast_step_number is not None,
            "failed_step_number": fail_fast_step_number,
            "skipped_step_numbers": [r.step_number for r in all_results if r.profiling.get("fail_fast")],
        }

        self._execution_stats.update(
            {
                "total_steps": len(nodes),
                # State counts: how the chain *ended up*. A rolled-back attempt
                # is deliberately absent — it is not part of the answer.
                "executed_steps": len(successful_steps) + len(failed_steps) if self.fail_fast else len(successful_steps),
                "successful_steps": len(successful_steps),
                "failed_steps": len(failed_steps),
                "skipped_steps": sum(r.skipped for r in all_results),
                "parallel_batches": batch_count,
                "total_time": total_time,
                "replan_events": len(replan_events),
                # Ledger count: how much work was *done*, attempts included.
                # ``len(step_results)`` cannot answer this — it is state, and a
                # RE-PLAN retry or loop-back iteration rewinds it. Exposed here
                # because ``ReasoningResult.trace`` (the ledger itself) is an
                # untyped optional field that callers persisting a run may drop.
                "step_executions": len(exec_trace.events),
            }
        )

        # Determine success (false if cancelled)
        chain_success = len(failed_steps) == 0 and not cancelled and not replan_failed

        # Aggregate token usage from the LEDGER, not from the rewindable
        # ``all_results`` state — see the state-vs-ledger note at the TraceEvent
        # append site. Summing from ``all_results`` made a RE-PLAN retry or a
        # loop-back iteration free: the discarded attempt's
        # ``StepExecutionResult`` was dropped by the rewind before this sum ran,
        # so a chain measured at 2 model calls billed 15 tokens instead of 30,
        # and a 3-iteration loop over a 2-step body billed 30 instead of 90.
        # Every ``StepExecutionResult`` that carries usage reaches
        # ``all_results`` via ``all_results.extend(batch_results)``, and the
        # same ``batch_results`` produce the events summed here — so for a chain
        # that never rewinds the two sums are identical, element for element.
        # The results appended to ``all_results`` outside that path (skipped
        # conditional branches, the synthetic RE-PLAN failure record) never
        # carry token usage and so contribute nothing either way.
        total_tokens = exec_trace.total_tokens()

        # Build final output for trace
        trace_output: dict[str, Any] = {
            "success": chain_success,
            "executed_steps": len(successful_steps),
            "failed_steps": len(failed_steps),
            "cancelled": cancelled,
            "total_execution_time": total_time,
            "token_usage": total_tokens,
        }
        # Include the final step result in trace output
        if termination is not None:
            trace_output["final_output"] = termination.result[:2000]
        elif successful_steps:
            last_successful = max(
                successful_steps,
                key=lambda r: (r.profiling.get("batch_index", -1) if has_planner else 0, r.step_number),
            )
            if last_successful.result:
                trace_output["final_output"] = last_successful.result[:2000]
        trace_output["replan_failed"] = replan_failed
        trace_output["replan_events"] = len(replan_events)

        # Update trace with final chain outcome
        trace.update(output=trace_output)

        # End the trace to complete it in Langfuse
        trace.end()

        # Finalize structured execution trace
        exec_trace.total_execution_time = total_time
        exec_trace.success = chain_success
        exec_trace.metadata = {
            "fail_fast": fail_fast_summary,
            "parallel_batches": batch_count,
            "cancelled": cancelled,
            "token_usage": total_tokens,
        }

        result = ReasoningResult(
            success=chain_success,
            history=current_history,
            step_results=all_results,
            total_execution_time=total_time,
            token_usage=total_tokens,
            replan_events=replan_events,
            trace=exec_trace,
            context_metadata=dict(context.metadata),
            metadata={
                "execution_stats": self._execution_stats.copy(),
                "parallel_batches": batch_count,
                "cancelled": cancelled,
                "fail_fast": fail_fast_summary,
                "replan": {
                    "enabled": replan_enabled,
                    "events": len(replan_events),
                    "chain_replans": chain_replans,
                    "per_step_replans": {str(k): v for k, v in per_step_replans.items()},
                    "rollback_target_visits": rollback_target_visits,
                    "failed": replan_failed,
                    "failure_reason": replan_fail_message if replan_failed else "",
                },
            },
        )

        if termination is not None:
            terminated = {
                "step_number": termination.step_number,
                "reason": step_by_number[termination.step_number].step_config.reason,
            }
            result.metadata["terminated"] = terminated
            exec_trace.metadata["terminated"] = terminated.copy()

        # Call chain_complete callback if registered
        if context.on_chain_complete:
            try:
                context.on_chain_complete(result)
            except Exception:
                pass

        # Log chain completion
        log_chain_complete(chain_success, total_time, len(successful_steps), len(nodes))

        # Raise cancellation error after cleanup. The partially-built result
        # is attached so callers can recover completed-step outputs.
        if cancelled:
            raise ExecutionCancelledError(
                f"Chain execution cancelled after completing {len(executed_nodes)}/{len(nodes)} steps. "
                "Partial results are available via exc.result (ReasoningResult).",
                result=result,
            )

        return result

    def _skip_conditional_branches(
        self,
        nodes: list[ExecutionNode],
        conditional_step: int,
        next_step: int,
        executed_nodes: set[int],
        all_results: list[StepExecutionResult],
    ) -> None:
        """Mark steps that should be skipped due to conditional routing.

        When a conditional step selects next_step=3, all other branches
        (steps not reachable from step 3) should be marked as skipped.
        Skipped steps are recorded in all_results with skipped=True.
        """
        step_by_number = {node.step.number: node for node in nodes}

        # Find all steps that should be skipped
        for node in nodes:
            if node.step.number == conditional_step:
                continue  # Don't skip the conditional step itself

            if node.step.number in executed_nodes:
                continue  # Already executed

            # Check if this step is reachable from the target next_step
            if not self._is_reachable_from_target(
                node.step.number, next_step, conditional_step, step_by_number
            ):
                # Mark as executed without actually running
                node.executed = True
                executed_nodes.add(node.step.number)
                # Record a skipped result so it's visible in ReasoningResult
                step = node.step
                step_type = getattr(step, "step_type", StepType.LLM)
                all_results.append(
                    StepExecutionResult(
                        step_number=step.number,
                        step_title=step.title,
                        step_type=step_type,
                        result="",
                        success=True,
                        skipped=True,
                    )
                )

    def _skip_all_conditional_branches(
        self,
        nodes: list[ExecutionNode],
        conditional_step: int,
        executed_nodes: set[int],
        all_results: list[StepExecutionResult],
    ) -> None:
        """Mark all downstream branches of a conditional step as skipped.

        Called when a conditional step has no matching branch and no default,
        i.e. next_step=None.  All steps that depend (directly or transitively)
        on the conditional step are skipped so execution can proceed cleanly.
        """
        # Collect all steps that depend on conditional_step (directly)
        for node in nodes:
            if node.step.number == conditional_step:
                continue
            if node.step.number in executed_nodes:
                continue
            if conditional_step in node.step.dependencies:
                node.executed = True
                executed_nodes.add(node.step.number)
                step = node.step
                step_type = getattr(step, "step_type", StepType.LLM)
                all_results.append(
                    StepExecutionResult(
                        step_number=step.number,
                        step_title=step.title,
                        step_type=step_type,
                        result="",
                        success=True,
                        skipped=True,
                    )
                )

    def _is_reachable_from_target(
        self,
        step_number: int,
        target_step: int,
        conditional_step: int,
        step_by_number: dict[int, ExecutionNode],
    ) -> bool:
        """Check if a step is reachable from the target next_step.

        A step is reachable if:
        - It IS the target step, OR
        - It depends on the target step (directly or transitively)
        """
        if step_number == target_step:
            return True

        node = step_by_number.get(step_number)
        if not node:
            return False

        # Check if this step depends on the target (directly or transitively)
        for dep_node in node.dependencies:
            dep_step = dep_node.step.number
            if dep_step == target_step:
                return True
            if dep_step != conditional_step:  # Don't trace back through conditional
                if self._is_reachable_from_target(dep_step, target_step, conditional_step, step_by_number):
                    return True

        return False

    def get_execution_stats(self) -> dict[str, Any]:
        """Get execution statistics from the last run."""
        return self._execution_stats.copy()
