"""Unified per-run descriptor for multi-objective evolution.

Packages the scattered signals from a single chain run — quality, cost (tokens),
time, complexity (graph shape), and fault-tolerance — into ONE
:class:`RunDescriptor`, so an evolution engine gets a canonical objective vector
(for a Pareto front, see :mod:`mmar_carl.pareto`) and a map-elites niche key from
a single object instead of reaching into ``ReasoningResult`` + ``ReasoningChain``
+ per-step results by hand.

``quality`` and ``fault_tolerance`` are *inputs* — CARL doesn't decide what
"quality" means (that's your metric) nor score fault-tolerance (that's your
LLM-judge / exceptional-scenario harness); it packages them next to the
structural signals it can extract.
"""

from __future__ import annotations

import bisect
from typing import Any, Optional

from pydantic import BaseModel, Field


class ComplexityDescriptor(BaseModel):
    """Structural shape of a chain's DAG (no run required — see :func:`extract_complexity`)."""

    num_steps: int = Field(..., description="Number of steps.")
    num_edges: int = Field(..., description="Total dependency edges.")
    max_depth: int = Field(..., description="Number of topological batches (critical-path depth).")
    max_parallel_batch: int = Field(..., description="Widest parallel batch (max steps runnable at once).")
    distinct_step_types: int = Field(..., description="Number of distinct step types used.")
    has_loops: bool = Field(default=False, description="True when any step declares a loop_back_to.")


class RunDescriptor(BaseModel):
    """Everything an evolution engine needs to score/place one chain run."""

    success: bool = Field(..., description="Whether the chain executed without failing.")
    quality: Optional[float] = Field(default=None, description="Target-metric score (caller-supplied).")
    total_tokens: int = Field(default=0)
    prompt_tokens: int = Field(default=0)
    completion_tokens: int = Field(default=0)
    wall_time_s: float = Field(default=0.0)
    llm_calls: int = Field(default=0, description="Steps that recorded token usage.")
    failed_steps: int = Field(default=0)
    skipped_steps: int = Field(default=0)
    complexity: ComplexityDescriptor
    fault_tolerance: Optional[float] = Field(
        default=None, description="Caller-supplied robustness score (e.g. LLM-as-a-judge)."
    )
    scores_by_metric: dict[str, float] = Field(default_factory=dict)

    # -- objective access -------------------------------------------------
    def value(self, key: str, *, default: float = 0.0) -> float:
        """Resolve a dotted objective key to a float.

        Supported keys: ``quality``, ``total_tokens``, ``prompt_tokens``,
        ``completion_tokens``, ``wall_time_s``, ``llm_calls``, ``failed_steps``,
        ``skipped_steps``, ``fault_tolerance``, ``complexity.<field>``, and
        ``metric.<name>`` (from ``scores_by_metric``). ``None`` → *default*.
        """
        if key.startswith("complexity."):
            v = getattr(self.complexity, key.split(".", 1)[1], None)
            return float(v) if v is not None else default
        if key.startswith("metric."):
            return float(self.scores_by_metric.get(key.split(".", 1)[1], default))
        v = getattr(self, key, None)
        if isinstance(v, bool):
            return float(v)
        return float(v) if isinstance(v, (int, float)) else default

    def as_objectives(self, keys: list[str], *, default: float = 0.0) -> tuple[float, ...]:
        """Project to an N-D objective vector for :func:`mmar_carl.pareto.pareto_front`."""
        return tuple(self.value(k, default=default) for k in keys)

    def behavior_key(self, bins: dict[str, list[float | int]]) -> tuple[int, ...]:
        """Discretise chosen dimensions into a map-elites niche key.

        ``bins`` maps an objective key to ascending bin edges; the value's bin
        index (via ``bisect``) becomes one coordinate of the returned tuple. The
        caller owns the niche layout; CARL only extracts + bins.
        """
        return tuple(bisect.bisect_right(edges, self.value(key)) for key, edges in bins.items())


# ---------------------------------------------------------------------------
# Extraction (duck-typed on ReasoningResult / ReasoningChain to avoid cycles).
# ---------------------------------------------------------------------------


def extract_complexity(chain: Any) -> ComplexityDescriptor:
    """Structural descriptor from a ``ReasoningChain`` (no execution needed)."""
    deps: dict[int, list[int]] = chain.get_step_dependencies()
    num_steps = len(deps)
    num_edges = sum(len(v) for v in deps.values())

    completed: set[int] = set()
    remaining = set(deps)
    batches: list[list[int]] = []
    while remaining:
        batch = [n for n in remaining if all(d in completed for d in deps[n])]
        if not batch:  # cycle guard — shouldn't happen for a valid chain
            break
        batches.append(batch)
        completed |= set(batch)
        remaining -= set(batch)

    distinct = len({getattr(s, "step_type", None) for s in chain.steps})
    has_loops = any(getattr(s, "loop_back_to", None) is not None for s in chain.steps)
    return ComplexityDescriptor(
        num_steps=num_steps,
        num_edges=num_edges,
        max_depth=len(batches),
        max_parallel_batch=max((len(b) for b in batches), default=0),
        distinct_step_types=distinct,
        has_loops=has_loops,
    )


def extract_run_descriptor(
    result: Any,
    chain: Any,
    *,
    quality: Optional[float] = None,
    scores_by_metric: Optional[dict[str, float]] = None,
    fault_tolerance: Optional[float] = None,
) -> RunDescriptor:
    """Package a single ``ReasoningResult`` (+ its ``ReasoningChain``) into a
    :class:`RunDescriptor`. ``quality`` / ``scores_by_metric`` / ``fault_tolerance``
    are caller-supplied (metric + judge outputs)."""
    tu = getattr(result, "token_usage", None) or {}
    step_results = getattr(result, "step_results", []) or []
    llm_calls = sum(1 for s in step_results if (getattr(s, "token_usage", None) or {}).get("total", 0) > 0)
    skipped = sum(1 for s in step_results if getattr(s, "skipped", False))
    if hasattr(result, "get_failed_steps"):
        failed = len(result.get_failed_steps())
    else:
        failed = sum(1 for s in step_results if not getattr(s, "success", True) and not getattr(s, "skipped", False))

    return RunDescriptor(
        success=bool(getattr(result, "success", False)),
        quality=quality,
        total_tokens=int(tu.get("total", 0)),
        prompt_tokens=int(tu.get("prompt", 0)),
        completion_tokens=int(tu.get("completion", 0)),
        wall_time_s=float(getattr(result, "total_execution_time", None) or 0.0),
        llm_calls=llm_calls,
        failed_steps=failed,
        skipped_steps=skipped,
        complexity=extract_complexity(chain),
        fault_tolerance=fault_tolerance,
        scores_by_metric=dict(scores_by_metric or {}),
    )


__all__ = [
    "ComplexityDescriptor",
    "RunDescriptor",
    "extract_complexity",
    "extract_run_descriptor",
]
