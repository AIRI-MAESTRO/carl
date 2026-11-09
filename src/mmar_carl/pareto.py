"""N-dimensional Pareto-dominance kernel (pure Python, no numpy).

Small, self-contained helpers for multi-objective selection. Feed them the
objective vectors produced by :meth:`mmar_carl.run_descriptor.RunDescriptor.as_objectives`
plus a per-objective ``sense`` (``"max"`` to maximise, ``"min"`` to minimise).
An evolution engine composes e.g.
``{quality: max, tokens: min, wall_time: min, num_steps: min, fault_tolerance: max}``.
Visualization and the selection policy stay with the caller.
"""

from __future__ import annotations

from typing import Literal, Sequence

Sense = Literal["max", "min"]


def dominates(a: Sequence[float], b: Sequence[float], senses: Sequence[Sense]) -> bool:
    """True when point ``a`` Pareto-dominates ``b`` — no worse on every objective
    (per its sense) and strictly better on at least one."""
    strictly_better = False
    for av, bv, sense in zip(a, b, senses):
        if sense == "min":
            av, bv = -av, -bv
        if av < bv:
            return False
        if av > bv:
            strictly_better = True
    return strictly_better


def pareto_front(points: Sequence[Sequence[float]], *, senses: Sequence[Sense]) -> list[int]:
    """Indices of the non-dominated points (the Pareto front)."""
    n = len(points)
    return [i for i in range(n) if not any(j != i and dominates(points[j], points[i], senses) for j in range(n))]


def pareto_rank(points: Sequence[Sequence[float]], senses: Sequence[Sense]) -> list[int]:
    """NSGA-style front index per point (0 = Pareto front, 1 = next front, …)."""
    n = len(points)
    ranks = [-1] * n
    remaining = set(range(n))
    rank = 0
    while remaining:
        current = [
            i for i in remaining if not any(dominates(points[j], points[i], senses) for j in remaining if j != i)
        ]
        if not current:  # pragma: no cover - cyclic dominance can't happen with a strict order
            current = list(remaining)
        for i in current:
            ranks[i] = rank
        remaining -= set(current)
        rank += 1
    return ranks


__all__ = ["Sense", "dominates", "pareto_front", "pareto_rank"]
