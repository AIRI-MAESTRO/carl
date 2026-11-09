"""Index ready waves for runs with a fixed graph and monotonic completion."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .models.enums import StepType

if TYPE_CHECKING:
    from .executor import ExecutionNode


class ReadyFrontier:
    """Track readiness while preserving the executor's batch boundaries."""

    @classmethod
    def for_static_graph(cls, nodes: list[ExecutionNode], *, replan_enabled: bool) -> ReadyFrontier | None:
        if replan_enabled or any(
            getattr(node.step, "step_type", None) == StepType.CONDITIONAL
            or getattr(node.step, "loop_back_to", None) is not None
            or getattr(node.step, "triggered_by", None)
            for node in nodes
        ):
            return None
        return cls(nodes)

    def __init__(self, nodes: list[ExecutionNode]) -> None:
        self._declaration_order = {node.step.number: index for index, node in enumerate(nodes)}
        self._pending_dependencies = {
            node.step.number: sum(not dependency.executed for dependency in node.dependencies)
            for node in nodes
            if not node.executed
        }
        self.ready_nodes = [node for node in nodes if self._pending_dependencies.get(node.step.number) == 0]

    def advance(self) -> None:
        """Expose the next wave after the executor marks this wave complete.

        Discard the index if a result injects steps before calling this method.
        This index relies on completion changing only for the current wave.
        """
        next_wave: list[ExecutionNode] = []
        for node in self.ready_nodes:
            if not node.executed:
                next_wave.append(node)
                continue
            for dependent in node.dependents:
                if dependent.executed:
                    continue
                number = dependent.step.number
                self._pending_dependencies[number] -= 1
                if self._pending_dependencies[number] == 0:
                    next_wave.append(dependent)

        # Edges are visited in completion-wave order, which can differ from
        # declaration order. Declaration order governs same-key memory merges.
        next_wave.sort(key=lambda node: self._declaration_order[node.step.number])
        self.ready_nodes = next_wave
