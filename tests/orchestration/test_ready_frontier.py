"""Parity with the full-graph readiness scan for static execution waves."""

from types import SimpleNamespace

import pytest

from mmar_carl import (
    LLMClientBase,
    ReasoningChain,
    ReasoningContext,
    ToolStepConfig,
    ToolStepDescription,
)
from mmar_carl._ready_frontier import ReadyFrontier
from mmar_carl.executor import DAGExecutor
from mmar_carl.models.enums import StepType


def make_step(number, dependencies=(), **kwargs):
    return SimpleNamespace(number=number, dependencies=dependencies, **kwargs)


def test_frontier_matches_scan_with_resume_and_out_of_order_declarations():
    nodes = DAGExecutor().build_execution_graph(
        [
            make_step(8, [7]),
            make_step(5, [1, 1, 2]),
            make_step(2),
            make_step(7, [1]),
            make_step(1),
            make_step(3, [8, 5]),
        ]
    )
    next(node for node in nodes if node.step.number == 2).executed = True

    frontier = ReadyFrontier.for_static_graph(nodes, replan_enabled=False)
    assert frontier is not None

    for expected in ([1], [5, 7], [8], [3], []):
        assert frontier.ready_nodes == [node for node in nodes if node.is_ready()]
        assert [node.step.number for node in frontier.ready_nodes] == expected
        for node in frontier.ready_nodes:
            node.executed = True
        frontier.advance()


def test_frontier_keeps_unstarted_nodes_ready():
    nodes = DAGExecutor().build_execution_graph(
        [
            make_step(3, [1]),
            make_step(2),
            make_step(1),
        ]
    )
    frontier = ReadyFrontier.for_static_graph(nodes, replan_enabled=False)
    assert frontier is not None

    nodes[2].executed = True
    frontier.advance()

    assert [node.step.number for node in frontier.ready_nodes] == [3, 2]


@pytest.mark.parametrize(
    ("step_fields", "replan_enabled"),
    [
        ({"step_type": StepType.CONDITIONAL}, False),
        ({"loop_back_to": 1}, False),
        ({"triggered_by": ["ready"]}, False),
        ({}, True),
    ],
)
def test_frontier_defers_graphs_that_can_change_readiness_outside_the_wave(step_fields, replan_enabled):
    nodes = DAGExecutor().build_execution_graph([make_step(1, **step_fields)])
    assert ReadyFrontier.for_static_graph(nodes, replan_enabled=replan_enabled) is None


class _UnusedLLM(LLMClientBase):
    async def get_response(self, prompt: str) -> str:
        raise AssertionError("Only tools should run")

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        raise AssertionError("Only tools should run")


@pytest.mark.asyncio
async def test_next_wave_sees_merged_history_from_entire_previous_wave():
    context = ReasoningContext(outer_context="", api=_UnusedLLM(), model="mock")
    context.register_tool("first", lambda: "first output")
    context.register_tool("sibling", lambda: "sibling output")
    context.register_tool("echo", lambda value: value)
    chain = ReasoningChain(
        steps=[
            ToolStepDescription(number=1, title="First", config=ToolStepConfig(tool_name="first")),
            ToolStepDescription(number=2, title="Sibling", config=ToolStepConfig(tool_name="sibling")),
            ToolStepDescription(
                number=3,
                title="Child",
                dependencies=[1],
                config=ToolStepConfig(tool_name="echo", input_mapping={"value": "$history[-1]"}),
            ),
        ],
        max_workers=2,
    )

    result = await chain.execute_async(context)

    assert result.success
    assert [event.batch_index for event in result.trace.events] == [0, 0, 1]
    assert "sibling output" in result.trace.get_event(3).inputs["value"]
