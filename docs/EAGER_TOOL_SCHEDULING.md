# Dependency-scoped tool scheduling

Use the experimental `execute_eager_tools()` host API for a static graph of
independent async tools. A step can start as soon as its dependencies have
succeeded and a worker is available. Results are returned in step declaration
order.

```python
from mmar_carl import ToolStepConfig, ToolStepDescription
from mmar_carl.eager_tools import execute_eager_tools


async def add_one(value):
    return {"value": value + 1}


async def double(value):
    return {"value": value * 2}


async def run():
    steps = [
        ToolStepDescription(
            number=1,
            title="Add one",
            config=ToolStepConfig(
                tool_name="add_one",
                input_mapping={"value": "$outer_context.value"},
            ),
        ),
        ToolStepDescription(
            number=2,
            title="Double",
            dependencies=[1],
            config=ToolStepConfig(
                tool_name="double",
                input_mapping={"value": "$steps.1.result_data.value"},
            ),
        ),
    ]
    return await execute_eager_tools(
        steps,
        tools={"add_one": add_one, "double": double},
        outer_context={"value": 20},
        max_concurrency=2,
    )
```

The host supplies an explicit mapping of trusted functions declared with
`async def`. These
functions must be stateless, safe to run in any order consistent with their
dependencies, and responsive to cancellation. Each step receives only the JSON
inputs it selects from `$outer_context` and its declared ancestors, or quoted
string literals. Mutating an input inside a tool cannot change another step's
input. CARL validates the graph and reference scope before calling tools.

A failed step blocks its descendants. Unrelated branches can finish, and a
blocked step is returned with `success=False` and `skipped=True`. A tool
timeout from `ToolStepConfig.timeout` fails that step. An overall `timeout`
cancels in-flight tools and waits for their cleanup before raising
`TimeoutError`. Run cancellation also waits for in-flight tool cleanup.
The scheduler sends at most one cancellation to each tool. A tool with its
own cancellation source must handle overlap with the scheduler's
cancellation. Use the scheduler's deadlines when it should own cleanup.

Use the regular `ReasoningChain.execute_async()` path for chains that use shared
memory or history, named events, conditional routing, checkpoints, dynamic
steps, or RE-PLAN. It uses batch boundaries to determine when those effects
become visible.
