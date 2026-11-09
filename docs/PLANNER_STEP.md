# PlannerStep

A planner selects a finite, ordered sequence of tool actions already declared in the chain. CARL validates the action IDs against a JSON Schema, then adds ordinary tool steps to the running DAG. An empty sequence is valid. A static step that depends on the planner waits for the selected sequence to finish.

```python
from mmar_carl import PlannerStepConfig, PlannerStepDescription, ToolStepConfig

plan = PlannerStepDescription(
    number=1,
    title="Choose checks",
    config=PlannerStepConfig(
        instruction="Select the checks requested by the user.",
        input_mapping={"request": "$outer_context.request"},
        actions={
            "inventory": ToolStepConfig(
                tool_name="check_inventory",
                input_mapping={"product_id": "$outer_context.product_id"},
            ),
            "price": ToolStepConfig(
                tool_name="check_price",
                input_mapping={"product_id": "$outer_context.product_id"},
            ),
        },
        max_actions=2,
    ),
)
```

The host registers the tools as usual. Supply validated inputs through
`ReasoningContext.runtime_input`. The planner prompt receives the explicit
`input_mapping` values and the action IDs and descriptions. Tool parameter
mappings stay in the chain. The model cannot provide tool arguments or add
another planner. Use trusted chain configurations and register only tools that
the host is willing to invoke. Preflight lists the primary and fallback tools.

`$steps.N.result_data.actions` reports the selected action IDs. For a nonempty
plan, `$steps.N.result_data.step_numbers` holds the numbers assigned to the
injected tools, in the same order. Tool results are recorded as ordinary step
results. Enable chain `fail_fast` if a tool failure should stop later actions.

`ReasoningContext.on_step_complete` runs when the planner has selected its
actions, while other steps in the same batch may still be running. For a
nonempty plan, `on_step_event(planner_number, "planner_actions_scheduled", payload)`
fires after the DAG has assigned the tool step numbers and before those tools
start. The payload contains parallel `actions` and `step_numbers` lists.

Format 13 introduces the new step. Existing chains migrate without changing
their steps. In-memory pause and resume use the existing scheduler. Chains
with a planner reject RE-PLAN, loop-back, trace replay and resume from a saved snapshot.
The sum of declared planner budgets must fit the chain's `max_injections`.

## Related step designs

| Need | CARL boundary |
| --- | --- |
| Typed model judgment | The `decision` step already has a separate implementation in #41. Rebase its format migration after the first new step merges. |
| START and END contracts | #10 explores durable entry and exit schemas. Treat these as chain boundaries so the graph and downstream consumers can validate a whole invocation. |
| Dynamic subgraph synthesis | The planner deliberately begins with finite tool actions. General generated subgraphs need a durable record of the graph, its injections and their remaining budget before snapshot resume or RE-PLAN can be supported. |
| HTTP, stop and agent loops | The older #4 overlaps the current `wait`, `code`, `map` and bounded `agent` steps. Ordinary HTTP calls can use host-registered tools. A terminate primitive needs a defined contract for in-flight work, pending steps and result status. |
| Remote workers and evolution | CARL supplies execution primitives. Work dispatch and receipts belong to the collaboration layer, and topology search belongs to EvoC. |
