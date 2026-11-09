# CARL workflows for small models

## Give each step one output contract

The default Russian and English prompt wrappers carry input, instructions and
history. Put the required answer format in the step. An application can still
set an explicit chain template when all its steps use the same style.

For a value consumed by code, use `StructuredOutputStep` with a narrow JSON
Schema and scoped `input_mapping`. For a human-facing answer, put a concise
format instruction in the final step. Keep examples, explanations and
machine-readable fields separate. Check the actual rendered prompt before
measuring parse failures.

## Extract, validate, execute

1. Extract a small domain state. A swap task needs a mapping of owners to
   objects, an ordered list of swaps, and a target owner.
2. Validate shape with JSON Schema and check domain invariants in a host tool.
   For example, verify that the initial objects form a permutation of the
   options and that each swap names two distinct owners. Schema conformance
   alone does not prove the extracted facts match the source.
3. Execute arithmetic, state updates, or constraint solving in a deterministic
   host tool when the operation has an exact implementation. Preserve evidence
   references for fields extracted from a document.
4. Return the verified result directly if it satisfies the output contract.
   Use a second model call when synthesis or a targeted repair is needed.

```python
from mmar_carl import StructuredOutputStepConfig, StructuredOutputStepDescription
from mmar_carl import ToolStepConfig, ToolStepDescription

extract = StructuredOutputStepDescription(
    number=1,
    title="Extract the operands",
    config=StructuredOutputStepConfig(
        instruction="Extract the two quantities to add from the question.",
        input_mapping={"question": "$outer_context.question"},
        output_schema={
            "type": "object",
            "properties": {"left": {"type": "integer"}, "right": {"type": "integer"}},
            "required": ["left", "right"],
            "additionalProperties": False,
        },
        repair_retries=0,
    ),
)
execute = ToolStepDescription(
    number=2,
    title="Add the quantities",
    dependencies=[1],
    config=ToolStepConfig(
        tool_name="add_operands",
        input_mapping={"operands": "$steps.1.result_data"},
    ),
)
```

The host registers `add_operands` and supplies the validated question in
`ReasoningContext.runtime_input`. `validate_spec()` checks that a producer is
a declared transitive predecessor when a consumer reads `$steps.N` in a
catalogued reference field. The meta mutation helpers run this check
automatically. Call `validate_spec()` explicitly before loading or evaluating
a candidate through other entry points. The predecessor still needs to succeed. For
conditional graphs, place consumers within the producer's branch or define a
join that explicitly handles missing outcomes. Supply host data through
`runtime_input` or memory instead of synthetic step numbers.

## Budget retries and additional reasoning

- Reserve completion capacity for the final answer. Provider limits may count
  reasoning and visible output together. Preserve finish reason and usage,
  including for attempts with no visible answer.
- On `LLMTokenLimitError`, a host policy can change the budget or select a
  supported reasoning setting. Charge the unsuccessful request to the run.
  A retry with unchanged settings spends budget without addressing the limit.
- Bound schema repair attempts and send the specific failed invariant for
  repair. Use `fail_fast` if downstream work requires the failed output.
- Compare an extra LLM step against one longer call on the same model. Where a
  deterministic solver is available, measure both solver-only and equal-tool
  baselines.

## Compare generation and evolution fairly

Use the versioned CARL JSON as the executable contract. A compact generation
schema or a sequence of typed edits can compile to the same contract. Validate
the candidate after each structural edit and reject stale or dangling inputs
before spending inference budget.

Freeze the benchmark, task families, model revisions, available tools, budgets,
scorer and retry policy before collecting comparison results. Record both
parse validity and answer quality. Count all attempted calls, total tokens,
elapsed time and incomplete cases. Keep renderer changes, extra inference and
tool access as separate experimental factors. A small development pilot can
choose what to test next. A claim that a small model matches a larger model
requires a paired held-out comparison and a predeclared non-inferiority margin.

[JSONSchemaBench](https://arxiv.org/abs/2501.10868) ·
[Compute-optimal test-time scaling](https://arxiv.org/abs/2408.03314)
