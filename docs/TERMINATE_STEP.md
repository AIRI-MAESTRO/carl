# TerminateStep

Use `TerminateStepDescription` when a workflow reaches a branch with a
complete answer. The chain stops after the current batch and returns that
answer through `ReasoningResult.get_final_output()`.

```python
TerminateStepDescription(
    number=3,
    title="Return accepted answer",
    dependencies=[2],
    config=TerminateStepConfig(
        output_source="$steps.2.result_data",
        reason="The answer passed validation",
    ),
)
```

Supply exactly one of `output` (a literal JSON value, including null) or
`output_source` (a context reference). A missing reference fails the step.
Strings are returned as text, while other JSON values are rendered as JSON.

- The scheduler lets the entire selected batch settle. Siblings may still
  perform side effects, including siblings waiting for a worker slot. Make
  a side-effectful step depend on a downstream branch if it must never run
  once an early answer is available.
- The lowest-numbered successful TerminateStep in a batch selects the output.
  Remaining steps are reported as skipped. No new planner actions, loop
  iterations, or RE-PLAN work are scheduled from that batch.
- Existing step failures remain failures. Host cancellation and fail-fast
  take priority. The termination step number and reason appear in the result
  and trace metadata under `terminated`.
- A snapshot that already executed a TerminateStep cannot resume past it.
  In-process pause and resume before termination use normal batch semantics.
- Chain format 15 introduces the new step type. The earlier formats migrate
  without rewriting existing steps.
