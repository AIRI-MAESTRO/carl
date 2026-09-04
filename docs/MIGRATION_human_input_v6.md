# HumanInputStep migration: chain format 5 to 6

Format 6 replaces the fallback-based HumanInputStep contract with one typed,
process-local text request.

## Host callback

Before:

```python
async def provider(prompt, future):
    value = await read_from_ui(prompt)
    future.set_result(value)
```

After:

```python
from mmar_carl import HumanInputRequest, HumanInputResponse

async def provider(request: HumanInputRequest) -> HumanInputResponse:
    value = await read_from_ui(request.prompt)
    return HumanInputResponse(
        request_id=request.request_id,
        value=value,
        actor_id="authenticated-user-id",
    )
```

Assign the callback to `context.on_human_input_requested` as before. CARL now owns
the callback task, timeout and cancellation cleanup. Synchronous callbacks may
return `HumanInputResponse` directly, but must not block.

## Missing input and timeout

`fallback_value` is removed. A missing callback returns `unavailable`; timeout
returns `timed_out`; cancellation returns `cancelled`. None is reported as a human
answer or written to memory. Tests and non-interactive jobs must install an explicit
fake-human callback.

Loading format-5 chain JSON through `ReasoningChain.migrate()` removes the serialized
legacy `fallback_value` and emits a warning about the semantic change. Format-6
chains are rejected by older runtimes through the normal newer-format check.

## Typed outcome and sensitive input

Read `StepExecutionResult.as_human_input_outcome()` for the Pydantic
`HumanInputOutcome`. `answered` is the only successful status.

Set `sensitive=True` together with `output_memory_key` to keep the raw value out of
the step result and model-visible history. The value is written only to
`$memory.human_input.<output_memory_key>`; applications remain responsible for
protecting and clearing that memory.
