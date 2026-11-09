# CodeStep adoption: chain format 6 to 7

Format 7 adds the public `code` step for executing an exact Python source string
produced earlier in a run. Existing format-6 chains need no structural rewrite.

## Do not migrate Command/BASH automatically

`CodeStep` is not an alias for command or shell execution. It never accepts a
model-selected executable, argv or shell script. The generated program contract is
exactly one synchronous function:

```python
def run(inputs):
    return {"answer": inputs["value"] * 2}
```

Store that exact source string in runtime context memory and reference it from
`CodeStepConfig.source`. CARL does not extract Markdown, repair syntax or ask an LLM
to retry.

## Serialized step configuration

```python
from mmar_carl import CodeStepConfig, CodeStepDescription

step = CodeStepDescription(
    number=2,
    title="Execute generated calculation",
    config=CodeStepConfig(
        source="$memory.generated.python_source",
        runtime_profile="python-safe-v1",
        input_mapping={"value": "$memory.input.value"},
        input_schema={
            "type": "object",
            "properties": {"value": {"type": "number"}},
            "required": ["value"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {"answer": {"type": "number"}},
            "required": ["answer"],
            "additionalProperties": False,
        },
        timeout_seconds=5,
        output_key="calculation",
    ),
)
```

Both schemas are mandatory. Only finite JSON values cross the runtime boundary,
and memory is written only after successful output validation.

## Host-owned runtime policy

The chain stores only `runtime_profile`. The host must separately attach a
runtime-only `CodeExecutionPolicy` to `ReasoningContext`:

```python
from mmar_carl import CodeExecutionPolicy, CodeRuntimeProfile

context.code_execution_policy = CodeExecutionPolicy(
    profiles={
        "python-safe-v1": CodeRuntimeProfile(
            runtime="docker",
            revision="python-runtime-v1",
            interpreter=("python", "-I", "-B"),
            prepare_config={
                "image": "python@sha256:<64-lowercase-hex-digest>",
            },
            cpu_limit=1,
            mem_limit="256m",
            pids_limit=32,
        ),
    },
)
```

The image must be pinned by digest. V1 requires strict non-host isolation, no
network, enforced time/output/CPU/memory/PID limits, controlled workspace files and
read-only host bind mounts. Local and best-effort runtimes are rejected before
generated code executes. The policy is excluded from context serialization so chain
data cannot grant itself execution authority.

## Typed outcomes

Read `StepExecutionResult.as_code_execution_outcome()` for the Pydantic
`CodeExecutionOutcome`. `completed` is the only successful status. Validation,
policy denial, runtime unavailability, timeout, cancellation, invalid output and
runtime failure remain distinct terminal states with source/runtime provenance.

Older CARL runtimes reject format-7 chains through the normal newer-format check.
`ReasoningChain.migrate()` only advances existing format-6 payloads to version 7; it
does not invent CodeSteps or runtime authority.
