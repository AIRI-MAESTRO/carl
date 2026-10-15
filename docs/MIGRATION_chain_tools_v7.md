# Chain format v7: embedded chain tools

CARL format 7 adds serialized `chain_tools` at the top level of a
`ReasoningChain`. Older chains migrate without behavioral changes:

```python
raw = json.loads(path.read_text())
migrated = ReasoningChain.migrate(raw)
chain = ReasoningChain.from_dict_typed(migrated)
```

The migration adds `"chain_tools": []`. A runtime that supports only format 6
must reject a v7 chain; otherwise it could silently discard executable nested
chain definitions.

## Creating an embedded chain tool

```python
child_tool = ChainToolDefinition.from_chain(
    name="summarize_document",
    description="Run the pinned document summarization chain.",
    chain=child_chain,
    input_schema={
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
        "additionalProperties": False,
    },
    output_schema={
        "type": "object",
        "properties": {"summary": {"type": "string"}},
        "required": ["summary"],
        "additionalProperties": False,
    },
    output_reference="$steps.1.result_data",
    allowed_tools=["summarize"],
)

parent_chain = ReasoningChain(
    steps=[agent_step],
    chain_tools=[child_tool],
)
```

Before execution, register every host tool listed in `allowed_tools`. Embedded
chain names are registered automatically. The child receives fresh mutable
state and cannot read parent history or memory.

`ChainToolOutcome.status` must be checked explicitly. Only `completed` has
`success=true` and a schema-valid `output`; cancellation, timeout, invalid
boundaries, missing capabilities, nested failure and recursion limits are
non-success outcomes. Snapshot, input and completed-output SHA-256 fields provide
invocation provenance; definition-level byte limits bound retained boundary data.
