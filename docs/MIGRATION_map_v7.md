# Migrating chains to format 7 (MapStep)

CARL chain format 7 adds the serialized `map` step type. Existing format-6 chains
need no structural changes:

```python
from mmar_carl import ReasoningChain

migrated = ReasoningChain.migrate(raw_chain)
assert migrated["format_version"] == 7
```

Format 7 runtimes continue to load migrated older steps. A runtime supporting only
format 6 must reject a format-7 chain rather than attempting to interpret an unknown
`map` step.

Before executing a MapStep, register its configured tool in `ReasoningContext` and
provide a JSON array through the configured source. `ReasoningChain.preflight()`
reports a missing Map tool before execution.

MapStep uses process-local cancellation. Async tool calls are cancelled and awaited;
a synchronous tool already running in a worker thread cannot be forcibly stopped and
may continue after item timeout or cancellation. This limitation is recorded in the
aggregate `enforcement_gaps` field.
