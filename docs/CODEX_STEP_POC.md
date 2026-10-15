# Codex-as-step PoC

## Outcome

The PoC is feasible. CARL can delegate a DAG node to a local Codex agent,
collect the final response and token usage, persist the Codex thread id, and
resume that thread from a later dependent step.

The implementation uses the official Python Codex SDK behind the optional
`mmar-carl[codex]` extra. The SDK starts a local Codex app-server process;
authentication remains in the host's existing Codex installation and is not
stored in a chain or `ReasoningContext`.

## Minimal contract

`CodexStepDescription` owns a serializable `CodexStepConfig` with:

- a static instruction and bounded named inputs resolved from CARL context;
- an optional working directory, model, and reasoning-effort override;
- `read-only` (default) or explicit `workspace-write` filesystem authority;
- a whole-step timeout;
- optional thread resumption through a CARL context reference;
- optional structured outcome storage in CARL memory.

The executor maps a completed turn into:

- `result`: Codex's final response;
- `result_data`: thread id, turn id, status, sandbox, cwd, and raw usage;
- `token_usage`: CARL's `prompt` / `completion` / `total` convention;
- `model`: the configured model override, when one was supplied.

## Safety boundary

The PoC deliberately excludes Codex `full-access`. Headless runs always use
`ApprovalMode.deny_all`, preventing a serialized chain from pausing for or
automatically gaining more authority. A chain must opt into
`workspace-write`; otherwise the Codex thread is read-only.

Credentials, environment overrides, arbitrary Codex launch arguments, and
approval policy are not part of the serialized step config.

## Try it

```bash
pip install 'mmar-carl[codex]'
codex login status
python examples/agents/codex_step_example.py
```

Hermetic tests run with the normal suite. The real local-runtime smoke test is
opt-in:

```bash
CARL_CODEX_LIVE=1 uv run --extra codex pytest \
  tests/agents/test_codex_step.py::test_live_local_codex_sdk_smoke \
  -m live -q
```

## Production follow-ups

1. Stream Codex turn events into CARL's progress and `on_llm_chunk` callbacks.
2. Define a host-owned policy object for allowed workspaces and
   `workspace-write`, analogous to CARL's command/code execution policies.
3. Add explicit cancellation telemetry and stress-test parallel Codex steps.
4. Decide whether durable thread ids belong in CARL memory, an external run
   store, or both.
5. Add model/cost normalization once the SDK exposes the actual resolved model
   and stable billing metadata for each turn.
6. Evaluate app-server reuse across steps to reduce startup latency while
   preserving concurrency and cleanup guarantees.

