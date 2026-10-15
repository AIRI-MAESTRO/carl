# Release Notes

Detailed release notes for CARL (Multi-step Agentic Reasoning with LLMs) library.

## Version 0.4.0 - 2026-08-27

### ✨ Headlines

CARL 0.4.0 is the _runtime execution_ release. 0.3.0 taught chains how to
reason, orchestrate, and observe; 0.4.0 teaches them how to **act** — run
commands, execute generated code, wait on real time and events, fan out over
data, and call other chains — under a single explicit rule:

> **A serialized chain is input, never authority.**

Every new execution capability separates the _request_ (serialized, portable,
diffable chain data) from the _permission_ (runtime-only, host-injected, never
written to chain JSON). Seven new step types ship with typed terminal outcomes,
bounded budgets, and fail-closed defaults. The chain format advances from
version 1 to version 9, each rung a deliberate compatibility gate.

Every capability in this release was designed against
[`docs/RUNTIME_STEP_SPEC.md`](docs/RUNTIME_STEP_SPEC.md) — a living
specification that records the accepted public contract, non-goals, and
decision log for each step type.

### 🧩 Seven New Step Types

| Step type                     | `StepType`      | Purpose                                                  |
| ----------------------------- | --------------- | -------------------------------------------------------- |
| `AgentStepDescription`        | `agent`         | Bounded ReAct tool loop with an explicit `finish`        |
| `CodeStepDescription`         | `code`          | Execute an exact, runtime-generated Python source string |
| `WaitStepDescription`         | `wait`          | Self-contained async timer / named-event wait            |
| `MapStepDescription`          | `map`           | Bounded ordered fan-out over a JSON array                |
| `CommandPlanStepDescription`  | `command_plan`  | LLM picks typed args for a host capability               |
| `CommandStepDescription`      | `command`       | Execute an argv command through a skill runtime          |
| `ShellSessionStepDescription` | `shell_session` | Run a static script in one shell process                 |

### 🤖 AgentStep — bounded tool loop

`AgentStepConfig` is a _bounded_ ReAct loop, deliberately not a second
orchestrator:

- **Explicit non-empty tool allowlist** — `tools=[...]` is required; `finish`
  is reserved and rejected in the allowlist.
- **Exactly one call per iteration.** Zero or multiple calls are protocol
  errors returned as observations: they consume an iteration and execute
  nothing.
- **`finish` is the only successful exit.** Exhausting `max_iterations` is not
  success.
- Layered budgets: `max_iterations` (8), `max_tool_calls` (12),
  `timeout_seconds` (120), `model_timeout_seconds` (60),
  `tool_timeout_seconds` (30), optional aggregate `max_tokens`,
  `max_tool_result_chars` (16k), `max_transcript_chars` (100k).
- Optional `output_schema` validates `finish.result`; `output_key` /
  `output_namespace` write to memory **only after** a valid finish.
- `input_mapping` values must be `$`-prefixed context references and are
  supplied as data — the model never rewrites the goal's inputs.

### 🐍 CodeStep — exact generated Python, host-sandboxed

`CodeStep` executes one exact source string produced earlier in the run. It is
**not** an alias for shell execution and never accepts a model-selected
executable, argv, or script.

- Fixed program contract: exactly one synchronous `def run(inputs): ...`.
  CARL does not extract Markdown, repair syntax, or ask an LLM to retry.
- `input_schema` **and** `output_schema` are both mandatory; only finite JSON
  values cross the runtime boundary, and memory is written only after
  successful output validation.
- The chain stores only a `runtime_profile` **name**. The host attaches a
  runtime-only `CodeExecutionPolicy` with `CodeRuntimeProfile` entries to
  `ReasoningContext.code_execution_policy`.
- Profile `python-safe-v1` requires strict non-host isolation, no network,
  digest-pinned images, and enforced time / output / CPU / memory / PID limits.
  Local and best-effort runtimes are rejected _before_ generated code runs.
- `StepExecutionResult.as_code_execution_outcome()` returns a typed
  `CodeExecutionOutcome`. `completed` is the only success; `invalid_source`,
  `invalid_input`, `denied`, `runtime_unavailable`, `timed_out`,
  `invalid_output`, `failed`, and `cancelled` stay distinct with
  source/runtime provenance.
- `ReasoningChain.required_code_profiles()` and `PreflightReport
.missing_code_profiles` surface a missing profile before execution.

See [`docs/MIGRATION_code_step_v7.md`](docs/MIGRATION_code_step_v7.md).

### ⏱️ WaitStep — real timers and events

A self-contained asynchronous wait with four discriminated conditions:

- `AfterWaitCondition` — relative, finite, non-negative duration.
- `AtWaitCondition` — absolute, **timezone-aware** timestamp.
- `EventWaitCondition` — a named process-local event.
- `AnyOfWaitCondition` — race at least two leaf conditions.

`StepExecutionResult.as_wait_outcome()` returns a typed `WaitOutcome`
(`trigger`, `elapsed_seconds`, winning `condition_index`, event `payload`).

The event bus was reworked to make this correct: parallel executor snapshots
now **share** `_event_bus_token` by reference rather than copying and merging
after `gather`. A WaitStep can therefore observe an event emitted by a sibling
in the _same_ parallel batch instead of deadlocking on it. Emission is
loop-safe from synchronous tool threads via `call_soon_threadsafe`.

### 🗺️ MapStep — bounded ordered fan-out

- `items_source` resolves once to a JSON array (`$outer_context`,
  `$memory.*`, `$steps.*`, `$metadata.*`, `$event.*`); shared
  `input_mapping` values resolve once too.
- Each element is passed to one registered tool under `item_parameter`, with
  optional zero-based `index_parameter`.
- Bounds: `max_items` (1000), `max_concurrency` (8, capped at 256),
  `item_timeout_seconds` (30).
- **Collect-all** contract: `MapOutcome` is an ordered, self-validating
  aggregate of `MapItemOutcome` records (`completed` / `failed` / `timed_out`
  / `cancelled`) in contiguous input order, with per-status counts that must
  reconcile. `StepExecutionResult.as_map_outcome()` returns it typed.
- The DAG executor commits _only_ the configured aggregate output from a
  MapStep snapshot — no other failed-step memory mutation leaks back.
- V1 has no retry, filter, reduce, per-item memory write, or graph mutation.
- Known limitation, recorded in `MapOutcome.enforcement_gaps`: a synchronous
  tool already running in a worker thread cannot be forcibly stopped and may
  outlive item timeout or cancellation.

See [`docs/MIGRATION_map_v7.md`](docs/MIGRATION_map_v7.md).

### 🖥️ CommandStep & ShellSessionStep — argv execution through a runtime

`CommandStepConfig` runs one executable from an explicit **argv list** — there
is no implicit shell, so metacharacters in resolved values cannot inject extra
commands. Resolved `input_mapping` values are appended as discrete argv tokens
_and_ exported as `CARL_ARG_<NAME>`; they are never string-interpolated. A
preceding LLM step therefore cannot silently replace the executable.

`ShellSessionStepConfig` is the deliberate counterpart: a **static** command
sequence executed by one POSIX shell process sharing cwd, variables,
functions, and filesystem state. The script is piped to `<shell> -s`, is
covered in full by the approval fingerprint, and `stop_on_error=True` prefixes
`set -e`. Authorizing the shell authorizes every command inside it, so
interpreter approval stays on by default.

Shared `_RuntimeExecutionConfig` controls:

- `runtime` — `local` / `docker` / `e2b` / `firejail` (or a registered backend).
- `enforcement_mode` — `strict` (default) fails _before_ launch if a requested
  control is not actually enforced; `best_effort` runs only where host policy
  permits and reports the gaps.
- `network` — fail-closed `none` default, plus `allowlist` (exact hosts) and
  `host`.
- `timeout`, `artifact_io_timeout`, `cpu_limit`, `mem_limit`, `pids_limit`,
  `max_output_bytes`, `allow_nonzero_exit`.
- No host environment inheritance: local execution receives only
  `PATH`/`LANG`/`LC_ALL` plus explicit `env` keys and CARL-derived variables.
  `CARL_ARG_*` and `CARL_ARTIFACT_*` prefixes are reserved.
- `output_key` is validated against the reserved command result-metadata keys
  (`exit_code`, `stdout`, `stderr`, `policy_decision`, `enforcement_report`,
  `capability_fingerprint`, …).

### 🔐 CommandPolicy — host-owned execution authority

Permission never lives in chain JSON. The application supplies a frozen
`CommandPolicy` on `ReasoningContext.command_policy`:

- `allowed_executables` are **exact strings** — never basenames or globs. For
  host-launching runtimes (`local`, `firejail`) the rule and the requested
  executable must be absolute paths, and the normalized path becomes the
  actual `exec` target.
- `approval_required_executables` are eligible only after the host callback
  approves the complete invocation.
- Unknown executables, runtimes, networks, env keys, and host cwd roots are
  denied.
- Ceilings: `max_timeout`, `max_cpu_limit`, `max_memory_bytes`,
  `max_pids_limit`, `max_output_bytes`, `max_argument_count`,
  `max_artifact_io_timeout`, plus `approval_timeout`,
  `network_enforcer_timeout`, `runtime_cleanup_timeout`.
- `require_approval_for_planned` is typed `Literal[True]` — an invariant, not
  a tuning knob. `require_approval_for_interpreters` defaults to `True`.
- `ReasoningContext.on_command_approval_requested` receives a frozen
  `CommandApprovalRequest`. The reviewer sees full `argv`, but `argv` is
  `exclude=True`: only the executable plus argument _placeholders_
  (`argv_preview`, `dynamic_argument_count`) reach serialization and the
  public step event. The `fingerprint` covers every execution-affecting
  value, including hashes of stdin and env values.

### 🧭 CommandPlanStep — typed capability planning

`CommandPlanStep` lets a model choose _arguments_, never a program.

- The serialized step carries an `instruction`, a requested `capability_ids`
  subset, and planner `input_mapping` references — nothing more.
- The runtime-only `CommandCapabilityRegistry` on `ReasoningContext` owns the
  capability ids, argument schemas, and trusted argv builders. It is never
  emitted by `chain.to_dict()`.
- The LLM never supplies an executable, argv prefix, runtime, network mode, or
  resource limit.
- A `CommandPlanRecord` is bound to a capability **revision and fingerprint**;
  a planned `CommandStep` consumes it via `plan_source` with
  `planned_capability_ids` narrowing (never widening) the host registry.
- Planned commands always cross the approval boundary.

### 🌐 Host-owned network enforcement

New `network_enforcement` module makes the egress trust boundary typed and
auditable:

- `NetworkEnforcer` protocol + `PreconfiguredNetworkEnforcer` attestation
  adapter, `ManagedNetworkProfile`, `NetworkEnforcementRequest` /
  `NetworkEnforcementPlan` / `NetworkEnforcementLease`, and the
  `DockerNetworkBinding` / `FirejailNetworkBinding` binding kinds.
- `PreconfiguredNetworkEnforcer` is explicitly **not** a provisioning system:
  it creates no Docker networks, configures no Firejail interfaces, and
  installs no firewall rules. The operator attests that each configured
  binding really enforces its profile's exact allowlist.
- `ReasoningContext.network_enforcer` is runtime-only and excluded from
  context dumps and durable snapshots. Neither an enforcer, a live lease, nor
  a raw binding may appear in chain JSON — only the JSON-safe
  `NetworkEnforcementPlan` (profile, revision, exact hosts, binding kind,
  opaque binding commitment, fingerprint) may enter an approval record or
  execution result.
- Runtime identity propagates across parallel, replay/resume, handoff, and
  supervisor paths without ever becoming durable.

### 📦 Portable artifacts

`models/artifacts.py` adds `ArtifactInput` / `ArtifactOutput` /
`ArtifactRecord`. Artifacts deliberately carry **bytes, not host paths**: a
producing step returns base64 content, and a later step declares that record
as an input for CARL to stage below the runtime's `in/` directory. Outputs are
collected from `out/` under an independent `artifact_io_timeout` and bounded
size limit. Chain JSON stays portable, and a serialized chain can never ask
the host to read an arbitrary filesystem path.

### 🧱 Skill runtime hardening

- `RuntimeCapabilities` declares per-backend static guarantees for
  `isolation`, `wall_time`, `output_limit`, `cpu_limit`, `memory_limit`,
  `pids_limit`, `network_none`, `network_allowlist`, `workspace_files`,
  `persistent_shell`, and `artifact_output_limit`. Custom runtimes that
  declare nothing are treated conservatively as unsupported.
- `EnforcementReport` + `assess_runtime_enforcement()` /
  `get_runtime_capabilities()` compare requested controls against real
  backend capability. `.gaps()`, `.fully_enforced()`, `.as_dict()`,
  `.with_control()`. `advisory` explicitly does not mean prevention.
- Docker, E2B, and Firejail backends were substantially reworked for
  persistent sessions, bounded workspace file I/O, process-group kills, and
  `network="allowlist"` that fails closed until managed egress attests.
- New `e2b` extra (`mmar-carl[e2b]`, `e2b>=2.38,<3`), also folded into
  `mmar-carl[all]`.

### 🧩 Chain as a tool

`ChainToolDefinition` embeds a complete `ReasoningChain.to_dict()` snapshot as
a callable, isolated tool, serialized at the chain's new top-level
`chain_tools` field.

- `ChainToolDefinition.from_chain(name=..., chain=..., input_schema=...,
output_schema=..., output_reference=..., allowed_tools=[...])`.
- The child receives validated arguments as JSON in `$outer_context` and
  fresh mutable state. It **cannot** read parent history, memory, messages,
  metadata, callbacks, event bus, or command/network authority.
- Bounds: `max_depth` (4), `timeout_seconds` (120), `max_input_bytes` /
  `max_output_bytes` (1 MB each). `snapshot_sha256` plus input and completed-
  output digests give invocation provenance.
- `ChainToolOutcome.status` must be checked explicitly: only `completed` is
  success. `failed`, `cancelled`, `timed_out`, `invalid_input`,
  `invalid_output`, `unavailable`, and `recursion_limit` are all non-success.
- `ChainBuilder.add_chain_tool(definition)`; embedded chain names register
  automatically, but every host tool named in `allowed_tools` must be
  registered before execution.

See [`docs/MIGRATION_chain_tools_v7.md`](docs/MIGRATION_chain_tools_v7.md).

### 🙋 HumanInputStep — typed lifecycle (breaking)

`fallback_value` is **removed**. Silent fallback success is gone.

- The host callback is now `async def provider(request: HumanInputRequest) ->
HumanInputResponse`, assigned to `context.on_human_input_requested`. CARL
  owns the callback task, timeout, and cancellation cleanup.
- `StepExecutionResult.as_human_input_outcome()` returns a typed
  `HumanInputOutcome`. `answered` is the **only** successful status; a missing
  callback yields `unavailable`, a timeout `timed_out`, cancellation
  `cancelled`. None of these is reported as a human answer or written to
  memory.
- `sensitive=True` (which now requires `output_memory_key`) keeps the raw
  value out of the step result and model-visible history, writing only to
  `$memory.human_input.<key>`.
- Config bounds: `prompt` ≤ 8192 chars, `min_length` / `max_length`
  (≤ 65536), finite `timeout`.

See [`docs/MIGRATION_human_input_v6.md`](docs/MIGRATION_human_input_v6.md).

### 🧾 Structured output validation

`StructuredOutputStepExecutor` now enforces the declared contract instead of
trusting the model:

- The schema is compiled and `check_schema`-validated **before any provider
  call**, so an invalid contract costs nothing.
- Remote references (`$ref`, `$dynamicRef`, `$recursiveRef` not starting with
  `#`) are rejected — validation can never perform network I/O.
- An unknown `$schema` dialect **fails closed** rather than silently falling
  back to the library's newest draft and changing the contract's meaning.
- Parsed output is validated with the dialect's `FORMAT_CHECKER`. Failure
  messages name the JSON path and the failed keyword but **never reflect the
  rejected instance** back into logs.
- Token usage and the resolved client `model` are now recorded on the step
  result for both success and failure paths. Streaming keeps `token_usage`
  empty rather than reporting fabricated zeros.
- `jsonschema>=4.23.0,<5.0.0` is now a **core dependency**.

### 🤖 LLM client & telemetry

- New `LLMClientBase.get_response_with_tools_and_usage(...)` returns
  `(content, tool_calls, usage)`. The base implementation delegates to
  `get_response_with_tools` and returns empty usage, so existing custom
  clients keep working; `OpenAICompatibleClient` overrides it with real
  prompt/completion counters.
- `RecordingLLMClient` / `PlayingLLMClient` persist and replay `usage` in
  cassettes.
- **Behavioural fix:** malformed tool-call `arguments` JSON is preserved as
  the raw string instead of being coerced to `{}`, so the AgentStep protocol
  validator cannot accidentally execute a no-argument tool after a parse
  failure.
- The legacy `httpx` import in `OpenAICompatibleClient` is now lazy — OpenAI
  SDK 3 ships `httpx2` and no longer installs `httpx`.

### 💰 Cost estimation

`estimate_chain_cost` now understands the new LLM-calling steps:

- `command_plan` counts one model call ("typed command capability selection").
- `agent` counts `max_iterations` calls (documented as an upper bound: one
  model call per AgentStep iteration).
- Prompt sizing reads `config.goal` when a step has no `aim`.

### 🔁 Serialization & chain format v9

`ReasoningChain.FORMAT_VERSION` advanced **1 → 9**, one rung per capability,
with a migration ladder in `ReasoningChain.migrate()`:

| Version | Adds                                                              |
| ------- | ----------------------------------------------------------------- |
| 2       | `shell_session` step + portable artifact declarations             |
| 3       | `CommandPlanStep` + planned `CommandStep` wire fields             |
| 4       | `agent` step                                                      |
| 5       | `wait` step                                                       |
| 6       | HumanInputStep `fallback_value` removal (rewrites configs, warns) |
| 7       | `code` step                                                       |
| 8       | `map` step                                                        |
| 9       | top-level `chain_tools`                                           |

Policy change, documented in
[`docs/SERIALIZATION_COMPAT.md`](docs/SERIALIZATION_COMPAT.md): adding a step
type is now treated as **forward-incompatible**. Bump `FORMAT_VERSION` and add
a rung even when no structural rewrite is needed, so an older reader raises the
typed `ChainFormatNewerError` at the chain boundary instead of an incidental
enum error halfway through step reconstruction.

`loop_back_to` and `loop_config` are now documented guaranteed round-trip
fields. Round-trip coverage spans all 19 step types across the dict, JSON, and
public JSON-loader paths.

### 🛡️ Preflight

`PreflightReport` gains `required_code_profiles` / `missing_code_profiles`,
folded into `all_present` and `format_text()` alongside tools, MCP servers,
and skills.

### 🔧 Breaking & Behavioural Changes

1. **`HumanInputStepConfig.fallback_value` removed.** Constructing a config
   with it now raises. The callback signature changed to
   `HumanInputRequest -> HumanInputResponse`. Missing providers and timeouts
   are explicit non-success outcomes, not answers.
2. **Chain format 1 → 9.** Chains written by 0.4.0 are rejected by 0.3.0
   runtimes via `ChainFormatNewerError`. Run `ReasoningChain.migrate(raw)` to
   lift older payloads forward.
3. **Structured output is now validated.** A step that previously "succeeded"
   with schema-violating JSON now fails. Invalid schemas, remote `$ref`s, and
   unknown `$schema` dialects fail before the provider call.
4. **`allow_unsafe_local` / `allow_unenforced_network` are ignored.** Old
   chains still load, but emit a `DeprecationWarning`; supply a host-owned
   `CommandPolicy` instead. Chain data cannot grant itself execution
   authority.
5. **The `bash` step type was renamed to `command`.** `StepType._missing_`
   keeps pre-merge `"bash"` JSON loadable while only the clearer `CommandStep`
   API is exposed.
6. **Event bus semantics changed.** Parallel snapshots now share the bus by
   reference instead of copying and merging after `gather`, so sibling
   emissions are visible within the same batch.
7. **Malformed tool-call arguments** are surfaced as the raw provider string
   rather than an empty dict.
8. **New core dependency:** `jsonschema>=4.23.0,<5.0.0`.

### Migration Guide from v0.3.0

1. **Lift your chain JSON.** `migrated = ReasoningChain.migrate(raw)` →
   `format_version == 9`. The 5 → 6 rung strips `fallback_value` and warns.
2. **Rewrite human-input providers** to the typed
   `HumanInputRequest`/`HumanInputResponse` shape, and install an explicit
   fake-human callback in tests and non-interactive jobs — there is no
   fallback any more.
3. **Audit structured-output schemas.** Remove remote `$ref`s, confirm the
   `$schema` dialect is one `jsonschema` recognises, and expect real
   validation failures where the model previously drifted.
4. **Inject host authority before running any execution step.** Set
   `context.command_policy`, and as applicable
   `context.command_capability_registry`, `context.network_enforcer`, and
   `context.code_execution_policy`. None of these can come from chain JSON.
5. **Replace `allow_unsafe_local` / `allow_unenforced_network`** with explicit
   `CommandPolicy` allowlists.
6. **Check terminal statuses explicitly.** `as_code_execution_outcome()`,
   `as_map_outcome()`, `as_wait_outcome()`, `as_human_input_outcome()`, and
   `ChainToolOutcome.status` all distinguish success from six-to-eight
   non-success states. Do not treat "no exception" as success.
7. **Register dependencies up front.** MapStep tools, chain-tool
   `allowed_tools`, and CodeStep runtime profiles are all reported by
   `chain.preflight(context)` before execution.
8. **Custom `LLMClientBase` subclasses** need no change — but override
   `get_response_with_tools_and_usage` if you want AgentStep token budgets to
   be enforced against real usage.

### 🗂️ Tests, Examples, Docs

- Suite grew from 542 to 666 test functions; **2,954 hermetic tests pass**
  (16 skipped, 15 live/mcp_live/skill_runtime_live deselected).
- New test modules: `test_agent_step.py`, `test_code_step.py`,
  `test_chain_as_tool.py`, `test_map_step.py`, `test_wait_step.py`,
  `test_command_steps.py`, `test_command_plan_steps.py`,
  `test_command_capabilities.py`, `test_managed_network_command_steps.py`,
  `test_network_enforcement.py`, `test_shell_session_artifacts.py`,
  `test_artifact_runtime_io.py`,
  `test_structured_output_schema_validation.py`.
- New examples: `agents/chain_as_tool_example.py`,
  `agents/code_step_example.py`, `orchestration/command_steps_example.py`,
  `orchestration/command_plan_capabilities_example.py`,
  `orchestration/command_llm_context_example.py`,
  `orchestration/map_step_example.py`,
  `orchestration/shell_session_artifacts_example.py`. Most run hermetically
  with no provider call.
- New docs: `RUNTIME_STEP_SPEC.md` (living capability specification with
  per-capability decision logs), plus migration guides for HumanInput (v6),
  CodeStep (v7), MapStep (v7), and chain tools (v7).
- New `skill_runtime_live` pytest marker for tests that exercise a real
  Docker / Firejail / E2B daemon; each skips gracefully when its backend is
  unavailable.

## Version 0.3.0 - 2026-06-02

### ✨ Headlines

CARL 0.3.0 is the largest release since the typed-step API. It graduates
multi-agent orchestration, sandboxed skill execution, observability /
visualization, and chain evolution from experimental to first-class. CARE
(Collaborative Agent Reasoning Editor) consumes the new typed metadata,
preflight, RunRecord, and pause/resume APIs.

### 🤝 Multi-Agent Orchestration

New first-class step types for hierarchical and ensemble patterns:

- `AgentHandoffStepDescription` — explicit sub-chain delegation with input/output mapping.
- `SupervisorStepDescription` — LLM-routed dispatch to specialist sub-chains.
- `DebateStepDescription` — round-robin multi-role debate + judge synthesis.
- `ParallelSamplingStepDescription` — N-sample voting (`MAJORITY_VOTE`) or LLM-judge aggregation (`BEST_OF_N`).
- `HumanInputStepDescription` — pause for human input (callable or webhook).
- `EvaluationStepDescription` — inline metric quality gate with `EvalFailAction` policies.
- `LoopConfig` (`while`/`until`) on any step + `ChainBuilder.add_while_loop`/ `add_until_loop`.
- True conditional branching — only the matched branch executes; siblings are skipped (was: all branches executed).
- Intra-chain event bus — `emit_event` + `event_dependencies` for fan-out triggers.

### 🧱 Sandboxed AgentSkill Execution

`AgentSkillStepConfig.runtime` now dispatches to four backends:

- `local` — host process (default).
- `docker` — `DockerSkillRuntime` runs scripts in containers with a bind-mounted `/workspace`.
- `e2b` — `E2BSkillRuntime` provisions an e2b.dev microVM per invocation; workspace round-trips via `sandbox.files`.
- `firejail` — `FirejailSkillRuntime` lightweight Linux process sandbox.

All backends self-register via `register_skill_runtime`; unknown values raise `SkillRuntimeError`. Network policy comes from the SKILL.md `allowed-tools` frontmatter (auto-allowlisted) or from `runtime_config["network"]`.

New `LLM_AGENT` execution mode runs a progressive-disclosure tool-calling loop (`run_script` / `read_file` / `write_file` / `list_resources`) inside the workspace, capturing `output_files` written to `/workspace/out/`.

### 🔗 Skill Resolver

URI-based skill identity:

- `github://owner/repo[/subdir][@ref]` — tarball download from
  `codeload.github.com`, no `git` required.
- `https://...tar.gz` — generic HTTPS tarball.
- `module://my_pkg.skills.pdf` — bundled Python package.
- Bare paths and skill names (cataloged via `SkillLoader.catalog_all`).

`trust_policy="sha_pinned"` verifies SHA256 of `SKILL.md` after extraction.
`filter_security_terms=True` strips password/encrypt/decrypt sections from LLM prompts. `list_cached_skills()` enumerates the `~/.cache/mmar_carl/skills/` cache.

### 🧠 Memory & State

- Three-layer memory: namespaced short-term, session metadata, optional long-term via `LTMBase` (`InMemoryLTM` / `JsonFileLTM`).
- COW (copy-on-write) memory isolation replaces deep-copy for parallel batches; bytes-saved profiling surfaced under `result.metadata`.
- `ReasoningContext(memory_schema={...})` validates `memory[namespace][key]` types at write-time; raises `MemorySchemaError`.
- `LazyMemoryValue` for deferred materialisation via `unwrap_lazy(...)`.
- Pre-execution `$memory.*` reference validation warns when a step reads a key that no prior step writes.
- Smart history truncation: `max_history_entries` + `history_truncation_strategy` (recency / token-budget).
- New dynamic references: `$ltm.key`, `$event.<name>`.

### 🛠️ Tool System

- Dynamic discovery: `ModuleToolSource`, `CallableToolSource`, `DictToolSource` + `ToolDiscoveryStepDescription`.
- `@carl_tool` decorator with tags and `name_prefix` namespacing.
- `ToolStepConfig.error_recovery`: `RAISE` / `SKIP` / `RETRY` / `FALLBACK`.
- Tool pipelining — chain tool outputs to subsequent tools via the existing `$history` / `$metadata` references.
- `context.register_tools_from_path(glob, *, tag_filter, name_prefix)` registers `@carl_tool`-decorated callables from disk in one call.
- Async-safe tool registry.

### 📊 Observability & Visualization

- `ExecutionTrace` attached to every `ReasoningResult.trace`: serialisable (`to_json` / `from_json`), diffable (`trace.diff`), Gantt-renderable (`format_gantt(format="text"|"mermaid")`), and replayable via `trace.to_html(path)` — a standalone animated playback file with zero external deps.
- `TraceAggregator` reports `{p50, p95, p99, mean, max}_ms` latency and token-usage percentiles across N runs.
- `ChainVisualizer(result, chain=..., evolution_result=...)` fluent facade: `.token_pie().gantt().heatmap(metric="tokens").print()`.
- `ReasoningResult` formatters: `format_token_pie`, `format_prompt_completion_breakdown`, `format_profiling_table`, `format_cost_by_model(pricing=...)`.
- `ReasoningChain` formatters: `to_mermaid`, `to_mermaid_critical_path`, `to_mermaid_heatmap(metric="tokens"|"latency"|"cost")`.
- Jupyter rich display (`_repr_markdown_`) on `ReasoningResult`, `EvolutionResult`, `DatasetEvaluationReport`, `CostEstimate`, `ChainVisualizer`.
- Optional `mmar-carl[viz]` extra pulls `matplotlib` for PNG output paths.

### 💰 Cost & Token Management

- `chain.estimate_cost(context, pricing={...})` returns a `CostEstimate` with a per-step `format_table()` projection — zero LLM calls.
- Per-step token budget enforcement.
- Per-model cost rollup via `result.format_cost_by_model(pricing=...)`.

### 🧪 Evaluation Framework

- `DatasetEvaluator` runs a chain against any `DataCase`-based dataset and emits a `DatasetEvaluationReport` with per-case scores, step outcomes, latencies, and step-level metric scores.
- `EvalSuite` lightweight golden-output regression harness.
- `EvaluationStepDescription` — inline quality gates inside a chain.
- `ChainTestHarness` for deterministic chain unit tests.
- Built-in match metrics: `ExactMatchMetric`, `CaseInsensitiveMatchMetric`, `ContainsMetric`, `RegexMatchMetric`.
- Case-aware metrics: `MetricBase.compute_async(output, *, case=None)` dispatched via `call_metric_async`.
- `DatasetEvaluationReport` formatters: `format_failure_heatmap`, `format_step_metric_heatmap`, `format_score_distribution`, `format_latency_histogram`, `format_cost_trend(pricing=...)`.

### 🤖 LLM Inference

- `AnthropicClient` — native (non-OpenAI-compatible) client implementing
  the full `LLMClientBase` surface plus Anthropic-only extras:
  extended thinking (`get_response_with_thinking`), prompt caching
  (`cache_system=True`), vision (`get_response_with_image` — URL or base64),
  native tool use, and streaming. Lazy `anthropic` SDK import.
- `StructuredOutputStepExecutor` now streams: when the client exposes
  `stream_response` and `on_llm_chunk` is wired, partial chunks are
  forwarded with `stage="structured_output"` and parsed early once the
  brace/bracket balance closes.
- `RecordingLLMClient` / `PlayingLLMClient` — pytest-vcr-style cassettes
  (`sha256(method + prompt + model + temperature)` keyed) for deterministic
  offline replay. Missing keys raise `CassetteMissError`.
- Native multi-turn message history (`ChatMessage` Pydantic model) on
  `get_response_with_messages` and `get_response_with_tools`.
- `RetryPolicy` with transient-only retries (exponential backoff + jitter,
  configurable `retry_on_status`). 401/403/404/422 fail fast by default.
- Typed introspection on `LLMClientBase`: `model_name`, `temperature`,
  `max_tokens`, `supports_streaming`.

### 🧬 Chain Generation & Evolution

- `ChainBuilder.from_description(task, llm_client, ...)` — LLM-planned
  chains with `max_retries` self-correction and full provenance
  (`planner_prompt` / `planner_reply` / `planner_attempts`) written to
  `chain.metadata`.
- `ChainEvolver` — evolutionary search over chain variants. Multi-objective
  fitness (`metric: MetricBase | list[MetricBase]` + optional `fitness_fn`),
  elitism, atomic checkpoint/resume, concurrent population evaluation,
  cost pre-flight via `EvolutionCostEstimate`.
- `ChainMutator` mutation kinds: `MODEL_SWAP`, `TEMPERATURE_SWAP`,
  `PROMPT_REWRITE`, `MAX_WORKERS`, `INSERT_STEP`, `DELETE_STEP`. Every
  mutation round-trips through `from_dict` validation; failures roll back.
- `EvolutionResult` formatters: `format_score_evolution`, `format_pareto`,
  `format_spend_vs_quality`, `format_mutation_effectiveness`,
  `to_lineage_mermaid` (parent edges, gold border on best chain).
- Top-level `format_runs_pareto(results, ...)` for cross-run Pareto charts.

### 🔌 MCP Integration

Graduated from experimental:

- Stdio, SSE, and streamable_http transports stabilised.
- `MCPServerConfig` field set frozen.
- `MCPSessionPool` — one `ClientSession` per server reused via
  `async with context.mcp_pool():`.
- `context.register_mcp_tools()` auto-discovery.
- `MCPResourceStepDescription` + `context.list_mcp_resources(server, ...)`
  for resource fetches as context input.
- Live integration tests under `pytest -m mcp_live` exercise a real
  `FastMCP` server subprocess.

### 🔄 Streaming, Cancellation, Pause/Resume

- `ReasoningContext.cancel()` / `is_cancelled()` / `reset_cancellation()`
  backed by a shared `_CancelToken` so parallel snapshots see each other's
  flips. Cancelled mid-step → `skipped=True,
error_message="cancelled by user"`. Partial outputs surfaced on the
  cancelled `ReasoningResult`.
- `context.request_pause()` / `ContextSnapshot` + `execute_async(...,
resume_from=...)` for pause/resume across processes.
- `chain.stream_async(ctx)` yields `StepExecutionResult` per step plus a
  terminal `ReasoningResult`.
- Streaming callbacks: `on_step_start`, `on_step_complete`, `on_progress`,
  `on_llm_chunk` (legacy `(chunk)` and extended `(chunk, *, step_number,
stage)` shapes supported transparently), `on_human_input_requested`,
  `on_step_event(step_num, event_type, payload)`. Events wired in
  Supervisor, ParallelSampling, Debate, and AgentSkill LLM_AGENT.

### 📦 Result & Chain Serialization

- Lossless `ReasoningResult.to_dict(*, full=True)` / `from_dict` /
  `to_json` / `from_json` / `save` / `load`.
- `StepExecutionResult.to_dict(*, truncate=False)` + `from_dict`.
- `RunRecord` Pydantic model bundles chain + input + result + timestamps
  - runtime info with `RunRecord.from_run(...)` convenience constructor.
- Chain JSON format version tag (`carl_version` / `format_version`) +
  `ChainFormatNewerError` typed exception for forward-compatibility warnings;
  `ReasoningChain.migrate(data, to_version)` migration ladder for format upgrades.
- `CareChainMetadata` / `CareContextFile` typed models at
  `models.care_metadata`; `ReasoningChain.set_care_metadata` /
  `get_care_metadata`; `ReasoningContext.from_chain_inputs(chain, api=...)`
  primes a fresh context from saved metadata + file paths.
- Typed views over `StepExecutionResult.result_data`: `SkillOutput`,
  `DebateTranscript` / `DebateTurn`, `ParallelSamples`, `SupervisorDecision`.

### 🛡️ Tool Registry & Preflight

- `ReasoningChain.required_tools()` / `required_mcp_servers()` /
  `required_skills()` — static introspection across the chain graph.
- `ReasoningChain.preflight(context) -> PreflightReport` — compares
  requirements against the context's registries and surfaces missing
  tools.

### ⚡ Performance & Reliability

- Step result caching (`StepCache`) memoizes per-step outputs.
- Parallel batch size auto-tuning.
- Per-step retry policy for LLM transient errors.
- Large memory payload optimisation via COW.

### 🗂️ Tests, Examples, Notebooks

- `tests/` reorganised into nine topic submodules (`agents`,
  `chain_lifecycle`, `evaluation`, `llm_inference`, `mcp`, `memory`,
  `orchestration`, `replan`, `tool_calling`) with auto-generated README
  per topic (`make docs-topic-index`).
- `examples/` mirrors the same topic structure with auto-generated README.
- `notebooks/02_visualizations_demo.ipynb` rebuilt from a Python source
  and runs offline via committed cassettes (`make notebook-smoke`).
- `@pytest.mark.live` opt-in marker for real-API smoke tests
  (`make test-live`).

### 🧰 Configuration & Builder

- Chain-level `default_llm_config` — steps without an `llm_config` inherit the
  chain default; per-step configs merge/override field-by-field.
- `StepGroup(name, steps, llm_config)` — apply config overrides to a set of
  steps at once (e.g. separate "creative" from "analytical" phases). Passed
  via `ReasoningChain(step_groups=[...])`.

### 🔧 Breaking & Behavioural Changes

- Conditional steps now execute **only the matched branch** — code that
  relied on every branch running must be updated.
- Parallel-step memory isolation switched from deep-copy to COW. Writes
  from parallel steps remain visible only to subsequent batches.
- `RetryPolicy` defaults to transient-only retries; 401/403/404/422 fail
  fast. Override via `retry_on_status` if you need the old behaviour.

### Migration Guide from v0.2.0

1. **Conditional chains** — verify only the matched branch should run.
   If you depended on side effects from non-matched branches, restructure
   the chain so those steps run unconditionally.
2. **Adopt typed metadata** — use `chain.set_care_metadata(...)` instead
   of writing into `chain.metadata["care"]` by hand.
3. **MCP** — drop the EXPERIMENTAL caveat; consider migrating to
   `MCPSessionPool` for hot paths.
4. **AgentSkill** — pin trusted skills with
   `trust_policy="sha_pinned"` + `skill_sha256=...`.
5. **Anthropic users** — replace OpenRouter passthrough with
   `AnthropicClient(AnthropicClientConfig(...))` to access extended
   thinking and prompt caching natively.

## Version 0.2.0 - 2026-04-15

### 🔧 Breaking Changes

#### LLM Client Refactoring

**Moved LLMClientBase to separate module**

- `LLMClientBase` moved from `mmar_carl.models.base` to `mmar_carl.models.llm_client_base`
- Update imports if you're directly importing this class:

  ```python
  # Old (deprecated)
  from mmar_carl.models.base import LLMClientBase

  # New (correct)
  from mmar_carl.models.llm_client_base import LLMClientBase
  ```

**Removed legacy mmar-llm-mapi integration**

- Completely removed integration with the deprecated `mmar-llm-mapi` library
- Removed `create_llm_client()` function
- Use `create_openai_client()` for OpenAI-compatible APIs instead

**Rationale:** Simplifies the architecture and removes dependencies on deprecated libraries.

---

### ✨ New Features

#### Comprehensive Test Suite

Added 74 new comprehensive tests (4,237 lines of test code):

**New Test Files:**

- `tests/test_conditional_steps.py` - 20 tests for conditional branching patterns
  - Built-in patterns: contains, equals, startswith, endswith, matches, empty, nonempty
  - Complex expressions with simpleeval
  - Multi-branch routing with default steps
  - Serialization and ChainBuilder integration

- `tests/test_advanced_tool_steps.py` - 13 tests for tool integration
  - Multi-step tool chains with parallel execution
  - Input mapping with $metadata, $history, $outer_context references
  - Tool error handling and parameter mapping
  - Complex data flow between tool steps

- `tests/test_llm_council.py` - 10 tests for multi-model voting patterns
  - Parallel council member execution
  - Per-step model overrides
  - Vote aggregation (unanimous, majority, split)
  - Council synthesis and complex scenarios

- `tests/test_structured_output_advanced.py` - 17 tests for JSON schema validation
  - Pydantic model validation
  - JSON schema validation (nested, complex)
  - Error recovery from invalid JSON
  - Strict vs lenient parsing modes

- `tests/test_execution_modes_advanced.py` - 14 tests for FAST/SELF_CRITIC modes
  - FAST mode single-pass behavior
  - SELF_CRITIC with custom evaluators
  - Evaluator chains and revision limits
  - Mixed execution modes and performance characteristics

**Enhanced Mock Infrastructure:**

- `tests/mocks.py` - 12 specialized mock clients for testing
  - MockLLMClient - Basic mock for general testing
  - ConditionalMockClient - Pattern-based responses
  - CouncilMockClient - Multi-model council simulation
  - ToolTrackingMockClient - Tracks tool execution flow
  - StructuredOutputMockClient - JSON response simulation
  - ReplanScenarioMockClient - RE-PLAN scenario simulation
  - ExecutionModeMockClient - FAST vs SELF_CRITIC mode testing
  - ChainBuilderMockClient - Chain building validation
  - And more...

**Test Coverage:** 266 tests total (192 existing + 74 new) - 100% pass rate

#### Examples Runner

**New `examples/runner.py`** - Universal runner for all examples

```bash
# Run any example with automatic setup
python examples/runner.py --example basic_chain_example

# List available examples
python examples/runner.py --list

# Run with custom parameters
python examples/runner.py --example llm_council_example --model claude-3-5-sonnet
```

Benefits:

- Simplified example execution without manual setup
- Better testing and demonstration capabilities
- Consistent environment across all examples
- Easy to add new examples

---

### 🐛 Bug Fixes

#### BUG-001 (HIGH): Conditional Steps Execute All Branches

**Problem:** Conditional steps were executing ALL possible branch target steps instead of only the matched branch.

**Fix:** DAG executor now respects conditional routing decisions

- Added `_skip_conditional_branches()` helper method
- Added `_is_reachable_from_target()` helper method
- Only executes steps that are reachable from the matched branch

**Before:**

```python
# Conditional step matches condition "int(value) >= 70"
# Routes to step 3 (High Score)
# BUG: Executes steps 1, 3, 4, 2 (all branches)
assert len(result.step_results) == 4  # Wrong!
```

**After:**

```python
# Conditional step matches condition "int(value) >= 70"
# Routes to step 3 (High Score)
# CORRECT: Executes only steps 1, 2, 3 (matched branch)
assert len(result.step_results) == 3  # Correct!
```

**Files Modified:** `src/mmar_carl/executor.py`

---

#### BUG-002 (MEDIUM): Step Type Field Missing from Serialization

**Problem:** The `step_type` field was not included when step descriptions were serialized using `model_dump()`.

**Fix:** Added `model_dump()` override to all 7 step description classes

- `LLMStepDescription`
- `ToolStepDescription`
- `MCPStepDescription`
- `MemoryStepDescription`
- `TransformStepDescription`
- `ConditionalStepDescription`
- `StructuredOutputStepDescription`

**Before:**

```python
step_dict = cond_step.model_dump()
assert "step_type" not in step_dict  # Missing!
```

**After:**

```python
step_dict = cond_step.model_dump()
assert "step_type" in step_dict  # Present!
assert step_dict["step_type"] == "conditional"
```

**Impact:** Serialization round-trips now work correctly. Chain save/load functionality is fixed.

**Files Modified:** `src/mmar_carl/models/steps.py`

---

#### BUG-003 (MEDIUM): Inconsistent Branch Definition Formats

**Problem:** ChainBuilder accepted tuples `("condition", step_number)` but direct construction required `ConditionalBranch` objects, creating API inconsistency.

**Fix:** ChainBuilder now accepts both formats with automatic normalization

**Before:**

```python
# ChainBuilder format - worked
branches=[("contains:positive", 3)]

# Direct construction format - failed
ConditionalStepDescription(
    branches=[("contains:positive", 3)]  # ValidationError!
)
```

**After:**

```python
# Both formats work everywhere!
ChainBuilder().add_conditional_step(
    branches=[("contains:positive", 3)]  # OK
)

ConditionalStepDescription(
    branches=[ConditionalBranch(condition="contains:positive", next_step=3)]  # OK
)

# ChainBuilder also accepts ConditionalBranch objects
ChainBuilder().add_conditional_step(
    branches=[ConditionalBranch(condition="contains:positive", next_step=3)]  # OK
)
```

**Files Modified:** `src/mmar_carl/chain.py`

---

#### BUG-004 (MEDIUM): String Literal Handling in Input Mapping

**Problem:** String literals in input mapping were not properly handled. Quoted strings like `'"value"'` returned `None`.

**Fix:** Added string literal detection to `resolve_context_reference()`

**Before:**

```python
input_mapping={"summary": '"Revenue Analysis"'}
# Result: summary parameter receives None
```

**After:**

```python
input_mapping={"summary": '"Revenue Analysis"'}
# Result: summary parameter receives "Revenue Analysis"
```

**Files Modified:** `src/mmar_carl/step_executors.py`

---

#### BUG-005 (MEDIUM): Type Coercion in Input Mapping

**Problem:** Input mapping didn't perform type coercion from strings to expected parameter types.

**Fix:** Added automatic type coercion based on `ToolParameter.type` field

**Before:**

```python
def calculate_sum(values: list[float]) -> dict:
    return {"sum": sum(values)}

input_mapping={"values": "[100, 200, 300]"}  # String, not list
# Error: unsupported operand type(s) for +: 'int' and 'str'
```

**After:**

```python
input_mapping={"values": "[100, 200, 300]"}
# Result: values parameter receives [100.0, 200.0, 300.0] (list of floats)
```

**Supported Types:** int, float, bool, list, dict

**Files Modified:** `src/mmar_carl/step_executors.py`

---

#### BUG-006 (LOW): Outer Context String Parsing

**Problem:** When `$outer_context` contained structured data as a JSON string, it wasn't parsed before being passed to tools.

**Fix:** Enhanced `$outer_context` handling to parse JSON strings automatically

**Before:**

```python
context = ReasoningContext(
    outer_context="[100, 200, 300]",  # String representation of list
    ...
)
input_mapping={"values": "$outer_context"}
# Error: unsupported operand type(s) for +: 'int' and 'str'
```

**After:**

```python
context = ReasoningContext(
    outer_context="[100, 200, 300]",
    ...
)
input_mapping={"values": "$outer_context"}
# Result: values parameter receives [100, 200, 300] (parsed list)
```

**Files Modified:** `src/mmar_carl/step_executors.py`

---

### 📝 Documentation Improvements

- Updated all examples to remove legacy endpoint/entrypoint references
- Improved import documentation in `__init__.py`
- Enhanced module structure documentation
- Clarified LLM client detection and usage patterns
- Updated README.md to reference separate release notes file

---

### 🗑️ Deprecated

The following features have been completely removed:

- `mmar-llm-mapi` integration code
- `create_llm_client()` function (use `create_openai_client()` instead)
- Legacy endpoint/entrypoint configuration options
- `examples/legacy_mmar_llm_example.py` file

---

### 🔒 Internal Changes

- Refactored LLM client detection logic for better separation of concerns
- Improved error handling in step executors
- Enhanced type safety throughout codebase
- Optimized import structure for better modularity
- Updated `__version__` to "0.2.0"

---

### Migration Guide from v0.1.0

If you're upgrading from v0.1.0, here's what you need to know:

**1. Update your imports (if applicable):**

```python
# If you were importing LLMClientBase directly
from mmar_carl.models.llm_client_base import LLMClientBase
```

**2. Use create_openai_client() for OpenAI-compatible APIs:**

```python
from mmar_carl import create_openai_client

client = create_openai_client(
    api_key="sk-or-v1-...",
    model="anthropic/claude-3.5-sonnet"
)
```

**3. No code changes required for bug fixes!** All bug fixes are backward compatible and will automatically improve your chains.

---

## Version 0.1.0

### 🚨 Deprecation of StepDescription

The unified `StepDescription` class is now **deprecated**. Use typed step classes instead:

```python
# ❌ Deprecated (will show warning)
StepDescription(
    number=1,
    title="Analysis",
    aim="Analyze data"
)

# ✅ Recommended
LLMStepDescription(
    number=1,
    title="Analysis",
    aim="Analyze data"
)
```

### 📊 Structured Logging

New logging system with configurable levels:

```python
import logging
from mmar_carl import set_log_level, get_logger

# Configure logging level
set_log_level(logging.DEBUG)  # or INFO, WARNING, ERROR

# Use logger directly
logger = get_logger()
logger.info("Custom log message")

# Automatic logging during execution:
# 2026-03-10 10:39:09 [INFO] mmar_carl: Starting chain 'My Chain' with 4 steps (max_workers=2)
# 2026-03-10 10:39:18 [INFO] mmar_carl: Chain execution completed successfully in 8.97s (4/4 steps)
# 2026-03-10 10:40:04 [WARNING] mmar_carl: Step 1 failed in 3.16s
```

### 🔍 Error Traceback Preservation

`StepExecutionResult` now includes `error_traceback` field for debugging:

```python
result = chain.execute(context)
if not result.success:
    for step in result.get_failed_steps():
        print(f"Step {step.step_number} failed: {step.error_message}")
        if step.error_traceback:
            print(f"Traceback:\n{step.error_traceback}")
```

### 🛡️ Memory Leak Fix

Context snapshots are now properly cleaned up in parallel execution, preventing event loop issues in long-running applications.

### Chain-Level Timeout

Set a maximum execution time for the entire chain:

```python
chain = ReasoningChain(
    steps=steps,
    timeout=300.0,  # 5 minutes max
)

# Or with ChainBuilder
chain = (ChainBuilder()
    .add_step(...)
    .with_timeout(300.0)
    .build())
```

### Per-Step Retry Configuration

Override retry attempts for specific steps (e.g., more retries for flaky API calls):

```python
LLMStepDescription(
    number=1,
    title="API Call",
    aim="Call external API",
    retry_max=5,  # More retries for this step
)
```

### Resource Cleanup

Properly close LLM clients when done:

```python
context = ReasoningContext(...)
try:
    result = chain.execute(context)
finally:
    await context.close()  # Release HTTP connections
```

### OpenAI-Compatible API Support

Use CARL with OpenRouter, Azure OpenAI, local LLMs (Ollama, vLLM, LM Studio), and any OpenAI-compatible API:

```python
from mmar_carl import create_openai_client, ReasoningContext, Language

# OpenRouter
client = create_openai_client(
    api_key="sk-or-v1-...",
    model="anthropic/claude-3.5-sonnet",
    extra_headers={"HTTP-Referer": "https://your-site.com"}
)

# Local LLM (Ollama)
client = create_openai_client(
    api_key="not-needed",
    model="llama3",
    base_url="http://localhost:11434/v1"
)

context = ReasoningContext(
    outer_context=data,
    api=client,
    language=Language.ENGLISH
)
```

### Per-Step LLM Configuration

Use different models for different reasoning steps:

```python
from mmar_carl import LLMStepDescription, LLMStepConfig

steps = [
    # Fast model for simple tasks
    LLMStepDescription(
        number=1,
        title="Quick Analysis",
        aim="Fast initial analysis",
        # Uses default model from context
    ),
    # Powerful model for complex reasoning
    LLMStepDescription(
        number=2,
        title="Deep Analysis",
        aim="Complex reasoning task",
        llm_config=LLMStepConfig(
            model="anthropic/claude-3.5-sonnet",
            temperature=0.3
        ),
        dependencies=[1]
    ),
]
```

### LLM Execution Modes (Production)

Each LLM step can choose one of two execution strategies via `LLMStepConfig.execution_mode`:

- `ExecutionMode.FAST`: single direct generation (default)
- `ExecutionMode.SELF_CRITIC`: generation with evaluator chain (all evaluators must approve)

```python
from mmar_carl import ExecutionMode, LLMStepConfig, LLMStepDescription
```

#### FAST mode

FAST is strict one-pass generation with no evaluator calls:

```python
LLMStepDescription(
    number=1,
    title="Quick Analysis",
    aim="Generate first-pass answer",
    llm_config=LLMStepConfig(execution_mode=ExecutionMode.FAST),
)
```

#### SELF_CRITIC mode (default LLM evaluator)

SELF_CRITIC runs evaluator(s) after generation. If any evaluator disapproves, the step is regenerated
until all approve or `self_critic_max_revisions` is reached.

Built-in `llm` evaluator behavior:

1. Uses the same step LLM client.
2. Produces strict JSON: `{"verdict":"APPROVE|DISAPPROVE","review":"..."}`.
3. Treats malformed/empty review as `DISAPPROVE`.

```python
LLMStepDescription(
    number=2,
    title="Quality-Controlled Answer",
    aim="Produce higher-quality answer",
    llm_config=LLMStepConfig(
        execution_mode=ExecutionMode.SELF_CRITIC,
        self_critic_evaluators=["llm"],  # built-in evaluator
        self_critic_max_revisions=1,
        self_critic_instruction="Prioritize factual consistency and concrete mitigation advice.",
    ),
)
```

#### Custom self-critic evaluator strategies

You can register custom evaluators by implementing `SelfCriticEvaluatorBase`.

```python
from mmar_carl import SelfCriticEvaluatorBase, SelfCriticDecision

class KeywordGuard(SelfCriticEvaluatorBase):
    async def evaluate(self, step, candidate, base_prompt, context, llm_client, retries):
        if "mitigation" in candidate.lower():
            return SelfCriticDecision("APPROVE", "Keyword present.", {"llm_calls": 0})
        return SelfCriticDecision("DISAPPROVE", "Missing mitigation keyword.", {"llm_calls": 0})

context.register_self_critic_evaluator("keyword_guard", KeywordGuard())
```

Use evaluator chains in a step (all-must-approve policy):

```python
LLMStepDescription(
    number=3,
    title="Final Review",
    aim="Enforce stricter quality rules",
    llm_config=LLMStepConfig(
        execution_mode=ExecutionMode.SELF_CRITIC,
        self_critic_evaluators=["llm", "keyword_guard"],
        self_critic_max_revisions=2,
        self_critic_disapprove_feedback={
            "keyword_guard": "You must explicitly include at least one mitigation item.",
        },
    ),
)
```

#### Execution diagnostics in the pipeline example

`examples/execution_modes_pipeline_example.py` now prints a detailed execution report similar to the
basic example style:

- Chain overview (steps, dependencies, execution plan, per-step mode config)
- Per-step execution results (status, timing, output preview/error)
- Per-step mode diagnostics (`llm_calls`, rounds, evaluator policy, evaluator verdicts by round)
- Final output section

### Chain-Level RE-PLAN Policy

RE-PLAN is a chain-level control policy, not an LLM execution mode.

- `ExecutionMode` defines **how a step runs** (`FAST`, `SELF_CRITIC`).
- `ReplanPolicy` defines **how chain control flow reacts** to intermediate outcomes.

This keeps concerns separate and allows combinations like:

- `FAST` + RE-PLAN
- `SELF_CRITIC` + RE-PLAN
- mixed execution modes + RE-PLAN

#### Minimal RE-PLAN configuration

```python
from mmar_carl import (
    ReplanAction,
    ReplanPolicy,
    RuleBasedReplanCheckerConfig,
    ReasoningChain,
)

policy = ReplanPolicy(
    enabled=True,
    checkers=[
        RuleBasedReplanCheckerConfig(
            name="retry_on_bad_output",
            result_substrings=["needs_retry"],
            action_on_match=ReplanAction.RETRY_CURRENT_STEP,
            feedback_on_match=["Use clearer assumptions and stronger validation."],
        )
    ],
)

chain = ReasoningChain(steps=steps, replan_policy=policy)
```

#### Checker types

- `RuleBasedReplanCheckerConfig`: deterministic/rule-based checks.
- `LLMReplanCheckerConfig`: LLM-based checker with strict structured verdict parsing into `ReplanVerdict`.
- `RegisteredReplanCheckerConfig`: references a custom checker registered in `ReasoningContext`.

#### Aggregation strategies

- `ANY`
- `ALL`
- `K_OF_N`
- `MANDATORY_PLUS_K_OF_REST`

Configure via `ReplanAggregationConfig`.

#### Checkpoints and rollback

Mark any step as a checkpoint with additive step fields:

- `checkpoint=True`
- `checkpoint_name="my_checkpoint"` (optional)

Rollback targets support:

- chain start
- current step
- nearest previous checkpoint
- named checkpoint
- specific step number

#### Triggers, feedback, and safeguards

Configure trigger points with `ReplanTriggerConfig`:

- after step
- after failed step
- checkpoint-only
- selected step numbers/types

Configure loop prevention with `ReplanBudgetConfig`:

- `max_replans_per_chain`
- `max_replans_per_step`
- `max_visits_per_checkpoint`
- repeated same-target protection

RE-PLAN feedback/hints are injected into retried LLM prompts and all replan evaluations/actions are recorded in `ReasoningResult.replan_events` plus summary metadata.

### 📊 Evaluation Metrics

Attach numeric evaluation metrics to individual steps or to the whole chain by subclassing `MetricBase`.

```python
from mmar_carl import MetricBase, LLMStepDescription, ReasoningChain

class WordCountMetric(MetricBase):
    @property
    def name(self) -> str:
        return "word_count"

    async def compute_async(self, text: str) -> float:
        return float(len(text.split()))
```

Attach to a **step** — scored after each step's output:

```python
step = LLMStepDescription(
    number=1,
    title="Analysis",
    aim="Analyse the data",
    metrics=[WordCountMetric()],
)

result = chain.execute(context)
print(result.step_results[0].metrics)   # {'word_count': 47.0}
```

Attach to the **chain** — scored on the final output:

```python
chain = ReasoningChain(steps=steps, metrics=[WordCountMetric()])
result = chain.execute(context)
print(result.metrics)                   # {'word_count': 82.0}
```

Scores are stored in `StepExecutionResult.metrics` and `ReasoningResult.metrics`, included in `to_dict()`, and sent to LangFuse (step scores appear in span output; chain scores are posted as LangFuse `score` objects).

**Key properties:**

- Any number of metrics on the same step or chain
- Metrics run only on **successful** outputs; failed steps are skipped
- A metric that raises an exception is silently skipped — it never aborts execution
- Implement `compute_async(text) -> float`; a sync `compute()` wrapper is provided for convenience

**Built-in examples** (`examples/metrics_example.py` — no API key needed):

| Metric                            | What it measures                            |
| --------------------------------- | ------------------------------------------- |
| `WordCountMetric`                 | Number of words in the output               |
| `SentenceLengthMetric`            | Average words per sentence                  |
| `KeywordCoverageMetric(keywords)` | Fraction of required keywords present (0–1) |
| `MockLLMJudgeMetric`              | Simulated LLM-as-a-judge score (0–10)       |

Run the example:

```bash
python examples/metrics_example.py
# or
make example-metrics
```

#### Metrics in Reflection

When calling `chain.reflect()`, metric scores are automatically fed into the reflection prompt so the LLM can reference concrete quality signals:

```python
from mmar_carl import ReflectionOptions

result = chain.execute(context)

reflection = chain.reflect(
    task_description="Analyse quarterly revenue",
    options=ReflectionOptions(
        include_metric_scores=True,      # True by default
        extra_feedback={                 # optional user context
            "audience": "C-level executives",
            "priority": "conciseness",
        },
    ),
)
```

`extra_feedback` accepts a `dict` (labelled entries) or a plain `str`. Set `include_metric_scores=False` to exclude scores from the prompt.

```bash
python examples/reflection_metrics_example.py
# or
make example-reflection-metrics
```

### Typed Step Description Classes

New inheritance-based step classes for better type safety:

- `LLMStepDescription` - LLM reasoning steps
- `ToolStepDescription` - External tool/function execution
- `MCPStepDescription` - MCP protocol calls
- `MemoryStepDescription` - Memory operations
- `TransformStepDescription` - Data transformations
- `ConditionalStepDescription` - Conditional branching

### Multi-Step Type Support

Execute different types of operations in your reasoning chains:

- **LLM**: Standard LLM reasoning (default)
- **TOOL**: Execute registered Python functions
- **MCP**: Call MCP protocol servers
- **MEMORY**: Read/write/append/delete/list operations
- **TRANSFORM**: Data transformations without LLM
- **CONDITIONAL**: Branch execution based on conditions

### Memory and Tool Registry

- Built-in memory storage with namespace isolation
- Tool registry for registering external functions
- Input mapping syntax: `$history[-1]`, `$memory.namespace.key`, `$metadata.key`

### JSON Serialization

- `chain.save("file.json")` / `ReasoningChain.load("file.json")`
- `chain.to_dict()` / `ReasoningChain.from_dict(data)`
- `chain.to_json()` / `ReasoningChain.from_json(json_str)`
