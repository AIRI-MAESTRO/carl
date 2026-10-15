# Maestro runtime capabilities: focused living specification

Status: **LIVING SPEC — A01-A05, A02 and A07 verified; A06 parked**
Implementation status: **A01 replacement is PR #18; A03 replacement is stacked PR #19; A04 is stacked PR #20; A05 Map is stacked PR #22; A02 CodeStep is stacked PR #23; A07 chain-as-tool is stacked PR #21**
Last updated: **2026-08-27**

## 1. Purpose

This document is the design gate for the small set of runtime capabilities that
remain in the current discussion. It separates verified observations, proposed
contracts, accepted decisions and later implementation evidence.

The current active scope is deliberately narrow:

1. a basic tool-using `AgentStep`;
2. a self-contained asynchronous `WaitStep`;
3. typed human-in-the-loop interaction;
4. dynamic `Map` fan-out;
5. parked orchestration research;
6. chain-as-tool composition;
7. execution of LLM-generated code through `CodeStep`.

Relative and absolute timers are variants of `WaitStep`, not separate public steps.
Global Terminate remains recorded only as a parked backlog decision. The reopened
capabilities are discussed and implemented one at a time in the order above. A05's
accepted contract is recorded in Section 11; A02 and A07 remain open and A06 is parked.

## 2. Decision workflow

Each active capability moves through these states:

1. `OPEN` — audit and requirements are incomplete.
2. `PROPOSED` — one complete contract is ready for review.
3. `ACCEPTED` — the contract has explicit project-owner approval.
4. `IMPLEMENTING` — a scoped implementation is in progress.
5. `VERIFIED` — the implementation passes its acceptance evidence.
6. `PARKED`, `REJECTED`, or `SUBSUMED` — deliberately deferred, excluded, or
   represented by another accepted capability.

Rules:

- Existing code is evidence, not an accepted design.
- A plausible recommendation does not change a status to `ACCEPTED`.
- Every accepted decision receives a dated decision-log entry.
- Runtime code is changed only after the corresponding contract is accepted.
- A capability must survive chain-format save/load and fail preflight before side
  effects when the host cannot support it.

## 3. Current scope and status

| ID | Capability | Status | Current position |
| --- | --- | --- | --- |
| A01 | Basic `AgentStep` / agent loop | `VERIFIED` | Bounded tool-using step, replacement published in PR #18; no orchestration or formal verified exit. |
| A02 | `CodeStep` | `VERIFIED` | Python-only strict-sandbox implementation published in stacked PR #23. |
| A03 | Self-contained `WaitStep` | `VERIFIED` | In-process asynchronous `After`, `At`, `Event`, and `AnyOf`; no durable resume. |
| A04 | `HumanInputStep` / HITL interaction | `VERIFIED` | Typed process-local text request with explicit non-success outcomes; AIITL excluded. |
| A05 | `MapStep` | `VERIFIED` | Bounded ordered `collect_all` fan-out of one registered tool over a JSON array. |
| A06 | Orchestration | `PARKED` | Explicitly skipped; no public router/planner/orchestrator abstraction accepted. |
| A07 | Chain as a tool | `VERIFIED` | Embedded typed snapshots execute as isolated AgentStep/ToolStep tools. |
| B01 | Separate fixed Delay type | `SUBSUMED` | `WaitStep` provides relative and absolute timer conditions. |
| B02 | Global Terminate | `PARKED` | No current use case justifies a global scheduler-control step. |

## 4. Explicitly out of scope

The topics below remain excluded or are not independently accepted. This does not
claim that an idea is universally useless, and it does not modify work in its own
separate branch. Map and chain-as-tool remain active in Section 3; orchestration is
parked rather than rejected.

| Removed topic | Disposition |
| --- | --- |
| Raw `HTTP request` step | Removed. Network access should be provided by separately registered tools/connectors if needed. |
| `Command` / `BASH` | Removed from this document. It has separate branches, audits and implementations. |
| AIITL | Removed. A04 covers human actors only. |
| Route / Supervisor | Not independently accepted; A06 is parked. |
| Planner / Orchestrator | Not independently accepted; A06 is parked. |
| Formal verified exit | Removed. The basic agent uses ordinary protocol completion only. |
| ReAct trace to chain / workshop article | Removed. No research or distillation package is planned here. |
| PR/versioning graph | Removed. Release organization is handled only after a separate request. |

The remaining exclusions constrain A01. Its verified v1 does not gain map,
orchestration, chain composition, AIITL, or formal verified completion implicitly;
later capabilities require their own contracts and PRs.

## 5. Candidate invariants for the remaining capabilities

Status: `OPEN`; these are recommendations to review, not accepted decisions.

- Host policy owns authority. Chain JSON may request or narrow a capability but may
  not grant itself execution permission, credentials, network, filesystem access or
  approval.
- `completed`, `incomplete`, `cancelled`, `budget_exhausted` and `failed` are
  distinct outcomes. A budget stop is never success.
- Timeout and cancellation must stop owned work, not merely stop awaiting it.
- Tool and code outputs are typed, bounded and explicitly marked as trusted or
  untrusted before they enter a model conversation.
- A capability used by a chain is declared in chain requirements and checked during
  preflight.
- Chain JSON round-trip may not replace required executable state with a callable,
  placeholder or ambient process object.
- Every model, tool and code call records identity, arguments/result hashes, usage,
  policy decision, redaction markers and executed-versus-replayed status.

## 6. Discussion order

The current order is:

1. **A01 Basic AgentStep — verified.** The minimal bounded tool loop is complete.
2. **A03 WaitStep — verified.** Self-contained asynchronous timers and events are implemented.
3. **A04 HumanInputStep — verified.** Typed human-only text interaction is complete.
4. **A05 Map — verified.** Bounded ordered tool fan-out and typed aggregation are implemented.
5. **A06 Orchestration — parked.** Reopen only for a concrete routing or planning use case.
6. **A07 Chain as a tool — verified.** Embedded typed snapshots execute as isolated tools.
7. **A02 CodeStep — verified.** Exact runtime-generated Python execution is published in PR #23 with a strict host-authorized sandbox contract.

Each remaining capability receives its own design decision, branch, compatibility
contract and PR. Reopening the sequence does not add those powers to A01 v1.

## 7. A01 — Basic AgentStep

Status: `VERIFIED`
Contract: `A01-contract-v1`

### 7.1 Why this capability exists

An ordinary LLM step makes one model call. A normal tool step calls one tool chosen
when the chain is authored. Neither can express the bounded dynamic loop:

1. the model observes the task and visible tool schemas;
2. it chooses a tool and arguments at runtime;
3. the runtime executes the call under host policy;
4. the model observes the result and decides whether another call is needed;
5. the model explicitly finishes with a final answer.

That runtime-sized number and order of calls is the distinct reason for an agent
step. It should be one component inside an otherwise explicit chain, not a second
general orchestrator.

### 7.2 Proposed public name

Decision `A01-001`: the public chain type is `agent` and the Python/API object is
`AgentStep`. It is a bounded ReAct loop: the step receives a goal and an explicitly
available set of tool capabilities, then iterates reason -> act -> observe until it
returns a result or reaches a runtime limit.

“Agent loop” describes the internal state machine; it does not create a second public
`AgentLoopStep` type.

Decision `A01-002`: v1 requires an explicit non-empty tool allowlist, accepts exactly
one model-directed call per iteration, and completes only through the runtime-provided
`finish` meta-tool. Zero or multiple calls are protocol errors returned as observations;
they consume an iteration and execute no requested tool.

### 7.3 Non-goals for v1

- Calling another chain as a tool or spawning subagents.
- Dynamic Map, Route, Supervisor, Planner or graph mutation.
- Human or AI interaction during the loop.
- Arbitrary inline code execution; A02 remains a separate step/capability.
- Global chain termination. `finish` closes only this AgentStep.
- Formal or proof-carrying verification of real-world task completion.
- Long-term memory, autonomous background operation or unbounded continuation.
- Automatic tool discovery beyond the explicitly requested and host-approved set.

### 7.4 Observed implementation state

Pull request #4 adds a small `AgentLoopStep` with these semantics:

- `tools=None` exposes every registered tool;
- a response with no tool calls terminates the loop;
- all tool calls emitted in one model turn run concurrently;
- tool exceptions become strings and are fed back to the model;
- `max_iterations` defaults to eight;
- exhausting the limit is successful by default unless
  `fail_on_max_iterations=True`;
- there is no mandatory wall-time, tool-call, token or cost budget;
- the transcript is not a durable replay contract.

A richer experimental `AgentStep` already demonstrates useful mechanisms: explicit
`finish`, tool filters/tags, host hooks, input slots, budgets, cancellation,
compaction, untrusted-output wrapping, typed output validation, events and
transcripts. It is not merge-ready evidence: invalid final schemas may eventually be
accepted and max-iteration termination may still be reported as success.

### 7.5 Accepted minimal public contract

Public JSON shape:

```json
{
  "number": 3,
  "title": "Research the requested topic",
  "step_type": "agent",
  "dependencies": [2],
  "step_config": {
    "goal": "Use the available evidence tools and answer the question.",
    "input_mapping": {
      "question": "$memory.input.question"
    },
    "tools": ["search_documents", "read_document"],
    "max_iterations": 8,
    "max_tool_calls": 12,
    "timeout_seconds": 120,
    "max_tokens": 20000,
    "model_timeout_seconds": 60,
    "tool_timeout_seconds": 30,
    "max_tool_result_chars": 16000,
    "max_transcript_chars": 100000,
    "output_schema": {
      "type": "object",
      "required": ["answer"],
      "properties": {
        "answer": {"type": "string"}
      }
    },
    "output_key": "research_result"
  }
}
```

Public fields:

| Field | Public type | Runtime rule |
| --- | --- | --- |
| `goal` | non-empty string | Static instruction; resolved inputs are supplied separately, not by unrestricted string interpolation. |
| `input_mapping` | map of name to context reference | Every resolved value is size-bounded and serialized as data. |
| `tools` | non-empty list of capability ids | Required in v1; there is no “all registered tools” default. |
| `system_prompt` | optional string | Appended after the runtime protocol and host context instructions. |
| `max_iterations` | positive integer | Finite model-iteration bound. |
| `max_tool_calls` | non-negative integer | Aggregate ordinary-tool-call bound; `finish` does not count. |
| `timeout_seconds` | positive finite number | Whole-step wall-clock bound. |
| `model_timeout_seconds` | positive finite number | Per-model-call wall-clock bound. |
| `tool_timeout_seconds` | positive finite number | Per-tool-call wall-clock bound. |
| `max_tokens` | optional positive integer | Enforced when the selected client reports aggregate provider usage. |
| `max_tool_result_chars` | integer at least 256 | Bounds each serialized observation passed to the model and trace. |
| `max_transcript_chars` | integer at least 1024 | Bounds each model request and the serialized diagnostic trace. |
| `output_schema` | optional JSON Schema subset | Validates the payload supplied to `finish`. |
| `output_key` | optional memory binding | Stores the typed final payload only on `completed`. |
| `output_namespace` | non-empty string | Namespace used with `output_key`; defaults to `agent`. |

The chain configuration selects tool ids but never stores Python callables, secrets,
approval callbacks or permission grants.

### 7.6 Tool capability contract

The model sees only the explicit tool ids requested by the AgentStep and present
in the host registry when preparation starts. A missing id rejects preparation
before the first model call.

For the accepted v1, every visible tool needs:

- a unique registry name;
- an inspectable callable using named parameters only;
- a JSON input schema derived from its Python signature (annotated parameters
  receive strict runtime type validation; unannotated parameters remain
  JSON-unconstrained in v1);
- a JSON-serializable output;
- timeout/cancellation behaviour and the AgentStep output-size limit.

Unknown/unavailable tools fail preflight and executor preparation. A model cannot
make them available by naming them in a response. Versioned revisions, declared
output schemas, side-effect classes, approval hooks and per-tool redaction policies
are explicitly deferred registry hardening; chain JSON cannot grant any of those
authorities.

### 7.7 Loop state machine

Accepted internal state machine:

`PREPARE -> THINK -> VALIDATE_CALLS -> ACT -> OBSERVE -> THINK -> FINISH`

1. **PREPARE:** resolve bounded inputs; intersect tool visibility with host policy;
   check budgets and schemas.
2. **THINK:** make one model call with the current bounded transcript.
3. **VALIDATE_CALLS:** require valid tool ids and schema-valid arguments. Invalid
   calls produce a typed observation and consume an iteration; repeated invalid
   calls hit the ordinary budget.
4. **ACT:** execute allowed calls under per-tool policy, timeout and cancellation.
5. **OBSERVE:** append typed, size-bounded, trust-labelled results to the transcript.
6. **FINISH:** accept only an explicit runtime-provided `finish` meta-tool with a
   schema-valid final payload.

A tool-free text response does not silently complete v1. The runtime returns one
corrective observation such as “call a tool or call finish”; continued refusal is
bounded by the same iteration budget.

### 7.8 Meaning of finish

`finish` is component-local protocol control, not B02 Terminate and not formal
verified exit. It means only:

> The model proposes that this AgentStep has enough information and supplies a final
> payload that satisfies the declared output schema.

The resulting `completed` status confirms protocol completion, not that every
real-world claim in the answer is true. The runtime must not call this
`completed_verified`.

Invalid finish payloads are returned to the model as typed validation observations.
After the remaining iteration budget is exhausted, the step is `incomplete`; the
validation failure is retained only as diagnostic evidence, never as successful
output.

### 7.9 Terminal outcomes

| Outcome | Meaning |
| --- | --- |
| `completed` | Explicit finish received and the final payload satisfies the configured schema. |
| `incomplete` | The bounded loop ended without a valid finish. |
| `budget_exhausted` | A model/tool/token/transcript/wall-time limit prevented another action. |
| `cancelled` | The host cancelled the step and owned in-flight work was stopped or marked with an explicit enforcement gap. |
| `failed` | Permanent runtime/provider/tool infrastructure failure made continuation impossible. |

`max_iterations`, `max_tool_calls`, invalid finishes and repeated invalid calls can
never produce `completed`.

### 7.10 Concurrency semantics

Exactly one call is accepted per model response. Therefore v1 has no intra-iteration
tool concurrency or call ordering ambiguity. If a response proposes multiple calls,
none of them execute; the model receives a protocol-error observation and may retry
within the same iteration budget.

### 7.11 Budgets, timeout and cancellation

The accepted v1 serializes and enforces:

- model iterations;
- total tool calls;
- whole-step wall time;
- per-model-call and per-tool-call wall time;
- serialized request, diagnostic-trace and tool-result characters;
- provider tokens when the client reports usage.

Budget use is monotonic and includes failed/invalid calls that consumed resources.
A timeout of a synchronous in-process tool cannot stop its Python thread; the result
records this as an enforcement gap. Tools that require a hard kill guarantee must be
registered through a killable external-process backend. Parent/run budget
composition and cost metering are deferred.

### 7.12 Permissions, approvals and untrusted observations

- External/tool output is framed as untrusted data and size-bounded before being
  inserted into the conversation. Framing is defense-in-depth, not a guarantee
  against prompt injection.
- Tool errors use typed observations (`protocol_error`, `timeout`, `permanent`)
  rather than pretending an exception string is a successful result.
- Permissions, approvals, secret resolution and redaction remain host/tool-registry
  responsibilities. AgentStep v1 does not add an approval callback or place secrets
  in chain JSON.

### 7.13 Transcript, output and replay

The semantic output is the typed payload passed to valid `finish`. Presentation
history may contain a summary but is not the data contract.

The v1 result records a bounded diagnostic transcript containing:

- each model response, proposed-call count and reported usage;
- each protocol error, tool observation and invalid finish;
- iteration/tool-call counts, enforcement gaps, terminal outcome and stop reason.

The existing model-client cassette wrappers now preserve tool-call usage and can
replay the model side of this loop. A durable per-tool executed-versus-replayed
provenance envelope, hashes and policy revisions remain deferred and must not be
claimed by v1.

### 7.14 Serialization and compatibility

- The public chain JSON stores only serializable configuration and capability ids.
- The stable public type value is `agent`; typed and legacy save/load paths preserve
  the complete AgentStep configuration.
- Existing draft `agent_loop` JSON is not silently aliased because its exit semantics
  differ.
- A versioned capability-revision declaration is deferred until the broader chain
  invocation/version contract is reopened.

### 7.15 Exact changes required in the current prototypes

1. Keep one canonical `AgentStep`; do not introduce the PR #4 prototype's
   incompatible `AgentLoopStep` public type.
2. Replace `tools=None -> all tools` with an explicit required allowlist in v1.
3. Replace tool-free-answer termination with explicit `finish`.
4. Replace success-on-iteration-exhaustion with typed `incomplete` or
   `budget_exhausted`.
5. Never accept an output-schema-invalid finish after a retry count.
6. Add mandatory finite iteration, tool-call and wall-time limits.
7. Reject multiple proposed calls and execute exactly one allowed call per iteration.
8. Validate tool ids/arguments through the host registry before every call.
9. Replace unbounded stringified exceptions with typed, size-bounded observations.
10. Return a bounded diagnostic transcript and preserve usage in model cassettes.
11. Poll cancellation before model calls, before each tool call and while awaiting
    model/tool work; record the known synchronous-thread enforcement gap.
12. Keep chain composition, HITL, graph planning and formal exit out of A01 v1.

Implementation areas are `models/enums.py`,
`models/config.py`, `models/steps.py`, `step_executors.py`, model-client tool-call
protocols, chain requirements/serialization and focused tests. The current `main`
branch does not contain the evo-metadata catalog subsystem, so these capability PRs
do not recreate that unrelated subsystem.

### 7.16 Acceptance criteria

1. JSON save/load/execute round-trip for the complete AgentStep configuration.
2. Preflight reports missing tool ids, and executor preparation rejects them before
   any model or tool call.
3. A deterministic fake model/tool scenario covers multi-turn act/observe/finish.
4. Tool-free text, invalid calls, invalid finish, max iterations, max tool calls,
   timeout and cancellation each produce the specified non-success outcome.
5. Multiple proposed calls execute nothing; ordinary calls execute one at a time.
6. Tool arguments are checked against callable binding and declared annotations;
   final output is checked against the configured schema subset.
7. Tool observations and transcripts enforce configured character limits and
   untrusted-data framing.
8. Reported token usage and tool/model counters are monotonic.
9. Record/replay preserves tool calls and their reported provider usage.
10. A clean built artifact installs with all direct dependencies and runs focused and
    supported hermetic suites without paid provider calls.

### 7.17 Decisions deferred beyond the accepted v1 core

The next discussion should resolve these in order:

1. Versioned tool capability metadata beyond the current callable registry.
2. Host approval hooks and side-effect classes.
3. Durable record/replay envelopes beyond the existing client cassette support.
4. Cost budgets once a trusted price source and currency contract exist.

### 7.18 Decision log

| Date | ID | Decision | Rationale |
| --- | --- | --- | --- |
| 2026-08-24 | A01-001 | Use one public `AgentStep` as a bounded ReAct loop that receives a goal and allowed tool capabilities and attempts to produce a result. | This is the minimal runtime-sized think/act/observe behaviour that cannot be represented by a fixed sequence of one-shot LLM and tool steps. |
| 2026-08-24 | A01-002 | Require a non-empty explicit tool allowlist, exactly one call per model iteration, and explicit schema-checked `finish`. | This prevents ambient authority, parallel side-effect ambiguity and accidental completion from ordinary model text. |

## 8. A02 — CodeStep

Status: `VERIFIED`; implementation published in stacked PR #23.
Contract: `A02-contract-v1`

### 8.1 Purpose and boundary

`CodeStep` executes a Python source string produced earlier in the same run. The
step does not call an LLM and does not carry authored inline code. Its `source`
field is one explicit context reference, for example `$memory.generated.code`,
whose resolved value must be a non-empty string.

This capability is distinct from Command/BASH: the executable and argv are not
model-selected, no shell is inserted, and the only public program contract is one
Python function:

```python
def run(inputs: dict) -> JSONValue:
    ...
```

The generated source is untrusted. AST parsing checks the exact entry-point shape
and improves diagnostics, but is not treated as an isolation boundary.

### 8.2 Exact source contract

- Python is the only v1 language.
- `source` must be a named context reference beginning with `$`; inline source is
  rejected.
- CARL executes the resolved string exactly. It does not strip Markdown fences,
  extract a code block, repair syntax, regenerate code or retry with an LLM.
- The source must parse and define exactly one synchronous top-level
  `run(inputs)` function with one positional parameter and no variadic parameters.
- CARL records the UTF-8 source byte length and SHA-256 before runtime preparation.
- Source text is not copied into the outcome or history.

### 8.3 Typed input and output

- `input_mapping` maps names to explicit context references. The resolved values
  form the single object passed as `inputs`.
- `input_schema` is required and must describe an object. CARL validates the input
  before starting a runtime.
- `output_schema` is required. The returned value must be finite,
  JSON-serializable and schema-valid.
- Schema validation uses a documented strict subset: `type`, `properties`,
  `required`, `items`, `enum` and `additionalProperties`. Unknown schema keywords
  are rejected rather than silently ignored.
- Optional successful output memory is written only after complete validation.
- Generated stdout and stderr are bounded diagnostics. The validated function
  return value is the step result and the only value appended to chain history.

### 8.4 Host-owned runtime authority

The serialized step requests a `runtime_profile` id. It cannot name an executable,
container image, environment variable or runtime preparation flag directly. A
runtime-only Pydantic `CodeExecutionPolicy` maps that id to a host-owned
`CodeRuntimeProfile` containing the runtime/revision, fixed interpreter argv,
pinned runtime configuration, resource ceilings and byte/time limits.

Missing policy/profile, unknown runtime or an unavailable interpreter is a typed
non-success outcome before generated code executes. Policy objects are excluded
from context serialization and copied explicitly into parallel/replay contexts.

### 8.5 Strict isolation profile

V1 has no unsafe-local or best-effort mode. The selected backend must attest all of:

- non-host isolation;
- enforced wall-clock timeout and bounded stdout/stderr collection;
- enforced CPU, memory and process limits;
- enforced no-network policy;
- controlled workspace inputs and bounded output handling;
- a read-only generated-code workspace output mount so code cannot grow a host bind
  mount.

The current Docker backend is the reference implementation after adding the
read-only output-mount mode. Current Local, Firejail and E2B capability reports do
not satisfy the complete v1 profile. Custom runtimes may qualify only when their
capability report and prepared handle attest the same controls.

The runtime receives only CARL-owned source, input and runner files through a
read-only input area. Network is always `none`; no host environment, credentials,
dependency installation, artifact input/output or writable host workspace is
exposed. Temporary in-runtime storage remains bounded by the runtime profile. The
semantic output byte ceiling gets 4096 additional bounded stdout bytes for the
CARL result envelope; generated diagnostics can consume that headroom and cause a
deterministic `invalid_output`, never an unbounded capture.

### 8.6 Lifecycle and outcomes

CARL races owned execution against host cancellation. Timeout or cancellation must
stop the runtime-owned process/container and bounded cleanup must complete before the
step returns. Only `completed` is successful. `CodeExecutionOutcome.status` is one
of `completed`, `invalid_source`, `invalid_input`, `denied`,
`runtime_unavailable`, `timed_out`, `invalid_output`, `failed` or `cancelled`.

The outcome records the source hash/size, requested profile, runtime name/revision,
effective limits, enforcement report, duration, bounded diagnostics, truncation
markers and Python version when available. It never fabricates a successful value
after timeout, non-zero exit, missing result envelope or failed validation.

### 8.7 Serialization and compatibility

Public chain type is `code`; Python classes are `CodeStepDescription` and
`CodeStepConfig`. All configuration, policy, profile and outcome records are
Pydantic models. The chain format advances from 6 to 7 so older runtimes reject a
serialized CodeStep before encountering an unknown step type. Older chains require
no structural rewrite.

### 8.8 Acceptance criteria

1. Typed and legacy JSON save/load/execute round trips preserve the complete config.
2. Missing policy/profile/runtime and insufficient capability reports fail before
   runtime preparation or generated-code execution.
3. Exact source resolution, byte cap, syntax, entry-point shape and source hash are
   covered hermetically.
4. Inputs and outputs reject non-JSON values, NaN/infinity, schema violations and
   configured byte-limit violations.
5. Success, source/input/output failure, runtime failure, timeout and cancellation
   produce the specified typed outcomes.
6. Cancellation and timeout stop owned work; runtime cleanup is attempted under a
   bounded shield and cleanup failure cannot become success.
7. The Docker command uses a pinned image, no network, fixed interpreter, read-only
   input/output mounts, read-only root, bounded tmpfs, and CPU/memory/PID limits.
8. Output/history never use `default=str`; only validated JSON values cross the
   boundary.
9. Public exports, migration notes and a no-provider example are verified. The
   evo-metadata catalog remains outside this branch and is not recreated here.
10. Focused, lifecycle and supported hermetic suites plus a clean installed artifact
    pass without paid model calls or a required live Docker daemon.

### 8.9 Decision log

| Date | ID | Decision | Rationale |
| --- | --- | --- | --- |
| 2026-08-24 | A02-001 | Park `CodeStep` before accepting an execution contract. | The intended use case was clear, but its execution contract was not yet safe enough to freeze. |
| 2026-08-27 | A02-002 | Reopen `CodeStep`. | The capability returned to scope while isolation remained a prerequisite. |
| 2026-08-27 | A02-003 | Consume one exact runtime context string and require `def run(inputs)`. | This matches LLM-generated code without mixing generation into execution or guessing Markdown extraction. |
| 2026-08-27 | A02-004 | Require explicit input/output schemas and JSON-only boundaries. | Deterministic validation prevents Python objects and `default=str` coercions from crossing the runtime boundary. |
| 2026-08-27 | A02-005 | Require a host-owned strict isolated runtime profile; exclude local and best-effort execution. | Generated source cannot grant itself execution authority, and AST filtering is not a sandbox. |
| 2026-08-27 | A02-006 | Disable network, host environment, dependencies and writable host workspace in v1. | This is the smallest reviewable authority surface for arbitrary generated computation. |
| 2026-08-27 | A02-007 | Use typed lifecycle outcomes and source/runtime provenance. | Callers must distinguish validation, policy, runtime, timeout, cancellation and output failures. |
| 2026-08-27 | A02-008 | Advance the chain format from 6 to 7 for serialized CodeStep. | Older runtimes must reject the new public type with an upgrade signal. |

### 8.10 Verification evidence

Verified locally on `agent/code-step`, functional commit `4bedf11`, and published
as stacked PR #23 on 2026-08-27:

- 36 focused CodeStep tests pass for exact source handling, strict JSON boundaries,
  schemas, policy/profile denial, isolation capability checks, runtime provenance,
  success/failure, timeout, cancellation, cleanup and Docker composition.
- 449 combined CodeStep, AgentStep, WaitStep, HumanInputStep, runtime, preflight,
  round-trip and lifecycle tests pass.
- The full default suite reaches 2872 passed, 25 skipped and 15 deselected. Its
  only failure is the pre-existing tilde-expansion test blocked by the workspace
  sandbox from creating `~/.carl_test_tmp_xyz`; that exact test passes with the
  required filesystem permission.
- Focused Ruff, Python compileall and `git diff --check` pass. Wheel and source
  distributions build, and a clean wheel installation passes the format-7
  CodeStep round-trip smoke test.
- No paid provider call or live Docker daemon is required. The hermetic strict
  runtime test double is not claimed as production isolation evidence; Docker
  isolation and flag composition are checked separately.

## 9. A03 — Self-contained WaitStep

Status: `VERIFIED`
Contract: `A03-contract-v1`

### 9.1 Purpose and boundary

`WaitStep` asynchronously waits inside the currently running CARL process. It keeps
the chain execution alive but does not block the event loop or occupy a worker
thread. The step is deliberately self-contained: there is no external scheduler,
suspension store, lease, durable delivery protocol, or separate `Resume` API.

Waiting state is process-local and is lost if the process exits. Durable restart and
exactly-once resume are explicitly outside v1 and must not be claimed by this step.

### 9.2 Pydantic condition model

All public models use Pydantic `BaseModel`; CARL does not introduce dataclasses for
configuration, state, events, results, or serializable domain objects. Tagged
variants use a Pydantic discriminated union:

```text
WaitCondition =
  | After(seconds)
  | At(timestamp)
  | Event(name)
  | AnyOf([After | At | Event, ...])
```

- `After` is a finite non-negative relative duration in seconds.
- `At` is a timezone-aware absolute timestamp. A past timestamp completes
  immediately.
- `Event` waits for a non-empty named event in the current `ReasoningContext`.
- `AnyOf` races at least two leaf conditions and reports the first completed one.
- `AllOf`, arbitrary callables, durable events and nested compositions are deferred.

### 9.3 Event semantics

Events are emitted through the existing `context.emit_event(name, payload)` API.
They are process-local, broadcast, non-consuming and level-triggered: an event that
already exists completes a later waiter immediately. Re-emission uses last-value-wins
payload semantics. Payloads consumed by `WaitStep` must be JSON-compatible.

The event bus is shared by parallel execution snapshots and exposes an awaitable
notification primitive; `WaitStep` must not poll `has_event()` or expose a public
`poll_interval`. Emission wakes all current waiters. Cancellation is an independent
signal and never appears as an ordinary successful wait result.

Event emission is an immediate process-local side effect. It is visible before the
emitting step finishes and is not rolled back if that step later fails. This is
required for a sibling `WaitStep` to wake within the same parallel batch; callers
that require transactional publication must emit only after their own work commits.

### 9.4 Public shape and result

```json
{
  "number": 3,
  "title": "Wait for data or timeout",
  "step_type": "wait",
  "step_config": {
    "condition": {
      "type": "any_of",
      "conditions": [
        {"type": "event", "name": "data_ready"},
        {"type": "after", "seconds": 60}
      ]
    },
    "output_memory_key": "data_wait"
  }
}
```

Successful result data identifies the trigger and elapsed monotonic duration. An
event result also carries its name and payload. When `output_memory_key` is set, the
same structured result is written to the `wait` namespace for downstream steps.
`After(0)` and `At(past)` complete immediately. Chain timeout remains an independent
upper bound. Cancellation produces the normal cancelled step outcome.

### 9.5 Required implementation changes

1. Add Pydantic condition/config models, `StepType.WAIT`, `WaitStepDescription`,
   executor registration, exports and chain round-trip support.
2. Replace event polling with an awaitable shared event-bus token.
3. Share the event bus across parallel context snapshots so sibling execution can
   wake a waiter in the same batch.
4. Race timers/events against the shared cancellation signal and clean up losing
   tasks deterministically.
5. Keep event payload and trigger identity structured; do not encode timeout as
   silent success or rely on human-readable history for downstream branching.

### 9.6 Acceptance criteria

- Relative, absolute, pre-emitted event and later-emitted event waits complete with
  the documented structured result.
- `AnyOf(Event, After)` deterministically reports the first observed trigger.
- Waiting does not block unrelated async work and uses no polling interval.
- A parallel sibling can wake a waiter through the shared CARL event bus.
- Cancellation promptly ends timer and event waits and leaves no pending tasks.
- Invalid durations, naive timestamps, empty event names, invalid `AnyOf` shapes and
  non-JSON event payloads fail deterministically.
- Chain save/load, public imports and focused regression tests pass.

### 9.7 Decision log

| Date | ID | Decision | Rationale |
| --- | --- | --- | --- |
| 2026-08-26 | A03-002 | Supersede durable Await/Resume with one self-contained in-process `WaitStep`. | The project requires no external persistence, scheduler, lease, or resume construction for this capability. |
| 2026-08-26 | A03-003 | Support `After`, `At`, `Event`, and `AnyOf` in v1; defer `AllOf`. | This covers relative and absolute timers plus event-or-timeout without public polling or special timeout flags. |
| 2026-08-26 | A03-004 | Implement all Wait models as Pydantic models and tagged discriminated unions; do not use dataclasses. | This is a project-wide CARL model convention explicitly required by the project owner. |
| 2026-08-26 | A03-005 | Treat event emission as an immediate, non-transactional process-local side effect. | A sibling waiter must observe the event before the emitting step completes; rollback would require a different transactional event abstraction. |

### 9.8 Verification evidence

Verified locally on replacement branch `agent/wait-step-main` on 2026-08-26:

- 20 focused WaitStep tests pass for validation, relative/absolute timers,
  pre-emitted and later events, sync-thread emission, JSON payload enforcement,
  `AnyOf`, memory output, cancellation cleanup and parallel sibling wake-up.
- 372 combined WaitStep, lifecycle, event, cancellation, pause/resume,
  round-trip, AgentStep and Command/BASH regression tests pass.
- The full default test run reached 2815 passed, 25 skipped and 15 deselected.
  Its only failure was an existing tilde-expansion test blocked from creating
  `~/.carl_test_tmp_xyz` by the filesystem sandbox; that exact test passed when
  rerun with the required filesystem permission.
- The evo-metadata-only registry/catalog is deliberately absent from current
  `main` and is not restored by this capability PR.
- Python compileall and `git diff --check` pass; the new focused test file passes
  Ruff. Repository-wide Ruff is not claimed because the checkout has pre-existing
  lint findings outside this change.

Known v1 boundaries are intentional: waiting is process-local, a running chain is
retained, restart durability is absent, events are broadcast/level-triggered with
last-value-wins payloads, `AnyOf` accepts leaf conditions only, and `AllOf` is not
implemented.

## 10. A04 — Human-in-the-loop interaction

Status: `VERIFIED`
Contract: `A04-contract-v1`

AI actors and AIITL policies are explicitly out of scope.

### 10.1 Observed implementation state

The public `human_input` step, Pydantic config/description, executor, chain
round-trip and examples already exist. The current contract is limited to a string
prompt, optional timeout, fallback string and optional memory key.

The executor waits for one raw `asyncio.Future` stored privately on its execution
context. The callback receives `(prompt, future)` and is responsible for resolving
that future. There is no request id or typed request/response model. Parallel DAG
steps execute on context snapshots, so `provide_human_input()` called on the parent
context cannot address a future owned by a running snapshot; only the leaked future
passed to the callback is reliably correlated.

If no callback is installed, fallback is returned immediately with `success=True`
and `timed_out=False`. Timeout also returns fallback with `success=True`. Both paths
write the fallback to memory/history as if it were human input. The executor does not
race chain cancellation, does not validate a runtime response beyond the static
`Future[str]` annotation, and schedules async callback work without owning its
failure or cleanup. Raw values enter model-visible history without a sensitivity or
redaction policy. There is no actor identity, provenance or separation between
ordinary input and approval.

### 10.2 Accepted v1 contract

Keep the existing public `HumanInputStep` and stable `human_input` type rather than
introducing a second `InteractionStep`. V1 remains process-local and handles ordinary
human input only; AI actors and security approval are separate future capabilities.

Replace the raw-future callback with a typed host callback:
`HumanInputRequest -> HumanInputResponse | Awaitable[HumanInputResponse]`. The
runtime request has an opaque request id, step identity, a Pydantic text-input
specification and an optional deadline. The typed response repeats the request
id, contains a schema-valid value and records host-asserted human actor/provenance.
The callback may await its own UI or transport, but CARL owns timeout, cancellation
and task cleanup.

Return a Pydantic `HumanInputOutcome` with distinct `answered`, `timed_out`,
`unavailable`, `cancelled`, `invalid_response` and `failed` statuses. Only `answered`
is successful and writes a value to the configured memory key. V1 has no implicit
fallback; automated runs install an explicit fake-human callback. Sensitive answers
must be redacted or omitted from model-visible history and ordinary traces.

### 10.3 Acceptance criteria

- Request, response and outcome are Pydantic models and survive their documented
  JSON boundaries.
- Multiple parallel requests are correlated independently and complete at most once.
- Missing UI/callback cannot fabricate successful input.
- Timeout, unavailable, cancelled, invalid response and callback failure are distinct
  outcomes; cancellation promptly stops owned callback work.
- Values are validated against the declared input variant before memory mutation.
- Sensitive values are redacted from model-visible history and ordinary trace.
- Legacy fallback-based chain JSON is migrated explicitly or rejected with an
  actionable compatibility error; it is never silently treated as a human answer.

### 10.4 Deliberately deferred

- V1 supports bounded text only. Confirm, choice and JSON forms require a later
  extension decision.
- `actor_id` is optional host-asserted provenance, not verified authorization.
- Ordinary answers enter result/history by default. `sensitive=True` requires an
  output memory key and redacts the value from result/history.
- Format-5 migration removes serialized fallback values with an actionable warning;
  it never converts them into human answers.
- A synchronous callback must return promptly. CARL can cancel and await owned async
  callback work but cannot preempt a blocking synchronous callback.

### 10.5 Decision log

| Date | ID | Decision | Rationale |
| --- | --- | --- | --- |
| 2026-08-27 | A04-001 | Keep the public `HumanInputStep` / `human_input` type and use a process-local text-only v1. | Repairs the existing capability without adding a competing InteractionStep or external scheduler. |
| 2026-08-27 | A04-002 | Only a schema-valid typed human response is successful; remove implicit fallback success. | Missing providers and timeouts must not be fabricated or attributed to a human. |
| 2026-08-27 | A04-003 | Replace the raw Future callback with `HumanInputRequest -> HumanInputResponse | Awaitable[...]` and opaque request ids. | Gives parallel requests explicit correlation and lets CARL own timeout, cancellation and async cleanup. |
| 2026-08-27 | A04-004 | Return typed answered, timed_out, unavailable, cancelled, invalid_response and failed outcomes. | Downstream hosts need to distinguish lifecycle states instead of parsing fallback strings. |
| 2026-08-27 | A04-005 | Redact explicitly sensitive answers from result/history and require a memory destination. | Prevents unconditional disclosure while preserving an explicit downstream data path. |
| 2026-08-27 | A04-006 | Bump chain format from 5 to 6 and warn while removing legacy serialized fallback values. | Older runtimes fail clearly and migration cannot silently preserve false human attribution. |

### 10.6 Audit evidence

Verified locally on branch `agent/human-input-step`, based on WaitStep replacement
head `0af2c0e`, on 2026-08-27:

- 36 focused HumanInputStep tests pass for Pydantic validation, typed sync/async
  callbacks, request correlation, timeout, failure, cancellation, parallel snapshots,
  memory writes, redaction, serialization and public model JSON round trips.
- The combined agents/orchestration/lifecycle suite passes with 1799 tests and 8
  skipped.
- The full default suite reaches 2831 passed, 25 skipped and 15 deselected. Its only
  failure is the existing tilde-expansion test blocked by the filesystem sandbox;
  that exact test passes with the required home-directory permission.
- The executable HumanInput example passes, including intentional timeout and
  unavailable outcomes. Generated topic README drift is clean.
- Fatal Ruff checks on every changed Python file, focused Ruff, compileall and
  `git diff --check` pass.
- A fresh 0.3.0 sdist/wheel installs in a clean Python 3.13 environment and executes
  the typed HumanInputStep contract from the installed artifact with chain format 6.
- Host migration instructions are recorded in `docs/MIGRATION_human_input_v6.md`.

## 11. A05 — Map

Status: `VERIFIED`
Contract: `A05-contract-v1`

### 11.1 Purpose and boundary

`MapStep` turns a runtime JSON array into bounded parallel calls to one registered
host tool. It is a data-parallel step over a fixed operation, not a general loop,
dynamic DAG builder, router, or orchestrator.

V1 accepts only an array resolved from `$outer_context`, `$memory.*`, `$steps.*`,
`$metadata.*`, or `$event.*`. Every array element may be any JSON value. The source,
shared input mappings, complete array size, and registered tool availability are
validated before the first invocation.

### 11.2 Invocation contract

One configured registered tool is invoked exactly once for every admitted item. The
current item is passed under `item_parameter`; an optional `index_parameter` receives
its zero-based input index. `input_mapping` supplies shared JSON arguments resolved
once from the parent context. The item and shared arguments are copied through strict
JSON serialization before fan-out, so item calls receive no shared mutable CARL memory.

V1 does not provide per-item memory writes, filter, reduce, retry, fail-fast,
per-item tool selection, graph mutation, or nested-chain semantics. A later
chain-as-tool capability can participate only by registering its own explicit tool
contract; Map does not special-case chains.

### 11.3 Bounds, ordering, and failures

- `max_items` rejects oversized input before side effects.
- `max_concurrency` bounds active calls.
- `item_timeout_seconds` applies independently to every admitted item.
- Results are returned in input order, independent of completion order.
- V1 always collects every item outcome; one failed or timed-out item does not stop
  its siblings.
- An empty array succeeds without invoking the tool.
- The step succeeds exactly when all items have status `completed` (vacuously true
  for an empty array).

Each item has a Pydantic `MapItemOutcome` with index, status, success, JSON output or
error details, and execution time. Status is one of `completed`, `failed`,
`timed_out`, or `cancelled`. Pydantic `MapOutcome` contains the ordered items,
terminal counts, configured concurrency bound, and any enforcement gaps.

The complete aggregate is written at most once, after collection, when
`output_memory_key` is configured. Partial aggregates are written as data even when
the step is unsuccessful, so successful siblings are not discarded.

### 11.4 Cancellation and enforcement boundary

Host cancellation prevents pending items from starting and cancels/awaits async work
owned by MapStep. Cancellation wins a same-turn race with tool completion. All items
still receive terminal outcomes before the aggregate is returned.

Python cannot forcibly stop a synchronous callable already running in a worker thread.
After its awaitable is cancelled or timed out, that callable may continue in the
background. The aggregate records this explicitly in `enforcement_gaps`; hard-kill
semantics require a future process-backed tool runtime.

### 11.5 Serialization and preflight

The public wire type is `map`, represented by `MapStepDescription` and
`MapStepConfig`. Map's configured tool appears in `ReasoningChain.required_tools()`
and missing registration fails preflight. Chain format 8 adds the serialized MapStep
type (CodeStep took format 7); migration requires no structural rewrite.

### 11.6 Decision log

| Date | ID | Decision | Rationale |
| --- | --- | --- | --- |
| 2026-08-27 | A05-001 | Use public `MapStep` / `map` over one runtime JSON array. | Provides bounded data parallelism without introducing a general loop or dynamic graph. |
| 2026-08-27 | A05-002 | Invoke one registered tool per item with optional index and shared context arguments. | Keeps capability declaration and preflight compatible with existing host tools. |
| 2026-08-27 | A05-003 | Bound item count, concurrency and per-item time; preserve input order and always collect all outcomes. | Makes resource use predictable and partial failures inspectable. |
| 2026-08-27 | A05-004 | Use Pydantic item and aggregate outcomes; success means every item completed and empty input succeeds. | Gives downstream hosts a stable typed lifecycle contract. |
| 2026-08-27 | A05-005 | Cancel owned async work and report the sync-thread preemption gap explicitly. | Avoids claiming hard cancellation that the process-local Python runtime cannot enforce. |
| 2026-08-27 | A05-006 | Exclude filter, reduce, retry, fail-fast, shared mutable memory and dynamic DAG behaviour from v1. | Keeps Map a focused deterministic fan-out primitive. |
| 2026-08-27 | A05-007 | Bump chain wire format from 7 to 8 for MapStep (CodeStep took 6→7). | Older runtimes must reject the unknown serialized type clearly. |

### 11.7 Audit evidence

Verified on branch `agent/map-step`, based exactly on published
`agent/human-input-step` head `a3db163`, and published as stacked PR #22 on
2026-08-27:

- 40 focused MapStep tests pass for Pydantic validation, every accepted source,
  JSON isolation, sync/async tools, shared arguments and index, item/concurrency/time
  bounds, input ordering, collect-all failures, typed timeouts, cancellation and
  cleanup, single aggregate writes, partial aggregate commits, preflight, factory,
  migration, serialization, public exports and typed accessors.
- The full supported hermetic suite passes with 2874 tests, 25 skipped and 16
  deselected. The one default-suite tilde-expansion test excluded by the workspace
  sandbox passes separately with the required home-directory permission.
- The executable Map example and generated topic README drift check pass.
- Fatal Ruff checks, full Ruff on new files, compileall and `git diff --check` pass.
- A fresh 0.3.0 sdist/wheel installs with declared dependencies in a clean Python
  3.13 environment; the installed artifact executes an ordered MapStep after JSON
  round-trip with format 8 and commits its aggregate output.
- No live model/provider call or external side effect is part of verification.
- Package version remains the inherited `0.3.0`; release selection is a separate
  project-owner decision.

## 12. A06 — Orchestration

Status: `PARKED`; explicitly skipped after A04.

No separate Route, Supervisor, Planner or Orchestrator step is accepted. Reopen only
for a concrete use case that is not already served by static DAG execution, Map, or
explicit nested-chain invocation.

## 13. A07 — Chain as a tool

Status: `VERIFIED`
Contract: `A07-contract-v1`

### 13.1 Why this capability exists

A completed CARL chain should be reusable inside an `AgentStep` or an ordinary
`ToolStep` without copying its steps into the caller and without introducing an
orchestrator. The composed unit must preserve the exact child-chain snapshot and
expose typed input and output boundaries.

### 13.2 Public surface

Chain-as-tool is not a new step type. A parent `ReasoningChain` owns a list of
Pydantic `ChainToolDefinition` models. Each definition contains:

- a tool name and description;
- `contract_version=1`;
- a complete embedded `ReasoningChain.to_dict()` snapshot and canonical SHA-256;
- object-shaped input JSON Schema;
- output JSON Schema and one explicit child-context output reference;
- an explicit host-tool allowlist, tags, maximum nesting depth, wall-clock timeout
  and input/output byte limits.

`ReasoningContext.register_chain_tool()` is the direct host API. During normal chain
execution, the parent automatically registers its embedded definitions before any
step runs. A conflicting tool name fails closed instead of overwriting a host tool.

### 13.3 Invocation and isolation

Validated keyword arguments are encoded as JSON and become the child's
`$outer_context`. Every invocation creates fresh history, memory, messages, metadata,
event bus and lifecycle state. The child receives the parent's LLM client and host
system prompt, plus only the registered tools named by `allowed_tools`.

Parent memory, history, messages, metadata, human-input callbacks, long-term memory,
command policy and network authority are not inherited. Nested chains must embed
their own chain-tool definitions; they cannot import sibling definitions from the
parent.

### 13.4 Completion contract

The callable returns a JSON-compatible Pydantic `ChainToolOutcome` envelope with
invocation id, snapshot/input/output hashes, nesting depth, duration, token usage,
executed-step count and one of:

- `completed` — the child succeeded and the selected output matches its schema;
- `failed` — the child or its runtime failed;
- `cancelled` — parent cancellation stopped owned work;
- `timed_out` — the invocation deadline stopped owned work;
- `invalid_input` or `invalid_output` — a boundary schema did not match;
- `unavailable` — the snapshot or required host capability is unavailable;
- `recursion_limit` — a digest cycle or maximum depth was detected.

Only `completed` has `success=true` and may expose `output`. The snapshot digest is
checked again at invocation so mutation after validation cannot change executed code
under the recorded provenance identity.

### 13.5 Preflight, serialization and compatibility

Parent preflight treats embedded chain names as internally provided and reports their
transitive `allowed_tools` as host requirements. Construction rejects a child snapshot
whose statically required host tools are not declared. Execution checks all declared
host tools and name collisions before the first parent step.

Chain wire format increases from 6 to 7 and serializes top-level `chain_tools`.
Migration adds an empty list to older chains. Older runtimes must reject v7 instead of
silently dropping executable composition.

### 13.6 Accepted v1 boundaries

- Definitions embed snapshots; external registry identifiers and remote fetching are
  deferred.
- The supported boundary-schema subset is `type`, `properties`, `required`, `items`,
  `enum` and `additionalProperties=false`.
- A synchronous host tool may still continue in its worker thread after cancellation;
  this is an inherited Python runtime limitation, not a durable sandbox guarantee.
- Nested trace streaming and shared mutable parent memory are deliberately excluded.

### 13.7 Acceptance evidence

Verified locally on `agent/chain-as-tool`, based on HumanInputStep PR #20 head
`a3db163`, on 2026-08-27:

- 14 focused chain-as-tool tests pass for Pydantic validation, schema and byte
  boundaries, snapshot digest integrity, JSON round trips, v6-to-v7 migration,
  preflight, isolation, typed outcomes, failure, timeout, cancellation, recursion,
  AgentStep schema exposure and argument rejection.
- The combined agents/orchestration/lifecycle suite passes with 1813 tests, 8
  skipped and 7 deselected.
- The full default suite reaches 2845 passed, 25 skipped and 15 deselected. Its
  only failure is the existing tilde-expansion test blocked by the filesystem
  sandbox; that exact test passes with the required home-directory permission.
- The executable chain-as-tool example, generated topic README drift, focused and
  fatal Ruff, compileall and `git diff --check` pass.
- A fresh 0.3.0 sdist/wheel installs in a clean Python 3.13 environment and executes
  the serialized typed child tool with chain format 7 from the installed artifact.
- Host migration instructions are recorded in `docs/MIGRATION_chain_tools_v7.md`.

## 14. Parked backlog

### 14.1 B01 — Fixed Delay

Status: `SUBSUMED` by decisions `B01-001`, `A03-001`, and `A03-002`.

No separate fixed-duration step is planned. A03 `After(duration)` and `At(timestamp)`
cover timer waiting inside `WaitStep`. The chain remains alive and restart durability
is not provided. Retry/backoff, rate limiting and host scheduling remain owned by
their respective policies rather than Wait.

### 14.2 B02 — Global Terminate

Status: `PARKED` by decision `B02-001`.

No public global Terminate step is planned. Ordinary early completion routes to END;
failure, cancellation, budget exhaustion and local Agent/Code outcomes remain owned
by their components. Reopen only for a concrete use case that requires cross-branch
global halt authority and defines deterministic arbitration among concurrent work.

The A01 `finish` meta-tool does not reopen B02: it exits only the AgentStep loop.

## 15. Global decision log

| Date | ID | Decision | Rationale |
| --- | --- | --- | --- |
| 2026-08-12 | META-001 | Use a living specification as the design gate before implementation. | Discuss and freeze contracts before changing runtime code. |
| 2026-08-12 | B02-001 | Park the public global Terminate step. | No distinct use case justified global scheduler authority over routing to END and component-local outcomes. |
| 2026-08-12 | B01-001 | Park the public fixed Delay step while keeping durable Await/Resume open. | Retry, rate limiting, polling, scheduling and interaction own the current use cases. |
| 2026-08-24 | SCOPE-001 | Active scope is A01 AgentStep, A02 CodeStep, A03 Await/Resume and A04 human HITL; the Section 4 topics are removed. | Explicit project-owner scope reduction; Command/BASH continues only in its separate workstream. |
| 2026-08-27 | SCOPE-002 | Reopen Map, orchestration, chain-as-tool and CodeStep while keeping AIITL and the other Section 4 exclusions out of scope. | Explicit project-owner roadmap revision after AgentStep and WaitStep verification. |
| 2026-08-27 | ORDER-002 | Continue in the order HumanInputStep, Map, orchestration, chain-as-tool, CodeStep. | Keeps each capability in a separate design and PR stage and defers untrusted code execution until composition contracts are understood. |
| 2026-08-27 | ORDER-003 | Park orchestration; Map, CodeStep and chain-as-tool are implemented. | No current routing/planning use case justifies a new public orchestration abstraction. |
| 2026-08-27 | A05-001 | Use public `MapStep` / `map` over one runtime JSON array. | Adds focused bounded data parallelism. |
| 2026-08-27 | A05-002 | Invoke one registered tool per item with optional index and shared inputs. | Reuses the declared host capability boundary. |
| 2026-08-27 | A05-003 | Bound size, concurrency and per-item time; preserve order and collect all. | Makes execution predictable and partial outcomes inspectable. |
| 2026-08-27 | A05-004 | Use Pydantic item/aggregate outcomes and succeed iff every item completes. | Establishes a typed lifecycle contract including empty input. |
| 2026-08-27 | A05-005 | Cancel owned async work and report sync-thread preemption gaps. | Matches enforceable process-local semantics. |
| 2026-08-27 | A05-006 | Exclude filter, reduce, retry, fail-fast, shared mutable memory and dynamic DAG behaviour from v1. | Prevents scope expansion into orchestration. |
| 2026-08-27 | A05-007 | Bump the chain wire format from 7 to 8 for MapStep (CodeStep took 6→7). | Gives older runtimes a clear upgrade boundary. |
| 2026-08-27 | A06-001 | Park orchestration without accepting Route, Supervisor, Planner or Orchestrator. | The project owner explicitly skipped this capability and moved to chain-as-tool. |
| 2026-08-27 | A07-001 | Implement chain-as-tool as a serialized parent-chain capability, not a new step type. | AgentStep and ToolStep already consume registered tools; composition should reuse that boundary. |
| 2026-08-27 | A07-002 | Embed a complete child snapshot with canonical digest instead of resolving an external id in v1. | Makes invocation portable and reproducible without registry availability or version-selection semantics. |
| 2026-08-27 | A07-003 | Use Pydantic definition and outcome contracts with typed input/output and fresh child mutable state. | Preserves CARL's model convention and prevents ambient parent memory/history leakage. |
| 2026-08-27 | A07-004 | Expose only explicitly allowed host tools and fail preflight on missing capabilities or name collisions. | A serialized chain may narrow host authority but may not grant itself new authority. |
| 2026-08-27 | A07-005 | Bound nested invocation by timeout, cancellation, depth and digest-cycle detection. | Nested work must not outlive its owner or recurse without a deterministic bound. |
| 2026-08-27 | A07-006 | Bump chain wire format from 8 to 9 for serialized top-level chain tools. | Older runtimes must reject executable composition they cannot preserve. |
| 2026-08-24 | ORDER-001 | Discuss the basic AgentStep first. | Explicit project-owner choice; v1 is narrowed so it does not depend on removed composition or later HITL support. |
| 2026-08-24 | A02-001 | Park `CodeStep` before accepting an execution contract. | Runtime execution of LLM-generated source remains a backlog use case; its public and isolation contracts are not frozen. |
| 2026-08-27 | A02-003 | Execute one exact context-sourced Python string through `def run(inputs)`. | Keeps generation separate and avoids implicit Markdown extraction or repair. |
| 2026-08-27 | A02-004 | Require strict JSON input/output schemas. | Python objects and lossy string coercion must not cross the boundary. |
| 2026-08-27 | A02-005 | Require host-owned strict isolated profiles and exclude local/best-effort execution. | Arbitrary generated code requires a real killable sandbox. |
| 2026-08-27 | A02-006 | Disable network, host environment, dependencies and writable host workspace in v1. | Minimizes authority and bounds host-side accumulation. |
| 2026-08-27 | A02-007 | Emit typed outcomes with source/runtime provenance. | Makes every non-success lifecycle state explicit. |
| 2026-08-27 | A02-008 | Bump the chain format from 6 to 7 for CodeStep. | Older runtimes fail with a version signal before an unknown type. |
| 2026-08-27 | A04-001 | Keep `HumanInputStep` and use a process-local text-only v1. | Avoids a competing public InteractionStep and defers forms and external delivery. |
| 2026-08-27 | A04-002 | Treat only a typed valid response as success and remove implicit fallback. | Timeout or missing integration is not human input. |
| 2026-08-27 | A04-003 | Use typed request/response callbacks with request ids and CARL-owned async lifecycle. | Supports correlation, timeout, cancellation and cleanup. |
| 2026-08-27 | A04-004 | Use a typed HumanInputOutcome status set. | Makes lifecycle states explicit. |
| 2026-08-27 | A04-005 | Redact explicitly sensitive answers and require a memory destination. | Preserves a deliberate downstream path without model-visible disclosure. |
| 2026-08-27 | A04-006 | Bump chain format from 5 to 6 and warn when migrating away legacy fallbacks. | Prevents silent false attribution and gives older runtimes an upgrade signal. |
| 2026-08-24 | A03-001 | Do not expose a separate Delay step; A03 supports durable `After`, `At`, and `Event` wake conditions. | Timer waiting can share persistence, worker release, restart and idempotent resume semantics; short sleeps remain an internal runtime detail. |
| 2026-08-26 | A03-002 | Supersede A03-001's durable Await/Resume design with one self-contained in-process `WaitStep`. | The project owner explicitly rejected external persistence, scheduler and resume constructions for this step. |
| 2026-08-26 | A03-003 | WaitStep v1 supports Pydantic `After`, `At`, `Event`, and `AnyOf` conditions; `AllOf` is deferred. | Covers relative/absolute timers and event-or-timeout composition without polling. |
| 2026-08-26 | A03-004 | Use Pydantic rather than dataclasses for all WaitStep models. | Mandatory CARL model convention. |
| 2026-08-26 | A03-005 | Event emission is immediate and is not rolled back with a failed emitting step. | Required for communication between concurrently executing siblings in one batch. |
| 2026-08-26 | A03-006 | Rebuild WaitStep on the main-based AgentStep replacement. | Keeps the one-capability-per-PR stack while correcting the original integration target. |
| 2026-08-26 | A03-007 | Bump chain wire-format version from 4 to 5 for WaitStep. | Older runtimes must reject serialized WaitStep chains with a clear upgrade signal instead of encountering an unknown step type. |
| 2026-08-24 | A01-001 | Use one public `AgentStep` as a bounded ReAct loop that receives a goal and allowed tool capabilities and attempts to produce a result. | Establishes the basic agent abstraction without adding a second public AgentLoop type or reintroducing orchestration. |
| 2026-08-24 | A01-002 | AgentStep v1 uses an explicit non-empty tool allowlist, exactly one call per iteration and explicit schema-checked `finish`. | The project owner accepted the minimal deterministic ReAct protocol and authorized implementation. |
| 2026-08-26 | A01-003 | Rebuild AgentStep from current `main` rather than merge it into `evo-metadata`. | `evo-metadata` was incorrectly treated as the integration target; replacement PR #18 preserves current Command/BASH behaviour and ports only AgentStep requirements. |
| 2026-08-26 | A01-004 | Bump chain wire-format version from 3 to 4 for AgentStep. | Older runtimes must reject serialized AgentStep chains with a clear upgrade signal instead of encountering an unknown step type. |

## 16. Audit evidence index

| Evidence | Reference | Used for |
| --- | --- | --- |
| Pull request #4 | head `e23e712e27e878cba53ce9fdd58e4879628f7f24` | Prototype AgentLoop and Code behaviour; Wait and Terminate backlog evidence |
| Current runtime checkout | stacked base `agent/agent-step-main` at `92f2dfd`; implementation branch `agent/wait-step-main` | Main-based self-contained WaitStep and shared process-local event bus |
| Experimental agent checkout | committed head `57db2e1`; additional uncommitted work exists | Rich AgentStep mechanisms and known terminal-semantics defects; evidence only |

All drift-prone observations must be rechecked against the exact implementation base
before a capability moves from `PROPOSED` to `ACCEPTED` and against a clean built
artifact before it moves to `VERIFIED`.

## 17. Implementation evidence index

- Replacement AgentStep branch: `agent/agent-step-main`, created from `origin/main`
  at `11a5298742ecb5aaa23612fdb06538b07c72d08f`; published as PR #18.
- Focused AgentStep suite: `24 passed`; the AgentStep plus Command/BASH and
  directly affected compatibility suite: `434 passed` with no paid/provider calls.
- Full supported hermetic AgentStep suite: `2792 passed`, `25 skipped`,
  `15 deselected`; its only sandbox failure was the existing tilde-expansion test,
  which passed independently with the required home-directory permission.
- Static gate: Ruff fatal/error checks on every changed Python file and
  `git diff --check` pass.
- Artifact gate: fresh `mmar_carl-0.3.0` wheel and sdist build; the wheel installs
  with all declared dependencies in a clean Python 3.13 environment and imports
  `AgentStepConfig`, `AgentStepDescription`, `AgentStepExecutor`, and both standard
  and `verify_ssl=False` OpenAI-compatible client paths under `openai==3.3.1`.
- Package version remains `0.3.0` only as the inherited `main` value. This branch
  must not be published under an already-used version; release/version selection
  is a separate project-owner decision.
- No live model/provider call is part of this verification.
- MapStep branch: `agent/map-step`, created from published HumanInputStep head
  `a3db163`; verified locally with 40 focused tests and the full supported hermetic
  suite at 2874 passed, 25 skipped and 16 deselected; published as stacked PR #22
  targeting `agent/human-input-step`.
- MapStep artifact gate: fresh 0.3.0 wheel and sdist; the clean installed wheel
  executes ordered format-7 MapStep serialization and memory-output behaviour.

Known v1 enforcement gaps:

- A synchronous in-process tool may continue in its worker thread after timeout
  or cancellation; the result records this explicitly. Hard-kill semantics require
  a killable process backend.
- `max_tokens` is enforceable only when the selected client reports usage.
- Tool revisions, output schemas, approvals, side-effect classes, durable tool
  replay and cost budgets remain deferred as recorded in Sections 7.6 and 7.17.
