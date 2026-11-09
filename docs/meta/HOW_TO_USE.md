# `mmar_carl.meta` — step-type meta-description: how to use it

`mmar_carl.meta` is a **token-efficient, self-updating description of every CARL
step type**. It exists so that _other systems_ — a chain generator (MAGE), an
evolutionary searcher (GigaEvo), a TUI, or a hand-written script — can answer
the questions they need to build and improve chains:

- _What step types exist?_
- _What fields does each one need (required vs optional)?_
- _Is this chain spec valid — do its dependencies resolve, are required fields
  present — without running it?_
- _Give me a minimal valid step I can splice in (mutation / generation)._
- _Which fields are safe to mutate?_

Everything structural is **introspected live** from the pydantic models + the
executor registry, so when a new step type is added the catalog updates itself
— no edits to this module. See [§8](#8-how-auto-update-works-and-the-drift-guard).

> Source: `src/mmar_carl/meta/`. Public API is re-exported from the top-level
> `mmar_carl` package.

---

## 1. Quick start

```python
from mmar_carl import get_step_catalog, validate_spec, auto_generate_minimal_step

cat = get_step_catalog()

print(cat.tier0())                       # one line per type — tiny
print(cat.tier2("tool"))                 # full card for one type

report = validate_spec({"steps": [...]}) # structured, no execution
print(report.format_text())

step = auto_generate_minimal_step("evaluation", number=3, dependencies=[2])
```

The catalog is built once and cached per process. Pass `refresh=True` to rebuild
(e.g. after registering a custom step type at runtime):
`get_step_catalog(refresh=True)`.

---

## 2. Progressive disclosure — pull only the tier you need

The whole point is to never send more than necessary to a model. Four tiers,
smallest first:

| Tier | Call                                       | What you get                                                                         | Rough size (all 16 types) |
| ---- | ------------------------------------------ | ------------------------------------------------------------------------------------ | ------------------------- |
| 0    | `cat.tier0()` / `cat.list_step_types()`    | name → one-line purpose                                                              | ~300 tokens               |
| 1    | `cat.tier1(include=None)` / `spec.card()`  | required + key-optional fields + a ready example                                     | ~3 KB                     |
| 2    | `cat.tier2(name)` / `spec.card(full=True)` | **every** field + descriptions, validators, choices, and inlined nested-model fields | ~1–3 KB / type            |
| 3    | `schema_for(name)`                         | raw JSON Schema of the config model, on demand                                       | per type                  |

### Tier 0 — the menu

```text
- llm: Chain-of-thought reasoning with an LLM. The default building block.
- tool: Call a registered Python function/tool with mapped inputs.
- memory: Read / write / append / delete / list namespaced shared memory.
- conditional: Branch the DAG: route to a step whose condition matches.
- supervisor: LLM routes a task to one of N registered specialist sub-chains.
- debate: Round-robin multi-role debate, then an LLM judge synthesis.
... (16 total)
```

### Tier 1 — a compact card

```text
### tool — Call a registered Python function/tool with mapped inputs.
  (step_config: ToolStepConfig)
  required:
    - tool_name: str
  key optional (8 total):
    - error_recovery: ToolErrorRecovery?
    - input_mapping: dict[str, str]
    - timeout: float=30.0
    - tool_description: str=''
    - allowed_tool_tags: list[str]?
    - output_key: str='result'
    - (+2 more optional fields; see tier 2)
  example: {"number": 1, "title": "tool step", "step_type": "tool", "dependencies": [],
            "step_config": {"tool_name": "web_search", "input_mapping": {"query": "$history[-1]"}}}
```

"Key optional" prioritises the fields a generator/mutator actually cares about
(the curated `mutable_fields`), then caps the rest. Tier 2 shows them all.

### Tier 3 — the raw schema, only when you need it

```python
schema = schema_for("structured_output")   # standard JSON Schema dict
```

For embedded-config types this is the config model's schema. The `llm` type is
inlined (no config model), so a schema is synthesised from its fields — you
still get a real JSON Schema back.

---

## 3. Generate a chain

The auto-updating planner prompt is already wired into
`ChainBuilder.from_description`, but you can render it yourself for your own
generator:

```python
prompt = cat.generation_prompt(
    max_steps=6,
    include=None,                       # default: all types; or ["llm","tool","memory"]
    detail="cards",                     # "cards" (tier-1 per type) or "menu" (tier-0 only)
    available_tools=["web_search"],     # surfaced for `tool` steps
    available_skills=["pdf"],           # surfaced for `agent_skill` steps
    extra_instructions="Be concise.",
)
```

It contains: the JSON envelope shape, the rules (string-not-array, no
`triggered_by`, max-steps), a per-type menu/cards, and the `$...` reference
grammar — and it **covers every registered type automatically**.

Then go straight from the model's JSON to a validated, constructed chain with
**one call**:

```python
from mmar_carl import validate_and_build

report, chain = validate_and_build(llm_output)   # llm_output is the parsed {"steps": [...]}
if report.valid:
    result = await chain.execute_async(ctx)
else:
    # feed precise, retry-ready feedback back to the planner
    feedback = "\n".join(e.format_for_llm() for e in report.errors)
```

`validate_and_build` normalises the spec to chain form (so cross-step checks
always run), validates it, and only then builds a `ReasoningChain` — returning
`(report, chain)`. On failure `chain` is `None` and every error (structural
_and_ any residual construction error) is one `ValidationIssue` shape. A green
report means the chain is structurally sound; runtime-only sub-agents
(`sub_chain` / `agents` / `base_step`) still need binding before execution.

---

## 4. Mutate / evolve a chain

Everything an evolutionary operator (GigaEvo-style) needs — INSERT, value
mutation, and DELETE — against the public API only.

### Read → mutate → validate, driven by data

`mutable_fields(name)` says which fields are safe to change; `field_choices` /
`field_constraint` say what a _valid_ value looks like; and
`get_field_by_path` / `set_field_by_path` resolve the path convention
(`step_config.<field>`, bare `<field>`, and list indices like
`step_config.branches.0.condition`).

```python
from mmar_carl import (
    get_step_catalog, mutable_fields, get_field_by_path, set_field_by_path,
    iter_mutable_values, validate_spec,
)
cat = get_step_catalog()

# temperature-swap within the valid range (no hand-coded 0..2)
temp = next(f for f in cat.step("llm").field("llm_config").nested_fields if f.name == "temperature")
set_field_by_path(step, "llm_config.temperature", (temp.constraints["ge"] + temp.constraints["le"]) / 2)

# model-swap / prompt-rewrite
set_field_by_path(step, "llm_config.model", "anthropic/claude-3.5-sonnet")
set_field_by_path(step, "aim", "Rewritten objective.")

# enum-swap using structured choices, not string parsing
choices = cat.step("evaluation").field_choices("evaluation_method")   # ['rule', 'llm']

# inspect every mutable target and its current value
for path, value in iter_mutable_values(step):
    ...

assert validate_spec(step).valid   # confirm the mutant is still well-formed
```

`set_field_by_path` creates intermediate dicts, so you can set
`llm_config.temperature` on a step that has no `llm_config` yet.

### INSERT_STEP

```python
new_step = auto_generate_minimal_step("parallel_sampling", number=4, dependencies=[3])
```

A minimal _valid_ step in the canonical serialization shape — passes
`validate_spec` and reconstructs through `from_dict`.

### DELETE_STEP

```python
from mmar_carl import find_leaf_steps, can_delete_step

for n in find_leaf_steps(spec["steps"]):        # steps nothing depends on
    if can_delete_step(spec["steps"], n):        # leaf + ≥1 step remains
        spec["steps"] = [s for s in spec["steps"] if s["number"] != n]
        break
```

Notes:

- `mutable_fields(name)` never includes runtime-only fields
  (`sub_chain` / `agents` / `base_step`).
- Nested config models (`llm_config`, `server`, `branches`, …) are expanded
  into `FieldSpec.nested_fields`, each carrying its own `choices` / `constraints`
  — so you never parse a JSON schema to learn a nested shape (see [§4a](#4a-field-level-detail-choices-constraints-nested-models)).

### 4a. Field-level detail (choices, constraints, nested models)

Each `FieldSpec` carries structured data a generator/mutator can act on
directly, without parsing the `.type` string or the tier-3 schema:

```python
f = cat.step("mcp").nested_spec("server")   # -> list[FieldSpec] for MCPServerConfig
[(x.name, x.choices) for x in f if x.choices]
# [('transport', ['stdio', 'http', 'sse'])]

cat.step("agent_skill").field_choices("execution_mode")
# ['llm', 'script', 'hybrid', 'subagent', 'llm_agent']   # Enum and Literal unified

cat.step("llm").field("llm_config").nested_fields    # LLMStepConfig fields, incl.
# temperature -> constraints {'ge': 0.0, 'le': 2.0}
```

- `FieldSpec.choices` — Enum/Literal values as `list[str]` (unified — no
  separate `enum[...]` vs `Literal[...]` grammars).
- `FieldSpec.constraints` — `{'ge': 0.0, 'le': 2.0, 'min_length': 1, ...}`.
- `FieldSpec.nested_model` / `container` / `nested_fields` — the nested pydantic
  model's own fields (`container` is `'scalar'` | `'list'` | `'dict'`).
- `FieldSpec.accepts_references` — True when the field takes `$...` strings.

---

## 5. Validate without executing

`validate_spec` runs **no LLM, no tools, no context** — pure structural and
schema checks. It accepts a single step dict, a list of steps, or a chain dict
(`{"steps": [...]}`).

```python
report = validate_spec({"steps": [
    {"number": 1, "title": "x", "step_type": "llm"},                    # missing aim
    {"number": 1, "title": "y", "step_type": "tool", "step_config": {}},# dup number + missing tool_name
    {"number": 3, "title": "z", "step_type": "memory",
     "dependencies": [9], "step_config": {"operation": "read"}},        # bad dep + missing memory_key
]})

print(report.valid)            # False
print(report.format_text())
```

```text
INVALID: 5 error(s), 0 warning(s)
  ✗ step 1 [aim] missing_required_field: llm steps require a non-empty 'aim'
  ✗ step 1 [step_config.tool_name] invalid_config: Field required
  ✗ step 3 [step_config.memory_key] invalid_config: Field required
  ✗ step 1 duplicate_number: duplicate step number 1
  ✗ step 3 [dependencies] missing_dependency: depends on non-existent step 9
```

What it checks:

| Scope             | Checks                                                                                                      |
| ----------------- | ----------------------------------------------------------------------------------------------------------- |
| Every step        | known `step_type`, `title` present, required fields / config validity (via `pydantic`), `llm.aim` non-empty |
| Chain / list only | unique `number`s, every dependency resolves, no cycles, `$memory` read with no writer (**warning**)         |

A **single** step is validated for its own correctness only — its
`dependencies` reference siblings that aren't present, so cross-step checks are
skipped for a lone step.

`ValidationReport` fields: `valid: bool`, `errors: list[ValidationIssue]`,
`warnings: list[ValidationIssue]`. Each `ValidationIssue` has `severity`,
`code` (stable machine code), `message`, `step`, `field`. It renders in Jupyter
(`_repr_markdown_`) and to stdout (`format_text()`).

---

## 6. Reference grammar & validation rules as data

The `$...` dynamic-reference grammar and the validation rules are exposed as
structured data so you can show them in a UI or feed them to a model:

```python
for r in cat.references:
    print(r.syntax, "→", r.returns, "—", r.description)
# $history[-1] | $history[0] | $history → str | list[str] — Output of a previous step ...
# $memory.<namespace>.<key> | $memory.<key> → Any — Read namespaced shared memory ...
# $ltm.<key>, $event.<name>, $metadata.<path>, $steps.<N>.result, $outer_context, literals ...

print(cat.reference_syntax())          # pre-formatted block
print(cat.common_fields_text())        # fields shared by every step (number, title, dependencies, ...)
cat.validation_rules                   # list of {code, severity, rule}
```

---

## 7. Serialize for an external service

The catalog is a pydantic model, so it round-trips to versioned JSON. There are
**two** ways to consume it from a separate service (e.g. MAGE / GigaEvo):

**(a) Live, versioned payload** — generate on demand and cache by version:

```python
blob = cat.to_json()                   # includes schema_version + carl_version
# ... ship over HTTP; on the other side:
from mmar_carl import StepCatalog
restored = StepCatalog.from_json(blob)
```

**(b) Committed snapshot** — `src/mmar_carl/meta/catalog_snapshot.json` is a
versioned artifact checked into the repo and shipped in the wheel. Regenerate
it whenever step types change:

```bash
make meta-snapshot          # writes catalog_snapshot.json
```

```python
from mmar_carl.meta import load_snapshot
snap = load_snapshot()                 # StepCatalog, or None if not shipped
```

The snapshot omits a timestamp and is compared **structurally**, so version
bumps never churn it — only an actual change to the step-type surface does
(`tests/meta/test_snapshot_drift.py` fails when it's stale, telling you to run
`make meta-snapshot`).

---

## 8. How auto-update works (and the drift guard)

You never hand-list step types here. The catalog is assembled by introspecting:

1. `StepDescriptionBase.__subclasses__()` — the typed step classes. The
   `step_type` is read via `cls.model_construct().step_type`.
2. the `_EXECUTORS` registry — the one site that must be updated for a type to
   run at all (used for executor names + the canonical type set).
3. each model's `model_fields` — for field names, types, required/optional,
   defaults, descriptions, and constraints.

A small **curated side-table** (`meta/_curated.py`) adds only what reflection
can't infer — `purpose`, a minimal `example`, `mutable_fields`, and
`required_overrides` (e.g. `llm.aim` is field-optional but validator-required).
Missing curated data degrades gracefully (docstring fallback, required-only
example).

### Adding a new step type

Follow the normal CARL steps (enum value, config class, `*StepDescription`
subclass, executor + `_EXECUTORS` entry). Then:

1. **Nothing** is required in `meta/` for it to appear in the catalog,
   `validate_spec`, `schema_for`, and the planner prompt.
2. Add a curated entry in `meta/_curated.py` (`purpose`, `example`,
   `mutable_fields`) for a good card and a known-good template — recommended,
   not required.
3. Run `make meta-snapshot` to refresh the committed snapshot.

### The drift guard

`check_drift()` (and `tests/meta/test_drift_guard.py`) assert that the
`StepType` enum, the executor registry, the discovered subclasses, the
`AnyStepDescription` union, and the `from_dict` dispatch all agree. If any site
falls out of sync the test fails — this is the safety net that catches the
class of bug that once left three types out of `AnyStepDescription`.

```python
from mmar_carl import check_drift
report = check_drift()
print(report.in_sync, report.issues)
```

---

## 9. API reference

Top-level (`from mmar_carl import ...`) and `mmar_carl.meta`:

| Symbol                                                                                 | Purpose                                                       |
| -------------------------------------------------------------------------------------- | ------------------------------------------------------------- |
| `get_step_catalog(*, refresh=False) -> StepCatalog`                                    | The cached live catalog.                                      |
| `validate_spec(spec) -> ValidationReport`                                              | No-execution validation of a step / list / chain.             |
| `validate_and_build(spec, *, use_typed_steps=True) -> (ValidationReport, chain\|None)` | Validate then build a `ReasoningChain` in one call.           |
| `auto_generate_minimal_step(name, *, number=1, title=None, dependencies=None) -> dict` | Minimal valid step spec (INSERT_STEP).                        |
| `schema_for(name) -> dict`                                                             | Raw JSON Schema for a step type **or config class** (tier 3). |
| `config_class_names() -> list[str]`                                                    | Nested config classes resolvable by `schema_for`.             |
| `mutable_fields(name) -> list[str]`                                                    | Fields safe to mutate.                                        |
| `reference_fields(name) -> list[str]`                                                  | Config field paths that accept `$...` references.             |
| `get_field_by_path(step, path, *, default=…) -> Any`                                   | Read a value at a `mutable_fields` path.                      |
| `set_field_by_path(step, path, value) -> None`                                         | Write a value (creates intermediate dicts).                   |
| `iter_mutable_values(step) -> list[tuple[str, Any]]`                                   | `(path, current_value)` for every mutable target.             |
| `find_leaf_steps(steps) -> set[int]`                                                   | Step numbers safe to delete (nothing depends).                |
| `can_delete_step(steps, number) -> bool`                                               | Is `number` a deletable leaf?                                 |
| `check_drift() -> DriftReport`                                                         | Are the registration sites in sync?                           |
| `StepCatalog`                                                                          | The catalog model — see methods below.                        |
| `StepTypeSpec`, `FieldSpec`, `ReferenceSpec`                                           | Per-type / per-field / per-reference data.                    |
| `ValidationReport`, `ValidationIssue`, `DriftReport`                                   | Structured results.                                           |

`StepCatalog` methods: `names()`, `step(name)`, `list_step_types()`, `tier0()`,
`tier1(include=None)`, `tier2(name)`, `reference_syntax()`,
`common_fields_text()`, `generation_prompt(...)`, `to_json()`, `from_json()`,
`format_text()`, `_repr_markdown_()`.

`StepTypeSpec` fields: `step_type`, `purpose`, `description_class`,
`config_class`, `executor_class`, `inlined`, `config_fields`,
`required_fields` (property), `optional_fields` (property),
`runtime_only_fields`, `mutable_fields`, `reference_fields`, `validators`,
`example`; methods `field(name)`, `field_choices(name)`,
`field_constraint(name, key)`, `nested_spec(name)`,
`card(*, full=False, max_optional=6)`, `_repr_markdown_()`.

`FieldSpec` fields: `name`, `type`, `required`, `default`, `description`,
`excluded`, `choices`, `constraints`, `nested_model`, `container`,
`nested_fields`, `accepts_references`.

`ValidationIssue` adds `format_for_llm()` — a retry-ready sentence keyed on the
stable `code`; `field` is always a dotted path.

`mmar_carl.meta` also exposes snapshot helpers: `build_catalog()`,
`load_snapshot()`, `snapshot_is_stale()`, `write_snapshot()`, `snapshot_payload()`,
`SNAPSHOT_PATH`.

---

## 10. Notes & gotchas

- **Runtime callers always get accurate data.** `get_step_catalog()` builds
  live from the current code; the committed snapshot is only for external
  consumers and the CI drift check, never for runtime behaviour.
- **`*StepDescription` classes can't emit a JSON Schema** (an inherited
  `cache: StepCache` field carries a `Callable`), which is why tier-3 schemas
  come from the _config_ class, and the `llm` type's schema is synthesised.
- **Runtime-only fields** (`sub_chain`, `agents`, `base_step`) never appear in
  a spec or in `mutable_fields`; `from_dict` injects placeholders for them, and
  CARE rebinds the real values from gigaevo-memory before execution.
- **Validation ≠ execution.** `validate_spec` confirms a chain is _well-formed_
  (structure + required fields + config shapes). It does not resolve `$memory`
  values or run anything — semantic/runtime errors still surface at execution.
