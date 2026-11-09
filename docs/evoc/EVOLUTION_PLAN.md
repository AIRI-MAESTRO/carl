# EvoC — Chain Evolution Engine: plan & CARL readiness

**Status:** CARL substrate **implemented** ✅ • **Audience:** EvoC implementers + CARL maintainers
**Scope:** EvoC is a **separate repository** that depends on **CARL** (`mmar-carl`) as its base
library. This document extends the original EvoC algorithm with the concrete CARL
primitives each step relies on, a readiness matrix, the CARL-side additions, and the
inconsistencies fixed.

> **Implementation status.** All CARL-side additions **A1–A10** below are now shipped and
> tested — CARL is ready to host EvoC. Quick map (all importable from top-level `mmar_carl`
> and from `mmar_carl.meta`):
>
> - **A1/A10 topology + DAG** (`mmar_carl.meta.mutations`): `rewire_dependencies`,
>   `change_step_type`, `insert_step_at`, `remove_step` (dependency + loop-edge healing),
>   `apply_records`, `MutationRecord`/`MutationError`, `find_root_steps`, `dependents_of`,
>   `steps_of_type`, `deletable_steps`, `expand_field_path`.
> - **A2 content sampler**: `sample_field_value`, `mutate_content`, `mutable_targets`.
> - **A3 descriptors** (`mmar_carl.run_descriptor`): `RunDescriptor`, `ComplexityDescriptor`,
>   `extract_run_descriptor`, `extract_complexity`, `behavior_key`, `as_objectives`.
> - **A4 dataset** (`mmar_carl.models.dataset`): `subsample`, `split`, `select_cases`;
>   `DatasetEvaluator(max_concurrent=…)`.
> - **A5 partition/diff**: `DatasetEvaluationReport.partition` → `CaseSplit`,
>   `.diff_by_case` → `CaseDiff`, `.by_label`.
> - **A6 capability library** (`mmar_carl.meta`): `CapabilityLibrary` / `CapabilityEntry`,
>   `generation_prompt(capability_library=…)`, `from_description(capability_library=…)`,
>   `SkillLoader.as_capability_library`.
> - **A7 lineage substrate**: `IndividualMetrics.id` / `parent_id` / `mutation_records`.
> - **A8 Pareto** (`mmar_carl.pareto`): `pareto_front`, `pareto_rank`, `dominates`.
> - **A9 freezing**: `StepDescriptionBase.frozen` + `frozen_step_numbers`; every mutation op
>   honours `respect_frozen=True`.
>
> Fixed inconsistencies: dangling `loop_back_to` now flagged by `validate_spec` (code
> `dangling_loop`); `remove_step`/`change_step_type` heal loop + dependency edges; a
> shared-mutable-state bug where a generated step could corrupt the cached catalog was
> fixed (deep-copied examples). §5 items #6 (str-enum equality already works) and #12
> (`ChainMutator` already uses the seeded rng) were verified non-issues.
>
> **Milestone 3 (polish) — now shipped too:** §5 #5 (`to_lineage_mermaid` recovers
> parent edges **exactly** from `IndividualMetrics.id`/`parent_id` when present, falling
> back to score-matching only for legacy history); §5 #7 (`set_field_by_path(..., validate=True)`
> rejects out-of-constraint values, and `validate_spec` now validates an llm step's inlined
> `llm_config` so an out-of-range `temperature`/`timeout` is caught pre-run); §5 #8
> (`EvolutionCostEstimate` is model-pool aware — `per_chain_cost_by_model` +
> `total_cost_usd_upper` so `MODEL_SWAP` children onto pricier models aren't under-estimated);
> §5 #9 (`expand_field_path` expands `*` wildcard reference paths); §5 #10 kept as the
> `test_every_catalog_example_validates` guard (the plan's sanctioned option); §5 #11 is
> a non-issue (`run_descriptor` never dereferences `result.trace`).

> **The one thing to read first.** In CARL's dataset-evaluation path,
> `CaseEvaluationResult.success` means _the chain executed without raising_ — **not**
> _the answer is correct_. Correctness is **score-vs-threshold**. Any EvoC code that
> partitions cases into "correct / incorrect" off `success` will silently mis-label
> every case where the chain ran but produced a wrong answer. Addition **A5**
> (`DatasetEvaluationReport.partition()`) makes correct/incorrect a first-class,
> score-derived concept.

---

## 0. Design principle — the CARL ⟷ EvoC boundary

CARL is the **substrate of primitives**; EvoC is the **orchestration policy**. Keeping
this line sharp is what lets EvoC evolve without re-deriving graph algorithms and lets
CARL stay reusable.

| Concern                                                                                                                   | Owner                                             |
| ------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------- |
| Step-type meta (catalog, schemas, field choices/constraints)                                                              | **CARL** (`mmar_carl.meta`)                       |
| No-execution validation of a chain spec                                                                                   | **CARL** (`validate_spec` / `validate_and_build`) |
| Structural + content **mutation primitives** (rewire, retype, insert-at, delete-with-healing, constrained value sampling) | **CARL** (to add — A1/A2)                         |
| Run **descriptors** (quality / cost / time / complexity / fault-tolerance)                                                | **CARL** (to add — A3)                            |
| Dataset **subset / partition / cross-run diff**                                                                           | **CARL** (partly exists; to extend — A4/A5)       |
| **Freeze** a step                                                                                                         | **CARL** (to add — A9)                            |
| Replayable **MutationRecord** + stable lineage IDs                                                                        | **CARL** (to add — A1/A7)                         |
| N-dimensional **Pareto** kernel                                                                                           | **CARL** (to add — A8)                            |
| EDA, the **map-elites loop**, niche archive                                                                               | **EvoC**                                          |
| **Concept library** content (domain knowledge)                                                                            | **EvoC** (CARL only ships the schema — A6)        |
| **Cascade-validation policy** (how many mutants, promotion thresholds)                                                    | **EvoC**                                          |
| **Decision-tree** learning / cross-experiment reuse strategy                                                              | **EvoC**                                          |
| **LLM-as-a-judge** fault-tolerance scoring                                                                                | **EvoC** (CARL's descriptor accepts the float)    |

Legend used below: **✓** CARL provides it today · **⊕** CARL addition required (see §4) ·
**▹** EvoC owns it.

---

## 1. The algorithm, extended

### Phase I — Opening (Debut)

| Step                                 | CARL support                                                                                                                                                                                                                                                                                                                       |
| ------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Analyze input data & summarize (EDA) | ▹ EvoC. CARL provides `DataCase` / `AbstractDataset` / `SimpleDataset` / `DataFrameDataset` ✓ and `subsample` / `split` ⊕ (A4) so EvoC's EDA and later cascade share one dataset API.                                                                                                                                              |
| Generate the initial chain **v0**    | ✓ `ChainBuilder.from_description(task, llm_client, available_tools=…, available_skills=…)` — already renders the **auto-updating** planner prompt from `StepCatalog.generation_prompt` (covers all 16 step types). ⊕ Accept a structured `capability_library` so tool/skill/concept knowledge arrives as data, not free-text (A6). |
| Explore external context             | ▹ EvoC performs the research. ⊕ CARL accepts the result as a `CapabilityLibrary` payload for generation/mutation (A6).                                                                                                                                                                                                             |
| Primary run of v0                    | ✓ `DatasetEvaluator.evaluate_async(chain, dataset)` → `DatasetEvaluationReport`; per-case `CaseEvaluationResult` carries `score`, `chain_output`, `success`, `token_usage`, `execution_time`, `step_outcomes`. ⊕ `extract_run_descriptor(...)` to package this into the Pareto/niche vector (A3).                                  |

### Phase II — Middlegame (the evolution loop)

**Analyze the last run — correct vs incorrect.**

- ▹ Summarization (LLM) is EvoC's.
- ⊕ **A5** makes the split first-class on the very report the evolver produces:
  `report.partition(threshold=…)` → `CaseSplit(correct=[…], incorrect=[…])`, and
  `report.by_label()` for keyed access. `CaseEvaluationResult` already exposes the
  inputs/outputs/expected EvoC needs to summarize. **Do not** derive correctness from
  `success` (see banner).

**Generate mutations** from three libraries:

- _(a) tool / AgentSkill library_ → ✓ `SkillLoader.catalog_all(payloads=True)` enumerates
  skills as data; the context tool registry lists tools. ⊕ `CapabilityLibrary` +
  `SkillLoader.as_capability_library(...)` unify these into one typed, token-budgeted
  payload (A6).
- _(b) CARL step-type library_ → ✓ `mmar_carl.meta` — `get_step_catalog()`, tiers,
  `schema_for`, `generation_prompt`. Fully auto-updating.
- _(c) concept library_ → ▹ EvoC-owned content; ⊕ carried as `CapabilityEntry(kind="concept")` (A6).

_Topology-level decisions_ (which step types, how to wire the graph):

| Decision                 | CARL support                                                                                                                                                                                                                                                          |
| ------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Add a new step           | ✓ `auto_generate_minimal_step(type, …)` builds a valid step; ⊕ `insert_step_at(spec, step, after=/before=, rewire_dependents=…)` for arbitrary-position insertion with healing (A1). Today `ChainMutator` only inserts _before the last step_ as a verification gate. |
| Remove an existing step  | ✓ `find_leaf_steps` / `can_delete_step` (leaf-only); ⊕ `remove_step(spec, n, heal=True)` — non-leaf removal that redirects dependents through the deleted step's own dependencies (A1).                                                                               |
| Change dependencies      | ⊕ `rewire_dependencies(spec, n, new_deps, heal=…)` — validates existence + acyclicity (A1). _Missing today._                                                                                                                                                          |
| (?) Change a step's type | ⊕ `change_step_type(spec, n, new_type, field_map=…)` — morph A→B, copy compatible fields, drop the rest (A1). _Missing today._                                                                                                                                        |

_Content-level decisions_ (change fields in a step):

- ✓ `mutable_fields(type)` (safe-to-mutate paths), `get_field_by_path` / `set_field_by_path`
  / `iter_mutable_values`, and `FieldSpec.choices` / `constraints` / `nested_fields`.
- ⊕ **A2** adds the missing piece: a _constraint-aware value sampler_
  (`sample_field_value(...)`, `mutate_content(...)`) so a mutation produces a valid value
  (temperature in `[0,2]`, an enum ≠ current) instead of relying on `set_field_by_path`,
  which is deliberately "dumb" and does **not** validate.

**Validate the new mutations** → ✓ `validate_spec(spec)` / `validate_and_build(spec)`
(no execution; structure + dependency resolution + config validity + `$memory` warnings).
Note the loop-edge gap in §5.

**Advance via map-elites (or another algorithm)** → ▹ the loop/archive is EvoC's.
CARL supplies the inputs: ⊕ `RunDescriptor.behavior_key(bins)` for niche keys and
⊕ `pareto_front(...)` for selection (A3/A8). `ChainEvolver` (population + elitism) stays
available as an alternative driver but is **not** map-elites.

**Save successful decisions into a decision tree; reuse across experiments** →
▹ the learning strategy is EvoC's. ⊕ CARL provides the substrate: a serializable
`MutationRecord` (op + step + path + before/after) with `apply_records(spec, records)`
for replay (A1), and stable individual IDs + `parent_id` + `mutation_records` on
`IndividualMetrics` (A7) so lineage is exact rather than score-matched.

### Phase III — Endgame (Pareto front over product metrics)

| Objective                                              | CARL support                                                                                                                                                       |
| ------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Quality** (target metric)                            | ✓ `MetricBase` (step + chain level, incl. LLM-judge); `report.mean_score`; `RunDescriptor.quality` (A3).                                                           |
| **Cost** (tokens/run)                                  | ✓ `ReasoningResult.token_usage`, `token_usage_by_step`; `CaseEvaluationResult.token_usage`; `RunDescriptor.total_tokens` (A3).                                     |
| **Time** (run time)                                    | ✓ `execution_time`, `ExecutionTrace`, `TraceAggregator`; `RunDescriptor.wall_time_s` (A3).                                                                         |
| **Complexity** (graph config, #steps)                  | ⊕ `ComplexityDescriptor` from `chain.get_execution_plan()` — `num_steps`, `num_edges`, `max_depth`, `max_parallel_batch`, `distinct_step_types`, `has_loops` (A3). |
| **Fault tolerance** (exceptional scenarios, LLM-judge) | ✓ `ReasoningResult.get_failed_steps()` + `step_outcomes`; ⊕ `RunDescriptor.fault_tolerance` accepts an EvoC-supplied judge float (A3).                             |
| Pareto computation                                     | ⊕ `pareto_front(points, senses)` / `pareto_rank(...)` — N-D dominance (A8). Today `EvolutionResult.format_pareto` is **2D only** (cost vs score).                  |

### Cross-cutting requirements ("what we want")

**Freeze a step / N steps — evolve only part of the pipeline.**
⊕ **A9**: a `frozen: bool` field on `StepDescriptionBase` (serialized through the existing
`_common_step_kwargs` path), `frozen_step_numbers(spec)`, and a `respect_frozen=True`
guard in every mutation primitive (A1/A2) and in `ChainMutator`. _No such concept exists today._

**Cascade validation (subsample → promote → robustness).**
▹ policy is EvoC's. ⊕ CARL primitives:

- `subsample(dataset, k)` / `split(dataset, ratio)` / `select_cases(dataset, labels)` (A4) —
  run many mutants on a cheap subset, then re-run survivors on a _specific_ case set.
- `report.partition(threshold)` (A5) — the "previously-worse" case set is `split.incorrect`.
- `report_new.diff_by_case(report_old, threshold)` → `CaseDiff(regressed, improved, unchanged)`
  keyed by `DataCase.label` (A5) — the robustness check ("did quality drop on
  previously-good cases?") becomes one call.
- `DatasetEvaluator(..., max_concurrent=k)` (A4) — bounded parallel per-case eval for speed.

---

## 2. Requirement readiness matrix (R1–R16)

| Req                            | Status               | Where in CARL                                                        | Gap → addition                                          |
| ------------------------------ | -------------------- | -------------------------------------------------------------------- | ------------------------------------------------------- |
| R1 EDA / dataset abstractions  | partial              | `models/dataset.py`                                                  | sampling/split/select → **A4**                          |
| R2 Generate v0                 | **exists**           | `ChainBuilder.from_description`, `meta.generation_prompt`            | structured capabilities → A6                            |
| R3 External-context hooks      | missing              | —                                                                    | structured `capability_library` → **A6**                |
| R4 Primary run + eval          | **exists**           | `DatasetEvaluator.evaluate_async`                                    | (concurrency knob → A4)                                 |
| R5 Correct/incorrect partition | partial              | `CaseEvaluationResult.score`, `all_results`                          | `success ≠ correct`; `partition()` → **A5**             |
| R6 Access/summarize subsets    | partial              | report formatters (display-only)                                     | queryable `CaseSplit` → **A5**                          |
| R7a Tool/skill library as data | partial              | `SkillLoader.catalog_all`, context tool registry                     | unify → **A6**                                          |
| R7b Step-type catalog          | **exists**           | `mmar_carl.meta`                                                     | —                                                       |
| R7c Concept library            | missing (EvoC-owned) | —                                                                    | schema only → A6                                        |
| R8 Topology mutations          | partial              | `ChainMutator` insert-before-last / delete-leaf; `meta` leaf helpers | rewire / retype / insert-at / delete-with-heal → **A1** |
| R9 Content mutations           | partial              | `meta` field-path + choices/constraints                              | constraint-aware sampler → **A2**                       |
| R10 Validate mutants           | **exists**           | `validate_spec` / `validate_and_build`                               | (loop-edge check → §5)                                  |
| R11 map-elites / niches        | missing              | `ChainEvolver` = population+elitism                                  | behavior descriptor → **A3** (loop is EvoC's)           |
| R12 Lineage / reusable ops     | partial              | `IndividualMetrics.mutation_kind/parent_score`, `to_lineage_mermaid` | `MutationRecord` + stable IDs → **A1/A7**               |
| R13 Run descriptors            | partial              | `results.py`, `chain.get_execution_plan`, `IndividualMetrics`        | unified `RunDescriptor` → **A3**                        |
| R14 Pareto front + viz         | partial              | `format_pareto` (2D)                                                 | N-D kernel → **A8**                                     |
| R15 Freeze steps               | missing              | —                                                                    | `frozen` field + guards → **A9**                        |
| R16 Cascade validation         | missing              | full-dataset only; `EvalSuiteDiff` on the _other_ stack              | subset/select + `diff_by_case` → **A4/A5**              |

**Overall readiness ≈ 60%.** Strong meta + validation + population GA; the four P0 gaps
(A1, A2, A3, A4/A5) are the primitives EvoC would otherwise be forced to hand-roll.

---

## 3. CARL ⟷ EvoC split — the non-goals

These stay in **EvoC** and must **not** be added to CARL:

- EDA proper (class-distribution profiling, input-length stats, embedding clustering).
  CARL ships only `subsample` / `split` / `select_cases`.
- The map-elites loop itself — archive management, niche-replacement, illumination.
  CARL supplies the behavior-descriptor extractor and the N-D Pareto kernel.
- Concept-library **content**. CARL owns only the `CapabilityLibrary` schema + rendering.
- Cascade-validation **policy** (counts, promotion thresholds, rounds).
- Decision-tree **strategy** (which mutations to prefer, cross-experiment reuse heuristics).
  CARL ships `MutationRecord` / `apply_records` + stable IDs.
- LLM-as-a-judge fault-tolerance **scoring**. `RunDescriptor.fault_tolerance` takes the float.
- A replacement evolutionary algorithm. `ChainEvolver` stays as-is; EvoC brings its own loop.
- External research / web-fetch execution.

---

## 4. CARL additions required

Every addition is **keyword-additive, backward-compatible, and introspection-driven**
(no per-step-type maintenance burden). Priority: **P0** = blocks EvoC, **P1** = closes a
requirement, **P2** = ergonomics.

### P0

**A1 — Topology mutation primitives + `MutationRecord`** — `src/mmar_carl/meta/mutations.py`
(pure dict-spec transforms, no `ReasoningChain` import; mirror `get_field_by_path` style).
Reqs R8, R12. Effort **L**.

```python
@dataclass
class MutationRecord:
    op: str                     # "rewire" | "retype" | "insert" | "remove" | "set_field"
    step_number: int | None
    path: str | None
    before: Any
    after: Any
    new_step_type: str | None = None

class MutationError(ValueError):        # carries the failing ValidationReport
    report: "ValidationReport | None"

# each op mutates a COPY, validates it, returns (new_spec, record); raises MutationError if illegal
def rewire_dependencies(spec, step_number, new_deps, *, heal=True, respect_frozen=True) -> tuple[dict, MutationRecord]
def change_step_type(spec, step_number, new_type, *, field_map=None, respect_frozen=True) -> tuple[dict, MutationRecord]
def insert_step_at(spec, new_step, *, after=None, before=None, inherit_deps=True, rewire_dependents=False) -> tuple[dict, MutationRecord]
def remove_step(spec, step_number, *, heal=True, respect_frozen=True) -> tuple[dict, MutationRecord]
def apply_records(spec, records: list[MutationRecord]) -> dict     # replay for decision-tree reuse
```

**A2 — Constraint-aware content-value sampler** — same module. Reqs R9. Effort **M**.

```python
def sample_field_value(step_type, path, *, current=None, rng, mode="random") -> Any   # respects choices / ge-le / min_length
def mutate_content(spec, step_number, path, *, rng, mode="perturb") -> tuple[dict, MutationRecord]
def mutable_targets(spec) -> list[tuple[int, str, Any]]           # (step_number, path, current) across ALL steps
```

**A3 — `RunDescriptor` + `ComplexityDescriptor`** — `src/mmar_carl/run_descriptor.py`.
Reqs R13, R11. Effort **M**.

```python
class ComplexityDescriptor(BaseModel):
    num_steps: int; num_edges: int; max_depth: int
    max_parallel_batch: int; distinct_step_types: int; has_loops: bool

class RunDescriptor(BaseModel):
    quality: float | None; total_tokens: int; wall_time_s: float; llm_calls: int
    complexity: ComplexityDescriptor; fault_tolerance: float | None
    scores_by_metric: dict[str, float] = {}
    def behavior_key(self, bins: dict[str, list[float|int]]) -> tuple      # map-elites niche key (EvoC picks bins)
    def as_objectives(self, keys: list[str]) -> tuple[float, ...]          # -> A8 Pareto

def extract_run_descriptor(result, chain, *, quality=None, scores_by_metric=None, fault_tolerance=None) -> RunDescriptor
def extract_complexity(chain) -> ComplexityDescriptor                      # structural, no run needed
```

**A4 — Dataset subsample / partition helpers** — `src/mmar_carl/models/dataset.py`
(+ optional `DatasetEvaluator(max_concurrent=…)`). Reqs R16, R1. Effort **S**.

```python
def subsample(dataset, k, *, rng, strategy="random", stratify_key=None) -> SimpleDataset
def split(dataset, ratio, *, rng) -> tuple[SimpleDataset, SimpleDataset]
def select_cases(dataset, labels: set[str]) -> SimpleDataset               # rebuild from case labels
```

**A5 — `CorrectIncorrectSplit` + cross-run per-case diff** — methods on
`DatasetEvaluationReport` (the same object `ChainEvolver` produces). Reqs R5, R6, R16.
Effort **M**.

```python
class CaseSplit(BaseModel):
    correct: list[CaseEvaluationResult]; incorrect: list[CaseEvaluationResult]
    def labels(self) -> tuple[set[str], set[str]]

class CaseDiff(BaseModel):
    regressed: list[str]; improved: list[str]; unchanged: list[str]
    score_delta_by_label: dict[str, float]

# on DatasetEvaluationReport:
def partition(self, *, correct=None, threshold=None) -> CaseSplit          # default: score >= threshold
def diff_by_case(self, other, *, threshold=None) -> CaseDiff               # matched by DataCase.label
def by_label(self) -> dict[str, CaseEvaluationResult]
```

### P1

**A6 — Structured tool/skill/concept `CapabilityLibrary`** — `meta/models.py`; accepted by
`from_description` and `generation_prompt`. Reqs R7a, R7c, R3, R2. Effort **M**.

```python
class CapabilityEntry(BaseModel):
    name: str; kind: Literal["tool","skill","concept"]; description: str
    when_to_use: str | None = None; example: str | None = None
class CapabilityLibrary(BaseModel):
    entries: list[CapabilityEntry]
    def render(self, *, max_tokens=None) -> str
# StepCatalog.generation_prompt(..., capability_library=None)
# ChainBuilder.from_description(..., capability_library=None)   # additive; existing lists still work
# SkillLoader.as_capability_library(search_paths=…) -> CapabilityLibrary
```

**A7 — Stable individual IDs + explicit parentage** — `chain_evolution.py`. Reqs R12. Effort **S**.

```python
IndividualMetrics.id: str | None            # uuid4 hex, assigned at creation
IndividualMetrics.parent_id: str | None
IndividualMetrics.mutation_records: list[MutationRecord] = []
# to_lineage_mermaid: use exact ids when present, fall back to score-matching only for old data
```

**A8 — N-dimensional Pareto kernel** — `src/mmar_carl/pareto.py` (pure, numpy-free).
Reqs R14, R13. Effort **S**.

```python
def dominates(a, b, senses) -> bool
def pareto_front(points, *, senses) -> list[int]     # indices of non-dominated points
def pareto_rank(points, senses) -> list[int]         # NSGA-style front index per point
```

**A9 — Frozen / protected-step support** — `models/steps.py` + mutator guards. Reqs R15. Effort **S**.

```python
StepDescriptionBase.frozen: bool = False             # serialized via existing _common_step_kwargs
def frozen_step_numbers(spec) -> set[int]
def deletable_steps(spec, *, respect_frozen=True) -> set[int]
# A1/A2 ops + ChainMutator accept respect_frozen=True (default) and raise/skip on frozen targets
```

### P2

**A10 — DAG query helpers** — `meta/mutations.py` (companions to `find_leaf_steps`).
Reqs R8, R9. Effort **S**.

```python
def find_root_steps(steps) -> set[int]               # no dependencies
def steps_of_type(steps, step_type) -> list[int]
def dependents_of(steps, step_number) -> set[int]    # inverse dependency edges (used by A1 healing)
```

---

## 5. Inconsistencies & fixes (things that will surprise an EvoC author)

1. **`success` ≠ correct** (`dataset.py`, `CaseEvaluationResult.success`). Means "chain
   executed", not "answer correct". → Fixed conceptually by **A5** `partition()`
   (correctness is score-vs-threshold) + a docstring/banner. **Highest priority.**
2. **Two non-interoperating eval stacks.** Cross-run per-case regression exists only on the
   `EvalSuite` / `EvalSuiteDiff` side, while `ChainEvolver` drives the `DatasetEvaluator` /
   `DatasetEvaluationReport` side. → **A5** `diff_by_case` brings regression detection to the
   report EvoC actually evolves against.
3. **`INSERT_STEP` is one hard-coded pattern** — always inserts before the numeric-last step
   as a verification gate (`ChainMutator._apply_insert_step`). Not a general operator. →
   **A1** `insert_step_at`.
4. **`DELETE_STEP` is loop-unaware.** Deleting a `loop_back_to` target or loop tail silently
   yields a broken loop that `validate_spec` does **not** catch (it checks numeric
   dependencies, not loop edges). → **A1** `remove_step` heals loop edges; extend
   `validate_spec` to flag dangling `loop_back_to`.
5. **Lineage by score-tolerance matching** (`to_lineage_mermaid`) is ambiguous on ties;
   `IndividualMetrics` has no stable id. → **A7** stable ids + explicit `parent_id`.
6. **`IndividualMetrics.mutation_kind` stored as a bare string**, not the `MutationKind`
   enum. Adding a kind later means old checkpoints compare unequal to enum members. →
   normalize on load, or store the enum.
7. **`set_field_by_path` is unguarded** — writes any value (e.g. `temperature=9.9`, an
   out-of-`choices` enum); invalid state only surfaces at `validate_spec`/execution. →
   **A2** constraint-aware sampler is the intended writer; optionally add a
   `validate=True` guard to `set_field_by_path`.
8. **`EvolutionCostEstimate` assumes a single model/pricing** (`pop×gens×cases ×
base_chain.estimate_cost`); systematically under-estimates multi-model chains — exactly
   what `MODEL_SWAP` produces. → per-step pricing in the estimate.
9. **`reference_fields` returns wildcard paths** (`step_config.branches.*.condition`) but
   `get_field_by_path` / `set_field_by_path` only accept concrete indices, and the `*`
   grammar is undocumented. → add wildcard expansion (`expand_wildcard(spec, path) ->
list[str]`) or document that callers must expand `*` against current list length.
10. **`auto_generate_minimal_step` curated examples aren't round-trip-validated inside the
    catalog build** (a drifted curated `example` would ship a step that fails
    `validate_spec`). Currently guarded only by a test
    (`test_every_catalog_example_validates`). → add a build-time assertion, or keep the
    test as the guard and document it.
11. **`ReasoningResult.trace` is `Optional[Any]`** (duck-typed via `hasattr(trace,
"to_dict")`). A descriptor extractor (A3) must treat it as possibly `None`.
12. **`ChainMutator` uses an unseeded per-call shuffle** for failure recovery, so which
    mutation lands is non-deterministic even under a seeded global rng — a reproducibility
    surprise for a seeded search. → thread the run rng through the shuffle.

---

## 6. Suggested sequencing

1. **Milestone 0 — unblock EvoC bring-up (P0):** A4 (dataset subset/split/select),
   A5 (partition + diff_by_case), A3 (RunDescriptor). These let EvoC run v0, split
   correct/incorrect, and get a Pareto/niche vector — enough to stand up the loop with the
   _existing_ `ChainMutator`.
2. **Milestone 1 — real mutation surface (P0):** A1 (topology ops + MutationRecord),
   A2 (constraint-aware content sampler), A10 (DAG queries). Now EvoC can do full
   topology + content mutation with healing and replayable records.
3. **Milestone 2 — evolution quality (P1):** A8 (N-D Pareto), A9 (frozen steps),
   A7 (stable lineage IDs), A6 (capability library).
4. **Milestone 3 — polish:** the §5 inconsistency fixes (loop-aware delete + validation,
   cost estimate per-model, deterministic mutator, wildcard expansion).

Each addition ships with hermetic tests and, where it touches the serialized catalog,
a `make meta-snapshot` refresh (the drift guard enforces it).

---

## 7. Appendix — key CARL entry points

| Concern                            | Symbol / file                                                                                                                                                                          |
| ---------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Step-type catalog                  | `mmar_carl.meta.get_step_catalog`, `StepCatalog` (`meta/models.py`)                                                                                                                    |
| Generation prompt                  | `StepCatalog.generation_prompt`; `ChainBuilder.from_description` (`chain.py`)                                                                                                          |
| Validation                         | `validate_spec`, `validate_and_build` (`meta/catalog.py`)                                                                                                                              |
| Field-level mutation               | `mutable_fields`, `get_field_by_path`, `set_field_by_path`, `iter_mutable_values` (`meta/catalog.py`); `FieldSpec.choices/constraints/nested_fields` (`meta/models.py`)                |
| DAG safety                         | `find_leaf_steps`, `can_delete_step` (`meta/catalog.py`)                                                                                                                               |
| Evolution loop                     | `ChainEvolver`, `ChainMutator`, `EvolutionResult`, `IndividualMetrics`, `MutationKind` (`chain_evolution.py`)                                                                          |
| Dataset eval                       | `DatasetEvaluator` (`dataset_evaluator.py`); `DatasetEvaluationReport`, `CaseEvaluationResult`, `DataCase` (`models/dataset.py`)                                                       |
| Cross-run regression (other stack) | `EvalSuite`, `EvalSuiteDiff` (`eval_suite.py`)                                                                                                                                         |
| Run metrics                        | `ReasoningResult.token_usage` / `token_usage_by_step` / `get_profiling_summary` / `get_failed_steps` (`models/results.py`); `ExecutionTrace`, `TraceAggregator` (`execution_trace.py`) |
| Chain structure                    | `chain.get_execution_plan` / `get_step_dependencies` / `get_steps_summary` (`chain.py`)                                                                                                |
| Skills/tools as data               | `SkillLoader.catalog_all(payloads=True)` (`skill_loader.py`); context tool registry (`models/context.py`)                                                                              |

---

_This document is a preparation spec. The A-items are CARL-side, additive, and
back-compatible; the EvoC repository consumes them plus the existing surface above. See
`docs/meta/HOW_TO_USE.md` for the step-type meta / validation / mutation-primitive API that
already exists today._
