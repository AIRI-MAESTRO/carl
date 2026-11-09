"""Structural + content mutation primitives for CARL chain specs.

These operate on plain chain-serialization dicts (``{"steps": [...]}``) — no
``ReasoningChain`` import, no execution — so an evolution engine (e.g. EvoC) can
mutate a chain and validate it cheaply. Every structural op mutates a *copy*,
runs :func:`validate_spec`, and returns ``(new_spec, MutationRecord)`` or raises
:class:`MutationError` (with the failing report attached) so the caller can try
the next candidate.

Primitives:

- **topology** — :func:`rewire_dependencies`, :func:`change_step_type`,
  :func:`insert_step_at`, :func:`remove_step` (dependency + loop-edge healing),
  :func:`apply_records`.
- **content** — :func:`sample_field_value` (constraint-aware), :func:`mutate_content`,
  :func:`mutable_targets`.
- **DAG queries** — :func:`find_root_steps`, :func:`find_leaf_steps`,
  :func:`dependents_of`, :func:`steps_of_type`, :func:`deletable_steps`,
  :func:`frozen_step_numbers`, :func:`expand_field_path`.

All ops accept ``respect_frozen=True`` (default) and refuse to touch a step whose
``frozen`` flag is set — this is how a caller freezes part of a pipeline.
"""

from __future__ import annotations

import copy
import random
from typing import Any, Optional

from .catalog import (
    auto_generate_minimal_step,
    find_leaf_steps,
    get_field_by_path,
    get_step_catalog,
    mutable_fields,
    set_field_by_path,
    validate_spec,
)
from .models import FieldSpec, MutationRecord, ValidationReport

_UNSET = object()


class MutationError(ValueError):
    """A mutation could not be applied (illegal target, frozen step, or the
    result failed validation). ``report`` holds the failing
    :class:`ValidationReport` when the cause was validation."""

    def __init__(self, message: str, *, report: Optional[ValidationReport] = None) -> None:
        super().__init__(message)
        self.report = report


# ---------------------------------------------------------------------------
# Spec/step helpers.
# ---------------------------------------------------------------------------


def _steps(spec: Any) -> list[dict[str, Any]]:
    if isinstance(spec, dict) and "steps" in spec:
        return spec["steps"]
    if isinstance(spec, list):
        return spec
    return [spec]


def _find(steps: list[dict[str, Any]], number: int) -> Optional[dict[str, Any]]:
    for s in steps:
        if s.get("number") == number:
            return s
    return None


def _numbers(steps: list[dict[str, Any]]) -> list[int]:
    return [s.get("number") for s in steps if s.get("number") is not None]


def _require_valid(spec: dict[str, Any], op: str) -> None:
    report = validate_spec(spec)
    if not report.valid:
        raise MutationError(f"{op} produced an invalid spec:\n{report.format_text()}", report=report)


def _step_type(step: dict[str, Any]) -> str:
    st = step.get("step_type", "llm")
    return st.value if hasattr(st, "value") else st


# ---------------------------------------------------------------------------
# DAG queries (A10) + frozen helpers (A9).
# ---------------------------------------------------------------------------


def find_root_steps(steps: list[dict[str, Any]]) -> set[int]:
    """Step numbers with no dependencies (chain entry points)."""
    return {s["number"] for s in steps if s.get("number") is not None and not (s.get("dependencies") or [])}


def dependents_of(steps: list[dict[str, Any]], step_number: int) -> set[int]:
    """Step numbers that depend (directly) on ``step_number`` — inverse edges."""
    return {s["number"] for s in steps if step_number in (s.get("dependencies") or [])}


def steps_of_type(steps: list[dict[str, Any]], step_type: str) -> list[int]:
    """Step numbers whose step_type is ``step_type`` (in list order)."""
    return [s["number"] for s in steps if _step_type(s) == step_type]


def frozen_step_numbers(steps: list[dict[str, Any]]) -> set[int]:
    """Step numbers marked ``frozen`` (must not be mutated/deleted)."""
    return {s["number"] for s in steps if s.get("frozen")}


def deletable_steps(steps: list[dict[str, Any]], *, respect_frozen: bool = True) -> set[int]:
    """Leaf steps that may be safely deleted (≥1 step remains; not frozen)."""
    if len(steps) < 2:
        return set()
    out = find_leaf_steps(steps)
    if respect_frozen:
        out -= frozen_step_numbers(steps)
    return out


def expand_field_path(step: dict[str, Any], path: str) -> list[str]:
    """Expand ``*`` segments in a field path against the current list lengths.

    e.g. ``'step_config.branches.*.condition'`` -> ``['step_config.branches.0.condition',
    ...]`` for however many branches exist. Paths without ``*`` return ``[path]``.
    """
    if "*" not in path:
        return [path]
    segs = path.split(".")
    out_paths: list[list[str]] = [[]]
    values: list[Any] = [step]
    for seg in segs:
        new_paths: list[list[str]] = []
        new_values: list[Any] = []
        for p, v in zip(out_paths, values):
            if seg == "*":
                seq = v if isinstance(v, (list, tuple)) else []
                for i, item in enumerate(seq):
                    new_paths.append(p + [str(i)])
                    new_values.append(item)
            else:
                nv = _UNSET
                if isinstance(v, dict) and seg in v:
                    nv = v[seg]
                new_paths.append(p + [seg])
                new_values.append(nv)
        out_paths, values = new_paths, new_values
    return [".".join(p) for p in out_paths]


# ---------------------------------------------------------------------------
# Topology mutations (A1).
# ---------------------------------------------------------------------------


def _guard_frozen(step: dict[str, Any], respect_frozen: bool, op: str) -> None:
    if respect_frozen and step.get("frozen"):
        raise MutationError(f"{op}: step {step.get('number')} is frozen")


def rewire_dependencies(
    spec: dict[str, Any], step_number: int, new_deps: list[int], *, respect_frozen: bool = True
) -> tuple[dict[str, Any], MutationRecord]:
    """Replace a step's dependency list. Validates existence + acyclicity."""
    new = copy.deepcopy(spec)
    steps = _steps(new)
    step = _find(steps, step_number)
    if step is None:
        raise MutationError(f"rewire_dependencies: no step {step_number}")
    _guard_frozen(step, respect_frozen, "rewire_dependencies")
    before = list(step.get("dependencies") or [])
    nums = set(_numbers(steps))
    for d in new_deps:
        if d not in nums:
            raise MutationError(f"rewire_dependencies: dependency {d} does not exist")
        if d == step_number:
            raise MutationError("rewire_dependencies: a step cannot depend on itself")
    step["dependencies"] = list(new_deps)
    _require_valid(new, "rewire_dependencies")
    return new, MutationRecord(
        op="rewire", step_number=step_number, path="dependencies", before=before, after=list(new_deps)
    )


def change_step_type(
    spec: dict[str, Any],
    step_number: int,
    new_type: str,
    *,
    field_map: Optional[dict[str, str]] = None,
    respect_frozen: bool = True,
) -> tuple[dict[str, Any], MutationRecord]:
    """Morph a step to ``new_type``, keeping common fields (number/title/deps/
    frozen) and any ``field_map``'d config values; incompatible config is dropped."""
    if new_type not in get_step_catalog().names():
        raise MutationError(f"change_step_type: unknown step type {new_type!r}")
    new = copy.deepcopy(spec)
    steps = _steps(new)
    old = _find(steps, step_number)
    if old is None:
        raise MutationError(f"change_step_type: no step {step_number}")
    _guard_frozen(old, respect_frozen, "change_step_type")
    old_type = _step_type(old)

    fresh = auto_generate_minimal_step(
        new_type,
        number=step_number,
        title=old.get("title"),
        dependencies=list(old.get("dependencies") or []),
    )
    # carry common control fields
    for key in ("checkpoint", "checkpoint_name", "replan_enabled", "frozen", "triggered_by"):
        if old.get(key) is not None:
            fresh[key] = old[key]
    # carry mapped config values
    for src_path, dst_path in (field_map or {}).items():
        val = get_field_by_path(old, src_path, default=_UNSET)
        if val is not _UNSET:
            set_field_by_path(fresh, dst_path, val)

    idx = steps.index(old)
    steps[idx] = fresh
    _require_valid(new, "change_step_type")
    return new, MutationRecord(
        op="retype", step_number=step_number, before=old_type, after=fresh, new_step_type=new_type
    )


def insert_step_at(
    spec: dict[str, Any],
    new_step: dict[str, Any],
    *,
    after: Optional[int] = None,
    before: Optional[int] = None,
    inherit_deps: bool = True,
    rewire_dependents: bool = False,
    respect_frozen: bool = True,
) -> tuple[dict[str, Any], MutationRecord]:
    """Insert ``new_step`` at an arbitrary position.

    - ``after=n``: the new step depends on ``n``. With ``rewire_dependents``,
      steps that depended on ``n`` are redirected through the new step (splice).
    - ``before=n``: the new step inherits ``n``'s dependencies and ``n`` is
      rewired to depend on the new step (a gate before ``n``).
    - neither: appended as an isolated step (keeps its own ``dependencies``).
    """
    if after is not None and before is not None:
        raise MutationError("insert_step_at: pass at most one of after/before")
    new = copy.deepcopy(spec)
    steps = _steps(new)
    nums = _numbers(steps)
    new_num = (max(nums) + 1) if nums else 1
    step = copy.deepcopy(new_step)
    step["number"] = new_num

    if after is not None:
        if after not in nums:
            raise MutationError(f"insert_step_at: 'after' step {after} does not exist")
        if inherit_deps or not step.get("dependencies"):
            step["dependencies"] = [after]
        if rewire_dependents:
            for s in steps:
                if after in (s.get("dependencies") or []):
                    if respect_frozen and s.get("frozen"):
                        raise MutationError(f"insert_step_at: would rewire frozen step {s['number']}")
                    s["dependencies"] = [new_num if d == after else d for d in s["dependencies"]]
        insert_idx = next((i + 1 for i, s in enumerate(steps) if s["number"] == after), len(steps))
    elif before is not None:
        target = _find(steps, before)
        if target is None:
            raise MutationError(f"insert_step_at: 'before' step {before} does not exist")
        if respect_frozen and target.get("frozen"):
            raise MutationError(f"insert_step_at: 'before' step {before} is frozen")
        if inherit_deps or not step.get("dependencies"):
            step["dependencies"] = list(target.get("dependencies") or [])
        target["dependencies"] = [new_num]
        insert_idx = next((i for i, s in enumerate(steps) if s["number"] == before), len(steps))
    else:
        step.setdefault("dependencies", [])
        insert_idx = len(steps)

    steps.insert(insert_idx, step)
    _require_valid(new, "insert_step_at")
    return new, MutationRecord(op="insert", step_number=new_num, after=step)


def remove_step(
    spec: dict[str, Any], step_number: int, *, heal: bool = True, respect_frozen: bool = True
) -> tuple[dict[str, Any], MutationRecord]:
    """Remove a step. With ``heal=True`` (default), dependents inherit the removed
    step's own dependencies (transitive bypass) and any loop edge pointing at it
    is cleared — so a non-leaf step can be removed without breaking the DAG."""
    new = copy.deepcopy(spec)
    steps = _steps(new)
    victim = _find(steps, step_number)
    if victim is None:
        raise MutationError(f"remove_step: no step {step_number}")
    _guard_frozen(victim, respect_frozen, "remove_step")
    if len(steps) < 2:
        raise MutationError("remove_step: refusing to remove the only step")

    victim_deps = [d for d in (victim.get("dependencies") or []) if d != step_number]
    # A frozen dependent would have to be rewired — refuse.
    if respect_frozen:
        for s in steps:
            if s.get("frozen") and step_number in (s.get("dependencies") or []):
                raise MutationError(f"remove_step: frozen step {s['number']} depends on {step_number}")

    steps.remove(victim)
    for s in steps:
        deps = s.get("dependencies") or []
        if step_number in deps:
            healed = [d for d in deps if d != step_number]
            if heal:
                for d in victim_deps:
                    if d not in healed:
                        healed.append(d)
            s["dependencies"] = healed
        # heal / break loop edges that referenced the removed step
        if s.get("loop_back_to") == step_number:
            if heal:
                s["loop_back_to"] = None
                s["loop_config"] = None
            else:
                raise MutationError(f"remove_step: step {s['number']} loops back to removed {step_number}")

    _require_valid(new, "remove_step")
    return new, MutationRecord(op="remove", step_number=step_number, before=victim)


def apply_records(spec: dict[str, Any], records: list[MutationRecord]) -> dict[str, Any]:
    """Best-effort replay of a mutation lineage onto ``spec`` (decision-tree reuse).

    Replays each record's net effect and validates once at the end. Structural
    ops are re-derived deterministically (e.g. 'remove' re-heals); 'insert' and
    'retype' replace/append the recorded resulting step.
    """
    new = copy.deepcopy(spec)
    steps = _steps(new)
    for rec in records:
        if rec.op == "set_field" and rec.step_number is not None and rec.path:
            step = _find(steps, rec.step_number)
            if step is not None:
                set_field_by_path(step, rec.path, rec.after)
        elif rec.op == "rewire" and rec.step_number is not None:
            step = _find(steps, rec.step_number)
            if step is not None:
                step["dependencies"] = list(rec.after or [])
        elif rec.op in ("retype",) and rec.step_number is not None and isinstance(rec.after, dict):
            step = _find(steps, rec.step_number)
            if step is not None:
                steps[steps.index(step)] = copy.deepcopy(rec.after)
        elif rec.op == "insert" and isinstance(rec.after, dict):
            steps.append(copy.deepcopy(rec.after))
        elif rec.op == "remove" and rec.step_number is not None:
            try:
                new = remove_step(new, rec.step_number, heal=True, respect_frozen=False)[0]
                steps = _steps(new)
            except MutationError:
                pass
        elif rec.op == "tool_code" and rec.path and isinstance(rec.after, str):
            # path = "metadata.tool_library.<tool name>"; after = the new code.
            tool_name = rec.path.rsplit(".", 1)[-1]
            try:
                new = mutate_tool_code(new, tool_name, rec.after, create=True)[0]
            except MutationError:
                pass
    _require_valid(new, "apply_records")
    return new


# ---------------------------------------------------------------------------
# Content mutation (A2) — constraint-aware value sampling.
# ---------------------------------------------------------------------------


def _resolve_fieldspec(step_type: str, path: str) -> Optional[FieldSpec]:
    """Navigate a (possibly nested / indexed) field path to its FieldSpec."""
    segs = path.split(".")
    if segs and segs[0] == "step_config":
        segs = segs[1:]
    fields = get_step_catalog().step(step_type).config_fields
    fs: Optional[FieldSpec] = None
    for seg in segs:
        if seg.isdigit() or seg == "*":
            continue  # stay on the list-of-model FieldSpec, descend into its nested_fields
        fs = next((f for f in fields if f.name == seg), None)
        if fs is None:
            return None
        fields = fs.nested_fields or []
    return fs


def _perturb_int(base: float, span: float, lo: Optional[float], hi: Optional[float], rng: random.Random) -> float:
    """±10% of a small integer domain rounds straight back to ``base`` — a
    silent no-op mutation. Step by at least ±1 and keep only in-bounds
    candidates, so a perturb always moves when the domain allows it."""
    step = max(1, round(span))
    cands = [c for c in (base - step, base + step) if (lo is None or c >= lo) and (hi is None or c <= hi)]
    return rng.choice(cands) if cands else base


def sample_field_value(
    step_type: str, path: str, *, rng: random.Random, current: Any = None, mode: str = "random"
) -> Any:
    """Sample a *valid* value for a field, honouring its choices / numeric range /
    length. ``mode='perturb'`` nudges a numeric ``current`` within +-10% (clamped;
    integers step by at least ±1 so the nudge can't silently no-op);
    ``mode='random'`` samples the whole domain. Returns ``current`` when the field
    has no known choices/constraints to sample from."""
    fs = _resolve_fieldspec(step_type, path)
    if fs is None:
        return current
    if fs.choices:
        alts = [c for c in fs.choices if c != current] or list(fs.choices)
        return rng.choice(alts)
    cons = fs.constraints or {}
    is_num = fs.type.startswith("int") or fs.type.startswith("float")
    if is_num:
        is_int = fs.type.startswith("int")
        # base to perturb around: current value, else the field default, else 1.0
        base = current if isinstance(current, (int, float)) else None
        if base is None and isinstance(fs.default, (int, float)):
            base = fs.default
        real_lo = cons.get("ge", cons.get("gt"))
        real_hi = cons.get("le", cons.get("lt"))
        lo, hi = real_lo, real_hi
        if lo is not None or hi is not None:
            # A missing bound is synthesised for `random`-mode sampling only;
            # perturb must clamp against REAL bounds, otherwise a value above
            # the synthetic window gets yanked back into it.
            if lo is None:
                lo = 0.0 if hi is None else min(0.0, hi)
            if hi is None:
                hi = lo + max(1.0, abs(lo))
            if mode == "perturb" and isinstance(base, (int, float)):
                span = (hi - lo) * 0.1 or 0.1
                if is_int:
                    val = _perturb_int(base, span, real_lo, real_hi, rng)
                else:
                    val = base + rng.uniform(-span, span)
                    val = min(max(val, lo), hi)
            else:
                val = rng.uniform(lo, hi)
                val = min(max(val, lo), hi)
        else:
            # unbounded numeric: perturb/scale around the base (always a number)
            if base is None:
                base = 1.0
            if mode == "perturb":
                span = abs(base) * 0.1 or 0.5
                val = _perturb_int(base, span, 0, None, rng) if is_int else base + rng.uniform(-span, span)
            else:
                val = base * rng.uniform(0.5, 1.5)
            val = max(val, 0.0)
        return int(round(val)) if is_int else val
    return current


def mutate_content(
    spec: dict[str, Any],
    step_number: int,
    path: str,
    *,
    rng: random.Random,
    mode: str = "perturb",
    respect_frozen: bool = True,
) -> tuple[dict[str, Any], MutationRecord]:
    """Read the current value at ``path``, sample a new valid one, set it, validate."""
    new = copy.deepcopy(spec)
    step = _find(_steps(new), step_number)
    if step is None:
        raise MutationError(f"mutate_content: no step {step_number}")
    _guard_frozen(step, respect_frozen, "mutate_content")
    if _resolve_fieldspec(_step_type(step), path) is None:
        # Without this, a typo'd path silently records a None -> None no-op
        # (sample_field_value returns `current` for unknown fields).
        raise MutationError(
            f"mutate_content: unknown field path {path!r} for step type {_step_type(step)!r} "
            f"— use mutable_targets() / expand_field_path() to enumerate valid paths"
        )
    current = get_field_by_path(step, path, default=None)
    value = sample_field_value(_step_type(step), path, rng=rng, current=current, mode=mode)
    set_field_by_path(step, path, value)
    _require_valid(new, "mutate_content")
    return new, MutationRecord(op="set_field", step_number=step_number, path=path, before=current, after=value)


def mutate_tool_code(
    spec: dict[str, Any],
    tool_name: str,
    new_code: str,
    *,
    description: str = "",
    create: bool = False,
) -> tuple[dict[str, Any], MutationRecord]:
    """Evolve a spec-carried code tool: append ``new_code`` as the tool's next
    version in ``spec["metadata"]["tool_library"]``.

    Steps that call the tool by plain name pick up the new version on the next
    run (binding registers latest under the plain name); steps pinned to
    ``"{name}@v{k}"`` are untouched. With ``create=True`` a missing tool (or
    missing library) is created instead of raising.

    Returns ``(new_spec, MutationRecord(op="tool_code"))`` — replayable via
    :func:`apply_records`.
    """
    from ..code_tools import TOOL_LIBRARY_KEY, CodeToolError, CodeToolLibrary

    new = copy.deepcopy(spec)
    lib_data = (new.get("metadata") or {}).get(TOOL_LIBRARY_KEY)
    if not lib_data and not create:
        raise MutationError("mutate_tool_code: spec carries no tool library")
    library = CodeToolLibrary.from_dict(lib_data or {})
    latest = library.get(tool_name)
    if latest is None and not create:
        raise MutationError(f"mutate_tool_code: no tool {tool_name!r} in the library")
    if latest is not None and latest.code == new_code:
        raise MutationError(f"mutate_tool_code: no-op (code unchanged for {tool_name!r})")
    try:
        library.add(tool_name, new_code, description=description)
    except CodeToolError as exc:
        raise MutationError(f"mutate_tool_code: {exc}") from exc
    library.attach_to(new)
    _require_valid(new, "mutate_tool_code")
    return new, MutationRecord(
        op="tool_code",
        path=f"metadata.{TOOL_LIBRARY_KEY}.{tool_name}",
        before=latest.code if latest is not None else None,
        after=new_code,
    )


def mutable_targets(spec: dict[str, Any], *, respect_frozen: bool = True) -> list[tuple[int, str, Any]]:
    """Whole-chain view of mutable content targets: ``(step_number, path, current)``
    for every mutable field of every (non-frozen) step."""
    out: list[tuple[int, str, Any]] = []
    for step in _steps(spec):
        if respect_frozen and step.get("frozen"):
            continue
        num = step.get("number")
        for path in mutable_fields(_step_type(step)):
            out.append((num, path, get_field_by_path(step, path, default=None)))
    return out


__all__ = [
    "MutationError",
    "MutationRecord",
    # topology
    "rewire_dependencies",
    "change_step_type",
    "insert_step_at",
    "remove_step",
    "apply_records",
    # content
    "sample_field_value",
    "mutate_content",
    "mutate_tool_code",
    "mutable_targets",
    # dag queries
    "find_root_steps",
    "find_leaf_steps",
    "dependents_of",
    "steps_of_type",
    "deletable_steps",
    "frozen_step_numbers",
    "expand_field_path",
]
