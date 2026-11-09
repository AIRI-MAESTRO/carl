"""Public functional surface of :mod:`mmar_carl.meta`.

Builds the live :class:`StepCatalog`, checks registration drift, validates step
/ chain specs without executing them, exposes per-type JSON schemas and
mutation helpers, and manages the committed snapshot used by external services
and the CI drift check.
"""

from __future__ import annotations

import contextlib
import copy
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel

from .._dependency_graph import dependency_back_edges
from ..models.enums import StepType
from ._curated import REFERENCE_FIELDS, REFERENCE_GRAMMAR, VALIDATION_RULES
from ._introspect import (
    build_step_type_spec,
    common_field_specs,
    discover_step_classes,
    executor_class_names,
)
from .models import (
    DriftReport,
    ReferenceSpec,
    StepCatalog,
    ValidationIssue,
    ValidationReport,
)

#: Committed, versioned snapshot consumed by external services + the CI drift test.
SNAPSHOT_PATH = Path(__file__).parent / "catalog_snapshot.json"

_CACHED: Optional[StepCatalog] = None


def _carl_version() -> str:
    try:
        from importlib.metadata import version

        return version("mmar-carl")
    except Exception:  # pragma: no cover - metadata may be unavailable in-tree
        return ""


# ---------------------------------------------------------------------------
# Build / access.
# ---------------------------------------------------------------------------


def build_catalog(*, generated_at: Optional[str] = None) -> StepCatalog:
    """Introspect the live models and assemble a fresh :class:`StepCatalog`."""
    desc_map = discover_step_classes()
    exec_names = executor_class_names()

    step_types = {}
    for st in StepType:  # stable enum-declaration order
        cls = desc_map.get(st)
        if cls is None:
            continue  # drift (no description class); surfaced by check_drift
        step_types[st.value] = build_step_type_spec(st, cls, exec_names.get(st, "?"))

    return StepCatalog(
        carl_version=_carl_version(),
        generated_at=generated_at,
        step_types=step_types,
        common_fields=common_field_specs(),
        references=[ReferenceSpec(**r) for r in REFERENCE_GRAMMAR],
        validation_rules=list(VALIDATION_RULES),
    )


def get_step_catalog(*, refresh: bool = False) -> StepCatalog:
    """Return the process-cached live catalog (built on first call).

    Always reflects the current code — runtime callers (validation, planner
    prompt, drift) get an accurate catalog regardless of the committed snapshot.
    """
    global _CACHED
    if _CACHED is None or refresh:
        _CACHED = build_catalog()
    return _CACHED


# ---------------------------------------------------------------------------
# Drift detection — are all the step-type registration sites in sync?
# ---------------------------------------------------------------------------


def _union_step_types() -> set[str]:
    import typing

    from ..models.steps import AnyStepDescription

    out: set[str] = set()
    for cls in typing.get_args(AnyStepDescription):
        try:
            out.add(cls.model_construct().step_type.value)
        except Exception:  # pragma: no cover
            continue
    return out


def check_drift() -> DriftReport:
    """Compare the step-type registration sites and report any divergence.

    Sites checked: the ``StepType`` enum, the executor registry
    (``_EXECUTORS``), the discovered ``*StepDescription`` subclasses, and the
    ``AnyStepDescription`` union. (Round-tripping each type through
    ``from_dict`` is covered by the dispatch test, which is too expensive for a
    runtime call.)
    """
    enum_types = {st.value for st in StepType}
    discovered = {st.value for st in discover_step_classes()}
    registry = {st.value for st in executor_class_names()}
    union = _union_step_types()

    issues: list[str] = []

    def _diff(label: str, have: set[str]) -> None:
        missing = enum_types - have
        extra = have - enum_types
        if missing:
            issues.append(f"{label} is missing: {', '.join(sorted(missing))}")
        if extra:
            issues.append(f"{label} has unknown types: {', '.join(sorted(extra))}")

    _diff("executor registry (_EXECUTORS)", registry)
    _diff("discovered *StepDescription subclasses", discovered)
    _diff("AnyStepDescription union", union)

    # Curated coverage is advisory (graceful degradation), surfaced as a note.
    from ._curated import CURATED_STEP_META

    uncurated = sorted(enum_types - set(CURATED_STEP_META))
    if uncurated:
        issues.append(f"note: step types with no curated metadata (degraded card): {', '.join(uncurated)}")

    blocking = [i for i in issues if not i.startswith("note:")]
    return DriftReport(in_sync=not blocking, step_types=sorted(enum_types), issues=issues)


# ---------------------------------------------------------------------------
# No-execution validation of a step / chain spec.
# ---------------------------------------------------------------------------


def _config_class_map() -> dict[str, Optional[type]]:
    out: dict[str, Optional[type]] = {}
    for st, cls in discover_step_classes().items():
        ann = cls.model_fields["config"].annotation if "config" in cls.model_fields else None
        out[st.value] = ann
    return out


def _iter_ref_strings(value: Any):
    """Yield every ``$...`` reference string nested anywhere in *value*."""
    if isinstance(value, str):
        if value.startswith("$"):
            yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _iter_ref_strings(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _iter_ref_strings(v)


def _validate_one_step(
    step: dict[str, Any], cfg_map: dict[str, Optional[type]], errors: list[ValidationIssue]
) -> BaseModel | None:
    num = step.get("number")
    st = step.get("step_type", "llm")
    st = st.value if hasattr(st, "value") else st

    if not isinstance(st, str) or st not in cfg_map:
        errors.append(
            ValidationIssue(
                severity="error",
                code="unknown_step_type",
                step=num,
                field="step_type",
                message=f"unknown step_type {st!r}; expected one of {', '.join(sorted(cfg_map))}",
            )
        )
        return None

    if not step.get("title"):
        errors.append(
            ValidationIssue(
                severity="error", code="missing_required_field", step=num, field="title", message="title is required"
            )
        )

    config_cls = cfg_map[st]
    if config_cls is None:  # inlined (llm): aim is validator-required
        if st == "llm" and not str(step.get("aim", "")).strip():
            errors.append(
                ValidationIssue(
                    severity="error",
                    code="missing_required_field",
                    step=num,
                    field="aim",
                    message="llm steps require a non-empty 'aim'",
                )
            )
        # The llm step inlines its config, so its `llm_config` (temperature range,
        # timeout > 0, …) would otherwise never be checked here — validate it
        # against LLMStepConfig so an out-of-range override is caught before run.
        llm_cfg = step.get("llm_config")
        if llm_cfg is not None:
            from pydantic import ValidationError

            from ..models.config import LLMStepConfig

            try:
                LLMStepConfig.model_validate(llm_cfg)
            except ValidationError as exc:
                for err in exc.errors():
                    loc = ".".join(str(p) for p in err.get("loc", ()))
                    errors.append(
                        ValidationIssue(
                            severity="error",
                            code="invalid_config",
                            step=num,
                            field=f"llm_config.{loc}" if loc else "llm_config",
                            message=err.get("msg", "invalid value"),
                        )
                    )
        return None

    data = step.get("step_config", step.get("config"))
    if data is None:
        errors.append(
            ValidationIssue(
                severity="error",
                code="missing_step_config",
                step=num,
                field="step_config",
                message=f"{st} steps require a 'step_config' object",
            )
        )
        return None

    from pydantic import ValidationError

    try:
        return config_cls.model_validate(data)
    except ValidationError as exc:
        for err in exc.errors():
            loc = ".".join(str(p) for p in err.get("loc", ()))
            errors.append(
                ValidationIssue(
                    severity="error",
                    code="invalid_config",
                    step=num,
                    field=f"step_config.{loc}" if loc else "step_config",
                    message=err.get("msg", "invalid value"),
                )
            )
        return None


def _reference_values(value: Any, path: list[str]) -> Iterator[str]:
    """Walk a declared reference field, expanding mapping and sequence wildcards."""
    if not path:
        if isinstance(value, str):
            yield value
        return
    key, *rest = path
    if key == "*":
        items = value.values() if isinstance(value, dict) else value if isinstance(value, list | tuple) else ()
        for item in items:
            yield from _reference_values(item, rest)
    elif isinstance(value, dict) and key in value:
        yield from _reference_values(value[key], rest)


def _step_result_dependency_issues(
    steps: list[dict[str, Any]], configs: list[BaseModel | None]
) -> Iterator[ValidationIssue]:
    """Require references to resolve to a declared transitive predecessor."""
    from ..step_executors import _condition_references

    dependencies = {step["number"]: step.get("dependencies") or [] for step in steps}
    producer_numbers = {str(number): number for number in dependencies}
    for step, model in zip(steps, configs, strict=True):
        step_type = step.get("step_type", "llm")
        fields = REFERENCE_FIELDS.get(step_type)
        if model is None or not fields:
            continue
        config = model.model_dump()
        if step_type == "memory" and config["operation"] not in ("write", "append"):
            continue
        number = step["number"]
        refs = [
            (field, ref, producer_numbers.get(ref.removeprefix("$steps.").split(".", 1)[0]))
            for field in fields
            for value in _reference_values(config, field.split(".")[1:])
            for ref in (
                _condition_references(value) if field == "step_config.branches.*.condition" else (value,)
            )
            if ref.startswith("$steps.") and ref != "$steps."
        ]
        if not refs:
            continue

        outstanding = {producer for _, _, producer in refs if producer is not None}
        ancestors: set[int] = set()
        pending = list(dependencies[number])
        while pending and outstanding:
            predecessor = pending.pop()
            if predecessor not in ancestors:
                ancestors.add(predecessor)
                outstanding.discard(predecessor)
                if outstanding:
                    pending.extend(dependencies[predecessor])

        for field, ref, producer in refs:
            if producer is None:
                code = "step_result_missing_producer"
                message = f"reference {ref!r} does not name a step in this chain"
            elif producer in outstanding:
                code = "step_result_without_dependency"
                message = f"reference {ref!r} reads step {producer}, which is not a dependency"
            else:
                continue
            yield ValidationIssue(severity="error", code=code, step=number, field=field, message=message)


def validate_spec(spec: Any) -> ValidationReport:
    """Validate a step or chain spec **without executing it**.

    Accepts a single step dict, a list of step dicts, or a chain dict
    (``{"steps": [...]}``). For a *single* step only that step's own
    correctness is checked (step_type, required fields, config shape) — its
    ``dependencies`` reference sibling steps that aren't present, so cross-step
    checks are meaningless. For a chain / list, structure is also validated:
    unique numbers, dependencies resolve, no cycles, explicit predecessors for
    declared step-result references, and a warning for
    ``$memory`` reads no step writes. Returns a structured
    :class:`ValidationReport`.
    """
    errors: list[ValidationIssue] = []
    warnings: list[ValidationIssue] = []

    single_step = isinstance(spec, dict) and "steps" not in spec
    if isinstance(spec, dict) and "steps" in spec:
        steps = spec["steps"]
    elif isinstance(spec, list):
        steps = spec
    else:
        steps = [spec]

    def shape_issue(field: str, message: str) -> None:
        errors.append(ValidationIssue(severity="error", code="invalid_spec", field=field, message=message))

    if not isinstance(steps, (list, tuple)):
        shape_issue("steps", "steps must be a list of step objects")
        return ValidationReport(valid=False, errors=errors)
    if not steps:
        errors.append(ValidationIssue(severity="error", code="empty", message="a chain must contain at least one step"))
        return ValidationReport(valid=False, errors=errors, warnings=warnings)

    for index, step in enumerate(steps):
        path = f"steps.{index}"
        if not isinstance(step, dict):
            shape_issue(path, "each step must be an object")
            continue
        if ("number" in step or not single_step) and type(step.get("number")) is not int:
            shape_issue(f"{path}.number", "step number must be an integer")
        parents = step.get("dependencies")
        if parents is not None and (
            not isinstance(parents, (list, tuple)) or any(type(parent) is not int for parent in parents)
        ):
            shape_issue(f"{path}.dependencies", "dependencies must be a list of step numbers")
        loop_back_to = step.get("loop_back_to")
        if loop_back_to is not None and type(loop_back_to) is not int:
            shape_issue(f"{path}.loop_back_to", "loop_back_to must be a step number")
    metadata = spec.get("metadata") if isinstance(spec, dict) else None
    if metadata is not None and not isinstance(metadata, dict):
        shape_issue("metadata", "metadata must be an object")
    if errors:
        return ValidationReport(valid=False, errors=errors)

    # Validate each step's payload before inspecting relationships.
    cfg_map = _config_class_map()
    configs = [_validate_one_step(s, cfg_map, errors) for s in steps]

    # Spec-carried code tools (metadata["tool_library"]) validate without
    # execution too: parse, right function name, no imports / dunder access.
    if metadata:
        lib_data = metadata.get("tool_library")
        if lib_data:
            from ..code_tools import CodeToolLibrary

            try:
                lib_errors = CodeToolLibrary.from_dict(lib_data).validate_all()
            except Exception as exc:  # malformed library structure
                lib_errors = [f"tool_library does not parse: {exc}"]
            for msg in lib_errors:
                errors.append(
                    ValidationIssue(
                        severity="error",
                        code="invalid_tool_code",
                        field="metadata.tool_library",
                        message=msg,
                    )
                )

    if single_step:
        # No sibling steps to resolve dependencies / detect cycles against.
        return ValidationReport(valid=not errors, errors=errors, warnings=warnings)

    # --- chain-level structural checks ---
    numbers = [s.get("number") for s in steps]
    seen: set[Any] = set()
    for n in numbers:
        if n in seen:
            errors.append(
                ValidationIssue(severity="error", code="duplicate_number", step=n, message=f"duplicate step number {n}")
            )
        seen.add(n)

    number_set = set(numbers)
    for s in steps:
        for dep in s.get("dependencies", []) or []:
            if dep not in number_set:
                errors.append(
                    ValidationIssue(
                        severity="error",
                        code="missing_dependency",
                        step=s.get("number"),
                        field="dependencies",
                        message=f"depends on non-existent step {dep}",
                    )
                )
        # loop edges must point at an existing step (a delete-without-heal can dangle one)
        lbt = s.get("loop_back_to")
        if lbt is not None and lbt not in number_set:
            errors.append(
                ValidationIssue(
                    severity="error",
                    code="dangling_loop",
                    step=s.get("number"),
                    field="loop_back_to",
                    message=f"loops back to non-existent step {lbt}",
                )
            )

    dependencies = {s.get("number"): s.get("dependencies") or [] for s in steps}
    for _, cycle_step in dependency_back_edges(dependencies):
        if cycle_step is not None:
            errors.append(
                ValidationIssue(
                    severity="error",
                    code="cycle",
                    step=cycle_step,
                    message=f"dependency cycle detected involving step {cycle_step}",
                )
            )
            break

    if not errors:
        errors.extend(_step_result_dependency_issues(steps, configs))

    # memory writers for the $memory warning
    writers: set[str] = set()
    for s in steps:
        if (s.get("step_type") == "memory") and isinstance(s.get("step_config"), dict):
            c = s["step_config"]
            if c.get("operation") in ("write", "append"):
                ns = c.get("namespace") or "default"
                writers.add(f"{ns}.{c.get('memory_key')}")

    for s in steps:
        for ref in _iter_ref_strings(s.get("step_config", {})):
            if ref.startswith("$memory."):
                tail = ref[len("$memory.") :]
                ns, _, key = tail.partition(".")
                full = f"{ns}.{key}" if key else f"default.{ns}"
                if full not in writers:
                    warnings.append(
                        ValidationIssue(
                            severity="warning",
                            code="memory_no_writer",
                            step=s.get("number"),
                            message=f"reads '{ref}' but no prior step writes it; pre-populate context.memory if intentional",
                        )
                    )

    return ValidationReport(valid=not errors, errors=errors, warnings=warnings)


# ---------------------------------------------------------------------------
# Schemas (tier 3) + mutation helpers.
# ---------------------------------------------------------------------------


def _config_class_registry() -> dict[str, type]:
    """All config classes plus their nested pydantic models, keyed by class name."""
    from ._introspect import _nested_of

    out: dict[str, type] = {}
    frontier: list[type] = []
    for cls in discover_step_classes().values():
        if "config" in cls.model_fields:
            cfg = cls.model_fields["config"].annotation
            out.setdefault(cfg.__name__, cfg)
            frontier.append(cfg)
    while frontier:
        model = frontier.pop()
        for fi in model.model_fields.values():
            nested, _ = _nested_of(fi.annotation)
            if nested is not None and nested.__name__ not in out:
                out[nested.__name__] = nested
                frontier.append(nested)
    return out


def schema_for(name: str) -> dict[str, Any]:
    """Raw JSON Schema for one step type's config, or a nested config class (tier 3, on demand).

    ``name`` may be a step type (``'tool'``) or a config-class name
    (``'MCPServerConfig'``, ``'ConditionalBranch'`` — see :func:`config_class_names`).
    For embedded step types this is the config model's schema. For the inlined
    ``llm`` type a synthetic model of the inlined fields is used (the
    description class itself can't generate a schema — its inherited ``cache``
    field carries a ``Callable``).
    """
    desc_map = {st.value: cls for st, cls in discover_step_classes().items()}
    if name in desc_map:
        cls = desc_map[name]
        if "config" in cls.model_fields:
            return cls.model_fields["config"].annotation.model_json_schema()

        # inlined (llm) -> synthesise a model from the type-specific fields only
        from pydantic import create_model

        from ..models.steps import StepDescriptionBase

        base = set(StepDescriptionBase.model_fields)
        fields = {
            n: (fi.annotation, fi)
            for n, fi in cls.model_fields.items()
            if n not in base and not getattr(fi, "exclude", False)
        }
        synth = create_model(f"{cls.__name__}Fields", **fields)  # type: ignore[call-overload]
        return synth.model_json_schema()

    registry = _config_class_registry()
    if name in registry:
        return registry[name].model_json_schema()
    raise KeyError(f"Unknown step type or config class {name!r}")


def config_class_names() -> list[str]:
    """Nested config-model class names resolvable by :func:`schema_for`."""
    return sorted(_config_class_registry())


def mutable_fields(name: str) -> list[str]:
    """Curated list of fields an evolutionary mutator may safely change."""
    return get_step_catalog().step(name).mutable_fields


def reference_fields(name: str) -> list[str]:
    """Config field paths that accept ``$...`` reference strings for ``name``.

    Symmetric with :func:`mutable_fields`; e.g. ``reference_fields('conditional')``
    -> ``['step_config.condition_context_key', 'step_config.branches.*.condition']``.
    """
    return list(get_step_catalog().step(name).reference_fields)


def auto_generate_minimal_step(
    name: str,
    *,
    number: int = 1,
    title: Optional[str] = None,
    dependencies: Optional[list[int]] = None,
) -> dict[str, Any]:
    """A minimal *valid* step spec for ``name`` — ready to splice as an
    INSERT_STEP mutation or seed a generated chain."""
    # deep copy: the catalog example is shared/cached — callers mutate the
    # returned spec (set_field_by_path etc.), which must not corrupt the catalog.
    spec = copy.deepcopy(get_step_catalog().step(name).example)
    spec["number"] = number
    if title is not None:
        spec["title"] = title
    if dependencies is not None:
        spec["dependencies"] = list(dependencies)
    return spec


# ---------------------------------------------------------------------------
# Field-path resolvers — read/write the paths that mutable_fields() returns.
# ---------------------------------------------------------------------------

_UNSET = object()


def get_field_by_path(step: dict[str, Any], path: str, *, default: Any = _UNSET) -> Any:
    """Read a value at a dotted/indexed path within a step dict.

    Paths match the :func:`mutable_fields` convention: ``'aim'`` (on the step),
    ``'step_config.temperature'`` (nested), ``'step_config.branches.0.condition'``
    (list index). Raises ``KeyError`` / ``IndexError`` when absent and no
    ``default`` is given.
    """
    cur: Any = step
    for seg in path.split("."):
        if isinstance(cur, (list, tuple)):
            try:
                cur = cur[int(seg)]
            except (ValueError, IndexError):
                if default is not _UNSET:
                    return default
                raise
        elif isinstance(cur, dict) and seg in cur:
            cur = cur[seg]
        elif default is not _UNSET:
            return default
        else:
            raise KeyError(path)
    return cur


def _write_field_by_path(step: dict[str, Any], path: str, value: Any) -> None:
    """In-place writer used by :func:`set_field_by_path` (no validation)."""
    segs = path.split(".")
    cur: Any = step
    for seg in segs[:-1]:
        if isinstance(cur, (list, tuple)):
            cur = cur[int(seg)]
        else:
            child = cur.get(seg)
            if not isinstance(child, (dict, list)):
                child = {}
                cur[seg] = child
            cur = child
    last = segs[-1]
    if isinstance(cur, list):
        cur[int(last)] = value
    else:
        cur[last] = value


def set_field_by_path(step: dict[str, Any], path: str, value: Any, *, validate: bool = False) -> None:
    """Set a value at a dotted/indexed path, creating intermediate dicts.

    Creates missing dict levels (e.g. ``step_config`` or a ``None`` ``llm_config``)
    so a mutator can set ``'step_config.llm_config.temperature'`` on a step that
    has neither yet. List indices must already exist.

    By default this writer is deliberately "dumb": it writes any value and lets
    invalid state surface later at :func:`validate_spec` / execution (the
    constraint-aware writer is :func:`mutate_content`). Pass ``validate=True`` to
    reject a value that would break the step's own config constraints (e.g.
    ``temperature=9.9``, an out-of-``choices`` enum): the write is trialled on a
    copy, checked with :func:`validate_spec` (a lone step validates itself), and
    only committed if valid — otherwise ``ValueError`` is raised and ``step`` is
    left untouched.

    Args:
        step: The step dict to mutate.
        path: Dotted/indexed :func:`mutable_fields` path.
        value: The value to write.
        validate: When True, validate the resulting step before committing.

    Raises:
        ValueError: When ``validate=True`` and the value violates the step's
            config constraints. The message carries the failing report text.
    """
    if not validate:
        _write_field_by_path(step, path, value)
        return

    trial = copy.deepcopy(step)
    _write_field_by_path(trial, path, value)
    report = validate_spec(trial)
    if not report.valid:
        raise ValueError(
            f"set_field_by_path: value {value!r} at {path!r} is invalid for a "
            f"{step.get('step_type', 'llm')!r} step — {report.format_text()}"
        )
    _write_field_by_path(step, path, value)


def iter_mutable_values(step: dict[str, Any]) -> list[tuple[str, Any]]:
    """``[(mutable_path, current_value_or_None), ...]`` for this step's type."""
    st = step.get("step_type", "llm")
    st = st.value if hasattr(st, "value") else st
    return [(p, get_field_by_path(step, p, default=None)) for p in mutable_fields(st)]


# ---------------------------------------------------------------------------
# DAG mutation-safety helpers (for DELETE_STEP).
# ---------------------------------------------------------------------------


def find_leaf_steps(steps: list[dict[str, Any]]) -> set[int]:
    """Step numbers no other step depends on — safe to delete."""
    depended: set[int] = set()
    for s in steps:
        depended.update(s.get("dependencies", []) or [])
    return {n for s in steps if (n := s.get("number")) is not None and n not in depended}


def can_delete_step(steps: list[dict[str, Any]], number: int) -> bool:
    """True if ``number`` is a leaf and at least one step remains after removal."""
    return len(steps) >= 2 and number in find_leaf_steps(steps)


# ---------------------------------------------------------------------------
# Quiet loading — build specs in bulk without the per-chain reference notes.
# ---------------------------------------------------------------------------

_QUIET_DEPTH = 0
_LOUD_VALIDATE_REFERENCES: Optional[Any] = None


def _no_reference_warnings(self: Any) -> None:
    """Stand-in for ``ReasoningChain._validate_references`` while quiet.

    The real method only emits warnings: it reads ``self.steps`` and assigns
    nothing back, so replacing it changes what a chain *says*, never what it
    does. :func:`quiet_reference_warnings` documents the swap.
    """


@contextlib.contextmanager
def quiet_reference_warnings() -> Iterator[None]:
    """Build chains without ``ReasoningChain``'s ``$memory`` / ``$history[N]``
    reference ``UserWarning``\\ s.

    Those warnings are for a human hand-writing one chain, and they stay on by
    default for exactly that reader. A *program* loading specs in bulk — an
    evolutionary search legitimately generating chains whose first step reads
    ``$history[-1]`` — otherwise pays one warning per chain per build. Wrap the
    load, not the run::

        with quiet_reference_warnings():
            report, chain = validate_and_build(spec)   # or ReasoningChain.from_dict(...)

    Covers every construction path (``from_dict``, ``ChainBuilder.build()``,
    the constructor), because the warnings come from ``__init__``, not from
    ``from_dict``.

    Implemented by swapping that one diagnostic method for a no-op rather than
    by installing a ``warnings`` filter, for two measured reasons. On 200 builds
    with an unrelated ``UserWarning`` raised alongside each, wrapping every
    build in ``catch_warnings()`` made the *unrelated* warning fire 200 times
    instead of 1 — mutating the filter list invalidates Python's per-location
    "warn once" registry, so a caller silencing one warning gets every other
    warning in the process repeated. The swap leaves it at 1. And it matches no
    message text, so a warning this is not meant to hide cannot be hidden by a
    reworded message later.

    Process-global and not thread-safe — the same caveat
    ``warnings.catch_warnings`` carries. Nesting is reference-counted. The
    mechanism is private; only the silence is the contract.
    """
    global _QUIET_DEPTH, _LOUD_VALIDATE_REFERENCES

    from ..chain import ReasoningChain  # lazy — keeps meta import-cycle-free

    if _QUIET_DEPTH == 0:
        _LOUD_VALIDATE_REFERENCES = ReasoningChain._validate_references
        ReasoningChain._validate_references = _no_reference_warnings  # type: ignore[method-assign]
    _QUIET_DEPTH += 1
    try:
        yield
    finally:
        _QUIET_DEPTH -= 1
        if _QUIET_DEPTH == 0:
            ReasoningChain._validate_references = _LOUD_VALIDATE_REFERENCES  # type: ignore[method-assign]
            _LOUD_VALIDATE_REFERENCES = None


# ---------------------------------------------------------------------------
# One-call generate -> validated, constructed chain.
# ---------------------------------------------------------------------------


def validate_and_build(
    spec: Any,
    *,
    use_typed_steps: bool = True,
    quiet: bool = False,
    resolve_sub_chain: Optional[Any] = None,
) -> tuple[ValidationReport, Any]:
    """Validate a spec and, only if valid, build a ``ReasoningChain``.

    Accepts the same shapes as :func:`validate_spec` and normalises them to
    chain form first, so cross-step dependency/cycle checks always run. Returns
    ``(report, chain)`` — ``chain`` is a constructed ``ReasoningChain`` when
    ``report.valid`` is True, else ``None``. Any residual construction error is
    folded into ``report`` as a ``build_failed`` :class:`ValidationIssue`, so
    callers get one error format instead of two.

    Note: runtime-only fields (``sub_chain`` / ``agents`` / ``base_step``) stay
    placeholder-injected; a green report means the chain is structurally sound,
    not that those sub-agents are bound.

    ``resolve_sub_chain`` is the rebinding hook for that gap. A step may carry a
    serializable ``sub_chain_ref`` — a gigaevo-memory entity id or any other
    pointer the caller understands — and this callable turns one ref into a
    chain: ``resolve_sub_chain(ref) -> ReasoningChain | None``. Every handoff
    step whose ref resolves gets its real sub-chain bound in place of the
    placeholder.

    **An unresolved ref is reported, never silently left as the placeholder.**
    That asymmetry is the point of the feature. Without it the failure mode is
    the worst kind available here: the chain is structurally valid, executes,
    and runs a step titled ``__placeholder__`` instead of the sub-agent the spec
    asked for — a wrong answer with no error anywhere, which is the same shape
    as the condition bug that made a typo'd branch indistinguishable from an
    unmet one. So a ref that is present and does not resolve — because no hook
    was passed, or because the hook returned None — becomes an
    ``unresolved_sub_chain_ref`` **warning** on the report naming the step.
    A warning rather than an error because inspecting a spec you cannot yet
    resolve is a legitimate thing to do; the caller is told, and execution will
    still fail loudly on the placeholder.

    ``quiet=True`` suppresses the reference ``UserWarning``\\ s for this build
    only — a convenience over :func:`quiet_reference_warnings`, which is the
    form to use when loading many specs. Note the asymmetry: ``quiet`` silences
    warnings, it never withholds *errors*. The returned ``report`` is
    byte-identical either way, so a bulk loader still learns about genuinely
    invalid specs.
    """
    # deep copy: from_dict rewrites step_config dicts into config objects
    # in place, which would corrupt the caller's spec.
    if isinstance(spec, dict) and "steps" in spec:
        chain_spec = copy.deepcopy(spec)
    elif isinstance(spec, list):
        chain_spec = {"steps": copy.deepcopy(spec)}
    else:
        chain_spec = {"steps": [copy.deepcopy(spec)]}

    report = validate_spec(chain_spec)
    if not report.valid:
        return report, None

    from ..chain import ReasoningChain  # lazy — keeps meta import-cycle-free

    try:
        # `quiet` covers construction only: the warnings come from
        # `ReasoningChain.__init__`, and `validate_spec` above never warns.
        with quiet_reference_warnings() if quiet else contextlib.nullcontext():
            chain = ReasoningChain.from_dict(chain_spec, use_typed_steps=use_typed_steps)
    except Exception as exc:  # residual pydantic/ValueError after a green validate_spec
        issues = [
            *report.errors,
            ValidationIssue(severity="error", code="build_failed", message=f"from_dict failed: {str(exc)[:300]}"),
        ]
        return ValidationReport(valid=False, errors=issues, warnings=report.warnings), None

    unresolved = _bind_sub_chain_refs(chain, resolve_sub_chain)
    if unresolved:
        report = ValidationReport(
            valid=report.valid,
            errors=list(report.errors),
            warnings=[*report.warnings, *unresolved],
        )
    return report, chain


def _bind_sub_chain_refs(chain: Any, resolve: Optional[Any]) -> list[ValidationIssue]:
    """Bind every ``sub_chain_ref`` this chain carries; report the ones that stay.

    Returns one warning per ref left unbound. Binding failures are collected
    rather than raised: one unresolvable sub-agent should not deny the caller
    the other nine, and the report is where a caller already looks.
    """
    issues: list[ValidationIssue] = []
    for step in getattr(chain, "steps", []) or []:
        ref = getattr(step, "sub_chain_ref", None)
        if not ref:
            continue
        resolved = None
        if resolve is not None:
            try:
                resolved = resolve(ref)
            except Exception as exc:  # a store lookup can fail; that is a report, not a crash
                issues.append(
                    ValidationIssue(
                        severity="warning",
                        code="unresolved_sub_chain_ref",
                        step=getattr(step, "number", None),
                        field="sub_chain_ref",
                        message=f"resolving sub_chain_ref {ref!r} raised {type(exc).__name__}: {str(exc)[:160]}",
                    )
                )
                continue
        if resolved is None:
            reason = "no resolve_sub_chain hook was given" if resolve is None else "the hook returned None"
            issues.append(
                ValidationIssue(
                    severity="warning",
                    code="unresolved_sub_chain_ref",
                    step=getattr(step, "number", None),
                    field="sub_chain_ref",
                    message=(
                        f"sub_chain_ref {ref!r} is unbound ({reason}); this step still holds the "
                        "placeholder sub-chain and will execute it"
                    ),
                )
            )
            continue
        step.sub_chain = resolved
    return issues


# ---------------------------------------------------------------------------
# Committed snapshot (external services + CI drift review).
# ---------------------------------------------------------------------------

# Fields compared for staleness — structural content only, so version bumps and
# timestamps don't cause spurious "stale snapshot" failures.
_SNAPSHOT_STRUCTURAL_KEYS = ("schema_version", "step_types", "common_fields", "references", "validation_rules")


def _structural(catalog_dump: dict[str, Any]) -> dict[str, Any]:
    return {k: catalog_dump.get(k) for k in _SNAPSHOT_STRUCTURAL_KEYS}


def snapshot_payload() -> dict[str, Any]:
    """The dict written to ``catalog_snapshot.json`` (timestamp omitted for stable diffs)."""
    return build_catalog(generated_at=None).model_dump()


def load_snapshot() -> Optional[StepCatalog]:
    """Load the committed snapshot, or None when it isn't shipped."""
    if not SNAPSHOT_PATH.exists():
        return None
    return StepCatalog.from_json(SNAPSHOT_PATH.read_text(encoding="utf-8"))


def snapshot_is_stale() -> tuple[bool, str]:
    """Compare the committed snapshot to the live catalog (structural only).

    Returns ``(stale, human_summary)``. ``stale`` is True when the snapshot is
    missing or its structural content differs from the live models — i.e. run
    ``make meta-snapshot``.
    """
    if not SNAPSHOT_PATH.exists():
        return True, "catalog_snapshot.json is missing — run `make meta-snapshot`"
    committed = _structural(json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8")))
    live = _structural(snapshot_payload())
    if committed == live:
        return False, "snapshot is up to date"
    live_types, committed_types = set(live["step_types"]), set(committed["step_types"])
    added = live_types - committed_types
    removed = committed_types - live_types
    bits = []
    if added:
        bits.append(f"added types: {', '.join(sorted(added))}")
    if removed:
        bits.append(f"removed types: {', '.join(sorted(removed))}")
    if not bits:
        bits.append("field-level changes within existing types")
    return True, "snapshot is stale (" + "; ".join(bits) + ") — run `make meta-snapshot`"


def write_snapshot() -> Path:
    """Write the committed snapshot to disk; returns its path."""
    SNAPSHOT_PATH.write_text(json.dumps(snapshot_payload(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return SNAPSHOT_PATH


__all__ = [
    "SNAPSHOT_PATH",
    "build_catalog",
    "get_step_catalog",
    "check_drift",
    "validate_spec",
    "quiet_reference_warnings",
    "validate_and_build",
    "schema_for",
    "config_class_names",
    "mutable_fields",
    "reference_fields",
    "auto_generate_minimal_step",
    "get_field_by_path",
    "set_field_by_path",
    "iter_mutable_values",
    "find_leaf_steps",
    "can_delete_step",
    "load_snapshot",
    "snapshot_is_stale",
    "snapshot_payload",
    "write_snapshot",
]
