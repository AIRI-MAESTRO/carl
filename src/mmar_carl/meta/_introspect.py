"""Live introspection of CARL step types into catalog data models.

This is the auto-update engine. It derives the structural half of the catalog
(which types exist, their classes, fields, types, required/optional, defaults,
descriptions, constraints) directly from the pydantic models + executor
registry + ``StepType`` enum, so a newly-added ``*StepDescription`` subclass
appears in the catalog with **zero edits here**.

Discovery is anchored on two independent sources that must agree (drift is
checked in :mod:`mmar_carl.meta.catalog`):

1. ``StepDescriptionBase.__subclasses__()`` — the typed step classes. The
   ``step_type`` is read via ``cls.model_construct().step_type``, which bypasses
   validation/required fields (verified to work for all classes).
2. ``mmar_carl.step_executors._EXECUTORS`` — the runtime registry; the one site
   that *must* be updated for a type to run, so it's the source for executor
   class names and the canonical type set.
"""

from __future__ import annotations

import copy
import enum
import types
import typing
from typing import Any

from pydantic import BaseModel

from ..models.enums import StepType
from ..models.steps import StepDescriptionBase
from ._curated import CURATED_STEP_META, REFERENCE_FIELDS
from .models import FieldSpec, StepTypeSpec

_NoneType = type(None)
_UnionType = getattr(types, "UnionType", object())

#: How deep to recurse into nested config models when expanding fields.
_MAX_NESTED_DEPTH = 2


# ---------------------------------------------------------------------------
# Compact type rendering — avoids the $defs/$ref explosion of json schema.
# ---------------------------------------------------------------------------


def render_type(annotation: Any) -> str:
    """Render a type annotation as a compact, human/LLM-readable string."""
    if annotation is None or annotation is _NoneType:
        return "None"
    if annotation is Any:
        return "Any"

    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)

    # Union / Optional (typing.Union and PEP-604 X | Y)
    if origin is typing.Union or origin is getattr(types, "UnionType", object()):
        non_none = [a for a in args if a is not _NoneType]
        optional = len(non_none) != len(args)
        inner = " | ".join(render_type(a) for a in non_none)
        if optional:
            return f"{inner}?" if len(non_none) == 1 else f"({inner})?"
        return inner

    if origin is typing.Literal:
        return "Literal[" + ", ".join(repr(a) for a in args) + "]"

    if origin is not None:
        name = getattr(origin, "__name__", None) or str(origin).replace("typing.", "")
        if args:
            return f"{name}[{', '.join(render_type(a) for a in args)}]"
        return name

    # Bare class
    if isinstance(annotation, type):
        if issubclass(annotation, enum.Enum):
            return "enum[" + "|".join(str(getattr(m, "value", m.name)) for m in annotation) + "]"
        return annotation.__name__

    # ForwardRef / TypeVar / anything else
    return str(annotation).replace("typing.", "")


# ---------------------------------------------------------------------------
# Choices, constraints, and nested-model detection — structured data a
# generator/mutator can act on without parsing the type string or the schema.
# ---------------------------------------------------------------------------

_CONSTRAINT_KEYS = ("ge", "gt", "le", "lt", "min_length", "max_length", "min_items", "max_items")

#: Friendly labels for rendering constraints in human-readable validator notes.
_CONSTRAINT_LABELS = {"ge": ">=", "gt": ">", "le": "<=", "lt": "<"}


def _constraints_dict(field_info: Any) -> dict[str, Any]:
    """Machine-readable field constraints, e.g. ``{'ge': 0.0, 'le': 2.0}``."""
    out: dict[str, Any] = {}
    for meta in getattr(field_info, "metadata", []) or []:
        for key in _CONSTRAINT_KEYS:
            val = getattr(meta, key, None)
            if val is not None:
                out[key] = val
    return out


def _constraints_text(constraints: dict[str, Any]) -> str:
    return ", ".join(f"{_CONSTRAINT_LABELS.get(k, k)} {v}" for k, v in constraints.items())


def _choices(annotation: Any) -> list[str] | None:
    """Valid values for an Enum / Literal field (unwrapping Optional), else None."""
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)
    if origin is typing.Union or origin is _UnionType:
        for a in (x for x in args if x is not _NoneType):
            got = _choices(a)
            if got:
                return got
        return None
    if origin is typing.Literal:
        return [str(a) for a in args]
    if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
        return [str(getattr(m, "value", m.name)) for m in annotation]
    return None


def _nested_of(annotation: Any) -> tuple[type | None, str | None]:
    """If *annotation* resolves to a nested pydantic model, return
    ``(model_class, container)`` where container is 'scalar' | 'list' | 'dict'.
    Unions of *multiple* models are ambiguous and return ``(None, None)``."""
    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)
    if origin is None:
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            return annotation, "scalar"
        return None, None
    if origin is typing.Union or origin is _UnionType:
        non_none = [a for a in args if a is not _NoneType]
        models = [a for a in non_none if isinstance(a, type) and issubclass(a, BaseModel)]
        if len(models) > 1:
            return None, None  # e.g. ToolSource union — too ambiguous to expand
        if len(non_none) == 1:
            return _nested_of(non_none[0])  # Optional[Model] / Optional[list[Model]]
        return None, None
    if origin in (list, set, frozenset, tuple):
        for a in args:
            if isinstance(a, type) and issubclass(a, BaseModel):
                return a, "list"
        return None, None
    if origin is dict:
        if len(args) == 2 and isinstance(args[1], type) and issubclass(args[1], BaseModel):
            return args[1], "dict"
        return None, None
    return None, None


# ---------------------------------------------------------------------------
# Field-spec extraction.
# ---------------------------------------------------------------------------


def _default_for(field_info: Any) -> Any:
    """A JSON-serialisable default, or None when there's no meaningful one."""
    from pydantic_core import PydanticUndefined

    default = getattr(field_info, "default", PydanticUndefined)
    if default is PydanticUndefined or default is None:
        return None
    if isinstance(default, enum.Enum):
        return default.value
    if isinstance(default, (str, int, float, bool, list, dict)):
        return default
    return None  # factory-built or exotic default -> omit


def field_specs(
    model_cls: type,
    *,
    only: set[str] | None = None,
    _depth: int = 0,
    _seen: frozenset[str] = frozenset(),
) -> list[FieldSpec]:
    """Return the :class:`FieldSpec` list for *model_cls*.

    Populates structured ``choices`` / ``constraints`` and recursively expands
    nested pydantic models into ``nested_fields`` (depth-limited by
    ``_MAX_NESTED_DEPTH`` and cycle-guarded via ``_seen``). *only* restricts to
    a subset of field names (used for the inlined LLM case).
    """
    specs: list[FieldSpec] = []
    for name, fi in model_cls.model_fields.items():
        if only is not None and name not in only:
            continue
        ann = fi.annotation
        nested_cls, container = _nested_of(ann)
        nested_name = nested_cls.__name__ if nested_cls is not None else None
        nested_fields = None
        if nested_cls is not None and _depth < _MAX_NESTED_DEPTH and nested_cls.__name__ not in _seen:
            nested_fields = field_specs(nested_cls, _depth=_depth + 1, _seen=_seen | {nested_cls.__name__})
        constraints = _constraints_dict(fi)
        specs.append(
            FieldSpec(
                name=name,
                type=render_type(ann),
                required=fi.is_required(),
                default=_default_for(fi),
                description=(fi.description or "").strip(),
                excluded=bool(getattr(fi, "exclude", False)),
                choices=_choices(ann),
                constraints=constraints or None,
                nested_model=nested_name,
                container=container,
                nested_fields=nested_fields,
            )
        )
    return specs


# ---------------------------------------------------------------------------
# Discovery.
# ---------------------------------------------------------------------------


def discover_step_classes() -> dict[StepType, type]:
    """Map every ``StepType`` -> its ``*StepDescription`` class via reflection."""
    out: dict[StepType, type] = {}
    for cls in StepDescriptionBase.__subclasses__():
        try:
            st = cls.model_construct().step_type
        except Exception:  # pragma: no cover - defensive; all 16 verified to work
            continue
        out[st] = cls
    return out


def executor_class_names() -> dict[StepType, str]:
    """Map ``StepType`` -> executor class name (lazy import avoids cycles)."""
    from ..step_executors import _EXECUTORS  # local import — keeps meta import-cycle-free

    return {st: type(ex).__name__ for st, ex in _EXECUTORS.items()}


def common_field_specs() -> list[FieldSpec]:
    """Fields shared by every step (declared on ``StepDescriptionBase``)."""
    return field_specs(StepDescriptionBase)


# ---------------------------------------------------------------------------
# Per-type spec assembly.
# ---------------------------------------------------------------------------


def _placeholder_value(field: FieldSpec) -> Any:
    t = field.type
    # Match the parametrized forms render_type emits for collections
    # (e.g. "list[int]"), not bare class names that merely start with list/dict.
    if t.startswith("list[") or t == "list":
        return []
    if t.startswith("dict[") or t == "dict":
        return {}
    if t in ("int", "float"):
        return 0
    if t == "bool":
        return False
    return "..."


def _fallback_example(step_type: str, inlined: bool, config_fields: list[FieldSpec]) -> dict[str, Any]:
    """A required-only skeleton for a type with no curated example."""
    req = {f.name: _placeholder_value(f) for f in config_fields if f.required}
    if inlined:
        return req or {"aim": "..."}
    return {"step_config": req}


def build_step_type_spec(step_type: StepType, desc_cls: type, executor_name: str) -> StepTypeSpec:
    st = step_type.value
    curated = CURATED_STEP_META.get(st, {})

    base_names = set(StepDescriptionBase.model_fields)
    own_names = set(desc_cls.model_fields) - base_names
    inlined = "config" not in desc_cls.model_fields

    if inlined:
        # LLM-style: salient fields are the type-specific inlined fields.
        type_specific = own_names - {"config"}
        cfg_fields = field_specs(desc_cls, only=type_specific)
        config_class = None
    else:
        config_ann = desc_cls.model_fields["config"].annotation
        config_class = getattr(config_ann, "__name__", str(config_ann))
        cfg_fields = field_specs(config_ann)

    # exclude=True, type-specific fields are runtime-only (sub_chain/agents/base_step).
    runtime_only = sorted(
        n for n in own_names if n != "config" and bool(getattr(desc_cls.model_fields[n], "exclude", False))
    )
    # Don't surface runtime-only fields as inlined config fields.
    cfg_fields = [f for f in cfg_fields if f.name not in runtime_only]

    # Apply curated required-overrides (validator-enforced but field-optional).
    required_overrides = set(curated.get("required_overrides", []))
    for f in cfg_fields:
        if f.name in required_overrides:
            f.required = True

    # Flag config fields that accept $... references (from the curated table).
    ref_top = {p.split(".")[1] for p in REFERENCE_FIELDS.get(st, []) if p.startswith("step_config.")}
    for f in cfg_fields:
        if f.name in ref_top:
            f.accepts_references = True

    # Build human-readable validator notes from overrides + field constraints.
    validators: list[str] = [f"{name} is required (validator-enforced)" for name in sorted(required_overrides)]
    for f in cfg_fields:
        if f.constraints:
            validators.append(f"{f.name}: {_constraints_text(f.constraints)}")

    # Assemble a ready-to-use example spec (chain-serialization shape).
    # deep copy so StepTypeSpec.example never aliases the module-level
    # CURATED_STEP_META dict (a mutator writing into a generated step must not
    # corrupt the catalog / the next build).
    payload = copy.deepcopy(curated.get("example")) or _fallback_example(st, inlined, cfg_fields)
    deps = payload.pop("dependencies", [])
    example = {"number": 1, "title": f"{st} step", "step_type": st, "dependencies": deps}
    example.update(payload)

    purpose = curated.get("purpose") or _docstring_first_line(desc_cls) or st

    return StepTypeSpec(
        step_type=st,
        purpose=purpose,
        description_class=desc_cls.__name__,
        config_class=config_class,
        executor_class=executor_name,
        inlined=inlined,
        config_fields=cfg_fields,
        runtime_only_fields=runtime_only,
        mutable_fields=list(curated.get("mutable_fields", [])),
        reference_fields=list(REFERENCE_FIELDS.get(st, [])),
        validators=validators,
        example=example,
    )


def _docstring_first_line(cls: type) -> str:
    doc = (cls.__doc__ or "").strip()
    if not doc:
        return ""
    return " ".join(doc.split("\n\n", 1)[0].split())


__all__ = [
    "render_type",
    "field_specs",
    "discover_step_classes",
    "executor_class_names",
    "common_field_specs",
    "build_step_type_spec",
]
