"""Pydantic data models for the CARL step-type meta-catalog.

These are pure data + presentation: they hold already-introspected metadata and
render it at several verbosity tiers. They deliberately import nothing from the
rest of CARL (no ``chain`` / ``step_executors`` / ``models.steps``), so the
catalog is cheap to serialise, safe to ship to an external service, and free of
import cycles. Live introspection lives in :mod:`mmar_carl.meta._introspect`;
assembly + behaviour (validation, drift, schemas) in :mod:`mmar_carl.meta.catalog`.

Token efficiency comes from *progressive disclosure*:

- **tier 0** — ``catalog.tier0()`` / ``list_step_types()``: one line per type (~200 tok).
- **tier 1** — ``spec.card()`` / ``catalog.tier1()``: required + key-optional fields
  + a ready-to-use example (~4 KB for all 16 types).
- **tier 2** — ``spec.card(full=True)`` / ``catalog.tier2(name)``: every field with
  descriptions, validators, runtime-only notes (~1-2 KB per type).
- **tier 3** — ``catalog.schema_for(name)`` (in ``catalog.py``): the raw JSON Schema
  of one type's config model, on demand.
"""

from __future__ import annotations

import json
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

#: Bumped when the serialized catalog shape changes incompatibly.
#: v2 added FieldSpec.choices/constraints/nested_model/container/nested_fields/accepts_references.
CATALOG_SCHEMA_VERSION = 2


class FieldSpec(BaseModel):
    """One field of a step's config (or an inlined LLM/common field).

    Beyond the type string, this carries *structured* data a generator/mutator
    can act on directly: enum/Literal ``choices``, numeric/length
    ``constraints``, and — for fields whose type is a nested pydantic model —
    the model's own ``nested_fields`` (so you never have to parse the tier-3
    JSON schema to learn a nested shape).
    """

    name: str = Field(..., description="Field name.")
    type: str = Field(..., description="Compact rendered type, e.g. 'list[int]', 'A | B?', 'Literal[...]'.")
    required: bool = Field(..., description="True when the field must be supplied.")
    default: Any = Field(default=None, description="Default value when optional (None if no meaningful default).")
    description: str = Field(default="", description="Field docstring from Field(description=...).")
    excluded: bool = Field(default=False, description="True when excluded from serialization (runtime-only).")
    choices: Optional[list[str]] = Field(
        default=None, description="Valid values for an Enum/Literal field (unified); None if not a choice field."
    )
    constraints: Optional[dict[str, Any]] = Field(
        default=None, description="Machine-readable constraints, e.g. {'ge': 0.0, 'le': 2.0, 'min_length': 1}."
    )
    nested_model: Optional[str] = Field(
        default=None,
        description="Class name when the field's type is a nested pydantic model (e.g. 'MCPServerConfig').",
    )
    container: Optional[str] = Field(
        default=None, description="'scalar' | 'list' | 'dict' when nested_model is set (e.g. list[ConditionalBranch])."
    )
    nested_fields: Optional[list["FieldSpec"]] = Field(
        default=None, description="The nested model's own fields (recursive, depth-limited)."
    )
    accepts_references: bool = Field(
        default=False, description="True when this field accepts $... dynamic-reference strings."
    )

    def one_line(self, *, with_desc: bool = False) -> str:
        default = "" if (self.default is None or self.required) else f"={self.default!r}"
        extra = ""
        if self.choices:
            extra += f" choices={self.choices}"
        if with_desc and self.constraints:
            extra += f" ({', '.join(f'{k} {v}' for k, v in self.constraints.items())})"
        tail = f" — {self.description}" if with_desc and self.description else ""
        return f"{self.name}: {self.type}{default}{extra}{tail}"


class StepTypeSpec(BaseModel):
    """Everything an external system needs to generate, mutate, or validate one step type."""

    step_type: str = Field(..., description="StepType enum value, e.g. 'tool'.")
    purpose: str = Field(..., description="One-line, planner-facing summary.")
    description_class: str = Field(..., description="*StepDescription class name.")
    config_class: Optional[str] = Field(
        default=None, description="*StepConfig class name (None when fields are inlined)."
    )
    executor_class: str = Field(..., description="Executor class name from the registry.")
    inlined: bool = Field(
        default=False, description="True when fields live on the description (llm), not an embedded config."
    )
    config_fields: list[FieldSpec] = Field(
        default_factory=list, description="Salient type-specific fields (config fields, or inlined llm fields)."
    )
    runtime_only_fields: list[str] = Field(
        default_factory=list,
        description="exclude=True fields bound at runtime (e.g. sub_chain, agents, base_step) — never appear in a spec.",
    )
    mutable_fields: list[str] = Field(
        default_factory=list, description="Curated: fields safe for an evolutionary mutator to change."
    )
    reference_fields: list[str] = Field(
        default_factory=list, description="Config fields that accept $... reference strings."
    )
    validators: list[str] = Field(
        default_factory=list, description="Human-readable validator rules introspection can't show."
    )
    example: dict[str, Any] = Field(
        default_factory=dict, description="Minimal valid spec in chain-serialization shape (step_config keyed)."
    )

    @property
    def required_fields(self) -> list[FieldSpec]:
        return [f for f in self.config_fields if f.required]

    @property
    def optional_fields(self) -> list[FieldSpec]:
        return [f for f in self.config_fields if not f.required]

    def field(self, name: str) -> Optional[FieldSpec]:
        """The :class:`FieldSpec` for a config field, or None."""
        for f in self.config_fields:
            if f.name == name:
                return f
        return None

    def field_choices(self, name: str) -> Optional[list[str]]:
        """Valid values for an Enum/Literal config field, or None."""
        f = self.field(name)
        return f.choices if f else None

    def field_constraint(self, name: str, key: str) -> Any:
        """A single constraint value (e.g. ``field_constraint('temperature', 'le')``), or None."""
        f = self.field(name)
        return (f.constraints or {}).get(key) if f else None

    def nested_spec(self, name: str) -> Optional[list[FieldSpec]]:
        """The nested model's fields for a config field whose type is a model."""
        f = self.field(name)
        return f.nested_fields if f else None

    def _key_optionals(self, limit: int) -> list[FieldSpec]:
        """Optional fields a generator/mutator most likely cares about: the
        curated ``mutable_fields`` first, then the rest, capped at *limit*."""
        mut = {m.split(".", 1)[1] for m in self.mutable_fields if m.startswith("step_config.")}
        mut |= {m for m in self.mutable_fields if "." not in m}
        ordered = sorted(self.optional_fields, key=lambda f: (f.name not in mut, f.name))
        return ordered[:limit]

    def card(self, *, full: bool = False, max_optional: int = 6) -> str:
        """tier-1 (compact) or tier-2 (full) text card for this step type."""
        loc = "inlined on the step" if self.inlined else f"step_config: {self.config_class}"
        lines = [f"### {self.step_type} — {self.purpose}", f"  ({loc})"]

        req = self.required_fields
        if req:
            lines.append("  required:")
            lines += [f"    - {f.one_line(with_desc=full)}" for f in req]

        opt = self.optional_fields
        shown = opt if full else self._key_optionals(max_optional)
        if shown:
            label = "  optional:" if full else f"  key optional ({len(opt)} total):"
            lines.append(label)
            lines += [f"    - {f.one_line(with_desc=full)}" for f in shown]
            if not full and len(opt) > len(shown):
                lines.append(f"    - (+{len(opt) - len(shown)} more optional fields; see tier 2)")

        # Inline nested-model shapes (tier 2 only) so consumers don't have to
        # fetch and parse the JSON schema to learn e.g. ConditionalBranch.
        if full:
            for f in self.config_fields:
                if f.nested_fields:
                    kind = {"list": "each item", "dict": "each value", "scalar": "fields"}.get(
                        f.container or "", "fields"
                    )
                    lines.append(f"  {f.name} ({f.nested_model}) {kind}:")
                    lines += [f"      - {nf.one_line(with_desc=False)}" for nf in f.nested_fields]

        if self.runtime_only_fields:
            lines.append(f"  runtime-only (bound before exec, omit from spec): {', '.join(self.runtime_only_fields)}")
        if full and self.reference_fields:
            lines.append(f"  accepts $refs in: {', '.join(self.reference_fields)}")
        if full and self.validators:
            lines.append("  validators: " + "; ".join(self.validators))
        if full and self.mutable_fields:
            lines.append("  mutable (safe to evolve): " + ", ".join(self.mutable_fields))

        lines.append("  example: " + json.dumps(self.example, ensure_ascii=False))
        return "\n".join(lines)

    def _repr_markdown_(self) -> str:
        return f"```text\n{self.card(full=True)}\n```"


class MutationRecord(BaseModel):
    """A single, replayable chain-spec mutation.

    Produced by the ``mmar_carl.meta`` mutation primitives; serialisable so an
    evolution engine can record lineage and replay a decision path
    (``apply_records``) on another chain.
    """

    op: str = Field(..., description="'rewire' | 'retype' | 'insert' | 'remove' | 'set_field' | 'tool_code'.")
    step_number: Optional[int] = Field(default=None, description="Target step number (the new step for 'insert').")
    path: Optional[str] = Field(default=None, description="Field path for 'set_field' (e.g. 'step_config.temperature').")
    before: Any = Field(default=None, description="Value/step before the mutation (for diff/undo context).")
    after: Any = Field(default=None, description="Value/step/deps after the mutation (used by apply_records).")
    new_step_type: Optional[str] = Field(default=None, description="Target step type for 'retype'.")

    def summary(self) -> str:
        if self.op == "set_field":
            return f"set {self.path} on step {self.step_number}: {self.before!r} -> {self.after!r}"
        if self.op == "rewire":
            return f"rewire step {self.step_number} deps: {self.before} -> {self.after}"
        if self.op == "retype":
            return f"retype step {self.step_number} -> {self.new_step_type}"
        if self.op == "insert":
            return f"insert step {self.step_number}"
        if self.op == "remove":
            return f"remove step {self.step_number}"
        if self.op == "tool_code":
            return f"evolve code tool at {self.path}"
        return f"{self.op} step {self.step_number}"


class CapabilityEntry(BaseModel):
    """One tool / AgentSkill / concept a generator or mutator may use.

    ``kind='concept'`` carries domain knowledge (e.g. "prefer an AgentSkill for
    PDF extraction") — the *content* is the caller's; CARL owns only the schema.
    """

    name: str
    kind: Literal["tool", "skill", "concept"]
    description: str
    when_to_use: Optional[str] = Field(default=None, description="When this capability is the right choice.")
    example: Optional[str] = Field(default=None, description="A short usage example.")

    def render(self) -> str:
        parts = [f"- [{self.kind}] {self.name}: {self.description}"]
        if self.when_to_use:
            parts.append(f" (use when: {self.when_to_use})")
        if self.example:
            parts.append(f" e.g. {self.example}")
        return "".join(parts)


class CapabilityLibrary(BaseModel):
    """A structured, token-budgeted library of tools / skills / concepts fed to
    generation and mutation (see :meth:`StepCatalog.generation_prompt`)."""

    entries: list[CapabilityEntry] = Field(default_factory=list)

    def by_kind(self, kind: str) -> list[CapabilityEntry]:
        return [e for e in self.entries if e.kind == kind]

    def render(self, *, max_tokens: Optional[int] = None) -> str:
        if not self.entries:
            return ""
        lines: list[str] = []
        for kind in ("tool", "skill", "concept"):
            group = self.by_kind(kind)
            if group:
                lines.append(f"{kind.capitalize()}s:")
                lines += [f"  {e.render()}" for e in group]
        text = "Available capabilities:\n" + "\n".join(lines)
        if max_tokens is not None and len(text) > max_tokens * 4:
            text = text[: max_tokens * 4] + "\n  …[capabilities truncated]"
        return text


class ReferenceSpec(BaseModel):
    """One ``$...`` dynamic-reference grammar entry."""

    prefix: str
    syntax: str
    description: str
    returns: str


class ValidationIssue(BaseModel):
    """A single problem found by :func:`validate_spec`.

    ``field`` is always a **dotted path** when present (e.g. ``'aim'``,
    ``'dependencies'``, ``'step_config.branches.0.next_step'``) — safe to show
    an LLM verbatim in a retry turn.
    """

    severity: str = Field(..., description="'error' or 'warning'.")
    code: str = Field(..., description="Stable machine code, e.g. 'missing_required_field'.")
    message: str
    step: Optional[int] = Field(default=None, description="Offending step number, when applicable.")
    field: Optional[str] = Field(default=None, description="Offending dotted field path, when applicable.")

    def format_for_llm(self) -> str:
        """A single, retry-oriented sentence keyed off the stable ``code``."""
        where = f"Step {self.step}" if self.step is not None else "The chain"
        fld = f" '{self.field}'" if self.field else ""
        templates = {
            "missing_required_field": f"{where} is missing required field{fld}. Add it.",
            "missing_step_config": f"{where} needs a 'step_config' object. Add one matching this step type.",
            "invalid_config": f"{where} has an invalid value at{fld}: {self.message}. Fix it.",
            "unknown_step_type": f"{where} uses an unknown step_type: {self.message}. Pick a listed type.",
            "duplicate_number": f"{where} reuses a step number. Give every step a unique 'number'.",
            "missing_dependency": f"{where} depends on a step that doesn't exist{(': ' + self.message) if self.message else ''}. Reference only existing step numbers.",
            "cycle": f"{self.message}. Remove the circular dependency.",
            "empty": "The chain has no steps. Produce at least one step.",
            "memory_no_writer": f"{where}{fld} {self.message}",
        }
        return templates.get(self.code, f"{where}{fld}: {self.message}")


class ValidationReport(BaseModel):
    """Structured, no-execution validation result for a step or chain spec."""

    valid: bool
    errors: list[ValidationIssue] = Field(default_factory=list)
    warnings: list[ValidationIssue] = Field(default_factory=list)

    def format_text(self) -> str:
        if self.valid and not self.warnings:
            return "valid: ok (0 errors, 0 warnings)"
        head = "valid" if self.valid else "INVALID"
        out = [f"{head}: {len(self.errors)} error(s), {len(self.warnings)} warning(s)"]
        for issue in self.errors:
            loc = f" step {issue.step}" if issue.step is not None else ""
            fld = f" [{issue.field}]" if issue.field else ""
            out.append(f"  ✗{loc}{fld} {issue.code}: {issue.message}")
        for issue in self.warnings:
            loc = f" step {issue.step}" if issue.step is not None else ""
            fld = f" [{issue.field}]" if issue.field else ""
            out.append(f"  ⚠{loc}{fld} {issue.code}: {issue.message}")
        return "\n".join(out)

    def _repr_markdown_(self) -> str:
        banner = "✅ **valid**" if self.valid else "❌ **invalid**"
        return f"{banner} — {len(self.errors)} error(s), {len(self.warnings)} warning(s)\n\n```text\n{self.format_text()}\n```"


class DriftReport(BaseModel):
    """Result of :func:`check_drift` — are the step-type registration sites in sync?"""

    in_sync: bool
    step_types: list[str] = Field(default_factory=list, description="Canonical StepType values.")
    issues: list[str] = Field(default_factory=list, description="Human-readable divergences (empty when in sync).")

    def format_text(self) -> str:
        if self.in_sync:
            return f"meta drift: ok ({len(self.step_types)} step types in sync)"
        return "meta drift: DETECTED\n" + "\n".join(f"  - {i}" for i in self.issues)

    def _repr_markdown_(self) -> str:
        banner = "✅ in sync" if self.in_sync else "⚠️ drift detected"
        return f"**meta drift** — {banner}\n\n```text\n{self.format_text()}\n```"


class StepCatalog(BaseModel):
    """The serialisable, token-efficient catalog of CARL step types.

    Build via :func:`mmar_carl.meta.get_step_catalog`. Ship to an external
    service via :meth:`to_json` (versioned) and reload via :meth:`from_json`.
    """

    schema_version: int = Field(default=CATALOG_SCHEMA_VERSION)
    carl_version: str = Field(default="", description="mmar-carl version this catalog was generated from.")
    generated_at: Optional[str] = Field(
        default=None, description="ISO timestamp when snapshotted (None for live/in-memory catalogs)."
    )
    step_types: dict[str, StepTypeSpec] = Field(default_factory=dict)
    common_fields: list[FieldSpec] = Field(
        default_factory=list, description="Fields shared by every step (from StepDescriptionBase)."
    )
    references: list[ReferenceSpec] = Field(default_factory=list)
    validation_rules: list[dict[str, str]] = Field(default_factory=list)

    # -- access -----------------------------------------------------------
    def names(self) -> list[str]:
        return list(self.step_types.keys())

    def step(self, name: str) -> StepTypeSpec:
        if name not in self.step_types:
            raise KeyError(f"Unknown step type {name!r}. Known: {', '.join(self.step_types)}")
        return self.step_types[name]

    def list_step_types(self) -> dict[str, str]:
        """tier 0 as data: ``{step_type: purpose}``."""
        return {name: spec.purpose for name, spec in self.step_types.items()}

    # -- tiers ------------------------------------------------------------
    def tier0(self) -> str:
        return "\n".join(f"- {name}: {spec.purpose}" for name, spec in self.step_types.items())

    def tier1(self, include: Optional[list[str]] = None) -> str:
        names = include or self.names()
        return "\n\n".join(self.step(n).card(full=False) for n in names)

    def tier2(self, name: str) -> str:
        return self.step(name).card(full=True)

    def reference_syntax(self) -> str:
        rows = [f"  {r.syntax:<34} {r.description} (-> {r.returns})" for r in self.references]
        return "Dynamic reference grammar ($...):\n" + "\n".join(rows)

    def common_fields_text(self) -> str:
        return "Common fields on every step:\n" + "\n".join(
            f"  - {f.one_line()}" + ("  *required" if f.required else "") for f in self.common_fields
        )

    # -- generation -------------------------------------------------------
    def generation_prompt(
        self,
        *,
        max_steps: int = 10,
        include: Optional[list[str]] = None,
        detail: str = "cards",
        available_tools: Optional[list[str]] = None,
        available_skills: Optional[list[str]] = None,
        capability_library: Optional["CapabilityLibrary"] = None,
        extra_instructions: str = "",
    ) -> str:
        """Render an auto-updating, token-budgeted planner prompt fragment.

        Replaces hand-written step-type documentation: every registered type
        appears automatically. ``detail='menu'`` emits only the tier-0 list;
        ``detail='cards'`` (default) adds a compact tier-1 card per type.
        """
        names = include or self.names()
        parts: list[str] = []
        parts.append(
            "You are a CARL chain planner. Produce a JSON plan: a single object "
            '{"steps": [ {step}, ... ]}. Each step object has: '
            '"number" (int, from 1, increasing), "title" (str), "step_type" (str, '
            'one of the types below), "dependencies" (list[int] of prior step numbers). '
            'LLM steps add "aim" (str, required). Every other type adds a "step_config" '
            "object whose shape matches its card below."
        )
        parts.append(
            "Rules (strict — wrong shapes are rejected by pydantic):\n"
            f"- Produce at most {max_steps} steps; fewer is better. Number them sequentially from 1.\n"
            '- Use "dependencies" to express the DAG; do NOT use "triggered_by".\n'
            "- String fields (aim, reasoning_questions, stage_action, example_reasoning) must be a "
            "single string, NOT an array — join multiple items with newlines.\n"
            '- "step_config" is an object (NOT array) whose shape matches the step type\'s card below.\n'
            "- Output JSON only (triple-backtick json fences are tolerated)."
        )
        parts.append("Available step types:\n" + self.tier0())
        if detail == "cards":
            parts.append("Step type details:\n\n" + self.tier1(include=names))
        parts.append(self.reference_syntax())
        if available_tools:
            parts.append(
                "You MAY reference these registered tools in `tool` steps "
                f"(use exactly these names): {', '.join(available_tools)}."
            )
        if available_skills:
            parts.append(
                f"You MAY reference these named AgentSkills in `agent_skill` steps: {', '.join(available_skills)}."
            )
        if capability_library is not None and capability_library.entries:
            parts.append(capability_library.render())
        if extra_instructions:
            parts.append("Additional constraints: " + extra_instructions)
        return "\n\n".join(parts)

    # -- serialization ----------------------------------------------------
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.model_dump(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_json(cls, data: str) -> "StepCatalog":
        return cls.model_validate_json(data)

    # -- display ----------------------------------------------------------
    def format_text(self) -> str:
        return (
            f"CARL step catalog (schema v{self.schema_version}, carl {self.carl_version or '?'}, "
            f"{len(self.step_types)} step types)\n\n" + self.tier0()
        )

    def _repr_markdown_(self) -> str:
        return (
            f"**CARL step catalog** — {len(self.step_types)} step types "
            f"(schema v{self.schema_version}, carl {self.carl_version or '?'})\n\n"
            f"```text\n{self.tier0()}\n```"
        )


__all__ = [
    "CATALOG_SCHEMA_VERSION",
    "FieldSpec",
    "StepTypeSpec",
    "ReferenceSpec",
    "ValidationIssue",
    "ValidationReport",
    "DriftReport",
    "StepCatalog",
    "MutationRecord",
    "CapabilityEntry",
    "CapabilityLibrary",
]
