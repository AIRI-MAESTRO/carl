"""Self-updating meta-description of CARL step types.

A single, token-efficient source of truth that external agent systems (chain
generators like MAGE, evolutionary searchers like GigaEvo) use to *generate*,
*mutate*, and *validate* CARL chains — without reading the source or blowing
their context window.

Everything structural is introspected live from the pydantic models + executor
registry, so a newly-added ``*StepDescription`` subclass appears automatically;
a small curated side-table adds the editorial bits reflection can't infer
(purpose, examples, safe-to-mutate fields). A drift guard
(:func:`check_drift`) and a committed snapshot keep the registration sites and
external consumers honest.

Quick start::

    from mmar_carl.meta import get_step_catalog, validate_spec, auto_generate_minimal_step

    cat = get_step_catalog()
    print(cat.tier0())                       # name -> purpose (tiny)
    print(cat.tier1(["llm", "tool"]))        # compact cards
    print(cat.tier2("tool"))                 # full card

    prompt = cat.generation_prompt(max_steps=6)      # auto-updating planner prompt
    report = validate_spec({"steps": [...]})         # no-execution validation
    step = auto_generate_minimal_step("evaluation")  # INSERT_STEP template
"""

from __future__ import annotations

from .catalog import (
    SNAPSHOT_PATH,
    auto_generate_minimal_step,
    build_catalog,
    can_delete_step,
    check_drift,
    config_class_names,
    find_leaf_steps,
    get_field_by_path,
    get_step_catalog,
    iter_mutable_values,
    load_snapshot,
    mutable_fields,
    reference_fields,
    schema_for,
    set_field_by_path,
    snapshot_is_stale,
    snapshot_payload,
    validate_and_build,
    validate_spec,
    write_snapshot,
)
from .models import (
    CATALOG_SCHEMA_VERSION,
    CapabilityEntry,
    CapabilityLibrary,
    DriftReport,
    FieldSpec,
    MutationRecord,
    ReferenceSpec,
    StepCatalog,
    StepTypeSpec,
    ValidationIssue,
    ValidationReport,
)
from .mutations import (
    MutationError,
    apply_records,
    change_step_type,
    deletable_steps,
    dependents_of,
    expand_field_path,
    find_root_steps,
    frozen_step_numbers,
    insert_step_at,
    mutable_targets,
    mutate_content,
    mutate_tool_code,
    remove_step,
    rewire_dependencies,
    sample_field_value,
    steps_of_type,
)

__all__ = [
    # primary entry points
    "get_step_catalog",
    "validate_spec",
    "validate_and_build",
    "auto_generate_minimal_step",
    "schema_for",
    "config_class_names",
    "mutable_fields",
    "reference_fields",
    "check_drift",
    # mutation helpers — content
    "get_field_by_path",
    "set_field_by_path",
    "iter_mutable_values",
    "sample_field_value",
    "mutate_content",
    "mutate_tool_code",
    "mutable_targets",
    "expand_field_path",
    # mutation helpers — topology
    "rewire_dependencies",
    "change_step_type",
    "insert_step_at",
    "remove_step",
    "apply_records",
    "MutationRecord",
    "MutationError",
    # capability library
    "CapabilityEntry",
    "CapabilityLibrary",
    # dag queries
    "find_leaf_steps",
    "can_delete_step",
    "find_root_steps",
    "dependents_of",
    "steps_of_type",
    "deletable_steps",
    "frozen_step_numbers",
    # build / snapshot
    "build_catalog",
    "load_snapshot",
    "snapshot_is_stale",
    "snapshot_payload",
    "write_snapshot",
    "SNAPSHOT_PATH",
    # data models
    "StepCatalog",
    "StepTypeSpec",
    "FieldSpec",
    "ReferenceSpec",
    "ValidationReport",
    "ValidationIssue",
    "DriftReport",
    "CATALOG_SCHEMA_VERSION",
]
