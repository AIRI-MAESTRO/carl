"""Drift guard — the single test that fails the moment a step-type registration
site falls out of sync.

This is the safety net that makes ``mmar_carl.meta`` (and the library's
step-type handling generally) auto-updating: adding a ``StepType`` without also
registering an executor, a description subclass, a union member, or a working
``from_dict`` branch surfaces here immediately. Before the meta work, three
types were missing from ``AnyStepDescription``; ``test_union_covers_every_step_type``
encodes exactly that invariant.
"""

from __future__ import annotations

import typing

import pytest

from mmar_carl import check_drift, get_step_catalog
from mmar_carl.chain import _create_typed_step_from_dict, _step_maps
from mmar_carl.meta._introspect import discover_step_classes, executor_class_names
from mmar_carl.models.enums import StepType
from mmar_carl.models.steps import AnyStepDescription

ALL_TYPES = {st.value for st in StepType}


def _union_types() -> set[str]:
    return {cls.model_construct().step_type.value for cls in typing.get_args(AnyStepDescription)}


def test_check_drift_reports_in_sync() -> None:
    report = check_drift()
    assert report.in_sync, report.format_text()


def test_executor_registry_covers_every_step_type() -> None:
    assert {st.value for st in executor_class_names()} == ALL_TYPES


def test_description_subclasses_cover_every_step_type() -> None:
    assert {st.value for st in discover_step_classes()} == ALL_TYPES


def test_union_covers_every_step_type() -> None:
    """Regression guard for the AnyStepDescription drift bug (3 types were missing)."""
    assert _union_types() == ALL_TYPES


def test_catalog_covers_every_step_type() -> None:
    assert set(get_step_catalog().names()) == ALL_TYPES


@pytest.mark.parametrize("step_type", list(StepType), ids=lambda s: s.value)
def test_from_dict_dispatch_handles_every_type(step_type: StepType) -> None:
    """Every type's catalog example reconstructs through the registry-driven
    ``_create_typed_step_from_dict`` — proves the dispatch has no missing branch."""
    example = get_step_catalog().step(step_type.value).example
    step = _create_typed_step_from_dict(example)
    assert step.step_type == step_type


def test_step_maps_are_complete() -> None:
    desc_map, cfg_map = _step_maps()
    assert set(desc_map) == ALL_TYPES
    # Only the inlined llm type has no config class.
    inlined = {k for k, v in cfg_map.items() if v is None}
    assert inlined == {"llm"}
