"""CapabilityLibrary injection + IndividualMetrics lineage substrate."""

from __future__ import annotations

from mmar_carl import CapabilityEntry, CapabilityLibrary, get_step_catalog
from mmar_carl.chain_evolution import IndividualMetrics
from mmar_carl.meta import MutationRecord


def test_capability_library_render() -> None:
    lib = CapabilityLibrary(
        entries=[
            CapabilityEntry(
                name="web_search", kind="tool", description="Search the web.", when_to_use="need fresh facts"
            ),
            CapabilityEntry(name="pdf", kind="skill", description="Extract PDF text."),
            CapabilityEntry(
                name="verify-before-answer", kind="concept", description="Add a check step before final output."
            ),
        ]
    )
    text = lib.render()
    assert "Tools:" in text and "web_search" in text and "need fresh facts" in text
    assert "Skills:" in text and "pdf" in text
    assert "Concepts:" in text
    assert lib.by_kind("tool")[0].name == "web_search"


def test_generation_prompt_includes_capabilities() -> None:
    lib = CapabilityLibrary(entries=[CapabilityEntry(name="calc", kind="tool", description="Arithmetic.")])
    prompt = get_step_catalog().generation_prompt(capability_library=lib)
    assert "Available capabilities" in prompt
    assert "calc" in prompt


def test_render_token_budget_truncates() -> None:
    lib = CapabilityLibrary(
        entries=[CapabilityEntry(name=f"t{i}", kind="tool", description="x" * 200) for i in range(50)]
    )
    assert "truncated" in lib.render(max_tokens=100)


def test_individual_metrics_lineage_fields() -> None:
    m = IndividualMetrics(
        score=0.8,
        id="abc123",
        parent_id="parent1",
        mutation_records=[
            MutationRecord(op="set_field", step_number=2, path="step_config.timeout", before=30, after=15)
        ],
    )
    # round-trips (the replayable substrate for a decision tree)
    restored = IndividualMetrics.model_validate_json(m.model_dump_json())
    assert restored.id == "abc123"
    assert restored.parent_id == "parent1"
    assert restored.mutation_records[0].op == "set_field"
    assert restored.mutation_records[0].after == 15


def test_skill_loader_as_capability_library() -> None:
    from mmar_carl import SkillLoader

    lib = SkillLoader().as_capability_library(search_paths=[])
    assert isinstance(lib, CapabilityLibrary)  # empty is fine when no skills installed
    assert all(e.kind == "skill" for e in lib.entries)
