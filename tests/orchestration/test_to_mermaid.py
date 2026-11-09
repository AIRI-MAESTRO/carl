"""Tests for ``ReasoningChain.to_mermaid()`` — the basic flowchart export.

Regression guard for the label line-break: Mermaid renders ``<br/>`` as a line
break inside a node label but shows a bare ``\\n`` as the literal two-character
escape, so the multi-line ``"N: title<br/>type"`` labels must use ``<br/>`` (the
same convention as ``to_mermaid_heatmap`` / ``to_lineage_mermaid``). A literal
``\\n`` here previously rendered as visible ``\\n`` text in the diagram.
"""

from __future__ import annotations

from mmar_carl import LLMStepDescription, ReasoningChain
from mmar_carl.models.config import TransformStepConfig
from mmar_carl.models.steps import MemoryStepDescription, TransformStepDescription
from mmar_carl.models.config import MemoryStepConfig
from mmar_carl.models.enums import MemoryOperation


def _linear() -> ReasoningChain:
    return ReasoningChain(steps=[
        LLMStepDescription(number=1, title="Gather data", aim="g"),
        LLMStepDescription(number=2, title="Analyse", aim="a", dependencies=[1]),
        LLMStepDescription(number=3, title="Summarise", aim="s", dependencies=[2]),
    ])


class TestStructure:
    def test_starts_with_flowchart(self):
        assert _linear().to_mermaid().startswith("flowchart TD")

    def test_one_node_per_step(self):
        out = _linear().to_mermaid()
        assert 'S1["' in out and 'S2["' in out and 'S3["' in out

    def test_dependency_edges(self):
        assert "S1 --> S2" in _linear().to_mermaid()
        assert "S2 --> S3" in _linear().to_mermaid()

    def test_linear_fallback_edges_when_no_deps(self):
        chain = ReasoningChain(steps=[
            LLMStepDescription(number=1, title="a", aim="a"),
            LLMStepDescription(number=2, title="b", aim="b"),
        ])
        assert "S1 --> S2" in chain.to_mermaid()  # implicit linear edge


class TestLabelLineBreak:
    def test_uses_br_not_literal_backslash_n(self):
        out = _linear().to_mermaid()
        label_lines = [ln for ln in out.splitlines() if ln.lstrip().startswith("S") and "[" in ln]
        assert label_lines
        for ln in label_lines:
            assert "<br/>" in ln, f"label must use <br/>, got: {ln}"
            assert "\\n" not in ln, f"label must not contain a literal backslash-n, got: {ln}"

    def test_type_label_present_after_break(self):
        out = _linear().to_mermaid()
        assert "<br/>🧠 LLM" in out  # "N: title<br/>🧠 LLM"


class TestStepTypeStyling:
    def test_classdefs_only_for_used_types(self):
        chain = ReasoningChain(steps=[
            LLMStepDescription(number=1, title="a", aim="a"),
            TransformStepDescription(number=2, title="t", dependencies=[1],
                                     config=TransformStepConfig(transform_type="format",
                                                                input_key="$history[-1]", output_format="{value}")),
            MemoryStepDescription(number=3, title="m", dependencies=[2],
                                  config=MemoryStepConfig(operation=MemoryOperation.WRITE,
                                                          memory_key="k", value_source="$history[-1]")),
        ])
        out = chain.to_mermaid()
        assert "classDef llm" in out and "classDef transform" in out and "classDef memory" in out
        assert "classDef tool" not in out          # tool never appears in this chain
        # each type's label uses <br/>
        for ln in out.splitlines():
            if ln.lstrip().startswith("S") and "[" in ln:
                assert "<br/>" in ln and "\\n" not in ln
