"""Unknown-kwarg visibility on typed step classes.

``StepDescriptionBase`` keeps pydantic's tolerant ``extra='ignore'``
behaviour (``extra='forbid'`` would break the
``Class(**step.model_dump())`` round-trip and LLM-planner parsing), but a
``model_validator(mode='before')`` now emits a ``UserWarning`` naming any
keys that are not model fields — so typos like ``llm_confg=`` or
unsupported kwargs (the historical ``llm_config`` on
``StructuredOutputStepDescription`` bug) are no longer silently swallowed.

Whitelisted keys that never warn:

- ``step_type`` — injected synthetically by every ``model_dump()`` override
- ``step_config`` — added by ``ReasoningChain.to_dict()`` serialization
"""

import warnings

import pytest

from mmar_carl import ReasoningChain
from mmar_carl.models.config import (
    ConditionalBranch,
    ConditionalStepConfig,
    LLMStepConfig,
    StructuredOutputStepConfig,
    ToolStepConfig,
)
from mmar_carl.models.steps import (
    ConditionalStepDescription,
    LLMStepDescription,
    StructuredOutputStepDescription,
    ToolStepDescription,
)


class TestUnknownKeyWarning:
    def test_typo_kwarg_warns(self):
        with pytest.warns(UserWarning, match=r"LLMStepDescription.*llm_confg"):
            step = LLMStepDescription(
                number=1,
                title="Analyze",
                aim="Analyze the data",
                llm_confg=LLMStepConfig(model="qwen/qwen3-8b"),  # typo: llm_config
            )
        # The typo'd value is still dropped — the warning is the only signal.
        assert step.llm_config is None

    def test_unsupported_kwarg_on_structured_output_warns(self):
        # The historical bug: llm_config passed to StructuredOutputStepDescription
        # was silently discarded because the class has no such field.
        if "llm_config" in StructuredOutputStepDescription.model_fields:
            pytest.skip("llm_config became a real field; this regression case no longer applies")
        with pytest.warns(UserWarning, match=r"StructuredOutputStepDescription.*llm_config"):
            StructuredOutputStepDescription(
                number=1,
                title="Extract",
                config=StructuredOutputStepConfig(output_schema={"type": "object"}),
                llm_config=LLMStepConfig(model="qwen/qwen3-8b"),
            )

    def test_multiple_unknown_keys_all_listed(self):
        with pytest.warns(UserWarning, match=r"\['alpha', 'zeta'\]"):
            LLMStepDescription(number=1, title="T", aim="A", zeta=1, alpha=2)

    def test_warning_fires_for_tool_step(self):
        with pytest.warns(UserWarning, match=r"ToolStepDescription.*retry_maxx"):
            ToolStepDescription(
                number=1,
                title="Fetch",
                config=ToolStepConfig(tool_name="fetch"),
                retry_maxx=3,
            )

    def test_warning_is_userwarning_not_deprecation(self):
        with pytest.warns(UserWarning) as record:
            LLMStepDescription(number=1, title="T", aim="A", bogus=True)
        categories = {w.category for w in record if "bogus" in str(w.message)}
        assert categories == {UserWarning}


class TestNoFalsePositives:
    def test_clean_construction_does_not_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            LLMStepDescription(
                number=1,
                title="Analyze",
                aim="Analyze the data",
                dependencies=[],
                llm_config=LLMStepConfig(model="qwen/qwen3-8b"),
                retry_max=2,
                timeout=30.0,
            )

    def test_model_dump_roundtrip_does_not_warn(self):
        # model_dump() injects a synthetic "step_type" key; the round-trip
        # Class(**step.model_dump()) must stay warning-free.
        step = LLMStepDescription(number=1, title="Analyze", aim="Analyze the data")
        dumped = step.model_dump()
        assert "step_type" in dumped
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            rebuilt = LLMStepDescription(**dumped)
        assert rebuilt.aim == step.aim

    def test_step_config_key_is_whitelisted(self):
        # ReasoningChain.to_dict() emits "step_config" alongside real fields.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ToolStepDescription(
                number=1,
                title="Fetch",
                config=ToolStepConfig(tool_name="fetch"),
                step_type="tool",
                step_config={"tool_name": "fetch"},
            )

    def test_chain_to_dict_from_dict_roundtrip_does_not_warn(self):
        chain = ReasoningChain(
            steps=[
                LLMStepDescription(number=1, title="Analyze", aim="Analyze the data"),
                ConditionalStepDescription(
                    number=2,
                    title="Route",
                    dependencies=[1],
                    config=ConditionalStepConfig(
                        branches=[ConditionalBranch(condition="contains:ok", next_step=3)],
                        default_step=3,
                    ),
                ),
                LLMStepDescription(number=3, title="Summarize", aim="Summarize", dependencies=[2]),
            ]
        )
        data = chain.to_dict()
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            restored = ReasoningChain.from_dict(data, use_typed_steps=True)
        assert len(restored.steps) == 3

    def test_validating_existing_instance_does_not_crash(self):
        step = LLMStepDescription(number=1, title="T", aim="A")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            revalidated = LLMStepDescription.model_validate(step)
        assert revalidated.number == 1

    def test_subclass_extra_fields_are_known(self):
        # Fields declared on a downstream subclass (the EvoC pattern) must not warn.
        class ExtendedStep(LLMStepDescription):
            evolution_tag: str = ""

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            step = ExtendedStep(number=1, title="T", aim="A", evolution_tag="gen-3")
        assert step.evolution_tag == "gen-3"
