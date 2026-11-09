"""Default LLM prompt wrappers leave the answer format to the step."""

import pytest

from mmar_carl import Language, PromptTemplate


@pytest.mark.parametrize(("language", "conflict"), [
    (Language.ENGLISH, "Response should be one paragraph"),
    (Language.RUSSIAN, "Ответ должен состоять из одного параграфа"),
])
def test_default_wrapper_does_not_override_step_output(language, conflict):
    contract = "Write one line of analysis, then a separate line beginning FINAL:"
    prompt = PromptTemplate().format_chain_prompt(
        "question", contract, history="previous evidence", language=language, system_prompt="system rule"
    )
    assert all(fragment in prompt for fragment in ("question", contract, "previous evidence", "system rule"))
    assert conflict not in prompt


def test_explicit_host_template_keeps_its_output_instruction():
    prompt = PromptTemplate(en_chain_template="{outer_context}\n{step_prompt}\nUse exactly one paragraph.")
    assert prompt.format_chain_prompt("input", "task", language=Language.ENGLISH).endswith("Use exactly one paragraph.")
