"""The catalog's examples must *run*, not merely validate.

Every other test that touches ``meta/_curated.py`` checks the examples against a
schema or renders them to a diagram. None of them ever executed one, and that
gap hid a dead template for as long as the catalog has existed: the ``transform``
example carried ``"Summary: {value}"`` while the executor renders the template
with ``output_format.format(input=input_data)``, so ``str.format`` raised
``KeyError: 'value'`` on contact.

Nobody noticed because of *where* it failed. The step dies in 0.00s with no
network call, so the chain simply reports one fewer completed step — but only
after the upstream steps have already run and been paid for. Measured on a
downstream consumer (EvoC, which inserts catalog-generated steps as evolutionary
mutations and pays a real model per step): 204 of 209 chain failures in a single
run were this one template, every one of them a ``1/2 steps`` chain that had
already bought an LLM response for step 1.

So: build a chain from each curated example and execute it. A schema check
cannot see a format string that names a key nobody binds.
"""

from __future__ import annotations

import pytest

from mmar_carl import (
    LLMClientBase,
    ReasoningContext,
    auto_generate_minimal_step,
    get_step_catalog,
    validate_and_build,
)


#: JSON, so a `structured_output` step has something parseable to work with.
#: Any step that only wants prose reads it as prose.
_REPLY = '{"result": "ok"}'


class _Stub(LLMClientBase):
    """Answers anything, so a failure here is structural rather than a model's."""

    async def get_response(self, prompt: str) -> str:
        return _REPLY

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        return _REPLY

    async def get_response_with_usage(self, prompt: str, retries: int = 3):
        return _REPLY, {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}


async def _run(spec: dict):
    report, chain = validate_and_build(spec, quiet=True)
    assert report.valid and chain is not None, f"spec did not build: {report}"
    # `execute_async`, not `execute`: the sync fallback nests an event loop
    # inside this one and warns that it can deadlock httpx-backed clients.
    return await chain.execute_async(ReasoningContext(outer_context="x", api=_Stub()))


@pytest.mark.asyncio
async def test_generated_transform_step_executes() -> None:
    """The regression itself: a `transform` after an `llm` must survive contact."""
    spec = {"steps": [
        auto_generate_minimal_step("llm", number=1),
        auto_generate_minimal_step("transform", number=2, dependencies=[1]),
    ]}
    result = await _run(spec)
    failures = [r for r in result.step_results if not r.success]
    assert not failures, [
        (r.step_number, r.step_title, r.error_message) for r in failures
    ]


def test_the_transform_example_binds_only_names_the_executor_supplies() -> None:
    """States the contract in the direction that broke, so a rename fails here.

    ``_execute_transform`` renders with ``format(input=...)`` and nothing else.
    A template naming any other field is a `KeyError` the moment it runs — which
    is a step failure, not a validation error, and therefore invisible to every
    schema-level test in this directory.
    """
    example = get_step_catalog().step("transform").example
    template = example["step_config"]["output_format"]
    assert template.format(input="probe") == "Summary: probe"


@pytest.mark.asyncio
@pytest.mark.parametrize("step_type", ["llm", "memory", "transform", "structured_output"])
async def test_minimal_steps_execute_standalone(step_type: str) -> None:
    """The insertable types a mutation operator can splice in unattended.

    Restricted to the types that need no ambient registration: `tool` wants a
    registered tool and `evaluation` wants something to score, so their failure
    without one is a real error rather than a dead template.
    """
    result = await _run({"steps": [auto_generate_minimal_step(step_type, number=1)]})
    failures = [r for r in result.step_results if not r.success]
    assert not failures, [
        (r.step_number, r.step_title, r.error_message) for r in failures
    ]
