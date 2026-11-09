"""Hermetic tests for the decision-model step (``StepType.DECISION``).

Every HTTP exchange goes through ``httpx.MockTransport``: the tests pin the
wire shape sent to each provider (OpenAI / Azure OpenAI Decisions API and the
OpenRouter Decisions API that serves Jev), the normalised answer shape, the
fail-closed policy for refusals / low confidence / malformed answers, retries,
credential handling, preflight, and routing a chain on a decision.
"""

from __future__ import annotations

import json
from typing import Any, Callable

import httpx
import pytest
from pydantic import ValidationError

from mmar_carl import (
    ConditionalBranch,
    ConditionalStepConfig,
    ConditionalStepDescription,
    DecisionChoice,
    DecisionLevel,
    DecisionQuestion,
    DecisionStepConfig,
    DecisionStepDescription,
    DecisionStepExecutor,
    Language,
    ReasoningChain,
    ReasoningContext,
    StepType,
    ToolStepConfig,
    ToolStepDescription,
)
from mmar_carl.step_executors import get_executor, register_executor
from tests.mocks import MockLLMClient

API_KEY = "test-decision-key-not-real"

DEPARTMENT = DecisionQuestion(
    name="department",
    type="choice",
    instructions="Which department should handle this complaint?",
    choices=[
        DecisionChoice(value="billing", description="Payments and refunds."),
        DecisionChoice(value="technical", description="Product problems."),
        DecisionChoice(value="other", description="All other requests."),
    ],
)
REFUND = DecisionQuestion(
    name="refund_requested", type="predicate", instructions="Does the customer ask for a refund?"
)
SEVERITY = DecisionQuestion(
    name="severity",
    type="score",
    instructions="How severe is this issue?",
    levels=[
        DecisionLevel(label="Cosmetic", description="No lost functionality."),
        DecisionLevel(label="Workaround available"),
        DecisionLevel(label="Fully blocked", description="No workaround."),
    ],
)


def _config(**overrides: Any) -> DecisionStepConfig:
    values: dict[str, Any] = {
        "provider": "openai",
        "model": "gpt-6-luna",
        "questions": [DEPARTMENT, REFUND, SEVERITY],
        "output_memory_key": "ticket",
    }
    values.update(overrides)
    return DecisionStepConfig(**values)


def _step(config: DecisionStepConfig, number: int = 1) -> DecisionStepDescription:
    return DecisionStepDescription(number=number, title="Triage ticket", config=config)


def _context(outer: Any = "I was charged twice for my order.") -> ReasoningContext:
    return ReasoningContext(
        outer_context=outer, api=MockLLMClient(), model="test", language=Language.ENGLISH,
        decision_endpoints={
            "openai": ("https://api.openai.com/v1/decisions", "OPENAI_API_KEY"),
            "openrouter": ("https://openrouter.ai/api/alpha/decisions", "OPENROUTER_API_KEY"),
            "azure_openai": ("https://res.openai.azure.com/openai/v1/decisions", "AZURE_OPENAI_API_KEY"),
        },
    )


class Recorder:
    """Mock transport that records requests and replays queued responses."""

    def __init__(self, *responses: httpx.Response | Callable[[httpx.Request], httpx.Response]) -> None:
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        response = self.responses.pop(0)
        return response(request) if callable(response) else response

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self)

    def payload(self, index: int = -1) -> dict[str, Any]:
        return json.loads(self.requests[index].content)


def _executor(recorder: Recorder, sleeps: list[float] | None = None) -> DecisionStepExecutor:
    async def fake_sleep(delay: float) -> None:
        if sleeps is not None:
            sleeps.append(delay)

    return DecisionStepExecutor(transport=recorder.transport, sleep=fake_sleep)


OPENAI_BODY = {
    "model": "gpt-6-luna-2026-10-01",
    "answers": [
        {
            "type": "choice",
            "name": "department",
            "choice": "billing",
            "probabilities": [
                {"value": "billing", "probability": 0.95},
                {"value": "technical", "probability": 0.03},
                {"value": "other", "probability": 0.02},
            ],
            "confidence": 0.93,
        },
        {"type": "predicate", "name": "refund_requested", "probability": 0.91},
        {
            "type": "score",
            "name": "severity",
            "score": 1.1,
            "probabilities": [
                {"value": 0, "label": "Cosmetic", "probability": 0.1},
                {"value": 1, "label": "Workaround available", "probability": 0.7},
                {"value": 2, "label": "Fully blocked", "probability": 0.2},
            ],
            "confidence": 0.55,
        },
    ],
    "usage": {"input_tokens": 162, "output_tokens": 0, "total_tokens": 162},
}

OPENROUTER_BODY = {
    "id": "dec-123",
    "provider": "TypeSafe",
    "model": "typesafe/jev-1.14-20261001",
    "answers": {
        "department": {
            "type": "choice",
            "choice": "billing",
            "probabilities": {"billing": 0.9, "technical": 0.06, "other": 0.04},
            "confidence": 0.88,
        },
        "refund_requested": {"type": "noul", "noul": 0.42},
        "severity": {
            "type": "score",
            "score": 0.2,
            "legend": {"0": "Cosmetic", "1": "Workaround available", "2": "Fully blocked"},
            "probabilities": {"0": 0.8, "1": 0.2, "2": 0.0},
            "confidence": 0.7,
        },
    },
    "usage": {"input_tokens": 447, "output_tokens": 0, "cost": 0.000019},
}


@pytest.fixture(autouse=True)
def _api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("OPENAI_API_KEY", "AZURE_OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.setenv(name, API_KEY)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestDecisionConfigValidation:
    def test_noul_is_accepted_as_predicate_alias(self) -> None:
        question = DecisionQuestion.model_validate({"name": "spam", "type": "noul", "instructions": "Is it spam?"})
        assert question.type == "predicate"

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"type": "choice", "choices": [{"value": "a"}]}, "2–255 choices"),
            ({"type": "choice", "choices": [{"value": "a"}, {"value": "a"}]}, "duplicate"),
            ({"type": "choice", "choices": [{"value": "a"}, {"value": "b"}], "fallback": "c"}, "fallback"),
            ({"type": "score", "levels": [{"label": "only"}]}, "2–10 levels"),
            ({"type": "score", "levels": [{"label": "a"}, {"label": "b"}], "fallback": "a"}, "fallback"),
            ({"type": "predicate", "min_confidence": 0.5}, "threshold"),
            ({"type": "predicate", "fallback": "yes"}, "bool"),
            ({"type": "predicate", "choices": [{"value": "a"}, {"value": "b"}]}, "no choices"),
        ],
    )
    def test_question_shape_is_validated(self, kwargs: dict[str, Any], match: str) -> None:
        with pytest.raises(ValidationError, match=match):
            DecisionQuestion(name="q", instructions="Question?", **kwargs)

    def test_question_name_must_be_identifier(self) -> None:
        with pytest.raises(ValidationError, match="safe identifier"):
            DecisionQuestion(name="not a name", type="predicate", instructions="Q?")

    @pytest.mark.parametrize(
        "overrides, match",
        [
            ({"state_source": "$outer_context", "state_template": "x"}, "not both"),
            ({"state_source": "outer_context"}, "'\\$' reference"),
            ({"input_mapping": {"a": "$outer_context"}}, "requires state_template"),
            ({"questions": [REFUND, REFUND]}, "unique"),
            ({"primary_question": "missing"}, "primary_question"),
            ({"provider": "azure_openai"}, "require base_url"),
            ({"base_url": "http://decisions.example.com/v1"}, "https"),
            ({"base_url": "https://user:pw@decisions.example.com/v1"}, "credentials"),
            ({"base_url": "https://decisions.example.com/v1?key=1"}, "query"),
            ({"api_key_env": "sk-live-123"}, "environment variable name"),
            ({"questions": []}, "at least 1"),
        ],
    )
    def test_step_config_is_validated(self, overrides: dict[str, Any], match: str) -> None:
        with pytest.raises(ValidationError, match=match):
            _config(**overrides)

    def test_localhost_http_and_defaults(self) -> None:
        config = _config(base_url="http://localhost:8080/v1", provider="openrouter")
        assert config.resolved_api_key_env == "OPENROUTER_API_KEY"
        assert config.resolved_primary_question == "department"


# ---------------------------------------------------------------------------
# OpenAI / Azure OpenAI protocol
# ---------------------------------------------------------------------------


class TestOpenAIProtocol:
    async def test_request_shape_and_normalised_answers(self) -> None:
        recorder = Recorder(httpx.Response(200, json=OPENAI_BODY))
        context = _context(json.dumps({"message": "I was charged twice.", "order": 42}))

        result = await _executor(recorder).execute(_step(_config()), context)

        assert result.success, result.error_message
        request = recorder.requests[0]
        assert str(request.url) == "https://api.openai.com/v1/decisions"
        assert request.headers["authorization"] == f"Bearer {API_KEY}"
        payload = recorder.payload()
        assert payload["model"] == "gpt-6-luna"
        # Structured state is sent as JSON text on the OpenAI protocol.
        assert json.loads(payload["input"]) == {"message": "I was charged twice.", "order": 42}
        assert payload["questions"][0] == {
            "type": "choice",
            "name": "department",
            "instructions": "Which department should handle this complaint?",
            "choices": [
                {"value": "billing", "description": "Payments and refunds."},
                {"value": "technical", "description": "Product problems."},
                {"value": "other", "description": "All other requests."},
            ],
        }
        assert payload["questions"][1] == {
            "type": "predicate",
            "name": "refund_requested",
            "instructions": "Does the customer ask for a refund?",
        }
        assert payload["questions"][2]["levels"][1] == {"label": "Workaround available"}

        answers = result.result_data["answers"]
        assert answers["department"]["value"] == "billing"
        assert answers["department"]["probabilities"] == {"billing": 0.95, "technical": 0.03, "other": 0.02}
        assert answers["department"]["confidence"] == 0.93
        assert answers["refund_requested"]["value"] is True
        assert answers["refund_requested"]["probability"] == 0.91
        assert answers["severity"]["value"] == pytest.approx(1.1)
        assert answers["severity"]["level"] == "Workaround available"
        assert answers["severity"]["probabilities"]["Fully blocked"] == 0.2

        assert result.step_type == StepType.DECISION
        assert result.result == "billing"
        assert result.model == "gpt-6-luna-2026-10-01"
        assert result.result_data["requested_model"] == "gpt-6-luna"
        assert result.result_data["decision"] == "billing"
        assert result.token_usage == {"prompt": 162, "completion": 0, "total": 162}
        assert context.memory_read("ticket", namespace="decision") == {
            "department": "billing",
            "refund_requested": True,
            "severity": pytest.approx(1.1),
        }
        assert "department=billing (confidence=0.93)" in result.updated_history[-1]
        assert "refund_requested=true (p=0.91)" in result.updated_history[-1]

    async def test_azure_uses_api_key_header_and_deployment_url(self) -> None:
        recorder = Recorder(httpx.Response(200, json=OPENAI_BODY))
        config = _config(provider="azure_openai", base_url="https://res.openai.azure.com/openai/v1/")

        result = await _executor(recorder).execute(_step(config), _context())

        assert result.success, result.error_message
        request = recorder.requests[0]
        assert str(request.url) == "https://res.openai.azure.com/openai/v1/decisions"
        assert request.headers["api-key"] == API_KEY
        assert "authorization" not in request.headers

    async def test_predicate_threshold_and_primary_question(self) -> None:
        recorder = Recorder(httpx.Response(200, json=OPENAI_BODY))
        strict_refund = REFUND.model_copy(update={"threshold": 0.95})
        config = _config(questions=[DEPARTMENT, strict_refund, SEVERITY], primary_question="refund_requested")

        result = await _executor(recorder).execute(_step(config), _context())

        assert result.success
        assert result.result == "false"
        assert result.result_data["answers"]["refund_requested"]["value"] is False

    async def test_refusal_fails_closed_by_default(self) -> None:
        body = {**OPENAI_BODY, "answers": [{"type": "refusal", "name": "department"}, *OPENAI_BODY["answers"][1:]]}
        recorder = Recorder(httpx.Response(200, json=body))
        context = _context()

        result = await _executor(recorder).execute(_step(_config()), context)

        assert not result.success
        assert "department (refused)" in result.error_message
        assert result.result_data["answers"]["department"]["refused"] is True
        assert context.memory_read("ticket", namespace="decision") is None

    async def test_refusal_uses_fallback(self) -> None:
        body = {**OPENAI_BODY, "answers": [{"type": "refusal", "name": "department"}, *OPENAI_BODY["answers"][1:]]}
        recorder = Recorder(httpx.Response(200, json=body))
        with_fallback = DEPARTMENT.model_copy(update={"fallback": "other"})

        result = await _executor(recorder).execute(
            _step(_config(questions=[with_fallback, REFUND, SEVERITY])), _context()
        )

        assert result.success, result.error_message
        assert result.result == "other"
        answer = result.result_data["answers"]["department"]
        assert answer["fallback_applied"] and answer["uncertain"] and answer["refused"]
        assert result.result_data["uncertain_questions"] == ["department"]

    async def test_low_confidence_is_uncertain(self) -> None:
        recorder = Recorder(httpx.Response(200, json=OPENAI_BODY))
        picky = SEVERITY.model_copy(update={"min_confidence": 0.6})

        result = await _executor(recorder).execute(_step(_config(questions=[DEPARTMENT, REFUND, picky])), _context())

        assert not result.success
        assert "severity (low confidence)" in result.error_message

    async def test_uncertainty_can_be_tolerated(self) -> None:
        body = {**OPENAI_BODY, "answers": OPENAI_BODY["answers"][:2]}  # severity missing → refused
        recorder = Recorder(httpx.Response(200, json=body))

        result = await _executor(recorder).execute(
            _step(_config(fail_on_uncertain=False, primary_question="severity")), _context()
        )

        assert result.success
        assert result.result == "refused"
        assert result.result_data["answers"]["severity"]["value"] is None
        assert "[uncertain]" in result.updated_history[-1]

    @pytest.mark.parametrize(
        "answer, match",
        [
            ({"type": "choice", "name": "department", "choice": "legal", "confidence": 0.9}, "unknown choice"),
            ({"type": "predicate", "name": "refund_requested", "probability": 1.4}, "outside \\[0, 1\\]"),
            ({"type": "score", "name": "severity", "score": 3.5}, "outside \\[0, 2\\]"),
            ({"type": "score", "name": "severity", "score": "high"}, "not a finite number"),
        ],
    )
    async def test_malformed_answers_fail(self, answer: dict[str, Any], match: str) -> None:
        answers = [a for a in OPENAI_BODY["answers"] if a["name"] != answer["name"]] + [answer]
        recorder = Recorder(httpx.Response(200, json={**OPENAI_BODY, "answers": answers}))

        result = await _executor(recorder).execute(_step(_config()), _context())

        assert not result.success
        assert result.error_message is not None
        import re

        assert re.search(match, result.error_message), result.error_message


# ---------------------------------------------------------------------------
# OpenRouter protocol (Jev and other decision models)
# ---------------------------------------------------------------------------


class TestOpenRouterProtocol:
    async def test_request_shape_and_normalised_answers(self) -> None:
        recorder = Recorder(httpx.Response(200, json=OPENROUTER_BODY))
        config = _config(provider="openrouter", model="typesafe/jev-1.14")
        context = _context(json.dumps({"message": "I was charged twice.", "order": 42}))

        result = await _executor(recorder).execute(_step(config), context)

        assert result.success, result.error_message
        request = recorder.requests[0]
        assert str(request.url) == "https://openrouter.ai/api/alpha/decisions"
        assert request.headers["authorization"] == f"Bearer {API_KEY}"
        payload = recorder.payload()
        # JSON state stays structured for Jev.
        assert payload["state"] == {"message": "I was charged twice.", "order": 42}
        assert payload["questions"]["department"] == {
            "type": "choice",
            "instructions": "Which department should handle this complaint?",
            "criteria": {
                "billing": "Payments and refunds.",
                "technical": "Product problems.",
                "other": "All other requests.",
            },
        }
        assert payload["questions"]["refund_requested"] == {
            "type": "noul",
            "instructions": "Does the customer ask for a refund?",
        }
        assert payload["questions"]["severity"]["criteria"] == [
            "Cosmetic: No lost functionality.",
            "Workaround available",
            "Fully blocked: No workaround.",
        ]

        answers = result.result_data["answers"]
        assert answers["department"]["value"] == "billing"
        assert answers["refund_requested"]["value"] is False
        assert answers["refund_requested"]["probability"] == 0.42
        assert answers["severity"]["level"] == "Cosmetic"
        assert answers["severity"]["probabilities"] == {
            "Cosmetic": 0.8,
            "Workaround available": 0.2,
            "Fully blocked": 0.0,
        }
        assert result.result_data["cost_usd"] == pytest.approx(0.000019)
        assert result.result_data["request_id"] == "dec-123"
        assert result.model == "typesafe/jev-1.14-20261001"
        assert result.token_usage["prompt"] == 447

    async def test_answers_without_wrapper_are_accepted(self) -> None:
        recorder = Recorder(httpx.Response(200, json=OPENROUTER_BODY["answers"]))
        config = _config(provider="openrouter", model="typesafe/jev-1.14")

        result = await _executor(recorder).execute(_step(config), _context())

        assert result.success, result.error_message
        assert result.result == "billing"
        assert result.model == "typesafe/jev-1.14"

    async def test_template_state_and_list_probabilities(self) -> None:
        body = {
            "answers": {
                "department": {"type": "choice", "choice": "other", "probabilities": [0.1, 0.1, 0.8], "confidence": 0.8}
            }
        }
        recorder = Recorder(httpx.Response(200, json=body))
        config = _config(
            provider="openrouter",
            model="~typesafe/jev-latest",
            questions=[DEPARTMENT],
            state_template="Customer {name} wrote: {message}",
            input_mapping={"name": "'Ann'", "message": "$outer_context"},
        )

        result = await _executor(recorder).execute(_step(config), _context("Where is my parcel?"))

        assert result.success, result.error_message
        assert recorder.payload()["state"] == "Customer Ann wrote: Where is my parcel?"
        assert result.result_data["answers"]["department"]["probabilities"]["other"] == 0.8


# ---------------------------------------------------------------------------
# Transport, credentials, limits
# ---------------------------------------------------------------------------


class TestTransport:
    async def test_retries_rate_limit_then_succeeds(self) -> None:
        recorder = Recorder(
            httpx.Response(429, headers={"retry-after": "2"}, json={"error": {"message": "slow down"}}),
            httpx.Response(503, json={"error": "busy"}),
            httpx.Response(200, json=OPENAI_BODY),
        )
        sleeps: list[float] = []

        result = await _executor(recorder, sleeps).execute(_step(_config()), _context())

        assert result.success, result.error_message
        assert len(recorder.requests) == 3
        assert sleeps == [2.0, 1.0]

    async def test_auth_error_is_not_retried_and_does_not_leak_key(self) -> None:
        recorder = Recorder(httpx.Response(401, json={"error": {"message": API_KEY}}))

        result = await _executor(recorder).execute(_step(_config()), _context())

        assert not result.success
        assert len(recorder.requests) == 1
        assert "HTTP 401" in result.error_message
        assert API_KEY not in result.error_message

    async def test_retries_exhausted_on_network_error(self) -> None:
        def boom(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused", request=request)

        recorder = Recorder(boom, boom)

        result = await _executor(recorder).execute(_step(_config(max_retries=1)), _context())

        assert not result.success
        assert "ConnectError" in result.error_message
        assert len(recorder.requests) == 2

    async def test_missing_key_fails_before_any_request(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY")
        recorder = Recorder()

        result = await _executor(recorder).execute(_step(_config()), _context())

        assert not result.success
        assert "$OPENAI_API_KEY" in result.error_message
        assert recorder.requests == []

    async def test_custom_key_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TEAM_DECISIONS_KEY", "team-key")
        recorder = Recorder(httpx.Response(200, json=OPENAI_BODY))

        context = _context()
        context.decision_endpoints["openai"] = ("https://api.openai.com/v1/decisions", "TEAM_DECISIONS_KEY")
        result = await _executor(recorder).execute(_step(_config(api_key_env="TEAM_DECISIONS_KEY")), context)

        assert result.success
        assert recorder.requests[0].headers["authorization"] == "Bearer team-key"

    async def test_state_size_limit(self) -> None:
        recorder = Recorder()

        result = await _executor(recorder).execute(_step(_config(max_state_bytes=1_024)), _context("x" * 2_000))

        assert not result.success
        assert "max_state_bytes" in result.error_message
        assert recorder.requests == []

    async def test_empty_state_fails(self) -> None:
        recorder = Recorder()

        result = await _executor(recorder).execute(_step(_config()), _context(""))

        assert not result.success
        assert "empty value" in result.error_message

    async def test_non_json_body_fails(self) -> None:
        recorder = Recorder(httpx.Response(200, text="<html>gateway</html>"))

        result = await _executor(recorder).execute(_step(_config()), _context())

        assert not result.success
        assert "non-JSON" in result.error_message


# ---------------------------------------------------------------------------
# Chain integration
# ---------------------------------------------------------------------------


@pytest.fixture
def mocked_decision_executor() -> Any:
    original = get_executor(StepType.DECISION)
    recorder = Recorder(httpx.Response(200, json=OPENAI_BODY))
    register_executor(StepType.DECISION, _executor(recorder))
    yield recorder
    register_executor(StepType.DECISION, original)


class TestChainIntegration:
    def test_executor_is_registered(self) -> None:
        assert isinstance(get_executor(StepType.DECISION), DecisionStepExecutor)

    async def test_decision_routes_a_conditional_branch(self, mocked_decision_executor: Recorder) -> None:
        chain = ReasoningChain(
            steps=[
                _step(_config()),
                ConditionalStepDescription(
                    number=2,
                    title="Route by department",
                    dependencies=[1],
                    config=ConditionalStepConfig(
                        condition_context_key="$memory.decision.ticket",
                        branches=[
                            ConditionalBranch(condition="value['department'] == 'billing'", next_step=3),
                            ConditionalBranch(condition="value['department'] == 'technical'", next_step=4),
                        ],
                        default_step=4,
                    ),
                ),
                ToolStepDescription(
                    number=3, title="Billing", config=ToolStepConfig(tool_name="billing", input_mapping={})
                ),
                ToolStepDescription(
                    number=4, title="Support", config=ToolStepConfig(tool_name="support", input_mapping={})
                ),
            ],
            max_workers=1,
        )
        context = _context()
        context.register_tool("billing", lambda: "refund issued")
        context.register_tool("support", lambda: "ticket opened")

        result = await chain.execute_async(context)

        assert result.success
        routed = next(sr for sr in result.step_results if sr.step_number == 2)
        assert routed.result_data["next_step"] == 3
        assert len(mocked_decision_executor.requests) == 1

    def test_unwritten_state_reference_warns(self) -> None:
        with pytest.warns(UserWarning, match=r"\$memory\.triage\.ticket"):
            ReasoningChain(steps=[_step(_config(state_source="$memory.triage.ticket"))])

    def test_serialization_never_contains_credentials(self) -> None:
        chain = ReasoningChain(steps=[_step(_config(provider="openrouter", model="typesafe/jev-1.14"))])

        data = chain.to_dict()
        restored = ReasoningChain.from_dict(json.loads(json.dumps(data)))

        assert data["format_version"] == ReasoningChain.FORMAT_VERSION
        assert data["steps"][0]["step_type"] == "decision"
        assert API_KEY not in json.dumps(data)
        assert restored.steps[0].step_config == chain.steps[0].step_config

    def test_preflight_reports_missing_credentials(self, monkeypatch: pytest.MonkeyPatch) -> None:
        chain = ReasoningChain(
            steps=[
                _step(_config()),
                _step(_config(provider="openrouter", model="typesafe/jev-1.14"), number=2),
            ]
        )
        monkeypatch.delenv("OPENROUTER_API_KEY")

        report = chain.preflight(_context())

        assert chain.required_decision_credentials() == ["OPENAI_API_KEY", "OPENROUTER_API_KEY"]
        assert report.missing_decision_credentials == ["OPENROUTER_API_KEY"]
        assert not report.all_present
        assert "decision credential env: OPENROUTER_API_KEY" in report.format_text()
        assert API_KEY not in report.format_text()

    def test_cost_estimate_notes_external_pricing(self) -> None:
        chain = ReasoningChain(steps=[_step(_config())])

        estimate = chain.estimate_cost(_context())

        assert "pricing is provider-specific" in estimate.steps[0].note


@pytest.mark.asyncio
async def test_untrusted_chain_cannot_send_host_credentials() -> None:
    recorder = Recorder()
    step = _step(_config(base_url="https://untrusted.example/v1"))
    result = await _executor(recorder).execute(step, _context())
    assert not result.success
    assert "not authorized" in result.error_message
    assert not recorder.requests


@pytest.mark.parametrize("template", ["{field:1000000000}", "{field.__class__}", "{field!r}"])
def test_state_template_rejects_formatting_operations(template: str) -> None:
    with pytest.raises(ValueError, match="plain mapped placeholders"):
        _config(state_template=template, input_mapping={"field": "data"})


def test_decision_score_requires_unique_labels() -> None:
    with pytest.raises(ValueError, match="duplicate level labels"):
        DecisionQuestion(name="score", type="score", instructions="Rate it",
                         levels=[DecisionLevel(label="Same"), DecisionLevel(label="Same")])
