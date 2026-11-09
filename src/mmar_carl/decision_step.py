"""
Decision-model step: ask a decision model typed questions about one state.

Decision models are a non-generative model family. They do not produce text
token by token; they read the whole state at once and return one typed answer
per question:

- ``predicate`` — probability that a condition holds (Jev calls it ``noul``)
- ``choice``    — one of the supplied values, a probability per value and a
  confidence
- ``score``     — probability-weighted index on an ordered rubric, a
  probability per level and a confidence

Two wire protocols are supported, selected by ``DecisionStepConfig.provider``:

- ``openai`` / ``azure_openai`` — OpenAI Decisions API (``POST /v1/decisions``,
  ``input`` + ``questions`` array, ``answers`` array with refusals)
- ``openrouter`` — OpenRouter Decisions API (``POST /api/alpha/decisions``,
  ``state`` + ``questions`` object keyed by name with ``criteria``); serves
  TypeSafe Jev and the other decision models listed by OpenRouter

Both are normalised into one answer shape in ``result_data["answers"]``, so a
chain can switch providers by changing ``provider`` and ``model`` only. Probability
scales are not comparable across models: re-tune ``threshold`` and
``min_confidence`` after a switch.

What the step does with an answer:

- ``result``        — the primary question's decided value as text (goes into history)
- ``result_data``   — provider, served model, normalised answers, usage, cost
- memory output     — ``output_memory_key`` writes ``{question: decided value}``
  for conditional routing (``$memory.decision.<key>``)
- fail-closed       — refusals, unknown choices, out-of-range numbers and
  answers below ``min_confidence`` fail the step unless the question has a
  ``fallback`` or ``fail_on_uncertain`` is False

Transport: one HTTPS request through ``httpx``. The API key is read from the host environment at run
time; serialized chains never contain it.

Known limitations: text and JSON state only (image parts and Jev's ``box``
question type are not wired yet); dependent questions must be split into
separate steps — every question in one step sees the same state.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import time
from string import Formatter
from typing import Any, Callable, Optional

from .models.config import DecisionQuestion, DecisionStepConfig
from .models.context import ReasoningContext
from .models.enums import Language, StepType
from .models.prompts import PromptTemplate
from .models.results import StepExecutionResult
from .models.steps import StepDescription
from .step_executors import StepExecutorBase, register_executor, resolve_context_reference

__all__ = [
    "DEFAULT_DECISION_BASE_URLS",
    "DecisionProviderError",
    "DecisionStepExecutor",
]

DEFAULT_DECISION_BASE_URLS: dict[str, str] = {
    "openai": "https://api.openai.com/v1",
    "openrouter": "https://openrouter.ai/api/alpha",
}

_RETRY_STATUSES = frozenset({408, 429, 500, 502, 503, 504})
_ERROR_DETAIL_LIMIT = 500
_MAX_BACKOFF_S = 8.0
_MAX_RETRY_AFTER_S = 30.0
# Score answers are probability-weighted averages; allow float noise at the ends.
_SCORE_TOLERANCE = 1e-6


class DecisionProviderError(RuntimeError):
    """The decision provider rejected the request or returned an unusable answer."""

    def __init__(self, message: str, *, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def _truncate(message: str) -> str:
    if len(message) <= _ERROR_DETAIL_LIMIT:
        return message
    return message[:_ERROR_DETAIL_LIMIT] + " …[truncated]"


def _as_probability(value: Any, *, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DecisionProviderError(f"{what} is not a number: {value!r}")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise DecisionProviderError(f"{what} is outside [0, 1]: {number!r}")
    return number


def _optional_probability(value: Any, *, what: str) -> Optional[float]:
    return None if value is None else _as_probability(value, what=what)


class DecisionStepExecutor(StepExecutorBase):
    """Executor that sends one decision request and normalises the typed answers.

    ``transport`` and ``sleep`` are injection seams for hermetic tests: pass an
    ``httpx.MockTransport`` and a no-op sleep to exercise the full request /
    retry / parsing path without network access.
    """

    def __init__(
        self,
        *,
        transport: Any = None,
        sleep: Callable[[float], Any] = asyncio.sleep,
    ) -> None:
        self._transport = transport
        self._sleep = sleep

    async def execute(
        self,
        step: StepDescription,
        context: ReasoningContext,
        prompt_template: Optional[PromptTemplate] = None,
    ) -> StepExecutionResult:
        start_time = time.time()
        config: DecisionStepConfig = step.step_config  # type: ignore[assignment]
        result_data: Optional[dict[str, Any]] = None

        try:
            endpoint = self._endpoint(config)
            key_env = config.resolved_api_key_env
            if context.decision_endpoints.get(config.provider) != (endpoint, key_env):
                raise DecisionProviderError("decision endpoint and credential are not authorized by the host")
            api_key = os.environ.get(key_env, "").strip()
            if not api_key:
                raise ValueError(
                    f"decision step needs an API key in ${config.resolved_api_key_env}"
                )
            state = self._resolve_state(config, context)
            payload = self._build_payload(config, state)
            body = await self._post(config, payload, api_key)

            answers = self._parse_answers(config, body)
            uncertain = self._apply_policy(config, answers)
            result_data = self._result_data(config, body, answers, uncertain)

            if uncertain and config.fail_on_uncertain:
                unresolved = [
                    name for name in uncertain if not answers[name]["fallback_applied"]
                ]
                if unresolved:
                    details = ", ".join(
                        f"{name} ({'refused' if answers[name]['refused'] else 'low confidence'})"
                        for name in unresolved
                    )
                    raise DecisionProviderError(f"decision model was uncertain: {details}")

            return self._success(step, context, config, body, answers, result_data, start_time)

        except Exception as exc:
            return self._failure(step, context, _truncate(str(exc)), start_time, result_data)

    # === State resolution ===

    @staticmethod
    def _resolve_state(config: DecisionStepConfig, context: ReasoningContext) -> Any:
        if config.state_template is not None:
            inputs: dict[str, str] = {}
            for name, source in config.input_mapping.items():
                is_reference = source.startswith("$") or (
                    len(source) >= 2 and source[0] == source[-1] and source[0] in "'\""
                )
                value = resolve_context_reference(source, context) if is_reference else source
                if value is None:
                    raise ValueError(f"decision input {name!r} resolved to an empty value")
                if not isinstance(value, str):
                    value = json.dumps(value, ensure_ascii=False, default=str)
                inputs[name] = value
            if not inputs:
                return config.state_template
            size = 0
            parts = []
            for literal, field, _, _ in Formatter().parse(config.state_template):
                value = inputs[field] if field is not None else ""
                size += len(literal.encode("utf-8")) + len(value.encode("utf-8"))
                if size > config.max_state_bytes:
                    raise ValueError("decision state exceeds max_state_bytes")
                parts.extend((literal, value))
            return "".join(parts)

        source = config.state_source or "$outer_context"
        value = resolve_context_reference(source, context)
        if value is None or (isinstance(value, (str, list, dict)) and not value):
            raise ValueError(f"decision state {source!r} resolved to an empty value")
        return value

    # === Request building ===

    def _build_payload(self, config: DecisionStepConfig, state: Any) -> dict[str, Any]:
        if config.provider == "openrouter":
            wire_state = self._openrouter_state(state)
            payload: dict[str, Any] = {
                "model": config.model,
                "state": wire_state,
                "questions": {q.name: self._openrouter_question(q) for q in config.questions},
            }
        else:
            wire_state = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, default=str)
            payload = {
                "model": config.model,
                "input": wire_state,
                "questions": [self._openai_question(q) for q in config.questions],
            }
        size = len(json.dumps(wire_state, ensure_ascii=False, default=str).encode("utf-8"))
        if size > config.max_state_bytes:
            raise ValueError(
                f"decision state is {size} bytes, exceeding max_state_bytes ({config.max_state_bytes})"
            )
        return payload

    @staticmethod
    def _openrouter_state(state: Any) -> Any:
        # OpenRouter accepts a string, a JSON object, or an array of text parts.
        if isinstance(state, (str, dict)):
            return state
        if isinstance(state, list) and all(isinstance(item, str) for item in state):
            return state
        return json.dumps(state, ensure_ascii=False, default=str)

    @staticmethod
    def _openai_question(question: DecisionQuestion) -> dict[str, Any]:
        wire: dict[str, Any] = {
            "type": question.type,
            "name": question.name,
            "instructions": question.instructions,
        }
        if question.type == "choice":
            wire["choices"] = [
                {"value": c.value, **({"description": c.description} if c.description else {})}
                for c in question.choices
            ]
        elif question.type == "score":
            wire["levels"] = [
                {"label": lv.label, **({"description": lv.description} if lv.description else {})}
                for lv in question.levels
            ]
        return wire

    @staticmethod
    def _openrouter_question(question: DecisionQuestion) -> dict[str, Any]:
        if question.type == "predicate":
            return {"type": "noul", "instructions": question.instructions}
        if question.type == "choice":
            return {
                "type": "choice",
                "instructions": question.instructions,
                "criteria": {c.value: c.description for c in question.choices},
            }
        return {
            "type": "score",
            "instructions": question.instructions,
            "criteria": [
                f"{lv.label}: {lv.description}" if lv.description else lv.label
                for lv in question.levels
            ],
        }

    # === Transport ===

    @staticmethod
    def _endpoint(config: DecisionStepConfig) -> str:
        base = config.base_url or DEFAULT_DECISION_BASE_URLS[config.provider]
        return base.rstrip("/") + "/decisions"

    @staticmethod
    def _headers(config: DecisionStepConfig, api_key: str) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if config.provider == "azure_openai":
            headers["api-key"] = api_key
        else:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    async def _post(self, config: DecisionStepConfig, payload: dict[str, Any], api_key: str) -> dict[str, Any]:
        import httpx  # noqa: PLC0415

        url = self._endpoint(config)
        headers = self._headers(config, api_key)
        client_kwargs: dict[str, Any] = {"timeout": config.timeout, "trust_env": False}
        if self._transport is not None:
            client_kwargs["transport"] = self._transport

        async with httpx.AsyncClient(**client_kwargs) as client:
            attempt = 0
            while True:
                try:
                    response = await client.post(url, json=payload, headers=headers)
                except httpx.TransportError as exc:
                    if attempt >= config.max_retries:
                        raise DecisionProviderError(
                            f"decision request failed: {type(exc).__name__}"
                        ) from exc
                    await self._sleep(min(0.5 * 2**attempt, _MAX_BACKOFF_S))
                    attempt += 1
                    continue

                if response.status_code in _RETRY_STATUSES and attempt < config.max_retries:
                    await self._sleep(self._retry_delay(response, attempt))
                    attempt += 1
                    continue
                if not response.is_success:
                    raise DecisionProviderError(
                        f"decision provider returned HTTP {response.status_code}",
                        status_code=response.status_code,
                    )
                try:
                    body = response.json()
                except ValueError as exc:
                    raise DecisionProviderError("decision provider returned non-JSON body") from exc
                if not isinstance(body, dict):
                    raise DecisionProviderError("decision provider returned a non-object body")
                return body

    @staticmethod
    def _retry_delay(response: Any, attempt: int) -> float:
        retry_after = response.headers.get("retry-after")
        if retry_after:
            try:
                return min(max(float(retry_after), 0.0), _MAX_RETRY_AFTER_S)
            except ValueError:
                pass
        return min(0.5 * 2**attempt, _MAX_BACKOFF_S)

    # === Answer parsing ===

    def _parse_answers(self, config: DecisionStepConfig, body: dict[str, Any]) -> dict[str, dict[str, Any]]:
        raw = body.get("answers")
        if config.provider == "openrouter":
            if raw is None:
                raw = body
            if not isinstance(raw, dict):
                raise DecisionProviderError("openrouter decision response has no answers object")
            by_name = raw
        else:
            if not isinstance(raw, list):
                raise DecisionProviderError("openai decision response has no answers array")
            by_name = {}
            for item in raw:
                if isinstance(item, dict) and isinstance(item.get("name"), str):
                    if item["name"] in by_name:
                        raise DecisionProviderError("decision response contains a duplicate answer")
                    by_name[item["name"]] = item

        parsed: dict[str, dict[str, Any]] = {}
        for question in config.questions:
            item = by_name.get(question.name)
            if not isinstance(item, dict) or item.get("type") == "refusal":
                parsed[question.name] = self._refused(question)
                continue
            parsed[question.name] = self._parse_one(question, item)
        return parsed

    @staticmethod
    def _refused(question: DecisionQuestion) -> dict[str, Any]:
        return {
            "type": question.type,
            "value": None,
            "probabilities": {},
            "confidence": None,
            "refused": True,
            "uncertain": True,
            "fallback_applied": False,
        }

    def _parse_one(self, question: DecisionQuestion, item: dict[str, Any]) -> dict[str, Any]:
        name = question.name
        answer: dict[str, Any] = {
            "type": question.type,
            "refused": False,
            "uncertain": False,
            "fallback_applied": False,
        }
        if question.type == "predicate":
            raw_probability = item.get("probability", item.get("noul"))
            probability = _as_probability(raw_probability, what=f"{name}.probability")
            answer.update(
                probability=probability,
                value=probability >= question.threshold,
                probabilities={"true": probability, "false": 1.0 - probability},
                confidence=None,
            )
            return answer

        if question.type == "choice":
            values = [c.value for c in question.choices]
            choice = item.get("choice")
            if choice not in values:
                raise DecisionProviderError(f"{name}: model returned unknown choice {choice!r}")
            answer.update(
                choice=choice,
                value=choice,
                probabilities=self._probabilities(item.get("probabilities"), values, what=name),
                confidence=_optional_probability(item.get("confidence"), what=f"{name}.confidence"),
            )
            return answer

        labels = [lv.label for lv in question.levels]
        score = item.get("score")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
            raise DecisionProviderError(f"{name}: score is not a finite number: {score!r}")
        top = len(labels) - 1
        if not -_SCORE_TOLERANCE <= score <= top + _SCORE_TOLERANCE:
            raise DecisionProviderError(f"{name}: score {score!r} is outside [0, {top}]")
        score = min(max(float(score), 0.0), float(top))
        answer.update(
            score=score,
            value=score,
            level=labels[int(round(score))],
            probabilities=self._probabilities(item.get("probabilities"), labels, what=name),
            confidence=_optional_probability(item.get("confidence"), what=f"{name}.confidence"),
        )
        return answer

    @staticmethod
    def _probabilities(raw: Any, keys: list[str], *, what: str) -> dict[str, float]:
        """Normalise provider probability shapes into ``{choice value | level label: p}``.

        Accepts a mapping keyed by value, label or level index; a list of
        ``{value|label, probability}`` objects (OpenAI); or a bare list of numbers
        in option order.
        """
        if raw is None:
            return {}
        result: dict[str, float] = {}

        def key_for(token: Any) -> Optional[str]:
            if isinstance(token, str) and token in keys:
                return token
            if isinstance(token, bool):
                return None
            index: Optional[int] = None
            if isinstance(token, int):
                index = token
            elif isinstance(token, str) and token.isdigit():
                index = int(token)
            if index is not None and 0 <= index < len(keys):
                return keys[index]
            return None

        if isinstance(raw, dict):
            items = list(raw.items())
        elif isinstance(raw, list):
            items = []
            for position, entry in enumerate(raw):
                if isinstance(entry, dict):
                    token = entry.get("label") if entry.get("label") in keys else entry.get("value")
                    items.append((token, entry.get("probability")))
                else:
                    items.append((position, entry))
        else:
            raise DecisionProviderError(f"{what}: probabilities have an unexpected shape")

        for token, probability in items:
            key = key_for(token)
            if key is None:
                raise DecisionProviderError(f"{what}: probability for unknown option {token!r}")
            result[key] = _as_probability(probability, what=f"{what}.probabilities[{key}]")
        return result

    # === Policy ===

    @staticmethod
    def _apply_policy(config: DecisionStepConfig, answers: dict[str, dict[str, Any]]) -> list[str]:
        """Mark uncertain answers and apply fallbacks; return the uncertain question names."""
        uncertain: list[str] = []
        for question in config.questions:
            answer = answers[question.name]
            low_confidence = question.min_confidence is not None and (
                answer.get("confidence") is None or answer["confidence"] < question.min_confidence
            )
            if not (answer["refused"] or low_confidence):
                continue
            answer["uncertain"] = True
            uncertain.append(question.name)
            if question.fallback is not None:
                answer["value"] = question.fallback
                answer["fallback_applied"] = True
        return uncertain

    # === Result assembly ===

    @staticmethod
    def _usage(body: dict[str, Any]) -> tuple[dict[str, int], Any, Optional[float]]:
        usage = body.get("usage")
        if not isinstance(usage, dict):
            return {}, None, None
        prompt = int(usage.get("input_tokens") or usage.get("inputTokens") or 0)
        completion = int(usage.get("output_tokens") or usage.get("outputTokens") or 0)
        total = int(usage.get("total_tokens") or prompt + completion)
        cost = usage.get("cost")
        cost_usd = float(cost) if isinstance(cost, (int, float)) and not isinstance(cost, bool) else None
        return {"prompt": prompt, "completion": completion, "total": total}, usage, cost_usd

    def _result_data(
        self,
        config: DecisionStepConfig,
        body: dict[str, Any],
        answers: dict[str, dict[str, Any]],
        uncertain: list[str],
    ) -> dict[str, Any]:
        _, raw_usage, cost_usd = self._usage(body)
        primary = config.resolved_primary_question
        return {
            "provider": config.provider,
            "requested_model": config.model,
            "model": body.get("model") or config.model,
            "request_id": body.get("id"),
            "primary_question": primary,
            "decision": answers[primary]["value"],
            "answers": answers,
            "uncertain_questions": uncertain,
            "usage": raw_usage,
            "cost_usd": cost_usd,
        }

    @staticmethod
    def _format_value(answer: dict[str, Any]) -> str:
        value = answer["value"]
        if value is None:
            return "refused"
        if isinstance(value, bool):
            return "true" if value else "false"
        if answer["type"] == "score" and not answer["fallback_applied"]:
            return f"{value:.2f} ({answer['level']})"
        return str(value)

    def _history_entry(
        self,
        step: StepDescription,
        context: ReasoningContext,
        answers: dict[str, dict[str, Any]],
    ) -> str:
        parts = []
        for name, answer in answers.items():
            detail = self._format_value(answer)
            if answer.get("probability") is not None:
                detail += f" (p={answer['probability']:.2f})"
            elif answer.get("confidence") is not None:
                detail += f" (confidence={answer['confidence']:.2f})"
            if answer["uncertain"]:
                detail += " [uncertain]"
            parts.append(f"{name}={detail}")
        summary = "; ".join(parts)
        if context.language == Language.ENGLISH:
            return f"Step {step.number}. {step.title} [DECISION]\nResult: {summary}\n"
        return f"Шаг {step.number}. {step.title} [DECISION]\nРезультат: {summary}\n"

    def _success(
        self,
        step: StepDescription,
        context: ReasoningContext,
        config: DecisionStepConfig,
        body: dict[str, Any],
        answers: dict[str, dict[str, Any]],
        result_data: dict[str, Any],
        start_time: float,
    ) -> StepExecutionResult:
        if config.output_memory_key:
            context.memory_write(
                config.output_memory_key,
                {name: answer["value"] for name, answer in answers.items()},
                namespace=config.output_namespace,
            )

        token_usage, _, _ = self._usage(body)
        updated_history = context.history.copy()
        updated_history.append(self._history_entry(step, context, answers))

        return StepExecutionResult(
            step_number=step.number,
            step_title=step.title,
            step_type=StepType.DECISION,
            result=self._format_value(answers[config.resolved_primary_question]),
            result_data=result_data,
            success=True,
            execution_time=time.time() - start_time,
            updated_history=updated_history,
            token_usage=token_usage,
            model=result_data["model"],
        )

    def _failure(
        self,
        step: StepDescription,
        context: ReasoningContext,
        error_message: str,
        start_time: float,
        result_data: Optional[dict[str, Any]],
    ) -> StepExecutionResult:
        return StepExecutionResult(
            step_number=step.number,
            step_title=step.title,
            step_type=StepType.DECISION,
            result="",
            result_data=result_data,
            success=False,
            error_message=error_message,
            execution_time=time.time() - start_time,
            updated_history=context.history.copy(),
        )


# Importing ``mmar_carl`` installs the built-in executor. Users can still
# replace it through the normal register_executor extension point.
register_executor(StepType.DECISION, DecisionStepExecutor())
