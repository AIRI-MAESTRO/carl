"""CodexStep PoC: local SDK delegation as a first-class CARL step."""

from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest
from pydantic import ValidationError

from mmar_carl import (
    CodexStepConfig,
    CodexStepDescription,
    CodexStepExecutor,
    Language,
    ReasoningChain,
    ReasoningContext,
    StepCache,
    StepType,
    check_codex_runtime,
    get_executor,
)
from mmar_carl import codex_step as codex_step_module


class _Sandbox:
    read_only = "read-only"
    workspace_write = "workspace-write"


class _ApprovalMode:
    deny_all = "deny-all"


class _FakeThread:
    def __init__(self, thread_id: str, owner: type["_FakeAsyncCodex"]) -> None:
        self.id = thread_id
        self._owner = owner

    async def run(self, prompt: str, **kwargs: Any) -> Any:
        self._owner.run_calls.append((self.id, prompt, kwargs))
        total = SimpleNamespace(input_tokens=17, output_tokens=5, total_tokens=22)
        usage = SimpleNamespace(total=total)
        return SimpleNamespace(
            id=f"turn-{len(self._owner.run_calls)}",
            status=SimpleNamespace(value="completed"),
            error=None,
            final_response=self._owner.response_override or f"Codex completed {self.id}",
            usage=usage,
        )


class _FakeAsyncCodex:
    start_calls: ClassVar[list[dict[str, Any]]] = []
    resume_calls: ClassVar[list[tuple[str, dict[str, Any]]]] = []
    run_calls: ClassVar[list[tuple[str, str, dict[str, Any]]]] = []
    response_override: ClassVar[str | None] = None

    @classmethod
    def reset(cls) -> None:
        cls.start_calls = []
        cls.resume_calls = []
        cls.run_calls = []
        cls.response_override = None

    async def __aenter__(self) -> "_FakeAsyncCodex":
        return self

    async def __aexit__(self, *_args: Any) -> None:
        return None

    async def thread_start(self, **kwargs: Any) -> _FakeThread:
        type(self).start_calls.append(kwargs)
        return _FakeThread(f"thread-{len(type(self).start_calls)}", type(self))

    async def thread_resume(self, thread_id: str, **kwargs: Any) -> _FakeThread:
        type(self).resume_calls.append((thread_id, kwargs))
        return _FakeThread(thread_id, type(self))


@pytest.fixture(autouse=True)
def fake_codex_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeAsyncCodex.reset()
    sdk = SimpleNamespace(
        AsyncCodex=_FakeAsyncCodex,
        Sandbox=_Sandbox,
        ApprovalMode=_ApprovalMode,
    )
    monkeypatch.setattr(codex_step_module, "_load_codex_sdk", lambda: sdk)


def _context() -> ReasoningContext:
    return ReasoningContext(
        outer_context='{"ticket":"CARL-42","priority":"high"}',
        api=object(),
        language=Language.ENGLISH,
    )


def test_executor_is_registered() -> None:
    assert isinstance(get_executor(StepType.CODEX), CodexStepExecutor)


def test_config_is_safe_by_default_and_rejects_full_access() -> None:
    config = CodexStepConfig(task="Inspect the repository")

    assert config.sandbox == "read-only"
    assert config.ephemeral is True

    with pytest.raises(ValidationError):
        CodexStepConfig(
            task="Inspect the repository",
            sandbox="full-access",  # type: ignore[arg-type]
        )


def test_codex_step_cannot_be_cached() -> None:
    with pytest.raises(ValidationError, match="cannot be cached"):
        CodexStepDescription(
            number=1,
            title="Unsafe cache",
            config=CodexStepConfig(task="Review files"),
            cache=StepCache(ttl=60),
        )


async def test_chain_runs_codex_and_resumes_thread() -> None:
    first = CodexStepDescription(
        number=1,
        title="Inspect issue",
        config=CodexStepConfig(
            task="Inspect this issue without changing files: {issue}",
            input_mapping={"issue": "$outer_context"},
            sandbox="read-only",
            reasoning_effort="medium",
            ephemeral=False,
            output_memory_key="inspection",
        ),
    )
    second = CodexStepDescription(
        number=2,
        title="Continue review",
        dependencies=[1],
        config=CodexStepConfig(
            task="Now summarize the inspection.",
            resume_session="$steps.1.result_data.session_id",
            sandbox="read-only",
        ),
    )
    context = _context()

    result = await ReasoningChain([first, second]).execute_async(context)

    assert result.success is True
    assert [item.step_type for item in result.step_results] == [
        StepType.CODEX,
        StepType.CODEX,
    ]
    assert result.step_results[0].result == "Codex completed thread-1"
    assert result.step_results[0].token_usage == {
        "prompt": 17,
        "completion": 5,
        "total": 22,
    }
    assert result.step_results[0].result_data["thread_id"] == "thread-1"
    assert result.step_results[0].result_data["session_id"] == "thread-1"
    assert context.memory_read("inspection", namespace="codex") == "Codex completed thread-1"

    start_kwargs = _FakeAsyncCodex.start_calls[0]
    assert start_kwargs["approval_mode"] == _ApprovalMode.deny_all
    assert start_kwargs["sandbox"] == _Sandbox.read_only
    assert start_kwargs["ephemeral"] is False
    assert _FakeAsyncCodex.resume_calls[0][0] == "thread-1"
    assert "CARL-42" in _FakeAsyncCodex.run_calls[0][1]
    assert _FakeAsyncCodex.run_calls[0][2]["effort"] == "medium"


async def test_missing_sdk_becomes_actionable_step_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing() -> Any:
        raise ImportError(
            "Optional dependency 'openai_codex' not found. Install with: "
            "pip install 'mmar-carl[codex]'"
        )

    monkeypatch.setattr(codex_step_module, "_load_codex_sdk", missing)
    step = CodexStepDescription(
        number=1,
        title="Codex",
        config=CodexStepConfig(task="Inspect files"),
    )

    result = await get_executor(StepType.CODEX).execute(step, _context())

    assert result.success is False
    assert "mmar-carl[codex]" in (result.error_message or "")


async def test_timeout_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    async def never_returns(**_kwargs: Any) -> tuple[str, Any]:
        await asyncio.sleep(1)
        raise AssertionError("unreachable")

    executor = CodexStepExecutor()
    monkeypatch.setattr(executor, "_run_turn", never_returns)
    step = CodexStepDescription(
        number=1,
        title="Slow Codex",
        config=CodexStepConfig(
            task="Inspect files",
            timeout=0.01,
        ),
    )

    result = await executor.execute(step, _context())

    assert result.success is False
    assert result.error_message == "Codex turn timed out after 0.01s"


@pytest.mark.live
async def test_live_local_codex_sdk_smoke(tmp_path: Any) -> None:
    """Opt-in feasibility probe using the host's existing Codex login."""

    if os.environ.get("CARL_CODEX_LIVE") != "1":
        pytest.skip("set CARL_CODEX_LIVE=1 to run the local Codex SDK smoke test")

    # Restore the real lazy loader hidden by the autouse hermetic fixture.
    from mmar_carl._optional_deps import require_openai_codex

    original_loader = codex_step_module._load_codex_sdk
    codex_step_module._load_codex_sdk = require_openai_codex
    try:
        step = CodexStepDescription(
            number=1,
            title="Live Codex probe",
            config=CodexStepConfig(
                task=(
                    "Reply with a short sentence containing CODEX_STEP_OK. "
                    "Do not modify files or run unnecessary commands."
                ),
                cwd=str(tmp_path),
                sandbox="read-only",
                timeout_seconds=180,
            ),
        )
        result = await get_executor(StepType.CODEX).execute(step, _context())
    finally:
        codex_step_module._load_codex_sdk = original_loader

    assert result.success is True, result.error_message
    assert "CODEX_STEP_OK" in result.result


# ---------------------------------------------------------------------------
# Runtime validation + preflight + session convention + output schema
# (same manner as the Claude Code step)
# ---------------------------------------------------------------------------


def _codex_chain(**config_kwargs: Any) -> ReasoningChain:
    config_kwargs.setdefault("task", "Inspect files")
    return ReasoningChain([
        CodexStepDescription(
            number=1,
            title="Codex",
            config=CodexStepConfig(**config_kwargs),
        ),
    ])


def test_check_codex_runtime_available() -> None:
    status = check_codex_runtime()

    assert status.available is True
    assert status.error is None


def test_check_codex_runtime_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing() -> Any:
        raise ImportError(
            "Optional dependency 'openai_codex' not found. Install with: "
            "pip install 'mmar-carl[codex]'"
        )

    monkeypatch.setattr(codex_step_module, "_load_codex_sdk", missing)

    status = check_codex_runtime()

    assert status.available is False
    assert "mmar-carl[codex]" in (status.error or "")


def test_preflight_reports_codex_runtime() -> None:
    report = _codex_chain().preflight(_context())

    assert report.required_codex_runtimes == ["openai-codex"]
    assert report.missing_codex_runtimes == []
    assert report.all_present is True
    assert "codex runtime" in report.format_text()


def test_preflight_flags_missing_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing() -> Any:
        raise ImportError("no sdk")

    monkeypatch.setattr(codex_step_module, "_load_codex_sdk", missing)

    report = _codex_chain().preflight(_context())

    assert report.missing_codex_runtimes == ["openai-codex"]
    assert report.all_present is False
    assert "codex runtime: openai-codex" in report.format_text()


async def test_thread_id_stored_and_resumed_from_memory() -> None:
    first = CodexStepDescription(
        number=1,
        title="Start",
        config=CodexStepConfig(task="Start work", ephemeral=False),
    )
    second = CodexStepDescription(
        number=2,
        title="Continue",
        dependencies=[1],
        config=CodexStepConfig(
            task="Continue work",
            resume_session="$memory.codex.step_1",
        ),
    )
    context = _context()

    result = await ReasoningChain([first, second]).execute_async(context)

    assert result.success is True
    assert context.memory_read("step_1", namespace="codex") == "thread-1"
    assert _FakeAsyncCodex.resume_calls[0][0] == "thread-1"


async def test_store_session_key_override() -> None:
    context = _context()

    await _codex_chain(store_session_key="session").execute_async(context)

    assert context.memory_read("session", namespace="codex") == "thread-1"


async def test_output_schema_parses_and_stores_object() -> None:
    _FakeAsyncCodex.response_override = '{"answer": 42}'
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "number"}},
        "required": ["answer"],
    }
    context = _context()

    result = await _codex_chain(
        output_schema=schema, output_memory_key="answer"
    ).execute_async(context)

    assert result.success is True
    step_result = result.step_results[0]
    assert step_result.result_data["structured_output"] == {"answer": 42}
    # memory receives the parsed object, not the raw text
    assert context.memory_read("answer", namespace="codex") == {"answer": 42}


async def test_output_schema_violation_fails_step() -> None:
    _FakeAsyncCodex.response_override = '{"answer": 42}'
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }

    result = await _codex_chain(output_schema=schema).execute_async(_context())

    step_result = result.step_results[0]
    assert step_result.success is False
    assert "does not match JSON Schema" in (step_result.error_message or "")


async def test_output_schema_without_json_response_fails_step() -> None:
    result = await _codex_chain(output_schema={"type": "object"}).execute_async(_context())

    step_result = result.step_results[0]
    assert step_result.success is False
    assert "no JSON payload" in (step_result.error_message or "")


async def test_long_provider_error_is_truncated(monkeypatch: pytest.MonkeyPatch) -> None:
    async def explode(**_kwargs: Any) -> tuple[str, Any]:
        raise RuntimeError("boom " * 500)

    executor = CodexStepExecutor()
    monkeypatch.setattr(executor, "_run_turn", explode)
    step = CodexStepDescription(
        number=1,
        title="Codex",
        config=CodexStepConfig(task="Inspect files"),
    )

    result = await executor.execute(step, _context())

    assert result.success is False
    assert result.error_message.endswith("…[truncated]")
    assert len(result.error_message) < 600


async def test_resume_session_accepts_literal_id() -> None:
    await _codex_chain(resume_session="thread-lit").execute_async(_context())

    assert _FakeAsyncCodex.resume_calls[0][0] == "thread-lit"


async def test_missing_placeholder_input_fails() -> None:
    result = await _codex_chain(
        task="Use {missing}", input_mapping={"other": "'x'"}
    ).execute_async(_context())

    step_result = result.step_results[0]
    assert step_result.success is False
    assert "input_mapping" in (step_result.error_message or "")


async def test_rendered_task_respects_max_input_bytes() -> None:
    result = await _codex_chain(
        task="x" * 2_000, max_input_bytes=1_024
    ).execute_async(_context())

    step_result = result.step_results[0]
    assert step_result.success is False
    assert "max_input_bytes" in (step_result.error_message or "")


def test_legacy_field_aliases_still_validate() -> None:
    config = CodexStepConfig(
        instruction="Inspect files",
        timeout_seconds=45.0,
        thread_id_source="$memory.codex.step_1",
        store_thread_key="session",
    )

    assert config.task == "Inspect files"
    assert config.timeout == 45.0
    assert config.resume_session == "$memory.codex.step_1"
    assert config.store_session_key == "session"
    # dumps normalize to the harmonized names
    dumped = config.model_dump()
    assert dumped["task"] == "Inspect files"
    assert "instruction" not in dumped
    assert "thread_id_source" not in dumped
