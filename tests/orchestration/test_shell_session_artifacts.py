"""Contract and local E2E tests for ShellSessionStep and portable artifacts."""

from __future__ import annotations

import base64
import os
from typing import Any

import pytest
from pydantic import ValidationError

from mmar_carl import (
    ArtifactInput,
    ArtifactOutput,
    ArtifactRecord,
    CommandPolicy,
    CommandStepConfig,
    ReasoningChain,
    ReasoningContext,
    ShellSessionStepConfig,
    ShellSessionStepDescription,
    ShellSessionStepExecutor,
    StepCache,
    StepDescription,
    StepType,
    create_step,
)
from mmar_carl.models.llm_client_base import LLMClientBase
from mmar_carl.skill_runtime import LocalSkillRuntime, register_skill_runtime
from mmar_carl.step_executors import get_executor


class _MockClient(LLMClientBase):
    async def get_response(self, prompt: str) -> str:
        return "mock"

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        return "mock"


def _local_policy() -> CommandPolicy:
    return CommandPolicy(
        allowed_executables=frozenset({"/bin/sh"}),
        allowed_runtimes=frozenset({"local"}),
        allowed_networks=frozenset({"host"}),
    )


def _context(*, approvals: list[Any] | None = None) -> ReasoningContext:
    def approve(request):
        if approvals is not None:
            approvals.append(request)
        return True

    return ReasoningContext(
        outer_context="test",
        api=_MockClient(),
        model="mock",
        command_policy=_local_policy(),
        on_command_approval_requested=approve,
    )


class TestArtifactModels:
    @pytest.mark.parametrize("path", ["", "/abs", "../x", "a/../x", "a//x", "a\\x", "./x"])
    def test_rejects_unsafe_paths(self, path: str) -> None:
        with pytest.raises(ValidationError):
            ArtifactOutput(name="result", path=path)

    def test_duplicate_paths_are_portably_rejected(self) -> None:
        with pytest.raises(ValidationError, match="case-folding"):
            ShellSessionStepConfig(
                commands=["true"],
                artifact_outputs=[
                    ArtifactOutput(name="one", path="Report.txt"),
                    ArtifactOutput(name="two", path="report.txt"),
                ],
            )

    def test_record_detects_size_hash_and_base64_tampering(self) -> None:
        record = ArtifactRecord.from_bytes(
            name="report",
            path="report.txt",
            media_type="text/plain",
            data=b"hello",
        )
        assert record.decode(max_bytes=5) == b"hello"

        bad_hash = record.model_copy(update={"sha256": "0" * 64})
        with pytest.raises(ValueError, match="sha256"):
            bad_hash.decode(max_bytes=10)
        bad_size = record.model_copy(update={"size_bytes": 4})
        with pytest.raises(ValueError, match="size"):
            bad_size.decode(max_bytes=10)
        bad_base64 = record.model_copy(update={"content_base64": "!!!!"})
        with pytest.raises(ValueError, match="base64"):
            bad_base64.decode(max_bytes=10)

    def test_record_rejects_oversize_before_decode(self) -> None:
        record = ArtifactRecord(
            name="x",
            path="x.bin",
            media_type="application/octet-stream",
            size_bytes=9,
            sha256="0" * 64,
            content_base64=base64.b64encode(b"123456789").decode(),
        )
        with pytest.raises(ValueError, match="exceeds"):
            record.decode(max_bytes=4)


class TestShellSessionWiring:
    def test_factory_registry_cache_and_legacy_conversion(self) -> None:
        config = ShellSessionStepConfig(commands=["pwd"])
        step = create_step(1, "shell", StepType.SHELL_SESSION, config=config)
        assert isinstance(step, ShellSessionStepDescription)
        assert isinstance(get_executor(StepType.SHELL_SESSION), ShellSessionStepExecutor)
        assert step.model_dump()["step_type"] == "shell_session"

        with pytest.raises(ValidationError, match="cannot be cached"):
            ShellSessionStepDescription(
                number=1,
                title="cached",
                config=config,
                cache=StepCache(),
            )

        with pytest.warns(DeprecationWarning):
            legacy = StepDescription(
                number=1,
                title="legacy",
                triggered_by=["ready"],
                step_type=StepType.SHELL_SESSION,
                step_config=config,
            )
        typed_legacy = legacy.to_typed_step()
        assert isinstance(typed_legacy, ShellSessionStepDescription)
        assert typed_legacy.triggered_by == ["ready"]

    def test_typed_and_legacy_json_round_trip(self) -> None:
        chain = ReasoningChain(
            steps=[
                ShellSessionStepDescription(
                    number=1,
                    title="session",
                    triggered_by=["ready"],
                    config=ShellSessionStepConfig(
                        commands=["VALUE=42", 'printf "%s" "$VALUE"'],
                    ),
                )
            ]
        )
        typed = ReasoningChain.from_dict_typed(chain.to_dict())
        assert isinstance(typed.steps[0], ShellSessionStepDescription)
        assert typed.steps[0].triggered_by == ["ready"]
        legacy = ReasoningChain.from_json(chain.to_json())
        assert legacy.steps[0].step_type == StepType.SHELL_SESSION
        assert isinstance(legacy.steps[0].step_config, ShellSessionStepConfig)
        assert legacy.steps[0].triggered_by == ["ready"]

    def test_raw_legacy_union_selects_shell_config(self) -> None:
        with pytest.warns(DeprecationWarning):
            legacy = StepDescription.model_validate(
                {
                    "number": 1,
                    "title": "raw shell",
                    "step_type": "shell_session",
                    "step_config": {"commands": ["pwd"], "shell": "/bin/sh"},
                }
            )
        assert isinstance(legacy.step_config, ShellSessionStepConfig)

    def test_mermaid_has_shell_style(self) -> None:
        chain = ReasoningChain(
            steps=[
                create_step(
                    1,
                    "session",
                    StepType.SHELL_SESSION,
                    config=ShellSessionStepConfig(commands=["true"]),
                )
            ]
        )
        diagram = chain.to_mermaid()
        assert "Shell session" in diagram
        assert "classDef shellsession" in diagram


@pytest.mark.asyncio
async def test_local_session_preserves_state_and_collects_artifact() -> None:
    config = ShellSessionStepConfig(
        shell="/bin/sh",
        commands=[
            "mkdir work",
            "cd work",
            "VALUE=forty-two",
            'render() { printf "%s" "$VALUE"; }',
            'render > "$CARL_ARTIFACT_OUT_REPORT"',
            'printf "%s" "${PWD##*/}"',
        ],
        artifact_outputs=[ArtifactOutput(name="report", path="report.txt", media_type="text/plain")],
        runtime="local",
        network="host",
    )
    result = await ReasoningChain(
        steps=[create_step(1, "session", StepType.SHELL_SESSION, config=config)]
    ).execute_async(_context())
    step_result = result.step_results[0]
    assert step_result.success
    assert step_result.result == "work"
    assert step_result.step_type == StepType.SHELL_SESSION
    assert step_result.result_data["command_count"] == 6
    assert "commands" not in step_result.result_data
    record = ArtifactRecord.model_validate(step_result.result_data["artifacts"]["report"])
    assert record.decode(max_bytes=100) == b"forty-two"
    assert not os.path.isabs(record.path)


@pytest.mark.asyncio
async def test_artifact_round_trip_between_two_sessions() -> None:
    first = ShellSessionStepDescription(
        number=1,
        title="produce",
        config=ShellSessionStepConfig(
            shell="/bin/sh",
            commands=['printf "portable" > "$CARL_ARTIFACT_OUT_TEXT"'],
            artifact_outputs=[ArtifactOutput(name="text", path="text.txt", media_type="text/plain")],
            runtime="local",
            network="host",
        ),
    )
    second = ShellSessionStepDescription(
        number=2,
        title="consume",
        dependencies=[1],
        config=ShellSessionStepConfig(
            shell="/bin/sh",
            commands=['cat "$CARL_ARTIFACT_IN_SOURCE" > "$CARL_ARTIFACT_OUT_COPY"'],
            artifact_inputs=[
                ArtifactInput(
                    name="source",
                    source="$steps.1.result_data.artifacts.text",
                    path="source.txt",
                )
            ],
            artifact_outputs=[ArtifactOutput(name="copy", path="copy.txt", media_type="text/plain")],
            runtime="local",
            network="host",
        ),
    )
    result = await ReasoningChain(steps=[first, second]).execute_async(_context())
    assert result.success
    copied = ArtifactRecord.model_validate(result.step_results[1].result_data["artifacts"]["copy"])
    assert copied.decode(max_bytes=100) == b"portable"


@pytest.mark.asyncio
async def test_command_step_uses_same_artifact_contract(tmp_path) -> None:
    helper = tmp_path / "copy-artifact"
    helper.write_text(
        '#!/bin/sh\ncat "$CARL_ARTIFACT_IN_SOURCE" > "$CARL_ARTIFACT_OUT_COPY"\n',
        encoding="utf-8",
    )
    helper.chmod(0o700)
    policy = CommandPolicy(
        allowed_executables=frozenset({str(helper)}),
        allowed_runtimes=frozenset({"local"}),
        allowed_networks=frozenset({"host"}),
    )
    context = ReasoningContext(
        outer_context="test",
        api=_MockClient(),
        command_policy=policy,
    )
    config = CommandStepConfig(
        command=[str(helper)],
        artifact_inputs=[ArtifactInput(name="source", source="'command-data'", path="source.txt")],
        artifact_outputs=[ArtifactOutput(name="copy", path="copy.txt")],
        runtime="local",
        network="host",
    )
    result = await ReasoningChain(steps=[create_step(1, "copy", StepType.COMMAND, config=config)]).execute_async(
        context
    )
    copied = ArtifactRecord.model_validate(result.step_results[0].result_data["artifacts"]["copy"])
    assert copied.decode(max_bytes=100) == b"command-data"


@pytest.mark.asyncio
async def test_literal_host_path_is_staged_as_data_not_read() -> None:
    config = ShellSessionStepConfig(
        shell="/bin/sh",
        commands=['cat "$CARL_ARTIFACT_IN_VALUE" > "$CARL_ARTIFACT_OUT_COPY"'],
        artifact_inputs=[ArtifactInput(name="value", source="'/etc/passwd'", path="value.txt")],
        artifact_outputs=[ArtifactOutput(name="copy", path="copy.txt")],
        runtime="local",
        network="host",
    )
    result = await ReasoningChain(
        steps=[create_step(1, "literal", StepType.SHELL_SESSION, config=config)]
    ).execute_async(_context())
    copied = ArtifactRecord.model_validate(result.step_results[0].result_data["artifacts"]["copy"])
    assert copied.decode(max_bytes=100) == b"/etc/passwd"


@pytest.mark.asyncio
async def test_ordinary_json_with_record_like_key_remains_json_data() -> None:
    context = _context()
    context.metadata["payload"] = {"sha256": "business-value", "count": 2}
    config = ShellSessionStepConfig(
        shell="/bin/sh",
        commands=['cat "$CARL_ARTIFACT_IN_VALUE" > "$CARL_ARTIFACT_OUT_COPY"'],
        artifact_inputs=[
            ArtifactInput(name="value", source="$metadata.payload", path="value.json")
        ],
        artifact_outputs=[ArtifactOutput(name="copy", path="copy.json")],
        runtime="local",
        network="host",
    )
    result = await ReasoningChain(
        steps=[create_step(1, "json", StepType.SHELL_SESSION, config=config)]
    ).execute_async(context)

    copied = ArtifactRecord.model_validate(
        result.step_results[0].result_data["artifacts"]["copy"]
    )
    assert copied.decode(max_bytes=100) == b'{"count":2,"sha256":"business-value"}'


@pytest.mark.asyncio
async def test_stop_on_error_and_approval_fingerprint_covers_script() -> None:
    approvals: list[Any] = []
    for command in ("printf one", "printf two"):
        config = ShellSessionStepConfig(
            shell="/bin/sh",
            commands=[command],
            runtime="local",
            network="host",
        )
        await ReasoningChain(
            steps=[create_step(1, "fingerprint", StepType.SHELL_SESSION, config=config)]
        ).execute_async(_context(approvals=approvals))
    assert len(approvals) == 2
    assert approvals[0].fingerprint != approvals[1].fingerprint
    assert approvals[0].stdin == b"set -e\nprintf one\n"
    assert "printf one" not in " ".join(approvals[0].argv_preview)

    failing = ShellSessionStepConfig(
        shell="/bin/sh",
        commands=["false", "printf should-not-run"],
        runtime="local",
        network="host",
    )
    result = await ReasoningChain(
        steps=[create_step(1, "fail-fast", StepType.SHELL_SESSION, config=failing)]
    ).execute_async(_context())
    assert not result.step_results[0].success
    assert "should-not-run" not in result.step_results[0].result


@pytest.mark.asyncio
async def test_approval_fingerprint_covers_artifact_content() -> None:
    approvals: list[Any] = []
    for value in ("one", "two"):
        config = ShellSessionStepConfig(
            shell="/bin/sh",
            commands=['cat "$CARL_ARTIFACT_IN_VALUE"'],
            artifact_inputs=[ArtifactInput(name="value", source=f"'{value}'", path="value.txt")],
            runtime="local",
            network="host",
        )
        await ReasoningChain(
            steps=[create_step(1, "artifact approval", StepType.SHELL_SESSION, config=config)]
        ).execute_async(_context(approvals=approvals))

    assert approvals[0].fingerprint != approvals[1].fingerprint
    assert (
        approvals[0].artifact_manifest["inputs"][0]["sha256"] != approvals[1].artifact_manifest["inputs"][0]["sha256"]
    )
    public_request = approvals[0].model_dump()
    assert "content_base64" not in str(public_request)
    assert "stdin" not in public_request
    assert "environment" not in public_request
    assert "artifact_inputs" not in public_request
    assert approvals[0].artifact_inputs == {"value": b"one"}
    assert set(public_request["artifact_manifest"]["inputs"][0]) == {
        "name",
        "path",
        "media_type",
        "size_bytes",
        "sha256",
        "max_bytes",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "commands, output",
    [
        (["true"], ArtifactOutput(name="missing", path="missing.bin")),
        (
            ['printf "12345" > "$CARL_ARTIFACT_OUT_RESULT"'],
            ArtifactOutput(name="result", path="result.bin", max_bytes=4),
        ),
        (
            ['ln -s /etc/passwd "$CARL_ARTIFACT_OUT_RESULT"'],
            ArtifactOutput(name="result", path="result.bin"),
        ),
    ],
    ids=["missing", "oversize", "symlink"],
)
async def test_invalid_declared_output_fails_step(
    commands: list[str],
    output: ArtifactOutput,
) -> None:
    result = await ReasoningChain(
        steps=[
            create_step(
                1,
                "invalid output",
                StepType.SHELL_SESSION,
                config=ShellSessionStepConfig(
                    shell="/bin/sh",
                    commands=commands,
                    artifact_outputs=[output],
                    runtime="local",
                    network="host",
                ),
            )
        ]
    ).execute_async(_context())
    assert not result.step_results[0].success
    assert result.step_results[0].result_data is None


@pytest.mark.asyncio
async def test_executor_rejects_runtime_that_violates_bounded_read_contract() -> None:
    class OversizeReaderRuntime(LocalSkillRuntime):
        name = "oversize-reader-test"

        async def read_file_bounded(
            self,
            handle,
            path,
            max_bytes,
            timeout=None,
        ) -> bytes:
            return b"12345"

    register_skill_runtime(OversizeReaderRuntime.name, OversizeReaderRuntime)
    policy = CommandPolicy(
        allowed_executables=frozenset({"/bin/sh"}),
        allowed_runtimes=frozenset({OversizeReaderRuntime.name}),
        allowed_networks=frozenset({"host"}),
        max_artifact_bytes=10,
        max_total_artifact_bytes=10,
    )
    context = ReasoningContext(
        outer_context="test",
        api=_MockClient(),
        command_policy=policy,
        on_command_approval_requested=lambda _request: True,
    )
    config = ShellSessionStepConfig(
        shell="/bin/sh",
        commands=["true"],
        artifact_outputs=[ArtifactOutput(name="result", path="result.bin", max_bytes=3)],
        runtime=OversizeReaderRuntime.name,
        network="host",
    )
    result = await ReasoningChain(
        steps=[create_step(1, "untrusted reader", StepType.SHELL_SESSION, config=config)]
    ).execute_async(context)
    assert not result.step_results[0].success
    assert "exceeds its per-file or remaining" in (
        result.step_results[0].error_message or ""
    )


@pytest.mark.asyncio
async def test_combined_artifact_budget_bounds_output_read_to_remaining_bytes() -> None:
    class RemainingBudgetRuntime(LocalSkillRuntime):
        name = "remaining-budget-test"
        requested_max_bytes: int | None = None

        async def read_file_bounded(
            self,
            handle,
            path,
            max_bytes,
            timeout=None,
        ) -> bytes:
            type(self).requested_max_bytes = max_bytes
            return b"x" * (max_bytes + 1)

    register_skill_runtime(RemainingBudgetRuntime.name, RemainingBudgetRuntime)
    policy = CommandPolicy(
        allowed_executables=frozenset({"/bin/sh"}),
        allowed_runtimes=frozenset({RemainingBudgetRuntime.name}),
        allowed_networks=frozenset({"host"}),
        max_artifact_bytes=5,
        max_total_artifact_bytes=5,
    )
    context = ReasoningContext(
        outer_context="four",
        api=_MockClient(),
        command_policy=policy,
        on_command_approval_requested=lambda _request: True,
    )
    config = ShellSessionStepConfig(
        shell="/bin/sh",
        commands=["true"],
        artifact_inputs=[ArtifactInput(name="input", source="$outer_context", path="in.txt")],
        artifact_outputs=[ArtifactOutput(name="output", path="out.txt", max_bytes=5)],
        runtime=RemainingBudgetRuntime.name,
        network="host",
    )
    result = await ReasoningChain(
        steps=[create_step(1, "combined cap", StepType.SHELL_SESSION, config=config)]
    ).execute_async(context)
    assert result.step_results[0].success is False
    assert RemainingBudgetRuntime.requested_max_bytes == 1
    assert "remaining host policy total" in (
        result.step_results[0].error_message or ""
    )


@pytest.mark.asyncio
async def test_separate_session_does_not_inherit_shell_state() -> None:
    steps = [
        ShellSessionStepDescription(
            number=1,
            title="set state",
            config=ShellSessionStepConfig(
                shell="/bin/sh",
                commands=["export CARL_SESSION_TEST=secret"],
                runtime="local",
                network="host",
            ),
        ),
        ShellSessionStepDescription(
            number=2,
            title="new process",
            dependencies=[1],
            config=ShellSessionStepConfig(
                shell="/bin/sh",
                commands=['printf "%s" "${CARL_SESSION_TEST-unset}"'],
                runtime="local",
                network="host",
            ),
        ),
    ]
    result = await ReasoningChain(steps=steps).execute_async(_context())
    assert result.step_results[1].result == "unset"


@pytest.mark.asyncio
async def test_session_uses_one_prepare_run_cleanup_cycle() -> None:
    class CountingRuntime(LocalSkillRuntime):
        name = "counting-shell-test"
        prepares = 0
        runs = 0
        cleanups = 0

        async def prepare(self, skill, workspace, config):
            type(self).prepares += 1
            return await super().prepare(skill, workspace, config)

        async def run(self, handle, cmd, **kwargs):
            type(self).runs += 1
            return await super().run(handle, cmd, **kwargs)

        async def cleanup(self, handle):
            type(self).cleanups += 1
            await super().cleanup(handle)

    register_skill_runtime(CountingRuntime.name, CountingRuntime)
    policy = CommandPolicy(
        allowed_executables=frozenset({"/bin/sh"}),
        allowed_runtimes=frozenset({CountingRuntime.name}),
        allowed_networks=frozenset({"host"}),
    )
    context = ReasoningContext(
        outer_context="test",
        api=_MockClient(),
        command_policy=policy,
        on_command_approval_requested=lambda _request: True,
    )
    result = await ReasoningChain(
        steps=[
            create_step(
                1,
                "count lifecycle",
                StepType.SHELL_SESSION,
                config=ShellSessionStepConfig(
                    shell="/bin/sh",
                    commands=["VALUE=ok", 'printf "%s" "$VALUE"'],
                    runtime=CountingRuntime.name,
                    network="host",
                ),
            )
        ]
    ).execute_async(context)
    assert result.step_results[0].success
    assert (CountingRuntime.prepares, CountingRuntime.runs, CountingRuntime.cleanups) == (1, 1, 1)


@pytest.mark.asyncio
async def test_oversize_input_fails_before_runtime_instantiation() -> None:
    class MustNotConstruct(LocalSkillRuntime):
        name = "artifact-preflight-test"
        constructions = 0

        def __init__(self) -> None:
            type(self).constructions += 1

    register_skill_runtime(MustNotConstruct.name, MustNotConstruct)
    policy = CommandPolicy(
        allowed_executables=frozenset({"/bin/sh"}),
        allowed_runtimes=frozenset({MustNotConstruct.name}),
        allowed_networks=frozenset({"host"}),
        max_artifact_bytes=3,
        max_total_artifact_bytes=3,
    )
    context = ReasoningContext(
        outer_context="test",
        api=_MockClient(),
        command_policy=policy,
        on_command_approval_requested=lambda _request: True,
    )
    config = ShellSessionStepConfig(
        shell="/bin/sh",
        commands=["true"],
        artifact_inputs=[ArtifactInput(name="value", source="'four'", path="value.txt")],
        runtime=MustNotConstruct.name,
        network="host",
    )
    result = await ReasoningChain(
        steps=[create_step(1, "preflight", StepType.SHELL_SESSION, config=config)]
    ).execute_async(context)
    assert not result.step_results[0].success
    assert "exceeds" in (result.step_results[0].error_message or "")
    assert MustNotConstruct.constructions == 0


@pytest.mark.asyncio
async def test_shell_session_timeout_is_bounded() -> None:
    config = ShellSessionStepConfig(
        shell="/bin/sh",
        commands=["sleep 5"],
        runtime="local",
        network="host",
        timeout=0.1,
    )
    result = await ReasoningChain(
        steps=[create_step(1, "timeout", StepType.SHELL_SESSION, config=config)]
    ).execute_async(_context())
    assert not result.step_results[0].success
    assert "timed out" in (result.step_results[0].error_message or "")
