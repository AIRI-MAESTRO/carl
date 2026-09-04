"""CodeStep v1: exact source, JSON boundaries and strict runtime lifecycle."""

from __future__ import annotations

import asyncio
import hashlib
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, ClassVar

import pytest
from pydantic import ValidationError

from mmar_carl import (
    CodeExecutionPolicy,
    CodeRuntimeProfile,
    CodeSourceError,
    CodeStepConfig,
    CodeStepDescription,
    DockerSkillRuntime,
    ReasoningChain,
    ReasoningContext,
    RuntimeCapabilities,
    RuntimeRunResult,
    SkillRuntimeHandle,
    StepType,
    register_skill_runtime,
)
from mmar_carl.code_execution import validate_code_source

INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "values": {"type": "array", "items": {"type": "number"}},
    },
    "required": ["values"],
    "additionalProperties": False,
}
OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {"sum": {"type": "number"}},
    "required": ["sum"],
    "additionalProperties": False,
}
SOURCE = "def run(inputs):\n    return {'sum': sum(inputs['values'])}\n"

pytestmark = pytest.mark.filterwarnings(
    "ignore:.*Execute generated code.*:UserWarning"
)


class HermeticCodeRuntime:
    """Test double running only fixture code in an owned local subprocess.

    It claims the strict capability surface so executor integration can be
    tested without a Docker daemon. It is never exposed as a production
    backend and is not evidence of host isolation.
    """

    name: ClassVar[str] = "code-test-strict"
    capabilities: ClassVar[RuntimeCapabilities] = RuntimeCapabilities(
        isolation="container",
        wall_time="enforced",
        output_limit="enforced",
        cpu_limit="enforced",
        memory_limit="enforced",
        pids_limit="enforced",
        network_none="enforced",
        workspace_files="enforced",
        artifact_output_limit="enforced",
    )
    prepare_calls: ClassVar[list[dict[str, Any]]] = []
    run_calls: ClassVar[list[list[str]]] = []
    cleanup_calls: ClassVar[int] = 0
    cancelled_runs: ClassVar[int] = 0
    cancelled_cleanups: ClassVar[int] = 0
    cleanup_delay: ClassVar[float] = 0.0
    fail_prepare: ClassVar[bool] = False
    fail_cleanup: ClassVar[bool] = False

    @classmethod
    def reset(cls) -> None:
        cls.prepare_calls = []
        cls.run_calls = []
        cls.cleanup_calls = 0
        cls.cancelled_runs = 0
        cls.cancelled_cleanups = 0
        cls.cleanup_delay = 0.0
        cls.fail_prepare = False
        cls.fail_cleanup = False

    async def prepare(
        self,
        skill: Any,
        workspace: Path | None,
        config: dict[str, Any],
    ) -> SkillRuntimeHandle:
        del skill, workspace
        type(self).prepare_calls.append(dict(config))
        if type(self).fail_prepare:
            from mmar_carl import SkillRuntimeError

            raise SkillRuntimeError("test runtime unavailable")
        root = Path(tempfile.mkdtemp(prefix="carl_code_test_"))
        (root / "in").mkdir()
        (root / "out").mkdir()
        return SkillRuntimeHandle(
            workspace_root=root,
            workspace_in=root / "in",
            workspace_out=root / "out",
            backend={
                "network_enforced": True,
                "workspace_output_mode": config.get("workspace_output_mode"),
                "workspace_in_in_runtime": str(root / "in"),
                "workspace_out_in_runtime": str(root / "out"),
            },
        )

    async def run(
        self,
        handle: SkillRuntimeHandle,
        cmd: list[str],
        *,
        env: dict[str, str] | None = None,
        stdin: bytes | None = None,
        timeout: float | None = None,
        cwd: str | None = None,
    ) -> RuntimeRunResult:
        del env, stdin
        type(self).run_calls.append(list(cmd))
        # The profile's first token is a runtime-local interpreter name. The
        # hermetic test double substitutes this test process's Python only.
        actual = [sys.executable, *cmd[1:]]
        started = asyncio.get_running_loop().time()
        proc: asyncio.subprocess.Process | None = None
        try:
            proc = await asyncio.create_subprocess_exec(
                *actual,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.CancelledError:
            type(self).cancelled_runs += 1
            if proc is not None:
                proc.kill()
                await proc.wait()
            raise
        except TimeoutError:
            assert proc is not None
            proc.kill()
            await proc.wait()
            return RuntimeRunResult(
                stdout=b"",
                stderr=f"[timeout after {timeout}s]".encode(),
                exit_code=124,
                duration_s=asyncio.get_running_loop().time() - started,
            )
        assert proc is not None
        limit = int(handle.backend.get("max_output_bytes", 1_000_000))
        stdout_truncated = len(stdout) > limit
        stderr_truncated = len(stderr) > limit
        return RuntimeRunResult(
            stdout=stdout[:limit],
            stderr=stderr[:limit],
            exit_code=proc.returncode or 0,
            duration_s=asyncio.get_running_loop().time() - started,
            stdout_truncated=stdout_truncated,
            stderr_truncated=stderr_truncated,
        )

    async def read_file(self, handle: SkillRuntimeHandle, path: str) -> bytes:
        return (handle.workspace_root / path).read_bytes()

    async def read_file_bounded(
        self,
        handle: SkillRuntimeHandle,
        path: str,
        max_bytes: int,
        timeout: float | None = None,
    ) -> bytes:
        del timeout
        data = await self.read_file(handle, path)
        if len(data) > max_bytes:
            raise ValueError("file too large")
        return data

    async def write_file(
        self, handle: SkillRuntimeHandle, path: str, data: bytes,
    ) -> None:
        target = handle.workspace_root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

    async def cleanup(self, handle: SkillRuntimeHandle) -> None:
        type(self).cleanup_calls += 1
        try:
            await asyncio.sleep(type(self).cleanup_delay)
        except asyncio.CancelledError:
            type(self).cancelled_cleanups += 1
            raise
        finally:
            shutil.rmtree(handle.workspace_root)
        if type(self).fail_cleanup:
            raise RuntimeError("cleanup failed")


class WeakCodeRuntime(HermeticCodeRuntime):
    name: ClassVar[str] = "code-test-weak"
    capabilities: ClassVar[RuntimeCapabilities] = RuntimeCapabilities(
        isolation="none",
        wall_time="enforced",
        output_limit="enforced",
    )


class UnknownIsolationRuntime(HermeticCodeRuntime):
    name: ClassVar[str] = "code-test-unknown-isolation"
    capabilities: ClassVar[RuntimeCapabilities] = RuntimeCapabilities(
        isolation="unknown",
        wall_time="enforced",
        output_limit="enforced",
        cpu_limit="enforced",
        memory_limit="enforced",
        pids_limit="enforced",
        network_none="enforced",
        workspace_files="enforced",
        artifact_output_limit="enforced",
    )


register_skill_runtime(HermeticCodeRuntime.name, HermeticCodeRuntime)
register_skill_runtime(WeakCodeRuntime.name, WeakCodeRuntime)
register_skill_runtime(UnknownIsolationRuntime.name, UnknownIsolationRuntime)


@pytest.fixture(autouse=True)
def reset_runtime() -> None:
    HermeticCodeRuntime.reset()
    WeakCodeRuntime.reset()
    UnknownIsolationRuntime.reset()


def profile(runtime: str = HermeticCodeRuntime.name, **updates: Any) -> CodeRuntimeProfile:
    values = {
        "runtime": runtime,
        "revision": "hermetic-test-v1",
        "interpreter": ("python", "-I", "-B"),
        "max_timeout_seconds": 1.0,
        "max_cleanup_seconds": 1.0,
        "max_source_bytes": 20_000,
        "max_input_bytes": 10_000,
        "max_output_bytes": 10_000,
    }
    values.update(updates)
    return CodeRuntimeProfile(**values)


def context(
    *,
    source: Any = SOURCE,
    values: Any = None,
    runtime: str = HermeticCodeRuntime.name,
    profile_updates: dict[str, Any] | None = None,
) -> ReasoningContext:
    ctx = ReasoningContext(
        outer_context="",
        api=None,
        code_execution_policy=CodeExecutionPolicy(
            profiles={"python-safe-v1": profile(runtime, **(profile_updates or {}))},
        ),
    )
    ctx.memory_write("source", source, namespace="generated")
    ctx.memory_write("values", [1, 2, 3] if values is None else values, namespace="input")
    return ctx


def config(**updates: Any) -> CodeStepConfig:
    values = {
        "source": "$memory.generated.source",
        "runtime_profile": "python-safe-v1",
        "input_mapping": {"values": "$memory.input.values"},
        "input_schema": INPUT_SCHEMA,
        "output_schema": OUTPUT_SCHEMA,
        "timeout_seconds": 0.5,
        "output_key": "answer",
    }
    values.update(updates)
    return CodeStepConfig(**values)


def step(**updates: Any) -> CodeStepDescription:
    return CodeStepDescription(number=1, title="Execute generated code", config=config(**updates))


async def execute(ctx: ReasoningContext, **updates: Any):
    result = await ReasoningChain([step(**updates)]).execute_async(ctx)
    return result.step_results[0]


class TestCodeContracts:
    def test_rejects_inline_source(self) -> None:
        with pytest.raises(ValidationError, match="context reference"):
            config(source=SOURCE)

    def test_rejects_unknown_schema_keyword(self) -> None:
        with pytest.raises(ValidationError, match="unsupported schema keys"):
            config(output_schema={"type": "number", "minimum": 0})

    def test_requires_object_input_schema(self) -> None:
        with pytest.raises(ValidationError, match="type='object'"):
            config(input_schema={"type": "array", "items": {"type": "number"}})

    def test_static_input_mapping_must_satisfy_schema(self) -> None:
        with pytest.raises(ValidationError, match="does not provide required"):
            config(input_mapping={})
        with pytest.raises(ValidationError, match="undeclared schema keys"):
            config(
                input_mapping={
                    "values": "$memory.input.values",
                    "extra": "$memory.input.extra",
                },
            )

    def test_rejects_markdown_and_wrong_entrypoint_before_prepare(self) -> None:
        cases = [
            "```python\n" + SOURCE + "```",
            "def compute(inputs):\n    return 1\n",
            "async def run(inputs):\n    return 1\n",
            "def run(value):\n    return value\n",
        ]
        for source in cases:
            with pytest.raises(CodeSourceError):
                # Config validates the reference; execution validates its value.
                validate_code_source(source)

    def test_docker_profile_requires_digest(self) -> None:
        with pytest.raises(ValidationError, match="pinned"):
            CodeRuntimeProfile(
                runtime="docker",
                revision="v1",
                prepare_config={"image": "python:3.12-slim"},
            )
        with pytest.raises(ValidationError, match="pinned"):
            CodeRuntimeProfile(
                runtime="docker",
                revision="v1",
                prepare_config={"image": "--pull=always@sha256:" + "a" * 64},
            )
        pinned = "python@sha256:" + "a" * 64
        item = CodeRuntimeProfile(
            runtime="docker",
            revision="v1",
            prepare_config={"image": pinned},
        )
        assert item.prepare_config["image"] == pinned

    def test_policy_is_runtime_only(self) -> None:
        ctx = context()
        assert "code_execution_policy" not in ctx.model_dump(mode="json")


class TestCodeExecution:
    async def test_success_uses_exact_source_and_writes_validated_output(self) -> None:
        ctx = context()

        result = await execute(ctx)

        outcome = result.as_code_execution_outcome()
        assert result.success is True
        assert result.step_type is StepType.CODE
        assert outcome.status == "completed"
        assert outcome.output == {"sum": 6}
        assert outcome.source_sha256 == hashlib.sha256(SOURCE.encode()).hexdigest()
        assert outcome.input_sha256 is not None
        assert outcome.output_sha256 is not None
        assert outcome.python_version
        assert ctx.memory_read("answer", namespace="code") == {"sum": 6}
        assert HermeticCodeRuntime.prepare_calls[0]["network"] == "none"
        assert HermeticCodeRuntime.prepare_calls[0]["workspace_output_mode"] == "read_only"
        assert HermeticCodeRuntime.cleanup_calls == 1

    async def test_generated_stdout_is_diagnostic_not_semantic_result(self) -> None:
        source = (
            "def run(inputs):\n"
            "    print('debug-only')\n"
            "    return {'sum': sum(inputs['values'])}\n"
        )
        result = await execute(context(source=source))
        outcome = result.as_code_execution_outcome()

        assert result.success is True
        assert result.result == '{"sum":6}'
        assert outcome.stdout == "debug-only"
        assert "debug-only" not in result.updated_history[-1]

    async def test_missing_policy_is_denied_before_prepare(self) -> None:
        ctx = context()
        ctx.code_execution_policy = None

        result = await execute(ctx)

        assert result.as_code_execution_outcome().status == "denied"
        assert HermeticCodeRuntime.prepare_calls == []

    async def test_missing_profile_is_denied_before_prepare(self) -> None:
        ctx = context()

        result = await execute(ctx, runtime_profile="unknown")

        assert result.as_code_execution_outcome().status == "denied"
        assert HermeticCodeRuntime.prepare_calls == []

    async def test_weak_runtime_is_denied_before_prepare(self) -> None:
        result = await execute(context(runtime=WeakCodeRuntime.name))

        outcome = result.as_code_execution_outcome()
        assert outcome.status == "denied"
        assert "strict CodeStep controls" in outcome.error_message
        assert WeakCodeRuntime.prepare_calls == []

    async def test_unknown_isolation_is_denied_even_if_controls_claim_enforced(self) -> None:
        result = await execute(context(runtime=UnknownIsolationRuntime.name))

        outcome = result.as_code_execution_outcome()
        assert outcome.status == "denied"
        assert "isolation" in outcome.error_message
        assert outcome.enforcement_report["isolation_accepted"] is False
        assert UnknownIsolationRuntime.prepare_calls == []

    async def test_invalid_source_is_typed_and_never_prepares(self) -> None:
        result = await execute(context(source="```python\n" + SOURCE + "```"))

        assert result.as_code_execution_outcome().status == "invalid_source"
        assert HermeticCodeRuntime.prepare_calls == []

    async def test_source_byte_limit_is_checked_before_prepare(self) -> None:
        result = await execute(context(), max_source_bytes=8)

        assert result.as_code_execution_outcome().status == "invalid_source"
        assert HermeticCodeRuntime.prepare_calls == []

    async def test_invalid_input_is_typed_and_never_prepares(self) -> None:
        result = await execute(context(values="not-a-list"))

        assert result.as_code_execution_outcome().status == "invalid_input"
        assert HermeticCodeRuntime.prepare_calls == []

    async def test_input_byte_limit_is_checked_before_prepare(self) -> None:
        result = await execute(context(values=list(range(100))), max_input_bytes=32)

        outcome = result.as_code_execution_outcome()
        assert outcome.status == "invalid_input"
        assert "limit is 32" in outcome.error_message
        assert HermeticCodeRuntime.prepare_calls == []

    async def test_non_finite_input_is_rejected(self) -> None:
        result = await execute(context(values=[float("nan")]))

        assert result.as_code_execution_outcome().status == "invalid_input"
        assert HermeticCodeRuntime.prepare_calls == []

    async def test_input_rejects_non_string_json_object_keys(self) -> None:
        result = await execute(context(values=[{1: "coercion is forbidden"}]))

        assert result.as_code_execution_outcome().status == "invalid_input"
        assert HermeticCodeRuntime.prepare_calls == []

    async def test_output_schema_violation_is_invalid_output_and_not_written(self) -> None:
        source = "def run(inputs):\n    return {'wrong': 1}\n"
        ctx = context(source=source)

        result = await execute(ctx)

        assert result.as_code_execution_outcome().status == "invalid_output"
        assert ctx.memory_read("answer", namespace="code") is None

    async def test_output_rejects_python_to_json_key_coercion(self) -> None:
        source = "def run(inputs):\n    return {1: 6}\n"

        result = await execute(context(source=source))

        outcome = result.as_code_execution_outcome()
        assert outcome.status == "invalid_output"
        assert "keys must be strings" in outcome.error_message

    async def test_output_byte_limit_is_invalid_output(self) -> None:
        source = "def run(inputs):\n    return 'x' * 200\n"

        result = await execute(
            context(source=source),
            output_schema={"type": "string"},
            max_output_bytes=64,
        )

        outcome = result.as_code_execution_outcome()
        assert outcome.status == "invalid_output"
        assert "exceeds max_output_bytes" in outcome.error_message

    async def test_runtime_exception_is_failed(self) -> None:
        source = "def run(inputs):\n    raise RuntimeError('boom')\n"

        result = await execute(context(source=source))

        outcome = result.as_code_execution_outcome()
        assert outcome.status == "failed"
        assert "RuntimeError: boom" in outcome.error_message
        assert HermeticCodeRuntime.cleanup_calls == 1

    async def test_nonzero_exit_without_envelope_is_runtime_failure(self) -> None:
        source = "import os\n\ndef run(inputs):\n    os._exit(7)\n"

        result = await execute(context(source=source))

        outcome = result.as_code_execution_outcome()
        assert outcome.status == "failed"
        assert "exited with code 7" in outcome.error_message

    async def test_user_value_error_is_runtime_failure_not_invalid_output(self) -> None:
        source = "def run(inputs):\n    raise ValueError('user failure')\n"

        result = await execute(context(source=source))

        outcome = result.as_code_execution_outcome()
        assert outcome.status == "failed"
        assert "ValueError: user failure" in outcome.error_message

    async def test_result_envelope_has_bounded_protocol_headroom(self) -> None:
        source = "def run(inputs):\n    return 'x' * 900\n"

        result = await execute(
            context(source=source),
            output_schema={"type": "string"},
            max_output_bytes=1_000,
        )

        outcome = result.as_code_execution_outcome()
        assert outcome.status == "completed"
        assert outcome.output == "x" * 900
        assert outcome.effective_limits["diagnostic_bytes"] == 5_096

    async def test_timeout_cancels_owned_runtime_and_cleans_up(self) -> None:
        source = "def run(inputs):\n    while True:\n        pass\n"

        result = await execute(context(source=source), timeout_seconds=0.05)

        assert result.as_code_execution_outcome().status == "timed_out"
        assert HermeticCodeRuntime.cancelled_runs == 1
        assert HermeticCodeRuntime.cleanup_calls == 1

    async def test_host_cancellation_cancels_owned_runtime_and_cleans_up(self) -> None:
        source = "def run(inputs):\n    while True:\n        pass\n"
        ctx = context(source=source)
        task = asyncio.create_task(execute(ctx, timeout_seconds=0.8))
        while not HermeticCodeRuntime.run_calls:
            await asyncio.sleep(0.01)
        ctx.cancel()

        result = await task

        assert result.as_code_execution_outcome().status == "cancelled"
        assert result.skipped is True
        assert HermeticCodeRuntime.cancelled_runs == 1
        assert HermeticCodeRuntime.cleanup_calls == 1

    async def test_cleanup_failure_cannot_become_success(self) -> None:
        HermeticCodeRuntime.fail_cleanup = True

        result = await execute(context())

        outcome = result.as_code_execution_outcome()
        assert outcome.status == "failed"
        assert "cleanup failed" in outcome.error_message

    async def test_cleanup_timeout_cannot_become_success(self) -> None:
        HermeticCodeRuntime.cleanup_delay = 0.2

        result = await execute(
            context(profile_updates={"max_cleanup_seconds": 0.02}),
        )

        outcome = result.as_code_execution_outcome()
        assert outcome.status == "failed"
        assert "cleanup timed out" in outcome.error_message
        assert HermeticCodeRuntime.cancelled_cleanups == 1

    async def test_parent_task_cancellation_waits_for_bounded_cleanup(self) -> None:
        HermeticCodeRuntime.cleanup_delay = 0.05
        source = "def run(inputs):\n    while True:\n        pass\n"
        task = asyncio.create_task(execute(context(source=source), timeout_seconds=0.8))
        while not HermeticCodeRuntime.run_calls:
            await asyncio.sleep(0.01)

        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task
        assert HermeticCodeRuntime.cancelled_runs == 1
        assert HermeticCodeRuntime.cleanup_calls == 1
        assert HermeticCodeRuntime.cancelled_cleanups == 0

    async def test_parent_cancellation_during_cleanup_is_reshielded(self) -> None:
        HermeticCodeRuntime.cleanup_delay = 0.1
        task = asyncio.create_task(execute(context()))
        while not HermeticCodeRuntime.cleanup_calls:
            await asyncio.sleep(0.01)

        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task
        assert HermeticCodeRuntime.cleanup_calls == 1
        assert HermeticCodeRuntime.cancelled_cleanups == 0


class TestCodeChainIntegration:
    def test_typed_and_legacy_round_trip(self) -> None:
        original = ReasoningChain([step()])
        payload = original.to_dict()

        typed = ReasoningChain.from_dict_typed(payload)
        legacy = ReasoningChain.from_dict(payload)

        assert payload["format_version"] == ReasoningChain.FORMAT_VERSION == 10
        assert payload["steps"][0]["step_type"] == "code"
        assert isinstance(typed.steps[0], CodeStepDescription)
        assert typed.steps[0].config == original.steps[0].config
        assert legacy.steps[0].step_type is StepType.CODE
        assert isinstance(legacy.steps[0].step_config, CodeStepConfig)

    async def test_saved_chain_executes_after_typed_load(self) -> None:
        restored = ReasoningChain.from_dict_typed(ReasoningChain([step()]).to_dict())

        result = await restored.execute_async(context())

        assert result.success is True
        assert result.step_results[0].as_code_execution_outcome().output == {"sum": 6}

    def test_preflight_reports_missing_and_present_profiles(self) -> None:
        chain = ReasoningChain([step()])
        missing = chain.preflight(ReasoningContext(outer_context="", api=None))
        present = chain.preflight(context())

        assert missing.required_code_profiles == ["python-safe-v1"]
        assert missing.missing_code_profiles == ["python-safe-v1"]
        assert missing.all_present is False
        assert present.missing_code_profiles == []
        assert present.all_present is True

    def test_docker_read_only_output_mount(self, tmp_path: Path) -> None:
        handle = SkillRuntimeHandle(
            workspace_root=tmp_path,
            workspace_in=tmp_path / "in",
            workspace_out=tmp_path / "out",
            backend={
                "image": "python@sha256:" + "a" * 64,
                "network_policy": "none",
                "workspace_output_mode": "read_only",
                "mem_limit": "256m",
                "cpu_limit": "1",
                "pids_limit": 32,
            },
        )

        command = DockerSkillRuntime()._build_docker_cmd(
            handle,
            ["python", "/workspace/in/runner.py"],
            env=None,
            timeout=1.0,
            cwd="/workspace/in",
        )

        assert f"{handle.workspace_in}:/workspace/in:ro" in command
        assert f"{handle.workspace_out}:/workspace/out:ro" in command
        network_index = command.index("--network")
        assert command[network_index + 1] == "none"
