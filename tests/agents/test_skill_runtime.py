"""Tests for the SkillRuntime protocol + LocalSkillRuntime + registry.

The SkillRuntime is the pluggable sandbox plug-in surface for AgentSkill
execution. This file covers:

* Protocol contract — LocalSkillRuntime implements every method.
* Behaviour — prepare/run/read/write/cleanup round-trip a tiny script.
* Registry — register / get / unknown-name handling, error message
  surfaces the available list.
* Executor wiring — unknown ``runtime`` in AgentSkillStepConfig fails
  the step cleanly with a SkillRuntimeError-derived message.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest

from mmar_carl.skill_runtime import (
    LocalSkillRuntime,
    SKILL_RUNTIME_REGISTRY,
    RuntimeCapabilities,
    SkillRuntimeError,
    SkillRuntimeHandle,
    assess_runtime_enforcement,
    get_runtime_capabilities,
    get_skill_runtime,
    list_skill_runtimes,
    register_skill_runtime,
)


# ---------------------------------------------------------------------------
# Protocol contract
# ---------------------------------------------------------------------------


class TestProtocolContract:
    def test_local_runtime_satisfies_protocol(self) -> None:
        rt = LocalSkillRuntime()
        # The Protocol is @runtime_checkable, but issubclass on a class
        # with ClassVars is brittle; assert method presence instead.
        for attr in ("prepare", "run", "read_file", "write_file", "cleanup"):
            assert callable(getattr(rt, attr)), f"missing {attr}"
        assert rt.name == "local"

    def test_handle_carries_workspace_paths(self) -> None:
        h = SkillRuntimeHandle(
            workspace_root=Path("/tmp/x"),
            workspace_in=Path("/tmp/x/in"),
            workspace_out=Path("/tmp/x/out"),
        )
        assert h.workspace_root == Path("/tmp/x")
        assert h.backend == {}


class TestCapabilityReports:
    def test_local_reports_network_none_as_unsupported(self) -> None:
        report = assess_runtime_enforcement(
            LocalSkillRuntime(),
            mode="strict",
            network="none",
            cpu_limit_requested=False,
            memory_limit_requested=False,
            pids_limit_requested=False,
        )
        assert report.isolation == "none"
        assert report.controls["wall_time"] == (
            "enforced" if os.name == "posix" else "advisory"
        )
        assert report.controls["output_limit"] == "enforced"
        assert report.controls["network"] == "unsupported"
        assert report.gaps == ("network",)

    def test_builtin_runtime_matrix_is_honest(self) -> None:
        from mmar_carl.docker_skill_runtime import DockerSkillRuntime
        from mmar_carl.e2b_skill_runtime import E2BSkillRuntime
        from mmar_carl.firejail_skill_runtime import FirejailSkillRuntime

        docker = get_runtime_capabilities(DockerSkillRuntime())
        firejail = get_runtime_capabilities(FirejailSkillRuntime())
        e2b = get_runtime_capabilities(E2BSkillRuntime())

        assert docker.isolation == "container"
        assert docker.network_none == "enforced"
        assert docker.network_allowlist == "unsupported"
        assert firejail.isolation == "process"
        assert firejail.cpu_limit == "unsupported"
        assert firejail.network_allowlist == "unsupported"
        assert e2b.isolation == "microvm"
        assert e2b.output_limit == "advisory"
        assert e2b.network_none == "enforced"
        assert e2b.network_allowlist == "enforced"

    def test_undeclared_custom_runtime_fails_closed(self) -> None:
        class RuntimeWithoutCapabilities:
            name = "undeclared"

        capabilities = get_runtime_capabilities(RuntimeWithoutCapabilities())
        assert capabilities == RuntimeCapabilities(isolation="unknown")
        report = assess_runtime_enforcement(
            RuntimeWithoutCapabilities(),
            mode="strict",
            network="none",
            cpu_limit_requested=True,
            memory_limit_requested=True,
            pids_limit_requested=True,
        )
        assert set(report.gaps) == {
            "wall_time",
            "output_limit",
            "cpu_limit",
            "memory_limit",
            "pids_limit",
            "network",
        }

        artifact_report = assess_runtime_enforcement(
            RuntimeWithoutCapabilities(),
            mode="strict",
            network="host",
            cpu_limit_requested=False,
            memory_limit_requested=False,
            pids_limit_requested=False,
            workspace_files_requested=True,
        )
        assert artifact_report.controls["workspace_files"] == "unsupported"
        assert "workspace_files" in artifact_report.gaps


# ---------------------------------------------------------------------------
# LocalSkillRuntime behaviour
# ---------------------------------------------------------------------------


class TestLocalSkillRuntime:
    async def test_prepare_creates_workspace_subdirs(self, tmp_path: Path) -> None:
        rt = LocalSkillRuntime()
        handle = await rt.prepare(skill=None, workspace=tmp_path / "ws", config={})
        assert handle.workspace_root.is_dir()
        assert handle.workspace_in.is_dir()
        assert handle.workspace_out.is_dir()
        assert handle.workspace_in.name == "in"
        assert handle.workspace_out.name == "out"

    async def test_prepare_without_workspace_creates_tempdir(self) -> None:
        rt = LocalSkillRuntime()
        handle = await rt.prepare(skill=None, workspace=None, config={})
        try:
            assert handle.workspace_root.is_dir()
            assert handle.workspace_root.name.startswith("carl_skill_")
        finally:
            await rt.cleanup(handle)
            assert not handle.workspace_root.exists()

    async def test_run_echoes_stdout(self, tmp_path: Path) -> None:
        rt = LocalSkillRuntime()
        handle = await rt.prepare(skill=None, workspace=tmp_path / "ws", config={})
        result = await rt.run(handle, [sys.executable, "-c", "print('hello')"])
        assert result.exit_code == 0
        assert result.stdout.strip() == b"hello"
        assert result.duration_s >= 0
        await rt.cleanup(handle)

    async def test_run_captures_nonzero_exit(self, tmp_path: Path) -> None:
        rt = LocalSkillRuntime()
        handle = await rt.prepare(skill=None, workspace=tmp_path / "ws", config={})
        result = await rt.run(
            handle,
            [sys.executable, "-c", "import sys; sys.exit(7)"],
        )
        assert result.exit_code == 7
        await rt.cleanup(handle)

    async def test_run_respects_timeout(self, tmp_path: Path) -> None:
        rt = LocalSkillRuntime()
        handle = await rt.prepare(skill=None, workspace=tmp_path / "ws", config={})
        result = await rt.run(
            handle,
            [sys.executable, "-c", "import time; time.sleep(5)"],
            timeout=0.2,
        )
        # Standard "killed by timeout" exit code in our implementation
        assert result.exit_code == 124
        assert b"timeout" in result.stderr.lower()
        await rt.cleanup(handle)

    async def test_timeout_kills_child_after_direct_leader_exits(
        self,
        tmp_path: Path,
    ) -> None:
        marker = tmp_path / "late-child.txt"
        child_code = (
            "import pathlib,time; "
            "time.sleep(0.4); "
            f"pathlib.Path({str(marker)!r}).write_text('escaped')"
        )
        parent_code = (
            "import subprocess,sys; "
            "subprocess.Popen([sys.executable, '-c', "
            f"{child_code!r}])"
        )
        rt = LocalSkillRuntime()
        handle = await rt.prepare(skill=None, workspace=tmp_path / "ws", config={})
        result = await rt.run(
            handle,
            [sys.executable, "-c", parent_code],
            timeout=0.05,
        )
        assert result.exit_code == 124
        await asyncio.sleep(0.6)
        assert not marker.exists()
        await rt.cleanup(handle)

    async def test_successful_command_cannot_leave_background_child(
        self,
        tmp_path: Path,
    ) -> None:
        marker = tmp_path / "escaped-child.txt"
        child_code = (
            "import pathlib,time; "
            "time.sleep(0.4); "
            f"pathlib.Path({str(marker)!r}).write_text('escaped')"
        )
        parent_code = (
            "import subprocess,sys; "
            "subprocess.Popen([sys.executable, '-c', "
            f"{child_code!r}], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)"
        )
        rt = LocalSkillRuntime()
        handle = await rt.prepare(skill=None, workspace=tmp_path / "ws", config={})
        result = await rt.run(handle, [sys.executable, "-c", parent_code])
        assert result.exit_code == 0
        await asyncio.sleep(0.6)
        assert not marker.exists()
        await rt.cleanup(handle)

    async def test_run_passes_env(self, tmp_path: Path) -> None:
        rt = LocalSkillRuntime()
        handle = await rt.prepare(skill=None, workspace=tmp_path / "ws", config={})
        result = await rt.run(
            handle,
            [sys.executable, "-c", "import os; print(os.environ.get('FOO'))"],
            env={"FOO": "bar", "PATH": ""},
        )
        assert result.stdout.strip() == b"bar"
        await rt.cleanup(handle)

    async def test_output_limit_bounds_both_streams_while_draining(
        self,
        tmp_path: Path,
    ) -> None:
        rt = LocalSkillRuntime()
        handle = await rt.prepare(skill=None, workspace=tmp_path / "ws", config={})
        handle.backend["max_output_bytes"] = 16
        result = await rt.run(
            handle,
            [
                sys.executable,
                "-c",
                "import sys; sys.stdout.write('o'*1000); sys.stderr.write('e'*1000)",
            ],
        )
        assert result.stdout == b"o" * 16
        assert result.stderr == b"e" * 16
        assert result.stdout_truncated is True
        assert result.stderr_truncated is True
        await rt.cleanup(handle)

    async def test_run_failure_to_launch_raises_runtime_error(
        self, tmp_path: Path,
    ) -> None:
        rt = LocalSkillRuntime()
        handle = await rt.prepare(skill=None, workspace=tmp_path / "ws", config={})
        with pytest.raises(SkillRuntimeError):
            await rt.run(handle, ["/no/such/executable/ever"])
        await rt.cleanup(handle)

    async def test_write_and_read_file_round_trip(self, tmp_path: Path) -> None:
        rt = LocalSkillRuntime()
        handle = await rt.prepare(skill=None, workspace=tmp_path / "ws", config={})
        await rt.write_file(handle, "out/result.txt", b"payload")
        data = await rt.read_file(handle, "out/result.txt")
        assert data == b"payload"
        await rt.cleanup(handle)

    async def test_write_creates_parent_dirs(self, tmp_path: Path) -> None:
        rt = LocalSkillRuntime()
        handle = await rt.prepare(skill=None, workspace=tmp_path / "ws", config={})
        await rt.write_file(handle, "deep/nested/path/file.txt", b"hello")
        assert (handle.workspace_root / "deep" / "nested" / "path" / "file.txt").exists()
        await rt.cleanup(handle)

    async def test_absolute_path_rejected(self, tmp_path: Path) -> None:
        rt = LocalSkillRuntime()
        handle = await rt.prepare(skill=None, workspace=tmp_path / "ws", config={})
        with pytest.raises(SkillRuntimeError, match="must be relative"):
            await rt.write_file(handle, "/etc/passwd", b"evil")
        await rt.cleanup(handle)

    async def test_path_escape_via_dotdot_rejected(self, tmp_path: Path) -> None:
        rt = LocalSkillRuntime()
        handle = await rt.prepare(skill=None, workspace=tmp_path / "ws", config={})
        with pytest.raises(SkillRuntimeError, match="escapes workspace"):
            await rt.write_file(handle, "../../etc/passwd", b"evil")
        await rt.cleanup(handle)

    async def test_cleanup_removes_workspace(self, tmp_path: Path) -> None:
        rt = LocalSkillRuntime()
        ws = tmp_path / "ws"
        handle = await rt.prepare(skill=None, workspace=ws, config={})
        assert ws.is_dir()
        await rt.cleanup(handle)
        assert not ws.exists()

    async def test_cleanup_is_idempotent(self, tmp_path: Path) -> None:
        rt = LocalSkillRuntime()
        handle = await rt.prepare(skill=None, workspace=tmp_path / "ws", config={})
        await rt.cleanup(handle)
        # Second call must not raise on missing dir
        await rt.cleanup(handle)

    async def test_cleanup_skips_when_persisted(self, tmp_path: Path) -> None:
        rt = LocalSkillRuntime()
        ws = tmp_path / "ws"
        handle = await rt.prepare(skill=None, workspace=ws, config={})
        handle.backend["persisted"] = True
        await rt.cleanup(handle)
        # Workspace should still exist
        assert ws.is_dir()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_local_preregistered(self) -> None:
        assert "local" in SKILL_RUNTIME_REGISTRY
        assert "local" in list_skill_runtimes()

    def test_get_returns_instance(self) -> None:
        rt = get_skill_runtime("local")
        assert isinstance(rt, LocalSkillRuntime)

    def test_unknown_runtime_raises_with_available_list(self) -> None:
        with pytest.raises(SkillRuntimeError) as exc:
            get_skill_runtime("nonexistent")
        msg = str(exc.value)
        assert "nonexistent" in msg
        # The available-list hint mentions 'local' (always registered)
        assert "local" in msg
        assert "backend prerequisites" in msg

    def test_register_custom_runtime(self) -> None:
        class StubRuntime:
            name = "stub-test-runtime"

            async def prepare(self, *args, **kwargs):
                raise NotImplementedError

            async def run(self, *args, **kwargs):
                raise NotImplementedError

            async def read_file(self, *args, **kwargs):
                raise NotImplementedError

            async def write_file(self, *args, **kwargs):
                raise NotImplementedError

            async def cleanup(self, *args, **kwargs):
                raise NotImplementedError

        register_skill_runtime("stub-test-runtime", StubRuntime)
        try:
            rt = get_skill_runtime("stub-test-runtime")
            assert isinstance(rt, StubRuntime)
            assert "stub-test-runtime" in list_skill_runtimes()
        finally:
            SKILL_RUNTIME_REGISTRY.pop("stub-test-runtime", None)

    def test_register_rejects_non_class(self) -> None:
        with pytest.raises(TypeError):
            register_skill_runtime("bad", "not-a-class")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Executor wiring
# ---------------------------------------------------------------------------


class TestExecutorWiring:
    @pytest.mark.asyncio
    async def test_unknown_runtime_fails_step_cleanly(
        self, tmp_path: Path,
    ) -> None:
        """When AgentSkillStepConfig.runtime points at a name that
        isn't registered, the step must fail with a SkillRuntimeError
        message — not blow up with an attribute error later."""
        from mmar_carl import (
            AgentSkillStepConfig,
            AgentSkillStepDescription,
            AgentSkillSource,
            ReasoningChain,
            ReasoningContext,
        )

        # Minimal valid skill on disk
        skill_dir = tmp_path / "noop_skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: noop\ndescription: noop test skill\n---\n# Noop\n"
        )

        chain = ReasoningChain(steps=[
            AgentSkillStepDescription(
                number=1, title="Noop",
                config=AgentSkillStepConfig(
                    skill=AgentSkillSource(path=str(skill_dir)),
                    task="do nothing",
                    runtime="not-a-real-runtime",
                ),
            ),
        ])

        # Fake LLM client — not actually called because runtime resolution
        # fails before LLM dispatch.
        from unittest.mock import AsyncMock, MagicMock
        api = MagicMock()
        api.get_response_with_retries = AsyncMock(return_value="ok")
        api.get_response = AsyncMock(return_value="ok")
        api.get_response_with_system = AsyncMock(return_value="ok")

        ctx = ReasoningContext(outer_context="N/A", api=api)
        result = await chain.execute_async(ctx)
        assert result.success is False
        failed = result.get_failed_steps()
        assert len(failed) == 1
        assert "not-a-real-runtime" in failed[0].error_message

    @pytest.mark.asyncio
    async def test_local_runtime_default_still_executes(
        self, tmp_path: Path,
    ) -> None:
        """The default ``runtime='local'`` must still drive the existing
        AgentSkill executor end-to-end — no behavioural regression."""
        from mmar_carl import (
            AgentSkillStepConfig,
            AgentSkillStepDescription,
            AgentSkillSource,
            ReasoningChain,
            ReasoningContext,
        )

        skill_dir = tmp_path / "noop_skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: noop\ndescription: a noop skill\n---\n# Noop\n"
            "Just answer concisely.\n"
        )

        from unittest.mock import AsyncMock, MagicMock
        api = MagicMock()
        api.get_response_with_retries = AsyncMock(return_value="hi")
        api.get_response = AsyncMock(return_value="hi")
        api.get_response_with_system = AsyncMock(return_value="hi")

        chain = ReasoningChain(steps=[
            AgentSkillStepDescription(
                number=1, title="Noop",
                config=AgentSkillStepConfig(
                    skill=AgentSkillSource(path=str(skill_dir)),
                    task="hello",
                    # runtime defaults to "local"
                ),
            ),
        ])
        ctx = ReasoningContext(outer_context="N/A", api=api)
        result = await chain.execute_async(ctx)
        assert result.success is True


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------


class TestExports:
    def test_module_exports_public_names(self) -> None:
        from mmar_carl import skill_runtime as mod
        for name in (
            "RuntimeRunResult", "SkillRuntime", "SkillRuntimeError",
            "SkillRuntimeHandle", "LocalSkillRuntime",
            "SKILL_RUNTIME_REGISTRY", "register_skill_runtime",
            "get_skill_runtime", "list_skill_runtimes",
        ):
            assert hasattr(mod, name), f"missing export: {name}"
