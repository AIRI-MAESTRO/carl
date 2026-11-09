""":class:`DockerSkillRuntime`.

The runtime shells out to the ``docker`` CLI (no Python SDK dep). These
tests stub ``shutil.which`` (so they don't need a real docker install)
and ``asyncio.create_subprocess_exec`` (so they don't need a daemon)
and assert the right ``docker run`` flags are emitted for every
documented runtime_config option.

Defaults under test:
- image: ``python:3.12-slim``
- network: ``--network none`` (fail closed)
- ``--memory 2g``, ``--cpus 1``, ``--pids-limit 128``
- dropped capabilities, no-new-privileges, read-only rootfs + bounded tmpfs
- workspace mount: ``in/`` ro, ``out/`` rw
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mmar_carl import (
    DockerNetworkBinding,
    DockerSkillRuntime,
    SkillRuntimeError,
    SkillRuntimeHandle,
    get_skill_runtime,
    list_skill_runtimes,
)

# ---------------------------------------------------------------------------
# Subprocess stubbing
# ---------------------------------------------------------------------------


def _fake_proc(
    *, stdout: bytes = b"ok", stderr: bytes = b"",
    returncode: int = 0, hang: bool = False,
) -> MagicMock:
    """Build a fake asyncio subprocess result."""
    proc = MagicMock()
    if hang:
        stopped = asyncio.Event()

        async def never_returns(*a: Any, **k: Any) -> Any:
            await stopped.wait()
            proc.returncode = -9
            return (b"", b"")
        proc.communicate = never_returns
        proc.kill = MagicMock(side_effect=stopped.set)
    else:
        proc.communicate = AsyncMock(return_value=(stdout, stderr))
        proc.kill = MagicMock()
    proc.returncode = None if hang else returncode
    return proc


class _DockerLifecycleStub:
    """Hermetic docker-run/removal dispatcher for lifecycle regressions."""

    def __init__(self, removal_results: list[MagicMock]) -> None:
        self.calls: list[list[str]] = []
        self.run_proc = _fake_proc(hang=True)
        self.run_proc.pid = 4242
        self._removal_results = iter(removal_results)

    async def create_subprocess(self, *cmd: str, **kwargs: Any) -> MagicMock:
        self.calls.append(list(cmd))
        if Path(cmd[0]).name == "docker" and list(cmd[1:3]) == ["rm", "--force"]:
            return next(self._removal_results)
        return self.run_proc

    @staticmethod
    def stop_cli(proc: MagicMock, **kwargs: Any) -> None:
        proc.kill()

    @property
    def container_name(self) -> str:
        run_cmd = self.calls[0]
        return run_cmd[run_cmd.index("--name") + 1]

    @property
    def removal_calls(self) -> list[list[str]]:
        return [
            call
            for call in self.calls
            if Path(call[0]).name == "docker" and call[1:3] == ["rm", "--force"]
        ]


@pytest.fixture
def docker_on_path():
    """Pretend docker CLI is installed."""
    with patch(
        "mmar_carl.docker_skill_runtime.shutil.which",
        return_value="/usr/local/bin/docker",
    ):
        yield


@pytest.fixture
def captured_calls():
    """List that records every captured docker_cmd from
    create_subprocess_exec.
    """
    return []


@pytest.fixture
def fake_subprocess(captured_calls):
    """Patch create_subprocess_exec to record the docker_cmd argv."""
    async def _fake(*cmd: str, **kwargs: Any) -> MagicMock:
        captured_calls.append(list(cmd))
        return _fake_proc()
    with patch(
        "mmar_carl.docker_skill_runtime.asyncio.create_subprocess_exec",
        new=_fake,
    ):
        yield


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_docker_runtime_pre_registered(self) -> None:
        assert "docker" in list_skill_runtimes()
        cls = get_skill_runtime("docker")
        assert isinstance(cls, DockerSkillRuntime)

    def test_name_attr(self) -> None:
        assert DockerSkillRuntime.name == "docker"


# ---------------------------------------------------------------------------
# prepare()
# ---------------------------------------------------------------------------


class TestPrepare:
    @pytest.mark.asyncio
    async def test_missing_docker_cli_raises(self, tmp_path: Path) -> None:
        with patch(
            "mmar_carl.docker_skill_runtime.shutil.which", return_value=None,
        ):
            runtime = DockerSkillRuntime()
            with pytest.raises(SkillRuntimeError, match="`docker` CLI on PATH"):
                await runtime.prepare(
                    skill=None, workspace=tmp_path / "ws", config={},
                )

    @pytest.mark.asyncio
    async def test_rejects_untyped_internal_network_binding(
        self, tmp_path: Path, docker_on_path,
    ) -> None:
        with pytest.raises(SkillRuntimeError, match="host-issued"):
            await DockerSkillRuntime().prepare(
                None,
                tmp_path / "ws",
                {
                    "network": "allowlist",
                    "network_allowlist": ["api.example.com"],
                    "_network_binding": "chain-selected-network",
                },
            )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "image",
        ["--network=host", " has-space", "python image", "", "bad\x00image"],
    )
    async def test_rejects_image_values_that_can_change_docker_cli_parsing(
        self, tmp_path: Path, docker_on_path, image: str,
    ) -> None:
        with pytest.raises(SkillRuntimeError, match="image reference"):
            await DockerSkillRuntime().prepare(
                None,
                tmp_path / "ws",
                {"image": image},
            )

    @pytest.mark.asyncio
    async def test_defaults_stamped_on_handle(
        self, tmp_path: Path, docker_on_path,
    ) -> None:
        runtime = DockerSkillRuntime()
        handle = await runtime.prepare(
            skill=None, workspace=tmp_path / "ws", config={},
        )
        assert handle.backend["isolation"] == "docker"
        assert handle.backend["image"] == "python:3.12-slim"
        assert handle.backend["mem_limit"] == "2g"
        assert handle.backend["cpu_limit"] == "1"
        assert handle.backend["pids_limit"] == 128
        assert "extra_args" not in handle.backend
        expected_user = (
            f"{os.getuid()}:{os.getgid()}" if os.name == "posix" else None
        )
        assert handle.backend["container_user"] == expected_user
        assert handle.backend["active_container_name"] is None
        assert handle.backend["network_policy"] == "none"
        assert handle.backend["network_enforced"] is True
        assert handle.backend["workspace_output_mode"] == "read_write"
        # Workspace dirs created.
        assert handle.workspace_in.is_dir()
        assert handle.workspace_out.is_dir()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("mem_limit", ["", "0m", "512", "garbage"])
    async def test_rejects_ambiguous_or_invalid_memory_limits(
        self,
        tmp_path: Path,
        docker_on_path,
        mem_limit: str,
    ) -> None:
        workspace = tmp_path / f"ws-{mem_limit or 'empty'}"
        with pytest.raises(SkillRuntimeError, match="explicit b/k/m/g suffix"):
            await DockerSkillRuntime().prepare(
                skill=None,
                workspace=workspace,
                config={"mem_limit": mem_limit},
            )
        assert not workspace.exists()

    @pytest.mark.asyncio
    async def test_rejects_memory_below_docker_minimum(
        self,
        tmp_path: Path,
        docker_on_path,
    ) -> None:
        workspace = tmp_path / "too-small-memory"
        with pytest.raises(SkillRuntimeError, match="at least 6 MiB"):
            await DockerSkillRuntime().prepare(
                skill=None,
                workspace=workspace,
                config={"mem_limit": "1b"},
            )
        assert not workspace.exists()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "config",
        [
            {"cpu_limit": 0},
            {"cpu_limit": float("inf")},
            {"pids_limit": 0},
            {"pids_limit": True},
            {"pids_limit": 1.5},
            {"pids_limit": "16"},
        ],
    )
    async def test_rejects_invalid_resources_before_workspace_creation(
        self,
        tmp_path: Path,
        docker_on_path,
        config: dict[str, Any],
    ) -> None:
        workspace = tmp_path / "invalid-resource"
        with pytest.raises(SkillRuntimeError, match="invalid Docker"):
            await DockerSkillRuntime().prepare(
                skill=None,
                workspace=workspace,
                config=config,
            )
        assert not workspace.exists()

    @pytest.mark.asyncio
    async def test_custom_runtime_config_recognised_keys(
        self, tmp_path: Path, docker_on_path,
    ) -> None:
        runtime = DockerSkillRuntime()
        handle = await runtime.prepare(
            skill=None, workspace=tmp_path / "ws",
            config={
                "image": "alpine:3.19",
                "mem_limit": "512m",
                "cpu_limit": "0.5",
                "pids_limit": 64,
            },
        )
        assert handle.backend["image"] == "alpine:3.19"
        assert handle.backend["mem_limit"] == "512m"
        assert handle.backend["cpu_limit"] == "0.5"
        assert handle.backend["pids_limit"] == 64

    @pytest.mark.asyncio
    async def test_rejects_chain_controlled_extra_args_before_workspace_creation(
        self,
        tmp_path: Path,
        docker_on_path,
    ) -> None:
        workspace = tmp_path / "unsafe-extra-args"
        with pytest.raises(SkillRuntimeError, match="host-owned runtime profile"):
            await DockerSkillRuntime().prepare(
                skill=None,
                workspace=workspace,
                config={"extra_args": ["--privileged", "--network", "host"]},
            )
        assert not workspace.exists()

    @pytest.mark.asyncio
    async def test_host_can_request_read_only_output_mount(
        self,
        tmp_path: Path,
        docker_on_path,
    ) -> None:
        handle = await DockerSkillRuntime().prepare(
            skill=None,
            workspace=tmp_path / "read-only-output",
            config={"workspace_output_mode": "read_only"},
        )

        assert handle.backend["workspace_output_mode"] == "read_only"

    @pytest.mark.asyncio
    async def test_rejects_unknown_output_mount_mode_before_workspace_creation(
        self,
        tmp_path: Path,
        docker_on_path,
    ) -> None:
        workspace = tmp_path / "invalid-output-mode"
        with pytest.raises(SkillRuntimeError, match="workspace_output_mode"):
            await DockerSkillRuntime().prepare(
                skill=None,
                workspace=workspace,
                config={"workspace_output_mode": "unbounded"},
            )
        assert not workspace.exists()

    @pytest.mark.asyncio
    async def test_unknown_network_policy_raises(
        self, tmp_path: Path, docker_on_path,
    ) -> None:
        runtime = DockerSkillRuntime()
        with pytest.raises(SkillRuntimeError, match="Unknown network policy"):
            await runtime.prepare(
                skill=None, workspace=tmp_path / "ws",
                config={"network": "wide-open"},
            )


# ---------------------------------------------------------------------------
# _build_docker_cmd — flag composition
# ---------------------------------------------------------------------------


class TestBuildDockerCmd:
    def _make_handle(
        self,
        tmp_path: Path,
        *,
        image: str = "python:3.12-slim",
        mem_limit: str = "2g",
        cpu_limit: str = "1",
        pids_limit: int = 128,
        network_policy: str = "none",
        network_allowlist: list[str] | None = None,
        extra_args: list[str] | None = None,
        network_binding: DockerNetworkBinding | None = None,
    ) -> SkillRuntimeHandle:
        ws = tmp_path / "ws"
        ws_in = ws / "in"
        ws_out = ws / "out"
        ws.mkdir()
        ws_in.mkdir()
        ws_out.mkdir()
        return SkillRuntimeHandle(
            workspace_root=ws, workspace_in=ws_in, workspace_out=ws_out,
            backend={
                "image": image,
                "mem_limit": mem_limit,
                "cpu_limit": cpu_limit,
                "pids_limit": pids_limit,
                "network_policy": network_policy,
                "network_allowlist": network_allowlist or [],
                "network_binding": network_binding,
                "extra_args": extra_args or [],
                "container_user": (
                    f"{os.getuid()}:{os.getgid()}"
                    if os.name == "posix"
                    else None
                ),
            },
        )

    def test_default_flags(self, tmp_path: Path) -> None:
        runtime = DockerSkillRuntime()
        handle = self._make_handle(tmp_path)
        cmd = runtime._build_docker_cmd(
            handle, ["python", "-c", "print(1)"],
            env=None, timeout=None, cwd=None,
        )
        assert cmd[:2] == ["docker", "run"]
        assert "--pull=never" in cmd
        assert "--rm" in cmd
        assert "--read-only" in cmd
        assert "--cap-drop=ALL" in cmd
        assert "--security-opt=no-new-privileges" in cmd
        assert cmd[cmd.index("--memory") + 1] == "2g"
        assert cmd[cmd.index("--memory-swap") + 1] == "2g"
        assert cmd[cmd.index("--cpus") + 1] == "1"
        assert cmd[cmd.index("--pids-limit") + 1] == "128"
        # tmpfs at /tmp
        assert "--tmpfs" in cmd
        assert "/tmp:rw,noexec,nosuid,nodev,size=64m" in cmd
        # Image immediately precedes the user cmd.
        img_idx = cmd.index("python:3.12-slim")
        assert cmd[img_idx + 1 :] == ["python", "-c", "print(1)"]

    def test_network_none(self, tmp_path: Path) -> None:
        runtime = DockerSkillRuntime()
        handle = self._make_handle(tmp_path, network_policy="none")
        cmd = runtime._build_docker_cmd(
            handle, ["echo"], env=None, timeout=None, cwd=None,
        )
        net_idx = cmd.index("--network")
        assert cmd[net_idx + 1] == "none"

    def test_network_host(self, tmp_path: Path) -> None:
        runtime = DockerSkillRuntime()
        handle = self._make_handle(tmp_path, network_policy="host")
        cmd = runtime._build_docker_cmd(
            handle, ["echo"], env=None, timeout=None, cwd=None,
        )
        net_idx = cmd.index("--network")
        assert cmd[net_idx + 1] == "host"

    def test_network_allowlist_fails_closed_without_egress_provider(
        self, tmp_path: Path,
    ) -> None:
        runtime = DockerSkillRuntime()
        handle = self._make_handle(
            tmp_path, network_policy="allowlist",
            network_allowlist=["my-net", "extra-host"],
        )
        cmd = runtime._build_docker_cmd(
            handle, ["echo"], env=None, timeout=None, cwd=None,
        )
        net_idx = cmd.index("--network")
        assert cmd[net_idx + 1] == "none"

    def test_empty_allowlist_falls_back_to_none(self, tmp_path: Path) -> None:
        runtime = DockerSkillRuntime()
        handle = self._make_handle(
            tmp_path, network_policy="allowlist", network_allowlist=[],
        )
        cmd = runtime._build_docker_cmd(
            handle, ["echo"], env=None, timeout=None, cwd=None,
        )
        net_idx = cmd.index("--network")
        assert cmd[net_idx + 1] == "none"

    def test_allowlist_uses_only_typed_host_binding(self, tmp_path: Path) -> None:
        handle = self._make_handle(
            tmp_path,
            network_policy="allowlist",
            network_allowlist=["api.example.com"],
            network_binding=DockerNetworkBinding("carl-egress-api"),
        )

        cmd = DockerSkillRuntime()._build_docker_cmd(
            handle, ["echo"], env=None, timeout=None, cwd=None,
        )

        assert cmd[cmd.index("--network") + 1] == "carl-egress-api"
        assert "api.example.com" not in cmd

    def test_workspace_mounts(self, tmp_path: Path) -> None:
        runtime = DockerSkillRuntime()
        handle = self._make_handle(tmp_path)
        cmd = runtime._build_docker_cmd(
            handle, ["echo"], env=None, timeout=None, cwd=None,
        )
        # Two -v entries: in/ ro, out/ rw.
        mounts = [cmd[i + 1] for i, x in enumerate(cmd) if x == "-v"]
        assert any(m.endswith(":/workspace/in:ro") for m in mounts)
        assert any(m.endswith(":/workspace/out") for m in mounts)

    def test_env_flags(self, tmp_path: Path) -> None:
        runtime = DockerSkillRuntime()
        handle = self._make_handle(tmp_path)
        cmd = runtime._build_docker_cmd(
            handle, ["echo"], env={"FOO": "bar", "BAZ": "qux"},
            timeout=None, cwd=None,
        )
        env_pairs = [cmd[i + 1] for i, x in enumerate(cmd) if x == "-e"]
        assert "FOO=bar" in env_pairs
        assert "BAZ=qux" in env_pairs

    def test_cwd_flag(self, tmp_path: Path) -> None:
        runtime = DockerSkillRuntime()
        handle = self._make_handle(tmp_path)
        cmd = runtime._build_docker_cmd(
            handle, ["echo"], env=None, timeout=None, cwd="/workspace/out",
        )
        assert "-w" in cmd
        assert cmd[cmd.index("-w") + 1] == "/workspace/out"

    def test_timeout_does_not_add_unapproved_environment(self, tmp_path: Path) -> None:
        runtime = DockerSkillRuntime()
        handle = self._make_handle(tmp_path)
        cmd = runtime._build_docker_cmd(
            handle, ["echo"], env=None, timeout=30.0, cwd=None,
        )
        env_pairs = [cmd[i + 1] for i, x in enumerate(cmd) if x == "-e"]
        assert not any(p.startswith("CARL_TIMEOUT_S=") for p in env_pairs)

    def test_extra_args_on_manual_handle_are_not_passed(self, tmp_path: Path) -> None:
        runtime = DockerSkillRuntime()
        handle = self._make_handle(
            tmp_path,
            extra_args=["--cap-drop", "ALL", "--user", "1000:1000"],
        )
        cmd = runtime._build_docker_cmd(
            handle, ["echo"], env=None, timeout=None, cwd=None,
        )
        assert "--cap-drop" not in cmd
        assert cmd.count("--cap-drop=ALL") == 1
        if os.name == "posix":
            user_flags = [i for i, value in enumerate(cmd) if value == "--user"]
            assert len(user_flags) == 1
            assert cmd[user_flags[-1] + 1] == f"{os.getuid()}:{os.getgid()}"

    @pytest.mark.skipif(os.name != "posix", reason="POSIX UID:GID contract")
    def test_host_uid_gid_is_the_only_user_flag(
        self, tmp_path: Path,
    ) -> None:
        runtime = DockerSkillRuntime()
        handle = self._make_handle(
            tmp_path,
            extra_args=["--user", "0:0"],
        )
        cmd = runtime._build_docker_cmd(
            handle, ["echo"], env=None, timeout=None, cwd=None,
        )
        user_flags = [i for i, value in enumerate(cmd) if value == "--user"]
        assert len(user_flags) == 1
        assert cmd[user_flags[-1] + 1] == f"{os.getuid()}:{os.getgid()}"

    def test_user_cmd_appended_after_image(self, tmp_path: Path) -> None:
        runtime = DockerSkillRuntime()
        handle = self._make_handle(tmp_path, image="alpine:3.19")
        cmd = runtime._build_docker_cmd(
            handle, ["sh", "-c", "echo hi"], env=None, timeout=None, cwd=None,
        )
        img_idx = cmd.index("alpine:3.19")
        assert cmd[img_idx + 1 :] == ["sh", "-c", "echo hi"]


# ---------------------------------------------------------------------------
# run() — end-to-end with stubbed subprocess
# ---------------------------------------------------------------------------


class TestRun:
    @pytest.mark.asyncio
    async def test_run_captures_stdout(
        self, tmp_path: Path, docker_on_path, fake_subprocess, captured_calls,
    ) -> None:
        runtime = DockerSkillRuntime()
        handle = await runtime.prepare(
            skill=None, workspace=tmp_path / "ws", config={},
        )
        result = await runtime.run(handle, ["python", "-c", "print(1)"])
        assert result.exit_code == 0
        assert result.stdout == b"ok"
        assert captured_calls  # one docker run dispatched
        assert captured_calls[0][:2] == ["/usr/local/bin/docker", "run"]

    @pytest.mark.asyncio
    async def test_run_timeout_returns_exit_124(
        self, tmp_path: Path, docker_on_path,
    ) -> None:
        lifecycle = _DockerLifecycleStub([_fake_proc(stdout=b"removed\n")])

        with (
            patch(
                "mmar_carl.docker_skill_runtime.asyncio.create_subprocess_exec",
                new=lifecycle.create_subprocess,
            ),
            patch(
                "mmar_carl.docker_skill_runtime._kill_process_group",
                side_effect=lifecycle.stop_cli,
            ),
        ):
            runtime = DockerSkillRuntime()
            handle = await runtime.prepare(
                skill=None, workspace=tmp_path / "ws", config={},
            )
            result = await runtime.run(handle, ["sleep", "100"], timeout=0.05)
        assert result.exit_code == 124
        assert b"[timeout after" in result.stderr
        assert lifecycle.removal_calls == [[
            "/usr/local/bin/docker", "rm", "--force", lifecycle.container_name,
        ]]
        assert handle.backend["active_container_name"] is None

    @pytest.mark.asyncio
    async def test_timeout_removal_failure_surfaces_and_cleanup_retries(
        self, tmp_path: Path, docker_on_path,
    ) -> None:
        # Regression guard: daemon cleanup must not depend on the local pid
        # being a MagicMock or on the docker CLI still being alive.
        lifecycle = _DockerLifecycleStub([
            _fake_proc(
                stderr=b"daemon refused forced removal",
                returncode=1,
            ),
            _fake_proc(stdout=b"removed\n"),
        ])

        with (
            patch(
                "mmar_carl.docker_skill_runtime.asyncio.create_subprocess_exec",
                new=lifecycle.create_subprocess,
            ),
            patch(
                "mmar_carl.docker_skill_runtime._kill_process_group",
                side_effect=lifecycle.stop_cli,
            ),
        ):
            runtime = DockerSkillRuntime()
            handle = await runtime.prepare(
                skill=None, workspace=tmp_path / "ws", config={},
            )
            with pytest.raises(
                SkillRuntimeError,
                match="failed to remove active container.*timeout/cancellation",
            ):
                await runtime.run(handle, ["sleep", "100"], timeout=0.05)

            assert (
                handle.backend["active_container_name"]
                == lifecycle.container_name
            )
            assert handle.workspace_root.exists()

            await runtime.cleanup(handle)

        assert len(lifecycle.removal_calls) == 2
        assert handle.backend["active_container_name"] is None
        assert not handle.workspace_root.exists()

    @pytest.mark.asyncio
    async def test_cancellation_identity_survives_removal_failure_and_cleanup_retries(
        self, tmp_path: Path, docker_on_path,
    ) -> None:
        lifecycle = _DockerLifecycleStub([
            _fake_proc(stderr=b"daemon unavailable", returncode=1),
            _fake_proc(stdout=b"removed\n"),
        ])

        with (
            patch(
                "mmar_carl.docker_skill_runtime.asyncio.create_subprocess_exec",
                new=lifecycle.create_subprocess,
            ),
            patch(
                "mmar_carl.docker_skill_runtime._kill_process_group",
                side_effect=lifecycle.stop_cli,
            ),
        ):
            runtime = DockerSkillRuntime()
            handle = await runtime.prepare(
                skill=None, workspace=tmp_path / "ws", config={},
            )
            task = asyncio.create_task(runtime.run(handle, ["sleep", "100"]))
            await asyncio.sleep(0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

            assert handle.backend["active_container_name"] == lifecycle.container_name
            assert handle.workspace_root.exists()
            await runtime.cleanup(handle)

        assert len(lifecycle.removal_calls) == 2
        assert handle.backend["active_container_name"] is None
        assert not handle.workspace_root.exists()

    @pytest.mark.asyncio
    async def test_oserror_on_launch_wrapped(
        self, tmp_path: Path, docker_on_path,
    ) -> None:
        async def _fail(*cmd, **kwargs):
            raise OSError("docker daemon not running")
        with patch(
            "mmar_carl.docker_skill_runtime.asyncio.create_subprocess_exec",
            new=_fail,
        ):
            runtime = DockerSkillRuntime()
            handle = await runtime.prepare(
                skill=None, workspace=tmp_path / "ws", config={},
            )
            with pytest.raises(SkillRuntimeError, match="failed to launch docker"):
                await runtime.run(handle, ["echo"])

    @pytest.mark.asyncio
    async def test_docker_cli_exit_125_is_runtime_error(
        self, tmp_path: Path, docker_on_path,
    ) -> None:
        async def _fail_before_container(*cmd, **kwargs):
            return _fake_proc(
                stderr=b"Cannot connect to the Docker daemon",
                returncode=125,
            )

        with patch(
            "mmar_carl.docker_skill_runtime.asyncio.create_subprocess_exec",
            new=_fail_before_container,
        ):
            runtime = DockerSkillRuntime()
            handle = await runtime.prepare(
                skill=None, workspace=tmp_path / "ws", config={},
            )
            with pytest.raises(SkillRuntimeError, match="before command execution"):
                await runtime.run(handle, ["echo"])

    @pytest.mark.asyncio
    async def test_runtime_config_flows_into_docker_args(
        self, tmp_path: Path, docker_on_path, fake_subprocess, captured_calls,
    ) -> None:
        """Custom mem/cpu/image from runtime_config must reach the
        emitted docker run argv."""
        runtime = DockerSkillRuntime()
        handle = await runtime.prepare(
            skill=None, workspace=tmp_path / "ws",
            config={
                "image": "alpine:3.19",
                "mem_limit": "512m",
                "cpu_limit": "0.5",
                "network": "host",
            },
        )
        await runtime.run(handle, ["true"])
        cmd = captured_calls[0]
        assert cmd[cmd.index("--memory") + 1] == "512m"
        assert cmd[cmd.index("--memory-swap") + 1] == "512m"
        assert cmd[cmd.index("--cpus") + 1] == "0.5"
        assert cmd[cmd.index("--network") + 1] == "host"
        assert "alpine:3.19" in cmd


# ---------------------------------------------------------------------------
# read_file / write_file / cleanup — same contract as LocalSkillRuntime
# ---------------------------------------------------------------------------


class TestFileOps:
    @pytest.mark.asyncio
    async def test_write_then_read_roundtrip(
        self, tmp_path: Path, docker_on_path,
    ) -> None:
        runtime = DockerSkillRuntime()
        handle = await runtime.prepare(
            skill=None, workspace=tmp_path / "ws", config={},
        )
        await runtime.write_file(handle, "out/result.txt", b"hello")
        data = await runtime.read_file(handle, "out/result.txt")
        assert data == b"hello"

    @pytest.mark.asyncio
    async def test_absolute_path_rejected(
        self, tmp_path: Path, docker_on_path,
    ) -> None:
        runtime = DockerSkillRuntime()
        handle = await runtime.prepare(
            skill=None, workspace=tmp_path / "ws", config={},
        )
        with pytest.raises(SkillRuntimeError, match="must be relative"):
            await runtime.write_file(handle, "/etc/passwd", b"x")

    @pytest.mark.asyncio
    async def test_dotdot_escape_rejected(
        self, tmp_path: Path, docker_on_path,
    ) -> None:
        runtime = DockerSkillRuntime()
        handle = await runtime.prepare(
            skill=None, workspace=tmp_path / "ws", config={},
        )
        with pytest.raises(SkillRuntimeError, match="escapes workspace"):
            await runtime.write_file(handle, "../escape.txt", b"x")

    @pytest.mark.asyncio
    async def test_cleanup_removes_workspace(
        self, tmp_path: Path, docker_on_path,
    ) -> None:
        runtime = DockerSkillRuntime()
        handle = await runtime.prepare(
            skill=None, workspace=tmp_path / "ws", config={},
        )
        await runtime.cleanup(handle)
        assert not handle.workspace_root.exists()

    @pytest.mark.asyncio
    async def test_cleanup_persisted_keeps_workspace(
        self, tmp_path: Path, docker_on_path,
    ) -> None:
        runtime = DockerSkillRuntime()
        handle = await runtime.prepare(
            skill=None, workspace=tmp_path / "ws", config={},
        )
        handle.backend["persisted"] = True
        await runtime.cleanup(handle)
        assert handle.workspace_root.exists()


# ---------------------------------------------------------------------------
# Top-level export sanity
# ---------------------------------------------------------------------------


def test_docker_runtime_importable_from_top_level() -> None:
    import mmar_carl
    assert hasattr(mmar_carl, "DockerSkillRuntime")
    assert mmar_carl.DockerSkillRuntime is DockerSkillRuntime
