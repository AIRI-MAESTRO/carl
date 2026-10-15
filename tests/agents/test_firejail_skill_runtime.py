""":class:`FirejailSkillRuntime`.

Linux-only lightweight sandbox sister to :class:`DockerSkillRuntime`.
Same hermetic test strategy: stub ``shutil.which`` so the tests don't
need firejail installed, and stub ``asyncio.create_subprocess_exec`` so
they don't actually run a sandbox — they assert the right ``firejail``
flags are emitted.

Defaults under test:
- ``--quiet`` + ``--noprofile`` (no surprise profile from /etc/firejail/).
- private tmp/dev, dropped capabilities, no-new-privileges, seccomp.
- ``--net=none`` (fail closed — fits the spec'd "P2" lightweight model).
- ``--private=<workspace>`` (private home points at the workspace).
- ``--rlimit-as`` virtual-memory cap (default 2 GiB).
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mmar_carl import (
    FirejailNetworkBinding,
    FirejailSkillRuntime,
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
    proc = MagicMock()
    if hang:
        async def never_returns(*a: Any, **k: Any) -> Any:
            await asyncio.sleep(10)
            return (b"", b"")
        proc.communicate = never_returns
    else:
        proc.communicate = AsyncMock(return_value=(stdout, stderr))
    proc.kill = MagicMock()
    proc.returncode = returncode
    return proc


@pytest.fixture
def firejail_on_path():
    """Pretend firejail CLI is installed."""
    def _which(name: str, **_kwargs: Any) -> str | None:
        return {
            "firejail": "/usr/bin/firejail",
            "env": "/usr/bin/env",
        }.get(name)

    with patch(
        "mmar_carl.firejail_skill_runtime.shutil.which",
        side_effect=_which,
    ):
        yield


@pytest.fixture
def captured_calls():
    return []


@pytest.fixture
def fake_subprocess(captured_calls):
    async def _fake(*cmd: str, **kwargs: Any) -> MagicMock:
        captured_calls.append(list(cmd))
        return _fake_proc()
    with patch(
        "mmar_carl.firejail_skill_runtime.asyncio.create_subprocess_exec",
        new=_fake,
    ):
        yield


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_firejail_runtime_pre_registered(self) -> None:
        assert "firejail" in list_skill_runtimes()
        instance = get_skill_runtime("firejail")
        assert isinstance(instance, FirejailSkillRuntime)

    def test_name_attr(self) -> None:
        assert FirejailSkillRuntime.name == "firejail"


# ---------------------------------------------------------------------------
# prepare()
# ---------------------------------------------------------------------------


class TestPrepare:
    @pytest.mark.asyncio
    async def test_missing_firejail_cli_raises(self, tmp_path: Path) -> None:
        with patch(
            "mmar_carl.firejail_skill_runtime.shutil.which",
            return_value=None,
        ):
            runtime = FirejailSkillRuntime()
            with pytest.raises(
                SkillRuntimeError, match="`firejail` CLI on PATH",
            ):
                await runtime.prepare(
                    skill=None, workspace=tmp_path / "ws", config={},
                )

    @pytest.mark.asyncio
    async def test_rejects_untyped_internal_network_binding(
        self, tmp_path: Path, firejail_on_path,
    ) -> None:
        with pytest.raises(SkillRuntimeError, match="host-issued"):
            await FirejailSkillRuntime().prepare(
                None,
                tmp_path / "ws",
                {
                    "network": "allowlist",
                    "network_allowlist": ["api.example.com"],
                    "_network_binding": "chain-selected-interface",
                },
            )

    @pytest.mark.asyncio
    async def test_defaults_stamped_on_handle(
        self, tmp_path: Path, firejail_on_path,
    ) -> None:
        runtime = FirejailSkillRuntime()
        handle = await runtime.prepare(
            skill=None, workspace=tmp_path / "ws", config={},
        )
        assert handle.backend["isolation"] == "firejail"
        assert handle.backend["firejail_executable"] == "/usr/bin/firejail"
        assert handle.backend["sandbox_env_executable"] == "/usr/bin/env"
        assert handle.backend["rlimit_as_mb"] == 2048
        assert handle.backend["rlimit_cpu_s"] is None
        assert "extra_args" not in handle.backend
        assert handle.backend["network_policy"] == "none"
        assert handle.backend["network_enforced"] is True
        assert handle.workspace_in.is_dir()
        assert handle.workspace_out.is_dir()

    @pytest.mark.asyncio
    async def test_memory_limit_preserves_exact_canonical_bytes(
        self,
        tmp_path: Path,
        firejail_on_path,
    ) -> None:
        runtime = FirejailSkillRuntime()
        one_byte = await runtime.prepare(
            skill=None,
            workspace=tmp_path / "one-byte",
            config={"mem_limit": "1b"},
        )
        over_one_mib = await runtime.prepare(
            skill=None,
            workspace=tmp_path / "over-one-mib",
            config={"mem_limit": "1025k"},
        )
        assert one_byte.backend["rlimit_as_bytes"] == 1
        assert over_one_mib.backend["rlimit_as_bytes"] == 1025 * 1024
        one_byte_cmd = runtime._build_firejail_cmd(one_byte, ["echo"], cwd=None)
        over_one_mib_cmd = runtime._build_firejail_cmd(
            over_one_mib,
            ["echo"],
            cwd=None,
        )
        assert "--rlimit-as=1" in one_byte_cmd
        assert f"--rlimit-as={1025 * 1024}" in over_one_mib_cmd

    @pytest.mark.asyncio
    async def test_rejects_ambiguous_memory_limit(
        self,
        tmp_path: Path,
        firejail_on_path,
    ) -> None:
        workspace = tmp_path / "ambiguous"
        with pytest.raises(SkillRuntimeError, match="explicit b/k/m/g suffix"):
            await FirejailSkillRuntime().prepare(
                skill=None,
                workspace=workspace,
                config={"mem_limit": "512"},
            )
        assert not workspace.exists()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "config",
        [
            {"rlimit_as_mb": 0},
            {"rlimit_cpu_s": float("inf")},
            {"rlimit_cpu_s": 1.5},
            {"pids_limit": 0},
            {"pids_limit": True},
        ],
    )
    async def test_rejects_invalid_resources_before_workspace_creation(
        self,
        tmp_path: Path,
        firejail_on_path,
        config: dict[str, Any],
    ) -> None:
        workspace = tmp_path / "invalid-resource"
        with pytest.raises(SkillRuntimeError, match="invalid"):
            await FirejailSkillRuntime().prepare(
                skill=None,
                workspace=workspace,
                config=config,
            )
        assert not workspace.exists()

    @pytest.mark.asyncio
    async def test_relative_discovery_is_resolved_before_run(
        self, tmp_path: Path,
    ) -> None:
        with patch(
            "mmar_carl.firejail_skill_runtime.shutil.which",
            side_effect=lambda name, **_kwargs: (
                "host-bin/firejail" if name == "firejail" else "/usr/bin/env"
            ),
        ):
            handle = await FirejailSkillRuntime().prepare(
                skill=None, workspace=tmp_path / "ws", config={},
            )

        launcher = Path(handle.backend["firejail_executable"])
        assert launcher.is_absolute()
        assert launcher == Path("host-bin/firejail").resolve()

    @pytest.mark.asyncio
    async def test_rejects_chain_controlled_profile(
        self, tmp_path: Path, firejail_on_path,
    ) -> None:
        workspace = tmp_path / "chain-profile"
        with pytest.raises(SkillRuntimeError, match="chain-controlled profiles"):
            await FirejailSkillRuntime().prepare(
                skill=None,
                workspace=workspace,
                config={"profile": "python"},
            )
        assert not workspace.exists()

    @pytest.mark.asyncio
    async def test_rejects_chain_controlled_extra_args_before_workspace_creation(
        self,
        tmp_path: Path,
        firejail_on_path,
    ) -> None:
        workspace = tmp_path / "unsafe-extra-args"
        with pytest.raises(SkillRuntimeError, match="host-owned runtime profile"):
            await FirejailSkillRuntime().prepare(
                skill=None,
                workspace=workspace,
                config={"extra_args": ["--net=eth0", "--rlimit-as=999999999"]},
            )
        assert not workspace.exists()

    @pytest.mark.asyncio
    async def test_unknown_network_policy_raises(
        self, tmp_path: Path, firejail_on_path,
    ) -> None:
        runtime = FirejailSkillRuntime()
        with pytest.raises(SkillRuntimeError, match="Unknown network policy"):
            await runtime.prepare(
                skill=None, workspace=tmp_path / "ws",
                config={"network": "wide-open"},
            )


# ---------------------------------------------------------------------------
# _build_firejail_cmd — flag composition
# ---------------------------------------------------------------------------


class TestBuildFirejailCmd:
    def _make_handle(
        self,
        tmp_path: Path,
        *,
        profile: Optional[str] = None,
        network_policy: str = "none",
        network_allowlist: list[str] | None = None,
        rlimit_as_mb: int = 2048,
        rlimit_cpu_s: Optional[int] = None,
        extra_args: list[str] | None = None,
        network_binding: FirejailNetworkBinding | None = None,
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
                "firejail_executable": "/usr/bin/firejail",
                "sandbox_env_executable": "/usr/bin/env",
                "profile": profile,
                "rlimit_as_mb": rlimit_as_mb,
                "rlimit_cpu_s": rlimit_cpu_s,
                "extra_args": extra_args or [],
                "network_policy": network_policy,
                "network_allowlist": network_allowlist or [],
                "network_binding": network_binding,
            },
        )

    def test_default_flags(self, tmp_path: Path) -> None:
        runtime = FirejailSkillRuntime()
        handle = self._make_handle(tmp_path)
        cmd = runtime._build_firejail_cmd(
            handle, ["python", "-c", "print(1)"], cwd=None,
        )
        assert cmd[0] == "/usr/bin/firejail"
        assert "--quiet" in cmd
        assert "--noprofile" in cmd
        assert "--private-tmp" in cmd
        assert "--private-dev" in cmd
        assert "--caps.drop=all" in cmd
        assert "--nonewprivs" in cmd
        assert "--seccomp" in cmd
        assert "--net=none" in cmd
        # rlimit-as in bytes
        assert any(c == f"--rlimit-as={2048 * 1024 * 1024}" for c in cmd)
        # private= points at workspace
        assert any(
            c.startswith("--private=") and str(handle.workspace_root) in c
            for c in cmd
        )
        # `--` separator before the user cmd
        sep = cmd.index("--")
        assert cmd[sep + 1 :] == ["python", "-c", "print(1)"]

    def test_manual_handle_cannot_select_profile(self, tmp_path: Path) -> None:
        runtime = FirejailSkillRuntime()
        handle = self._make_handle(tmp_path, profile="python")
        cmd = runtime._build_firejail_cmd(handle, ["echo"], cwd=None)
        assert "--profile=python" not in cmd
        assert "--noprofile" in cmd

    def test_network_none(self, tmp_path: Path) -> None:
        runtime = FirejailSkillRuntime()
        handle = self._make_handle(tmp_path, network_policy="none")
        cmd = runtime._build_firejail_cmd(handle, ["echo"], cwd=None)
        assert "--net=none" in cmd

    def test_network_host_omits_net_flag(self, tmp_path: Path) -> None:
        """Host policy → no ``--net=*`` flag (firejail's documented
        default behaviour)."""
        runtime = FirejailSkillRuntime()
        handle = self._make_handle(tmp_path, network_policy="host")
        cmd = runtime._build_firejail_cmd(handle, ["echo"], cwd=None)
        assert not any(c.startswith("--net=") for c in cmd)

    def test_network_allowlist_fails_closed_without_egress_provider(
        self, tmp_path: Path,
    ) -> None:
        runtime = FirejailSkillRuntime()
        handle = self._make_handle(
            tmp_path, network_policy="allowlist",
            network_allowlist=["br-care0", "extra"],
        )
        cmd = runtime._build_firejail_cmd(handle, ["echo"], cwd=None)
        assert "--net=none" in cmd

    def test_empty_allowlist_falls_back_to_none(self, tmp_path: Path) -> None:
        runtime = FirejailSkillRuntime()
        handle = self._make_handle(
            tmp_path, network_policy="allowlist", network_allowlist=[],
        )
        cmd = runtime._build_firejail_cmd(handle, ["echo"], cwd=None)
        assert "--net=none" in cmd

    def test_allowlist_uses_only_typed_host_binding(self, tmp_path: Path) -> None:
        handle = self._make_handle(
            tmp_path,
            network_policy="allowlist",
            network_allowlist=["api.example.com"],
            network_binding=FirejailNetworkBinding("carl-egress0"),
        )

        cmd = FirejailSkillRuntime()._build_firejail_cmd(
            handle, ["echo"], cwd=None,
        )

        assert "--net=carl-egress0" in cmd
        assert "api.example.com" not in cmd

    def test_rlimit_as_in_bytes(self, tmp_path: Path) -> None:
        runtime = FirejailSkillRuntime()
        handle = self._make_handle(tmp_path, rlimit_as_mb=512)
        cmd = runtime._build_firejail_cmd(handle, ["echo"], cwd=None)
        assert f"--rlimit-as={512 * 1024 * 1024}" in cmd

    def test_rlimit_cpu_optional(self, tmp_path: Path) -> None:
        runtime = FirejailSkillRuntime()
        handle = self._make_handle(tmp_path, rlimit_cpu_s=15)
        cmd = runtime._build_firejail_cmd(handle, ["echo"], cwd=None)
        assert "--rlimit-cpu=15" in cmd

    def test_rlimit_cpu_omitted_when_none(self, tmp_path: Path) -> None:
        runtime = FirejailSkillRuntime()
        handle = self._make_handle(tmp_path, rlimit_cpu_s=None)
        cmd = runtime._build_firejail_cmd(handle, ["echo"], cwd=None)
        assert not any(c.startswith("--rlimit-cpu=") for c in cmd)

    def test_chdir_flag(self, tmp_path: Path) -> None:
        runtime = FirejailSkillRuntime()
        handle = self._make_handle(tmp_path)
        cmd = runtime._build_firejail_cmd(handle, ["echo"], cwd="/work")
        assert "--chdir=/work" in cmd

    def test_extra_args_on_manual_handle_are_not_passed(self, tmp_path: Path) -> None:
        runtime = FirejailSkillRuntime()
        handle = self._make_handle(
            tmp_path, extra_args=["--seccomp", "--nogroups"],
        )
        cmd = runtime._build_firejail_cmd(handle, ["echo"], cwd=None)
        # seccomp is part of CARL's fixed baseline, not copied from the manual
        # handle. The other chain-supplied token must remain absent.
        assert cmd.count("--seccomp") == 1
        assert "--nogroups" not in cmd

    def test_user_cmd_after_double_dash(self, tmp_path: Path) -> None:
        runtime = FirejailSkillRuntime()
        handle = self._make_handle(tmp_path)
        cmd = runtime._build_firejail_cmd(
            handle, ["sh", "-c", "echo hi"], cwd=None,
        )
        sep = cmd.index("--")
        assert cmd[sep + 1 :] == ["sh", "-c", "echo hi"]

    def test_requires_prepared_absolute_launcher(self, tmp_path: Path) -> None:
        runtime = FirejailSkillRuntime()
        handle = self._make_handle(tmp_path)
        handle.backend["firejail_executable"] = "firejail"

        with pytest.raises(SkillRuntimeError, match="absolute launcher path"):
            runtime._build_firejail_cmd(handle, ["true"], cwd=None)

    def test_requested_env_uses_post_separator_wrapper(
        self, tmp_path: Path,
    ) -> None:
        runtime = FirejailSkillRuntime()
        handle = self._make_handle(tmp_path)
        requested_env = {
            "PATH": "/sandbox/bin",
            "CUSTOM": "value with spaces=and-equals",
        }

        cmd = runtime._build_firejail_cmd(
            handle, ["/usr/bin/env"], env=requested_env, cwd=None,
        )

        separator = cmd.index("--")
        assert not any(item.startswith("--env=") for item in cmd[:separator])
        assert cmd[separator + 1 :] == [
            "/usr/bin/env",
            "-i",
            "--",
            "PATH=/sandbox/bin",
            "CUSTOM=value with spaces=and-equals",
            "/usr/bin/env",
        ]


# ---------------------------------------------------------------------------
# run() — end-to-end with stubbed subprocess
# ---------------------------------------------------------------------------


class TestRun:
    @pytest.mark.asyncio
    async def test_run_captures_stdout(
        self, tmp_path: Path, firejail_on_path, fake_subprocess, captured_calls,
    ) -> None:
        runtime = FirejailSkillRuntime()
        handle = await runtime.prepare(
            skill=None, workspace=tmp_path / "ws", config={},
        )
        result = await runtime.run(handle, ["python", "-c", "print(1)"])
        assert result.exit_code == 0
        assert result.stdout == b"ok"
        assert captured_calls
        assert captured_calls[0][0] == "/usr/bin/firejail"

    @pytest.mark.asyncio
    async def test_step_env_cannot_control_firejail_launcher(
        self, tmp_path: Path, firejail_on_path,
    ) -> None:
        captured: dict[str, Any] = {}

        async def _capture(*cmd: str, **kwargs: Any) -> MagicMock:
            captured["cmd"] = list(cmd)
            captured["kwargs"] = kwargs
            return _fake_proc()

        with patch(
            "mmar_carl.firejail_skill_runtime.asyncio.create_subprocess_exec",
            new=_capture,
        ):
            runtime = FirejailSkillRuntime()
            handle = await runtime.prepare(
                skill=None, workspace=tmp_path / "ws", config={},
            )
            requested_env = {
                "PATH": str(tmp_path / "attacker-bin"),
                "LD_PRELOAD": str(tmp_path / "evil.so"),
                "FIREJAIL_FILE_COPY_LIMIT": "0",
                "CUSTOM": "sandbox-only",
            }
            await runtime.run(handle, ["/usr/bin/true"], env=requested_env)

        command = captured["cmd"]
        launcher_env = captured["kwargs"]["env"]
        assert command[0] == "/usr/bin/firejail"
        assert launcher_env == runtime._trusted_launcher_env(handle)
        assert launcher_env["PATH"] == os.defpath
        assert "LD_PRELOAD" not in launcher_env
        assert "FIREJAIL_FILE_COPY_LIMIT" not in launcher_env

        separator = command.index("--")
        assert not any(
            item.startswith("--env=") for item in command[:separator]
        )
        wrapper = command[separator + 1 :]
        assert wrapper[:3] == ["/usr/bin/env", "-i", "--"]
        assignment_end = 3 + len(requested_env)
        sandbox_env = dict(
            item.split("=", 1) for item in wrapper[3:assignment_end]
        )
        assert sandbox_env == requested_env
        assert wrapper[assignment_end:] == ["/usr/bin/true"]

    @pytest.mark.asyncio
    async def test_run_timeout_returns_exit_124(
        self, tmp_path: Path, firejail_on_path,
    ) -> None:
        async def _hang(*cmd, **kwargs):
            return _fake_proc(hang=True)
        with patch(
            "mmar_carl.firejail_skill_runtime.asyncio.create_subprocess_exec",
            new=_hang,
        ):
            runtime = FirejailSkillRuntime()
            handle = await runtime.prepare(
                skill=None, workspace=tmp_path / "ws", config={},
            )
            result = await runtime.run(handle, ["sleep", "100"], timeout=0.05)
        assert result.exit_code == 124
        assert b"[timeout after" in result.stderr

    @pytest.mark.asyncio
    async def test_oserror_on_launch_wrapped(
        self, tmp_path: Path, firejail_on_path,
    ) -> None:
        async def _fail(*cmd, **kwargs):
            raise OSError("permission denied")
        with patch(
            "mmar_carl.firejail_skill_runtime.asyncio.create_subprocess_exec",
            new=_fail,
        ):
            runtime = FirejailSkillRuntime()
            handle = await runtime.prepare(
                skill=None, workspace=tmp_path / "ws", config={},
            )
            with pytest.raises(SkillRuntimeError, match="failed to launch firejail"):
                await runtime.run(handle, ["echo"])

    @pytest.mark.asyncio
    async def test_safe_runtime_config_flows_into_firejail_args(
        self, tmp_path: Path, firejail_on_path, fake_subprocess,
        captured_calls,
    ) -> None:
        runtime = FirejailSkillRuntime()
        handle = await runtime.prepare(
            skill=None, workspace=tmp_path / "ws",
            config={
                "rlimit_as_mb": 256,
                "network": "host",
            },
        )
        await runtime.run(handle, ["true"])
        cmd = captured_calls[0]
        assert "--noprofile" in cmd
        assert f"--rlimit-as={256 * 1024 * 1024}" in cmd
        # host policy → no --net flag
        assert not any(c.startswith("--net=") for c in cmd)


# ---------------------------------------------------------------------------
# File ops — same contract as Local + Docker
# ---------------------------------------------------------------------------


class TestFileOps:
    @pytest.mark.asyncio
    async def test_write_then_read_roundtrip(
        self, tmp_path: Path, firejail_on_path,
    ) -> None:
        runtime = FirejailSkillRuntime()
        handle = await runtime.prepare(
            skill=None, workspace=tmp_path / "ws", config={},
        )
        await runtime.write_file(handle, "out/result.txt", b"hello")
        data = await runtime.read_file(handle, "out/result.txt")
        assert data == b"hello"

    @pytest.mark.asyncio
    async def test_absolute_path_rejected(
        self, tmp_path: Path, firejail_on_path,
    ) -> None:
        runtime = FirejailSkillRuntime()
        handle = await runtime.prepare(
            skill=None, workspace=tmp_path / "ws", config={},
        )
        with pytest.raises(SkillRuntimeError, match="must be relative"):
            await runtime.write_file(handle, "/etc/passwd", b"x")

    @pytest.mark.asyncio
    async def test_dotdot_escape_rejected(
        self, tmp_path: Path, firejail_on_path,
    ) -> None:
        runtime = FirejailSkillRuntime()
        handle = await runtime.prepare(
            skill=None, workspace=tmp_path / "ws", config={},
        )
        with pytest.raises(SkillRuntimeError, match="escapes workspace"):
            await runtime.write_file(handle, "../escape.txt", b"x")

    @pytest.mark.asyncio
    async def test_cleanup_removes_workspace(
        self, tmp_path: Path, firejail_on_path,
    ) -> None:
        runtime = FirejailSkillRuntime()
        handle = await runtime.prepare(
            skill=None, workspace=tmp_path / "ws", config={},
        )
        await runtime.cleanup(handle)
        assert not handle.workspace_root.exists()

    @pytest.mark.asyncio
    async def test_cleanup_persisted_keeps_workspace(
        self, tmp_path: Path, firejail_on_path,
    ) -> None:
        runtime = FirejailSkillRuntime()
        handle = await runtime.prepare(
            skill=None, workspace=tmp_path / "ws", config={},
        )
        handle.backend["persisted"] = True
        await runtime.cleanup(handle)
        assert handle.workspace_root.exists()


# ---------------------------------------------------------------------------
# Top-level export sanity
# ---------------------------------------------------------------------------


def test_firejail_runtime_importable_from_top_level() -> None:
    import mmar_carl
    assert hasattr(mmar_carl, "FirejailSkillRuntime")
    assert mmar_carl.FirejailSkillRuntime is FirejailSkillRuntime
