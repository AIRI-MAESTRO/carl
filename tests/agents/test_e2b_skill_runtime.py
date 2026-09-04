""":class:`E2BSkillRuntime`.

Remote SaaS sandbox via the e2b.dev micro-VM service. The real ``e2b``
SDK is **not** required to run these tests — the runtime accepts a
``sandbox_factory=...`` injection that swaps in a mock sandbox, so we
verify the contract end-to-end without an e2b account or network.

A separate test asserts that the lazy SDK import path produces a clear
``SkillRuntimeError`` when the SDK is missing, so users without
``mmar-carl[e2b]`` installed get a helpful message instead of a cryptic
``ImportError``.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mmar_carl import (
    E2BSkillRuntime,
    SkillRuntimeError,
    SkillRuntimeHandle,
    get_skill_runtime,
    list_skill_runtimes,
)

# ---------------------------------------------------------------------------
# Mock sandbox harness — duck-types the e2b SDK surface the runtime uses.
# ---------------------------------------------------------------------------


def _make_command_result(
    *, stdout: bytes | str = b"ok", stderr: bytes | str = b"",
    exit_code: int = 0,
) -> MagicMock:
    r = MagicMock()
    r.stdout = stdout
    r.stderr = stderr
    r.exit_code = exit_code
    return r


def _make_mock_sandbox(
    *,
    run_result: Any = None,
    run_raises: Exception | None = None,
    wait_raises: Exception | None = None,
    files_read_returns: bytes = b"",
    files_write_raises: Exception | None = None,
) -> MagicMock:
    sandbox = MagicMock()

    command_handle = MagicMock()
    command_handle.pid = 123
    command_handle.send_stdin = AsyncMock()
    command_handle.close_stdin = AsyncMock()
    command_handle.kill = AsyncMock()
    if wait_raises is not None:
        command_handle.wait = AsyncMock(side_effect=wait_raises)
    else:
        command_handle.wait = AsyncMock(
            return_value=run_result or _make_command_result(),
        )

    # commands.run -> current SDK background command handle
    if run_raises is not None:
        sandbox.commands.run = AsyncMock(side_effect=run_raises)
    else:
        sandbox.commands.run = AsyncMock(return_value=command_handle)
    sandbox.command_handle = command_handle

    # files.read / files.write -> AsyncMock
    sandbox.files.read = AsyncMock(return_value=files_read_returns)
    sandbox.files.make_dir = AsyncMock(return_value=True)
    if files_write_raises is not None:
        sandbox.files.write = AsyncMock(side_effect=files_write_raises)
    else:
        sandbox.files.write = AsyncMock()
    sandbox.files.remove = AsyncMock()

    sandbox.kill = AsyncMock()
    return sandbox


def _factory_for(sandbox: Any):
    """Return a sandbox_factory that yields the given sandbox.

    The runtime calls ``await factory(template=..., api_key=..., **kw)`` —
    we ignore the kwargs and return a pre-built mock.
    """
    async def _factory(**_kwargs: Any) -> Any:
        return sandbox
    return _factory


def _staged_exec_request(sandbox: Any) -> dict[str, Any]:
    request_calls = [
        call
        for call in sandbox.files.write.await_args_list
        if str(call.args[0]).endswith(".json")
    ]
    assert len(request_calls) == 1
    return json.loads(request_calls[0].args[1])


@pytest.fixture
def api_key_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("E2B_API_KEY", "test-key-1234")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_e2b_runtime_pre_registered(self) -> None:
        assert "e2b" in list_skill_runtimes()
        instance = get_skill_runtime("e2b")
        assert isinstance(instance, E2BSkillRuntime)

    def test_name_attr(self) -> None:
        assert E2BSkillRuntime.name == "e2b"

    def test_remote_timeout_is_reported_as_advisory_until_live_attested(self) -> None:
        assert E2BSkillRuntime.capabilities.wall_time == "advisory"


# ---------------------------------------------------------------------------
# prepare()
# ---------------------------------------------------------------------------


class TestPrepare:
    @pytest.mark.asyncio
    async def test_repeated_cancellation_during_create_kills_late_sandbox(
        self,
        tmp_path: Path,
        api_key_env: None,
    ) -> None:
        sandbox = _make_mock_sandbox()
        create_started = asyncio.Event()
        allow_create = asyncio.Event()

        async def blocking_factory(**_kwargs: Any) -> MagicMock:
            create_started.set()
            await allow_create.wait()
            return sandbox

        runtime = E2BSkillRuntime(sandbox_factory=blocking_factory)
        task = asyncio.create_task(runtime.prepare(None, tmp_path / "ws", {}))
        try:
            await asyncio.wait_for(create_started.wait(), timeout=1.0)
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
            allow_create.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=1.0)
        finally:
            allow_create.set()
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

        sandbox.kill.assert_awaited_once()
        assert not (tmp_path / "ws").exists()

    @pytest.mark.asyncio
    async def test_cancellation_after_create_kills_remote_sandbox(
        self,
        tmp_path: Path,
        api_key_env: None,
    ) -> None:
        sandbox = _make_mock_sandbox()
        entered_make_dir = asyncio.Event()
        release_make_dir = asyncio.Event()

        async def block_make_dir(_path: str) -> None:
            entered_make_dir.set()
            await release_make_dir.wait()

        sandbox.files.make_dir = AsyncMock(side_effect=block_make_dir)
        runtime = E2BSkillRuntime(sandbox_factory=_factory_for(sandbox))
        task = asyncio.create_task(
            runtime.prepare(None, tmp_path / "ws", {}),
        )
        await entered_make_dir.wait()
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task
        sandbox.kill.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_repeated_cancellation_after_create_finishes_remote_kill(
        self,
        tmp_path: Path,
        api_key_env: None,
    ) -> None:
        sandbox = _make_mock_sandbox()
        entered_make_dir = asyncio.Event()
        release_make_dir = asyncio.Event()
        kill_started = asyncio.Event()
        allow_kill = asyncio.Event()
        kill_finished = asyncio.Event()

        async def block_make_dir(_path: str) -> None:
            entered_make_dir.set()
            await release_make_dir.wait()

        async def block_kill() -> None:
            kill_started.set()
            await allow_kill.wait()
            kill_finished.set()

        sandbox.files.make_dir = AsyncMock(side_effect=block_make_dir)
        sandbox.kill = AsyncMock(side_effect=block_kill)
        runtime = E2BSkillRuntime(sandbox_factory=_factory_for(sandbox))
        task = asyncio.create_task(runtime.prepare(None, tmp_path / "ws", {}))
        try:
            await asyncio.wait_for(entered_make_dir.wait(), timeout=1.0)
            task.cancel()
            await asyncio.wait_for(kill_started.wait(), timeout=1.0)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
            allow_kill.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=1.0)
        finally:
            release_make_dir.set()
            allow_kill.set()
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

        assert kill_finished.is_set()
        sandbox.kill.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_missing_api_key_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("E2B_API_KEY", raising=False)
        runtime = E2BSkillRuntime(
            sandbox_factory=_factory_for(_make_mock_sandbox()),
        )
        with pytest.raises(SkillRuntimeError, match="E2B_API_KEY"):
            await runtime.prepare(
                skill=None, workspace=tmp_path / "ws", config={},
            )

    @pytest.mark.asyncio
    async def test_custom_api_key_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``api_key_env`` lets callers point at a custom env var."""
        monkeypatch.setenv("CARE_E2B_KEY", "custom-1234")
        runtime = E2BSkillRuntime(
            sandbox_factory=_factory_for(_make_mock_sandbox()),
        )
        handle = await runtime.prepare(
            skill=None, workspace=tmp_path / "ws",
            config={"api_key_env": "CARE_E2B_KEY"},
        )
        assert handle.backend["api_key_env"] == "CARE_E2B_KEY"

    @pytest.mark.asyncio
    async def test_defaults_stamped_on_handle(
        self, tmp_path: Path, api_key_env: None,
    ) -> None:
        sandbox = _make_mock_sandbox()
        runtime = E2BSkillRuntime(sandbox_factory=_factory_for(sandbox))
        handle = await runtime.prepare(
            skill=None, workspace=tmp_path / "ws", config={},
        )
        assert handle.backend["isolation"] == "e2b"
        assert handle.backend["template"] == "base"
        assert handle.backend["api_key_env"] == "E2B_API_KEY"
        assert handle.backend["workspace_root_in_sandbox"] == "/workspace"
        assert handle.backend["network_policy"] == "none"
        assert handle.backend["network_enforced"] is True
        assert handle.backend["sandbox"] is sandbox
        assert handle.workspace_in.is_dir()
        assert handle.workspace_out.is_dir()
        assert [call.args[0] for call in sandbox.files.make_dir.await_args_list] == [
            "/workspace",
            "/workspace/in",
            "/workspace/out",
        ]

    @pytest.mark.asyncio
    async def test_custom_template(
        self, tmp_path: Path, api_key_env: None,
    ) -> None:
        sandbox = _make_mock_sandbox()
        runtime = E2BSkillRuntime(sandbox_factory=_factory_for(sandbox))
        handle = await runtime.prepare(
            skill=None, workspace=tmp_path / "ws",
            config={"template": "python-data-science"},
        )
        assert handle.backend["template"] == "python-data-science"

    @pytest.mark.asyncio
    async def test_extra_create_kwargs_forwarded(
        self, tmp_path: Path, api_key_env: None,
    ) -> None:
        """``extra_create_kwargs`` is splatted into the factory call."""
        captured_kwargs: dict[str, Any] = {}

        async def capturing_factory(**kwargs: Any) -> MagicMock:
            captured_kwargs.update(kwargs)
            return _make_mock_sandbox()

        runtime = E2BSkillRuntime(sandbox_factory=capturing_factory)
        await runtime.prepare(
            skill=None, workspace=tmp_path / "ws",
            config={
                "template": "base",
                "extra_create_kwargs": {"metadata": {"task": "x"}},
            },
        )
        assert captured_kwargs["allow_internet_access"] is False
        assert captured_kwargs["secure"] is True
        assert captured_kwargs["lifecycle"] == {"on_timeout": "kill"}
        assert captured_kwargs["timeout"] == 300
        assert captured_kwargs["request_timeout"] == 30.0
        assert captured_kwargs["template"] == "base"
        assert captured_kwargs["api_key"] == "test-key-1234"
        assert captured_kwargs["metadata"] == {"task": "x"}

    @pytest.mark.asyncio
    async def test_native_allowlist_is_forwarded_to_provider(
        self, tmp_path: Path, api_key_env: None,
    ) -> None:
        captured_kwargs: dict[str, Any] = {}

        async def capturing_factory(**kwargs: Any) -> MagicMock:
            captured_kwargs.update(kwargs)
            return _make_mock_sandbox()

        runtime = E2BSkillRuntime(sandbox_factory=capturing_factory)
        handle = await runtime.prepare(
            None,
            tmp_path / "ws",
            {
                "network": "allowlist",
                "network_allowlist": ["api.example.com"],
            },
        )

        assert "allow_internet_access" not in captured_kwargs
        assert captured_kwargs["network"] == {
            "allow_out": ["api.example.com"],
            "allow_public_traffic": False,
        }
        assert handle.backend["network_enforced"] is True

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "key",
        ["network", "allow_internet_access", "secure", "lifecycle", "volume_mounts", "mcp"],
    )
    async def test_extra_create_kwargs_cannot_override_network_controls(
        self, tmp_path: Path, api_key_env: None, key: str,
    ) -> None:
        runtime = E2BSkillRuntime(sandbox_factory=_factory_for(_make_mock_sandbox()))

        with pytest.raises(SkillRuntimeError, match="may contain only"):
            await runtime.prepare(
                None,
                tmp_path / "ws",
                {"extra_create_kwargs": {key: True}},
            )

    @pytest.mark.asyncio
    async def test_sandbox_lifetime_is_host_bounded(
        self, tmp_path: Path, api_key_env: None,
    ) -> None:
        runtime = E2BSkillRuntime(sandbox_factory=_factory_for(_make_mock_sandbox()))
        with pytest.raises(SkillRuntimeError, match="from 1 to 300"):
            await runtime.prepare(
                None,
                tmp_path / "ws",
                {"extra_create_kwargs": {"timeout": 301}},
            )

    @pytest.mark.asyncio
    async def test_factory_failure_wrapped(
        self, tmp_path: Path, api_key_env: None,
    ) -> None:
        async def failing_factory(**kwargs: Any) -> Any:
            raise RuntimeError("network unreachable")

        runtime = E2BSkillRuntime(sandbox_factory=failing_factory)
        with pytest.raises(SkillRuntimeError, match="failed to start sandbox"):
            await runtime.prepare(
                skill=None, workspace=tmp_path / "ws", config={},
            )

    @pytest.mark.asyncio
    async def test_missing_sdk_raises_when_no_factory(
        self, tmp_path: Path, api_key_env: None,
    ) -> None:
        """Without an injected factory, the runtime tries to import the
        real e2b SDK. When it's missing we surface a clear message."""
        runtime = E2BSkillRuntime()  # no factory → uses default lazy import
        # Patch the SDK import to raise ImportError.
        with patch.dict("sys.modules", {"e2b": None}):
            with pytest.raises(SkillRuntimeError) as exc_info:
                await runtime.prepare(
                    skill=None, workspace=tmp_path / "ws", config={},
                )
        msg = str(exc_info.value)
        # The error chain: outer "failed to start sandbox" wraps the
        # inner "requires the e2b SDK" message. Surface either one.
        assert "e2b" in msg.lower()

    @pytest.mark.asyncio
    async def test_unknown_network_policy_raises(
        self, tmp_path: Path, api_key_env: None,
    ) -> None:
        runtime = E2BSkillRuntime(
            sandbox_factory=_factory_for(_make_mock_sandbox()),
        )
        with pytest.raises(SkillRuntimeError, match="Unknown network policy"):
            await runtime.prepare(
                skill=None, workspace=tmp_path / "ws",
                config={"network": "wide-open"},
            )

    @pytest.mark.asyncio
    async def test_workspace_root_must_be_absolute(
        self, tmp_path: Path, api_key_env: None,
    ) -> None:
        runtime = E2BSkillRuntime(
            sandbox_factory=_factory_for(_make_mock_sandbox()),
        )
        with pytest.raises(SkillRuntimeError, match="absolute sandbox path"):
            await runtime.prepare(
                skill=None,
                workspace=tmp_path / "ws",
                config={"workspace_root_in_sandbox": "relative"},
            )

    @pytest.mark.asyncio
    async def test_workspace_setup_failure_kills_sandbox(
        self, tmp_path: Path, api_key_env: None,
    ) -> None:
        sandbox = _make_mock_sandbox()
        sandbox.files.make_dir = AsyncMock(side_effect=RuntimeError("disk full"))
        runtime = E2BSkillRuntime(sandbox_factory=_factory_for(sandbox))

        with pytest.raises(SkillRuntimeError, match="failed to initialize workspace"):
            await runtime.prepare(
                skill=None, workspace=tmp_path / "ws", config={},
            )

        sandbox.kill.assert_awaited_once()
        assert not (tmp_path / "ws").exists()


# ---------------------------------------------------------------------------
# run() — happy path + error mapping
# ---------------------------------------------------------------------------


class TestRun:
    @pytest.mark.asyncio
    async def test_run_captures_stdout(
        self, tmp_path: Path, api_key_env: None,
    ) -> None:
        sandbox = _make_mock_sandbox(
            run_result=_make_command_result(
                stdout=b"hello\n", exit_code=0,
            ),
        )
        runtime = E2BSkillRuntime(sandbox_factory=_factory_for(sandbox))
        handle = await runtime.prepare(
            skill=None, workspace=tmp_path / "ws", config={},
        )
        argv = ["echo", "$(touch /tmp/not-executed)", "x; whoami"]
        result = await runtime.run(handle, argv)
        assert result.exit_code == 0
        assert result.stdout == b"hello\n"
        # The SDK still receives a string, but it contains only CARL's fixed
        # launcher and random internal paths. User argv lives in JSON and is
        # consumed by os.execvpe, never by E2B's implicit login shell.
        sandbox.commands.run.assert_awaited_once()
        cmd_arg = sandbox.commands.run.await_args[0][0]
        assert cmd_arg.startswith("/usr/bin/python3 /workspace/.carl-exec-")
        assert "touch" not in cmd_arg
        assert "whoami" not in cmd_arg
        assert _staged_exec_request(sandbox)["argv"] == argv
        helper_calls = [
            call
            for call in sandbox.files.write.await_args_list
            if str(call.args[0]).endswith(".py")
        ]
        assert len(helper_calls) == 1
        assert b"os.execvpe" in helper_calls[0].args[1]
        removed = {call.args[0] for call in sandbox.files.remove.await_args_list}
        assert len(removed) == 2
        assert any(path.endswith(".json") for path in removed)
        assert any(path.endswith(".py") for path in removed)

    @pytest.mark.asyncio
    async def test_run_handles_str_stdout(
        self, tmp_path: Path, api_key_env: None,
    ) -> None:
        """Some e2b SDK versions return str rather than bytes — coerce."""
        sandbox = _make_mock_sandbox(
            run_result=_make_command_result(stdout="text out", stderr="err"),
        )
        runtime = E2BSkillRuntime(sandbox_factory=_factory_for(sandbox))
        handle = await runtime.prepare(
            skill=None, workspace=tmp_path / "ws", config={},
        )
        result = await runtime.run(handle, ["echo"])
        assert result.stdout == b"text out"
        assert result.stderr == b"err"

    @pytest.mark.asyncio
    async def test_run_propagates_env_and_cwd(
        self, tmp_path: Path, api_key_env: None,
    ) -> None:
        sandbox = _make_mock_sandbox()
        runtime = E2BSkillRuntime(sandbox_factory=_factory_for(sandbox))
        handle = await runtime.prepare(
            skill=None, workspace=tmp_path / "ws", config={},
        )
        await runtime.run(
            handle, ["pwd"], env={"FOO": "bar"}, cwd="/work",
        )
        kwargs = sandbox.commands.run.await_args.kwargs
        assert kwargs["envs"] == {
            "PATH": "/usr/bin:/bin",
            "LANG": "C",
            "LC_ALL": "C",
        }
        assert _staged_exec_request(sandbox)["env"] == {"FOO": "bar"}
        assert kwargs["cwd"] == "/work"
        assert kwargs["background"] is True
        assert kwargs["stdin"] is False

    @pytest.mark.asyncio
    async def test_run_forwards_stdin_and_closes_it(
        self, tmp_path: Path, api_key_env: None,
    ) -> None:
        sandbox = _make_mock_sandbox(
            run_result=_make_command_result(stdout=b"input"),
        )
        runtime = E2BSkillRuntime(sandbox_factory=_factory_for(sandbox))
        handle = await runtime.prepare(
            skill=None, workspace=tmp_path / "ws", config={},
        )
        result = await runtime.run(handle, ["cat"], stdin=b"input")
        assert result.stdout == b"input"
        sandbox.command_handle.send_stdin.assert_awaited_once_with(b"input")
        sandbox.command_handle.close_stdin.assert_awaited_once()
        assert sandbox.commands.run.await_args.kwargs["stdin"] is True

    @pytest.mark.asyncio
    async def test_run_timeout_returns_exit_124(
        self, tmp_path: Path, api_key_env: None,
    ) -> None:
        import asyncio
        sandbox = _make_mock_sandbox(run_raises=asyncio.TimeoutError())
        runtime = E2BSkillRuntime(sandbox_factory=_factory_for(sandbox))
        handle = await runtime.prepare(
            skill=None, workspace=tmp_path / "ws", config={},
        )
        result = await runtime.run(handle, ["sleep", "100"], timeout=1.0)
        assert result.exit_code == 124
        assert b"[timeout after" in result.stderr
        assert sandbox.files.remove.await_count == 2

    @pytest.mark.asyncio
    async def test_run_sandbox_error_wrapped(
        self, tmp_path: Path, api_key_env: None,
    ) -> None:
        sandbox = _make_mock_sandbox(run_raises=RuntimeError("sandbox died"))
        runtime = E2BSkillRuntime(sandbox_factory=_factory_for(sandbox))
        handle = await runtime.prepare(
            skill=None, workspace=tmp_path / "ws", config={},
        )
        with pytest.raises(SkillRuntimeError, match="sandbox command failed"):
            await runtime.run(handle, ["echo"])
        assert sandbox.files.remove.await_count == 2

    @pytest.mark.asyncio
    async def test_cancellation_removes_staged_invocation_files(
        self,
        tmp_path: Path,
        api_key_env: None,
    ) -> None:
        sandbox = _make_mock_sandbox()
        entered_wait = asyncio.Event()

        async def wait_forever():
            entered_wait.set()
            await asyncio.Event().wait()

        sandbox.command_handle.wait = AsyncMock(side_effect=wait_forever)
        runtime = E2BSkillRuntime(sandbox_factory=_factory_for(sandbox))
        handle = await runtime.prepare(
            skill=None,
            workspace=tmp_path / "ws",
            config={},
        )
        task = asyncio.create_task(runtime.run(handle, ["sleep", "100"]))
        await entered_wait.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        sandbox.command_handle.kill.assert_awaited_once()
        assert sandbox.files.remove.await_count == 2

    @pytest.mark.asyncio
    async def test_nonzero_command_is_not_a_runtime_failure(
        self, tmp_path: Path, api_key_env: None,
    ) -> None:
        class CommandExit(Exception):
            exit_code = 7
            stdout = "partial"
            stderr = "failed"

        sandbox = _make_mock_sandbox(wait_raises=CommandExit("exit 7"))
        runtime = E2BSkillRuntime(sandbox_factory=_factory_for(sandbox))
        handle = await runtime.prepare(
            skill=None, workspace=tmp_path / "ws", config={},
        )
        result = await runtime.run(handle, ["false"])
        assert result.exit_code == 7
        assert result.stdout == b"partial"
        assert result.stderr == b"failed"

    @pytest.mark.asyncio
    async def test_run_without_sandbox_raises(self, tmp_path: Path) -> None:
        """If the handle is malformed (no live sandbox), surface
        a clear error instead of crashing inside the SDK."""
        runtime = E2BSkillRuntime()
        bogus_handle = SkillRuntimeHandle(
            workspace_root=tmp_path, workspace_in=tmp_path,
            workspace_out=tmp_path, backend={},
        )
        with pytest.raises(SkillRuntimeError, match="no live sandbox"):
            await runtime.run(bogus_handle, ["echo"])


# ---------------------------------------------------------------------------
# File ops — translated to absolute sandbox paths
# ---------------------------------------------------------------------------


class TestFileOps:
    @pytest.mark.asyncio
    async def test_write_uses_sandbox_path(
        self, tmp_path: Path, api_key_env: None,
    ) -> None:
        sandbox = _make_mock_sandbox()
        runtime = E2BSkillRuntime(sandbox_factory=_factory_for(sandbox))
        handle = await runtime.prepare(
            skill=None, workspace=tmp_path / "ws", config={},
        )
        await runtime.write_file(handle, "out/result.txt", b"hi")
        # files.write was called with /workspace/out/result.txt.
        args = sandbox.files.write.await_args.args
        assert args[0] == "/workspace/out/result.txt"
        assert args[1] == b"hi"

    @pytest.mark.asyncio
    async def test_read_returns_bytes(
        self, tmp_path: Path, api_key_env: None,
    ) -> None:
        sandbox = _make_mock_sandbox(files_read_returns=b"contents")
        runtime = E2BSkillRuntime(sandbox_factory=_factory_for(sandbox))
        handle = await runtime.prepare(
            skill=None, workspace=tmp_path / "ws", config={},
        )
        data = await runtime.read_file(handle, "in/data.txt")
        assert data == b"contents"
        # Path translated to /workspace/in/data.txt.
        assert sandbox.files.read.await_args.args[0] == "/workspace/in/data.txt"

    @pytest.mark.asyncio
    async def test_read_coerces_str_to_bytes(
        self, tmp_path: Path, api_key_env: None,
    ) -> None:
        sandbox = _make_mock_sandbox(files_read_returns="string content")
        runtime = E2BSkillRuntime(sandbox_factory=_factory_for(sandbox))
        handle = await runtime.prepare(
            skill=None, workspace=tmp_path / "ws", config={},
        )
        data = await runtime.read_file(handle, "in/data.txt")
        assert data == b"string content"

    @pytest.mark.asyncio
    async def test_custom_workspace_root(
        self, tmp_path: Path, api_key_env: None,
    ) -> None:
        sandbox = _make_mock_sandbox()
        runtime = E2BSkillRuntime(sandbox_factory=_factory_for(sandbox))
        handle = await runtime.prepare(
            skill=None, workspace=tmp_path / "ws",
            config={"workspace_root_in_sandbox": "/data"},
        )
        await runtime.write_file(handle, "out/x.txt", b"x")
        assert sandbox.files.write.await_args.args[0] == "/data/out/x.txt"

    @pytest.mark.asyncio
    async def test_absolute_path_rejected(
        self, tmp_path: Path, api_key_env: None,
    ) -> None:
        sandbox = _make_mock_sandbox()
        runtime = E2BSkillRuntime(sandbox_factory=_factory_for(sandbox))
        handle = await runtime.prepare(
            skill=None, workspace=tmp_path / "ws", config={},
        )
        with pytest.raises(SkillRuntimeError, match="must be relative"):
            await runtime.write_file(handle, "/etc/passwd", b"x")

    @pytest.mark.asyncio
    async def test_dotdot_escape_rejected(
        self, tmp_path: Path, api_key_env: None,
    ) -> None:
        sandbox = _make_mock_sandbox()
        runtime = E2BSkillRuntime(sandbox_factory=_factory_for(sandbox))
        handle = await runtime.prepare(
            skill=None, workspace=tmp_path / "ws", config={},
        )
        with pytest.raises(SkillRuntimeError, match="escapes workspace"):
            await runtime.write_file(handle, "../escape.txt", b"x")

    @pytest.mark.asyncio
    async def test_write_error_wrapped(
        self, tmp_path: Path, api_key_env: None,
    ) -> None:
        sandbox = _make_mock_sandbox(
            files_write_raises=RuntimeError("quota exceeded"),
        )
        runtime = E2BSkillRuntime(sandbox_factory=_factory_for(sandbox))
        handle = await runtime.prepare(
            skill=None, workspace=tmp_path / "ws", config={},
        )
        with pytest.raises(SkillRuntimeError, match="write_file"):
            await runtime.write_file(handle, "out/x.txt", b"x")


# ---------------------------------------------------------------------------
# cleanup
# ---------------------------------------------------------------------------


class TestCleanup:
    @pytest.mark.asyncio
    async def test_cleanup_kills_sandbox(
        self, tmp_path: Path, api_key_env: None,
    ) -> None:
        sandbox = _make_mock_sandbox()
        runtime = E2BSkillRuntime(sandbox_factory=_factory_for(sandbox))
        handle = await runtime.prepare(
            skill=None, workspace=tmp_path / "ws", config={},
        )
        await runtime.cleanup(handle)
        sandbox.kill.assert_awaited_once()
        # Local workspace dir removed.
        assert not handle.workspace_root.exists()
        # Sandbox cleared from handle.
        assert handle.backend["sandbox"] is None

    @pytest.mark.asyncio
    async def test_cleanup_persisted_keeps_everything(
        self, tmp_path: Path, api_key_env: None,
    ) -> None:
        sandbox = _make_mock_sandbox()
        runtime = E2BSkillRuntime(sandbox_factory=_factory_for(sandbox))
        handle = await runtime.prepare(
            skill=None, workspace=tmp_path / "ws", config={},
        )
        handle.backend["persisted"] = True
        await runtime.cleanup(handle)
        sandbox.kill.assert_not_awaited()
        assert handle.workspace_root.exists()
        assert handle.backend["sandbox"] is sandbox

    @pytest.mark.asyncio
    async def test_cleanup_surfaces_kill_errors_after_local_cleanup(
        self, tmp_path: Path, api_key_env: None,
    ) -> None:
        sandbox = _make_mock_sandbox()
        sandbox.kill = AsyncMock(side_effect=RuntimeError("kill failed"))
        runtime = E2BSkillRuntime(sandbox_factory=_factory_for(sandbox))
        handle = await runtime.prepare(
            skill=None, workspace=tmp_path / "ws", config={},
        )
        with pytest.raises(SkillRuntimeError, match=r"sandbox\.kill\(\) failed"):
            await runtime.cleanup(handle)
        assert not handle.workspace_root.exists()
        # Preserve the live reference so a later cleanup retry can kill it.
        assert handle.backend["sandbox"] is sandbox


# ---------------------------------------------------------------------------
# Top-level export sanity
# ---------------------------------------------------------------------------


def test_e2b_runtime_importable_from_top_level() -> None:
    import mmar_carl
    assert hasattr(mmar_carl, "E2BSkillRuntime")
    assert mmar_carl.E2BSkillRuntime is E2BSkillRuntime


def test_command_result_coercion_handles_missing_exit_code() -> None:
    """An SDK result without an exit_code defaults to 0 — matches the
    other backends' convention for the happy path."""
    result_obj = MagicMock(spec=["stdout", "stderr"])
    result_obj.stdout = b"x"
    result_obj.stderr = b""
    coerced = E2BSkillRuntime._coerce_result(result_obj, duration_s=0.1)
    assert coerced.exit_code == 0
    assert coerced.stdout == b"x"
    assert coerced.duration_s == 0.1
