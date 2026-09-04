"""Hermetic checks for the staged-workspace artifact runtime contract."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mmar_carl.docker_skill_runtime import DockerSkillRuntime
from mmar_carl.e2b_skill_runtime import E2BSkillRuntime
from mmar_carl.firejail_skill_runtime import FirejailSkillRuntime
from mmar_carl.skill_runtime import (
    LocalSkillRuntime,
    SkillRuntimeError,
    SkillRuntimeHandle,
    assess_runtime_enforcement,
)


def _handle(workspace: Path, **backend: object) -> SkillRuntimeHandle:
    workspace_in = workspace / "in"
    workspace_out = workspace / "out"
    workspace_in.mkdir(parents=True)
    workspace_out.mkdir()
    return SkillRuntimeHandle(
        workspace_root=workspace,
        workspace_in=workspace_in,
        workspace_out=workspace_out,
        backend=dict(backend),
    )


@pytest.mark.asyncio
async def test_local_prepare_exposes_all_runtime_workspace_coordinates(
    tmp_path: Path,
) -> None:
    LocalSkillRuntime._unsafe_warned = True
    runtime = LocalSkillRuntime()
    handle = await runtime.prepare(None, tmp_path / "workspace", {})

    assert handle.backend["workspace_root_in_runtime"] == str(handle.workspace_root)
    assert handle.backend["workspace_in_in_runtime"] == str(handle.workspace_in)
    assert handle.backend["workspace_out_in_runtime"] == str(handle.workspace_out)


@pytest.mark.asyncio
async def test_docker_prepare_exposes_container_workspace_coordinates(
    tmp_path: Path,
) -> None:
    with patch(
        "mmar_carl.docker_skill_runtime.shutil.which",
        return_value="/usr/bin/docker",
    ):
        handle = await DockerSkillRuntime().prepare(None, tmp_path / "workspace", {})

    assert handle.backend["workspace_root_in_runtime"] == "/workspace"
    assert handle.backend["workspace_in_in_runtime"] == "/workspace/in"
    assert handle.backend["workspace_out_in_runtime"] == "/workspace/out"


@pytest.mark.asyncio
async def test_firejail_uses_private_home_coordinates_not_host_source_path(
    tmp_path: Path,
) -> None:
    with patch(
        "mmar_carl.firejail_skill_runtime.shutil.which",
        side_effect=lambda name, **_kwargs: f"/usr/bin/{name}",
    ):
        runtime = FirejailSkillRuntime()
        handle = await runtime.prepare(None, tmp_path / "workspace", {})

    runtime_root = Path(handle.backend["workspace_root_in_runtime"])
    assert handle.backend["workspace_in_in_runtime"] == str(runtime_root / "in")
    assert handle.backend["workspace_out_in_runtime"] == str(runtime_root / "out")
    assert runtime_root != handle.workspace_root

    command = runtime._build_firejail_cmd(
        handle,
        ["pwd"],
        cwd=str(runtime_root / "out"),
    )
    assert f"--private={handle.workspace_root}" in command
    assert f"--chdir={runtime_root / 'out'}" in command


@pytest.mark.asyncio
async def test_e2b_prepare_exposes_remote_workspace_coordinates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("E2B_API_KEY", "test-key")
    sandbox = MagicMock()
    sandbox.files.make_dir = AsyncMock()
    sandbox.kill = AsyncMock()

    async def factory(**_kwargs: object) -> object:
        return sandbox

    handle = await E2BSkillRuntime(sandbox_factory=factory).prepare(
        None,
        tmp_path / "workspace",
        {},
    )

    assert handle.backend["workspace_root_in_runtime"] == "/workspace"
    assert handle.backend["workspace_in_in_runtime"] == "/workspace/in"
    assert handle.backend["workspace_out_in_runtime"] == "/workspace/out"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "runtime",
    [
        LocalSkillRuntime(),
        DockerSkillRuntime(),
        FirejailSkillRuntime(),
        E2BSkillRuntime(),
    ],
    ids=["local", "docker", "firejail", "e2b"],
)
async def test_cleanup_surfaces_workspace_removal_failure(
    tmp_path: Path,
    runtime: object,
) -> None:
    handle = _handle(tmp_path / type(runtime).__name__)
    with patch(
        "mmar_carl.skill_runtime.shutil.rmtree",
        side_effect=PermissionError("workspace remains"),
    ):
        with pytest.raises(PermissionError, match="workspace remains"):
            await runtime.cleanup(handle)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "runtime",
    [LocalSkillRuntime(), DockerSkillRuntime(), FirejailSkillRuntime()],
    ids=["local", "docker", "firejail"],
)
async def test_host_workspace_backends_read_regular_files_with_hard_limit(
    tmp_path: Path,
    runtime: object,
) -> None:
    handle = _handle(tmp_path / type(runtime).__name__)
    (handle.workspace_out / "result.bin").write_bytes(b"abcd")

    assert (
        await runtime.read_file_bounded(
            handle,
            "out/result.bin",
            max_bytes=4,
            timeout=1.0,
        )
        == b"abcd"
    )
    with pytest.raises(SkillRuntimeError, match="exceeds its 3-byte limit"):
        await runtime.read_file_bounded(
            handle,
            "out/result.bin",
            max_bytes=3,
            timeout=1.0,
        )


@pytest.mark.asyncio
async def test_bounded_read_rejects_directory_symlink_and_fifo(
    tmp_path: Path,
) -> None:
    handle = _handle(tmp_path / "workspace")
    runtime = LocalSkillRuntime()

    with pytest.raises(SkillRuntimeError, match="not a regular file"):
        await runtime.read_file_bounded(handle, "out", max_bytes=10)

    target = handle.workspace_out / "target.txt"
    target.write_bytes(b"secret")
    link = handle.workspace_out / "link.txt"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlinks are unavailable on this platform")
    with pytest.raises(SkillRuntimeError, match="not a regular file"):
        await runtime.read_file_bounded(handle, "out/link.txt", max_bytes=10)

    real_dir = handle.workspace_out / "real"
    real_dir.mkdir()
    (real_dir / "nested.txt").write_bytes(b"nested")
    linked_dir = handle.workspace_out / "linked-dir"
    linked_dir.symlink_to(real_dir, target_is_directory=True)
    with pytest.raises(SkillRuntimeError, match="not a regular file"):
        await runtime.read_file_bounded(
            handle,
            "out/linked-dir/nested.txt",
            max_bytes=10,
        )

    if hasattr(os, "mkfifo"):
        fifo = handle.workspace_out / "pipe"
        os.mkfifo(fifo)
        with pytest.raises(SkillRuntimeError, match="not a regular file"):
            await runtime.read_file_bounded(
                handle,
                "out/pipe",
                max_bytes=10,
                timeout=1.0,
            )


def test_docker_attaches_stdin_only_when_requested(tmp_path: Path) -> None:
    runtime = DockerSkillRuntime()
    handle = _handle(
        tmp_path / "workspace",
        image="alpine:3.20",
        network_policy="none",
    )

    without_stdin = runtime._build_docker_cmd(
        handle,
        ["true"],
        env=None,
        timeout=None,
        cwd=None,
    )
    with_stdin = runtime._build_docker_cmd(
        handle,
        ["cat"],
        env=None,
        timeout=None,
        cwd=None,
        stdin=b"",
    )

    assert "-i" not in without_stdin
    assert "-i" in with_stdin


def _e2b_handle(tmp_path: Path, files: object) -> SkillRuntimeHandle:
    sandbox = MagicMock()
    sandbox.files = files
    return _handle(
        tmp_path / "workspace",
        sandbox=sandbox,
        workspace_root_in_sandbox="/workspace",
    )


class _AsyncChunkReader:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = iter(chunks)
        self.closed = False

    def __aiter__(self) -> _AsyncChunkReader:
        return self

    async def __anext__(self) -> bytes:
        try:
            return next(self._chunks)
        except StopIteration as exc:
            raise StopAsyncIteration from exc

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_e2b_bounded_read_preflights_type_and_size(tmp_path: Path) -> None:
    files = MagicMock()
    files.get_info = AsyncMock(
        return_value=SimpleNamespace(
            type=SimpleNamespace(value="file"),
            size=4,
        )
    )
    files.read = AsyncMock(return_value=bytearray(b"data"))
    handle = _e2b_handle(tmp_path, files)

    data = await E2BSkillRuntime().read_file_bounded(
        handle,
        "out/data.bin",
        max_bytes=4,
        timeout=1.0,
    )

    assert data == b"data"
    files.get_info.assert_awaited_once_with(
        "/workspace/out/data.bin",
        request_timeout=1.0,
    )
    files.read.assert_awaited_once_with(
        "/workspace/out/data.bin",
        format="stream",
        request_timeout=1.0,
        stream_idle_timeout=1.0,
    )


@pytest.mark.asyncio
async def test_e2b_stream_read_is_chunk_bounded_and_closed_on_overflow(
    tmp_path: Path,
) -> None:
    reader = _AsyncChunkReader([b"abcd", b"excess-data"])
    files = MagicMock()
    files.get_info = AsyncMock(
        return_value=SimpleNamespace(
            type=SimpleNamespace(value="file"),
            size=4,
        )
    )
    files.read = AsyncMock(return_value=reader)
    handle = _e2b_handle(tmp_path, files)

    with pytest.raises(SkillRuntimeError, match="exceeds its 4-byte limit"):
        await E2BSkillRuntime().read_file_bounded(
            handle,
            "out/data.bin",
            max_bytes=4,
        )

    assert reader.closed is True


@pytest.mark.asyncio
async def test_e2b_bounded_read_supports_legacy_posthoc_only_shim(
    tmp_path: Path,
) -> None:
    class LegacyFiles:
        def __init__(self) -> None:
            self.calls = 0

        async def read(self, _path: str) -> bytes:
            self.calls += 1
            return b"data"

    files = LegacyFiles()
    handle = _e2b_handle(tmp_path, files)

    assert (
        await E2BSkillRuntime().read_file_bounded(
            handle,
            "out/data.bin",
            max_bytes=4,
        )
        == b"data"
    )
    assert files.calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("file_type", "reported_size", "content", "message"),
    [
        ("symlink", 1, b"x", "not a regular file"),
        ("file", 5, b"large", "exceeds its 4-byte limit"),
        ("file", 1, b"grew!", "exceeds its 4-byte limit"),
    ],
)
async def test_e2b_bounded_read_rejects_unsafe_or_oversized_results(
    tmp_path: Path,
    file_type: str,
    reported_size: int,
    content: bytes,
    message: str,
) -> None:
    files = MagicMock()
    files.get_info = AsyncMock(
        return_value=SimpleNamespace(
            type=SimpleNamespace(value=file_type),
            size=reported_size,
        )
    )
    files.read = AsyncMock(return_value=content)
    handle = _e2b_handle(tmp_path, files)

    with pytest.raises(SkillRuntimeError, match=message):
        await E2BSkillRuntime().read_file_bounded(
            handle,
            "out/data.bin",
            max_bytes=4,
        )

    if file_type != "file" or reported_size > 4:
        files.read.assert_not_awaited()


@pytest.mark.asyncio
async def test_e2b_write_creates_remote_parent_directory(tmp_path: Path) -> None:
    files = MagicMock()
    files.make_dir = AsyncMock()
    files.write = AsyncMock()
    handle = _e2b_handle(tmp_path, files)

    await E2BSkillRuntime().write_file(
        handle,
        "out/nested/result.bin",
        b"data",
    )

    files.make_dir.assert_awaited_once_with("/workspace/out/nested")
    files.write.assert_awaited_once_with(
        "/workspace/out/nested/result.bin",
        b"data",
    )


def test_artifact_output_enforcement_is_reported_only_when_requested() -> None:
    requested = assess_runtime_enforcement(
        LocalSkillRuntime(),
        mode="strict",
        network="host",
        cpu_limit_requested=False,
        memory_limit_requested=False,
        pids_limit_requested=False,
        artifact_outputs_requested=True,
    )
    omitted = assess_runtime_enforcement(
        LocalSkillRuntime(),
        mode="strict",
        network="host",
        cpu_limit_requested=False,
        memory_limit_requested=False,
        pids_limit_requested=False,
    )
    remote = assess_runtime_enforcement(
        E2BSkillRuntime(),
        mode="best_effort",
        network="host",
        cpu_limit_requested=False,
        memory_limit_requested=False,
        pids_limit_requested=False,
        artifact_outputs_requested=True,
    )

    assert requested.controls["artifact_output_limit"] == "enforced"
    assert omitted.controls["artifact_output_limit"] == "not_requested"
    assert remote.controls["artifact_output_limit"] == "advisory"
    assert "artifact_output_limit" in remote.gaps
