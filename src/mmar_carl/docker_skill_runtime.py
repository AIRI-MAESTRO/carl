"""DockerSkillRuntime — sandbox AgentSkill execution inside a Docker container.

Default profile: ``python:3.12-slim`` image, ``--network none``,
``--cpus=1``, ``--memory=2g``, ``--pids-limit=128``, ``--read-only`` rootfs,
all capabilities dropped, no-new-privileges, and a bounded hardened ``/tmp``
tmpfs. On POSIX hosts the container process
runs as the host UID:GID so nested bind-mounted outputs remain removable.
The host workspace is mounted so
``read_file`` / ``write_file`` use the existing :class:`LocalSkillRuntime`
path — they don't go through Docker exec.

Every invocation also sets ``--pull=never``. The host must provision the
image before execution, so an approved command cannot make the Docker daemon
contact a registry as a control-plane side effect.

Implementation note
-------------------
We shell out to the ``docker`` CLI rather than depending on the
``docker`` Python SDK. Reasons: (a) no new optional dependency to declare,
(b) the CLI is what users actually have installed, and (c) the call surface
is small enough that we don't need the SDK's higher-level container objects.
Hermetic tests stub ``asyncio.create_subprocess_exec`` so they don't need a
live daemon.

The runtime self-registers as ``"docker"`` in the shared
:data:`~mmar_carl.skill_runtime.SKILL_RUNTIME_REGISTRY` on module import, so
``AgentSkillStepConfig(runtime="docker")`` works without any extra wiring.

``runtime_config`` recognised keys
----------------------------------

* ``image`` — Docker image reference. Default ``"python:3.12-slim"``.
* ``network`` — ``"none"`` / ``"allowlist"`` / ``"host"``.
  Hostname allowlists require a managed egress provider. Until one is
  supplied, this backend reports them unsupported and falls back to
  ``--network none`` rather than confusing a hostname with a network id.
* ``mem_limit`` — Docker ``--memory`` value. Default ``"2g"``.
* ``cpu_limit`` — Docker ``--cpus`` value. Default ``"1"``.
* ``pids_limit`` — Docker ``--pids-limit`` value. Default ``128``.
* ``workspace_output_mode`` — ``"read_write"`` (default) or ``"read_only"``.
  CodeStep uses the latter so generated code cannot grow a host output bind mount.
Arbitrary ``docker run`` flags are intentionally rejected. Runtime profiles
that need extra mounts/capabilities must be supplied by trusted host code; a
serializable chain cannot weaken the fixed sandbox flags.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import re
import shutil
import tempfile
import uuid
import warnings
from pathlib import Path
from typing import Any, ClassVar, Optional

from .command_policy import normalize_memory_limit, parse_memory_limit_bytes
from .skill_runtime import (
    RuntimeCapabilities,
    RuntimeRunResult,
    SkillRuntimeError,
    SkillRuntimeHandle,
    _communicate_bounded,
    _kill_process_group,
    _read_file_bounded_async,
    _remove_workspace_tree,
    register_skill_runtime,
    resolve_network_policy,
)

_log = logging.getLogger(__name__)


_DEFAULT_IMAGE = "python:3.12-slim"
_DEFAULT_MEM_LIMIT = "2g"
_DEFAULT_CPU_LIMIT = "1"
_DEFAULT_PIDS_LIMIT = 128
_DEFAULT_TMPFS_SIZE = "64m"
_DOCKER_CONTROL_TIMEOUT_S = 2.0
_DOCKER_CONTROL_OUTPUT_BYTES = 4_096
_SAFE_IMAGE_RE = re.compile(r"[^\s\x00]+\Z")


def _validate_image_reference(value: Any) -> str:
    """Reject values Docker could reinterpret as later CLI options."""

    if (
        not isinstance(value, str)
        or not value
        or value.startswith("-")
        or _SAFE_IMAGE_RE.fullmatch(value) is None
    ):
        raise SkillRuntimeError(
            "invalid Docker image reference: expected one non-option token"
        )
    return value


class DockerSkillRuntime:
    """Run skill scripts inside a fresh Docker container per call.

    Containers are ephemeral (``--rm``). The host ``in/`` and ``out/``
    workspace directories are mounted at ``/workspace/in`` and
    ``/workspace/out``; ``in/`` is read-only while ``out/`` is read-write by
    default and may be host-selected read-only.
    The container root is
    read-only with an ephemeral tmpfs at ``/tmp`` so a misbehaving
    skill can't leave files on the host disk.

    Network access is governed by :func:`resolve_network_policy`. ``none``
    and ``host`` are implemented directly; hostname allowlists fail closed
    until a managed egress provider is configured.
    """

    name: ClassVar[str] = "docker"
    capabilities: ClassVar[RuntimeCapabilities] = RuntimeCapabilities(
        isolation="container",
        wall_time="enforced",
        output_limit="enforced",
        cpu_limit="enforced",
        memory_limit="enforced",
        pids_limit="enforced",
        network_none="enforced",
        # A hostname list is not a Docker network. Managed egress lands in a
        # later layer; until then allowlist must fail strict preflight.
        network_allowlist="unsupported",
        workspace_files="enforced",
        artifact_output_limit="enforced",
    )

    async def prepare(
        self,
        skill: Any,
        workspace: Optional[Path],
        config: dict[str, Any],
    ) -> SkillRuntimeHandle:
        # Up-front check: docker CLI must be on PATH. Surface a clear
        # `SkillRuntimeError` rather than failing inside `run` later.
        discovered_docker = shutil.which("docker")
        if discovered_docker is None:
            raise SkillRuntimeError(
                "DockerSkillRuntime requires the `docker` CLI on PATH. "
                "Install Docker Desktop / Docker Engine, or pick a "
                "different runtime via AgentSkillStepConfig(runtime=...)."
            )

        manifest_allowed = config.get("_manifest_allowed_tools")
        policy, allowlist = resolve_network_policy(
            config, manifest_allowed_tools=manifest_allowed,
        )
        from .network_enforcement import DockerNetworkBinding  # noqa: PLC0415

        network_binding = config.get("_network_binding")
        if network_binding is not None and (
            policy != "allowlist"
            or not isinstance(network_binding, DockerNetworkBinding)
        ):
            raise SkillRuntimeError(
                "Docker managed egress requires a host-issued "
                "DockerNetworkBinding with network='allowlist'"
            )
        if policy == "host":
            warnings.warn(
                "DockerSkillRuntime selected network='host' — unrestricted "
                "container egress is enabled.",
                UserWarning,
                stacklevel=3,
            )

        if config.get("extra_args"):
            raise SkillRuntimeError(
                "DockerSkillRuntime does not accept chain-controlled extra_args; "
                "use a host-owned runtime profile instead"
            )
        workspace_output_mode = config.get("workspace_output_mode", "read_write")
        if workspace_output_mode not in ("read_write", "read_only"):
            raise SkillRuntimeError(
                "DockerSkillRuntime workspace_output_mode must be "
                "'read_write' or 'read_only'"
            )
        image = _validate_image_reference(config.get("image", _DEFAULT_IMAGE))

        raw_mem_limit = (
            _DEFAULT_MEM_LIMIT
            if config.get("mem_limit") is None
            else config["mem_limit"]
        )
        try:
            mem_limit = normalize_memory_limit(str(raw_mem_limit))
        except (TypeError, ValueError) as exc:
            raise SkillRuntimeError(
                f"invalid Docker memory limit {raw_mem_limit!r}: {exc}"
            ) from exc
        if parse_memory_limit_bytes(mem_limit) < 6 * 1024**2:
            raise SkillRuntimeError(
                "invalid Docker memory limit: Docker requires at least 6 MiB"
            )

        raw_cpu_limit = (
            _DEFAULT_CPU_LIMIT
            if config.get("cpu_limit") is None
            else config["cpu_limit"]
        )
        try:
            if isinstance(raw_cpu_limit, bool):
                raise ValueError("boolean is not a CPU limit")
            parsed_cpu_limit = float(raw_cpu_limit)
            if not math.isfinite(parsed_cpu_limit) or parsed_cpu_limit <= 0:
                raise ValueError("CPU limit must be finite and positive")
        except (TypeError, ValueError) as exc:
            raise SkillRuntimeError(
                f"invalid Docker CPU limit {raw_cpu_limit!r}: {exc}"
            ) from exc

        raw_pids_limit = (
            _DEFAULT_PIDS_LIMIT
            if config.get("pids_limit") is None
            else config["pids_limit"]
        )
        try:
            if isinstance(raw_pids_limit, bool) or not isinstance(raw_pids_limit, int):
                raise TypeError("PID limit must be an integer")
            pids_limit = raw_pids_limit
            if pids_limit <= 0:
                raise ValueError("PID limit must be positive")
        except (TypeError, ValueError) as exc:
            raise SkillRuntimeError(
                f"invalid Docker PID limit {raw_pids_limit!r}: {exc}"
            ) from exc

        if workspace is None:
            workspace = Path(tempfile.mkdtemp(prefix="carl_skill_docker_"))
        workspace.mkdir(parents=True, exist_ok=True)
        workspace_in = workspace / "in"
        workspace_out = workspace / "out"
        workspace_in.mkdir(exist_ok=True)
        workspace_out.mkdir(exist_ok=True)

        return SkillRuntimeHandle(
            workspace_root=workspace,
            workspace_in=workspace_in,
            workspace_out=workspace_out,
            backend={
                "isolation": "docker",
                "docker_executable": os.path.abspath(discovered_docker),
                "image": image,
                "mem_limit": mem_limit,
                "cpu_limit": str(raw_cpu_limit),
                "pids_limit": pids_limit,
                "container_user": (
                    f"{os.getuid()}:{os.getgid()}"
                    if os.name == "posix"
                    else None
                ),
                # Set immediately before each ``docker run``. It is cleared
                # only after normal completion or confirmed forced removal.
                # A failed removal deliberately leaves the name here so
                # ``cleanup`` can retry without losing the daemon-owned
                # resource identifier.
                "active_container_name": None,
                "network_policy": policy,
                "network_allowlist": allowlist,
                # Docker DOES enforce these — distinct from LocalSkillRuntime
                # which only validates the contract.
                "network_enforced": (
                    policy == "none"
                    or (
                        policy == "allowlist"
                        and network_binding is not None
                    )
                ),
                "network_binding": network_binding,
                "workspace_root_in_runtime": "/workspace",
                "workspace_in_in_runtime": "/workspace/in",
                "workspace_out_in_runtime": "/workspace/out",
                "workspace_output_mode": workspace_output_mode,
            },
        )

    async def run(
        self,
        handle: SkillRuntimeHandle,
        cmd: list[str],
        *,
        env: Optional[dict[str, str]] = None,
        stdin: Optional[bytes] = None,
        timeout: Optional[float] = None,
        cwd: Optional[str] = None,
    ) -> RuntimeRunResult:
        pending_container = handle.backend.get("active_container_name")
        if pending_container:
            raise SkillRuntimeError(
                "DockerSkillRuntime: a previous container still requires "
                f"cleanup: {pending_container}"
            )

        container_name = f"carl-runtime-{uuid.uuid4().hex[:12]}"
        docker_cmd = self._build_docker_cmd(
            handle,
            cmd,
            env=env,
            stdin=stdin,
            timeout=timeout,
            cwd=cwd,
            container_name=container_name,
        )
        handle.backend["active_container_name"] = container_name

        loop_start = asyncio.get_event_loop().time()
        try:
            proc = await asyncio.create_subprocess_exec(
                *docker_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                stdin=asyncio.subprocess.PIPE if stdin is not None else None,
                start_new_session=(os.name == "posix"),
            )
        except OSError as exc:
            # No docker CLI process was created, so this invocation could not
            # have registered the named container with the daemon.
            self._clear_active_container(handle, container_name)
            raise SkillRuntimeError(
                f"DockerSkillRuntime: failed to launch docker CLI: {exc}"
            ) from exc

        communicate_task = asyncio.create_task(
            _communicate_bounded(
                proc,
                stdin=stdin,
                max_output_bytes=handle.backend.get("max_output_bytes"),
            )
        )

        async def stop_invocation() -> None:
            _kill_process_group(proc, include_exited_leader=True)
            # Killing the client does not reliably stop a daemon-owned
            # container. Force removal is stronger than ``docker kill`` and
            # its exit status is checked. This is intentionally attempted
            # even when ``proc.pid`` is unavailable: the stable container
            # name, not the local CLI pid, is the daemon resource identity.
            removal_error: SkillRuntimeError | None = None
            try:
                await self._remove_active_container(
                    handle,
                    reason="timeout/cancellation",
                )
            except SkillRuntimeError as exc:
                removal_error = exc
            try:
                await asyncio.wait_for(communicate_task, timeout=1.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                communicate_task.cancel()
            if removal_error is not None:
                raise removal_error

        try:
            stdout_b, stderr_b, stdout_truncated, stderr_truncated = await asyncio.wait_for(
                asyncio.shield(communicate_task), timeout=timeout,
            )
        except asyncio.CancelledError:
            try:
                await stop_invocation()
            except SkillRuntimeError:
                # Preserve cancellation as control flow. The tracked name
                # stays on the handle so executor cleanup can retry.
                _log.warning(
                    "DockerSkillRuntime: removal failed during cancellation",
                    exc_info=True,
                )
            raise
        except asyncio.TimeoutError:
            await stop_invocation()
            duration = asyncio.get_event_loop().time() - loop_start
            return RuntimeRunResult(
                stdout=b"",
                stderr=f"[timeout after {timeout}s]".encode(),
                exit_code=124,
                duration_s=duration,
            )

        duration = asyncio.get_event_loop().time() - loop_start
        if proc.returncode == 125:
            message = stderr_b.decode("utf-8", errors="replace").strip()
            raise SkillRuntimeError(
                "DockerSkillRuntime: docker run failed before command execution"
                + (f": {message}" if message else "")
            )
        # A non-negative command status means attached ``docker run --rm``
        # completed and the daemon removed the stopped container. A CLI
        # process killed by an external signal retains the name for cleanup.
        if proc.returncode is not None and proc.returncode >= 0:
            self._clear_active_container(handle, container_name)
        return RuntimeRunResult(
            stdout=stdout_b,
            stderr=stderr_b,
            exit_code=proc.returncode or 0,
            duration_s=duration,
            stdout_truncated=stdout_truncated,
            stderr_truncated=stderr_truncated,
        )

    async def read_file(self, handle: SkillRuntimeHandle, path: str) -> bytes:
        target = self._resolve_workspace_path(handle, path)
        return await asyncio.to_thread(target.read_bytes)

    async def read_file_bounded(
        self,
        handle: SkillRuntimeHandle,
        path: str,
        max_bytes: int,
        timeout: Optional[float] = None,
    ) -> bytes:
        return await _read_file_bounded_async(
            handle.workspace_root,
            path,
            max_bytes,
            timeout,
        )

    async def write_file(
        self, handle: SkillRuntimeHandle, path: str, data: bytes,
    ) -> None:
        target = self._resolve_workspace_path(handle, path)
        target.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(target.write_bytes, data)

    async def cleanup(self, handle: SkillRuntimeHandle) -> None:
        # Retry daemon cleanup before touching the bind-mounted workspace.
        # If removal cannot be confirmed, preserve both the container name
        # and workspace for a later retry instead of hiding the leak.
        if handle.backend.get("active_container_name"):
            await self._remove_active_container(handle, reason="cleanup")
        if handle.backend.get("persisted"):
            return
        await _remove_workspace_tree(handle.workspace_root)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_docker_cmd(
        self,
        handle: SkillRuntimeHandle,
        cmd: list[str],
        *,
        env: Optional[dict[str, str]],
        timeout: Optional[float],
        cwd: Optional[str],
        stdin: Optional[bytes] = None,
        container_name: Optional[str] = None,
    ) -> list[str]:
        """Assemble the full ``docker run`` argv for one ``cmd``.

        Defaults: ``--rm`` ephemeral container, ``--read-only`` rootfs,
        a bounded noexec/nosuid/nodev ``/tmp`` tmpfs, ``in/`` mounted
        read-only and ``out/`` read-write. Resource limits
        + network policy come from ``handle.backend``.
        """
        backend = handle.backend
        docker_executable = backend.get("docker_executable", "docker")
        docker: list[str] = [
            str(docker_executable), "run", "--rm",
            # Sandbox execution must never trigger a daemon-side registry
            # download after command approval. Operators provision and pin the
            # image locally; missing images fail closed.
            "--pull=never",
            "--memory", str(backend.get("mem_limit", _DEFAULT_MEM_LIMIT)),
            # Match memory+swap to the RAM ceiling so Docker cannot grant an
            # additional equal amount of swap behind the host policy's back.
            "--memory-swap", str(backend.get("mem_limit", _DEFAULT_MEM_LIMIT)),
            "--cpus", str(backend.get("cpu_limit", _DEFAULT_CPU_LIMIT)),
            "--pids-limit", str(backend.get("pids_limit", _DEFAULT_PIDS_LIMIT)),
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            "--read-only",
            "--tmpfs", f"/tmp:rw,noexec,nosuid,nodev,size={_DEFAULT_TMPFS_SIZE}",
        ]
        if container_name:
            docker += ["--name", container_name]
        if stdin is not None:
            # ``docker run`` does not attach its stdin without ``-i`` even
            # when the CLI process itself owns a pipe.
            docker.append("-i")

        # Network policy — Docker enforces this for real.
        policy = backend.get("network_policy", "none")
        if policy == "none":
            docker += ["--network", "none"]
        elif policy == "host":
            docker += ["--network", "host"]
        elif policy == "allowlist":
            from .network_enforcement import DockerNetworkBinding  # noqa: PLC0415

            binding = backend.get("network_binding")
            if not isinstance(binding, DockerNetworkBinding):
                # Hostnames are not Docker network ids. Fail closed until a
                # NetworkEnforcer supplies a typed host-owned route.
                docker += ["--network", "none"]
            else:
                docker += ["--network", binding.network_name]

        # Workspace mount. ``in/`` is always read-only. CodeStep requests a
        # host-owned read-only ``out/`` mount because its result travels over
        # bounded stdout and generated code must not grow a host bind mount.
        output_suffix = ":ro" if backend.get("workspace_output_mode") == "read_only" else ""
        docker += [
            "-v", f"{handle.workspace_in}:/workspace/in:ro",
            "-v", f"{handle.workspace_out}:/workspace/out{output_suffix}",
        ]

        if cwd is not None:
            docker += ["-w", cwd]

        if env:
            for k, v in env.items():
                docker += ["-e", f"{k}={v}"]

        # Bind-mounted output files must remain owned/removable by the host
        # user. This value is host-owned and cannot be overridden by chain
        # configuration.
        container_user = backend.get("container_user")
        if container_user:
            docker += ["--user", str(container_user)]

        # Image + the actual command to run inside the container.
        docker.append(str(backend.get("image", _DEFAULT_IMAGE)))
        docker += cmd
        return docker

    async def _remove_active_container(
        self,
        handle: SkillRuntimeHandle,
        *,
        reason: str,
    ) -> None:
        """Force-remove the tracked container and confirm daemon success.

        Docker owns the container independently of the local CLI process, so
        terminating that process alone is insufficient. A successful
        ``docker rm --force`` response (or an explicit "no such container"
        response after ``--rm`` won the race) is the acknowledgement that
        lets us forget the name. Every other result preserves it for retry.
        """

        container_name = handle.backend.get("active_container_name")
        if not container_name:
            return

        try:
            remover = await asyncio.create_subprocess_exec(
                str(handle.backend.get("docker_executable", "docker")),
                "rm",
                "--force",
                str(container_name),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=(os.name == "posix"),
            )
        except OSError as exc:
            raise SkillRuntimeError(
                "DockerSkillRuntime: failed to launch container removal "
                f"during {reason}: {exc}"
            ) from exc

        communicate_task = asyncio.create_task(remover.communicate())
        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                asyncio.shield(communicate_task),
                timeout=_DOCKER_CONTROL_TIMEOUT_S,
            )
        except asyncio.CancelledError:
            _kill_process_group(remover, include_exited_leader=True)
            communicate_task.cancel()
            await asyncio.gather(communicate_task, return_exceptions=True)
            raise
        except asyncio.TimeoutError as exc:
            _kill_process_group(remover, include_exited_leader=True)
            communicate_task.cancel()
            await asyncio.gather(communicate_task, return_exceptions=True)
            raise SkillRuntimeError(
                "DockerSkillRuntime: timed out removing active container "
                f"{container_name!r} during {reason}"
            ) from exc
        except Exception as exc:  # noqa: BLE001 — backend protocol boundary
            _kill_process_group(remover, include_exited_leader=True)
            communicate_task.cancel()
            await asyncio.gather(communicate_task, return_exceptions=True)
            raise SkillRuntimeError(
                "DockerSkillRuntime: container removal command failed "
                f"for {container_name!r} during {reason}: {exc}"
            ) from exc

        output = (stderr_b or stdout_b)[:_DOCKER_CONTROL_OUTPUT_BYTES]
        detail = output.decode("utf-8", errors="replace").strip()
        missing = "no such container" in detail.lower()
        if remover.returncode == 0 or missing:
            self._clear_active_container(handle, str(container_name))
            return

        raise SkillRuntimeError(
            "DockerSkillRuntime: failed to remove active container "
            f"{container_name!r} during {reason} "
            f"(docker exit {remover.returncode})"
            + (f": {detail}" if detail else "")
        )

    @staticmethod
    def _clear_active_container(
        handle: SkillRuntimeHandle,
        container_name: str,
    ) -> None:
        """Clear only the invocation that this caller actually owns."""

        if handle.backend.get("active_container_name") == container_name:
            handle.backend["active_container_name"] = None

    @staticmethod
    def _resolve_workspace_path(
        handle: SkillRuntimeHandle, path: str,
    ) -> Path:
        """Same safety contract as LocalSkillRuntime — reject absolute
        paths and ``..`` segments that escape the workspace.
        """
        import os as _os
        if _os.path.isabs(path):
            raise SkillRuntimeError(
                f"workspace paths must be relative: got {path!r}"
            )
        target = (handle.workspace_root / path).resolve()
        root = handle.workspace_root.resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise SkillRuntimeError(
                f"path {path!r} escapes workspace {root}"
            ) from exc
        return target


# Self-register on import.
register_skill_runtime(DockerSkillRuntime.name, DockerSkillRuntime)


__all__ = ["DockerSkillRuntime"]
