"""FirejailSkillRuntime — lightweight Linux-only sandbox for AgentSkills.

Sister to :class:`DockerSkillRuntime`, but trades the
container overhead for a process-level sandbox from
`Firejail <https://firejail.wordpress.com/>`_. Same SkillRuntime
protocol; same network-policy contract; same workspace-mount convention
— but the cost per call is a fork/exec instead of a container start.

Implementation note
-------------------
We shell out to the ``firejail`` CLI via subprocess. Self-registers as
``"firejail"`` in the shared
:data:`~mmar_carl.skill_runtime.SKILL_RUNTIME_REGISTRY` on module import.
Hermetic tests stub ``asyncio.create_subprocess_exec`` so they don't
need firejail installed.

The launcher and sandbox environments are deliberately separate.  The
absolute Firejail executable is resolved during :meth:`prepare`, then
started with a small host-owned environment.  Variables requested by a
step are applied by a trusted absolute ``env`` executable *after* the
Firejail option separator.  They therefore never become environment
variables or control options of the privileged launcher process itself.

``runtime_config`` recognised keys
-----------------------------------

Firejail profiles are host authority, not serializable chain data.  This
runtime uses one fixed in-code hardening baseline; chain-supplied ``profile``
and arbitrary CLI arguments are rejected.
* ``network`` — ``"none"`` / ``"allowlist"`` / ``"host"``.
  Mapping:
  - ``"none"`` → ``--net=none`` (no networking).
  - ``"host"`` → no ``--net=*`` flag (firejail leaves host networking
    intact).
  - ``"allowlist"`` → ``--net=none`` until a managed egress provider
    supplies a real bridge/filter. Hostnames are never treated as bridge ids.
* ``rlimit_as_mb`` — virtual-address-space cap in MiB. Default 2048
  (matches the Docker backend's 2 GiB ``--memory`` default).
* ``rlimit_cpu_s`` — CPU-time cap in seconds. ``None`` (default) means
  no rlimit; firejail doesn't enforce wall-clock time by default.
Arbitrary Firejail flags are intentionally rejected. A serializable chain must
not be able to override the fixed network/private/resource boundary; richer
profiles belong to trusted host configuration.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import shutil
import tempfile
import warnings
from pathlib import Path
from typing import Any, ClassVar, Optional

from .command_policy import parse_memory_limit_bytes
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


_DEFAULT_RLIMIT_AS_MB = 2048
_LAUNCHER_LOCALE = "C"


def _memory_limit_bytes(config: dict[str, Any]) -> int:
    """Translate public/legacy memory controls to an exact byte ceiling."""

    if config.get("mem_limit") is not None:
        value = config["mem_limit"]
        if not isinstance(value, str):
            raise SkillRuntimeError(
                "invalid memory limit for firejail: mem_limit requires an "
                "explicit b/k/m/g suffix"
            )
        try:
            return parse_memory_limit_bytes(value)
        except ValueError as exc:
            raise SkillRuntimeError(
                f"invalid memory limit for firejail: {value!r}: {exc}"
            ) from exc

    legacy_mb = config.get("rlimit_as_mb", _DEFAULT_RLIMIT_AS_MB)
    if (
        isinstance(legacy_mb, bool)
        or not isinstance(legacy_mb, (int, float))
        or not math.isfinite(float(legacy_mb))
        or float(legacy_mb) <= 0
    ):
        raise SkillRuntimeError(
            f"invalid legacy Firejail rlimit_as_mb: {legacy_mb!r}"
        )
    return max(1, int(float(legacy_mb) * 1024**2))


class FirejailSkillRuntime:
    """Run skill scripts inside a per-call firejail sandbox.

    Defaults: ``--noprofile`` (predictable), ``--quiet`` (no firejail
    chatter on stderr), private home/tmp/dev, dropped capabilities,
    no-new-privileges + seccomp, ``--net=none`` (no network), and an
    ``--rlimit-as`` virtual-memory cap.

    The workspace lives on the host filesystem; firejail's ``--private``
    flag points the sandbox's home directory at it. ``read_file`` /
    ``write_file`` operate on the host workspace directly (no sandbox
    round-trip needed since the host has full access to its own dir).

    Firejail enforces ``network='none'`` with ``--net=none``. Hostname
    allowlists fail closed to the same no-network mode until a managed egress
    provider exists; the handle reports that allowlist as not enforced.
    """

    name: ClassVar[str] = "firejail"
    capabilities: ClassVar[RuntimeCapabilities] = RuntimeCapabilities(
        isolation="process",
        wall_time="enforced",
        output_limit="enforced",
        # ``--rlimit-cpu`` limits CPU seconds, not requested CPU cores.
        cpu_limit="unsupported",
        # rlimit-as is per-process virtual address space, not aggregate RSS;
        # rlimit-nproc is shared by every process under the real host UID.
        memory_limit="advisory",
        pids_limit="advisory",
        network_none="enforced",
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
        # Up-front check: firejail must be on PATH. Surface a clear
        # `SkillRuntimeError` rather than failing inside `run` later.
        discovered_firejail = shutil.which("firejail")
        if discovered_firejail is None:
            raise SkillRuntimeError(
                "FirejailSkillRuntime requires the `firejail` CLI on PATH. "
                "Install via `apt install firejail` / `dnf install firejail` "
                "on Linux, or pick a different runtime via "
                "AgentSkillStepConfig(runtime=...)."
            )
        # ``shutil.which`` can return a relative path when the host PATH has
        # relative entries.  Resolve it once here; a later step-controlled
        # PATH must not participate in launcher selection.
        firejail_executable = str(Path(discovered_firejail).resolve())
        discovered_env = shutil.which("env", path=os.defpath)
        if discovered_env is None:
            raise SkillRuntimeError(
                "FirejailSkillRuntime requires the POSIX `env` utility in "
                f"the trusted system path {os.defpath!r}"
            )
        sandbox_env_executable = str(Path(discovered_env).resolve())

        manifest_allowed = config.get("_manifest_allowed_tools")
        policy, allowlist = resolve_network_policy(
            config, manifest_allowed_tools=manifest_allowed,
        )
        from .network_enforcement import FirejailNetworkBinding  # noqa: PLC0415

        network_binding = config.get("_network_binding")
        if network_binding is not None and (
            policy != "allowlist"
            or not isinstance(network_binding, FirejailNetworkBinding)
        ):
            raise SkillRuntimeError(
                "Firejail managed egress requires a host-issued "
                "FirejailNetworkBinding with network='allowlist'"
            )
        if policy == "host":
            warnings.warn(
                "FirejailSkillRuntime selected network='host' — host networking is enabled.",
                UserWarning,
                stacklevel=3,
            )

        if config.get("extra_args"):
            raise SkillRuntimeError(
                "FirejailSkillRuntime does not accept chain-controlled extra_args; "
                "use a host-owned runtime profile instead"
            )
        if config.get("profile"):
            raise SkillRuntimeError(
                "FirejailSkillRuntime does not accept chain-controlled profiles; "
                "use a host-owned runtime implementation instead"
            )

        rlimit_as_bytes = _memory_limit_bytes(config)
        raw_rlimit_cpu_s = config.get("rlimit_cpu_s")
        if raw_rlimit_cpu_s is None:
            rlimit_cpu_s = None
        elif (
            isinstance(raw_rlimit_cpu_s, bool)
            or not isinstance(raw_rlimit_cpu_s, int)
            or raw_rlimit_cpu_s <= 0
        ):
            raise SkillRuntimeError(
                "invalid Firejail CPU-time limit: expected a positive integer "
                f"number of seconds, got {raw_rlimit_cpu_s!r}"
            )
        else:
            rlimit_cpu_s = raw_rlimit_cpu_s
        pids_limit = config.get("pids_limit")
        if pids_limit is not None and (
            isinstance(pids_limit, bool)
            or not isinstance(pids_limit, int)
            or pids_limit <= 0
        ):
            raise SkillRuntimeError(
                f"invalid Firejail PID limit: {pids_limit!r}"
            )

        if workspace is None:
            workspace = Path(tempfile.mkdtemp(prefix="carl_skill_firejail_"))
        workspace.mkdir(parents=True, exist_ok=True)
        workspace_in = workspace / "in"
        workspace_out = workspace / "out"
        workspace_in.mkdir(exist_ok=True)
        workspace_out.mkdir(exist_ok=True)

        # ``--private=<directory>`` mounts that directory *at the current
        # user's home path* inside Firejail; the host source path is not the
        # portable coordinate for commands in the sandbox. Resolve the home
        # from passwd rather than ``$HOME`` because command environments are
        # intentionally minimal.
        try:
            import pwd

            passwd_entry = pwd.getpwuid(os.getuid())
            runtime_root = Path(passwd_entry.pw_dir)
            runtime_user = passwd_entry.pw_name
        except (ImportError, KeyError):  # pragma: no cover - Linux runtime
            runtime_root = Path.home()
            runtime_user = str(os.getuid()) if hasattr(os, "getuid") else "carl"

        return SkillRuntimeHandle(
            workspace_root=workspace,
            workspace_in=workspace_in,
            workspace_out=workspace_out,
            backend={
                "isolation": "firejail",
                "firejail_executable": firejail_executable,
                "sandbox_env_executable": sandbox_env_executable,
                "rlimit_as_mb": rlimit_as_bytes / 1024**2,
                "rlimit_as_bytes": rlimit_as_bytes,
                "rlimit_cpu_s": rlimit_cpu_s,
                "pids_limit": pids_limit,
                "network_policy": policy,
                "network_allowlist": allowlist,
                # network=none is enforced; RLIMIT-based resource controls are
                # reported separately as advisory capabilities.
                "network_enforced": (
                    policy == "none"
                    or (
                        policy == "allowlist"
                        and network_binding is not None
                    )
                ),
                "network_binding": network_binding,
                "workspace_root_in_runtime": str(runtime_root),
                "workspace_in_in_runtime": str(runtime_root / "in"),
                "workspace_out_in_runtime": str(runtime_root / "out"),
                "runtime_user": runtime_user,
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
        firejail_cmd = self._build_firejail_cmd(
            handle, cmd, env=env, cwd=cwd,
        )

        loop_start = asyncio.get_event_loop().time()
        try:
            proc = await asyncio.create_subprocess_exec(
                *firejail_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                stdin=asyncio.subprocess.PIPE if stdin is not None else None,
                env=self._trusted_launcher_env(handle),
                start_new_session=(os.name == "posix"),
            )
        except OSError as exc:
            raise SkillRuntimeError(
                f"FirejailSkillRuntime: failed to launch firejail CLI: {exc}"
            ) from exc

        communicate_task = asyncio.create_task(
            _communicate_bounded(
                proc,
                stdin=stdin,
                max_output_bytes=handle.backend.get("max_output_bytes"),
            )
        )
        try:
            stdout_b, stderr_b, stdout_truncated, stderr_truncated = await asyncio.wait_for(
                asyncio.shield(communicate_task), timeout=timeout,
            )
        except asyncio.CancelledError:
            _kill_process_group(proc, include_exited_leader=True)
            try:
                await asyncio.wait_for(communicate_task, timeout=1.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                communicate_task.cancel()
            raise
        except asyncio.TimeoutError:
            _kill_process_group(proc, include_exited_leader=True)
            try:
                await asyncio.wait_for(communicate_task, timeout=1.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                communicate_task.cancel()
            duration = asyncio.get_event_loop().time() - loop_start
            return RuntimeRunResult(
                stdout=b"",
                stderr=f"[timeout after {timeout}s]".encode(),
                exit_code=124,
                duration_s=duration,
            )

        duration = asyncio.get_event_loop().time() - loop_start
        _kill_process_group(proc, include_exited_leader=True)
        return RuntimeRunResult(
            stdout=stdout_b,
            stderr=stderr_b,
            exit_code=proc.returncode or 0,
            duration_s=duration,
            stdout_truncated=stdout_truncated,
            stderr_truncated=stderr_truncated,
        )

    async def read_file(
        self, handle: SkillRuntimeHandle, path: str,
    ) -> bytes:
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
        if handle.backend.get("persisted"):
            return
        await _remove_workspace_tree(handle.workspace_root)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_firejail_cmd(
        self,
        handle: SkillRuntimeHandle,
        cmd: list[str],
        *,
        env: Optional[dict[str, str]] = None,
        cwd: Optional[str],
    ) -> list[str]:
        """Assemble the ``firejail [opts...] -- cmd args...`` argv."""
        backend = handle.backend
        firejail_executable = backend.get("firejail_executable")
        if not isinstance(firejail_executable, str) or not os.path.isabs(
            firejail_executable,
        ):
            raise SkillRuntimeError(
                "FirejailSkillRuntime requires an absolute launcher path "
                "resolved by prepare()"
            )
        firejail: list[str] = [
            firejail_executable,
            "--quiet",
        ]

        # Fixed host-owned baseline. --noprofile avoids distribution profile
        # drift, so every security-relevant primitive is explicit here.
        firejail.extend(
            [
                "--noprofile",
                "--private-tmp",
                "--private-dev",
                "--caps.drop=all",
                "--nonewprivs",
                "--seccomp",
            ]
        )

        # Network policy.
        policy = backend.get("network_policy", "none")
        if policy == "none":
            firejail.append("--net=none")
        elif policy == "host":
            # Firejail leaves host networking intact when no --net flag
            # is provided. We pass nothing — documented behaviour.
            pass
        elif policy == "allowlist":
            from .network_enforcement import FirejailNetworkBinding  # noqa: PLC0415

            binding = backend.get("network_binding")
            if not isinstance(binding, FirejailNetworkBinding):
                firejail.append("--net=none")
            else:
                firejail.append(f"--net={binding.interface_name}")

        # Private home rooted at the workspace.
        firejail.append(f"--private={handle.workspace_root}")

        # Resource limits.
        rlimit_as_bytes = backend.get("rlimit_as_bytes")
        if rlimit_as_bytes is None:
            # Compatibility for manually constructed/older handles.
            rlimit_as_bytes = int(
                float(backend.get("rlimit_as_mb", _DEFAULT_RLIMIT_AS_MB))
                * 1024**2
            )
        firejail.append(f"--rlimit-as={int(rlimit_as_bytes)}")
        rlimit_cpu_s = backend.get("rlimit_cpu_s")
        if rlimit_cpu_s is not None:
            firejail.append(f"--rlimit-cpu={rlimit_cpu_s}")
        pids_limit = backend.get("pids_limit")
        if pids_limit is not None:
            firejail.append(f"--rlimit-nproc={int(pids_limit)}")

        if cwd is not None:
            firejail.append(f"--chdir={cwd}")

        sandbox_cmd = cmd
        if env is not None:
            if not cmd or "=" in cmd[0]:
                raise SkillRuntimeError(
                    "FirejailSkillRuntime cannot apply an explicit environment "
                    "to an empty command or an executable containing '='"
                )
            env_executable = backend.get("sandbox_env_executable")
            if not isinstance(env_executable, str) or not os.path.isabs(
                env_executable,
            ):
                raise SkillRuntimeError(
                    "FirejailSkillRuntime requires an absolute sandbox env "
                    "executable resolved by prepare()"
                )

            assignments: list[str] = []
            for key, value in env.items():
                if (
                    not isinstance(key, str)
                    or not isinstance(value, str)
                    or not key
                    or "=" in key
                    or "\x00" in key
                    or "\x00" in value
                ):
                    raise SkillRuntimeError(
                        f"invalid sandbox environment variable: {key!r}"
                    )
                assignments.append(f"{key}={value}")

            # Firejail's own --env option is intentionally not used here:
            # Firejail consults several stored variables (including PATH and
            # FIREJAIL_*) while constructing the sandbox.  A fixed env wrapper
            # after the option separator applies values only to the target.
            sandbox_cmd = [
                env_executable,
                "-i",
                "--",
                *assignments,
                *cmd,
            ]

        # ``--`` separates firejail's own flags from the command.
        firejail.append("--")
        firejail += sandbox_cmd
        return firejail

    @staticmethod
    def _trusted_launcher_env(handle: SkillRuntimeHandle) -> dict[str, str]:
        """Return the fixed environment used only to start Firejail.

        In particular, ``PATH``, loader variables, ``HOME``, and Firejail
        control variables from a step cannot reach the launcher boundary.
        The sandboxed command receives its requested values from the
        post-separator ``env`` wrapper.
        """

        backend = handle.backend
        runtime_root = str(backend.get("workspace_root_in_runtime") or "/")
        runtime_user = str(backend.get("runtime_user") or "carl")
        return {
            "PATH": os.defpath,
            "HOME": runtime_root,
            "USER": runtime_user,
            "LOGNAME": runtime_user,
            "LANG": _LAUNCHER_LOCALE,
            "LC_ALL": _LAUNCHER_LOCALE,
        }

    @staticmethod
    def _resolve_workspace_path(
        handle: SkillRuntimeHandle, path: str,
    ) -> Path:
        """Same safety contract as ``LocalSkillRuntime`` — reject
        absolute paths and ``..`` segments that escape the workspace.
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
register_skill_runtime(FirejailSkillRuntime.name, FirejailSkillRuntime)


__all__ = ["FirejailSkillRuntime"]
