"""Pluggable execution backends for AgentSkill scripts.

The CARE ecosystem needs to run untrusted AgentSkill
scripts in real sandboxes (Docker / E2B / Firejail) instead of the
default host-level subprocess execution. Today
``AgentSkillStepConfig.runtime`` is a declarative string field that no
code branched on; this module turns it into a real protocol with a
registry so CARE can plug in additional sandboxes without forking CARL.

Reference backends shipped here:

* :class:`LocalSkillRuntime` — wraps the existing
  ``asyncio.create_subprocess_exec`` behaviour on the host (no
  isolation). Always available. Emits a one-time warning on first use
  so callers know they are not sandboxed.

Plus the registration plumbing (:func:`register_skill_runtime`,
:func:`get_skill_runtime`) and the typed error
(:class:`SkillRuntimeError`) raised when the requested runtime can't
be loaded.

DockerSkillRuntime, E2BSkillRuntime, and FirejailSkillRuntime implement the
same protocol. Docker and Firejail use their system CLIs; E2B uses the
optional ``mmar-carl[e2b]`` dependency.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import signal
import stat
import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal, Optional, Protocol, runtime_checkable

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Network policy contract
# ---------------------------------------------------------------------------

NetworkPolicy = Literal["none", "allowlist", "host"]
"""Network access policy for a skill runtime invocation.

* ``"none"`` (default) — request no egress. Docker and Firejail enforce it;
  local records the request but cannot isolate host networking.
* ``"allowlist"`` — request an egress filter for the exact hosts in
  ``runtime_config["network_allowlist"]``. A backend must explicitly attest
  enforcement; the reference Docker/Firejail implementations currently fail
  closed until managed egress is available.
* ``"host"`` — full host networking (escape hatch). Backends MUST log
  a ``UserWarning`` when this is selected so it shows up in CARE's TUI
  as a banner.
"""


_WEBFETCH_DOMAIN_RE = re.compile(
    r"WebFetch\s*\(\s*domain\s*:\s*([^\s)]+)\s*\)",
    re.IGNORECASE,
)


def parse_network_allowlist_from_allowed_tools(
    allowed_tools: list[str] | str | None,
) -> list[str]:
    """Extract egress hosts from a skill's ``allowed-tools`` declaration.

    Tokens of the form ``WebFetch(domain:host.example)`` are recognised
    — the domain is pulled out and returned. Bare ``WebFetch`` (no
    domain constraint) is skipped: an unconstrained network policy is
    a deliberate choice for the chain author, not the skill manifest.

    Args:
        allowed_tools: Either the raw frontmatter string
            (``"Bash(git:*) WebFetch(domain:api.x.com)"``), the
            pre-tokenised list (``["Bash(git:*)", "WebFetch(domain:api.x.com)"]``),
            or ``None`` for skills with no restrictions.

    Returns:
        Sorted unique list of domain hosts. Empty when no
        ``WebFetch(domain:*)`` tokens are present.
    """
    if allowed_tools is None:
        return []
    if isinstance(allowed_tools, str):
        text = allowed_tools
    else:
        text = " ".join(allowed_tools)
    matches = _WEBFETCH_DOMAIN_RE.findall(text)
    return sorted({m.strip() for m in matches if m.strip()})


def resolve_network_policy(
    runtime_config: dict[str, Any] | None,
    *,
    manifest_allowed_tools: list[str] | str | None = None,
    default: NetworkPolicy = "none",
) -> tuple[NetworkPolicy, list[str]]:
    """Normalise the runtime's network policy + allowlist.

    Returns ``(policy, allowlist)`` where:

    * ``policy`` is one of ``"none"`` / ``"allowlist"`` / ``"host"`` —
      validated; an unknown value raises :class:`SkillRuntimeError`.
    * ``allowlist`` is the merged allowlist when ``policy == "allowlist"``.
      Hosts come from ``runtime_config["network_allowlist"]`` plus the
      manifest's ``WebFetch(domain:*)`` declarations (skill manifest
      becomes the source of truth for what egress the skill needs).

    Args:
        runtime_config: ``AgentSkillStepConfig.runtime_config`` (or
            ``None`` for the default ``"none"`` policy).
        manifest_allowed_tools: The skill's allowed-tools tokens; only
            consulted when policy is ``"allowlist"``.
        default: Policy when ``runtime_config`` is ``None`` or doesn't
            set a ``"network"`` key. Defaults to ``"none"`` — fail
            closed.

    Raises:
        SkillRuntimeError: When the policy string isn't one of the
            three recognised values.
    """
    cfg = runtime_config or {}
    raw = cfg.get("network", default)
    if raw not in ("none", "allowlist", "host"):
        raise SkillRuntimeError(
            f"Unknown network policy {raw!r}. "
            f"Valid options: 'none', 'allowlist', 'host'."
        )
    policy: NetworkPolicy = raw  # type: ignore[assignment]

    if policy != "allowlist":
        return policy, []

    explicit = cfg.get("network_allowlist") or []
    if not isinstance(explicit, (list, tuple, set)):
        raise SkillRuntimeError(
            "runtime_config['network_allowlist'] must be a list of host strings."
        )
    explicit_hosts = [str(h).strip() for h in explicit if str(h).strip()]

    manifest_hosts = parse_network_allowlist_from_allowed_tools(
        manifest_allowed_tools,
    )

    # Provider APIs may accept CIDRs and wildcard-like selectors in the same
    # field as hostnames.  CARL's public contract is deliberately narrower:
    # exact hostname/IP entries only, with one shared canonicalizer used by
    # managed-network plans as well.
    from .network_enforcement import (  # noqa: PLC0415
        NetworkEnforcerError,
        normalize_network_host,
    )

    try:
        selectors = [*explicit_hosts, *manifest_hosts]
        if not selectors:
            raise NetworkEnforcerError(
                "a managed network allowlist must contain at least one host"
            )
        # The same host may be declared by both the serialized runtime config
        # and the skill manifest.  Normalize first, then merge those two
        # authority-adjacent declarations deterministically.
        merged = sorted({normalize_network_host(host) for host in selectors})
    except (NetworkEnforcerError, TypeError) as exc:
        raise SkillRuntimeError(f"invalid network allowlist: {exc}") from exc
    return policy, merged


class SkillRuntimeError(RuntimeError):
    """Raised when a skill runtime fails to set up or execute.

    Distinct from a skill-script error (which surfaces through the
    return value's exit code / stderr): a ``SkillRuntimeError`` means
    the sandbox itself broke — missing docker daemon, unknown runtime
    name, container failed to start, etc. CARE renders this
    differently from "skill failed" in its TUI.
    """


ControlSupport = Literal["enforced", "advisory", "unsupported"]
ControlStatus = Literal["enforced", "advisory", "unsupported", "not_requested"]
EnforcementMode = Literal["strict", "best_effort"]


@dataclass(frozen=True)
class RuntimeCapabilities:
    """Static guarantees a runtime can make for one invocation.

    ``advisory`` means the backend records or applies a control only after the
    fact (for example E2B output slicing); it is not equivalent to prevention.
    Custom runtimes that do not declare capabilities are treated
    conservatively as unsupported.

    ``artifact_output_limit`` describes bounded artifact *collection* into
    CARL memory. It is not a workspace disk quota and does not stop a command
    from creating a larger temporary file before collection rejects it.
    """

    isolation: Literal["none", "process", "container", "microvm", "unknown"]
    wall_time: ControlSupport = "unsupported"
    output_limit: ControlSupport = "unsupported"
    cpu_limit: ControlSupport = "unsupported"
    memory_limit: ControlSupport = "unsupported"
    pids_limit: ControlSupport = "unsupported"
    network_none: ControlSupport = "unsupported"
    network_allowlist: ControlSupport = "unsupported"
    workspace_files: ControlSupport = "unsupported"
    persistent_shell: ControlSupport = "unsupported"
    artifact_output_limit: ControlSupport = "unsupported"


@dataclass(frozen=True)
class EnforcementReport:
    """Per-invocation record of requested and actually supported controls."""

    runtime: str
    isolation: str
    mode: EnforcementMode
    controls: dict[str, ControlStatus]

    @property
    def gaps(self) -> tuple[str, ...]:
        return tuple(
            name
            for name, status in self.controls.items()
            if status in ("advisory", "unsupported")
        )

    @property
    def fully_enforced(self) -> bool:
        return not self.gaps

    def as_dict(self) -> dict[str, Any]:
        return {
            "runtime": self.runtime,
            "isolation": self.isolation,
            "mode": self.mode,
            "fully_enforced": self.fully_enforced,
            "controls": dict(self.controls),
            "gaps": list(self.gaps),
        }

    def with_control(self, name: str, status: ControlStatus) -> EnforcementReport:
        controls = dict(self.controls)
        controls[name] = status
        return EnforcementReport(
            runtime=self.runtime,
            isolation=self.isolation,
            mode=self.mode,
            controls=controls,
        )


_UNKNOWN_CAPABILITIES = RuntimeCapabilities(isolation="unknown")


def get_runtime_capabilities(runtime: Any) -> RuntimeCapabilities:
    """Return a runtime declaration, defaulting to fail-closed unknowns."""

    capabilities = getattr(runtime, "capabilities", None)
    return capabilities if isinstance(capabilities, RuntimeCapabilities) else _UNKNOWN_CAPABILITIES


def assess_runtime_enforcement(
    runtime: Any,
    *,
    mode: EnforcementMode,
    network: NetworkPolicy,
    cpu_limit_requested: bool,
    memory_limit_requested: bool,
    pids_limit_requested: bool,
    workspace_files_requested: bool = False,
    artifact_outputs_requested: bool = False,
    network_allowlist_override: ControlSupport | None = None,
) -> EnforcementReport:
    """Build the report used for strict preflight and result provenance.

    ``network_allowlist_override`` is reserved for a host-owned
    :class:`~mmar_carl.network_enforcement.NetworkEnforcer` plan.  It must
    never be populated from serialized runtime configuration: a hostname list
    alone is not evidence that an egress filter exists.
    """

    capabilities = get_runtime_capabilities(runtime)
    network_status: ControlStatus
    if network == "host":
        network_status = "not_requested"
    elif network == "none":
        network_status = capabilities.network_none
    else:
        network_status = (
            network_allowlist_override
            if network_allowlist_override is not None
            else capabilities.network_allowlist
        )

    return EnforcementReport(
        runtime=str(getattr(runtime, "name", type(runtime).__name__)),
        isolation=capabilities.isolation,
        mode=mode,
        controls={
            "wall_time": capabilities.wall_time,
            "output_limit": capabilities.output_limit,
            "cpu_limit": capabilities.cpu_limit if cpu_limit_requested else "not_requested",
            "memory_limit": capabilities.memory_limit if memory_limit_requested else "not_requested",
            "pids_limit": capabilities.pids_limit if pids_limit_requested else "not_requested",
            "network": network_status,
            "workspace_files": (
                capabilities.workspace_files
                if workspace_files_requested
                else "not_requested"
            ),
            "artifact_output_limit": (
                capabilities.artifact_output_limit
                if artifact_outputs_requested
                else "not_requested"
            ),
        },
    )


@dataclass
class RuntimeRunResult:
    """Outcome of a single command execution inside a skill runtime."""

    stdout: bytes
    stderr: bytes
    exit_code: int
    duration_s: float
    stdout_truncated: bool = False
    stderr_truncated: bool = False


async def _communicate_bounded(
    proc: asyncio.subprocess.Process,
    *,
    stdin: bytes | None,
    max_output_bytes: int | None,
) -> tuple[bytes, bytes, bool, bool]:
    """Drain both pipes while retaining at most ``max_output_bytes`` each."""

    if max_output_bytes is None:
        stdout, stderr = await proc.communicate(input=stdin)
        return stdout, stderr, False, False

    async def feed_stdin() -> None:
        if proc.stdin is None:
            return
        try:
            if stdin:
                proc.stdin.write(stdin)
                await proc.stdin.drain()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            proc.stdin.close()

    async def drain(stream: asyncio.StreamReader | None) -> tuple[bytes, bool]:
        if stream is None:
            return b"", False
        kept = bytearray()
        truncated = False
        while True:
            chunk = await stream.read(64 * 1024)
            if not chunk:
                break
            if max_output_bytes is None:
                kept.extend(chunk)
                continue
            remaining = max_output_bytes - len(kept)
            if remaining > 0:
                kept.extend(chunk[:remaining])
            if len(chunk) > max(remaining, 0):
                truncated = True
        return bytes(kept), truncated

    _, stdout_pair, stderr_pair = await asyncio.gather(
        feed_stdin(),
        drain(proc.stdout),
        drain(proc.stderr),
    )
    await proc.wait()
    return stdout_pair[0], stderr_pair[0], stdout_pair[1], stderr_pair[1]


def _kill_process_group(
    proc: asyncio.subprocess.Process,
    *,
    include_exited_leader: bool = False,
) -> None:
    """Kill the command and descendants created in its POSIX session."""
    if proc.returncode is not None and not include_exited_leader:
        return
    pid = getattr(proc, "pid", None)
    # `killpg` reads 0 as "the caller's own process group" and 1 as init's.
    # Signalling either takes down the host application rather than the child
    # — and where the caller is itself PID 1 (a container), that is suicide.
    # A real child is always a plain int > 1; anything else (a mock, a closed
    # transport, a Windows handle) must fall through to the per-process kill.
    if os.name == "posix" and isinstance(pid, int) and pid > 1:
        target = pid
    else:
        target = None
    try:
        if target is not None:
            os.killpg(target, signal.SIGKILL)
        else:
            proc.kill()
    except (PermissionError, ProcessLookupError, TypeError):
        try:
            proc.kill()
        except ProcessLookupError:
            pass


def _validate_bounded_read_limit(max_bytes: int) -> None:
    if (
        isinstance(max_bytes, bool)
        or not isinstance(max_bytes, int)
        or max_bytes < 0
    ):
        raise SkillRuntimeError(
            f"max_bytes must be a non-negative integer, got {max_bytes!r}"
        )


def _workspace_relative_parts(path: str) -> tuple[str, ...]:
    """Return safe path components for descriptor-relative workspace I/O."""

    if not isinstance(path, str) or not path or "\x00" in path:
        raise SkillRuntimeError("workspace path must be a non-empty string")
    candidate = Path(path)
    if candidate.is_absolute() or candidate.drive:
        raise SkillRuntimeError(
            f"workspace paths must be relative: got {path!r}"
        )
    parts = tuple(part for part in candidate.parts if part not in ("", "."))
    if not parts or any(part == ".." for part in parts):
        raise SkillRuntimeError(f"path {path!r} escapes workspace")
    return parts


def _read_file_bounded_sync(
    workspace_root: Path,
    path: str,
    max_bytes: int,
) -> bytes:
    """Read one regular workspace file without following symlinks.

    POSIX hosts traverse from an open workspace directory descriptor using
    ``O_NOFOLLOW``. This prevents an output path from being swapped for a
    symlink between validation and open. The file size is checked before the
    result buffer is allocated, and every subsequent read is capped so a file
    that grows concurrently cannot exceed ``max_bytes`` in memory.
    """

    _validate_bounded_read_limit(max_bytes)
    parts = _workspace_relative_parts(path)
    root = workspace_root.resolve()

    if os.name == "posix":
        directory_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        directory_flags |= getattr(os, "O_DIRECTORY", 0)
        directory_flags |= getattr(os, "O_NOFOLLOW", 0)
        file_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        file_flags |= getattr(os, "O_NOFOLLOW", 0)
        # A FIFO must never block the worker thread before fstat rejects it.
        file_flags |= getattr(os, "O_NONBLOCK", 0)

        descriptors: list[int] = []
        try:
            current_fd = os.open(root, directory_flags)
            descriptors.append(current_fd)
            for component in parts[:-1]:
                current_fd = os.open(
                    component,
                    directory_flags,
                    dir_fd=current_fd,
                )
                descriptors.append(current_fd)
            file_fd = os.open(parts[-1], file_flags, dir_fd=current_fd)
            descriptors.append(file_fd)
            file_stat = os.fstat(file_fd)
            if not stat.S_ISREG(file_stat.st_mode):
                raise SkillRuntimeError(
                    f"artifact output {path!r} is not a regular file"
                )
            if file_stat.st_size > max_bytes:
                raise SkillRuntimeError(
                    f"artifact output {path!r} exceeds its {max_bytes}-byte limit"
                )

            kept = bytearray()
            while len(kept) < max_bytes:
                chunk = os.read(file_fd, min(64 * 1024, max_bytes - len(kept)))
                if not chunk:
                    break
                kept.extend(chunk)
            if len(kept) == max_bytes and os.read(file_fd, 1):
                raise SkillRuntimeError(
                    f"artifact output {path!r} exceeds its {max_bytes}-byte limit"
                )
            return bytes(kept)
        except SkillRuntimeError:
            raise
        except OSError as exc:
            raise SkillRuntimeError(
                f"artifact output {path!r} is unavailable or not a regular file: {exc}"
            ) from exc
        finally:
            for descriptor in reversed(descriptors):
                try:
                    os.close(descriptor)
                except OSError:
                    pass

    # Windows fallback: lstat every component before opening. Windows does
    # not provide portable descriptor-relative traversal through ``os.open``.
    target = root
    try:
        for component in parts:
            target = target / component
            component_stat = target.lstat()
            if stat.S_ISLNK(component_stat.st_mode):
                raise SkillRuntimeError(
                    f"artifact output {path!r} must not contain symlinks"
                )
        if not stat.S_ISREG(component_stat.st_mode):
            raise SkillRuntimeError(
                f"artifact output {path!r} is not a regular file"
            )
        if component_stat.st_size > max_bytes:
            raise SkillRuntimeError(
                f"artifact output {path!r} exceeds its {max_bytes}-byte limit"
            )
        kept = bytearray()
        with target.open("rb", buffering=0) as stream:
            while len(kept) < max_bytes:
                chunk = stream.read(min(64 * 1024, max_bytes - len(kept)))
                if not chunk:
                    break
                kept.extend(chunk)
            if len(kept) == max_bytes and stream.read(1):
                raise SkillRuntimeError(
                    f"artifact output {path!r} exceeds its {max_bytes}-byte limit"
                )
        return bytes(kept)
    except SkillRuntimeError:
        raise
    except OSError as exc:
        raise SkillRuntimeError(
            f"artifact output {path!r} is unavailable or not a regular file: {exc}"
        ) from exc


async def _read_file_bounded_async(
    workspace_root: Path,
    path: str,
    max_bytes: int,
    timeout: float | None,
) -> bytes:
    """Run the bounded regular-file reader without blocking the event loop."""

    operation = asyncio.to_thread(
        _read_file_bounded_sync,
        workspace_root,
        path,
        max_bytes,
    )
    try:
        if timeout is None:
            return await operation
        return await asyncio.wait_for(operation, timeout=timeout)
    except asyncio.TimeoutError as exc:
        raise SkillRuntimeError(
            f"artifact output read timed out after {timeout}s"
        ) from exc


async def _remove_workspace_tree(workspace_root: Path) -> None:
    """Remove an owned workspace idempotently without hiding real failures."""

    try:
        await asyncio.to_thread(shutil.rmtree, workspace_root)
    except FileNotFoundError:
        # ``cleanup`` is an idempotent protocol operation.
        return


@dataclass
class SkillRuntimeHandle:
    """Opaque handle returned by :meth:`SkillRuntime.prepare`.

    Carries the workspace root (always present — even containerised
    runtimes mount a host-side dir for I/O) and a free-form ``backend``
    dict that each runtime uses to stash whatever bookkeeping it needs
    (container id, sandbox session token, …).
    """

    workspace_root: Path
    workspace_in: Path
    workspace_out: Path
    backend: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class SkillRuntime(Protocol):
    """Protocol every skill execution backend must satisfy.

    Methods are all async so backends with network setup (Docker
    daemon, E2B API) don't block the chain's event loop. ``prepare``
    builds a workspace and returns a :class:`SkillRuntimeHandle`; the
    same handle threads through ``run`` / ``read_file`` /
    ``write_file`` / ``cleanup`` so the backend can stay stateful
    across calls (e.g. keep a long-lived container).
    """

    name: ClassVar[str]
    capabilities: ClassVar[RuntimeCapabilities]

    async def prepare(
        self,
        skill: Any,
        workspace: Optional[Path],
        config: dict[str, Any],
    ) -> SkillRuntimeHandle:
        """Create a fresh workspace + any backend state."""
        ...

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
        """Execute one command inside the runtime."""
        ...

    async def read_file(self, handle: SkillRuntimeHandle, path: str) -> bytes:
        """Read a file path *relative to ``workspace_root``*."""
        ...

    async def read_file_bounded(
        self,
        handle: SkillRuntimeHandle,
        path: str,
        max_bytes: int,
        timeout: Optional[float] = None,
    ) -> bytes:
        """Read one regular workspace file with a hard byte limit."""
        ...

    async def write_file(
        self, handle: SkillRuntimeHandle, path: str, data: bytes,
    ) -> None:
        """Write ``data`` to a path relative to ``workspace_root``."""
        ...

    async def cleanup(self, handle: SkillRuntimeHandle) -> None:
        """Tear down workspace + any backend state. Idempotent."""
        ...


# ---------------------------------------------------------------------------
# LocalSkillRuntime — the always-available reference backend
# ---------------------------------------------------------------------------


class LocalSkillRuntime:
    """Run scripts as host subprocesses with no isolation.

    Wraps the long-standing ``asyncio.create_subprocess_exec`` path
    that the AgentSkill executor used before the runtime protocol was
    introduced. Always available — no extras required — but **not a
    sandbox**: scripts inherit the caller's env, credentials, and
    filesystem access.

    On first use per process, emits a single ``UserWarning`` so users
    who flip to this runtime (default) know they are running unsafe
    code on the host. CARE's TUI promotes this into a visible banner.
    """

    name: ClassVar[str] = "local"
    capabilities: ClassVar[RuntimeCapabilities] = RuntimeCapabilities(
        isolation="none",
        # POSIX starts every command in a new session and kills the complete
        # process group. The Windows fallback can only kill the direct child;
        # without a Job Object, descendant wall-time enforcement is advisory.
        wall_time="enforced" if os.name == "posix" else "advisory",
        output_limit="enforced",
        workspace_files="enforced",
        artifact_output_limit="enforced",
    )
    _unsafe_warned: ClassVar[bool] = False

    async def prepare(
        self,
        skill: Any,
        workspace: Optional[Path],
        config: dict[str, Any],
    ) -> SkillRuntimeHandle:
        if not LocalSkillRuntime._unsafe_warned:
            warnings.warn(
                "LocalSkillRuntime executes skill scripts directly on the host "
                "with no sandboxing. For untrusted skills, install the "
                "Docker Engine and set `runtime='docker'`. This warning "
                "fires once per process.",
                UserWarning,
                stacklevel=3,
            )
            LocalSkillRuntime._unsafe_warned = True

        # resolve the network policy contract even though
        # the local backend can't enforce it. We validate the policy
        # string (caught typos surface as a SkillRuntimeError up-front
        # instead of silently failing in the Docker rewire), stash the
        # resolved policy + allowlist on the handle so callers /
        # introspection tests can read them, and emit a one-time warning
        # for the `host` escape hatch.
        manifest_allowed = config.get("_manifest_allowed_tools")
        policy, allowlist = resolve_network_policy(
            config, manifest_allowed_tools=manifest_allowed,
        )
        if policy == "host":
            warnings.warn(
                "Skill runtime selected network='host' — full host networking "
                "is enabled. Use 'none' or 'allowlist' for untrusted skills.",
                UserWarning,
                stacklevel=3,
            )

        if workspace is None:
            workspace = Path(tempfile.mkdtemp(prefix="carl_skill_"))
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
                "isolation": "none",
                "network_policy": policy,
                "network_allowlist": allowlist,
                # LocalSkillRuntime can't actually enforce any of this —
                # surface a single canonical flag so CARE's TUI knows
                # the policy is advisory rather than enforced.
                "network_enforced": False,
                "workspace_root_in_runtime": str(workspace),
                "workspace_in_in_runtime": str(workspace_in),
                "workspace_out_in_runtime": str(workspace_out),
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
        loop_start = asyncio.get_event_loop().time()
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                stdin=asyncio.subprocess.PIPE if stdin is not None else None,
                cwd=cwd,
                env=env,
                start_new_session=(os.name == "posix"),
            )
        except OSError as exc:
            # Distinguish "exec failed to start at all" from "script
            # crashed" — the former is a runtime-level failure.
            raise SkillRuntimeError(
                f"LocalSkillRuntime: failed to launch {cmd[0]!r}: {exc}"
            ) from exc

        max_output_bytes = handle.backend.get("max_output_bytes")
        communicate_task = asyncio.create_task(
            _communicate_bounded(
                proc,
                stdin=stdin,
                max_output_bytes=max_output_bytes,
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
                exit_code=124,  # standard timeout exit code
                duration_s=duration,
            )

        duration = asyncio.get_event_loop().time() - loop_start
        # A step owns its whole process group. Do not let a daemonized child
        # survive merely because the direct command exited successfully.
        _kill_process_group(proc, include_exited_leader=True)
        return RuntimeRunResult(
            stdout=stdout_b,
            stderr=stderr_b,
            exit_code=proc.returncode or 0,
            duration_s=duration,
            stdout_truncated=stdout_truncated,
            stderr_truncated=stderr_truncated,
        )

    async def read_file(self, handle: SkillRuntimeHandle, path: str) -> bytes:
        target = self._resolve_path(handle, path)
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
        target = self._resolve_path(handle, path)
        target.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(target.write_bytes, data)

    async def cleanup(self, handle: SkillRuntimeHandle) -> None:
        if handle.backend.get("persisted"):
            return
        await _remove_workspace_tree(handle.workspace_root)

    @staticmethod
    def _resolve_path(handle: SkillRuntimeHandle, path: str) -> Path:
        """Resolve a workspace-relative path safely.

        Absolute paths are rejected — they could escape the workspace,
        which defeats the (already-thin) safety story of LocalSkillRuntime.
        ``..`` segments are tolerated by ``Path.resolve()`` but the
        result must still sit inside ``workspace_root``.
        """
        if os.path.isabs(path):
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


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


SKILL_RUNTIME_REGISTRY: dict[str, type] = {}


def register_skill_runtime(name: str, cls: type) -> None:
    """Register a :class:`SkillRuntime` implementation under ``name``.

    The name is what users put in ``AgentSkillStepConfig.runtime``.
    Re-registering an existing name overwrites silently (lets test
    harnesses inject stub backends without juggling fixture order).
    """
    if not isinstance(cls, type):
        raise TypeError(f"runtime must be a class, got {type(cls).__name__}")
    SKILL_RUNTIME_REGISTRY[name] = cls


def get_skill_runtime(name: str) -> "SkillRuntime":
    """Look up + instantiate the runtime registered under ``name``.

    Raises :class:`SkillRuntimeError` when the name is unknown — CARE
    catches this and presents a friendly "install extras" message.
    """
    try:
        cls = SKILL_RUNTIME_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(SKILL_RUNTIME_REGISTRY)) or "<none>"
        raise SkillRuntimeError(
            f"Unknown skill runtime {name!r}. "
            f"Available: {available}. "
            "Install the backend prerequisites or call "
            "register_skill_runtime() with a custom backend."
        ) from exc
    return cls()  # type: ignore[no-any-return]


def list_skill_runtimes() -> list[str]:
    """Return the sorted list of registered runtime names."""
    return sorted(SKILL_RUNTIME_REGISTRY)


# Pre-register the always-available local backend on import. Other
# backends self-register via their own modules (`docker_skill_runtime.py`,
# etc.) and gate on their optional dependency being importable.
register_skill_runtime(LocalSkillRuntime.name, LocalSkillRuntime)


__all__ = [
    "ControlStatus",
    "ControlSupport",
    "EnforcementMode",
    "EnforcementReport",
    "RuntimeRunResult",
    "RuntimeCapabilities",
    "SkillRuntime",
    "SkillRuntimeError",
    "SkillRuntimeHandle",
    "LocalSkillRuntime",
    "NetworkPolicy",
    "parse_network_allowlist_from_allowed_tools",
    "resolve_network_policy",
    "assess_runtime_enforcement",
    "get_runtime_capabilities",
    "SKILL_RUNTIME_REGISTRY",
    "register_skill_runtime",
    "get_skill_runtime",
    "list_skill_runtimes",
]
