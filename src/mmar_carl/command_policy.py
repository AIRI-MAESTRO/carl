"""Host-owned authorization policy for runtime command steps.

A chain is serializable input, so it cannot grant itself permission to run a
program.  The application supplies one compact :class:`CommandPolicy` through
``ReasoningContext``.  CARL evaluates the complete invocation and, when
needed, sends a typed :class:`CommandApprovalRequest` to the host callback.

The host remains responsible for UI and persisted approval scopes.  This
module deliberately does not implement users, roles, or an approval database.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

CommandDecisionOutcome = Literal["allow", "deny", "require_approval"]
CommandSource = Literal["static", "planned"]
CommandInvocationKind = Literal["command", "shell_session"]
NetworkPolicyName = Literal["none", "allowlist", "host"]


class CommandDecision(BaseModel):
    """Authorization outcome for one normalized invocation."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    outcome: CommandDecisionOutcome
    reason: str
    executable: str
    authorized_executable: str | None = None
    authorized_working_dir: str | None = None
    matched_rule: str | None = None


class CommandApprovalRequest(BaseModel):
    """Invocation details at the trusted CARL-to-host boundary.

    ``argv`` is available to the approval callback so a reviewer can inspect
    dynamic targets.  It is excluded from serialization and from the public
    step event, which exposes only the executable plus argument placeholders.
    The fingerprint covers every execution-affecting value, including hashes
    of stdin and env values.
    """

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    request_id: str
    fingerprint: str
    step_number: int
    step_title: str
    executable: str
    authorized_executable: str
    argv: tuple[str, ...] = Field(exclude=True, repr=False)
    argv_preview: tuple[str, ...]
    dynamic_argument_count: int = Field(ge=0)
    source: CommandSource
    invocation_kind: CommandInvocationKind
    capability_id: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$",
    )
    capability_revision: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$",
    )
    capability_fingerprint: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    arguments_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    environment: dict[str, str] = Field(exclude=True, repr=False)
    environment_keys: tuple[str, ...]
    stdin: bytes | None = Field(exclude=True, repr=False)
    stdin_sha256: str | None
    runtime: str
    working_dir: str | None
    network: NetworkPolicyName
    network_allowlist: tuple[str, ...]
    resources: dict[str, Any]
    enforcement_report: dict[str, Any]
    artifact_inputs: dict[str, bytes] = Field(exclude=True, repr=False)
    artifact_manifest: dict[str, Any]
    enforcement_mode: Literal["strict", "best_effort"]
    reason: str

    @model_validator(mode="after")
    def validate_capability_provenance(self) -> CommandApprovalRequest:
        provenance = (
            self.capability_id,
            self.capability_revision,
            self.capability_fingerprint,
            self.arguments_sha256,
        )
        if self.source == "planned":
            if self.invocation_kind != "command":
                raise ValueError("planned approval is only valid for a command invocation")
            if any(value is None for value in provenance):
                raise ValueError(
                    "planned approval requires complete command capability provenance"
                )
        elif any(value is not None for value in provenance):
            raise ValueError("static approval cannot carry command capability provenance")
        return self

    @classmethod
    def create(
        cls,
        *,
        step_number: int,
        step_title: str,
        static_argument_count: int,
        argv: Sequence[str],
        authorized_executable: str,
        env: dict[str, str],
        stdin: bytes | None,
        runtime: str,
        working_dir: str | None,
        network: NetworkPolicyName,
        network_allowlist: Sequence[str],
        resources: dict[str, Any],
        enforcement_mode: Literal["strict", "best_effort"],
        reason: str,
        enforcement_report: dict[str, Any] | None = None,
        artifact_manifest: dict[str, Any] | None = None,
        artifact_inputs: dict[str, bytes] | None = None,
        source: CommandSource = "static",
        invocation_kind: CommandInvocationKind = "command",
        requested_executable: str | None = None,
        capability_id: str | None = None,
        capability_revision: str | None = None,
        capability_fingerprint: str | None = None,
        arguments_sha256: str | None = None,
    ) -> CommandApprovalRequest:
        normalized_argv = [authorized_executable, *list(argv)[1:]]
        stdin_hash = hashlib.sha256(stdin).hexdigest() if stdin is not None else None
        canonical_payload: dict[str, Any] = {
            "argv": normalized_argv,
            "artifact_manifest": artifact_manifest or {},
            "enforcement_mode": enforcement_mode,
            "enforcement_report": enforcement_report or {},
            "env": dict(sorted(env.items())),
            "network": network,
            "network_allowlist": sorted(network_allowlist),
            "resources": resources,
            "runtime": runtime,
            "source": source,
            "invocation_kind": invocation_kind,
            "stdin_sha256": stdin_hash,
            "working_dir": working_dir,
        }
        if capability_id is not None:
            canonical_payload["capability"] = {
                "id": capability_id,
                "revision": capability_revision,
                "fingerprint": capability_fingerprint,
                "arguments_sha256": arguments_sha256,
            }
        canonical = json.dumps(
            canonical_payload,
            # Escaping non-ASCII keeps canonicalization total even for a
            # malformed surrogate in approval-only metadata. Actual argv/env
            # must still pass the explicit UTF-8 invocation gate.
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        fingerprint = hashlib.sha256(canonical).hexdigest()
        static_count = max(1, min(static_argument_count, len(normalized_argv)))
        preview = [_truncate(normalized_argv[0])]
        preview.extend(f"<static-arg-{index}>" for index in range(1, static_count))
        preview.extend(f"<dynamic-arg-{index + 1}>" for index in range(len(normalized_argv) - static_count))
        return cls(
            request_id=f"command-{step_number}-{uuid.uuid4().hex}",
            fingerprint=fingerprint,
            step_number=step_number,
            step_title=step_title,
            executable=requested_executable or argv[0],
            authorized_executable=authorized_executable,
            argv=normalized_argv,
            argv_preview=preview,
            dynamic_argument_count=len(normalized_argv) - static_count,
            source=source,
            invocation_kind=invocation_kind,
            capability_id=capability_id,
            capability_revision=capability_revision,
            capability_fingerprint=capability_fingerprint,
            arguments_sha256=arguments_sha256,
            environment=dict(env),
            environment_keys=sorted(env),
            stdin=stdin,
            stdin_sha256=stdin_hash,
            runtime=runtime,
            working_dir=working_dir,
            network=network,
            network_allowlist=sorted(network_allowlist),
            resources=resources,
            enforcement_report=dict(enforcement_report or {}),
            artifact_inputs=dict(artifact_inputs or {}),
            artifact_manifest=dict(artifact_manifest or {}),
            enforcement_mode=enforcement_mode,
            reason=reason,
        )


class CommandPolicy(BaseModel):
    """Small exact-match executable policy owned by the application host.

    Executable rules are exact strings, never basenames or globs. For runtimes
    that launch a host binary (local and Firejail), applicable rules and the
    requested executable must be absolute paths; the normalized path becomes
    the actual ``exec`` target. Container/VM runtimes may use an image-local
    token such as ``python``.

    An executable in ``approval_required_executables`` is eligible only after
    the callback approves the complete invocation.  Unknown executables,
    runtimes, networks, env keys, and host cwd roots are denied.

    For ``ShellSessionStep`` the executable is the shell. Approving it covers
    the complete script fingerprint but does not create an allowlist for
    programs launched by that script.
    """

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    allowed_executables: frozenset[str] = Field(default_factory=frozenset)
    approval_required_executables: frozenset[str] = Field(default_factory=frozenset)
    allowed_runtimes: frozenset[str] = Field(default_factory=frozenset)
    allowed_networks: frozenset[NetworkPolicyName] = Field(
        default_factory=lambda: frozenset({"none"}),
    )
    allowed_network_hosts: frozenset[str] = Field(default_factory=frozenset)
    allowed_env_keys: frozenset[str] = Field(default_factory=frozenset)
    local_working_roots: tuple[str, ...] = ()
    allow_best_effort: bool = False
    allow_planned_local: bool = False
    # Planned commands always cross the approval boundary. This is an
    # invariant, not a chain/host tuning knob; ``Literal[True]`` rejects an
    # accidental attempt to disable it at policy construction time.
    require_approval_for_planned: Literal[True] = True
    require_approval_for_interpreters: bool = True
    approval_timeout: float = Field(default=300.0, gt=0)
    network_enforcer_timeout: float = Field(default=30.0, gt=0)
    runtime_cleanup_timeout: float = Field(default=30.0, gt=0)
    max_timeout: float = Field(default=300.0, gt=0)
    max_artifact_io_timeout: float = Field(default=30.0, gt=0)
    max_cpu_limit: float | None = Field(default=None, gt=0)
    max_memory_bytes: int | None = Field(default=None, gt=0)
    max_pids_limit: int | None = Field(default=None, gt=0)
    max_output_bytes: int = Field(default=1_000_000, gt=0)
    max_argument_count: int = Field(default=256, gt=0)
    max_argv_bytes: int = Field(default=256_000, gt=0)
    max_environment_keys: int = Field(default=128, gt=0)
    max_environment_bytes: int = Field(default=256_000, gt=0)
    max_stdin_bytes: int = Field(default=1_000_000, gt=0)
    max_artifact_bytes: int = Field(default=1_000_000, gt=0)
    max_total_artifact_bytes: int = Field(default=4_000_000, gt=0)
    max_artifact_count: int = Field(default=32, gt=0)

    @field_validator("local_working_roots")
    @classmethod
    def normalize_working_roots(cls, roots: tuple[str, ...]) -> tuple[str, ...]:
        relative = [root for root in roots if not os.path.isabs(root)]
        if relative:
            raise ValueError(f"local_working_roots must be absolute paths: {relative!r}")
        return tuple(_normalize_local_path(root) for root in roots)

    @model_validator(mode="after")
    def validate_rules(self) -> CommandPolicy:
        if self.max_total_artifact_bytes < self.max_artifact_bytes:
            raise ValueError("max_total_artifact_bytes must be at least max_artifact_bytes")
        overlap = self.allowed_executables & self.approval_required_executables
        if overlap:
            raise ValueError(f"executable rules overlap: {sorted(overlap)!r}")
        if self.allowed_runtimes & _HOST_EXECUTION_RUNTIMES:
            normalized_allowed = {
                _normalize_local_path(rule) for rule in self.allowed_executables if os.path.isabs(rule)
            }
            normalized_approval = {
                _normalize_local_path(rule) for rule in self.approval_required_executables if os.path.isabs(rule)
            }
            normalized_overlap = normalized_allowed & normalized_approval
            if normalized_overlap:
                raise ValueError(
                    f"local executable rules overlap after path normalization: {sorted(normalized_overlap)!r}"
                )
        return self

    def evaluate(
        self,
        command: Sequence[str],
        *,
        runtime: str,
        network: NetworkPolicyName,
        enforcement_mode: Literal["strict", "best_effort"],
        working_dir: str | None,
        explicit_env_keys: Sequence[str],
        timeout: float = 30.0,
        artifact_io_timeout: float = 30.0,
        cpu_limit: float | None = None,
        mem_limit: str | None = None,
        pids_limit: int | None = None,
        max_output_bytes: int = 1_000_000,
        artifact_count: int = 0,
        environment: Mapping[str, str] | None = None,
        stdin: bytes | None = None,
        network_allowlist: Sequence[str] = (),
        source: CommandSource = "static",
        invocation_kind: CommandInvocationKind = "command",
    ) -> CommandDecision:
        """Authorize metadata and return the exact executable to launch."""

        if not command:
            return self._deny("", "command must contain at least one executable token")
        executable = command[0]
        if not isinstance(executable, str):
            return self._deny("", "command tokens must be strings")
        if runtime not in self.allowed_runtimes:
            return self._deny(executable, f"runtime {runtime!r} is not allowed")
        if network not in self.allowed_networks:
            return self._deny(executable, f"network policy {network!r} is not allowed")
        if network == "allowlist":
            requested_hosts = {_normalize_network_host(host) for host in network_allowlist}
            allowed_hosts = {_normalize_network_host(host) for host in self.allowed_network_hosts}
            disallowed_hosts = sorted(requested_hosts - allowed_hosts)
            if disallowed_hosts:
                return self._deny(
                    executable,
                    f"network hosts are not allowed: {disallowed_hosts!r}",
                )
        if enforcement_mode == "best_effort" and not self.allow_best_effort:
            return self._deny(executable, "best-effort enforcement is not allowed")
        # Reject malformed raw tokens before host path normalization; pathlib
        # may itself raise while resolving a lone surrogate. A second check
        # below covers the longer final normalized executable.
        invocation_error = self._authorize_invocation_shape(
            command,
            environment=environment or {},
            stdin=stdin,
        )
        if invocation_error is not None:
            return self._deny(executable, invocation_error)
        resource_error = self._authorize_resources(
            timeout=timeout,
            artifact_io_timeout=artifact_io_timeout,
            cpu_limit=cpu_limit,
            mem_limit=mem_limit,
            pids_limit=pids_limit,
            max_output_bytes=max_output_bytes,
            artifact_count=artifact_count,
        )
        if resource_error is not None:
            return self._deny(executable, resource_error)
        disallowed_env = sorted(set(explicit_env_keys) - self.allowed_env_keys)
        if disallowed_env:
            return self._deny(
                executable,
                f"environment keys are not allowed: {disallowed_env!r}",
            )
        authorized_working_dir, cwd_error = self._authorize_working_dir(
            runtime,
            working_dir,
        )
        if cwd_error is not None:
            return self._deny(executable, cwd_error)
        if runtime == "local" and source == "planned" and not self.allow_planned_local:
            return self._deny(executable, "LLM-planned host execution is not allowed")
        if source == "planned" and not self.require_approval_for_planned:
            return self._deny(
                executable,
                "application policy must require approval for LLM-planned commands",
            )

        match = self._match_executable(executable, runtime)
        if match is None:
            return self._deny(executable, "executable is not present in the exact allowlist")
        authorized, matched_rule, requires_approval = match

        # The host-owned normalized executable is what will actually run. Its
        # path may be longer than a chain-provided symlink, so invocation caps
        # must cover the final argv rather than only the requested spelling.
        invocation_error = self._authorize_invocation_shape(
            [authorized, *command[1:]],
            environment=environment or {},
            stdin=stdin,
        )
        if invocation_error is not None:
            return self._deny(executable, invocation_error)

        if self.require_approval_for_interpreters and (
            invocation_kind == "shell_session" or _is_interpreter(authorized)
        ):
            requires_approval = True
        if source == "planned":
            requires_approval = True
        if requires_approval:
            return CommandDecision(
                outcome="require_approval",
                reason="executable requires host approval",
                executable=executable,
                authorized_executable=authorized,
                authorized_working_dir=authorized_working_dir,
                matched_rule=matched_rule,
            )
        return CommandDecision(
            outcome="allow",
            reason="invocation matches the application policy",
            executable=executable,
            authorized_executable=authorized,
            authorized_working_dir=authorized_working_dir,
            matched_rule=matched_rule,
        )

    def _match_executable(
        self,
        executable: str,
        runtime: str,
    ) -> tuple[str, str, bool] | None:
        if runtime not in _HOST_EXECUTION_RUNTIMES:
            if executable in self.approval_required_executables:
                return executable, executable, True
            if executable in self.allowed_executables:
                return executable, executable, False
            return None
        if not os.path.isabs(executable):
            return None
        normalized = _normalize_local_path(executable)
        for rule in sorted(self.approval_required_executables):
            if os.path.isabs(rule) and _normalize_local_path(rule) == normalized:
                return normalized, rule, True
        for rule in sorted(self.allowed_executables):
            if os.path.isabs(rule) and _normalize_local_path(rule) == normalized:
                return normalized, rule, False
        return None

    def _authorize_invocation_shape(
        self,
        command: Sequence[str],
        *,
        environment: Mapping[str, str],
        stdin: bytes | None,
    ) -> str | None:
        if len(command) > self.max_argument_count:
            return (
                f"argument count {len(command)} exceeds host maximum "
                f"{self.max_argument_count}"
            )
        argv_size = _bounded_utf8_size(command, self.max_argv_bytes)
        if argv_size is None:
            return (
                "argv is not valid UTF-8 or exceeds host maximum "
                f"{self.max_argv_bytes} bytes"
            )
        if len(environment) > self.max_environment_keys:
            return (
                f"environment key count {len(environment)} exceeds host maximum "
                f"{self.max_environment_keys}"
            )
        env_parts = [part for item in environment.items() for part in item]
        env_size = _bounded_utf8_size(env_parts, self.max_environment_bytes)
        if env_size is None:
            return (
                "environment is not valid UTF-8 or exceeds host maximum "
                f"{self.max_environment_bytes} bytes"
            )
        if stdin is not None and len(stdin) > self.max_stdin_bytes:
            return f"stdin exceeds host maximum {self.max_stdin_bytes} bytes"
        return None

    def _authorize_working_dir(
        self,
        runtime: str,
        working_dir: str | None,
    ) -> tuple[str | None, str | None]:
        if working_dir is None:
            return None, None
        if not working_dir.strip() or "\x00" in working_dir:
            return None, "working directory must be non-empty and contain no NUL bytes"
        if runtime not in _HOST_EXECUTION_RUNTIMES:
            return working_dir, None
        if not self.local_working_roots:
            return None, "explicit host working directories are not allowed"
        try:
            candidate = Path(working_dir).expanduser().resolve(strict=False)
        except (OSError, ValueError) as exc:
            return None, f"invalid working directory: {exc}"
        for root_value in self.local_working_roots:
            root = Path(root_value).expanduser().resolve(strict=False)
            try:
                candidate.relative_to(root)
                return str(candidate), None
            except ValueError:
                continue
        return None, f"working directory {str(candidate)!r} is outside allowed host roots"

    def _authorize_resources(
        self,
        *,
        timeout: float,
        artifact_io_timeout: float,
        cpu_limit: float | None,
        mem_limit: str | None,
        pids_limit: int | None,
        max_output_bytes: int,
        artifact_count: int,
    ) -> str | None:
        checks = (
            (timeout, self.max_timeout, "timeout"),
            (artifact_io_timeout, self.max_artifact_io_timeout, "artifact I/O timeout"),
            (cpu_limit, self.max_cpu_limit, "CPU limit"),
            (pids_limit, self.max_pids_limit, "PID limit"),
            (max_output_bytes, self.max_output_bytes, "output byte limit"),
            (artifact_count, self.max_artifact_count, "artifact count"),
        )
        for requested, maximum, label in checks:
            if requested is not None and maximum is not None and requested > maximum:
                return f"requested {label} {requested!r} exceeds host maximum {maximum!r}"
        if mem_limit is not None and self.max_memory_bytes is not None:
            try:
                requested_memory = parse_memory_limit_bytes(mem_limit)
            except ValueError as exc:
                return str(exc)
            if requested_memory > self.max_memory_bytes:
                return (
                    f"requested memory limit {mem_limit!r} exceeds host maximum "
                    f"{self.max_memory_bytes} bytes"
                )
        return None

    @staticmethod
    def _deny(executable: str, reason: str) -> CommandDecision:
        return CommandDecision(outcome="deny", reason=reason, executable=executable)


_INTERPRETERS = frozenset({"bash", "dash", "fish", "node", "perl", "pwsh", "ruby", "sh", "zsh"})
_HOST_EXECUTION_RUNTIMES = frozenset({"firejail", "local"})
_MEMORY_LIMIT_RE = re.compile(r"([1-9][0-9]*)([bkmg])", re.IGNORECASE)


def normalize_memory_limit(value: str) -> str:
    """Return one unambiguous positive Docker-style byte limit."""

    match = _MEMORY_LIMIT_RE.fullmatch(value.strip())
    if match is None:
        raise ValueError(
            "mem_limit must be a positive integer with an explicit b/k/m/g suffix"
        )
    return f"{int(match.group(1))}{match.group(2).lower()}"


def parse_memory_limit_bytes(value: str) -> int:
    """Parse a normalized b/k/m/g limit using binary multiples."""

    normalized = normalize_memory_limit(value)
    amount = int(normalized[:-1])
    multiplier = {"b": 1, "k": 1024, "m": 1024**2, "g": 1024**3}[normalized[-1]]
    return amount * multiplier


def _is_interpreter(executable: str) -> bool:
    name = Path(executable).name.lower()
    return name in _INTERPRETERS or name.startswith("python")


def _normalize_local_path(value: str) -> str:
    return str(Path(value).expanduser().resolve(strict=False))


def _normalize_network_host(value: str) -> str:
    return value.strip().rstrip(".").lower()


def _bounded_utf8_size(values: Sequence[str], limit: int) -> int | None:
    """Count UTF-8 bytes without encoding an already-obviously huge value."""

    total = 0
    for value in values:
        if not isinstance(value, str):
            return None
        remaining = limit - total
        if remaining < 0 or len(value) > remaining:
            return None
        try:
            total += len(value.encode("utf-8")) + 1
        except UnicodeEncodeError:
            return None
        if total > limit:
            return None
    return total


def _truncate(value: str, limit: int = 160) -> str:
    return value if len(value) <= limit else f"{value[:limit]}…"


__all__ = [
    "CommandApprovalRequest",
    "CommandDecision",
    "CommandPolicy",
    "normalize_memory_limit",
    "parse_memory_limit_bytes",
]
