"""Host-owned network allowlist enforcement contracts.

The objects in this module deliberately live outside chain configuration.  A
chain may request an exact hostname/IP allowlist, but only the host application
can map that request to a trusted, pre-provisioned network profile.

``PreconfiguredNetworkEnforcer`` is an attestation adapter, not a provisioning
system.  It does not create Docker networks, configure Firejail interfaces, or
install firewall rules.  The operator is responsible for ensuring that each
configured binding really enforces the profile's exact allowlist.  This module
only makes that trust boundary typed, immutable, provenance-bound, and
auditable before a runtime starts.
"""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, Protocol, runtime_checkable

NetworkRuntime = Literal["docker", "firejail"]
NetworkBindingKind = Literal["docker_network", "firejail_interface"]

_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z", re.ASCII)
_HOST_LABEL_RE = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\Z", re.ASCII)
_DOCKER_NETWORK_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z", re.ASCII)
_FIREJAIL_INTERFACE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,14}\Z", re.ASCII)
_DOCKER_BUILTIN_NETWORKS = frozenset({"bridge", "default", "host", "none"})


class NetworkEnforcerError(RuntimeError):
    """The host network enforcer rejected or could not attest a request."""


def _validate_identifier(value: str, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if _IDENTIFIER_RE.fullmatch(value) is None:
        raise NetworkEnforcerError(f"{field_name} must be 1-128 ASCII characters matching [A-Za-z0-9][A-Za-z0-9_.-]*")
    return value


def _validate_runtime(value: str) -> NetworkRuntime:
    if value not in ("docker", "firejail"):
        raise NetworkEnforcerError(f"unsupported managed-network runtime {value!r}; expected 'docker' or 'firejail'")
    return value


def normalize_network_host(value: str) -> str:
    """Return one canonical exact hostname/IP or reject broader selectors.

    CIDRs, URLs, wildcard selectors, ports, and Unicode hostnames are not
    accepted.  This is shared by runtime policy parsing and managed-network
    attestations so a backend cannot interpret a CARL "host" as a broader
    provider-specific network selector.
    """
    if not isinstance(value, str):
        raise TypeError("network allowlist hosts must be strings")

    host = value.strip().removesuffix(".")
    if not host or host.endswith("."):
        raise NetworkEnforcerError(f"network allowlist entries must be exact hostnames or IPs: {value!r}")
    if any(character.isspace() for character in host) or any(
        marker in host for marker in ("\x00", "://", "/", "*", "%", "[", "]")
    ):
        raise NetworkEnforcerError(f"network allowlist entries must be exact hostnames or IPs: {value!r}")

    try:
        return ipaddress.ip_address(host).compressed.lower()
    except ValueError:
        pass

    # Do not reinterpret a malformed IPv4 address as a DNS hostname.
    if all(character.isdigit() or character == "." for character in host):
        raise NetworkEnforcerError(f"network allowlist entries must be exact hostnames or IPs: {value!r}")

    try:
        ascii_host = host.encode("ascii").decode("ascii").lower()
    except UnicodeError as exc:
        raise NetworkEnforcerError("network allowlist hostnames must be ASCII; use an explicit IDNA A-label") from exc

    if len(ascii_host) > 253:
        raise NetworkEnforcerError("network allowlist hostname exceeds 253 ASCII characters")
    labels = ascii_host.split(".")
    if any(_HOST_LABEL_RE.fullmatch(label) is None for label in labels):
        raise NetworkEnforcerError(f"network allowlist entries must be exact hostnames or IPs: {value!r}")
    return ascii_host


def normalize_network_allowlist_hosts(hosts: Sequence[str]) -> tuple[str, ...]:
    """Canonicalize a non-empty sequence of exact hostname/IP selectors."""

    if isinstance(hosts, (str, bytes, bytearray)) or not isinstance(hosts, Sequence):
        raise TypeError("hosts must be a sequence of hostname/IP strings")
    if not hosts:
        raise NetworkEnforcerError("a managed network allowlist must contain at least one host")

    normalized = tuple(normalize_network_host(host) for host in hosts)
    if len(normalized) != len(set(normalized)):
        raise NetworkEnforcerError("network allowlist contains duplicate normalized hosts")
    return tuple(sorted(normalized))


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _fingerprint(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class DockerNetworkBinding:
    """A host-selected, already provisioned Docker network."""

    network_name: str

    def __post_init__(self) -> None:
        if not isinstance(self.network_name, str):
            raise TypeError("network_name must be a string")
        if _DOCKER_NETWORK_RE.fullmatch(self.network_name) is None:
            raise NetworkEnforcerError("network_name must be a safe 1-128 character Docker network name")
        if self.network_name.lower() in _DOCKER_BUILTIN_NETWORKS:
            raise NetworkEnforcerError(
                "network_name must identify a dedicated managed-egress network, not a Docker built-in network"
            )

    @property
    def kind(self) -> Literal["docker_network"]:
        return "docker_network"


@dataclass(frozen=True, slots=True)
class FirejailNetworkBinding:
    """A host-selected, already configured Firejail network interface."""

    interface_name: str

    def __post_init__(self) -> None:
        if not isinstance(self.interface_name, str):
            raise TypeError("interface_name must be a string")
        if _FIREJAIL_INTERFACE_RE.fullmatch(self.interface_name) is None:
            raise NetworkEnforcerError("interface_name must be a safe Linux interface name of at most 15 characters")

    @property
    def kind(self) -> Literal["firejail_interface"]:
        return "firejail_interface"


type NetworkBinding = DockerNetworkBinding | FirejailNetworkBinding


def _binding_kind(binding: NetworkBinding) -> NetworkBindingKind:
    if isinstance(binding, DockerNetworkBinding):
        return "docker_network"
    if isinstance(binding, FirejailNetworkBinding):
        return "firejail_interface"
    raise TypeError("binding must be DockerNetworkBinding or FirejailNetworkBinding")


def _validate_binding_for_runtime(runtime: NetworkRuntime, binding: NetworkBinding) -> None:
    expected: NetworkBindingKind = "docker_network" if runtime == "docker" else "firejail_interface"
    actual = _binding_kind(binding)
    if actual != expected:
        raise NetworkEnforcerError(f"{runtime!r} profile requires {expected!r}, not {actual!r}")


def _binding_commitment(binding: NetworkBinding) -> str:
    value = binding.network_name if isinstance(binding, DockerNetworkBinding) else binding.interface_name
    return hashlib.sha256(f"{_binding_kind(binding)}\x00{value}".encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class NetworkEnforcementRequest:
    """A runtime's request for one exact, normalized egress allowlist."""

    runtime: NetworkRuntime
    hosts: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "runtime", _validate_runtime(self.runtime))
        object.__setattr__(self, "hosts", normalize_network_allowlist_hosts(self.hosts))


@dataclass(frozen=True, slots=True)
class ManagedNetworkProfile:
    """Host-owned mapping from an exact allowlist to trusted infrastructure.

    Creating this object attests that ``binding`` was provisioned outside CARL
    to enforce exactly ``hosts``.  No validation in this module can inspect the
    external firewall or gateway configuration behind that binding.
    """

    profile: str
    runtime: NetworkRuntime
    hosts: tuple[str, ...]
    binding: NetworkBinding = field(repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "profile", _validate_identifier(self.profile, field_name="profile"))
        runtime = _validate_runtime(self.runtime)
        object.__setattr__(self, "runtime", runtime)
        object.__setattr__(self, "hosts", normalize_network_allowlist_hosts(self.hosts))
        _validate_binding_for_runtime(runtime, self.binding)


@dataclass(frozen=True, slots=True)
class NetworkEnforcementPlan:
    """Stable public preflight attestation suitable for approval and logs."""

    enforcer_id: str
    revision: str
    profile: str
    runtime: NetworkRuntime
    hosts: tuple[str, ...]
    binding_kind: NetworkBindingKind
    binding_commitment: str = field(repr=False)
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "enforcer_id",
            _validate_identifier(self.enforcer_id, field_name="enforcer_id"),
        )
        object.__setattr__(
            self,
            "revision",
            _validate_identifier(self.revision, field_name="revision"),
        )
        object.__setattr__(self, "profile", _validate_identifier(self.profile, field_name="profile"))
        runtime = _validate_runtime(self.runtime)
        object.__setattr__(self, "runtime", runtime)
        object.__setattr__(self, "hosts", normalize_network_allowlist_hosts(self.hosts))
        expected_kind: NetworkBindingKind = "docker_network" if runtime == "docker" else "firejail_interface"
        if self.binding_kind != expected_kind:
            raise NetworkEnforcerError(f"{runtime!r} plan requires binding_kind={expected_kind!r}")
        if not re.fullmatch(r"[0-9a-f]{64}", self.binding_commitment):
            raise NetworkEnforcerError("binding_commitment must be a SHA-256 commitment")
        object.__setattr__(self, "fingerprint", _fingerprint(self._payload()))

    def _payload(self) -> dict[str, Any]:
        return {
            "binding_kind": self.binding_kind,
            "binding_commitment": self.binding_commitment,
            "enforcer_id": self.enforcer_id,
            "hosts": list(self.hosts),
            "profile": self.profile,
            "revision": self.revision,
            "runtime": self.runtime,
        }

    def as_dict(self) -> dict[str, Any]:
        """Return a detached, JSON-safe public approval record."""

        return {**self._payload(), "fingerprint": self.fingerprint}


@dataclass(frozen=True, slots=True)
class NetworkEnforcementLease:
    """Opaque runtime-only proof that an approved binding was acquired."""

    plan: NetworkEnforcementPlan
    binding: NetworkBinding = field(repr=False)
    _owner_token: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.plan, NetworkEnforcementPlan):
            raise TypeError("plan must be a NetworkEnforcementPlan")
        _validate_binding_for_runtime(self.plan.runtime, self.binding)
        if _binding_kind(self.binding) != self.plan.binding_kind:
            raise NetworkEnforcerError("lease binding kind does not match its enforcement plan")
        if not hmac.compare_digest(
            _binding_commitment(self.binding),
            self.plan.binding_commitment,
        ):
            raise NetworkEnforcerError("lease binding does not match its approved opaque commitment")


@runtime_checkable
class NetworkEnforcer(Protocol):
    """Runtime-only host contract for managed network allowlist bindings.

    Implementations are trusted security providers. ``acquire`` must be
    transactional: on error or :class:`asyncio.CancelledError` it must roll
    back any partially provisioned external resources before returning.
    ``release`` must be idempotent and cancellation-safe. CARL additionally
    shields bounded release, but a provider remains responsible for its own
    remote create/teardown races. Any change to firewall semantics behind an
    unchanged binding must bump ``revision``; a binding value change is also
    detected by the public opaque commitment.
    """

    enforcer_id: str
    revision: str

    def plan(self, request: NetworkEnforcementRequest) -> NetworkEnforcementPlan:
        """Return a side-effect-free public preflight plan."""
        ...

    async def acquire(self, plan: NetworkEnforcementPlan) -> NetworkEnforcementLease:
        """Transactionally acquire the binding attested by ``plan``."""
        ...

    async def release(self, lease: NetworkEnforcementLease) -> None:
        """Release a lease idempotently and safely under cancellation."""
        ...


@dataclass(frozen=True, slots=True, init=False)
class PreconfiguredNetworkEnforcer:
    """Attest exact allowlists using operator-provisioned network profiles.

    ``plan`` only performs an immutable mapping lookup and has no side effects.
    ``acquire`` and ``release`` are async for protocol compatibility, but this
    implementation does not provision or tear down infrastructure.  It trusts
    the operator's external network/firewall setup and returns the configured
    typed binding as an attested lease.
    """

    enforcer_id: str
    revision: str
    _profiles: Mapping[tuple[NetworkRuntime, tuple[str, ...]], ManagedNetworkProfile] = field(repr=False)
    _owner_token: object = field(repr=False, compare=False)

    def __init__(
        self,
        enforcer_id: str,
        revision: str,
        profiles: Sequence[ManagedNetworkProfile],
    ) -> None:
        enforcer_id = _validate_identifier(enforcer_id, field_name="enforcer_id")
        revision = _validate_identifier(revision, field_name="revision")
        if isinstance(profiles, (str, bytes, bytearray)) or not isinstance(profiles, Sequence):
            raise TypeError("profiles must be a sequence of ManagedNetworkProfile objects")

        by_request: dict[tuple[NetworkRuntime, tuple[str, ...]], ManagedNetworkProfile] = {}
        profile_names: set[str] = set()
        for profile in profiles:
            if not isinstance(profile, ManagedNetworkProfile):
                raise TypeError("profiles must contain ManagedNetworkProfile objects")
            key = (profile.runtime, profile.hosts)
            if key in by_request:
                raise NetworkEnforcerError(
                    "duplicate managed-network profile for exact request "
                    f"runtime={profile.runtime!r}, hosts={profile.hosts!r}"
                )
            if profile.profile in profile_names:
                raise NetworkEnforcerError(f"duplicate managed-network profile name: {profile.profile!r}")
            by_request[key] = profile
            profile_names.add(profile.profile)

        object.__setattr__(self, "enforcer_id", enforcer_id)
        object.__setattr__(self, "revision", revision)
        object.__setattr__(self, "_profiles", MappingProxyType(by_request))
        object.__setattr__(self, "_owner_token", object())

    def plan(self, request: NetworkEnforcementRequest) -> NetworkEnforcementPlan:
        if not isinstance(request, NetworkEnforcementRequest):
            raise TypeError("request must be a NetworkEnforcementRequest")
        try:
            profile = self._profiles[(request.runtime, request.hosts)]
        except KeyError as exc:
            raise NetworkEnforcerError(
                "no preconfigured managed-network profile matches the exact request "
                f"runtime={request.runtime!r}, hosts={request.hosts!r}"
            ) from exc
        return self._plan_for_profile(profile)

    async def acquire(self, plan: NetworkEnforcementPlan) -> NetworkEnforcementLease:
        profile, expected = self._validate_plan(plan)
        return NetworkEnforcementLease(
            plan=expected,
            binding=profile.binding,
            _owner_token=self._owner_token,
        )

    async def release(self, lease: NetworkEnforcementLease) -> None:
        if not isinstance(lease, NetworkEnforcementLease):
            raise TypeError("lease must be a NetworkEnforcementLease")
        if lease._owner_token is not self._owner_token:
            raise NetworkEnforcerError("network enforcement lease belongs to another enforcer")
        profile, expected = self._validate_plan(lease.plan)
        if lease.plan != expected or lease.binding != profile.binding:
            raise NetworkEnforcerError("network enforcement lease no longer matches its profile")
        # Preconfigured profiles are operator-owned and remain provisioned.

    def _validate_plan(
        self,
        plan: NetworkEnforcementPlan,
    ) -> tuple[ManagedNetworkProfile, NetworkEnforcementPlan]:
        if not isinstance(plan, NetworkEnforcementPlan):
            raise TypeError("plan must be a NetworkEnforcementPlan")
        if plan.enforcer_id != self.enforcer_id or plan.revision != self.revision:
            raise NetworkEnforcerError("network enforcement plan belongs to another enforcer revision")
        try:
            profile = self._profiles[(plan.runtime, plan.hosts)]
        except KeyError as exc:
            raise NetworkEnforcerError("network enforcement plan has no matching preconfigured profile") from exc
        expected = self._plan_for_profile(profile)
        if plan != expected or not hmac.compare_digest(plan.fingerprint, expected.fingerprint):
            raise NetworkEnforcerError("network enforcement plan does not match its host profile")
        return profile, expected

    def _plan_for_profile(self, profile: ManagedNetworkProfile) -> NetworkEnforcementPlan:
        return NetworkEnforcementPlan(
            enforcer_id=self.enforcer_id,
            revision=self.revision,
            profile=profile.profile,
            runtime=profile.runtime,
            hosts=profile.hosts,
            binding_kind=_binding_kind(profile.binding),
            binding_commitment=_binding_commitment(profile.binding),
        )


__all__ = [
    "DockerNetworkBinding",
    "FirejailNetworkBinding",
    "ManagedNetworkProfile",
    "NetworkEnforcementLease",
    "NetworkEnforcementPlan",
    "NetworkEnforcementRequest",
    "NetworkEnforcer",
    "NetworkEnforcerError",
    "PreconfiguredNetworkEnforcer",
    "normalize_network_allowlist_hosts",
    "normalize_network_host",
]
