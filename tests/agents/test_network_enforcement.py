"""Focused tests for the host-owned managed-network contract."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError

import pytest

from mmar_carl.network_enforcement import (
    DockerNetworkBinding,
    FirejailNetworkBinding,
    ManagedNetworkProfile,
    NetworkEnforcementLease,
    NetworkEnforcementPlan,
    NetworkEnforcementRequest,
    NetworkEnforcer,
    NetworkEnforcerError,
    PreconfiguredNetworkEnforcer,
)


def _docker_profile(
    *,
    profile: str = "prod-web",
    hosts: tuple[str, ...] = ("api.example.com",),
    network_name: str = "carl-egress-prod",
) -> ManagedNetworkProfile:
    return ManagedNetworkProfile(
        profile=profile,
        runtime="docker",
        hosts=hosts,
        binding=DockerNetworkBinding(network_name),
    )


def _enforcer(*profiles: ManagedNetworkProfile) -> PreconfiguredNetworkEnforcer:
    return PreconfiguredNetworkEnforcer(
        enforcer_id="host-egress",
        revision="2026-08-12",
        profiles=profiles or (_docker_profile(),),
    )


class TestNetworkEnforcementRequest:
    def test_normalizes_and_sorts_exact_hostname_and_ip_hosts(self) -> None:
        request = NetworkEnforcementRequest(
            runtime="docker",
            hosts=(" API.Example.COM. ", "2001:0DB8:0:0::1", "192.0.2.4"),
        )

        assert request.hosts == ("192.0.2.4", "2001:db8::1", "api.example.com")

    @pytest.mark.parametrize(
        "host",
        [
            "",
            "https://api.example.com",
            "api.example.com/path",
            "*.example.com",
            "api example.com",
            "api_example.com",
            "example.com:443",
            "[2001:db8::1]",
            "2001:db8::1%eth0",
            "127.000.0.1",
            "999.1.1.1",
            "éxample.com",
            "example.com..",
        ],
    )
    def test_rejects_non_exact_or_ambiguous_hosts(self, host: str) -> None:
        with pytest.raises(NetworkEnforcerError, match="allowlist"):
            NetworkEnforcementRequest(runtime="docker", hosts=(host,))

    def test_rejects_empty_allowlist(self) -> None:
        with pytest.raises(NetworkEnforcerError, match="at least one"):
            NetworkEnforcementRequest(runtime="docker", hosts=())

    def test_rejects_duplicates_after_normalization(self) -> None:
        with pytest.raises(NetworkEnforcerError, match="duplicate normalized"):
            NetworkEnforcementRequest(
                runtime="docker",
                hosts=("API.EXAMPLE.COM", "api.example.com."),
            )

    def test_rejects_runtime_without_a_typed_binding(self) -> None:
        with pytest.raises(NetworkEnforcerError, match="unsupported"):
            NetworkEnforcementRequest(
                runtime="e2b",  # type: ignore[arg-type]
                hosts=("api.example.com",),
            )

    def test_request_is_immutable(self) -> None:
        request = NetworkEnforcementRequest(
            runtime="docker",
            hosts=("api.example.com",),
        )
        with pytest.raises(FrozenInstanceError):
            request.runtime = "firejail"  # type: ignore[misc]


class TestTypedBindingsAndProfiles:
    @pytest.mark.parametrize(
        "network_name",
        [
            "",
            "-leading-option",
            "has space",
            "network/name",
            "x" * 129,
            "host",
            "bridge",
            "none",
        ],
    )
    def test_rejects_unsafe_docker_network_names(self, network_name: str) -> None:
        with pytest.raises(NetworkEnforcerError, match="network_name"):
            DockerNetworkBinding(network_name)

    @pytest.mark.parametrize(
        "interface_name",
        ["", "-leading-option", "has space", "eth0:1", "interface-name16"],
    )
    def test_rejects_unsafe_firejail_interface_names(self, interface_name: str) -> None:
        with pytest.raises(NetworkEnforcerError, match="interface_name"):
            FirejailNetworkBinding(interface_name)

    def test_rejects_binding_for_the_wrong_runtime(self) -> None:
        with pytest.raises(NetworkEnforcerError, match="requires 'docker_network'"):
            ManagedNetworkProfile(
                profile="wrong-binding",
                runtime="docker",
                hosts=("api.example.com",),
                binding=FirejailNetworkBinding("carl0"),
            )

    def test_firejail_profile_has_typed_interface_binding(self) -> None:
        profile = ManagedNetworkProfile(
            profile="linux-egress",
            runtime="firejail",
            hosts=("packages.example.org",),
            binding=FirejailNetworkBinding("carl-egress0"),
        )
        enforcer = _enforcer(profile)

        plan = enforcer.plan(
            NetworkEnforcementRequest(
                runtime="firejail",
                hosts=("packages.example.org",),
            )
        )

        assert plan.binding_kind == "firejail_interface"

    def test_lease_rejects_same_kind_binding_that_breaks_approved_commitment(self) -> None:
        enforcer = _enforcer()
        plan = enforcer.plan(
            NetworkEnforcementRequest(
                runtime="docker",
                hosts=("api.example.com",),
            )
        )

        with pytest.raises(NetworkEnforcerError, match="opaque commitment"):
            NetworkEnforcementLease(
                plan=plan,
                binding=DockerNetworkBinding("different-egress-route"),
                _owner_token=object(),
            )


class TestPreconfiguredNetworkEnforcer:
    def test_implements_runtime_protocol(self) -> None:
        assert isinstance(_enforcer(), NetworkEnforcer)

    def test_plan_is_stable_public_json_without_binding_value(self) -> None:
        enforcer = _enforcer(_docker_profile(hosts=("API.Example.COM.", "2001:db8::1")))
        request = NetworkEnforcementRequest(
            runtime="docker",
            hosts=("2001:0db8:0:0::1", "api.example.com"),
        )

        first = enforcer.plan(request)
        second = enforcer.plan(request)

        assert first == second
        assert first.as_dict() == {
            "binding_kind": "docker_network",
            "binding_commitment": "fd8d9355aa687337f4f3485b2d6687fc8fc3ff18ebc5e0f347988407dd67ef4a",
            "enforcer_id": "host-egress",
            "hosts": ["2001:db8::1", "api.example.com"],
            "profile": "prod-web",
            "revision": "2026-08-12",
            "runtime": "docker",
            "fingerprint": "16fca189523d7296d2309ddd1999cf9333b1173a087202207a4b7fced9ce88a4",
        }
        assert "carl-egress-prod" not in json.dumps(first.as_dict(), sort_keys=True)

    def test_binding_swap_changes_public_plan_and_approval_commitment(self) -> None:
        first = _enforcer(_docker_profile(network_name="carl-egress-a"))
        second = _enforcer(_docker_profile(network_name="carl-egress-b"))
        request = NetworkEnforcementRequest(
            runtime="docker",
            hosts=("api.example.com",),
        )

        first_plan = first.plan(request)
        second_plan = second.plan(request)

        assert first_plan.binding_commitment != second_plan.binding_commitment
        assert first_plan.fingerprint != second_plan.fingerprint
        assert "carl-egress-a" not in json.dumps(first_plan.as_dict())
        assert "carl-egress-b" not in json.dumps(second_plan.as_dict())

    def test_plan_requires_an_exact_runtime_and_host_set(self) -> None:
        enforcer = _enforcer()

        with pytest.raises(NetworkEnforcerError, match="exact request"):
            enforcer.plan(
                NetworkEnforcementRequest(
                    runtime="docker",
                    hosts=("other.example.com",),
                )
            )
        with pytest.raises(NetworkEnforcerError, match="exact request"):
            enforcer.plan(
                NetworkEnforcementRequest(
                    runtime="firejail",
                    hosts=("api.example.com",),
                )
            )

    def test_rejects_duplicate_exact_request_mappings(self) -> None:
        with pytest.raises(NetworkEnforcerError, match="duplicate managed-network"):
            _enforcer(
                _docker_profile(profile="first"),
                _docker_profile(profile="second", network_name="another-net"),
            )

    def test_rejects_reused_profile_name(self) -> None:
        with pytest.raises(NetworkEnforcerError, match="duplicate.*name"):
            _enforcer(
                _docker_profile(profile="same"),
                _docker_profile(
                    profile="same",
                    hosts=("other.example.com",),
                    network_name="other-net",
                ),
            )

    def test_constructor_detaches_from_mutable_profile_sequence(self) -> None:
        profiles = [_docker_profile()]
        enforcer = PreconfiguredNetworkEnforcer("host-egress", "v1", profiles)
        profiles.clear()

        plan = enforcer.plan(
            NetworkEnforcementRequest(
                runtime="docker",
                hosts=("api.example.com",),
            )
        )

        assert plan.profile == "prod-web"

    @pytest.mark.asyncio
    async def test_acquire_returns_opaque_matching_typed_lease_and_release_is_idempotent(
        self,
    ) -> None:
        enforcer = _enforcer()
        plan = enforcer.plan(
            NetworkEnforcementRequest(
                runtime="docker",
                hosts=("api.example.com",),
            )
        )

        lease = await enforcer.acquire(plan)

        assert lease.plan == plan
        assert lease.binding == DockerNetworkBinding("carl-egress-prod")
        with pytest.raises(TypeError):
            json.dumps(lease)
        await enforcer.release(lease)
        await enforcer.release(lease)

    @pytest.mark.asyncio
    async def test_acquire_rejects_another_enforcer_revision(self) -> None:
        profile = _docker_profile()
        first = PreconfiguredNetworkEnforcer("host-egress", "v1", (profile,))
        second = PreconfiguredNetworkEnforcer("host-egress", "v2", (profile,))
        first_plan = first.plan(
            NetworkEnforcementRequest(
                runtime="docker",
                hosts=("api.example.com",),
            )
        )

        with pytest.raises(NetworkEnforcerError, match="another enforcer revision"):
            await second.acquire(first_plan)

    @pytest.mark.asyncio
    async def test_acquire_rejects_plan_with_wrong_profile(self) -> None:
        enforcer = _enforcer()
        forged = NetworkEnforcementPlan(
            enforcer_id="host-egress",
            revision="2026-08-12",
            profile="different-profile",
            runtime="docker",
            hosts=("api.example.com",),
            binding_kind="docker_network",
            binding_commitment="0" * 64,
        )

        with pytest.raises(NetworkEnforcerError, match="does not match"):
            await enforcer.acquire(forged)

    @pytest.mark.asyncio
    async def test_release_rejects_lease_owned_by_another_enforcer(self) -> None:
        profile = _docker_profile()
        first = PreconfiguredNetworkEnforcer("first", "v1", (profile,))
        second = PreconfiguredNetworkEnforcer("second", "v1", (profile,))
        plan = first.plan(
            NetworkEnforcementRequest(
                runtime="docker",
                hosts=("api.example.com",),
            )
        )
        lease = await first.acquire(plan)

        with pytest.raises(NetworkEnforcerError, match="another enforcer"):
            await second.release(lease)
