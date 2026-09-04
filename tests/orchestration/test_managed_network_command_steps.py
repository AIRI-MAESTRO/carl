"""CommandStep integration tests for host-owned managed egress."""

from __future__ import annotations

import asyncio
import shutil
import tempfile
from pathlib import Path
from typing import Any, ClassVar

import pytest

from mmar_carl import (
    CommandApprovalRequest,
    CommandPolicy,
    CommandStepConfig,
    DockerNetworkBinding,
    ManagedNetworkProfile,
    NetworkEnforcementPlan,
    NetworkEnforcementRequest,
    PreconfiguredNetworkEnforcer,
    ReasoningChain,
    ReasoningContext,
    StepType,
    create_step,
)
from mmar_carl.models.llm_client_base import LLMClientBase
from mmar_carl.skill_runtime import (
    SKILL_RUNTIME_REGISTRY,
    RuntimeCapabilities,
    RuntimeRunResult,
    SkillRuntimeHandle,
)
from mmar_carl.step_executors import CommandStepExecutor


class _Client(LLMClientBase):
    async def get_response(self, prompt: str) -> str:
        return "unused"

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        return "unused"


class _ManagedDockerRuntime:
    name = "docker"
    capabilities = RuntimeCapabilities(
        isolation="container",
        wall_time="enforced",
        output_limit="enforced",
        network_none="enforced",
        network_allowlist="unsupported",
    )
    calls: ClassVar[list[tuple[str, Any]]] = []
    attach_binding = True
    fail_run = False

    @classmethod
    def reset(cls) -> None:
        cls.calls = []
        cls.attach_binding = True
        cls.fail_run = False

    async def prepare(self, skill: Any, workspace: Any, config: dict[str, Any]) -> SkillRuntimeHandle:
        type(self).calls.append(("prepare", dict(config)))
        root = Path(tempfile.mkdtemp(prefix="managed-egress-test-"))
        workspace_in = root / "in"
        workspace_out = root / "out"
        workspace_in.mkdir()
        workspace_out.mkdir()
        binding = config.get("_network_binding")
        return SkillRuntimeHandle(
            workspace_root=root,
            workspace_in=workspace_in,
            workspace_out=workspace_out,
            backend={
                "network_enforced": binding is not None,
                "network_binding": binding if type(self).attach_binding else None,
                "workspace_in_in_runtime": "/workspace/in",
                "workspace_out_in_runtime": "/workspace/out",
            },
        )

    async def run(self, handle: SkillRuntimeHandle, cmd: list[str], **kwargs: Any) -> RuntimeRunResult:
        type(self).calls.append(("run", list(cmd)))
        if type(self).fail_run:
            raise RuntimeError("runtime exploded")
        return RuntimeRunResult(
            stdout=b"managed-ok",
            stderr=b"",
            exit_code=0,
            duration_s=0.0,
        )

    async def cleanup(self, handle: SkillRuntimeHandle) -> None:
        type(self).calls.append(("cleanup", None))
        shutil.rmtree(handle.workspace_root)


class _SpyEnforcer:
    def __init__(
        self,
        *,
        revision: str = "r1",
        fail_acquire: bool = False,
        fail_release: bool = False,
        acquire_delay: float = 0.0,
    ) -> None:
        self.inner = PreconfiguredNetworkEnforcer(
            "host-egress",
            revision,
            (
                ManagedNetworkProfile(
                    profile="api-only",
                    runtime="docker",
                    hosts=("api.example.com",),
                    binding=DockerNetworkBinding("carl-egress-api"),
                ),
            ),
        )
        self.enforcer_id = self.inner.enforcer_id
        self.revision = self.inner.revision
        self.calls: list[str] = []
        self.fail_acquire = fail_acquire
        self.fail_release = fail_release
        self.acquire_delay = acquire_delay

    def plan(self, request: NetworkEnforcementRequest) -> NetworkEnforcementPlan:
        self.calls.append("plan")
        return self.inner.plan(request)

    async def acquire(self, plan: NetworkEnforcementPlan):
        self.calls.append("acquire")
        if self.acquire_delay:
            await asyncio.sleep(self.acquire_delay)
        if self.fail_acquire:
            raise RuntimeError("acquire failed")
        return await self.inner.acquire(plan)

    async def release(self, lease):
        self.calls.append("release")
        if self.fail_release:
            raise RuntimeError("release failed")
        await self.inner.release(lease)


@pytest.fixture(autouse=True)
def _managed_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    _ManagedDockerRuntime.reset()
    monkeypatch.setitem(SKILL_RUNTIME_REGISTRY, "docker", _ManagedDockerRuntime)


def _policy(*, approval: bool = False) -> CommandPolicy:
    executable_field = "approval_required_executables" if approval else "allowed_executables"
    return CommandPolicy(
        **{executable_field: frozenset({"safe-tool"})},
        allowed_runtimes=frozenset({"docker"}),
        allowed_networks=frozenset({"allowlist"}),
        allowed_network_hosts=frozenset({"api.example.com"}),
        require_approval_for_interpreters=False,
    )


async def _run(
    *,
    enforcer: Any,
    policy: CommandPolicy | None = None,
    approval: Any = None,
):
    config = CommandStepConfig(
        command=["safe-tool", "fetch"],
        runtime="docker",
        network="allowlist",
        network_allowlist=["api.example.com"],
    )
    chain = ReasoningChain(steps=[create_step(1, "managed", StepType.COMMAND, config=config)])
    context = ReasoningContext(
        outer_context="test",
        api=_Client(),
        command_policy=policy or _policy(),
        network_enforcer=enforcer,
        on_command_approval_requested=approval,
    )
    return (await chain.execute_async(context)).step_results[0]


@pytest.mark.asyncio
async def test_exact_profile_is_fingerprinted_acquired_attached_and_released() -> None:
    enforcer = _SpyEnforcer()

    result = await _run(enforcer=enforcer)

    assert result.success is True
    assert enforcer.calls == ["plan", "acquire", "release"]
    prepare = _ManagedDockerRuntime.calls[0][1]
    assert prepare["_network_binding"] == DockerNetworkBinding("carl-egress-api")
    assert result.result_data["network_enforced"] is True
    public = result.result_data["network_enforcement"]
    assert public["profile"] == "api-only"
    assert public["hosts"] == ["api.example.com"]
    assert "carl-egress-api" not in str(public)


@pytest.mark.asyncio
async def test_approval_covers_public_enforcer_plan_but_not_private_binding() -> None:
    enforcer = _SpyEnforcer()
    requests: list[CommandApprovalRequest] = []

    def approve(request: CommandApprovalRequest) -> bool:
        requests.append(request)
        return True

    result = await _run(enforcer=enforcer, policy=_policy(approval=True), approval=approve)

    assert result.success is True
    managed = requests[0].resources["network_enforcement"]
    assert managed == result.result_data["network_enforcement"]
    assert "carl-egress-api" not in str(requests[0].model_dump())


@pytest.mark.asyncio
async def test_enforcer_revision_changes_command_approval_fingerprint() -> None:
    requests: list[CommandApprovalRequest] = []

    def approve(request: CommandApprovalRequest) -> bool:
        requests.append(request)
        return True

    first = await _run(
        enforcer=_SpyEnforcer(revision="r1"),
        policy=_policy(approval=True),
        approval=approve,
    )
    second = await _run(
        enforcer=_SpyEnforcer(revision="r2"),
        policy=_policy(approval=True),
        approval=approve,
    )

    assert first.success and second.success
    assert requests[0].fingerprint != requests[1].fingerprint


@pytest.mark.asyncio
async def test_missing_enforcer_fails_strict_preflight_before_runtime_prepare() -> None:
    result = await _run(enforcer=None)

    assert result.success is False
    assert "cannot strictly enforce" in result.error_message
    assert _ManagedDockerRuntime.calls == []


@pytest.mark.asyncio
async def test_native_e2b_allowlist_does_not_call_host_binding_enforcer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _NativeE2BRuntime(_ManagedDockerRuntime):
        name = "e2b"
        capabilities = RuntimeCapabilities(
            isolation="microvm",
            wall_time="enforced",
            output_limit="enforced",
            network_none="enforced",
            network_allowlist="enforced",
        )
        calls: ClassVar[list[tuple[str, Any]]] = []

        async def prepare(
            self,
            skill: Any,
            workspace: Any,
            config: dict[str, Any],
        ) -> SkillRuntimeHandle:
            handle = await super().prepare(skill, workspace, config)
            handle.backend["network_enforced"] = True
            return handle

    _NativeE2BRuntime.calls = []
    monkeypatch.setitem(SKILL_RUNTIME_REGISTRY, "e2b", _NativeE2BRuntime)
    enforcer = _SpyEnforcer()
    policy = CommandPolicy(
        allowed_executables=frozenset({"safe-tool"}),
        allowed_runtimes=frozenset({"e2b"}),
        allowed_networks=frozenset({"allowlist"}),
        allowed_network_hosts=frozenset({"api.example.com"}),
        require_approval_for_interpreters=False,
    )
    config = CommandStepConfig(
        command=["safe-tool", "fetch"],
        runtime="e2b",
        network="allowlist",
        network_allowlist=["api.example.com"],
    )
    chain = ReasoningChain(steps=[create_step(1, "native-e2b", StepType.COMMAND, config=config)])
    context = ReasoningContext(
        outer_context="test",
        api=_Client(),
        command_policy=policy,
        network_enforcer=enforcer,
    )

    result = (await chain.execute_async(context)).step_results[0]

    assert result.success is True
    assert enforcer.calls == []
    assert [name for name, _ in _NativeE2BRuntime.calls] == [
        "prepare",
        "run",
        "cleanup",
    ]
    prepare_config = _NativeE2BRuntime.calls[0][1]
    assert prepare_config["network"] == "allowlist"
    assert prepare_config["network_allowlist"] == ["api.example.com"]
    assert "_network_binding" not in prepare_config
    assert result.result_data["network_enforced"] is True
    assert result.result_data["network_enforcement"] is None


@pytest.mark.asyncio
async def test_denied_executable_does_not_call_host_network_enforcer() -> None:
    enforcer = _SpyEnforcer()
    denied_policy = CommandPolicy(
        allowed_executables=frozenset(),
        allowed_runtimes=frozenset({"docker"}),
        allowed_networks=frozenset({"allowlist"}),
        allowed_network_hosts=frozenset({"api.example.com"}),
    )

    result = await _run(enforcer=enforcer, policy=denied_policy)

    assert result.success is False
    assert "exact allowlist" in result.error_message
    assert enforcer.calls == []
    assert _ManagedDockerRuntime.calls == []


@pytest.mark.asyncio
async def test_unmatched_host_profile_fails_before_acquire_or_runtime_prepare() -> None:
    enforcer = PreconfiguredNetworkEnforcer("host-egress", "r1", ())

    result = await _run(enforcer=enforcer)

    assert result.success is False
    assert "rejected by the host" in result.error_message
    assert _ManagedDockerRuntime.calls == []


@pytest.mark.asyncio
async def test_custom_enforcer_cannot_attest_a_different_host_set() -> None:
    class _WrongPlanEnforcer(_SpyEnforcer):
        def plan(self, request: NetworkEnforcementRequest) -> NetworkEnforcementPlan:
            self.calls.append("plan")
            return NetworkEnforcementPlan(
                enforcer_id=self.enforcer_id,
                revision=self.revision,
                profile="wrong-hosts",
                runtime="docker",
                hosts=("other.example.com",),
                binding_kind="docker_network",
                binding_commitment="0" * 64,
            )

    enforcer = _WrongPlanEnforcer()

    result = await _run(enforcer=enforcer)

    assert result.success is False
    assert "does not exactly match" in result.error_message
    assert enforcer.calls == ["plan"]
    assert _ManagedDockerRuntime.calls == []


@pytest.mark.asyncio
async def test_acquire_failure_does_not_prepare_runtime() -> None:
    enforcer = _SpyEnforcer(fail_acquire=True)

    result = await _run(enforcer=enforcer)

    assert result.success is False
    assert "managed network acquisition failed" in result.error_message
    assert enforcer.calls == ["plan", "acquire"]
    assert _ManagedDockerRuntime.calls == []


@pytest.mark.asyncio
async def test_acquire_is_bounded_by_host_policy_timeout() -> None:
    enforcer = _SpyEnforcer(acquire_delay=0.1)
    policy = _policy().model_copy(update={"network_enforcer_timeout": 0.01})

    result = await _run(enforcer=enforcer, policy=policy)

    assert result.success is False
    assert "managed network acquisition timed out" in result.error_message
    assert _ManagedDockerRuntime.calls == []


@pytest.mark.asyncio
async def test_runtime_binding_mismatch_fails_and_still_releases_lease() -> None:
    enforcer = _SpyEnforcer()
    _ManagedDockerRuntime.attach_binding = False

    result = await _run(enforcer=enforcer)

    assert result.success is False
    assert "exact host-issued network binding" in result.error_message
    assert enforcer.calls == ["plan", "acquire", "release"]
    assert [name for name, _ in _ManagedDockerRuntime.calls] == ["prepare", "cleanup"]


@pytest.mark.asyncio
async def test_runtime_failure_still_cleans_runtime_then_releases_network() -> None:
    enforcer = _SpyEnforcer()
    _ManagedDockerRuntime.fail_run = True

    result = await _run(enforcer=enforcer)

    assert result.success is False
    assert [name for name, _ in _ManagedDockerRuntime.calls] == [
        "prepare",
        "run",
        "cleanup",
    ]
    assert enforcer.calls == ["plan", "acquire", "release"]


@pytest.mark.asyncio
async def test_release_failure_is_not_reported_as_success() -> None:
    enforcer = _SpyEnforcer(fail_release=True)

    result = await _run(enforcer=enforcer)

    assert result.success is False
    assert "managed network cleanup failed" in result.error_message
    assert [name for name, _ in _ManagedDockerRuntime.calls] == [
        "prepare",
        "run",
        "cleanup",
    ]


@pytest.mark.asyncio
async def test_cancellation_during_release_finishes_release_and_is_preserved() -> None:
    class _BlockingReleaseEnforcer(_SpyEnforcer):
        def __init__(self) -> None:
            super().__init__()
            self.release_started = asyncio.Event()
            self.allow_release = asyncio.Event()
            self.release_finished = asyncio.Event()

        async def release(self, lease: Any) -> None:
            self.calls.append("release")
            self.release_started.set()
            await self.allow_release.wait()
            await self.inner.release(lease)
            self.release_finished.set()

    enforcer = _BlockingReleaseEnforcer()
    task = asyncio.create_task(_run(enforcer=enforcer))
    try:
        await asyncio.wait_for(enforcer.release_started.wait(), timeout=1.0)
        task.cancel()
        # Let cancellation reach the executor while release remains blocked.
        await asyncio.sleep(0)
        enforcer.allow_release.set()

        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1.0)
    finally:
        enforcer.allow_release.set()
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    assert enforcer.release_finished.is_set()
    assert enforcer.calls == ["plan", "acquire", "release"]
    assert [name for name, _ in _ManagedDockerRuntime.calls] == [
        "prepare",
        "run",
        "cleanup",
    ]


@pytest.mark.asyncio
async def test_repeated_cancellation_during_runtime_cleanup_still_releases_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cleanup_started = asyncio.Event()
    allow_cleanup = asyncio.Event()
    cleanup_finished = asyncio.Event()

    async def blocking_cleanup(
        self: _ManagedDockerRuntime,
        handle: SkillRuntimeHandle,
    ) -> None:
        type(self).calls.append(("cleanup", None))
        cleanup_started.set()
        await allow_cleanup.wait()
        shutil.rmtree(handle.workspace_root)
        cleanup_finished.set()

    monkeypatch.setattr(_ManagedDockerRuntime, "cleanup", blocking_cleanup)
    enforcer = _SpyEnforcer()
    config = CommandStepConfig(
        command=["safe-tool", "fetch"],
        runtime="docker",
        network="allowlist",
        network_allowlist=["api.example.com"],
    )
    step = create_step(1, "managed", StepType.COMMAND, config=config)
    context = ReasoningContext(
        outer_context="test",
        api=_Client(),
        command_policy=_policy(),
        network_enforcer=enforcer,
    )
    # Exercise the executor task directly so Task.cancelling() observes the
    # exact task that consumes and defers each cancellation request.
    task = asyncio.create_task(CommandStepExecutor().execute(step, context))
    try:
        await asyncio.wait_for(cleanup_started.wait(), timeout=1.0)
        task.cancel()
        # Wait until the executor has consumed and deferred the first request,
        # then issue a genuinely distinct second cancel while cleanup remains
        # blocked. This avoids relying on two same-turn cancel() calls being
        # delivered separately by the event loop.
        for _ in range(100):
            await asyncio.sleep(0)
            if task.cancelling() == 0:
                break
        assert task.cancelling() == 0
        task.cancel()
        await asyncio.sleep(0)
        allow_cleanup.set()

        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1.0)
    finally:
        allow_cleanup.set()
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    assert cleanup_finished.is_set()
    assert enforcer.calls == ["plan", "acquire", "release"]
    assert enforcer.calls.count("release") == 1
    assert [name for name, _ in _ManagedDockerRuntime.calls] == [
        "prepare",
        "run",
        "cleanup",
    ]


@pytest.mark.asyncio
async def test_parallel_contexts_share_runtime_only_enforcer() -> None:
    enforcer = _SpyEnforcer()
    config = CommandStepConfig(
        command=["safe-tool", "fetch"],
        runtime="docker",
        network="allowlist",
        network_allowlist=["api.example.com"],
    )
    chain = ReasoningChain(
        steps=[
            create_step(1, "one", StepType.COMMAND, config=config),
            create_step(2, "two", StepType.COMMAND, config=config),
        ],
        max_workers=2,
    )
    context = ReasoningContext(
        outer_context="test",
        api=_Client(),
        command_policy=_policy(),
        network_enforcer=enforcer,
    )

    result = await chain.execute_async(context)

    assert result.success is True
    assert enforcer.calls.count("plan") == 2
    assert enforcer.calls.count("acquire") == 2
    assert enforcer.calls.count("release") == 2


@pytest.mark.asyncio
async def test_replay_context_preserves_runtime_only_enforcer() -> None:
    enforcer = _SpyEnforcer()
    config = CommandStepConfig(
        command=["safe-tool", "fetch"],
        runtime="docker",
        network="allowlist",
        network_allowlist=["api.example.com"],
    )
    chain = ReasoningChain(steps=[create_step(1, "managed", StepType.COMMAND, config=config)])

    def context() -> ReasoningContext:
        return ReasoningContext(
            outer_context="test",
            api=_Client(),
            command_policy=_policy(),
            network_enforcer=enforcer,
        )

    original = await chain.execute_async(context())
    replayed = await chain.replay(original.trace, context())

    assert original.success is True
    assert replayed.success is True
    assert enforcer.calls.count("acquire") == 2
