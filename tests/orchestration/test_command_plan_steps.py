"""Integration tests for typed command planning and planned execution.

These tests exercise the complete trust boundary with a scripted LLM and a
recording runtime.  The model may select a host-declared capability and fill
its typed argument object, but it never supplies an executable or raw argv.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from mmar_carl import (
    CommandApprovalRequest,
    CommandCapability,
    CommandCapabilityRegistry,
    CommandPlanStepConfig,
    CommandPlanStepDescription,
    CommandPolicy,
    CommandStepConfig,
    CommandStepDescription,
    ReasoningChain,
    ReasoningContext,
    StepDescription,
    StepType,
)
from mmar_carl.models.llm_client_base import LLMClientBase
from mmar_carl.skill_runtime import (
    RuntimeCapabilities,
    RuntimeRunResult,
    SkillRuntimeHandle,
    register_skill_runtime,
)
from mmar_carl.step_executors import CommandPlanStepExecutor


class _TextArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    repeat: int


class _ScriptedClient(LLMClientBase):
    def __init__(
        self,
        response: str,
        *,
        usage: dict[str, int] | None = None,
        delay: float = 0.0,
    ) -> None:
        self.response = response
        self.usage = usage or {}
        self.delay = delay
        self.prompts: list[str] = []

    @property
    def model_name(self) -> str:
        return "scripted-command-planner"

    async def get_response(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if self.delay:
            await asyncio.sleep(self.delay)
        return self.response

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        return await self.get_response(prompt)

    async def get_response_with_usage(
        self,
        prompt: str,
        retries: int = 3,
    ) -> tuple[str, dict[str, int]]:
        return await self.get_response(prompt), dict(self.usage)


class _RecordingRuntime:
    name = "command_plan_recording"
    capabilities = RuntimeCapabilities(
        isolation="unknown",
        wall_time="enforced",
        output_limit="enforced",
        cpu_limit="enforced",
        memory_limit="enforced",
        pids_limit="enforced",
        network_none="enforced",
        network_allowlist="enforced",
        workspace_files="enforced",
        artifact_output_limit="enforced",
    )
    calls: ClassVar[list[dict[str, Any]]] = []

    @classmethod
    def reset(cls) -> None:
        cls.calls = []

    async def prepare(self, skill: Any, workspace: Any, config: dict[str, Any]) -> SkillRuntimeHandle:
        root = Path(tempfile.mkdtemp(prefix="carl_command_plan_"))
        workspace_in = root / "in"
        workspace_out = root / "out"
        workspace_in.mkdir()
        workspace_out.mkdir()
        type(self).calls.append({"kind": "prepare", "config": dict(config)})
        return SkillRuntimeHandle(
            workspace_root=root,
            workspace_in=workspace_in,
            workspace_out=workspace_out,
            backend={
                "network_enforced": True,
                "workspace_in_in_runtime": str(workspace_in),
                "workspace_out_in_runtime": str(workspace_out),
            },
        )

    async def run(
        self,
        handle: SkillRuntimeHandle,
        cmd: Sequence[str],
        *,
        env: dict[str, str] | None = None,
        stdin: bytes | None = None,
        timeout: float | None = None,
        cwd: str | None = None,
    ) -> RuntimeRunResult:
        type(self).calls.append(
            {
                "kind": "run",
                "cmd": list(cmd),
                "env": dict(env or {}),
                "stdin": stdin,
                "timeout": timeout,
                "cwd": cwd,
            }
        )
        return RuntimeRunResult(stdout=b"planned-ok", stderr=b"", exit_code=0, duration_s=0.0)

    async def cleanup(self, handle: SkillRuntimeHandle) -> None:
        type(self).calls.append({"kind": "cleanup"})
        shutil.rmtree(handle.workspace_root)


@pytest.fixture(autouse=True)
def _recording_runtime() -> None:
    register_skill_runtime(_RecordingRuntime.name, _RecordingRuntime)
    _RecordingRuntime.reset()
    yield
    _RecordingRuntime.reset()


def _response(*, text: str = "hello", repeat: int = 2) -> str:
    return json.dumps(
        {
            "capability_id": "text.print",
            "arguments": {"text": text, "repeat": repeat},
        }
    )


def _capability(
    *,
    capability_id: str = "text.print",
    revision: str = "r1",
    executable: str = "safe-tool",
    calls: list[_TextArguments] | None = None,
) -> CommandCapability:
    def build(arguments: BaseModel) -> Sequence[str]:
        assert isinstance(arguments, _TextArguments)
        if calls is not None:
            calls.append(arguments)
        return (arguments.text, str(arguments.repeat))

    return CommandCapability(
        capability_id=capability_id,
        description="Print validated text a bounded number of times",
        executable=executable,
        static_args=("print",),
        argument_model=_TextArguments,
        argv_builder=build,
        revision=revision,
    )


def _registry(
    *,
    calls: list[_TextArguments] | None = None,
    revision: str = "r1",
    executable: str = "safe-tool",
    max_prompt_bytes: int = 128_000,
    max_input_value_bytes: int = 32_000,
    max_response_bytes: int = 64_000,
) -> CommandCapabilityRegistry:
    return CommandCapabilityRegistry(
        [_capability(calls=calls, revision=revision, executable=executable)],
        max_prompt_bytes=max_prompt_bytes,
        max_input_value_bytes=max_input_value_bytes,
        max_response_bytes=max_response_bytes,
    )


def _policy(
    *,
    executable: str = "safe-tool",
    runtime: str = _RecordingRuntime.name,
    allow_planned_local: bool = False,
    require_approval_for_planned: bool = True,
) -> CommandPolicy:
    return CommandPolicy(
        allowed_executables=frozenset({executable}),
        allowed_runtimes=frozenset({runtime}),
        allowed_networks=frozenset({"none"}),
        allow_planned_local=allow_planned_local,
        require_approval_for_planned=require_approval_for_planned,
        require_approval_for_interpreters=False,
    )


def _planner(
    *,
    capability_ids: list[str] | None = None,
    input_mapping: dict[str, str] | None = None,
    timeout: float = 30.0,
) -> CommandPlanStepDescription:
    return CommandPlanStepDescription(
        number=1,
        title="Plan a typed command",
        config=CommandPlanStepConfig(
            instruction="Print the supplied text twice.",
            capability_ids=capability_ids or ["text.print"],
            input_mapping=input_mapping or {},
        ),
        timeout=timeout,
    )


def _planned_command(
    *,
    number: int = 2,
    plan_source: str = "$steps.1.result_data.plan",
    dependencies: list[int] | None = None,
    runtime: str = _RecordingRuntime.name,
) -> CommandStepDescription:
    return CommandStepDescription(
        number=number,
        title="Run the typed plan",
        dependencies=[1] if dependencies is None else dependencies,
        config=CommandStepConfig(
            plan_source=plan_source,
            planned_capability_ids=["text.print"],
            runtime=runtime,
        ),
    )


def _context(
    client: LLMClientBase,
    *,
    registry: CommandCapabilityRegistry | None = None,
    policy: CommandPolicy | None = None,
    approval: Any = None,
    outer_context: str = "input",
) -> ReasoningContext:
    return ReasoningContext(
        outer_context=outer_context,
        api=client,
        model="mock",
        command_capability_registry=registry,
        command_policy=policy,
        on_command_approval_requested=approval,
    )


def _run_calls() -> list[dict[str, Any]]:
    return [call for call in _RecordingRuntime.calls if call["kind"] == "run"]


@pytest.mark.asyncio
async def test_mock_llm_plan_runs_exact_host_built_argv_once_and_reports_usage() -> None:
    builder_calls: list[_TextArguments] = []
    approvals: list[CommandApprovalRequest] = []
    client = _ScriptedClient(
        _response(text="dynamic value", repeat=3),
        usage={"prompt": 31, "completion": 9, "total": 40},
    )
    context = _context(
        client,
        registry=_registry(calls=builder_calls),
        policy=_policy(),
        approval=lambda request: approvals.append(request) or True,
    )
    chain = ReasoningChain(steps=[_planner(), _planned_command()], max_workers=1)

    result = await chain.execute_async(context)

    assert result.success is True
    assert len(builder_calls) == 1
    assert builder_calls[0].model_dump() == {"text": "dynamic value", "repeat": 3}
    assert [call["cmd"] for call in _run_calls()] == [
        ["safe-tool", "print", "dynamic value", "3"]
    ]
    assert len(approvals) == 1
    assert approvals[0].argv == ("safe-tool", "print", "dynamic value", "3")
    assert result.step_results[0].token_usage == {"prompt": 31, "completion": 9, "total": 40}
    assert result.step_results[0].model == "scripted-command-planner"
    assert result.step_results[1].result_data["command_source"] == "planned"
    assert result.step_results[1].result_data["capability_id"] == "text.print"


@pytest.mark.asyncio
async def test_prompt_exposes_only_selected_manifest_and_two_field_output_envelope() -> None:
    other = _capability(capability_id="text.other", revision="r2")
    registry = CommandCapabilityRegistry([_capability(), other])
    client = _ScriptedClient(_response())
    context = _context(client, registry=registry, outer_context='{"secret": "value"}')
    planner = _planner(input_mapping={"payload": "$outer_context"})

    result = await ReasoningChain(steps=[planner], max_workers=1).execute_async(context)

    assert result.success is True
    assert len(client.prompts) == 1
    request = json.loads(client.prompts[0].split("\n\n", 1)[1])
    assert request["inputs"] == {"payload": {"secret": "value"}}
    assert [item["capability_id"] for item in request["capabilities"]] == ["text.print"]
    assert set(request["capabilities"][0]) == {
        "capability_id",
        "description",
        "arguments_schema",
    }
    assert set(request["required_output"]) == {"capability_id", "arguments"}
    assert "safe-tool" not in client.prompts[0]
    assert "text.other" not in client.prompts[0]


@pytest.mark.parametrize(
    "response",
    [
        "not JSON",
        '{"command":["/bin/sh","-c","id"]}',
        (
            '{"capability_id":"text.print","arguments":{"text":"x","repeat":1},'
            '"command":["/bin/sh"]}'
        ),
        (
            '{"capability_id":"text.print","arguments":{"text":"x","repeat":1},'
            '"executable":"/bin/sh"}'
        ),
    ],
)
@pytest.mark.asyncio
async def test_planner_rejects_malformed_extra_and_raw_command_responses(response: str) -> None:
    builder_calls: list[_TextArguments] = []
    client = _ScriptedClient(response)
    context = _context(client, registry=_registry(calls=builder_calls))

    result = await ReasoningChain(steps=[_planner()], max_workers=1).execute_async(context)

    assert result.success is False
    assert "invalid typed plan" in result.step_results[0].error_message
    assert builder_calls == []
    assert _RecordingRuntime.calls == []


@pytest.mark.parametrize(
    "response",
    [
        '{"capability_id":"text.print","arguments":{"text":"x","repeat":"2"}}',
        (
            '{"capability_id":"text.print",'
            '"arguments":{"text":"x","repeat":2,"unexpected":true}}'
        ),
    ],
)
@pytest.mark.asyncio
async def test_planner_applies_strict_argument_types_and_forbids_extras(response: str) -> None:
    builder_calls: list[_TextArguments] = []
    context = _context(_ScriptedClient(response), registry=_registry(calls=builder_calls))

    result = await ReasoningChain(steps=[_planner()], max_workers=1).execute_async(context)

    assert result.success is False
    assert "invalid typed plan" in result.step_results[0].error_message
    assert builder_calls == []
    assert _RecordingRuntime.calls == []


@pytest.mark.asyncio
async def test_missing_or_unknown_registry_fails_before_provider_or_runtime() -> None:
    missing_client = _ScriptedClient(_response())
    missing = await ReasoningChain(steps=[_planner()], max_workers=1).execute_async(
        _context(missing_client)
    )
    assert missing.success is False
    assert "host-owned CommandCapabilityRegistry" in missing.step_results[0].error_message
    assert missing_client.prompts == []

    unknown_client = _ScriptedClient(_response())
    unknown = await ReasoningChain(
        steps=[_planner(capability_ids=["text.missing"])],
        max_workers=1,
    ).execute_async(_context(unknown_client, registry=_registry()))
    assert unknown.success is False
    assert "unknown capability_id" in unknown.step_results[0].error_message
    assert unknown_client.prompts == []
    assert _RecordingRuntime.calls == []


@pytest.mark.asyncio
async def test_planned_command_without_registry_fails_before_runtime_prepare() -> None:
    source_registry = _registry()
    record = source_registry.validate_plan(json.loads(_response()))
    context = _context(_ScriptedClient("unused"), policy=_policy())
    context.metadata["plan"] = record.model_dump(mode="json")

    result = await ReasoningChain(
        steps=[_planned_command(number=1, plan_source="$metadata.plan", dependencies=[])],
        max_workers=1,
    ).execute_async(context)

    assert result.success is False
    assert "host-owned CommandCapabilityRegistry" in result.step_results[0].error_message
    assert _RecordingRuntime.calls == []


def test_command_config_requires_exactly_one_static_or_planned_source() -> None:
    with pytest.raises(ValidationError, match="exactly one of command or plan_source"):
        CommandStepConfig()
    with pytest.raises(ValidationError, match="exactly one of command or plan_source"):
        CommandStepConfig(
            command=["safe-tool"],
            plan_source="$metadata.plan",
            planned_capability_ids=["text.print"],
        )
    with pytest.raises(ValidationError, match="requires planned_capability_ids"):
        CommandStepConfig(plan_source="$metadata.plan")
    with pytest.raises(ValidationError, match="externally supplied record"):
        CommandStepConfig(
            plan_source="$history[-1]",
            planned_capability_ids=["text.print"],
        )
    with pytest.raises(ValidationError, match="cannot append untyped input_mapping"):
        CommandStepConfig(
            plan_source="$metadata.plan",
            planned_capability_ids=["text.print"],
            input_mapping={"unsafe": "$outer_context"},
        )
    with pytest.raises(ValidationError, match="only valid with plan_source"):
        CommandStepConfig(command=["safe-tool"], planned_capability_ids=["text.print"])


@pytest.mark.parametrize(
    ("consumer", "message"),
    [
        (
            _planned_command(dependencies=[]),
            "must depend on CommandPlanStep 1",
        ),
        (
            _planned_command(plan_source="$steps.1.plan"),
            r"must use '\$steps\.<number>\.result_data\.plan'",
        ),
        (
            CommandStepDescription(
                number=2,
                title="Capability mismatch",
                dependencies=[1],
                config=CommandStepConfig(
                    plan_source="$steps.1.result_data.plan",
                    planned_capability_ids=["text.other"],
                    runtime=_RecordingRuntime.name,
                ),
            ),
            "must accept every capability offered",
        ),
    ],
)
def test_chain_rejects_missing_planner_dependency_malformed_source_and_id_mismatch(
    consumer: CommandStepDescription,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        ReasoningChain(steps=[_planner(), consumer], max_workers=1)


def test_chain_rejects_plan_source_that_points_to_a_non_planner_step() -> None:
    static = CommandStepDescription(
        number=1,
        title="Not a planner",
        config=CommandStepConfig(command=["safe-tool"], runtime=_RecordingRuntime.name),
    )
    consumer = _planned_command()

    with pytest.raises(ValueError, match="must reference an existing CommandPlanStep"):
        ReasoningChain(steps=[static, consumer], max_workers=1)


@pytest.mark.parametrize("kind", ["forged_arguments", "stale_registry", "forged_fingerprint"])
@pytest.mark.asyncio
async def test_forged_or_stale_plan_is_rejected_before_builder_and_prepare(kind: str) -> None:
    original = _registry()
    record = original.validate_plan(json.loads(_response(text="trusted", repeat=1)))
    dumped = record.model_dump(mode="json")
    builder_calls: list[_TextArguments] = []
    if kind == "forged_arguments":
        dumped["arguments"]["text"] = "forged"
        runtime_registry = _registry(calls=builder_calls)
    elif kind == "stale_registry":
        runtime_registry = _registry(calls=builder_calls, revision="r2")
    else:
        dumped["capability_fingerprint"] = "0" * 64
        runtime_registry = _registry(calls=builder_calls)
    context = _context(
        _ScriptedClient("unused"),
        registry=runtime_registry,
        policy=_policy(),
        approval=lambda _request: True,
    )
    context.metadata["plan"] = dumped

    result = await ReasoningChain(
        steps=[_planned_command(number=1, plan_source="$metadata.plan", dependencies=[])],
        max_workers=1,
    ).execute_async(context)

    assert result.success is False
    assert "failed to build runtime invocation" in result.step_results[0].error_message
    assert builder_calls == []
    assert _RecordingRuntime.calls == []


@pytest.mark.asyncio
async def test_static_command_regression_does_not_require_registry_or_planned_approval() -> None:
    context = _context(
        _ScriptedClient("unused"),
        policy=_policy(require_approval_for_planned=True),
    )
    step = CommandStepDescription(
        number=1,
        title="Static command",
        config=CommandStepConfig(command=["safe-tool", "static"], runtime=_RecordingRuntime.name),
    )

    result = await ReasoningChain(steps=[step], max_workers=1).execute_async(context)

    assert result.success is True
    assert [call["cmd"] for call in _run_calls()] == [["safe-tool", "static"]]
    assert result.step_results[0].result_data["command_source"] == "static"
    assert result.step_results[0].result_data["approval"] == "not_required"


@pytest.mark.asyncio
async def test_planned_local_execution_is_denied_before_builder_or_runtime_by_default() -> None:
    executable = "/usr/bin/printf"
    builder_calls: list[_TextArguments] = []
    registry = _registry(calls=builder_calls, executable=executable)
    record = registry.validate_plan(json.loads(_response()))
    context = _context(
        _ScriptedClient("unused"),
        registry=registry,
        policy=_policy(executable=executable, runtime="local"),
    )
    context.metadata["plan"] = record.model_dump(mode="json")

    result = await ReasoningChain(
        steps=[
            _planned_command(
                number=1,
                plan_source="$metadata.plan",
                dependencies=[],
                runtime="local",
            )
        ],
        max_workers=1,
    ).execute_async(context)

    assert result.success is False
    assert "LLM-planned host execution is not allowed" in result.step_results[0].error_message
    assert builder_calls == []
    assert _RecordingRuntime.calls == []


@pytest.mark.asyncio
async def test_planned_execution_always_requests_approval_and_public_event_redacts_arguments() -> None:
    secret = "model-selected-secret"
    approvals: list[CommandApprovalRequest] = []
    events: list[tuple[str, dict[str, Any]]] = []
    context = _context(
        _ScriptedClient(_response(text=secret, repeat=1)),
        registry=_registry(),
        policy=_policy(),
        approval=lambda request: approvals.append(request) or True,
    )
    context.on_step_event = lambda _number, kind, payload: events.append((kind, payload))

    result = await ReasoningChain(
        steps=[_planner(), _planned_command()],
        max_workers=1,
    ).execute_async(context)

    assert result.success is True
    assert len(approvals) == 1
    request = approvals[0]
    assert request.source == "planned"
    assert request.argv == ("safe-tool", "print", secret, "1")
    assert request.argv_preview == (
        "safe-tool",
        "<static-arg-1>",
        "<dynamic-arg-1>",
        "<dynamic-arg-2>",
    )
    planner_result = result.step_results[0].result_data
    assert request.capability_id == "text.print"
    assert request.capability_revision == planner_result["capability_revision"]
    assert request.capability_fingerprint == planner_result["capability_fingerprint"]
    assert request.arguments_sha256 == planner_result["arguments_sha256"]
    public = next(payload for kind, payload in events if kind == "command.approval_requested")
    assert "argv" not in public
    assert "environment" not in public
    assert "stdin" not in public
    assert "artifact_inputs" not in public
    assert secret not in repr(public)
    assert public["source"] == "planned"
    assert public["capability_id"] == "text.print"
    assert public["capability_revision"] == planner_result["capability_revision"]
    assert public["capability_fingerprint"] == planner_result["capability_fingerprint"]
    assert public["arguments_sha256"] == planner_result["arguments_sha256"]

    complete_request = {
        **request.model_dump(),
        "argv": request.argv,
        "environment": request.environment,
        "stdin": request.stdin,
        "artifact_inputs": request.artifact_inputs,
    }
    with pytest.raises(ValidationError, match="complete command capability provenance"):
        CommandApprovalRequest.model_validate(
            {**complete_request, "capability_fingerprint": None}
        )
    with pytest.raises(ValidationError, match="static approval cannot carry"):
        CommandApprovalRequest.model_validate({**complete_request, "source": "static"})


@pytest.mark.asyncio
async def test_missing_planned_approval_denies_before_prepare() -> None:
    builder_calls: list[_TextArguments] = []
    context = _context(
        _ScriptedClient(_response()),
        registry=_registry(calls=builder_calls),
        policy=_policy(),
    )

    result = await ReasoningChain(
        steps=[_planner(), _planned_command()],
        max_workers=1,
    ).execute_async(context)

    assert result.success is False
    assert result.step_results[0].success is True
    assert "not approved" in result.step_results[1].error_message
    assert builder_calls == []
    assert _RecordingRuntime.calls == []


@pytest.mark.asyncio
async def test_policy_denies_planned_executable_before_builder() -> None:
    builder_calls: list[_TextArguments] = []
    registry = _registry(calls=builder_calls, executable="denied-tool")
    record = registry.validate_plan(json.loads(_response()))
    context = _context(
        _ScriptedClient("unused"),
        registry=registry,
        policy=_policy(executable="different-tool"),
        approval=lambda _request: True,
    )
    context.metadata["plan"] = record.model_dump(mode="json")

    result = await ReasoningChain(
        steps=[_planned_command(number=1, plan_source="$metadata.plan", dependencies=[])],
        max_workers=1,
    ).execute_async(context)

    assert result.success is False
    assert "exact allowlist" in result.step_results[0].error_message
    assert builder_calls == []
    assert _RecordingRuntime.calls == []


def test_disabling_planned_approval_is_rejected_at_policy_construction() -> None:
    with pytest.raises(ValidationError, match="Input should be True"):
        _policy(require_approval_for_planned=False)


def test_registry_is_runtime_only_in_context_serialization_and_snapshot() -> None:
    registry = _registry()
    context = _context(_ScriptedClient(_response()), registry=registry, policy=_policy())

    dumped = context.model_dump()
    snapshot = context.snapshot().model_dump()

    assert context.command_capability_registry is registry
    assert "command_capability_registry" not in dumped
    assert "command_policy" not in dumped
    assert "command_capability_registry" not in snapshot
    assert "command_policy" not in snapshot
    assert "safe-tool" not in json.dumps(dumped, default=str)
    assert "safe-tool" not in json.dumps(snapshot, default=str)


def test_network_enforcer_is_runtime_only_in_context_serialization_and_snapshot() -> None:
    from mmar_carl.network_enforcement import PreconfiguredNetworkEnforcer

    enforcer = PreconfiguredNetworkEnforcer("test", "v1", ())
    context = _context(_ScriptedClient(_response()), registry=_registry(), policy=_policy())
    context.network_enforcer = enforcer

    dumped = context.model_dump()
    durable_snapshot = context.snapshot()
    snapshot = durable_snapshot.model_dump()

    assert context.network_enforcer is enforcer
    assert "network_enforcer" not in dumped
    assert "network_enforcer" not in snapshot
    context.restore(durable_snapshot)
    assert context.network_enforcer is enforcer


@pytest.mark.asyncio
async def test_parallel_context_snapshots_preserve_runtime_only_registry() -> None:
    builder_calls: list[_TextArguments] = []
    registry = _registry(calls=builder_calls)
    record_one = registry.validate_plan(json.loads(_response(text="one", repeat=1)))
    record_two = registry.validate_plan(json.loads(_response(text="two", repeat=2)))
    context = _context(
        _ScriptedClient("unused"),
        registry=registry,
        policy=_policy(),
        approval=lambda _request: True,
    )
    context.metadata.update(
        {
            "plan_one": record_one.model_dump(mode="json"),
            "plan_two": record_two.model_dump(mode="json"),
        }
    )
    steps = [
        _planned_command(
            number=1,
            plan_source="$metadata.plan_one",
            dependencies=[],
        ),
        _planned_command(
            number=2,
            plan_source="$metadata.plan_two",
            dependencies=[],
        ),
    ]

    result = await ReasoningChain(steps=steps, max_workers=2).execute_async(context)

    assert result.success is True
    assert len(builder_calls) == 2
    assert sorted(call["cmd"] for call in _run_calls()) == [
        ["safe-tool", "print", "one", "1"],
        ["safe-tool", "print", "two", "2"],
    ]


@pytest.mark.asyncio
async def test_planner_input_and_complete_prompt_are_host_bounded_before_provider_call() -> None:
    input_client = _ScriptedClient(_response())
    input_context = _context(
        input_client,
        registry=_registry(max_input_value_bytes=8),
        outer_context="x" * 100,
    )
    input_result = await ReasoningChain(
        steps=[_planner(input_mapping={"payload": "$outer_context"})],
        max_workers=1,
    ).execute_async(input_context)
    assert input_result.success is False
    assert "planner input 'payload' exceeds host limit" in input_result.step_results[0].error_message
    assert input_client.prompts == []

    baseline = _registry()
    manifest_bytes = len(
        json.dumps(
            baseline.manifest(),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )
    prompt_client = _ScriptedClient(_response())
    prompt_context = _context(
        prompt_client,
        registry=_registry(max_prompt_bytes=manifest_bytes),
    )
    prompt_result = await ReasoningChain(steps=[_planner()], max_workers=1).execute_async(
        prompt_context
    )
    assert prompt_result.success is False
    assert "command planner prompt exceeds host limit" in prompt_result.step_results[0].error_message
    assert prompt_client.prompts == []


@pytest.mark.asyncio
async def test_planner_response_is_host_bounded_before_json_validation() -> None:
    client = _ScriptedClient(_response(text="x" * 200, repeat=1))
    context = _context(client, registry=_registry(max_response_bytes=32))

    result = await ReasoningChain(steps=[_planner()], max_workers=1).execute_async(context)

    assert result.success is False
    assert result.step_results[0].error_message == (
        "command planner returned an invalid typed plan (ValueError)"
    )
    assert len(client.prompts) == 1
    assert _RecordingRuntime.calls == []


@pytest.mark.asyncio
async def test_planner_timeout_fails_without_command_execution() -> None:
    client = _ScriptedClient(_response(), delay=0.1)
    context = _context(client, registry=_registry())
    step = _planner(timeout=0.01)

    result = await CommandPlanStepExecutor().execute(step, context)

    assert result.success is False
    assert "timed out after 0.01s" in result.error_message
    assert len(client.prompts) == 1
    assert _RecordingRuntime.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("retry_max", "timeout", "expected"),
    [
        (11, None, "retry_max"),
        (None, float("inf"), "timeout"),
    ],
)
async def test_legacy_planner_cannot_bypass_retry_or_timeout_bounds(
    retry_max: int | None,
    timeout: float | None,
    expected: str,
) -> None:
    client = _ScriptedClient(_response())
    context = _context(client, registry=_registry())
    with pytest.warns(DeprecationWarning):
        step = StepDescription(
            number=1,
            title="Legacy planner",
            step_type=StepType.COMMAND_PLAN,
            step_config=CommandPlanStepConfig(
                instruction="Choose a typed command",
                capability_ids=["text.print"],
            ),
            retry_max=retry_max,
            timeout=timeout,
        )

    result = await CommandPlanStepExecutor().execute(step, context)

    assert result.success is False
    assert expected in result.error_message
    assert client.prompts == []


@pytest.mark.asyncio
async def test_replay_command_plan_trace_uses_recorded_envelope_without_live_llm() -> None:
    registry = _registry()
    original_client = _ScriptedClient(_response(text="replayed", repeat=2))
    chain = ReasoningChain(steps=[_planner()], max_workers=1)
    original = await chain.execute_async(_context(original_client, registry=registry))
    assert original.success is True
    assert original.trace is not None

    replay_client = _ScriptedClient("THIS MUST NOT BE CALLED")
    replayed = await chain.replay(
        original.trace,
        _context(replay_client, registry=registry),
    )

    assert replayed.success is True
    assert replay_client.prompts == []
    assert replayed.step_results[0].result_data["plan"] == original.step_results[0].result_data["plan"]


@pytest.mark.asyncio
async def test_execute_from_trace_replays_command_plan_prefix_for_planned_command() -> None:
    registry = _registry()
    chain = ReasoningChain(steps=[_planner(), _planned_command()], max_workers=1)
    original_client = _ScriptedClient(_response(text="from-trace", repeat=4))
    original = await chain.execute_async(
        _context(
            original_client,
            registry=registry,
            policy=_policy(),
            approval=lambda _request: True,
        )
    )
    assert original.success is True
    assert original.trace is not None
    _RecordingRuntime.reset()

    resumed_client = _ScriptedClient("THIS MUST NOT BE CALLED")
    resumed = await chain.execute_from_trace(
        original.trace,
        from_step=2,
        context=_context(
            resumed_client,
            registry=registry,
            policy=_policy(),
            approval=lambda _request: True,
        ),
    )

    assert resumed.success is True
    assert resumed_client.prompts == []
    assert [call["cmd"] for call in _run_calls()] == [
        ["safe-tool", "print", "from-trace", "4"]
    ]


@pytest.mark.asyncio
async def test_replay_command_plan_requires_runtime_registry() -> None:
    registry = _registry()
    chain = ReasoningChain(steps=[_planner()], max_workers=1)
    original = await chain.execute_async(
        _context(_ScriptedClient(_response()), registry=registry)
    )
    assert original.trace is not None

    with pytest.raises(
        ValueError,
        match="replaying CommandPlanStep requires a host-owned CommandCapabilityRegistry",
    ):
        await chain.replay(original.trace, _context(_ScriptedClient("unused")))


@pytest.mark.asyncio
async def test_replay_command_plan_rejects_stale_registry_before_live_llm_or_runtime() -> None:
    original_registry = _registry(revision="r1")
    chain = ReasoningChain(steps=[_planner()], max_workers=1)
    original = await chain.execute_async(
        _context(_ScriptedClient(_response()), registry=original_registry)
    )
    assert original.trace is not None
    replay_client = _ScriptedClient("unused")

    with pytest.raises(ValueError, match="command capability changed; re-plan required"):
        await chain.replay(
            original.trace,
            _context(replay_client, registry=_registry(revision="r2")),
        )

    assert replay_client.prompts == []
    assert _RecordingRuntime.calls == []
