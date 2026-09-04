"""Hermetic typed command-planning example without an external LLM or process.

The scripted client may select one host-declared capability and fill its
strict JSON arguments.  A trusted builder owns the executable and argv
prefix, the host approves the complete provenance-bound invocation, and a
recording runtime proves which argv list would be executed.  No shell, API
key, network request, or subprocess is involved.

Usage:
    python examples/orchestration/command_plan_capabilities_example.py
"""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from mmar_carl import (
    CommandApprovalRequest,
    CommandCapability,
    CommandCapabilityRegistry,
    CommandPlanStepConfig,
    CommandPlanStepDescription,
    CommandPolicy,
    CommandStepConfig,
    CommandStepDescription,
    LLMClientBase,
    ReasoningChain,
    ReasoningContext,
    RuntimeCapabilities,
    RuntimeRunResult,
    SkillRuntimeHandle,
    register_skill_runtime,
)


class PrintArguments(BaseModel):
    """The only dynamic values the scripted model is allowed to provide."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=80)
    repeat: int = Field(ge=1, le=3)


class ScriptedPlanner(LLMClientBase):
    """A deterministic stand-in for an LLM; it never calls a provider."""

    response = json.dumps(
        {
            "capability_id": "text.print",
            "arguments": {"text": "hello typed world", "repeat": 2},
        },
        separators=(",", ":"),
    )

    def __init__(self) -> None:
        self.prompts: list[str] = []

    @property
    def model_name(self) -> str:
        return "scripted-planner"

    async def get_response(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        return await self.get_response(prompt)


class RecordingRuntime:
    """A portable runtime that records argv but launches no process."""

    name = "command_plan_example_recording"
    capabilities = RuntimeCapabilities(
        isolation="unknown",
        wall_time="enforced",
        output_limit="enforced",
        network_none="enforced",
    )
    calls: ClassVar[list[dict[str, Any]]] = []

    async def prepare(
        self,
        skill: Any,
        workspace: Path | None,
        config: dict[str, Any],
    ) -> SkillRuntimeHandle:
        del skill, workspace
        root = Path(tempfile.mkdtemp(prefix="carl_command_plan_example_"))
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
        del handle
        type(self).calls.append(
            {
                "kind": "run",
                "argv": list(cmd),
                "env": dict(env or {}),
                "stdin": stdin,
                "timeout": timeout,
                "cwd": cwd,
            }
        )
        return RuntimeRunResult(
            stdout=b"recording runtime accepted the typed argv",
            stderr=b"",
            exit_code=0,
            duration_s=0.0,
        )

    async def cleanup(self, handle: SkillRuntimeHandle) -> None:
        type(self).calls.append({"kind": "cleanup"})
        shutil.rmtree(handle.workspace_root)


def build_registry() -> CommandCapabilityRegistry:
    def build_print_argv(arguments: BaseModel) -> Sequence[str]:
        assert isinstance(arguments, PrintArguments)
        return (arguments.text, str(arguments.repeat))

    return CommandCapabilityRegistry(
        [
            CommandCapability(
                capability_id="text.print",
                description="Print bounded validated text a bounded number of times",
                executable="demo-text-tool",
                static_args=("emit",),
                argument_model=PrintArguments,
                argv_builder=build_print_argv,
                revision="r1",
            )
        ]
    )


def build_chain() -> ReasoningChain:
    return ReasoningChain(
        trace_name="typed command planning without shell",
        max_workers=1,
        steps=[
            CommandPlanStepDescription(
                number=1,
                title="Select a typed capability",
                config=CommandPlanStepConfig(
                    instruction="Print a short greeting twice.",
                    capability_ids=["text.print"],
                ),
            ),
            CommandStepDescription(
                number=2,
                title="Execute the approved typed plan",
                dependencies=[1],
                config=CommandStepConfig(
                    plan_source="$steps.1.result_data.plan",
                    planned_capability_ids=["text.print"],
                    runtime=RecordingRuntime.name,
                    network="none",
                ),
            ),
        ],
    )


async def main() -> None:
    register_skill_runtime(RecordingRuntime.name, RecordingRuntime)
    RecordingRuntime.calls.clear()
    registry = build_registry()
    client = ScriptedPlanner()
    approvals: list[CommandApprovalRequest] = []

    def approve(request: CommandApprovalRequest) -> bool:
        expected = registry.capabilities["text.print"]
        assert request.source == "planned"
        assert request.invocation_kind == "command"
        assert request.capability_id == expected.capability_id
        assert request.capability_revision == expected.revision
        assert request.capability_fingerprint == expected.fingerprint
        assert request.arguments_sha256 is not None
        assert request.argv == (
            "demo-text-tool",
            "emit",
            "hello typed world",
            "2",
        )
        approvals.append(request)
        return True

    context = ReasoningContext(
        outer_context="",
        api=client,
        model="scripted-planner",
        command_capability_registry=registry,
        command_policy=CommandPolicy(
            allowed_executables=frozenset({"demo-text-tool"}),
            allowed_runtimes=frozenset({RecordingRuntime.name}),
            allowed_networks=frozenset({"none"}),
            require_approval_for_interpreters=False,
        ),
        on_command_approval_requested=approve,
    )

    result = await build_chain().execute_async(context)
    run_calls = [call for call in RecordingRuntime.calls if call["kind"] == "run"]

    assert result.success
    assert len(client.prompts) == 1
    assert len(approvals) == 1
    assert len(run_calls) == 1
    assert run_calls[0]["argv"] == [
        "demo-text-tool",
        "emit",
        "hello typed world",
        "2",
    ]

    approval = approvals[0]
    print(f"planner_calls={len(client.prompts)}")
    print(f"planner_response={ScriptedPlanner.response}")
    print(
        "approval_provenance="
        + json.dumps(
            {
                "source": approval.source,
                "capability_id": approval.capability_id,
                "capability_revision": approval.capability_revision,
                "capability_fingerprint": approval.capability_fingerprint,
                "arguments_sha256": approval.arguments_sha256,
                "argv_preview": approval.argv_preview,
            },
            sort_keys=True,
        )
    )
    print("runtime_argv=" + json.dumps(run_calls[0]["argv"]))
    print("shell_used=false")
    print(f"stdout={result.step_results[1].result}")
    print(f"success={str(result.success).lower()}")


if __name__ == "__main__":
    asyncio.run(main())
