"""One-process shell state plus portable artifacts, without an LLM call.

Each ``ShellSessionStep`` starts one fresh POSIX shell. Commands inside that
step share cwd, variables and functions; a later step receives only declared,
hash-verified artifacts, never the previous process or a temporary host path.

Usage:
    python examples/orchestration/shell_session_artifacts_example.py
"""

from __future__ import annotations

import asyncio
import shutil
import warnings

from mmar_carl import (
    ArtifactInput,
    ArtifactOutput,
    ArtifactRecord,
    CommandPolicy,
    LLMClientBase,
    ReasoningChain,
    ReasoningContext,
    ShellSessionStepConfig,
    ShellSessionStepDescription,
)


class MockClient(LLMClientBase):
    """Shell steps do not call the client; the context still requires one."""

    async def get_response(self, prompt: str) -> str:
        return "mock"

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        return "mock"


async def main() -> None:
    shell = shutil.which("sh")
    if shell is None:
        raise RuntimeError("this example needs a POSIX sh executable")

    steps = [
        ShellSessionStepDescription(
            number=1,
            title="Normalize text in one shell",
            config=ShellSessionStepConfig(
                shell=shell,
                commands=[
                    "mkdir work",
                    "cd work",
                    'normalize() { tr "[:lower:]" "[:upper:]"; }',
                    ('normalize < "$CARL_ARTIFACT_IN_SOURCE" > "$CARL_ARTIFACT_OUT_NORMALIZED"'),
                    'printf "%s" "${PWD##*/}"',
                ],
                artifact_inputs=[
                    ArtifactInput(
                        name="source",
                        source="'hello from a declared artifact'",
                        path="source.txt",
                        media_type="text/plain",
                    )
                ],
                artifact_outputs=[
                    ArtifactOutput(
                        name="normalized",
                        path="normalized.txt",
                        media_type="text/plain",
                    )
                ],
                runtime="local",
                network="none",
                enforcement_mode="best_effort",
            ),
        ),
        ShellSessionStepDescription(
            number=2,
            title="Consume the previous artifact in a fresh shell",
            dependencies=[1],
            config=ShellSessionStepConfig(
                shell=shell,
                commands=[
                    ('wc -w < "$CARL_ARTIFACT_IN_TEXT" > "$CARL_ARTIFACT_OUT_COUNT"'),
                ],
                artifact_inputs=[
                    ArtifactInput(
                        name="text",
                        source="$steps.1.result_data.artifacts.normalized",
                        path="normalized.txt",
                        media_type="text/plain",
                    )
                ],
                artifact_outputs=[ArtifactOutput(name="count", path="word-count.txt", media_type="text/plain")],
                runtime="local",
                network="none",
                enforcement_mode="best_effort",
            ),
        ),
    ]

    policy = CommandPolicy(
        approval_required_executables=frozenset({shell}),
        allowed_runtimes=frozenset({"local"}),
        allowed_networks=frozenset({"none"}),
        allow_best_effort=True,
    )

    def approve(request) -> bool:
        print(
            "approval:",
            request.step_number,
            request.authorized_executable,
            request.fingerprint[:12],
            request.artifact_manifest,
        )
        return True

    context = ReasoningContext(
        outer_context="",
        api=MockClient(),
        command_policy=policy,
        on_command_approval_requested=approve,
    )
    warnings.simplefilter("ignore", UserWarning)
    result = await ReasoningChain(steps=steps).execute_async(context)

    first = result.step_results[0]
    second = result.step_results[1]
    normalized = ArtifactRecord.model_validate(first.result_data["artifacts"]["normalized"])
    count = ArtifactRecord.model_validate(second.result_data["artifacts"]["count"])
    print("step 1 shell state:", first.result)
    print("normalized:", normalized.decode(max_bytes=1_000).decode())
    print("word count:", count.decode(max_bytes=1_000).decode().strip())
    print("runtime gaps:", second.result_data["enforcement_report"]["gaps"])


if __name__ == "__main__":
    asyncio.run(main())
