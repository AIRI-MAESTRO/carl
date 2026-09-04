"""
Example: Command Steps in CARL.

This example demonstrates the ``command`` step type, which runs an OS command
through the sandbox ``SkillRuntime`` — the same isolation layer AgentSkill
SCRIPT mode uses. Highlights:

- argv-list commands (never a shell string) — no shell-injection surface
- ``input_mapping`` values resolved from context and passed as discrete argv
  tokens + ``CARL_ARG_*`` env vars (never string-interpolated)
- fail-closed authority: the host supplies an explicit ``CommandPolicy``;
  networking defaults to ``none``, timeouts are always enforced
- piping data to stdin, capping output, and tolerating non-zero exits

No LLM or API key required — command steps make no LLM calls. The commands use
coreutils (``printf`` / ``cat`` / ``wc``), so this runs on Linux/macOS.

Usage:
    python examples/orchestration/command_steps_example.py
"""

import asyncio
import shutil
import warnings

from mmar_carl import (
    CommandPolicy,
    CommandStepConfig,
    LLMClientBase,
    ReasoningChain,
    ReasoningContext,
    StepType,
    create_step,
)


class MockClient(LLMClientBase):
    """Minimal mock client — command steps never call it."""

    async def get_response(self, prompt: str) -> str:
        return "mock"

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        return "mock"


def require_executable(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(f"required demo executable is missing: {name}")
    return path


PRINTF = require_executable("printf")
WC = require_executable("wc")
BASH = require_executable("bash")
FALSE = require_executable("false")
SLEEP = require_executable("sleep")

HOST_POLICY = CommandPolicy(
    allowed_executables=frozenset({PRINTF, WC, FALSE, SLEEP}),
    approval_required_executables=frozenset({BASH}),
    allowed_runtimes=frozenset({"local"}),
    allowed_networks=frozenset({"host"}),
)


def make_context(outer_context: str = "", *, with_policy: bool = True) -> ReasoningContext:
    return ReasoningContext(
        outer_context=outer_context,
        api=MockClient(),
        model="mock",
        command_policy=HOST_POLICY if with_policy else None,
        # A real host renders the typed request and waits for a user decision.
        # This deterministic demo approves only the predeclared BASH rule.
        on_command_approval_requested=lambda request: request.executable == BASH,
    )


async def run_one(
    title: str,
    config: CommandStepConfig,
    *,
    outer_context: str = "",
    with_policy: bool = True,
) -> None:
    chain = ReasoningChain(steps=[create_step(1, title, StepType.COMMAND, config=config)])
    result = await chain.execute_async(make_context(outer_context, with_policy=with_policy))
    sr = result.step_results[0]
    status = "OK" if sr.success else f"FAIL ({sr.error_message})"
    print(f"\n=== {title} — {status} ===")
    if sr.success:
        print(f"exit={sr.result_data['exit_code']}  stdout={sr.result!r}")


async def main() -> None:
    # The local (no-isolation) runtime warns once; silence it for the demo.
    warnings.simplefilter("ignore", UserWarning)

    # 1. Basic command.
    await run_one(
        "echo",
        CommandStepConfig(command=[PRINTF, "%s", "hello from command"], network="host"),
    )

    # 2. input_mapping — a context reference passed as a discrete argv token.
    #    The value is resolved from outer_context and appended safely (no shell).
    await run_one(
        "count words",
        CommandStepConfig(
            command=[WC, "-w"],
            stdin_source="$outer_context",
            network="host",
        ),
        outer_context="one two three four five",
    )

    # 3. Shell features on purpose — use bash -lc, still argv-safe.
    await run_one(
        "shell pipeline",
        CommandStepConfig(
            command=[BASH, "-lc", "printf '%s\\n' a b c | sort -r | tr '\\n' ' '"],
            network="host",
        ),
    )

    # 4. Injection payloads are inert — they are single arguments, not code.
    await run_one(
        "injection is inert",
        CommandStepConfig(
            command=[PRINTF, "%s"],
            input_mapping={"payload": '"; rm -rf / #"'},  # double-quoted literal
            network="host",
        ),
    )

    # 5. Tolerate a non-zero exit code.
    await run_one(
        "nonzero exit tolerated",
        CommandStepConfig(command=[FALSE], network="host", allow_nonzero_exit=True),
    )

    # 6. Fail-closed guard: local runtime requires explicit opt-in.
    await run_one(
        "missing host policy (expected FAIL)",
        CommandStepConfig(command=[PRINTF, "%s", "should not run"], network="host"),
        with_policy=False,
    )

    # 7. Timeout is always enforced.
    await run_one(
        "timeout (expected FAIL)",
        CommandStepConfig(command=[SLEEP, "5"], network="host", timeout=0.5),
    )

    print(
        "\nFor real isolation, install a sandbox backend and set runtime='docker' "
        "(or 'e2b'/'firejail') and provide an application-owned policy."
    )


if __name__ == "__main__":
    asyncio.run(main())
