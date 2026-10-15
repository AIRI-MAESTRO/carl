"""
Tests for the ``command`` step type (:class:`CommandStepExecutor`).

A command step runs an OS command through the sandbox ``SkillRuntime`` — the
same isolation layer AgentSkill SCRIPT mode uses. These tests lock in the
security posture:

- The command is an argv list; ``input_mapping`` values are passed as
  discrete argv tokens + ``CARL_ARG_*`` env vars, never string-interpolated.
- The host environment is NOT inherited (only a minimal PATH/HOME/LANG base
  plus explicit config env).
- A serialized chain cannot authorize execution. The host must provide an
  exact ``CommandPolicy`` through ``ReasoningContext``.
- Networking defaults to fail-closed ``none``; the policy is threaded into
  the runtime's ``prepare`` config.
- Timeouts, non-zero exits, output caps, and JSON round-tripping behave.

The real-command tests use coreutils (``printf`` / ``cat`` / ``false`` /
``sleep``) and are Linux/POSIX-hermetic. Routing/isolation contract tests use
a recording fake runtime so they don't depend on any host binary.
"""

import asyncio
import os
import shutil
import time
import warnings
from typing import ClassVar

import pytest

from mmar_carl import (
    CommandApprovalRequest,
    CommandPolicy,
    CommandStepConfig,
    CommandStepDescription,
    ReasoningChain,
    ReasoningContext,
    StepCache,
    StepDescription,
    StepType,
    create_step,
)
from mmar_carl.models.llm_client_base import LLMClientBase
from mmar_carl.skill_runtime import (
    RuntimeCapabilities,
    RuntimeRunResult,
    SkillRuntimeError,
    SkillRuntimeHandle,
    register_skill_runtime,
)
from mmar_carl.step_executors import CommandStepExecutor, get_executor

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockClient(LLMClientBase):
    async def get_response(self, prompt: str) -> str:
        return "mock"

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        return "mock"


def make_context(
    outer_context: str = "test",
    *,
    policy: CommandPolicy | None = None,
    approval_handler=None,
) -> ReasoningContext:
    return ReasoningContext(
        outer_context=outer_context,
        api=MockClient(),
        model="mock",
        command_policy=policy,
        on_command_approval_requested=approval_handler,
    )


def command_chain(config: CommandStepConfig) -> ReasoningChain:
    return ReasoningChain(steps=[create_step(1, "command", StepType.COMMAND, config=config)])


def _test_host_policy(config: CommandStepConfig) -> tuple[CommandStepConfig, CommandPolicy]:
    """Build an explicit policy for one test invocation.

    Local execution uses the actual absolute executable path and declares host
    networking honestly. Production applications should define a reusable
    policy rather than deriving one from chain input as this test helper does.
    """

    if config.runtime == "local":
        executable = config.command[0]
        resolved = executable if os.path.isabs(executable) else shutil.which(executable)
        assert resolved is not None, f"test executable is missing: {executable}"
        config = config.model_copy(
            update={
                "command": [resolved, *config.command[1:]],
                "network": "host",
            }
        )
    policy = CommandPolicy(
        allowed_executables=frozenset({config.command[0]}),
        allowed_runtimes=frozenset({config.runtime}),
        allowed_networks=frozenset({config.network}),
        allowed_network_hosts=frozenset(config.network_allowlist),
        allowed_env_keys=frozenset(config.env),
        local_working_roots=(config.working_dir,) if config.working_dir else (),
        allow_best_effort=config.enforcement_mode == "best_effort",
        require_approval_for_interpreters=False,
    )
    return config, policy


async def run_command(
    config: CommandStepConfig,
    *,
    outer_context: str = "test",
    with_policy: bool = True,
    policy: CommandPolicy | None = None,
    approval_handler=None,
):
    """Execute a single-step command chain and return its result."""
    if with_policy and policy is None:
        config, policy = _test_host_policy(config)
    result = await command_chain(config).execute_async(
        make_context(
            outer_context,
            policy=policy,
            approval_handler=approval_handler,
        )
    )
    return result.step_results[0]


class RecordingRuntime:
    """Fake SkillRuntime that records calls and returns canned output.

    ``get_skill_runtime`` instantiates the registered class fresh per call, so
    the recording buffers live at class scope.
    """

    name = "recording"
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
    calls: ClassVar[list[dict]] = []
    stdout: bytes = b"canned-stdout"
    stderr: bytes = b""
    exit_code: int = 0
    cleaned: ClassVar[list] = []
    cleanup_error: ClassVar[Exception | None] = None
    runtime_path_padding: ClassVar[str] = ""

    @classmethod
    def reset(cls) -> None:
        cls.calls = []
        cls.cleaned = []
        cls.stdout = b"canned-stdout"
        cls.stderr = b""
        cls.exit_code = 0
        cls.cleanup_error = None
        cls.runtime_path_padding = ""

    async def prepare(self, skill, workspace, config):
        import tempfile
        from pathlib import Path

        ws = Path(workspace) if workspace else Path(tempfile.mkdtemp(prefix="rec_"))
        ws.mkdir(parents=True, exist_ok=True)
        RecordingRuntime.calls.append({"kind": "prepare", "config": config})
        runtime_out = str(ws) + RecordingRuntime.runtime_path_padding
        return SkillRuntimeHandle(
            workspace_root=ws,
            workspace_in=ws,
            workspace_out=ws,
            backend={
                "network_enforced": True,
                "workspace_in_in_runtime": runtime_out,
                "workspace_out_in_runtime": runtime_out,
            },
        )

    async def run(self, handle, cmd, *, env=None, stdin=None, timeout=None, cwd=None):
        RecordingRuntime.calls.append(
            {
                "kind": "run",
                "cmd": list(cmd),
                "env": dict(env or {}),
                "stdin": stdin,
                "timeout": timeout,
                "cwd": cwd,
            }
        )
        return RuntimeRunResult(
            stdout=RecordingRuntime.stdout,
            stderr=RecordingRuntime.stderr,
            exit_code=RecordingRuntime.exit_code,
            duration_s=0.0,
        )

    async def read_file(self, handle, path):  # pragma: no cover - unused
        return b""

    async def write_file(self, handle, path, data):  # pragma: no cover - unused
        return None

    async def cleanup(self, handle):
        RecordingRuntime.cleaned.append(handle)
        if RecordingRuntime.cleanup_error is not None:
            raise RecordingRuntime.cleanup_error


@pytest.fixture()
def recording_runtime():
    register_skill_runtime("recording", RecordingRuntime)
    RecordingRuntime.reset()
    yield RecordingRuntime
    RecordingRuntime.reset()


@pytest.fixture(autouse=True)
def _silence_unsafe_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        yield


# ---------------------------------------------------------------------------
# Registration & wiring
# ---------------------------------------------------------------------------


class TestWiring:
    def test_executor_registered(self):
        assert isinstance(get_executor(StepType.COMMAND), CommandStepExecutor)

    def test_create_step_builds_typed_description(self):
        step = create_step(1, "b", StepType.COMMAND, config=CommandStepConfig(command=["echo"]))
        assert isinstance(step, CommandStepDescription)
        assert step.step_type == StepType.COMMAND
        assert step.step_config.command == ["echo"]

    def test_create_step_rejects_wrong_config(self):
        with pytest.raises(ValueError, match="COMMAND steps require CommandStepConfig"):
            create_step(1, "b", StepType.COMMAND, config=None)

    def test_model_dump_includes_step_type(self):
        step = create_step(1, "b", StepType.COMMAND, config=CommandStepConfig(command=["echo"]))
        assert step.model_dump()["step_type"] == "command"

    def test_rejects_colliding_result_key(self):
        with pytest.raises(ValueError, match="collides with command result metadata"):
            CommandStepConfig(command=["true"], output_key="exit_code")

    def test_rejects_case_colliding_input_names(self):
        with pytest.raises(ValueError, match="unique after uppercasing"):
            CommandStepConfig(command=["true"], input_mapping={"file": "'a'", "FILE": "'b'"})

    def test_rejects_case_insensitive_reserved_or_duplicate_env(self):
        with pytest.raises(ValueError, match="CARL-reserved"):
            CommandStepConfig(command=["true"], env={"carl_arg_value": "x"})
        with pytest.raises(ValueError, match="unique after uppercasing"):
            CommandStepConfig(command=["true"], env={"PATH": "a", "Path": "b"})

    def test_network_allowlist_shape_is_explicit(self):
        with pytest.raises(ValueError, match="requires at least one host"):
            CommandStepConfig(command=["true"], network="allowlist")
        with pytest.raises(ValueError, match="only valid"):
            CommandStepConfig(
                command=["true"],
                network="none",
                network_allowlist=["api.example.com"],
            )
        with pytest.raises(ValueError, match="exact hostnames"):
            CommandStepConfig(
                command=["true"],
                network="allowlist",
                network_allowlist=["https://api.example.com/path"],
            )
        normalized = CommandStepConfig(
            command=["true"],
            network="allowlist",
            network_allowlist=["API.EXAMPLE.COM."],
        )
        assert normalized.network_allowlist == ["api.example.com"]

    def test_legacy_permission_flags_are_ignored(self):
        with pytest.warns(DeprecationWarning, match="ignored"):
            config = CommandStepConfig.model_validate(
                {
                    "command": ["true"],
                    "allow_unsafe_local": True,
                    "allow_unenforced_network": True,
                }
            )
        assert "allow_unsafe_local" not in config.model_dump()
        assert "allow_unenforced_network" not in config.model_dump()

    @pytest.mark.parametrize("field", ["timeout", "artifact_io_timeout", "cpu_limit"])
    def test_float_controls_must_be_finite(self, field):
        with pytest.raises(ValueError):
            CommandStepConfig(command=["true"], **{field: float("inf")})

    @pytest.mark.parametrize("value", ["", "0", "512", "garbage", "1.5g"])
    def test_memory_limit_requires_positive_integer_and_unit(self, value):
        with pytest.raises(ValueError, match="explicit b/k/m/g suffix"):
            CommandStepConfig(command=["true"], mem_limit=value)

    def test_memory_limit_is_canonicalized(self):
        assert CommandStepConfig(command=["true"], mem_limit="2G").mem_limit == "2g"

    @pytest.mark.parametrize("working_dir", ["", "   ", "bad\x00path"])
    def test_working_dir_rejects_empty_or_nul(self, working_dir):
        with pytest.raises(ValueError, match="working_dir"):
            CommandStepConfig(command=["true"], working_dir=working_dir)


# ---------------------------------------------------------------------------
# Execution behaviour (real local runtime, coreutils)
# ---------------------------------------------------------------------------


class TestExecution:
    async def test_basic_success(self):
        sr = await run_command(CommandStepConfig(command=["printf", "%s", "hello"]))
        assert sr.success is True
        assert sr.result == "hello"
        assert sr.result_data["exit_code"] == 0
        assert sr.result_data["stdout"] == "hello"
        assert sr.step_type == StepType.COMMAND

    async def test_history_updated(self):
        sr = await run_command(CommandStepConfig(command=["printf", "%s", "hi"]))
        assert any("hi" in h for h in sr.updated_history)

    async def test_nonzero_exit_fails_by_default(self):
        sr = await run_command(CommandStepConfig(command=["false"]))
        assert sr.success is False
        assert "exited with code 1" in sr.error_message

    async def test_nonzero_exit_allowed(self):
        sr = await run_command(CommandStepConfig(command=["false"], allow_nonzero_exit=True))
        assert sr.success is True
        assert sr.result_data["exit_code"] == 1

    async def test_timeout(self):
        sr = await run_command(CommandStepConfig(command=["sleep", "5"], timeout=0.3))
        assert sr.success is False
        assert "timed out after 0.3s" in sr.error_message

    async def test_timeout_kills_child_process_group(self):
        started = time.perf_counter()
        sr = await run_command(
            CommandStepConfig(
                command=["sh", "-c", "sleep 3 & wait"],
                timeout=0.2,
            )
        )
        elapsed = time.perf_counter() - started
        assert sr.success is False
        assert elapsed < 1.5

    async def test_stdin_piped(self):
        sr = await run_command(CommandStepConfig(command=["cat"], stdin_source="'piped-in'"))
        assert sr.result == "piped-in"

    async def test_output_capped(self):
        # yes | head would be unbounded; use printf of a known long string.
        sr = await run_command(
            CommandStepConfig(
                command=["printf", "%s", "X" * 100],
                max_output_bytes=10,
            )
        )
        assert sr.result == "X" * 10
        assert sr.result_data["stdout_truncated"] is True

    async def test_explicit_working_dir_is_not_deleted(self, tmp_path):
        sentinel = tmp_path / "keep-me.txt"
        sentinel.write_text("caller-owned")
        sr = await run_command(
            CommandStepConfig(
                command=["true"],
                working_dir=str(tmp_path),
            )
        )
        assert sr.success is True
        assert sentinel.read_text() == "caller-owned"


# ---------------------------------------------------------------------------
# Security posture
# ---------------------------------------------------------------------------


class TestSecurity:
    @pytest.mark.parametrize(
        "output_key",
        [
            "network_enforcement",
            "command_source",
            "capability_id",
            "capability_revision",
            "capability_fingerprint",
        ],
    )
    def test_output_key_cannot_collide_with_command_metadata(self, output_key):
        with pytest.raises(ValueError, match="collides"):
            CommandStepConfig(
                command=["true"],
                output_key=output_key,
            )

    async def test_unsafe_local_blocked_by_default(self):
        sr = await run_command(CommandStepConfig(command=["echo", "no"]), with_policy=False)
        assert sr.success is False
        assert "host-owned CommandPolicy" in sr.error_message

    async def test_host_env_not_inherited(self):
        os.environ["CARL_BASH_SECRET"] = "leaked"
        try:
            sr = await run_command(
                CommandStepConfig(
                    command=["sh", "-c", 'printf %s "${CARL_BASH_SECRET:-ABSENT}"'],
                )
            )
        finally:
            os.environ.pop("CARL_BASH_SECRET", None)
        assert sr.result == "ABSENT"

    async def test_explicit_env_passed(self):
        sr = await run_command(
            CommandStepConfig(
                command=["sh", "-c", 'printf %s "$MY_VAR"'],
                env={"MY_VAR": "provided"},
            )
        )
        assert sr.result == "provided"

    async def test_input_mapping_not_string_interpolated(self):
        # A shell-injection payload must be an inert argv token, not executed.
        # Double-quote wrapping makes resolve_context_reference return the
        # literal verbatim (the payload itself contains shell metacharacters).
        payload = "; touch /tmp/carl_pwned #"
        sr = await run_command(
            CommandStepConfig(
                command=["printf", "%s"],
                input_mapping={"payload": f'"{payload}"'},
            )
        )
        assert sr.success is True
        # The payload is echoed verbatim as a single argument (printf, no shell).
        assert sr.result == payload
        assert not os.path.exists("/tmp/carl_pwned")

    async def test_input_mapping_exposed_as_env(self):
        sr = await run_command(
            CommandStepConfig(
                command=["sh", "-c", 'printf %s "$CARL_ARG_NAME"'],
                input_mapping={"name": "'world'"},
            )
        )
        assert sr.result == "world"

    async def test_bad_runtime_name_fails_cleanly(self):
        sr = await run_command(CommandStepConfig(command=["echo"], runtime="does-not-exist"))
        assert sr.success is False
        assert "Unknown skill runtime" in sr.error_message


# ---------------------------------------------------------------------------
# Runtime routing / isolation contract (fake recording runtime)
# ---------------------------------------------------------------------------


class TestRuntimeRouting:
    async def test_routes_through_runtime(self, recording_runtime):
        sr = await run_command(CommandStepConfig(command=["whatever"], runtime="recording"))
        assert sr.success is True
        assert sr.result == "canned-stdout"
        kinds = [c["kind"] for c in recording_runtime.calls]
        assert kinds == ["prepare", "run"]
        assert recording_runtime.cleaned, "cleanup must always run"

    async def test_argv_and_env_construction(self, recording_runtime):
        await run_command(
            CommandStepConfig(
                command=["mytool", "--flag"],
                input_mapping={"x": "'val'"},
                runtime="recording",
            )
        )
        run_call = next(c for c in recording_runtime.calls if c["kind"] == "run")
        # Resolved value appended as a discrete argv token.
        assert run_call["cmd"] == ["mytool", "--flag", "val"]
        # And exposed as CARL_ARG_X in the env.
        assert run_call["env"]["CARL_ARG_X"] == "val"
        # Host env not inherited: a random host var should be absent.
        assert "PYTEST_CURRENT_TEST" not in run_call["env"]

    async def test_network_policy_threaded_to_prepare(self, recording_runtime):
        await run_command(
            CommandStepConfig(
                command=["x"],
                runtime="recording",
                network="allowlist",
                network_allowlist=["api.example.com"],
            )
        )
        prep = next(c for c in recording_runtime.calls if c["kind"] == "prepare")
        assert prep["config"]["network"] == "allowlist"
        assert prep["config"]["network_allowlist"] == ["api.example.com"]

    async def test_unenforced_restricted_network_fails_closed(self, recording_runtime):
        class AdvisoryRuntime(RecordingRuntime):
            name = "advisory"

            async def prepare(self, skill, workspace, config):
                handle = await super().prepare(skill, workspace, config)
                handle.backend["network_enforced"] = False
                return handle

        register_skill_runtime("advisory", AdvisoryRuntime)
        sr = await run_command(CommandStepConfig(command=["x"], runtime="advisory"))
        assert sr.success is False
        assert "contradicted its declared network='none' enforcement" in sr.error_message
        assert recording_runtime.cleaned

    async def test_runtime_cannot_downgrade_declared_network_enforcement(
        self,
        recording_runtime,
    ):
        class ContradictoryRuntime(RecordingRuntime):
            name = "contradictory"

            async def prepare(self, skill, workspace, config):
                handle = await super().prepare(skill, workspace, config)
                handle.backend["network_enforced"] = False
                return handle

        register_skill_runtime("contradictory", ContradictoryRuntime)
        sr = await run_command(
            CommandStepConfig(
                command=["x"],
                runtime="contradictory",
                enforcement_mode="best_effort",
            )
        )
        assert sr.success is False
        assert "contradicted its declared" in sr.error_message
        assert recording_runtime.cleaned

    async def test_resource_limits_threaded_to_prepare(self, recording_runtime):
        await run_command(
            CommandStepConfig(
                command=["x"],
                runtime="recording",
                cpu_limit=2.0,
                mem_limit="512m",
                pids_limit=64,
            )
        )
        prep = next(c for c in recording_runtime.calls if c["kind"] == "prepare")
        assert prep["config"]["cpu_limit"] == 2.0
        assert prep["config"]["mem_limit"] == "512m"
        assert prep["config"]["pids_limit"] == 64

    async def test_timeout_forwarded(self, recording_runtime):
        await run_command(CommandStepConfig(command=["x"], runtime="recording", timeout=12.5))
        run_call = next(c for c in recording_runtime.calls if c["kind"] == "run")
        assert run_call["timeout"] == 12.5

    async def test_docker_uses_requested_limits_and_container_cwd(self, monkeypatch):
        from mmar_carl.docker_skill_runtime import DockerSkillRuntime

        monkeypatch.setattr("mmar_carl.docker_skill_runtime.shutil.which", lambda _: "/usr/bin/docker")
        runtime = DockerSkillRuntime()
        handle = await runtime.prepare(
            None,
            None,
            {
                "network": "none",
                "cpu_limit": 2.0,
                "mem_limit": "512m",
                "pids_limit": 64,
            },
        )
        try:
            docker_argv = runtime._build_docker_cmd(
                handle,
                ["true"],
                env=None,
                timeout=1.0,
                cwd=str(handle.backend["workspace_out_in_runtime"]),
            )
        finally:
            await runtime.cleanup(handle)

        assert docker_argv[docker_argv.index("--cpus") + 1] == "2.0"
        assert docker_argv[docker_argv.index("--memory") + 1] == "512m"
        assert docker_argv[docker_argv.index("--pids-limit") + 1] == "64"
        assert docker_argv[docker_argv.index("-w") + 1] == "/workspace/out"

    async def test_runtime_error_is_step_failure(self, recording_runtime):
        class Boom(RecordingRuntime):
            name = "boom"

            async def run(self, *a, **k):
                raise SkillRuntimeError("sandbox exploded")

        register_skill_runtime("boom", Boom)
        sr = await run_command(CommandStepConfig(command=["x"], runtime="boom"))
        assert sr.success is False
        assert "sandbox exploded" in sr.error_message

    async def test_cleanup_error_marks_successful_execution_failed(
        self,
        recording_runtime,
    ):
        recording_runtime.cleanup_error = RuntimeError("sandbox still alive")
        sr = await run_command(CommandStepConfig(command=["x"], runtime="recording"))
        assert sr.success is False
        assert "sandbox cleanup failed" in sr.error_message
        assert "sandbox still alive" in sr.error_message

    async def test_cleanup_error_is_reported_alongside_runtime_failure(
        self,
        recording_runtime,
    ):
        class BoomAndLeak(RecordingRuntime):
            name = "boom-and-leak"

            async def run(self, *args, **kwargs):
                raise SkillRuntimeError("primary sandbox failure")

        register_skill_runtime(BoomAndLeak.name, BoomAndLeak)
        recording_runtime.cleanup_error = RuntimeError("sandbox still alive")
        result = await run_command(
            CommandStepConfig(command=["x"], runtime=BoomAndLeak.name)
        )
        assert result.success is False
        assert "primary sandbox failure" in (result.error_message or "")
        assert "cleanup also failed" in (result.error_message or "")
        assert "sandbox still alive" in (result.error_message or "")


# ---------------------------------------------------------------------------
# Host policy and approvals
# ---------------------------------------------------------------------------


class TestCommandPolicy:
    def test_policy_float_limits_must_be_finite(self):
        with pytest.raises(ValueError):
            CommandPolicy(max_timeout=float("inf"))
        with pytest.raises(ValueError):
            CommandPolicy(network_enforcer_timeout=float("inf"))
        with pytest.raises(ValueError):
            CommandPolicy(runtime_cleanup_timeout=float("inf"))

    @pytest.mark.parametrize(
        ("resource", "value", "policy_override", "expected"),
        [
            ("timeout", 11.0, {"max_timeout": 10.0}, "timeout"),
            (
                "artifact_io_timeout",
                11.0,
                {"max_artifact_io_timeout": 10.0},
                "artifact I/O timeout",
            ),
            ("cpu_limit", 3.0, {"max_cpu_limit": 2.0}, "CPU limit"),
            ("mem_limit", "2g", {"max_memory_bytes": 1024**3}, "memory limit"),
            ("pids_limit", 65, {"max_pids_limit": 64}, "PID limit"),
            (
                "max_output_bytes",
                101,
                {"max_output_bytes": 100},
                "output byte limit",
            ),
            ("artifact_count", 3, {"max_artifact_count": 2}, "artifact count"),
        ],
    )
    def test_policy_rejects_chain_resource_requests_above_host_caps(
        self,
        resource,
        value,
        policy_override,
        expected,
    ):
        policy = CommandPolicy(
            allowed_executables=frozenset({"tool"}),
            allowed_runtimes=frozenset({"recording"}),
            **policy_override,
        )
        resources = {
            "timeout": 1.0,
            "artifact_io_timeout": 1.0,
            "cpu_limit": None,
            "mem_limit": None,
            "pids_limit": None,
            "max_output_bytes": 10,
            "artifact_count": 0,
        }
        resources[resource] = value
        decision = policy.evaluate(
            ["tool"],
            runtime="recording",
            network="none",
            enforcement_mode="strict",
            working_dir=None,
            explicit_env_keys=(),
            **resources,
        )
        assert decision.outcome == "deny"
        assert expected in decision.reason

    async def test_resource_denial_happens_before_runtime_prepare(self, recording_runtime):
        policy = CommandPolicy(
            allowed_executables=frozenset({"tool"}),
            allowed_runtimes=frozenset({"recording"}),
            max_timeout=1.0,
        )
        result = await run_command(
            CommandStepConfig(command=["tool"], runtime="recording", timeout=2.0),
            policy=policy,
        )
        assert result.success is False
        assert "exceeds host maximum" in (result.error_message or "")
        assert not recording_runtime.calls

    async def test_omitted_optional_resources_are_capped_by_host_policy(
        self,
        recording_runtime,
    ):
        policy = CommandPolicy(
            allowed_executables=frozenset({"tool"}),
            allowed_runtimes=frozenset({"recording"}),
            max_cpu_limit=1.5,
            max_memory_bytes=512 * 1024**2,
            max_pids_limit=32,
            require_approval_for_interpreters=False,
        )
        result = await run_command(
            CommandStepConfig(command=["tool"], runtime="recording"),
            policy=policy,
        )
        assert result.success is True
        prepare = next(
            call for call in recording_runtime.calls if call["kind"] == "prepare"
        )
        assert prepare["config"]["cpu_limit"] == 1.5
        assert prepare["config"]["mem_limit"] == f"{512 * 1024**2}b"
        assert prepare["config"]["pids_limit"] == 32
        controls = result.result_data["enforcement_report"]["controls"]
        assert controls["cpu_limit"] == "enforced"
        assert controls["memory_limit"] == "enforced"
        assert controls["pids_limit"] == "enforced"

    @pytest.mark.parametrize(
        ("policy_override", "command", "environment", "stdin", "expected"),
        [
            ({"max_argument_count": 1}, ["tool", "x"], {}, None, "argument count"),
            ({"max_argv_bytes": 4}, ["tool", "x"], {}, None, "argv"),
            (
                {"max_environment_keys": 1},
                ["tool"],
                {"A": "1", "B": "2"},
                None,
                "environment key count",
            ),
            (
                {"max_environment_bytes": 3},
                ["tool"],
                {"A": "long"},
                None,
                "environment",
            ),
            ({"max_stdin_bytes": 2}, ["tool"], {}, b"abc", "stdin"),
            ({}, ["tool", "\ud800"], {}, None, "valid UTF-8"),
            ({}, ["tool"], {"A": "\ud800"}, None, "valid UTF-8"),
        ],
    )
    def test_policy_bounds_complete_invocation_shape(
        self,
        policy_override,
        command,
        environment,
        stdin,
        expected,
    ):
        policy = CommandPolicy(
            allowed_executables=frozenset({"tool"}),
            allowed_runtimes=frozenset({"recording"}),
            **policy_override,
        )
        decision = policy.evaluate(
            command,
            runtime="recording",
            network="none",
            enforcement_mode="strict",
            working_dir=None,
            explicit_env_keys=(),
            environment=environment,
            stdin=stdin,
        )
        assert decision.outcome == "deny"
        assert expected in decision.reason

    def test_policy_caps_final_normalized_executable_path(self, tmp_path):
        target_dir = tmp_path / ("long-" + "x" * 120)
        target_dir.mkdir()
        target = target_dir / "tool"
        target.write_text("#!/bin/sh\n")
        link = tmp_path / "t"
        link.symlink_to(target)
        requested_size = len(str(link).encode()) + 1
        policy = CommandPolicy(
            allowed_executables=frozenset({str(target)}),
            allowed_runtimes=frozenset({"local"}),
            max_argv_bytes=requested_size,
        )
        decision = policy.evaluate(
            [str(link)],
            runtime="local",
            network="none",
            enforcement_mode="strict",
            working_dir=None,
            explicit_env_keys=(),
        )
        assert decision.outcome == "deny"
        assert "argv" in decision.reason

    def test_local_surrogate_path_is_denied_before_path_resolution(self):
        malformed = "/tmp/\ud800"
        policy = CommandPolicy(
            allowed_executables=frozenset({"/bin/true"}),
            allowed_runtimes=frozenset({"local"}),
        )
        decision = policy.evaluate(
            [malformed],
            runtime="local",
            network="none",
            enforcement_mode="strict",
            working_dir=None,
            explicit_env_keys=(),
        )
        assert decision.outcome == "deny"
        assert "valid UTF-8" in decision.reason

    async def test_actual_runtime_artifact_env_is_rechecked_before_write_or_run(
        self,
        recording_runtime,
    ):
        recording_runtime.runtime_path_padding = "x" * 300
        policy = CommandPolicy(
            allowed_executables=frozenset({"tool"}),
            allowed_runtimes=frozenset({"recording"}),
            max_environment_bytes=100,
        )
        result = await run_command(
            CommandStepConfig(
                command=["tool"],
                runtime="recording",
                artifact_outputs=[{"name": "out", "path": "out.txt"}],
            ),
            policy=policy,
        )
        assert result.success is False
        assert "runtime-final invocation denied" in (result.error_message or "")
        assert not any(call["kind"] == "run" for call in recording_runtime.calls)

    async def test_oversize_dynamic_stdin_fails_before_runtime_prepare(
        self,
        recording_runtime,
    ):
        policy = CommandPolicy(
            allowed_executables=frozenset({"tool"}),
            allowed_runtimes=frozenset({"recording"}),
            max_stdin_bytes=3,
        )
        result = await run_command(
            CommandStepConfig(
                command=["tool"],
                runtime="recording",
                stdin_source="$outer_context",
            ),
            outer_context="four",
            policy=policy,
        )
        assert result.success is False
        assert "stdin exceeds" in (result.error_message or "")
        assert not recording_runtime.calls

    def test_shell_session_always_requires_interpreter_approval(self):
        policy = CommandPolicy(
            allowed_executables=frozenset({"custom-shell"}),
            allowed_runtimes=frozenset({"docker"}),
        )
        decision = policy.evaluate(
            ["custom-shell", "-s"],
            runtime="docker",
            network="none",
            enforcement_mode="strict",
            working_dir=None,
            explicit_env_keys=[],
            invocation_kind="shell_session",
        )
        assert decision.outcome == "require_approval"

    async def test_approval_uses_immutable_execution_snapshot(
        self,
        recording_runtime,
        monkeypatch,
    ):
        executable = "/reviewed/executable"
        config = CommandStepConfig(
            command=[executable, "reviewed"],
            runtime="local",
            env={"SAFE": "reviewed"},
            timeout=12.5,
        )
        step = create_step(1, "command", StepType.COMMAND, config=config)
        chain = ReasoningChain(steps=[step])
        policy = CommandPolicy(
            approval_required_executables=frozenset({executable}),
            allowed_runtimes=frozenset({"local"}),
            allowed_env_keys=frozenset({"SAFE"}),
        )

        async def mutate_after_review(_request):
            step.config.command[:] = ["evil", "changed"]
            step.config.env.clear()
            step.config.env["UNREVIEWED"] = "changed"
            step.config.timeout = 0.01
            step.config.network = "host"
            monkeypatch.setenv("PATH", "/unreviewed/path")
            return True

        from mmar_carl.skill_runtime import SKILL_RUNTIME_REGISTRY

        monkeypatch.setitem(SKILL_RUNTIME_REGISTRY, "local", RecordingRuntime)
        monkeypatch.setenv("PATH", "/reviewed/path")
        result = await chain.execute_async(
            make_context(policy=policy, approval_handler=mutate_after_review),
        )

        sr = result.step_results[0]
        assert sr.success is True
        prepare_call = next(call for call in recording_runtime.calls if call["kind"] == "prepare")
        run_call = next(call for call in recording_runtime.calls if call["kind"] == "run")
        assert prepare_call["config"]["network"] == "none"
        assert run_call["cmd"] == [executable, "reviewed"]
        assert run_call["env"]["SAFE"] == "reviewed"
        assert run_call["env"]["PATH"] == "/reviewed/path"
        assert "UNREVIEWED" not in run_call["env"]
        assert run_call["timeout"] == 12.5

    async def test_allow_nonzero_exit_changes_approval_fingerprint(
        self,
        recording_runtime,
    ):
        requests = []
        policy = CommandPolicy(
            approval_required_executables=frozenset({"x"}),
            allowed_runtimes=frozenset({"recording"}),
        )

        for allow_nonzero in (False, True):
            await run_command(
                CommandStepConfig(
                    command=["x"],
                    runtime="recording",
                    allow_nonzero_exit=allow_nonzero,
                ),
                policy=policy,
                approval_handler=lambda request: requests.append(request) or True,
            )

        assert requests[0].fingerprint != requests[1].fingerprint
        assert requests[0].resources["allow_nonzero_exit"] is False
        assert requests[1].resources["allow_nonzero_exit"] is True

    async def test_denied_command_does_not_instantiate_runtime(self):
        class ConstructorRuntime(RecordingRuntime):
            name = "constructor"
            constructed = 0

            def __init__(self):
                type(self).constructed += 1

        register_skill_runtime("constructor", ConstructorRuntime)
        policy = CommandPolicy(
            allowed_executables=frozenset({"allowed"}),
            allowed_runtimes=frozenset({"constructor"}),
        )
        sr = await run_command(
            CommandStepConfig(command=["denied"], runtime="constructor"),
            policy=policy,
        )
        assert sr.success is False
        assert "exact allowlist" in sr.error_message
        assert ConstructorRuntime.constructed == 0

    def test_local_rule_is_exact_path_not_basename(self):
        allowed = shutil.which("printf")
        assert allowed is not None
        policy = CommandPolicy(
            allowed_executables=frozenset({allowed}),
            allowed_runtimes=frozenset({"local"}),
            allowed_networks=frozenset({"host"}),
        )
        decision = policy.evaluate(
            ["/tmp/printf", "hello"],
            runtime="local",
            network="host",
            enforcement_mode="strict",
            working_dir=None,
            explicit_env_keys=(),
        )
        assert decision.outcome == "deny"

    async def test_local_exec_uses_policy_normalized_path(self, tmp_path):
        target = shutil.which("printf")
        assert target is not None
        link = tmp_path / "trusted-printf"
        link.symlink_to(target)
        config = CommandStepConfig(
            command=[str(link), "%s", "ok"],
            network="host",
        )
        policy = CommandPolicy(
            allowed_executables=frozenset({str(link)}),
            allowed_runtimes=frozenset({"local"}),
            allowed_networks=frozenset({"host"}),
        )
        sr = await run_command(config, policy=policy)
        assert sr.success is True
        assert sr.result == "ok"
        assert sr.result_data["policy_decision"]["authorized_executable"] == str(link.resolve())

    async def test_local_exec_uses_policy_normalized_working_dir(self, tmp_path):
        root = tmp_path / "root"
        target = root / "cwd"
        target.mkdir(parents=True)
        link = tmp_path / "cwd-link"
        link.symlink_to(target, target_is_directory=True)
        executable = shutil.which("pwd")
        assert executable is not None
        config = CommandStepConfig(
            command=[executable],
            network="host",
            working_dir=str(link),
        )
        policy = CommandPolicy(
            allowed_executables=frozenset({executable}),
            allowed_runtimes=frozenset({"local"}),
            allowed_networks=frozenset({"host"}),
            local_working_roots=(str(root),),
        )
        sr = await run_command(config, policy=policy)
        assert sr.success is True
        assert sr.result.strip() == str(target.resolve())
        assert sr.result_data["policy_decision"]["authorized_working_dir"] == str(target.resolve())

    def test_firejail_requires_an_absolute_host_executable(self):
        policy = CommandPolicy(
            allowed_executables=frozenset({"python"}),
            allowed_runtimes=frozenset({"firejail"}),
        )
        decision = policy.evaluate(
            ["python", "-V"],
            runtime="firejail",
            network="none",
            enforcement_mode="strict",
            working_dir=None,
            explicit_env_keys=(),
        )
        assert decision.outcome == "deny"

    def test_local_symlink_cannot_bypass_approval_rule(self, tmp_path):
        target = shutil.which("sh")
        assert target is not None
        link = tmp_path / "trusted-runner"
        link.symlink_to(target)
        policy = CommandPolicy(
            approval_required_executables=frozenset({target}),
            allowed_runtimes=frozenset({"local"}),
            allowed_networks=frozenset({"host"}),
            require_approval_for_interpreters=False,
        )
        decision = policy.evaluate(
            [str(link), "-c", "true"],
            runtime="local",
            network="host",
            enforcement_mode="strict",
            working_dir=None,
            explicit_env_keys=(),
        )
        assert decision.outcome == "require_approval"
        assert decision.matched_rule == target

    def test_local_rules_cannot_overlap_through_symlinks(self, tmp_path):
        target = shutil.which("true")
        assert target is not None
        link = tmp_path / "same-target"
        link.symlink_to(target)
        with pytest.raises(ValueError, match="overlap after path normalization"):
            CommandPolicy(
                allowed_executables=frozenset({target}),
                approval_required_executables=frozenset({str(link)}),
                allowed_runtimes=frozenset({"local"}),
            )

    def test_host_working_roots_must_be_absolute(self):
        with pytest.raises(ValueError, match="must be absolute"):
            CommandPolicy(local_working_roots=("relative/path",))

    async def test_approval_callback_receives_complete_request(self, recording_runtime):
        requests: list[CommandApprovalRequest] = []

        async def approve(request: CommandApprovalRequest) -> bool:
            requests.append(request)
            return True

        policy = CommandPolicy(
            approval_required_executables=frozenset({"python"}),
            allowed_runtimes=frozenset({"recording"}),
            require_approval_for_interpreters=False,
        )
        sr = await run_command(
            CommandStepConfig(
                command=["python", "-c", "print('ok')"],
                input_mapping={"target": "'dynamic-value'"},
                runtime="recording",
            ),
            policy=policy,
            approval_handler=approve,
        )
        assert sr.success is True
        assert sr.result_data["approval"] == "approved"
        assert len(requests) == 1
        assert requests[0].argv[-1] == "dynamic-value"
        assert requests[0].argv_preview[-1] == "<dynamic-arg-1>"
        assert requests[0].environment["CARL_ARG_TARGET"] == "dynamic-value"
        assert requests[0].enforcement_report["fully_enforced"] is True
        assert requests[0].fingerprint

    async def test_public_approval_events_redact_dynamic_argv(self, recording_runtime):
        events: list[tuple[str, dict]] = []
        policy = CommandPolicy(
            approval_required_executables=frozenset({"dangerous"}),
            allowed_runtimes=frozenset({"recording"}),
            require_approval_for_interpreters=False,
        )
        context = make_context(
            policy=policy,
            approval_handler=lambda _request: True,
        )
        context.on_step_event = lambda _number, kind, payload: events.append((kind, payload))
        config = CommandStepConfig(
            command=["dangerous", "--token", "static-secret", "--target"],
            input_mapping={"target": "'dynamic-secret'"},
            runtime="recording",
        )
        result = await command_chain(config).execute_async(context)
        assert result.success is True
        requested = next(payload for kind, payload in events if kind == "command.approval_requested")
        resolved = next(payload for kind, payload in events if kind == "command.approval_resolved")
        assert "argv" not in requested
        assert "environment" not in requested
        assert "stdin" not in requested
        assert requested["argv_preview"][-1] == "<dynamic-arg-1>"
        assert requested["argv_preview"][1] == "<static-arg-1>"
        assert "dynamic-secret" not in repr(requested)
        assert "static-secret" not in repr(requested)
        assert resolved["fingerprint"] == requested["fingerprint"]
        assert resolved["status"] == "approved"

    async def test_missing_approval_handler_denies_before_prepare(self, recording_runtime):
        policy = CommandPolicy(
            approval_required_executables=frozenset({"dangerous"}),
            allowed_runtimes=frozenset({"recording"}),
            require_approval_for_interpreters=False,
        )
        sr = await run_command(
            CommandStepConfig(command=["dangerous"], runtime="recording"),
            policy=policy,
        )
        assert sr.success is False
        assert "not approved" in sr.error_message
        assert not recording_runtime.calls

    async def test_future_approval_is_awaited(self, recording_runtime):
        def approve_later(_request):
            loop = asyncio.get_running_loop()
            future = loop.create_future()
            loop.call_soon(future.set_result, True)
            return future

        policy = CommandPolicy(
            approval_required_executables=frozenset({"dangerous"}),
            allowed_runtimes=frozenset({"recording"}),
            require_approval_for_interpreters=False,
        )
        sr = await run_command(
            CommandStepConfig(command=["dangerous"], runtime="recording"),
            policy=policy,
            approval_handler=approve_later,
        )
        assert sr.success is True

    async def test_approval_timeout_fails_closed(self, recording_runtime):
        async def never(_request):
            await asyncio.sleep(1)
            return True

        policy = CommandPolicy(
            approval_required_executables=frozenset({"dangerous"}),
            allowed_runtimes=frozenset({"recording"}),
            require_approval_for_interpreters=False,
            approval_timeout=0.01,
        )
        sr = await run_command(
            CommandStepConfig(command=["dangerous"], runtime="recording"),
            policy=policy,
            approval_handler=never,
        )
        assert sr.success is False
        assert "approval timed out" in sr.error_message
        assert not recording_runtime.calls

    async def test_parallel_steps_keep_host_policy_and_use_unique_approval_ids(
        self,
        recording_runtime,
    ):
        requests: list[CommandApprovalRequest] = []

        async def approve(request: CommandApprovalRequest) -> bool:
            requests.append(request)
            return True

        policy = CommandPolicy(
            approval_required_executables=frozenset({"dangerous"}),
            allowed_runtimes=frozenset({"recording"}),
            require_approval_for_interpreters=False,
        )
        chain = ReasoningChain(
            steps=[
                create_step(
                    number,
                    f"command {number}",
                    StepType.COMMAND,
                    config=CommandStepConfig(
                        command=["dangerous"],
                        runtime="recording",
                    ),
                )
                for number in (1, 2)
            ]
        )
        result = await chain.execute_async(make_context(policy=policy, approval_handler=approve))
        assert result.success is True
        assert len(requests) == 2
        assert len({request.request_id for request in requests}) == 2

    def test_fingerprint_covers_env_and_stdin(self):
        base = dict(
            step_number=1,
            step_title="approve",
            static_argument_count=1,
            argv=["tool", "dynamic"],
            authorized_executable="tool",
            runtime="recording",
            working_dir=None,
            network="none",
            network_allowlist=[],
            resources={"timeout": 1.0},
            enforcement_mode="strict",
            reason="test",
        )
        first = CommandApprovalRequest.create(env={"A": "1"}, stdin=b"x", **base)
        changed_env = CommandApprovalRequest.create(env={"A": "2"}, stdin=b"x", **base)
        changed_stdin = CommandApprovalRequest.create(env={"A": "1"}, stdin=b"y", **base)
        assert len({first.fingerprint, changed_env.fingerprint, changed_stdin.fingerprint}) == 3
        assert len({first.request_id, changed_env.request_id, changed_stdin.request_id}) == 3

    def test_fingerprint_canonicalization_handles_surrogate_metadata(self):
        request = CommandApprovalRequest.create(
            step_number=1,
            step_title="approve",
            static_argument_count=1,
            argv=["tool"],
            authorized_executable="tool",
            env={},
            stdin=None,
            runtime="recording",
            working_dir="/workspace/\ud800",
            network="none",
            network_allowlist=[],
            resources={"timeout": 1.0},
            enforcement_mode="strict",
            reason="test",
        )
        assert request.fingerprint

    def test_policy_rejects_unapproved_best_effort_and_env(self):
        policy = CommandPolicy(
            allowed_executables=frozenset({"tool"}),
            allowed_runtimes=frozenset({"recording"}),
        )
        best_effort = policy.evaluate(
            ["tool"],
            runtime="recording",
            network="none",
            enforcement_mode="best_effort",
            working_dir=None,
            explicit_env_keys=(),
        )
        env = policy.evaluate(
            ["tool"],
            runtime="recording",
            network="none",
            enforcement_mode="strict",
            working_dir=None,
            explicit_env_keys=("LD_PRELOAD",),
        )
        assert best_effort.outcome == "deny"
        assert env.outcome == "deny"

    def test_policy_restricts_individual_network_hosts(self):
        policy = CommandPolicy(
            allowed_executables=frozenset({"tool"}),
            allowed_runtimes=frozenset({"docker"}),
            allowed_networks=frozenset({"allowlist"}),
            allowed_network_hosts=frozenset({"api.example.com"}),
        )
        allowed = policy.evaluate(
            ["tool"],
            runtime="docker",
            network="allowlist",
            network_allowlist=("API.EXAMPLE.COM.",),
            enforcement_mode="strict",
            working_dir=None,
            explicit_env_keys=(),
        )
        denied = policy.evaluate(
            ["tool"],
            runtime="docker",
            network="allowlist",
            network_allowlist=("other.example.com",),
            enforcement_mode="strict",
            working_dir=None,
            explicit_env_keys=(),
        )
        assert allowed.outcome == "allow"
        assert denied.outcome == "deny"
        assert "other.example.com" in denied.reason

    def test_one_policy_can_cover_local_and_sandbox_rules(self):
        local = shutil.which("true")
        assert local is not None
        policy = CommandPolicy(
            allowed_executables=frozenset({local, "python"}),
            allowed_runtimes=frozenset({"local", "docker"}),
            allowed_networks=frozenset({"host", "none"}),
            require_approval_for_interpreters=False,
        )
        local_decision = policy.evaluate(
            [local],
            runtime="local",
            network="host",
            enforcement_mode="strict",
            working_dir=None,
            explicit_env_keys=(),
        )
        docker_decision = policy.evaluate(
            ["python"],
            runtime="docker",
            network="none",
            enforcement_mode="strict",
            working_dir=None,
            explicit_env_keys=(),
        )
        assert local_decision.outcome == "allow"
        assert docker_decision.outcome == "allow"

    def test_command_step_rejects_cache(self):
        with pytest.raises(ValueError, match="cannot be cached"):
            CommandStepDescription(
                number=1,
                title="side effect",
                config=CommandStepConfig(command=["true"]),
                cache=StepCache(),
            )


# ---------------------------------------------------------------------------
# Serialization round-trip
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_host_authority_is_not_serialized_or_snapshotted(self):
        policy = CommandPolicy(
            allowed_executables=frozenset({"tool"}),
            allowed_runtimes=frozenset({"recording"}),
        )
        context = make_context(
            policy=policy,
            approval_handler=lambda _request: True,
        )
        dumped = context.model_dump()
        assert "command_policy" not in dumped
        assert "on_command_approval_requested" not in dumped
        snapshot = context.snapshot().model_dump()
        assert "command_policy" not in snapshot
        assert "on_command_approval_requested" not in snapshot

    def test_roundtrip_typed(self):
        chain = command_chain(
            CommandStepConfig(
                command=["grep", "-n", "TODO"],
                input_mapping={"file": "$history[-1]"},
                network="none",
                timeout=15.0,
            )
        )
        rebuilt = ReasoningChain.from_dict(chain.to_dict(), use_typed_steps=True)
        step = rebuilt.steps[0]
        assert isinstance(step, CommandStepDescription)
        assert step.config.command == ["grep", "-n", "TODO"]
        assert step.config.input_mapping == {"file": "$history[-1]"}
        assert step.config.timeout == 15.0

    def test_roundtrip_legacy_reconstruct(self):
        chain = command_chain(CommandStepConfig(command=["echo", "hi"]))
        rebuilt = ReasoningChain.from_dict(chain.to_dict())
        # Legacy path stores a reconstructed CommandStepConfig on step_config.
        step = rebuilt.steps[0]
        cfg = getattr(step, "config", None) or getattr(step, "step_config", None)
        assert cfg.command == ["echo", "hi"]

    def test_legacy_json_name_migrates_to_command(self):
        data = command_chain(CommandStepConfig(command=["echo", "hi"])).to_dict()
        data["steps"][0]["step_type"] = "bash"
        rebuilt = ReasoningChain.from_dict(data, use_typed_steps=True)
        assert isinstance(rebuilt.steps[0], CommandStepDescription)
        assert rebuilt.steps[0].step_type == StepType.COMMAND

    def test_legacy_step_description_converts_to_typed(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            legacy = StepDescription(
                number=1,
                title="legacy command",
                step_type=StepType.COMMAND,
                step_config=CommandStepConfig(command=["true"]),
            )
        assert isinstance(legacy.to_typed_step(), CommandStepDescription)
