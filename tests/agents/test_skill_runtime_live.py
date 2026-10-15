"""Live-daemon smoke tests for the sandbox skill runtimes.

CARE-M1 (Docker / Firejail) and CARE-M2 (E2B) — the unit suite proves
each runtime emits the right flags / dispatches the right SDK calls.
This live suite proves the same backends actually *work* end-to-end
against the real daemon / CLI / SaaS endpoint when available.

Marker
------
Tests are decorated with ``@pytest.mark.skill_runtime_live``. The
default ``pytest`` run deselects them so the hermetic-mock suite stays
fast. Opt in with::

    pytest -m skill_runtime_live tests/agents/

Each test gracefully **skips** (not fails) when its backend isn't
available — missing CLI, missing API key, etc. This keeps the live
suite safe to run on any host: Docker also requires the fixed image to be
cached; Linux+firejail hosts run the firejail test; E2B additionally requires
both ``E2B_API_KEY`` and the explicit ``CARL_RUN_E2B_LIVE=1`` cost opt-in.

Coverage scope
--------------
Docker tests use only the fixed ``python:3.12-slim`` image and first prove
that it is already present in the daemon's local cache.  The live suite never
intentionally pulls an image.  Every host/daemon/provider await also has a
test-side deadline independent of the runtime's own command timeout.

Besides the cheapest runtime smoke calls, Docker covers one complete
``CommandStep -> Docker -> portable artifact`` path and checks the effective
cgroup values from inside the container.  These are plumbing/enforcement
checks, not benchmarks.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import uuid
from collections.abc import Awaitable
from typing import Any

import pytest

from mmar_carl import (
    ArtifactInput,
    ArtifactOutput,
    ArtifactRecord,
    CommandPolicy,
    CommandStepConfig,
    DockerNetworkBinding,
    DockerSkillRuntime,
    E2BSkillRuntime,
    FirejailSkillRuntime,
    ManagedNetworkProfile,
    PreconfiguredNetworkEnforcer,
    ReasoningChain,
    ReasoningContext,
    StepType,
    create_step,
)
from mmar_carl.models.llm_client_base import LLMClientBase

_DOCKER_IMAGE = "python:3.12-slim"
_HOST_PREFLIGHT_TIMEOUT_S = 10.0
_RUNTIME_PREPARE_TIMEOUT_S = 5.0
_RUNTIME_COMMAND_TIMEOUT_S = 6.0
_RUNTIME_HOST_DEADLINE_S = 12.0
_RUNTIME_CLEANUP_TIMEOUT_S = 6.0
_CHAIN_HOST_DEADLINE_S = 18.0

class _NoLLMClient(LLMClientBase):
    """Make an accidental LLM call fail instead of reaching a provider."""

    async def get_response(self, prompt: str) -> str:
        raise AssertionError("live runtime smoke tests must not call an LLM")

    async def get_response_with_retries(
        self,
        prompt: str,
        retries: int = 3,
    ) -> str:
        raise AssertionError("live runtime smoke tests must not call an LLM")


async def _with_host_deadline[T](
    awaitable: Awaitable[T],
    *,
    timeout: float,
    operation: str,
) -> T:
    """Apply a test-harness deadline in addition to backend timeouts."""

    try:
        return await asyncio.wait_for(awaitable, timeout=timeout)
    except TimeoutError:
        pytest.fail(f"{operation} exceeded the {timeout:g}s live-test host deadline")


async def _host_exec(
    *argv: str,
    timeout: float = _HOST_PREFLIGHT_TIMEOUT_S,
) -> tuple[int, bytes, bytes]:
    """Run one bounded, read-only host preflight command."""

    proc: asyncio.subprocess.Process | None = None
    try:
        async with asyncio.timeout(timeout):
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
    except TimeoutError:
        if proc is not None and proc.returncode is None:
            proc.kill()
            try:
                await asyncio.wait_for(proc.wait(), timeout=1.0)
            except TimeoutError:
                pass
        raise
    assert proc.returncode is not None
    return proc.returncode, stdout, stderr


async def _require_cached_docker_image() -> str:
    """Skip without pulling when the daemon or fixed cached image is absent."""

    docker = shutil.which("docker")
    if docker is None:
        pytest.skip("docker CLI not on PATH")

    try:
        info_rc, _info_out, info_err = await _host_exec(
            docker,
            "info",
            "--format",
            "{{.ServerVersion}}",
        )
    except TimeoutError:
        pytest.skip("docker daemon preflight timed out")
    if info_rc != 0:
        detail = info_err.decode("utf-8", errors="replace").strip()[:200]
        pytest.skip(f"docker daemon unavailable: {detail or 'docker info failed'}")

    try:
        image_rc, image_out, image_err = await _host_exec(
            docker,
            "image",
            "ls",
            "--no-trunc",
            "--filter",
            f"reference={_DOCKER_IMAGE}",
            "--format",
            "{{.ID}}",
        )
    except TimeoutError:
        pytest.skip("docker image-cache preflight timed out")
    if image_rc != 0:
        detail = image_err.decode("utf-8", errors="replace").strip()[:200]
        pytest.skip(
            f"fixed live-test image {_DOCKER_IMAGE!r} is not cached; "
            f"the suite will not pull it ({detail or 'image inspect failed'})"
        )
    image_ids = {
        line.strip()
        for line in image_out.decode("ascii", errors="strict").splitlines()
        if line.strip()
    }
    assert len(image_ids) == 1, (
        "cached Docker tag must resolve to exactly one immutable image id"
    )
    image_id = image_ids.pop()
    assert image_id.startswith("sha256:"), "docker image ls returned no image id"

    # Verify the immutable ID itself is readable before any live run. Docker
    # Desktop/containerd installations may reject inspect-by-tag even though
    # image ls and inspect-by-ID both work.
    verify_rc, _verify_out, verify_err = await _host_exec(
        docker, "inspect", "--type", "image", image_id,
    )
    assert verify_rc == 0, (
        "cached immutable Docker image cannot be inspected: "
        + verify_err.decode("utf-8", errors="replace")[:200]
    )
    return image_id


async def _prepare_runtime(
    runtime: Any,
    workspace: Any,
    config: dict[str, Any],
) -> Any:
    return await _with_host_deadline(
        runtime.prepare(skill=None, workspace=workspace, config=config),
        timeout=_RUNTIME_PREPARE_TIMEOUT_S,
        operation=f"{runtime.name} prepare",
    )


async def _cleanup_runtime(runtime: Any, handle: Any) -> None:
    await _with_host_deadline(
        runtime.cleanup(handle),
        timeout=_RUNTIME_CLEANUP_TIMEOUT_S,
        operation=f"{runtime.name} cleanup",
    )


def _docker_command_policy() -> CommandPolicy:
    return CommandPolicy(
        allowed_executables=frozenset({"python"}),
        allowed_runtimes=frozenset({"docker"}),
        allowed_networks=frozenset({"none"}),
        require_approval_for_interpreters=False,
    )


async def _run_docker_command_step(
    config: CommandStepConfig,
    *,
    policy: CommandPolicy | None = None,
    network_enforcer: Any = None,
) -> Any:
    """Execute one real Docker CommandStep without any LLM/provider call."""

    await _require_cached_docker_image()
    context = ReasoningContext(
        outer_context="live-runtime-smoke",
        api=_NoLLMClient(),
        model="no-llm",
        command_policy=policy or _docker_command_policy(),
        network_enforcer=network_enforcer,
    )
    chain = ReasoningChain(
        steps=[create_step(1, "docker-live", StepType.COMMAND, config=config)],
    )
    result = await _with_host_deadline(
        chain.execute_async(context),
        timeout=_CHAIN_HOST_DEADLINE_S,
        operation="CommandStep Docker E2E",
    )
    assert len(result.step_results) == 1
    return result.step_results[0]


def _prefixed_json(stdout: str, prefix: str) -> dict[str, Any]:
    matching = [line for line in stdout.splitlines() if line.startswith(prefix)]
    assert len(matching) == 1, f"expected exactly one {prefix!r} line in {stdout!r}"
    return json.loads(matching[0][len(prefix):])


# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------


@pytest.mark.skill_runtime_live
@pytest.mark.asyncio
async def test_docker_runtime_runs_echo(tmp_path) -> None:
    """Run ``echo hello`` in the fixed, already-cached Docker image."""

    image = await _require_cached_docker_image()
    runtime = DockerSkillRuntime()
    handle = await _prepare_runtime(
        runtime,
        tmp_path / "ws",
        # The runtime itself emits ``--pull=never``; the cache preflight makes
        # the skip reason explicit before a container invocation is attempted.
        {"image": image},
    )
    try:
        result = await _with_host_deadline(
            runtime.run(
                handle,
                ["echo", "hello-from-docker"],
                timeout=_RUNTIME_COMMAND_TIMEOUT_S,
            ),
            timeout=_RUNTIME_HOST_DEADLINE_S,
            operation="docker echo",
        )
        assert result.exit_code == 0, (
            "docker echo failed: "
            + result.stderr.decode("utf-8", errors="replace")[:300]
        )
        assert b"hello-from-docker" in result.stdout
    finally:
        await _cleanup_runtime(runtime, handle)


@pytest.mark.skill_runtime_live
@pytest.mark.asyncio
async def test_docker_runtime_enforces_network_none(tmp_path) -> None:
    """Prove Python starts, routes stay on loopback, and numeric egress is blocked."""

    image = await _require_cached_docker_image()
    runtime = DockerSkillRuntime()
    handle = await _prepare_runtime(
        runtime,
        tmp_path / "ws",
        {"image": image, "network": "none"},
    )
    try:
        script = (
            "import json, socket\n"
            "from pathlib import Path\n"
            "interfaces = sorted(name for _index, name in socket.if_nameindex())\n"
            "ipv4 = [line.split()[0] for line in "
            "Path('/proc/net/route').read_text().splitlines()[1:] "
            "if len(line.split()) > 1 and line.split()[1] == '00000000']\n"
            "ipv6 = [line.split()[-1] for line in "
            "Path('/proc/net/ipv6_route').read_text().splitlines() "
            "if line.split() and line.split()[0] == '0' * 32]\n"
            "print('CARL_NETWORK_SENTINEL=' + "
            "json.dumps({'interfaces': interfaces, "
            "'default_routes': ipv4 + ipv6}), flush=True)\n"
            "try:\n"
            "    socket.create_connection(('1.1.1.1', 53), timeout=1.0)\n"
            "except OSError as exc:\n"
            "    print('CARL_NETWORK_BLOCKED=' + type(exc).__name__, flush=True)\n"
            "    raise SystemExit(23)\n"
            "raise SystemExit(0)\n"
        )
        result = await _with_host_deadline(
            runtime.run(
                handle,
                ["python", "-c", script],
                timeout=_RUNTIME_COMMAND_TIMEOUT_S,
            ),
            timeout=_RUNTIME_HOST_DEADLINE_S,
            operation="docker network-none probe",
        )
        stdout = result.stdout.decode("utf-8", errors="replace")
        sentinel = _prefixed_json(stdout, "CARL_NETWORK_SENTINEL=")
        assert "lo" in sentinel["interfaces"]
        assert set(sentinel["default_routes"]) <= {"lo"}, (
            "Docker network=none exposed a non-loopback default route: "
            f"{sentinel['default_routes']!r}"
        )
        assert "CARL_NETWORK_BLOCKED=" in stdout
        assert result.exit_code == 23, (
            "the probe did not reach the expected blocked-egress branch; "
            f"exit={result.exit_code}, stdout={stdout!r}, stderr={result.stderr!r}"
        )
    finally:
        await _cleanup_runtime(runtime, handle)


@pytest.mark.skill_runtime_live
@pytest.mark.asyncio
async def test_command_step_docker_round_trips_artifact() -> None:
    """Exercise policy, staging, Docker, collection, hash, and cleanup."""

    script = (
        "import os\n"
        "from pathlib import Path\n"
        "source = Path(os.environ['CARL_ARTIFACT_IN_SOURCE']).read_bytes()\n"
        "Path(os.environ['CARL_ARTIFACT_OUT_COPY']).write_bytes(b'docker:' + source)\n"
        "print('CARL_COMMAND_SENTINEL', flush=True)\n"
    )
    config = CommandStepConfig(
        command=["python", "-c", script],
        runtime="docker",
        network="none",
        timeout=_RUNTIME_COMMAND_TIMEOUT_S,
        artifact_io_timeout=_RUNTIME_COMMAND_TIMEOUT_S,
        artifact_inputs=[
            ArtifactInput(
                name="source",
                source="'artifact-input'",
                path="source.txt",
                media_type="text/plain",
            ),
        ],
        artifact_outputs=[
            ArtifactOutput(
                name="copy",
                path="copy.bin",
                media_type="application/octet-stream",
            ),
        ],
    )

    step_result = await _run_docker_command_step(config)
    assert step_result.success, step_result.error_message
    assert "CARL_COMMAND_SENTINEL" in step_result.result_data["stdout"]
    assert step_result.result_data["network_enforced"] is True
    report = step_result.result_data["enforcement_report"]
    assert report["controls"]["workspace_files"] == "enforced"
    assert report["controls"]["artifact_output_limit"] == "enforced"
    record = ArtifactRecord.model_validate(
        step_result.result_data["artifacts"]["copy"],
    )
    assert record.decode(max_bytes=100) == b"docker:artifact-input"


@pytest.mark.skill_runtime_live
@pytest.mark.asyncio
async def test_command_step_docker_cgroup_limits_are_effective() -> None:
    """Read effective CPU, memory, and PID limits from inside the container."""

    script = (
        "import json\n"
        "from pathlib import Path\n"
        "def first(paths):\n"
        "    for raw in paths:\n"
        "        path = Path(raw)\n"
        "        if path.is_file():\n"
        "            return {'path': raw, 'value': path.read_text().strip()}\n"
        "    return None\n"
        "payload = {\n"
        "    'memory': first(['/sys/fs/cgroup/memory.max', "
        "'/sys/fs/cgroup/memory/memory.limit_in_bytes']),\n"
        "    'pids': first(['/sys/fs/cgroup/pids.max', "
        "'/sys/fs/cgroup/pids/pids.max']),\n"
        "    'cpu_max': first(['/sys/fs/cgroup/cpu.max']),\n"
        "    'cpu_quota': first(['/sys/fs/cgroup/cpu/cpu.cfs_quota_us', "
        "'/sys/fs/cgroup/cpu.cfs_quota_us']),\n"
        "    'cpu_period': first(['/sys/fs/cgroup/cpu/cpu.cfs_period_us', "
        "'/sys/fs/cgroup/cpu.cfs_period_us']),\n"
        "}\n"
        "print('CARL_CGROUP_SENTINEL=' + json.dumps(payload), flush=True)\n"
    )
    config = CommandStepConfig(
        command=["python", "-c", script],
        runtime="docker",
        network="none",
        timeout=_RUNTIME_COMMAND_TIMEOUT_S,
        cpu_limit=0.5,
        mem_limit="64m",
        pids_limit=32,
    )

    step_result = await _run_docker_command_step(config)
    assert step_result.success, step_result.error_message
    payload = _prefixed_json(
        step_result.result_data["stdout"],
        "CARL_CGROUP_SENTINEL=",
    )

    assert payload["memory"] is not None, "container exposes no memory cgroup limit"
    assert int(payload["memory"]["value"]) == 64 * 1024 * 1024
    assert payload["pids"] is not None, "container exposes no PID cgroup limit"
    assert int(payload["pids"]["value"]) == 32

    if payload["cpu_max"] is not None:
        quota_text, period_text = payload["cpu_max"]["value"].split()
        assert quota_text != "max", "container CPU cgroup is unlimited"
        cpu_ratio = int(quota_text) / int(period_text)
    else:
        assert payload["cpu_quota"] is not None, (
            "container exposes neither cgroup-v2 cpu.max nor cgroup-v1 quota"
        )
        assert payload["cpu_period"] is not None
        cpu_ratio = (
            int(payload["cpu_quota"]["value"])
            / int(payload["cpu_period"]["value"])
        )
    assert cpu_ratio == pytest.approx(0.5, abs=0.01)

    report = step_result.result_data["enforcement_report"]
    assert report["controls"]["cpu_limit"] == "enforced"
    assert report["controls"]["memory_limit"] == "enforced"
    assert report["controls"]["pids_limit"] == "enforced"


@pytest.mark.skill_runtime_live
@pytest.mark.asyncio
async def test_command_step_docker_uses_managed_internal_egress_profile() -> None:
    """Reach one managed destination while public numeric egress stays blocked.

    This is a live proof of CARL's typed binding and Docker topology plumbing,
    not a general public-Internet hostname firewall.  The operator-owned
    profile is an internal Docker network containing exactly one destination
    container with the approved DNS alias.
    """

    image = await _require_cached_docker_image()
    docker = shutil.which("docker")
    assert docker is not None
    suffix = uuid.uuid4().hex[:10]
    network_name = f"carl-egress-live-{suffix}"
    server_name = f"carl-egress-dst-{suffix}"
    try:
        rc, _stdout, stderr = await _host_exec(
            docker, "network", "create", "--internal", network_name,
        )
        assert rc == 0, stderr.decode("utf-8", errors="replace")[:300]
        rc, _stdout, stderr = await _host_exec(
            docker,
            "run",
            "--detach",
            "--rm",
            "--pull=never",
            "--memory",
            "64m",
            "--memory-swap",
            "64m",
            "--cpus",
            "0.25",
            "--pids-limit",
            "32",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            "--read-only",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,nodev,size=8m",
            "--network",
            network_name,
            "--network-alias",
            "allowed.test",
            "--name",
            server_name,
            image,
            "python",
            "-u",
            "-m",
            "http.server",
            "8080",
            "--bind",
            "0.0.0.0",
        )
        assert rc == 0, stderr.decode("utf-8", errors="replace")[:300]

        script = (
            "import socket, time\n"
            "deadline = time.monotonic() + 3\n"
            "while True:\n"
            "    try:\n"
            "        conn = socket.create_connection(('allowed.test', 8080), 1)\n"
            "        break\n"
            "    except OSError:\n"
            "        if time.monotonic() >= deadline: raise\n"
            "        time.sleep(0.05)\n"
            "conn.sendall(b'GET / HTTP/1.0\\r\\nHost: allowed.test\\r\\n\\r\\n')\n"
            "reply = conn.recv(64)\n"
            "conn.close()\n"
            "assert reply.startswith(b'HTTP/1.0 200')\n"
            "print('CARL_ALLOWED_DESTINATION_OK', flush=True)\n"
            "try:\n"
            "    socket.create_connection(('1.1.1.1', 53), 1)\n"
            "except OSError:\n"
            "    print('CARL_PUBLIC_EGRESS_BLOCKED', flush=True)\n"
            "else:\n"
            "    raise SystemExit('unexpected public egress')\n"
        )
        enforcer = PreconfiguredNetworkEnforcer(
            "live-docker-egress",
            "r1",
            (
                ManagedNetworkProfile(
                    profile="single-internal-destination",
                    runtime="docker",
                    hosts=("allowed.test",),
                    binding=DockerNetworkBinding(network_name),
                ),
            ),
        )
        policy = CommandPolicy(
            allowed_executables=frozenset({"python"}),
            allowed_runtimes=frozenset({"docker"}),
            allowed_networks=frozenset({"allowlist"}),
            allowed_network_hosts=frozenset({"allowed.test"}),
            require_approval_for_interpreters=False,
        )
        step_result = await _run_docker_command_step(
            CommandStepConfig(
                command=["python", "-c", script],
                runtime="docker",
                network="allowlist",
                network_allowlist=["allowed.test"],
                timeout=_RUNTIME_COMMAND_TIMEOUT_S,
            ),
            policy=policy,
            network_enforcer=enforcer,
        )

        assert step_result.success, step_result.error_message
        assert "CARL_ALLOWED_DESTINATION_OK" in step_result.result_data["stdout"]
        assert "CARL_PUBLIC_EGRESS_BLOCKED" in step_result.result_data["stdout"]
        assert step_result.result_data["network_enforced"] is True
        assert step_result.result_data["network_enforcement"]["profile"] == (
            "single-internal-destination"
        )
    finally:
        # Always attempt both cleanups: a timed-out CLI can still have
        # completed daemon-side after the client deadline.
        try:
            await _host_exec(docker, "rm", "--force", server_name)
        finally:
            await _host_exec(docker, "network", "rm", network_name)


# ---------------------------------------------------------------------------
# Firejail (Linux only)
# ---------------------------------------------------------------------------


@pytest.mark.skill_runtime_live
@pytest.mark.asyncio
async def test_firejail_runtime_runs_echo(tmp_path) -> None:
    """Run ``echo hello`` via the real ``firejail`` CLI."""
    if shutil.which("firejail") is None:
        pytest.skip("firejail CLI not on PATH (Linux-only)")

    runtime = FirejailSkillRuntime()
    handle = await _prepare_runtime(
        runtime,
        tmp_path / "ws",
        {},
    )
    try:
        result = await _with_host_deadline(
            runtime.run(
                handle,
                ["echo", "hello-from-firejail"],
                timeout=_RUNTIME_COMMAND_TIMEOUT_S,
            ),
            timeout=_RUNTIME_HOST_DEADLINE_S,
            operation="firejail echo",
        )
        assert result.exit_code == 0, (
            "firejail run failed: "
            + result.stderr.decode("utf-8", errors="replace")[:300]
        )
        assert b"hello-from-firejail" in result.stdout
    finally:
        await _cleanup_runtime(runtime, handle)


# ---------------------------------------------------------------------------
# E2B (requires API key)
# ---------------------------------------------------------------------------


@pytest.mark.skill_runtime_live
@pytest.mark.asyncio
async def test_e2b_runtime_runs_echo(tmp_path) -> None:
    """Spin up a real e2b sandbox and run ``echo``.

    Skips when:
    - ``CARL_RUN_E2B_LIVE=1`` isn't explicitly set (paid-provider guard).
    - ``E2B_API_KEY`` isn't set in the environment.
    - The ``e2b`` Python SDK isn't installed.
    """
    if os.environ.get("CARL_RUN_E2B_LIVE") != "1":
        pytest.skip("set CARL_RUN_E2B_LIVE=1 to authorize the provider smoke call")
    if not os.environ.get("E2B_API_KEY"):
        pytest.skip("E2B_API_KEY not set — e2b sandbox unreachable")
    try:
        import e2b  # noqa: F401
    except ImportError:
        pytest.skip("e2b SDK not installed (`pip install e2b`)")

    runtime = E2BSkillRuntime()
    handle = await _with_host_deadline(
        runtime.prepare(
            skill=None,
            workspace=tmp_path / "ws",
            config={"extra_create_kwargs": {"timeout": 60}},
        ),
        timeout=30.0,
        operation="e2b prepare",
    )
    try:
        result = await _with_host_deadline(
            runtime.run(
                handle,
                ["echo", "hello-from-e2b"],
                timeout=_RUNTIME_COMMAND_TIMEOUT_S,
            ),
            timeout=_RUNTIME_HOST_DEADLINE_S,
            operation="e2b echo",
        )
        assert result.exit_code == 0, (
            "e2b sandbox run failed: "
            + result.stderr.decode("utf-8", errors="replace")[:300]
        )
        assert b"hello-from-e2b" in result.stdout
    finally:
        await _cleanup_runtime(runtime, handle)


# ---------------------------------------------------------------------------
# Hermetic guards — these run in the default suite and assert the
# marker plumbing is intact so a contributor who drops the marker from
# a test surfaces it here.
# ---------------------------------------------------------------------------


def test_skill_runtime_live_marker_declared_in_pyproject() -> None:
    from pathlib import Path

    pyproject = (
        Path(__file__).resolve().parents[2] / "pyproject.toml"
    ).read_text(encoding="utf-8")
    assert "skill_runtime_live:" in pyproject, (
        "The `skill_runtime_live` marker must be declared in "
        "pyproject.toml's [tool.pytest.ini_options] markers."
    )
    assert "not skill_runtime_live" in pyproject, (
        "The default pytest addopts must deselect `skill_runtime_live` "
        "so the hermetic suite stays fast."
    )


def test_every_live_test_carries_marker() -> None:
    """Every async test in this module must remain explicitly opt-in."""
    import ast
    from pathlib import Path

    text = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(text)
    async_tests = [
        node
        for node in tree.body
        if isinstance(node, ast.AsyncFunctionDef) and node.name.startswith("test_")
    ]
    assert async_tests, "live runtime module unexpectedly contains no async tests"
    for node in async_tests:
        decorators = {
            ast.unparse(decorator)
            for decorator in node.decorator_list
        }
        assert "pytest.mark.skill_runtime_live" in decorators, (
            f"{node.name} hits a real backend but lacks the opt-in live marker"
        )
        assert "pytest.mark.asyncio" in decorators, (
            f"{node.name} is async but lacks the pytest asyncio marker"
        )
