"""E2BSkillRuntime — remote SaaS sandbox for AgentSkills via e2b.dev.

Sister to :class:`DockerSkillRuntime` /
:class:`FirejailSkillRuntime`, but the sandbox is a hosted micro-VM
managed by `E2B <https://e2b.dev/>`_ instead of a local container or
Linux process. Per-call latency is higher (network round-trip), but
isolation is the strongest of the three reference backends — sandboxes
are full VMs with no host filesystem access.

Implementation note
-------------------
The runtime self-registers as ``"e2b"`` on import. The actual ``e2b``
SDK is imported *lazily* inside :meth:`prepare`, so ``mmar-carl`` users
without the ``mmar-carl[e2b]`` extra never pay the import cost. Missing SDK
→ ``SkillRuntimeError`` with the canonical install hint.

A ``sandbox_factory`` kwarg lets tests inject a mock sandbox class
without having to install the SDK or hit the e2b network. Production
callers leave it as the default ``None``, in which case the lazy
import path runs.

Sandbox protocol
----------------
The runtime expects the injected / SDK-provided sandbox object to
support (informally — duck-typed):

* ``await Sandbox.create(template=..., api_key=...)`` — async factory.
* ``await sandbox.commands.run(cmd, background=True, ...)`` — returns a
  command handle supporting ``send_stdin`` / ``close_stdin`` / ``wait`` /
  ``kill``. ``wait`` yields stdout, stderr, and exit status.
* ``await sandbox.files.write(path, content)`` — write a bytes/str
  file at an absolute path inside the sandbox.
* ``await sandbox.files.read(path)`` — read a bytes/str file.
* ``await sandbox.kill()`` — terminate.

These are the names used by the current ``e2b`` Python SDK (`v1+`).
The runtime is forgiving about return-value shapes: a result with
``.stdout`` (bytes/str) is accepted whether ``.exit_code`` is on the
result or buried in ``.error``.

``runtime_config`` recognised keys
----------------------------------

* ``template`` — e2b sandbox template id. Default ``"base"``.
* ``api_key_env`` — env var name that holds the e2b API key. Default
  ``"E2B_API_KEY"``. The runtime reads ``os.environ[api_key_env]`` and
  passes the value to ``Sandbox.create``.
* ``extra_create_kwargs`` — bounded ``timeout`` and string ``metadata`` only.
  Security, lifecycle, network, volume, MCP, and connection controls remain
  host-owned and cannot be replaced by serialized runtime configuration.
* ``workspace_root_in_sandbox`` — absolute path inside the sandbox
  that maps to the host workspace. Default ``"/workspace"``. The
  runtime mirrors host files at this path via the SDK's ``files``
  API on each ``write_file`` / ``read_file``.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import tempfile
import uuid
import warnings
from pathlib import Path, PurePosixPath
from typing import Any, Callable, ClassVar, Optional

from .skill_runtime import (
    RuntimeCapabilities,
    RuntimeRunResult,
    SkillRuntimeError,
    SkillRuntimeHandle,
    _remove_workspace_tree,
    register_skill_runtime,
    resolve_network_policy,
)

_log = logging.getLogger(__name__)


_DEFAULT_TEMPLATE = "base"
_DEFAULT_API_KEY_ENV = "E2B_API_KEY"
_DEFAULT_WS_IN_SANDBOX = "/workspace"
_E2B_CREATE_REQUEST_TIMEOUT_S = 30.0
_E2B_MAX_SANDBOX_LIFETIME_S = 300
_E2B_EXEC_PYTHON = "/usr/bin/python3"
_E2B_LAUNCHER_ENV = {"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"}
_E2B_EXEC_HELPER = b"""\
import json
import os
import sys

with open(sys.argv[1], encoding="utf-8") as request_file:
    request = json.load(request_file)
argv = request.get("argv")
env = request.get("env")
if not isinstance(argv, list) or not argv or not all(isinstance(v, str) for v in argv):
    raise SystemExit("invalid CARL argv request")
if not isinstance(env, dict) or not all(
    isinstance(k, str) and isinstance(v, str) for k, v in env.items()
):
    raise SystemExit("invalid CARL environment request")
os.execvpe(argv[0], argv, env)
"""


class E2BSkillRuntime:
    """Run skill scripts inside an e2b.dev hosted micro-VM sandbox.

    The sandbox lives for the duration of the handle (created in
    :meth:`prepare`, killed in :meth:`cleanup`). Every ``run`` reuses
    the same sandbox — distinct from the per-call container model of
    :class:`DockerSkillRuntime`. Restricted policies are passed through the
    E2B 2.x create API: ``none`` disables Internet access and ``allowlist``
    supplies an exact outbound selector. Provider lifecycle and total runtime
    deadlines remain separate advisory controls.

    Parameters
    ----------
    sandbox_factory:
        Optional injection hook. When supplied, ``prepare()`` calls
        ``await sandbox_factory(**create_kwargs)`` instead of importing
        the e2b SDK. Used by tests to swap in a mock sandbox; do NOT
        use in production.
    """

    name: ClassVar[str] = "e2b"
    capabilities: ClassVar[RuntimeCapabilities] = RuntimeCapabilities(
        isolation="microvm",
        # The provider timeout does not cover helper staging/cleanup, and a
        # failed remote kill is not yet attested. A live-test harness deadline
        # does not turn that end-to-end lifecycle into a runtime guarantee.
        wall_time="advisory",
        # The SDK hands CARL a complete response, so slicing is post-hoc.
        output_limit="advisory",
        workspace_files="enforced",
        # The supported SDK streams with a hard collector cap. Keep this
        # conservative because injected legacy shims fall back to buffered
        # reads and remote stat/read are not one descriptor-pinned operation.
        artifact_output_limit="advisory",
        network_none="enforced",
        network_allowlist="enforced",
    )

    def __init__(
        self,
        *,
        sandbox_factory: Optional[Callable[..., Any]] = None,
    ) -> None:
        self._sandbox_factory = sandbox_factory

    async def prepare(
        self,
        skill: Any,
        workspace: Optional[Path],
        config: dict[str, Any],
    ) -> SkillRuntimeHandle:
        manifest_allowed = config.get("_manifest_allowed_tools")
        policy, allowlist = resolve_network_policy(
            config, manifest_allowed_tools=manifest_allowed,
        )
        if policy == "host":
            warnings.warn(
                "E2BSkillRuntime selected network='host' — unrestricted sandbox egress is enabled.",
                UserWarning,
                stacklevel=3,
            )

        template = config.get("template", _DEFAULT_TEMPLATE)
        api_key_env = config.get("api_key_env", _DEFAULT_API_KEY_ENV)
        api_key = os.environ.get(api_key_env)
        if api_key is None:
            raise SkillRuntimeError(
                f"E2BSkillRuntime requires an API key in the "
                f"{api_key_env!r} environment variable. Sign up at "
                f"https://e2b.dev and export the key, or override "
                f"`runtime_config['api_key_env']`."
            )

        ws_in_sandbox = config.get(
            "workspace_root_in_sandbox", _DEFAULT_WS_IN_SANDBOX,
        )
        sandbox_root = PurePosixPath(str(ws_in_sandbox))
        if not sandbox_root.is_absolute() or ".." in sandbox_root.parts:
            raise SkillRuntimeError(
                "workspace_root_in_sandbox must be an absolute sandbox path"
            )
        extra_create_kwargs = dict(config.get("extra_create_kwargs") or {})
        allowed_create_keys = {"metadata", "timeout"}
        unsupported_keys = extra_create_kwargs.keys() - allowed_create_keys
        if unsupported_keys:
            raise SkillRuntimeError(
                "E2B extra_create_kwargs may contain only host-bounded "
                f"'metadata' and 'timeout'; rejected: {sorted(unsupported_keys)!r}"
            )
        timeout_value = extra_create_kwargs.get("timeout")
        if timeout_value is not None and (
            isinstance(timeout_value, bool)
            or not isinstance(timeout_value, int)
            or not 1 <= timeout_value <= _E2B_MAX_SANDBOX_LIFETIME_S
        ):
            raise SkillRuntimeError(
                "E2B sandbox timeout must be an integer from 1 to "
                f"{_E2B_MAX_SANDBOX_LIFETIME_S} seconds"
            )
        metadata = extra_create_kwargs.get("metadata")
        if metadata is not None and (
            not isinstance(metadata, dict)
            or not all(
                isinstance(key, str) and isinstance(value, str)
                for key, value in metadata.items()
            )
        ):
            raise SkillRuntimeError("E2B sandbox metadata must be a string-to-string mapping")
        # Force security and timeout teardown regardless of runtime config.
        extra_create_kwargs.setdefault("timeout", _E2B_MAX_SANDBOX_LIFETIME_S)
        extra_create_kwargs["request_timeout"] = _E2B_CREATE_REQUEST_TIMEOUT_S
        extra_create_kwargs["secure"] = True
        extra_create_kwargs["lifecycle"] = {"on_timeout": "kill"}
        if policy == "none":
            extra_create_kwargs["allow_internet_access"] = False
        elif policy == "allowlist":
            extra_create_kwargs["network"] = {
                "allow_out": list(allowlist),
                "allow_public_traffic": False,
            }

        factory = self._sandbox_factory or self._default_sandbox_factory
        created_local_workspace = False
        create_task = asyncio.ensure_future(
            factory(template=template, api_key=api_key, **extra_create_kwargs)
        )
        try:
            sandbox, create_cancellation = await self._finish_shielded_task(
                create_task
            )
        except SkillRuntimeError:
            raise
        except Exception as exc:
            raise SkillRuntimeError(
                f"E2BSkillRuntime: failed to start sandbox "
                f"(template={template!r}): {exc}"
            ) from exc

        if create_cancellation is not None:
            # The provider may complete creation after the caller cancels.
            # Finish the same bounded create request, kill the returned paid
            # resource under repeated cancellation, then preserve the original
            # cancellation identity. Provider-side timeout+kill limits the
            # residual server race where no client ever receives an ID.
            kill_task = asyncio.ensure_future(sandbox.kill())
            try:
                await self._finish_shielded_task(kill_task)
            except Exception:  # noqa: BLE001
                _log.warning(
                    "E2BSkillRuntime: sandbox.kill() failed after create cancellation",
                    exc_info=True,
                )
            raise create_cancellation

        try:
            for path in (
                str(sandbox_root),
                str(sandbox_root / "in"),
                str(sandbox_root / "out"),
            ):
                await sandbox.files.make_dir(path)

            if workspace is None:
                workspace = Path(tempfile.mkdtemp(prefix="carl_skill_e2b_"))
                created_local_workspace = True
            workspace.mkdir(parents=True, exist_ok=True)
            workspace_in = workspace / "in"
            workspace_out = workspace / "out"
            workspace_in.mkdir(exist_ok=True)
            workspace_out.mkdir(exist_ok=True)
        except asyncio.CancelledError as cancellation:
            # No handle exists yet, so the executor cannot clean this paid
            # remote resource for us. Finish the same kill + local cleanup
            # sequence despite repeated cancellation, then preserve the first
            # cancellation identity.
            task = asyncio.current_task()
            if task is not None:
                task.uncancel()

            async def cleanup_partial_prepare() -> None:
                try:
                    await sandbox.kill()
                finally:
                    if created_local_workspace and workspace is not None:
                        await _remove_workspace_tree(workspace)

            cleanup_task = asyncio.ensure_future(cleanup_partial_prepare())
            try:
                await self._finish_shielded_task(cleanup_task)
            except Exception:  # noqa: BLE001
                _log.warning(
                    "E2BSkillRuntime: cleanup failed after prepare cancellation",
                    exc_info=True,
                )
            raise cancellation
        except Exception as exc:
            async def cleanup_failed_prepare() -> None:
                try:
                    await sandbox.kill()
                finally:
                    if created_local_workspace and workspace is not None:
                        await _remove_workspace_tree(workspace)

            cleanup_task = asyncio.ensure_future(cleanup_failed_prepare())
            cleanup_cancellation: asyncio.CancelledError | None = None
            try:
                _, cleanup_cancellation = await self._finish_shielded_task(
                    cleanup_task
                )
            except Exception:  # noqa: BLE001
                _log.warning(
                    "E2BSkillRuntime: sandbox.kill() failed after prepare error",
                    exc_info=True,
                )
            if cleanup_cancellation is not None:
                raise cleanup_cancellation
            raise SkillRuntimeError(
                f"E2BSkillRuntime: failed to initialize workspace: {exc}"
            ) from exc

        return SkillRuntimeHandle(
            workspace_root=workspace,
            workspace_in=workspace_in,
            workspace_out=workspace_out,
            backend={
                "isolation": "e2b",
                "template": template,
                "api_key_env": api_key_env,
                "workspace_root_in_sandbox": str(sandbox_root),
                "sandbox": sandbox,
                "network_policy": policy,
                "network_allowlist": allowlist,
                "network_enforced": policy in ("none", "allowlist"),
                "workspace_root_in_runtime": str(sandbox_root),
                "workspace_in_in_runtime": str(sandbox_root / "in"),
                "workspace_out_in_runtime": str(sandbox_root / "out"),
            },
        )

    async def run(
        self,
        handle: SkillRuntimeHandle,
        cmd: list[str],
        *,
        env: Optional[dict[str, str]] = None,
        stdin: Optional[bytes] = None,
        timeout: Optional[float] = None,
        cwd: Optional[str] = None,
    ) -> RuntimeRunResult:
        sandbox = handle.backend.get("sandbox")
        if sandbox is None:
            raise SkillRuntimeError(
                "E2BSkillRuntime: handle has no live sandbox. Did "
                "`prepare()` complete successfully?"
            )

        # E2B's current SDK accepts only a command string and starts it through
        # a login shell. Never put chain-controlled argv/env in that string:
        # stage a structured request and invoke a fixed helper that calls
        # execvpe(), preserving argv boundaries and excluding shell builtins,
        # functions, substitutions, and metacharacter interpretation.
        invocation_id = uuid.uuid4().hex
        sandbox_root = PurePosixPath(
            str(handle.backend.get("workspace_root_in_sandbox", _DEFAULT_WS_IN_SANDBOX))
        )
        helper_path = str(sandbox_root / f".carl-exec-{invocation_id}.py")
        request_path = str(sandbox_root / f".carl-exec-{invocation_id}.json")
        request_bytes = json.dumps(
            {"argv": list(cmd), "env": dict(env or {})},
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        cmd_str = self._shell_join(
            [_E2B_EXEC_PYTHON, helper_path, request_path]
        )
        loop_start = asyncio.get_event_loop().time()
        command_handle = None
        primary_failed = False
        try:
            try:
                await sandbox.files.write(helper_path, _E2B_EXEC_HELPER)
                await sandbox.files.write(request_path, request_bytes)
            except Exception as exc:
                raise SkillRuntimeError(
                    "E2BSkillRuntime: failed to stage argv-preserving "
                    f"launcher: {exc}"
                ) from exc
            command_handle = await sandbox.commands.run(
                cmd_str,
                background=True,
                envs=dict(_E2B_LAUNCHER_ENV),
                stdin=stdin is not None,
                timeout=timeout,
                cwd=cwd,
            )
            if stdin is not None:
                if stdin:
                    await command_handle.send_stdin(stdin)
                await command_handle.close_stdin()
            result = await command_handle.wait()
        except asyncio.CancelledError:
            primary_failed = True
            if command_handle is not None:
                await self._kill_command_handle(command_handle)
            raise
        except asyncio.TimeoutError:
            if command_handle is not None:
                await self._kill_command_handle(command_handle)
            duration = asyncio.get_event_loop().time() - loop_start
            run_result = RuntimeRunResult(
                stdout=b"",
                stderr=f"[timeout after {timeout}s]".encode(),
                exit_code=124,
                duration_s=duration,
            )
        except Exception as exc:
            # Current E2B raises CommandExitException from handle.wait() for a
            # normal non-zero exit. Preserve it as a command result.
            if getattr(exc, "exit_code", None) is not None:
                duration = asyncio.get_event_loop().time() - loop_start
                run_result = self._coerce_result(
                    exc,
                    duration_s=duration,
                    max_output_bytes=handle.backend.get("max_output_bytes"),
                )
            else:
                primary_failed = True
                if command_handle is not None:
                    await self._kill_command_handle(command_handle)
                raise SkillRuntimeError(
                    f"E2BSkillRuntime: sandbox command failed: {exc}"
                ) from exc
        else:
            duration = asyncio.get_event_loop().time() - loop_start
            run_result = self._coerce_result(
                result,
                duration_s=duration,
                max_output_bytes=handle.backend.get("max_output_bytes"),
            )
        finally:
            try:
                await self._remove_invocation_files(
                    sandbox,
                    (request_path, helper_path),
                )
            except asyncio.CancelledError:
                # Preserve cancellation identity. Executor cleanup will kill
                # the complete sandbox even if per-file cleanup was interrupted.
                raise
            except SkillRuntimeError:
                if primary_failed:
                    _log.warning(
                        "E2BSkillRuntime: launcher cleanup also failed",
                        exc_info=True,
                    )
                else:
                    raise
        return run_result

    async def read_file(
        self, handle: SkillRuntimeHandle, path: str,
    ) -> bytes:
        # workspace paths are relative to the host workspace; we mirror
        # them via the SDK's files API at <ws_in_sandbox>/<rel_path>.
        self._reject_unsafe_relative_path(path)
        sandbox = self._require_sandbox(handle)
        sandbox_path = self._sandbox_path(handle, path)
        try:
            content = await sandbox.files.read(sandbox_path)
        except Exception as exc:
            raise SkillRuntimeError(
                f"E2BSkillRuntime: read_file({path!r}) failed: {exc}"
            ) from exc
        return content if isinstance(content, bytes) else str(content).encode()

    async def read_file_bounded(
        self,
        handle: SkillRuntimeHandle,
        path: str,
        max_bytes: int,
        timeout: Optional[float] = None,
    ) -> bytes:
        """Best-effort bounded read through the remote E2B filesystem API.

        Newer SDKs expose ``get_info``; use it to reject non-regular and
        already-oversized files before transfer, then consume the streaming
        reader while retaining at most ``max_bytes + 1`` bytes. Older injected
        SDK shims without streaming fall back to a post-hoc byte check, which
        is why this runtime advertises the control as advisory.
        """

        if (
            isinstance(max_bytes, bool)
            or not isinstance(max_bytes, int)
            or max_bytes < 0
        ):
            raise SkillRuntimeError(
                f"max_bytes must be a non-negative integer, got {max_bytes!r}"
            )
        self._reject_unsafe_relative_path(path)
        sandbox = self._require_sandbox(handle)
        sandbox_path = self._sandbox_path(handle, path)

        async def read_checked() -> bytes:
            get_info = getattr(sandbox.files, "get_info", None)
            if get_info is not None and inspect.iscoroutinefunction(get_info):
                try:
                    info = await get_info(sandbox_path, request_timeout=timeout)
                except TypeError:
                    # Compatibility with injected/older SDK shims that do not
                    # yet accept the request_timeout keyword.
                    info = await get_info(sandbox_path)
                file_type = getattr(info, "type", None)
                type_value = getattr(file_type, "value", file_type)
                if type_value != "file":
                    raise SkillRuntimeError(
                        f"artifact output {path!r} is not a regular file"
                    )
                size = getattr(info, "size", None)
                if isinstance(size, int) and size > max_bytes:
                    raise SkillRuntimeError(
                        f"artifact output {path!r} exceeds its "
                        f"{max_bytes}-byte limit"
                    )

            try:
                content = await sandbox.files.read(
                    sandbox_path,
                    format="stream",
                    request_timeout=timeout,
                    stream_idle_timeout=timeout,
                )
            except TypeError:
                # Compatibility path only: e2b>=2.38 supports streaming, but
                # custom sandbox_factory shims may expose the older API.
                content = await sandbox.files.read(sandbox_path)

            if hasattr(content, "__aiter__"):
                kept = bytearray()
                try:
                    async for chunk in content:
                        if isinstance(chunk, str):
                            chunk_bytes = chunk.encode()
                        elif isinstance(chunk, bytes):
                            chunk_bytes = chunk
                        else:
                            chunk_bytes = bytes(chunk)
                        remaining_with_sentinel = max_bytes + 1 - len(kept)
                        if remaining_with_sentinel > 0:
                            kept.extend(chunk_bytes[:remaining_with_sentinel])
                        if len(kept) > max_bytes:
                            raise SkillRuntimeError(
                                f"artifact output {path!r} exceeds its "
                                f"{max_bytes}-byte limit"
                            )
                    return bytes(kept)
                finally:
                    close = getattr(content, "aclose", None)
                    if callable(close):
                        close_result = close()
                        if inspect.isawaitable(close_result):
                            await close_result

            if isinstance(content, str):
                data = content.encode()
            elif isinstance(content, (bytes, bytearray, memoryview)):
                data = bytes(content)
            else:
                data = str(content).encode()
            if len(data) > max_bytes:
                raise SkillRuntimeError(
                    f"artifact output {path!r} exceeds its {max_bytes}-byte limit"
                )
            return data

        try:
            if timeout is None:
                return await read_checked()
            return await asyncio.wait_for(read_checked(), timeout=timeout)
        except SkillRuntimeError:
            raise
        except asyncio.TimeoutError as exc:
            raise SkillRuntimeError(
                f"artifact output read timed out after {timeout}s"
            ) from exc
        except Exception as exc:
            raise SkillRuntimeError(
                f"E2BSkillRuntime: read_file_bounded({path!r}) failed: {exc}"
            ) from exc

    async def write_file(
        self, handle: SkillRuntimeHandle, path: str, data: bytes,
    ) -> None:
        self._reject_unsafe_relative_path(path)
        sandbox = self._require_sandbox(handle)
        sandbox_path = self._sandbox_path(handle, path)
        try:
            await sandbox.files.make_dir(str(PurePosixPath(sandbox_path).parent))
            await sandbox.files.write(sandbox_path, data)
        except Exception as exc:
            raise SkillRuntimeError(
                f"E2BSkillRuntime: write_file({path!r}) failed: {exc}"
            ) from exc

    async def cleanup(self, handle: SkillRuntimeHandle) -> None:
        sandbox = handle.backend.get("sandbox")
        cleanup_error: Exception | None = None
        if sandbox is not None and not handle.backend.get("persisted"):
            try:
                await sandbox.kill()
            except asyncio.CancelledError:
                # Keep the reference so a retry can still terminate the
                # remote sandbox.
                raise
            except Exception as exc:  # noqa: BLE001
                cleanup_error = exc
            else:
                handle.backend["sandbox"] = None
        # Local workspace dir cleanup mirrors the other backends.
        if not handle.backend.get("persisted"):
            await _remove_workspace_tree(handle.workspace_root)
        if cleanup_error is not None:
            raise SkillRuntimeError(
                f"E2BSkillRuntime: sandbox.kill() failed during cleanup: {cleanup_error}"
            ) from cleanup_error

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    async def _finish_shielded_task(
        task: asyncio.Future[Any],
    ) -> tuple[Any, asyncio.CancelledError | None]:
        """Finish one lifecycle task despite repeated parent cancellation."""

        cancellation: asyncio.CancelledError | None = None
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError as exc:
                cancellation = cancellation or exc
                current = asyncio.current_task()
                if current is not None:
                    current.uncancel()
        return task.result(), cancellation

    @staticmethod
    async def _default_sandbox_factory(**kwargs: Any) -> Any:
        """Lazy import of the real e2b SDK.

        Raises :class:`SkillRuntimeError` with the install hint if the
        SDK isn't available.
        """
        try:
            from e2b import AsyncSandbox  # type: ignore[import-not-found]
        except ImportError as exc:
            raise SkillRuntimeError(
                "E2BSkillRuntime requires the e2b SDK. Install with "
                "`pip install mmar-carl[e2b]` (or `pip install e2b`)."
            ) from exc

        # The exact e2b SDK API has shifted across versions; we use
        # `AsyncSandbox.create` which is the v1+ shape.
        return await AsyncSandbox.create(**kwargs)

    @staticmethod
    async def _kill_command_handle(command_handle: Any) -> None:
        try:
            await command_handle.kill()
        except Exception:  # noqa: BLE001
            _log.warning(
                "E2BSkillRuntime: command kill failed",
                exc_info=True,
            )

    @staticmethod
    async def _remove_invocation_files(
        sandbox: Any,
        paths: tuple[str, ...],
    ) -> None:
        """Remove staged argv/env material before a reusable sandbox continues."""

        remove = getattr(sandbox.files, "remove", None)
        if not callable(remove):
            raise SkillRuntimeError(
                "E2BSkillRuntime: SDK does not provide files.remove() for "
                "argv launcher cleanup"
            )
        failures: list[str] = []
        for path in paths:
            try:
                await remove(path)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 — remote SDK boundary
                failures.append(f"{path}: {exc}")
        if failures:
            raise SkillRuntimeError(
                "E2BSkillRuntime: failed to remove staged argv launcher files: "
                + "; ".join(failures)
            )

    @staticmethod
    def _require_sandbox(handle: SkillRuntimeHandle) -> Any:
        sandbox = handle.backend.get("sandbox")
        if sandbox is None:
            raise SkillRuntimeError(
                "E2BSkillRuntime: handle has no live sandbox."
            )
        return sandbox

    @staticmethod
    def _sandbox_path(handle: SkillRuntimeHandle, rel: str) -> str:
        """Translate a workspace-relative path to an absolute path
        inside the sandbox."""
        root = handle.backend.get(
            "workspace_root_in_sandbox", _DEFAULT_WS_IN_SANDBOX,
        )
        # Use posix style joining since the sandbox is Linux.
        if root.endswith("/"):
            return f"{root}{rel}"
        return f"{root}/{rel}"

    @staticmethod
    def _reject_unsafe_relative_path(path: str) -> None:
        """Reject absolute paths and ``..`` segments — same contract
        as the other reference backends."""
        if os.path.isabs(path):
            raise SkillRuntimeError(
                f"workspace paths must be relative: got {path!r}"
            )
        # Reject any segment that's exactly `..` (escape attempt).
        parts = path.replace("\\", "/").split("/")
        if any(p == ".." for p in parts):
            raise SkillRuntimeError(
                f"path {path!r} escapes workspace"
            )

    @staticmethod
    def _shell_join(cmd: list[str]) -> str:
        """Join the fixed internal launcher argv into a command string.

        Chain-controlled argv is stored in a JSON request and never reaches
        this shell-parsed boundary.
        """
        import shlex
        return shlex.join(cmd)

    @staticmethod
    def _coerce_result(
        result: Any,
        *,
        duration_s: float,
        max_output_bytes: int | None = None,
    ) -> RuntimeRunResult:
        """Normalise the SDK's result object into ``RuntimeRunResult``."""
        stdout = getattr(result, "stdout", b"") or b""
        stderr = getattr(result, "stderr", b"") or b""
        exit_code = getattr(result, "exit_code", None)
        if exit_code is None:
            exit_code = getattr(result, "exitCode", 0)
        if isinstance(stdout, str):
            stdout = stdout.encode("utf-8", errors="replace")
        if isinstance(stderr, str):
            stderr = stderr.encode("utf-8", errors="replace")
        stdout_truncated = max_output_bytes is not None and len(stdout) > max_output_bytes
        stderr_truncated = max_output_bytes is not None and len(stderr) > max_output_bytes
        if max_output_bytes is not None:
            stdout = stdout[:max_output_bytes]
            stderr = stderr[:max_output_bytes]
        return RuntimeRunResult(
            stdout=stdout,
            stderr=stderr,
            exit_code=int(exit_code) if exit_code is not None else 0,
            duration_s=duration_s,
            stdout_truncated=stdout_truncated,
            stderr_truncated=stderr_truncated,
        )


# Self-register on import.
register_skill_runtime(E2BSkillRuntime.name, E2BSkillRuntime)


__all__ = ["E2BSkillRuntime"]
