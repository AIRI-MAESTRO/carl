"""Typed contracts and deterministic validation for :class:`CodeStep`."""

from __future__ import annotations

import ast
import json
import math
import re
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_validator,
    model_validator,
)

from .command_policy import normalize_memory_limit

CodeExecutionStatus = Literal[
    "completed",
    "invalid_source",
    "invalid_input",
    "denied",
    "runtime_unavailable",
    "timed_out",
    "invalid_output",
    "failed",
    "cancelled",
]

_IDENTIFIER_RE = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{0,127}\Z")
_DOCKER_DIGEST_RE = re.compile(r"(?!-)[^\s\x00]+@sha256:[0-9a-f]{64}\Z")
_JSON_TYPES = frozenset({"object", "array", "string", "number", "integer", "boolean", "null"})
_SCHEMA_KEYS = frozenset(
    {"type", "properties", "required", "items", "enum", "additionalProperties"}
)


class CodeSchemaError(ValueError):
    """Raised when a CodeStep schema or JSON value violates the v1 subset."""


class CodeSourceError(ValueError):
    """Raised when generated source violates the exact v1 entry-point contract."""


class CodeRuntimeProfile(BaseModel):
    """One immutable host-owned runtime profile available to CodeStep."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    runtime: str = Field(min_length=1, max_length=128)
    revision: str = Field(min_length=1, max_length=256)
    interpreter: tuple[str, ...] = Field(default=("python", "-I", "-B"), min_length=1)
    prepare_config: dict[str, JsonValue] = Field(default_factory=dict)
    cpu_limit: float = Field(default=1.0, gt=0)
    mem_limit: str = Field(default="256m")
    pids_limit: int = Field(default=32, gt=0)
    max_timeout_seconds: float = Field(default=30.0, gt=0)
    max_cleanup_seconds: float = Field(default=30.0, gt=0)
    max_source_bytes: int = Field(default=20_000, gt=0)
    max_input_bytes: int = Field(default=1_000_000, gt=0)
    max_output_bytes: int = Field(default=1_000_000, gt=0)

    @field_validator("runtime")
    @classmethod
    def _runtime_must_be_an_identifier(cls, value: str) -> str:
        value = value.strip()
        if _IDENTIFIER_RE.fullmatch(value) is None:
            raise ValueError("runtime must match [A-Za-z][A-Za-z0-9_.-]{0,127}")
        return value

    @field_validator("revision")
    @classmethod
    def _revision_must_be_non_empty(cls, value: str) -> str:
        value = value.strip()
        if not value or "\x00" in value:
            raise ValueError("revision must be non-empty and contain no NUL")
        return value

    @field_validator("interpreter")
    @classmethod
    def _interpreter_must_be_fixed_argv(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) > 16:
            raise ValueError("interpreter cannot contain more than 16 argv tokens")
        if any(not token or "\x00" in token for token in value):
            raise ValueError("interpreter tokens must be non-empty and contain no NUL")
        return value

    @field_validator("mem_limit")
    @classmethod
    def _normalize_memory_limit(cls, value: str) -> str:
        return normalize_memory_limit(value)

    @field_validator("prepare_config")
    @classmethod
    def _prepare_config_must_be_bounded_and_offline(
        cls, value: dict[str, JsonValue],
    ) -> dict[str, JsonValue]:
        forbidden = {"_network_binding", "extra_args", "workspace_output_mode"} & value.keys()
        if forbidden:
            raise ValueError(f"prepare_config contains CARL-reserved keys: {sorted(forbidden)!r}")
        if value.get("network", "none") != "none":
            raise ValueError("CodeStep runtime profiles must use network='none'")
        encoded = canonical_json_bytes(value)
        if len(encoded) > 16_384:
            raise ValueError("prepare_config cannot exceed 16384 JSON bytes")
        return value

    @model_validator(mode="after")
    def _docker_image_must_be_pinned(self) -> CodeRuntimeProfile:
        if self.runtime == "docker":
            image = self.prepare_config.get("image")
            if not isinstance(image, str) or _DOCKER_DIGEST_RE.fullmatch(image) is None:
                raise ValueError(
                    "docker CodeRuntimeProfile requires an image pinned as "
                    "name@sha256:<64 lowercase hex characters>"
                )
        return self


class CodeExecutionPolicy(BaseModel):
    """Runtime-only host authority mapping profile ids to strict sandboxes."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    profiles: dict[str, CodeRuntimeProfile] = Field(min_length=1)

    @field_validator("profiles")
    @classmethod
    def _profile_ids_must_be_identifiers(
        cls, value: dict[str, CodeRuntimeProfile],
    ) -> dict[str, CodeRuntimeProfile]:
        invalid = [name for name in value if _IDENTIFIER_RE.fullmatch(name) is None]
        if invalid:
            raise ValueError(f"profile ids are invalid: {invalid!r}")
        return value

    def resolve(self, profile_id: str) -> CodeRuntimeProfile | None:
        """Return the exact host profile requested by a serialized step."""

        return self.profiles.get(profile_id)


class CodeExecutionOutcome(BaseModel):
    """Canonical typed outcome emitted by CodeStep."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    status: CodeExecutionStatus
    profile_id: str = Field(min_length=1, max_length=128)
    runtime: str | None = Field(default=None, max_length=128)
    runtime_revision: str | None = Field(default=None, max_length=256)
    source_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    source_bytes: int | None = Field(default=None, ge=0)
    input_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    output_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    output: JsonValue | None = None
    python_version: str | None = Field(default=None, max_length=128)
    stdout: str = ""
    stderr: str = ""
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    elapsed_seconds: float = Field(ge=0.0)
    effective_limits: dict[str, JsonValue] = Field(default_factory=dict)
    enforcement_report: dict[str, JsonValue] = Field(default_factory=dict)
    error_message: str | None = Field(default=None, max_length=2048)

    @model_validator(mode="after")
    def _status_fields_must_be_consistent(self) -> CodeExecutionOutcome:
        if self.status == "completed":
            if self.error_message is not None or self.output_sha256 is None:
                raise ValueError("completed outcome requires output hash and no error")
        elif self.output is not None or self.output_sha256 is not None:
            raise ValueError("non-completed outcome cannot contain output")
        return self


def canonical_json_bytes(value: Any) -> bytes:
    """Encode one finite JSON value canonically or raise CodeSchemaError."""

    _assert_json_value(value, path="$", depth=0, ancestors=frozenset())
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise CodeSchemaError(f"value is not finite JSON: {exc}") from exc


def _assert_json_value(
    value: Any,
    *,
    path: str,
    depth: int,
    ancestors: frozenset[int],
) -> None:
    if depth > 64:
        raise CodeSchemaError(f"{path}: JSON nesting exceeds 64 levels")
    if value is None or isinstance(value, (bool, str, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CodeSchemaError(f"{path}: JSON number must be finite")
        return
    if isinstance(value, (list, dict)):
        identity = id(value)
        if identity in ancestors:
            raise CodeSchemaError(f"{path}: cyclic values are not JSON")
        child_ancestors = ancestors | {identity}
        if isinstance(value, list):
            for index, item in enumerate(value):
                _assert_json_value(
                    item,
                    path=f"{path}[{index}]",
                    depth=depth + 1,
                    ancestors=child_ancestors,
                )
            return
        for key, item in value.items():
            if not isinstance(key, str):
                raise CodeSchemaError(f"{path}: JSON object keys must be strings")
            _assert_json_value(
                item,
                path=f"{path}.{key}",
                depth=depth + 1,
                ancestors=child_ancestors,
            )
        return
    raise CodeSchemaError(f"{path}: {type(value).__name__} is not a JSON value")


def validate_code_schema(schema: dict[str, Any], *, require_object: bool = False) -> None:
    """Validate the strict JSON Schema subset supported by CodeStep v1."""

    _validate_schema_node(schema, path="$schema")
    schema_types = schema.get("type")
    normalized = [schema_types] if isinstance(schema_types, str) else schema_types
    if require_object and normalized != ["object"]:
        raise CodeSchemaError("input_schema must have type='object'")


def _validate_schema_node(schema: Any, *, path: str) -> None:
    if not isinstance(schema, dict) or not schema:
        raise CodeSchemaError(f"{path}: schema must be a non-empty object")
    unknown = set(schema) - _SCHEMA_KEYS
    if unknown:
        raise CodeSchemaError(f"{path}: unsupported schema keys {sorted(unknown)!r}")
    raw_types = schema.get("type")
    if isinstance(raw_types, str):
        types = [raw_types]
    elif isinstance(raw_types, list) and raw_types:
        types = raw_types
    else:
        raise CodeSchemaError(f"{path}.type must be a JSON type or non-empty list")
    if any(not isinstance(item, str) or item not in _JSON_TYPES for item in types):
        raise CodeSchemaError(f"{path}.type contains an unsupported JSON type")
    if len(types) != len(set(types)):
        raise CodeSchemaError(f"{path}.type entries must be unique")

    if "enum" in schema:
        enum = schema["enum"]
        if not isinstance(enum, list) or not enum:
            raise CodeSchemaError(f"{path}.enum must be a non-empty list")
        canonical_json_bytes(enum)

    object_keys = {"properties", "required", "additionalProperties"} & schema.keys()
    if object_keys and "object" not in types:
        raise CodeSchemaError(f"{path}: object keywords require type='object'")
    properties = schema.get("properties", {})
    if not isinstance(properties, dict) or any(not isinstance(key, str) for key in properties):
        raise CodeSchemaError(f"{path}.properties must be an object")
    for key, child in properties.items():
        _validate_schema_node(child, path=f"{path}.properties.{key}")
    required = schema.get("required", [])
    if (
        not isinstance(required, list)
        or any(not isinstance(key, str) for key in required)
        or len(required) != len(set(required))
    ):
        raise CodeSchemaError(f"{path}.required must contain unique strings")
    missing_properties = set(required) - set(properties)
    if missing_properties:
        raise CodeSchemaError(
            f"{path}.required references undeclared properties {sorted(missing_properties)!r}"
        )
    if not isinstance(schema.get("additionalProperties", True), bool):
        raise CodeSchemaError(f"{path}.additionalProperties must be boolean")

    if "items" in schema:
        if "array" not in types:
            raise CodeSchemaError(f"{path}.items requires type='array'")
        _validate_schema_node(schema["items"], path=f"{path}.items")


def validate_code_value(
    value: Any,
    schema: dict[str, Any],
    *,
    max_bytes: int | None = None,
) -> bytes:
    """Validate one bounded finite JSON value and return canonical bytes."""

    encoded = canonical_json_bytes(value)
    if max_bytes is not None and len(encoded) > max_bytes:
        raise CodeSchemaError(f"value is {len(encoded)} bytes; limit is {max_bytes}")
    _validate_value_node(value, schema, path="$")
    return encoded


def _validate_value_node(value: Any, schema: dict[str, Any], *, path: str) -> None:
    raw_types = schema["type"]
    types = [raw_types] if isinstance(raw_types, str) else raw_types
    if not any(_matches_json_type(value, item) for item in types):
        raise CodeSchemaError(
            f"{path}: expected {' | '.join(types)}, got {type(value).__name__}"
        )
    if "enum" in schema:
        encoded = canonical_json_bytes(value)
        if all(encoded != canonical_json_bytes(candidate) for candidate in schema["enum"]):
            raise CodeSchemaError(f"{path}: value is not in enum")
    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                raise CodeSchemaError(f"{path}: missing required key {key!r}")
        properties = schema.get("properties", {})
        if schema.get("additionalProperties", True) is False:
            extra = set(value) - set(properties)
            if extra:
                raise CodeSchemaError(f"{path}: additional properties {sorted(extra)!r}")
        for key, child in properties.items():
            if key in value:
                _validate_value_node(value[key], child, path=f"{path}.{key}")
    elif isinstance(value, list) and "items" in schema:
        for index, item in enumerate(value):
            _validate_value_node(item, schema["items"], path=f"{path}[{index}]")


def _matches_json_type(value: Any, expected: str) -> bool:
    if expected == "null":
        return value is None
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "string":
        return isinstance(value, str)
    if expected == "array":
        return isinstance(value, list)
    if expected == "object":
        return isinstance(value, dict)
    return False


def validate_code_source(source: str) -> ast.Module:
    """Parse exact source and enforce one synchronous ``run(inputs)`` entry point."""

    if not source.strip():
        raise CodeSourceError("source cannot be blank")
    try:
        tree = ast.parse(source, filename="<carl-generated-code>", mode="exec")
    except (SyntaxError, ValueError) as exc:
        message = getattr(exc, "msg", str(exc))
        line = getattr(exc, "lineno", None)
        suffix = f" at line {line}" if line is not None else ""
        raise CodeSourceError(f"source does not parse: {message}{suffix}") from exc
    async_runs = [node for node in tree.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "run"]
    runs = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run"]
    if async_runs:
        raise CodeSourceError("run(inputs) must be synchronous in CodeStep v1")
    if len(runs) != 1:
        raise CodeSourceError("source must define exactly one top-level function named 'run'")
    run = runs[0]
    args = run.args
    positional = [*args.posonlyargs, *args.args]
    if (
        len(positional) != 1
        or positional[0].arg != "inputs"
        or args.vararg is not None
        or args.kwarg is not None
        or args.kwonlyargs
        or args.defaults
        or args.kw_defaults
    ):
        raise CodeSourceError("entry point must have the exact signature def run(inputs)")
    if run.decorator_list:
        raise CodeSourceError("run(inputs) must not use decorators")
    return tree


CODE_RESULT_PREFIX = "__CARL_CODE_RESULT_V1__="

CODE_RUNNER_SOURCE = r'''import importlib.util
import inspect
import json
import math
import platform
import sys

PREFIX = "__CARL_CODE_RESULT_V1__="


def emit(payload):
    raw = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    sys.__stdout__.write(PREFIX + raw + "\n")
    sys.__stdout__.flush()


def validate_json(value, depth=0, ancestors=None):
    if depth > 64:
        raise ValueError("JSON nesting exceeds 64 levels")
    if value is None or isinstance(value, (bool, str, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON number must be finite")
        return
    if isinstance(value, (list, dict)):
        ancestors = set() if ancestors is None else ancestors
        identity = id(value)
        if identity in ancestors:
            raise ValueError("cyclic values are not JSON")
        child_ancestors = ancestors | {identity}
        if isinstance(value, list):
            for item in value:
                validate_json(item, depth + 1, child_ancestors)
            return
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("JSON object keys must be strings")
            validate_json(item, depth + 1, child_ancestors)
        return
    raise TypeError(type(value).__name__ + " is not a JSON value")


def main():
    source_path, input_path, max_output_text = sys.argv[1:]
    max_output_bytes = int(max_output_text)
    try:
        spec = importlib.util.spec_from_file_location("carl_generated_code", source_path)
        if spec is None or spec.loader is None:
            emit({"status": "invalid_source", "error": "source module could not be loaded"})
            return 2
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except SyntaxError as exc:
        emit({"status": "invalid_source", "error": str(exc)[:1024]})
        return 2
    except BaseException as exc:
        emit({
            "status": "failed",
            "error": (type(exc).__name__ + ": " + str(exc))[:1024],
        })
        return 1

    function = getattr(module, "run", None)
    if not callable(function) or inspect.iscoroutinefunction(function):
        emit({"status": "invalid_source", "error": "source did not define synchronous run(inputs)"})
        return 2

    try:
        with open(input_path, "r", encoding="utf-8") as stream:
            inputs = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        emit({"status": "invalid_input", "error": str(exc)[:1024]})
        return 2

    try:
        output = function(inputs)
    except BaseException as exc:
        emit({
            "status": "failed",
            "error": (type(exc).__name__ + ": " + str(exc))[:1024],
        })
        return 1

    try:
        validate_json(output)
        output_json = json.dumps(
            output,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if len(output_json.encode("utf-8")) > max_output_bytes:
            emit({"status": "invalid_output", "error": "output exceeds max_output_bytes"})
            return 3
    except (TypeError, ValueError, UnicodeError) as exc:
        emit({"status": "invalid_output", "error": str(exc)[:1024]})
        return 3

    emit({
        "status": "completed",
        "output": output,
        "python_version": platform.python_version(),
    })
    return 0


raise SystemExit(main())
'''


__all__ = [
    "CODE_RESULT_PREFIX",
    "CODE_RUNNER_SOURCE",
    "CodeExecutionOutcome",
    "CodeExecutionPolicy",
    "CodeExecutionStatus",
    "CodeRuntimeProfile",
    "CodeSchemaError",
    "CodeSourceError",
    "canonical_json_bytes",
    "validate_code_schema",
    "validate_code_source",
    "validate_code_value",
]
