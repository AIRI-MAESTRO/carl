"""Serializable contracts for exposing an embedded CARL chain as a tool."""

from __future__ import annotations

import hashlib
import json
import math
import re
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_TOOL_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]{0,63}$")
_OUTPUT_REFERENCE_PREFIXES = (
    "$history",
    "$memory.",
    "$metadata.",
    "$steps.",
    "$outer_context",
)


class ChainToolStatus(str, Enum):
    """Terminal status of one nested-chain invocation."""

    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    INVALID_INPUT = "invalid_input"
    INVALID_OUTPUT = "invalid_output"
    UNAVAILABLE = "unavailable"
    RECURSION_LIMIT = "recursion_limit"


class ChainToolDefinition(BaseModel):
    """Versioned, JSON-serializable definition of an embedded chain tool.

    ``chain_snapshot`` is a complete :meth:`ReasoningChain.to_dict` payload.
    The nested run receives its validated arguments as JSON in
    ``$outer_context`` and can see only the host tools named by
    ``allowed_tools``. Parent history, memory, messages, metadata, callbacks,
    event bus and command/network authority are not inherited.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    contract_version: Literal[1] = 1
    name: str = Field(..., min_length=1, max_length=64)
    description: str = Field(..., min_length=1, max_length=2000)
    chain_snapshot: dict[str, Any]
    snapshot_sha256: str = Field(default="", pattern=r"^[0-9a-f]{64}$|^$")
    input_schema: dict[str, Any] = Field(
        ...,
        description="JSON Schema subset for the keyword arguments accepted by the tool.",
    )
    output_schema: dict[str, Any] = Field(
        ...,
        description="JSON Schema subset for the value selected from the child context.",
    )
    output_reference: str = Field(
        ...,
        min_length=1,
        description="Child-context reference resolved after successful execution.",
    )
    allowed_tools: list[str] = Field(
        default_factory=list,
        description="Explicit host-tool capability allowlist copied into the child context.",
    )
    tags: list[str] = Field(
        default_factory=lambda: ["chain"],
        description="Tags attached to the registered chain tool.",
    )
    max_depth: int = Field(default=4, ge=1, le=32)
    timeout_seconds: float = Field(default=120.0, gt=0.0, le=86_400.0)
    max_input_bytes: int = Field(default=1_000_000, ge=1, le=16_000_000)
    max_output_bytes: int = Field(default=1_000_000, ge=1, le=16_000_000)

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        value = value.strip()
        if not _TOOL_NAME_RE.fullmatch(value):
            raise ValueError(
                "name must start with a letter or underscore and contain only letters, digits, underscores or hyphens"
            )
        if value == "finish":
            raise ValueError("'finish' is reserved by AgentStep")
        return value

    @field_validator("description")
    @classmethod
    def _strip_description(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("description must not be blank")
        return value

    @field_validator("chain_snapshot", "input_schema", "output_schema")
    @classmethod
    def _copy_json_object(cls, value: dict[str, Any]) -> dict[str, Any]:
        try:
            encoded = json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True)
        except (TypeError, ValueError) as exc:
            raise ValueError("value must contain only finite JSON-compatible data") from exc
        decoded = json.loads(encoded)
        if not isinstance(decoded, dict):
            raise ValueError("value must be a JSON object")  # noqa: TRY004
        return decoded

    @field_validator("input_schema")
    @classmethod
    def _input_must_be_object_schema(cls, value: dict[str, Any]) -> dict[str, Any]:
        if value.get("type") != "object":
            raise ValueError("input_schema must declare type='object'")
        return value

    @field_validator("input_schema", "output_schema")
    @classmethod
    def _schemas_use_supported_subset(cls, value: dict[str, Any]) -> dict[str, Any]:
        _validate_schema_definition(value)
        return value

    @field_validator("chain_snapshot")
    @classmethod
    def _validate_snapshot_shape(cls, value: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(value.get("format_version"), int):
            raise ValueError(  # noqa: TRY004
                "chain_snapshot must contain an integer format_version"
            )
        if not isinstance(value.get("steps"), list):
            raise ValueError("chain_snapshot must contain a steps list")  # noqa: TRY004
        return value

    @field_validator("output_reference")
    @classmethod
    def _validate_output_reference(cls, value: str) -> str:
        value = value.strip()
        if not value.startswith(_OUTPUT_REFERENCE_PREFIXES):
            raise ValueError("output_reference must read from child history, memory, metadata, steps or outer_context")
        return value

    @field_validator("allowed_tools", "tags")
    @classmethod
    def _normalize_names(cls, value: list[str]) -> list[str]:
        normalized = [item.strip() for item in value]
        if any(not item for item in normalized):
            raise ValueError("names must not be blank")
        if len(normalized) != len(set(normalized)):
            raise ValueError("names must be unique")
        return normalized

    @field_validator("timeout_seconds")
    @classmethod
    def _finite_timeout(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("timeout_seconds must be finite")
        return value

    @model_validator(mode="after")
    def _reject_direct_self_capability(self) -> ChainToolDefinition:
        if self.name in self.allowed_tools:
            raise ValueError("a chain tool cannot grant itself as a host capability")
        calculated = self.calculate_snapshot_sha256()
        if self.snapshot_sha256 and self.snapshot_sha256 != calculated:
            raise ValueError("snapshot_sha256 does not match chain_snapshot")
        object.__setattr__(self, "snapshot_sha256", calculated)
        return self

    @classmethod
    def from_chain(
        cls,
        *,
        name: str,
        description: str,
        chain: Any,
        input_schema: dict[str, Any],
        output_schema: dict[str, Any],
        output_reference: str,
        allowed_tools: list[str] | None = None,
        tags: list[str] | None = None,
        max_depth: int = 4,
        timeout_seconds: float = 120.0,
        max_input_bytes: int = 1_000_000,
        max_output_bytes: int = 1_000_000,
    ) -> ChainToolDefinition:
        """Freeze ``chain`` through its public JSON snapshot surface."""
        if not hasattr(chain, "to_dict"):
            raise TypeError("chain must provide to_dict()")
        return cls(
            name=name,
            description=description,
            chain_snapshot=chain.to_dict(),
            input_schema=input_schema,
            output_schema=output_schema,
            output_reference=output_reference,
            allowed_tools=allowed_tools or [],
            tags=["chain"] if tags is None else tags,
            max_depth=max_depth,
            timeout_seconds=timeout_seconds,
            max_input_bytes=max_input_bytes,
            max_output_bytes=max_output_bytes,
        )

    def calculate_snapshot_sha256(self) -> str:
        """Calculate the canonical SHA-256 identity of the embedded snapshot."""
        encoded = json.dumps(
            self.chain_snapshot,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def calculate_contract_sha256(self) -> str:
        """Calculate the identity of the complete invocation contract."""
        encoded = json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


class ChainToolOutcome(BaseModel):
    """Typed result envelope returned by every chain-tool invocation."""

    model_config = ConfigDict(extra="forbid")

    contract_version: Literal[1] = 1
    invocation_id: str = Field(..., min_length=1)
    tool_name: str = Field(..., min_length=1)
    snapshot_sha256: str = Field(..., pattern=r"^[0-9a-f]{64}$")
    depth: int = Field(..., ge=1)
    status: ChainToolStatus
    success: bool
    input_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    output_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    output: Any = None
    error_code: str | None = None
    error_message: str | None = None
    duration_seconds: float = Field(..., ge=0.0)
    token_usage: dict[str, int] = Field(default_factory=dict)
    executed_steps: int = Field(default=0, ge=0)

    @field_validator("token_usage")
    @classmethod
    def _validate_usage(cls, value: dict[str, int]) -> dict[str, int]:
        if any(isinstance(item, bool) or item < 0 for item in value.values()):
            raise ValueError("token_usage values must be non-negative integers")
        return value

    @model_validator(mode="after")
    def _validate_terminal_shape(self) -> ChainToolOutcome:
        expected_success = self.status is ChainToolStatus.COMPLETED
        if self.success is not expected_success:
            raise ValueError("success must be true only for completed outcomes")
        if expected_success:
            if self.error_code is not None or self.error_message is not None:
                raise ValueError("completed outcome cannot contain an error")
            if self.output_sha256 is None:
                raise ValueError("completed outcome requires output_sha256")
        elif self.output is not None:
            raise ValueError("non-completed outcome cannot expose output")
        elif self.output_sha256 is not None:
            raise ValueError("non-completed outcome cannot contain output_sha256")
        return self


def validate_chain_tool_value(value: Any, schema: dict[str, Any], *, path: str = "$") -> None:
    """Validate the bounded JSON-Schema subset used by chain-tool boundaries."""
    if not isinstance(schema, dict):
        raise TypeError(f"{path}: schema must be an object")

    expected = schema.get("type")
    if expected is not None:
        expected_types = expected if isinstance(expected, list) else [expected]
        if not any(_matches_type(value, item) for item in expected_types):
            raise ValueError(f"{path}: expected {' | '.join(map(str, expected_types))}, got {type(value).__name__}")

    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{path}: value is not in the declared enum")

    if isinstance(value, dict):
        required = schema.get("required", [])
        if not isinstance(required, list):
            raise TypeError(f"{path}: required must be an array")
        for key in required:
            if key not in value:
                raise ValueError(f"{path}: missing required key '{key}'")
        properties = schema.get("properties", {})
        if not isinstance(properties, dict):
            raise TypeError(f"{path}: properties must be an object")
        if schema.get("additionalProperties") is False:
            unexpected = sorted(set(value) - set(properties))
            if unexpected:
                raise ValueError(f"{path}: unexpected key '{unexpected[0]}'")
        for key, child_schema in properties.items():
            if key in value:
                validate_chain_tool_value(value[key], child_schema, path=f"{path}.{key}")
    elif isinstance(value, list) and isinstance(schema.get("items"), dict):
        for index, item in enumerate(value):
            validate_chain_tool_value(item, schema["items"], path=f"{path}[{index}]")


def _validate_schema_definition(schema: dict[str, Any], *, path: str = "$") -> None:
    allowed_keywords = {
        "type",
        "properties",
        "required",
        "items",
        "enum",
        "additionalProperties",
    }
    unsupported = sorted(set(schema) - allowed_keywords)
    if unsupported:
        raise ValueError(f"{path}: unsupported schema keyword '{unsupported[0]}'")

    supported_types = {"null", "boolean", "integer", "number", "string", "array", "object"}
    declared_type = schema.get("type")
    if declared_type is not None:
        declared_types = declared_type if isinstance(declared_type, list) else [declared_type]
        if not declared_types or any(item not in supported_types for item in declared_types):
            raise ValueError(f"{path}: type must use the supported JSON types")

    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        raise ValueError(f"{path}: properties must be an object")  # noqa: TRY004
    for key, child_schema in properties.items():
        if not isinstance(key, str) or not isinstance(child_schema, dict):
            raise ValueError(  # noqa: TRY004
                f"{path}: properties must map string names to schemas"
            )
        _validate_schema_definition(child_schema, path=f"{path}.properties.{key}")

    required = schema.get("required", [])
    if not isinstance(required, list) or any(not isinstance(key, str) for key in required):
        raise ValueError(f"{path}: required must be an array of strings")
    if len(required) != len(set(required)):
        raise ValueError(f"{path}: required names must be unique")
    undeclared = sorted(set(required) - set(properties))
    if undeclared:
        raise ValueError(f"{path}: required key '{undeclared[0]}' has no property schema")

    if "items" in schema:
        items = schema["items"]
        if not isinstance(items, dict):
            raise ValueError(f"{path}: items must be a schema object")
        _validate_schema_definition(items, path=f"{path}.items")

    if "enum" in schema and not isinstance(schema["enum"], list):
        raise ValueError(f"{path}: enum must be an array")

    if "additionalProperties" in schema and schema["additionalProperties"] is not False:
        raise ValueError(f"{path}: only additionalProperties=false is supported")


def _matches_type(value: Any, expected: Any) -> bool:
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


__all__ = [
    "ChainToolDefinition",
    "ChainToolOutcome",
    "ChainToolStatus",
    "validate_chain_tool_value",
]
