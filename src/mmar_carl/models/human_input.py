"""Typed process-local contracts for :class:`HumanInputStep`."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_validator,
    model_validator,
)

HumanInputStatus = Literal[
    "answered",
    "timed_out",
    "unavailable",
    "cancelled",
    "invalid_response",
    "failed",
]


class HumanInputRequest(BaseModel):
    """One text request emitted by a running HumanInputStep."""

    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(min_length=1, max_length=128)
    step_number: int
    step_title: str = Field(max_length=512)
    kind: Literal["text"] = "text"
    prompt: str = Field(min_length=1, max_length=8192)
    min_length: int = Field(default=0, ge=0)
    max_length: int = Field(default=4096, ge=1, le=65536)
    sensitive: bool = False
    deadline: datetime | None = None

    @field_validator("request_id", "prompt")
    @classmethod
    def _required_text_must_be_non_empty(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("value cannot be empty")
        return normalized

    @field_validator("deadline")
    @classmethod
    def _deadline_must_be_timezone_aware(
        cls, value: datetime | None,
    ) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("deadline must be timezone-aware")
        return value

    @model_validator(mode="after")
    def _length_bounds_must_be_ordered(self) -> HumanInputRequest:
        if self.min_length > self.max_length:
            raise ValueError("min_length cannot exceed max_length")
        return self


class HumanInputResponse(BaseModel):
    """Host-provided response to one HumanInputRequest."""

    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(min_length=1, max_length=128)
    value: str = Field(max_length=65536)
    actor_id: str | None = Field(default=None, max_length=256)
    responded_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    provenance: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("request_id")
    @classmethod
    def _request_id_must_be_non_empty(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("request_id cannot be empty")
        return normalized

    @field_validator("actor_id")
    @classmethod
    def _actor_id_must_be_non_empty(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("actor_id cannot be empty")
        return normalized

    @field_validator("responded_at")
    @classmethod
    def _responded_at_must_be_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("responded_at must be timezone-aware")
        return value

    @field_validator("provenance")
    @classmethod
    def _provenance_must_be_bounded(
        cls, value: dict[str, JsonValue],
    ) -> dict[str, JsonValue]:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
        if len(encoded) > 8192:
            raise ValueError("provenance cannot exceed 8192 UTF-8 bytes")
        return value


class HumanInputOutcome(BaseModel):
    """Canonical structured outcome emitted by HumanInputStep."""

    model_config = ConfigDict(extra="forbid")

    status: HumanInputStatus
    request_id: str = Field(min_length=1, max_length=128)
    elapsed_seconds: float = Field(ge=0.0)
    value: str | None = Field(default=None, max_length=65536)
    actor_id: str | None = Field(default=None, max_length=256)
    responded_at: datetime | None = None
    provenance: dict[str, JsonValue] = Field(default_factory=dict)
    redacted: bool = False
    error_message: str | None = Field(default=None, max_length=512)

    @field_validator("request_id")
    @classmethod
    def _request_id_must_be_non_empty(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("request_id cannot be empty")
        return normalized

    @field_validator("actor_id")
    @classmethod
    def _actor_id_must_be_non_empty(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("actor_id cannot be empty")
        return normalized

    @field_validator("responded_at")
    @classmethod
    def _responded_at_must_be_timezone_aware(
        cls, value: datetime | None,
    ) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("responded_at must be timezone-aware")
        return value

    @model_validator(mode="after")
    def _status_fields_must_be_consistent(self) -> HumanInputOutcome:
        if self.status == "answered":
            if self.error_message is not None:
                raise ValueError("answered outcome cannot contain error_message")
            if self.redacted and self.value is not None:
                raise ValueError("redacted answered outcome cannot contain value")
            if not self.redacted and self.value is None:
                raise ValueError("non-redacted answered outcome requires value")
        elif self.value is not None or self.redacted:
            raise ValueError("non-answered outcome cannot contain a value")
        return self


__all__ = [
    "HumanInputOutcome",
    "HumanInputRequest",
    "HumanInputResponse",
    "HumanInputStatus",
]
