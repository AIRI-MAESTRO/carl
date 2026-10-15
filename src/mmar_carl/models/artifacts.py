"""Portable, bounded file artifacts for runtime-backed steps.

Artifacts deliberately carry bytes, not host paths.  A producing step returns
an :class:`ArtifactRecord` with base64 content; a later step declares that
record as an input and CARL stages it below the runtime's ``in/`` directory.
This keeps chain JSON portable and prevents a serialized chain from asking the
host to read an arbitrary filesystem path.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import re

from pydantic import BaseModel, ConfigDict, Field, field_validator

_ARTIFACT_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


def validate_artifact_name(value: str) -> str:
    """Return an environment-safe logical artifact name."""

    if _ARTIFACT_NAME_RE.fullmatch(value) is None:
        raise ValueError("artifact names must match [A-Za-z_][A-Za-z0-9_]*")
    return value


def validate_utf8_text(value: str) -> str:
    """Reject lone surrogates that cannot cross the JSON/runtime boundary."""

    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError("value must be valid UTF-8") from exc
    return value


def validate_artifact_path(value: str) -> str:
    """Validate a portable path relative to the runtime ``in/`` or ``out/``.

    Paths always use POSIX separators because Docker/E2B are POSIX runtimes.
    Empty, absolute, dot-segment, duplicate-separator, backslash, and NUL paths
    are rejected before any backend sees them.
    """

    validate_utf8_text(value)
    if not value or "\x00" in value or "\\" in value or value.startswith("/"):
        raise ValueError("artifact paths must be non-empty relative POSIX paths")
    parts = value.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise ValueError("artifact paths must not contain empty, '.' or '..' segments")
    return value


class ArtifactInput(BaseModel):
    """One context value staged as bytes below the runtime ``in/`` directory."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(..., description="Logical name used in CARL_ARTIFACT_IN_<NAME>.")
    source: str = Field(..., min_length=1, description="CARL context reference or quoted literal.")
    path: str = Field(..., description="Relative target path below the runtime in/ directory.")
    media_type: str = Field(default="application/octet-stream", min_length=1)
    max_bytes: int | None = Field(
        default=None,
        gt=0,
        description="Optional per-input limit; host policy may impose a smaller limit.",
    )

    _validate_name = field_validator("name")(validate_artifact_name)
    _validate_path = field_validator("path")(validate_artifact_path)

    _validate_media_type = field_validator("media_type")(validate_utf8_text)


class ArtifactOutput(BaseModel):
    """One required file collected from the runtime ``out/`` directory."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(..., description="Logical key in result_data['artifacts'].")
    path: str = Field(..., description="Relative source path below the runtime out/ directory.")
    media_type: str = Field(default="application/octet-stream", min_length=1)
    max_bytes: int | None = Field(
        default=None,
        gt=0,
        description="Optional per-output limit; host policy may impose a smaller limit.",
    )

    _validate_name = field_validator("name")(validate_artifact_name)
    _validate_path = field_validator("path")(validate_artifact_path)
    _validate_media_type = field_validator("media_type")(validate_utf8_text)


class ArtifactRecord(BaseModel):
    """Serializable content-addressed artifact returned by a completed step."""

    model_config = ConfigDict(frozen=True)

    name: str
    path: str
    media_type: str
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    content_base64: str = Field(repr=False)

    _validate_name = field_validator("name")(validate_artifact_name)
    _validate_path = field_validator("path")(validate_artifact_path)
    _validate_media_type = field_validator("media_type")(validate_utf8_text)

    @classmethod
    def from_bytes(
        cls,
        *,
        name: str,
        path: str,
        media_type: str,
        data: bytes,
    ) -> ArtifactRecord:
        return cls(
            name=name,
            path=path,
            media_type=media_type,
            size_bytes=len(data),
            sha256=hashlib.sha256(data).hexdigest(),
            content_base64=base64.b64encode(data).decode("ascii"),
        )

    def decode(self, *, max_bytes: int) -> bytes:
        """Decode and verify content without trusting serialized metadata."""

        # A valid base64 encoding of at most N bytes occupies at most
        # 4*ceil(N/3) characters. Reject oversized payloads before allocating
        # the decoded buffer.
        encoded_limit = 4 * ((max_bytes + 2) // 3)
        if len(self.content_base64) > encoded_limit:
            raise ValueError(f"artifact {self.name!r} exceeds the {max_bytes}-byte input limit")
        try:
            data = base64.b64decode(self.content_base64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError(f"artifact {self.name!r} has invalid base64 content") from exc
        if len(data) > max_bytes:
            raise ValueError(f"artifact {self.name!r} exceeds the {max_bytes}-byte input limit")
        if len(data) != self.size_bytes:
            raise ValueError(f"artifact {self.name!r} size metadata does not match content")
        if hashlib.sha256(data).hexdigest() != self.sha256:
            raise ValueError(f"artifact {self.name!r} sha256 does not match content")
        return data


__all__ = ["ArtifactInput", "ArtifactOutput", "ArtifactRecord"]
