"""Host-owned capabilities for turning typed plans into command arguments.

A model may choose a capability and provide JSON arguments, but it never
chooses an executable or constructs ``argv``.  The application owns this
registry and the trusted builders in it.  Serialized plans therefore remain
requests, not authority to execute a program.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_CAPABILITY_ID_RE = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{0,127}\Z", re.ASCII)
_REVISION_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z", re.ASCII)


def _validate_capability_id(value: str) -> str:
    if _CAPABILITY_ID_RE.fullmatch(value) is None:
        raise ValueError(
            "capability_id must be 1-128 ASCII characters matching "
            "[A-Za-z][A-Za-z0-9_.-]*"
        )
    return value


def _validate_revision(value: str) -> str:
    if _REVISION_RE.fullmatch(value) is None:
        raise ValueError(
            "revision must be 1-128 ASCII characters matching "
            "[A-Za-z0-9][A-Za-z0-9_.-]*"
        )
    return value


def _validate_utf8(value: str, *, field_name: str, allow_empty: bool = True) -> str:
    if not allow_empty and not value:
        raise ValueError(f"{field_name} must not be empty")
    if "\x00" in value:
        raise ValueError(f"{field_name} must not contain NUL")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{field_name} must be valid UTF-8") from exc
    return value


def _canonical_json(value: Any, *, field_name: str) -> str:
    try:
        text = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        text.encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise ValueError(f"{field_name} must contain valid JSON data") from exc
    return text


def _normalize_json_object(value: dict[str, Any], *, field_name: str) -> dict[str, Any]:
    """Return a detached value containing only JSON-native objects."""

    return json.loads(_canonical_json(value, field_name=field_name))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class CommandPlanEnvelope(BaseModel):
    """Untrusted structured output produced by a command-planning model."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    capability_id: str
    arguments: dict[str, Any]

    _validate_id = field_validator("capability_id")(_validate_capability_id)

    @field_validator("arguments")
    @classmethod
    def normalize_arguments(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _normalize_json_object(value, field_name="arguments")


class CommandPlanRecord(BaseModel):
    """JSON-safe, provenance-bound capability plan stored in step output."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    capability_id: str
    capability_revision: str
    capability_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    arguments: dict[str, Any]
    arguments_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    _validate_id = field_validator("capability_id")(_validate_capability_id)
    _validate_capability_revision = field_validator("capability_revision")(_validate_revision)

    @field_validator("arguments")
    @classmethod
    def normalize_arguments(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _normalize_json_object(value, field_name="arguments")

    @model_validator(mode="after")
    def validate_arguments_digest(self) -> CommandPlanRecord:
        canonical = _canonical_json(self.arguments, field_name="arguments")
        if not _constant_time_equal(self.arguments_sha256, _sha256_text(canonical)):
            raise ValueError("arguments_sha256 does not match arguments")
        return self


@dataclass(frozen=True, slots=True)
class CommandCapability:
    """One host-owned typed command capability.

    ``argv_builder`` receives a strictly validated ``argument_model`` instance
    and returns only the dynamic suffix. It must be a pure deterministic
    compiler with no I/O or side effects because approval inspects the compiled
    argv before execution. The executable and static prefix are declared
    independently so policy can reject an ineligible plan before compilation.
    """

    capability_id: str
    description: str
    executable: str
    static_args: tuple[str, ...]
    argument_model: type[BaseModel]
    argv_builder: Callable[[BaseModel], Sequence[str]] = field(repr=False, compare=False)
    revision: str
    _arguments_schema_json: str = field(init=False, repr=False, compare=False)
    _fingerprint: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        _validate_capability_id(self.capability_id)
        _validate_revision(self.revision)
        _validate_utf8(self.description, field_name="description", allow_empty=False)
        _validate_utf8(self.executable, field_name="executable", allow_empty=False)

        if not isinstance(self.static_args, tuple):
            raise TypeError("static_args must be a tuple")
        for index, argument in enumerate(self.static_args):
            if not isinstance(argument, str):
                raise TypeError(f"static_args[{index}] must be a string")
            _validate_utf8(argument, field_name=f"static_args[{index}]")

        if not isinstance(self.argument_model, type) or not issubclass(self.argument_model, BaseModel):
            raise TypeError("argument_model must be a BaseModel subclass")
        if self.argument_model.model_config.get("extra") != "forbid":
            raise ValueError("argument_model must set ConfigDict(extra='forbid')")
        if not callable(self.argv_builder):
            raise TypeError("argv_builder must be callable")

        schema_json = _canonical_json(
            self.argument_model.model_json_schema(),
            field_name="argument_model JSON schema",
        )
        fingerprint_payload = {
            "argument_schema": json.loads(schema_json),
            "capability_id": self.capability_id,
            "description": self.description,
            "executable": self.executable,
            "revision": self.revision,
            "static_args": list(self.static_args),
        }
        object.__setattr__(self, "_arguments_schema_json", schema_json)
        object.__setattr__(
            self,
            "_fingerprint",
            _sha256_text(_canonical_json(fingerprint_payload, field_name="capability fingerprint")),
        )

    @property
    def fingerprint(self) -> str:
        """Stable digest of every declarative execution/validation field."""

        return self._fingerprint

    @property
    def arguments_schema(self) -> dict[str, Any]:
        """Return a detached JSON-safe copy of the snapshotted schema."""

        return json.loads(self._arguments_schema_json)


@dataclass(frozen=True, slots=True)
class ValidatedCommandPlan:
    """Immutable plan validated without invoking the capability builder."""

    capability_id: str
    capability_revision: str
    capability_fingerprint: str
    executable: str
    static_args: tuple[str, ...]
    arguments_json: str
    arguments_sha256: str


@dataclass(frozen=True, slots=True)
class ResolvedCommandInvocation:
    """Immutable result of resolving one provenance-bound plan."""

    capability_id: str
    capability_revision: str
    capability_fingerprint: str
    argv: tuple[str, ...]
    static_prefix_count: int
    arguments_json: str


@dataclass(frozen=True, slots=True, init=False)
class CommandCapabilityRegistry:
    """Immutable host-owned mapping of capability IDs to trusted builders."""

    _capabilities: Mapping[str, CommandCapability] = field(repr=False)
    max_prompt_bytes: int
    max_input_value_bytes: int
    max_response_bytes: int

    def __init__(
        self,
        capabilities: Sequence[CommandCapability],
        *,
        max_prompt_bytes: int = 128_000,
        max_input_value_bytes: int = 32_000,
        max_response_bytes: int = 64_000,
    ) -> None:
        _validate_positive_int(max_prompt_bytes, field_name="max_prompt_bytes")
        _validate_positive_int(max_input_value_bytes, field_name="max_input_value_bytes")
        _validate_positive_int(max_response_bytes, field_name="max_response_bytes")
        if isinstance(capabilities, (str, bytes, bytearray)) or not isinstance(capabilities, Sequence):
            raise TypeError("capabilities must be a sequence")
        by_id: dict[str, CommandCapability] = {}
        for capability in capabilities:
            if not isinstance(capability, CommandCapability):
                raise TypeError("capabilities must contain CommandCapability instances")
            if capability.capability_id in by_id:
                raise ValueError(f"duplicate capability_id: {capability.capability_id!r}")
            by_id[capability.capability_id] = capability
        object.__setattr__(self, "_capabilities", MappingProxyType(by_id))
        object.__setattr__(self, "max_prompt_bytes", max_prompt_bytes)
        object.__setattr__(self, "max_input_value_bytes", max_input_value_bytes)
        object.__setattr__(self, "max_response_bytes", max_response_bytes)
        self._validate_manifest_size(self._manifest_entries(self._select(None)))

    @property
    def capabilities(self) -> Mapping[str, CommandCapability]:
        return self._capabilities

    def manifest(self, selected_ids: Sequence[str] | None = None) -> list[dict[str, Any]]:
        """Return the selected model-visible metadata, never trusted builders."""

        selected = self._select(selected_ids)
        manifest = self._manifest_entries(selected)
        self._validate_manifest_size(manifest)
        return manifest

    @staticmethod
    def _manifest_entries(selected: Sequence[CommandCapability]) -> list[dict[str, Any]]:
        return [
            {
                "capability_id": capability.capability_id,
                "description": capability.description,
                "arguments_schema": capability.arguments_schema,
                "revision": capability.revision,
                "fingerprint": capability.fingerprint,
            }
            for capability in selected
        ]

    def _validate_manifest_size(self, manifest: list[dict[str, Any]]) -> None:
        encoded = _canonical_json(manifest, field_name="capability manifest").encode("utf-8")
        if len(encoded) > self.max_prompt_bytes:
            raise ValueError(
                f"capability manifest exceeds max_prompt_bytes "
                f"({len(encoded)} > {self.max_prompt_bytes})"
            )

    def validate_plan(
        self,
        envelope: CommandPlanEnvelope | Mapping[str, Any],
        selected_ids: Sequence[str] | None = None,
    ) -> CommandPlanRecord:
        """Strictly validate untrusted JSON arguments and bind provenance."""

        plan = CommandPlanEnvelope.model_validate(envelope, strict=True)
        capability = self._select_capability(plan.capability_id, selected_ids)
        validated_arguments, canonical = self._validate_arguments(capability, plan.arguments)
        return CommandPlanRecord(
            capability_id=capability.capability_id,
            capability_revision=capability.revision,
            capability_fingerprint=capability.fingerprint,
            arguments=validated_arguments,
            arguments_sha256=_sha256_text(canonical),
        )

    def resolve(
        self,
        record: CommandPlanRecord | Mapping[str, Any],
        selected_ids: Sequence[str] | None = None,
    ) -> ResolvedCommandInvocation:
        """Revalidate a stored plan, then invoke its pure trusted builder once."""

        return self.materialize(self.validate_record(record, selected_ids))

    def validate_record(
        self,
        record: CommandPlanRecord | Mapping[str, Any],
        selected_ids: Sequence[str] | None = None,
    ) -> ValidatedCommandPlan:
        """Validate provenance and arguments without calling ``argv_builder``.

        This phase lets the executor reject an ineligible executable/runtime
        through ``CommandPolicy`` before any host builder code is evaluated.
        """

        plan = CommandPlanRecord.model_validate(record, strict=True)
        capability = self._select_capability(plan.capability_id, selected_ids)
        if plan.capability_revision != capability.revision:
            raise ValueError("command plan capability revision does not match the registry")
        if not _constant_time_equal(plan.capability_fingerprint, capability.fingerprint):
            raise ValueError("command plan capability fingerprint does not match the registry")

        _validated_arguments, canonical = self._validate_arguments(capability, plan.arguments)
        if not _constant_time_equal(plan.arguments_sha256, _sha256_text(canonical)):
            raise ValueError("command plan arguments changed after validation")

        return ValidatedCommandPlan(
            capability_id=capability.capability_id,
            capability_revision=capability.revision,
            capability_fingerprint=capability.fingerprint,
            executable=capability.executable,
            static_args=capability.static_args,
            arguments_json=canonical,
            arguments_sha256=plan.arguments_sha256,
        )

    def materialize(self, plan: ValidatedCommandPlan) -> ResolvedCommandInvocation:
        """Compile a validated plan into immutable argv exactly once.

        ``argv_builder`` is trusted host code but MUST be pure, deterministic,
        and free of I/O or other side effects: approval needs to inspect its
        output before execution. Runtime effects belong exclusively in the
        approved ``CommandStep`` invocation.
        """

        if not isinstance(plan, ValidatedCommandPlan):
            raise TypeError("plan must be a ValidatedCommandPlan")
        capability = self._select_capability(plan.capability_id, None)
        if plan.capability_revision != capability.revision:
            raise ValueError("validated plan capability revision no longer matches the registry")
        if not _constant_time_equal(plan.capability_fingerprint, capability.fingerprint):
            raise ValueError("validated plan capability fingerprint no longer matches the registry")
        if plan.executable != capability.executable or plan.static_args != capability.static_args:
            raise ValueError("validated plan command prefix no longer matches the registry")
        if not _constant_time_equal(plan.arguments_sha256, _sha256_text(plan.arguments_json)):
            raise ValueError("validated plan arguments changed before materialization")

        arguments = json.loads(plan.arguments_json)
        model_arguments = capability.argument_model.model_validate(arguments, strict=True)
        suffix = capability.argv_builder(model_arguments)
        normalized_suffix = _validate_argv_suffix(suffix)
        argv = (capability.executable, *capability.static_args, *normalized_suffix)
        return ResolvedCommandInvocation(
            capability_id=capability.capability_id,
            capability_revision=capability.revision,
            capability_fingerprint=capability.fingerprint,
            argv=argv,
            static_prefix_count=1 + len(capability.static_args),
            arguments_json=plan.arguments_json,
        )

    def _select(self, selected_ids: Sequence[str] | None) -> tuple[CommandCapability, ...]:
        if selected_ids is None:
            return tuple(self._capabilities[capability_id] for capability_id in sorted(self._capabilities))
        if isinstance(selected_ids, (str, bytes, bytearray)) or not isinstance(selected_ids, Sequence):
            raise TypeError("selected_ids must be a sequence of capability IDs")

        seen: set[str] = set()
        selected: list[CommandCapability] = []
        for capability_id in selected_ids:
            if not isinstance(capability_id, str):
                raise TypeError("selected_ids must contain strings")
            _validate_capability_id(capability_id)
            if capability_id in seen:
                raise ValueError(f"duplicate selected capability_id: {capability_id!r}")
            seen.add(capability_id)
            try:
                selected.append(self._capabilities[capability_id])
            except KeyError as exc:
                raise ValueError(f"unknown capability_id: {capability_id!r}") from exc
        return tuple(selected)

    def _select_capability(
        self,
        capability_id: str,
        selected_ids: Sequence[str] | None,
    ) -> CommandCapability:
        selected = {capability.capability_id: capability for capability in self._select(selected_ids)}
        try:
            return selected[capability_id]
        except KeyError as exc:
            if capability_id in self._capabilities:
                raise ValueError(f"capability_id {capability_id!r} is outside the selected capability set") from exc
            raise ValueError(f"unknown capability_id: {capability_id!r}") from exc

    @staticmethod
    def _validate_arguments(
        capability: CommandCapability,
        arguments: dict[str, Any],
    ) -> tuple[dict[str, Any], str]:
        model_arguments = capability.argument_model.model_validate(arguments, strict=True)
        dumped = model_arguments.model_dump(mode="json", round_trip=True)
        normalized = _normalize_json_object(dumped, field_name="validated capability arguments")
        return normalized, _canonical_json(normalized, field_name="validated capability arguments")


def _validate_argv_suffix(value: Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise TypeError("argv_builder must return a sequence of strings")
    suffix: list[str] = []
    for index, argument in enumerate(value):
        if not isinstance(argument, str):
            raise TypeError(f"argv_builder result[{index}] must be a string")
        suffix.append(_validate_utf8(argument, field_name=f"argv_builder result[{index}]"))
    return tuple(suffix)


def _constant_time_equal(left: str, right: str) -> bool:
    return hmac.compare_digest(left, right)


def _validate_positive_int(value: int, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if value <= 0:
        raise ValueError(f"{field_name} must be positive")
    return value


__all__ = [
    "CommandCapability",
    "CommandCapabilityRegistry",
    "CommandPlanEnvelope",
    "CommandPlanRecord",
    "ResolvedCommandInvocation",
    "ValidatedCommandPlan",
]
