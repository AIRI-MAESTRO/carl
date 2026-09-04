"""Focused contract tests for host-owned command capabilities."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import FrozenInstanceError
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from mmar_carl.command_capabilities import (
    CommandCapability,
    CommandCapabilityRegistry,
    CommandPlanEnvelope,
    CommandPlanRecord,
)


class CountArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    repeat: int


class CoercibleArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    count: int


def _capability(
    *,
    capability_id: str = "text.count",
    revision: str = "r1",
    executable: str = "/usr/bin/printf",
    static_args: tuple[str, ...] = ("--",),
    description: str = "Print validated text",
    builder: Any = None,
) -> CommandCapability:
    return CommandCapability(
        capability_id=capability_id,
        description=description,
        executable=executable,
        static_args=static_args,
        argument_model=CountArguments,
        argv_builder=builder or (lambda args: (args.text,) * args.repeat),
        revision=revision,
    )


def test_validate_plan_and_resolve_produce_transparent_record_and_immutable_argv() -> None:
    calls = 0

    def build(arguments: BaseModel) -> Sequence[str]:
        nonlocal calls
        calls += 1
        assert isinstance(arguments, CountArguments)
        return [arguments.text] * arguments.repeat

    registry = CommandCapabilityRegistry([_capability(builder=build)])
    record = registry.validate_plan(
        {"capability_id": "text.count", "arguments": {"repeat": 2, "text": "привет"}},
        selected_ids=["text.count"],
    )

    assert calls == 0
    assert record.model_dump(mode="json") == {
        "capability_id": "text.count",
        "capability_revision": "r1",
        "capability_fingerprint": registry.capabilities["text.count"].fingerprint,
        "arguments": {"repeat": 2, "text": "привет"},
        "arguments_sha256": hashlib.sha256(
            '{"repeat":2,"text":"привет"}'.encode(),
        ).hexdigest(),
    }

    invocation = registry.resolve(record.model_dump(mode="json"), selected_ids=["text.count"])
    assert calls == 1
    assert invocation.argv == ("/usr/bin/printf", "--", "привет", "привет")
    assert invocation.static_prefix_count == 2
    assert invocation.arguments_json == '{"repeat":2,"text":"привет"}'
    with pytest.raises(TypeError):
        invocation.argv[0] = "changed"  # type: ignore[index]


def test_validation_is_strict_and_extra_arguments_are_forbidden() -> None:
    registry = CommandCapabilityRegistry(
        [
            CommandCapability(
                capability_id="strict.count",
                description="Strict count",
                executable="count",
                static_args=(),
                argument_model=CoercibleArguments,
                argv_builder=lambda args: (str(args.count),),
                revision="1",
            )
        ]
    )

    with pytest.raises(ValidationError, match="valid integer"):
        registry.validate_plan({"capability_id": "strict.count", "arguments": {"count": "2"}})
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        registry.validate_plan(
            {"capability_id": "strict.count", "arguments": {"count": 2, "executable": "/bin/sh"}}
        )


def test_capability_requires_strict_id_revision_and_extra_forbid_model() -> None:
    class PermissiveArguments(BaseModel):
        value: str

    with pytest.raises(ValueError, match="capability_id"):
        _capability(capability_id="shell command")
    with pytest.raises(ValueError, match="capability_id"):
        _capability(capability_id="текст")
    with pytest.raises(ValueError, match="revision"):
        _capability(revision="release 1")
    with pytest.raises(ValueError, match="extra='forbid'"):
        CommandCapability(
            capability_id="bad.model",
            description="Bad model",
            executable="bad",
            static_args=(),
            argument_model=PermissiveArguments,
            argv_builder=lambda _: (),
            revision="1",
        )


def test_capability_fingerprint_covers_declarative_contract() -> None:
    base = _capability()
    assert base.fingerprint == _capability().fingerprint
    assert base.fingerprint != _capability(revision="r2").fingerprint
    assert base.fingerprint != _capability(executable="/usr/bin/wc").fingerprint
    assert base.fingerprint != _capability(static_args=("-c",)).fingerprint
    assert base.fingerprint != _capability(description="Changed planner meaning").fingerprint

    class DifferentArguments(BaseModel):
        model_config = ConfigDict(extra="forbid")

        text: str

    different_schema = CommandCapability(
        capability_id="text.count",
        description=base.description,
        executable=base.executable,
        static_args=base.static_args,
        argument_model=DifferentArguments,
        argv_builder=lambda args: (args.text,),
        revision=base.revision,
    )
    assert base.fingerprint != different_schema.fingerprint


def test_registry_is_immutable_and_rejects_duplicate_capabilities() -> None:
    capability = _capability()
    registry = CommandCapabilityRegistry([capability])
    with pytest.raises(TypeError):
        registry.capabilities["other"] = capability  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        registry.max_prompt_bytes = 1  # type: ignore[misc]
    with pytest.raises(ValueError, match="duplicate capability_id"):
        CommandCapabilityRegistry([capability, capability])


def test_registry_enforces_host_owned_prompt_and_input_bounds() -> None:
    registry = CommandCapabilityRegistry(
        [_capability()],
        max_prompt_bytes=10_000,
        max_input_value_bytes=123,
    )
    assert registry.max_prompt_bytes == 10_000
    assert registry.max_input_value_bytes == 123
    assert registry.max_response_bytes == 64_000

    with pytest.raises(ValueError, match="max_prompt_bytes"):
        CommandCapabilityRegistry([_capability()], max_prompt_bytes=10)
    with pytest.raises(ValueError, match="must be positive"):
        CommandCapabilityRegistry([], max_input_value_bytes=0)
    with pytest.raises(TypeError, match="must be an integer"):
        CommandCapabilityRegistry([], max_prompt_bytes=True)  # type: ignore[arg-type]


def test_manifest_contains_only_requested_json_safe_metadata() -> None:
    registry = CommandCapabilityRegistry(
        [_capability(), _capability(capability_id="text.other", revision="r2")]
    )
    manifest = registry.manifest(["text.other"])
    assert [entry["capability_id"] for entry in manifest] == ["text.other"]
    assert set(manifest[0]) == {
        "capability_id",
        "description",
        "arguments_schema",
        "revision",
        "fingerprint",
    }
    assert "argv_builder" not in json.dumps(manifest)
    json.dumps(manifest, allow_nan=False)

    with pytest.raises(ValueError, match="duplicate selected"):
        registry.manifest(["text.count", "text.count"])
    with pytest.raises(ValueError, match="unknown capability_id"):
        registry.manifest(["text.missing"])


def test_resolve_rejects_unselected_unknown_and_stale_capabilities_before_builder() -> None:
    calls = 0

    def build(arguments: BaseModel) -> Sequence[str]:
        nonlocal calls
        calls += 1
        return ("unused",)

    original = CommandCapabilityRegistry([_capability(builder=build)])
    record = original.validate_plan(
        CommandPlanEnvelope(capability_id="text.count", arguments={"text": "x", "repeat": 1})
    )

    with pytest.raises(ValueError, match="outside the selected"):
        original.resolve(record, selected_ids=[])
    assert calls == 0

    stale_registry = CommandCapabilityRegistry([_capability(revision="r2", builder=build)])
    with pytest.raises(ValueError, match="revision"):
        stale_registry.resolve(record)
    assert calls == 0

    unknown = record.model_copy(update={"capability_id": "text.missing"})
    with pytest.raises(ValueError, match="unknown capability_id"):
        original.resolve(unknown)
    assert calls == 0


def test_validate_record_is_side_effect_free_until_explicit_materialization() -> None:
    calls = 0

    def build(arguments: BaseModel) -> Sequence[str]:
        nonlocal calls
        calls += 1
        return (arguments.text, str(arguments.repeat))  # type: ignore[attr-defined]

    registry = CommandCapabilityRegistry([_capability(builder=build)])
    record = registry.validate_plan(
        CommandPlanEnvelope(
            capability_id="text.count",
            arguments={"text": "x", "repeat": 2},
        )
    )

    validated = registry.validate_record(record)
    assert calls == 0
    assert validated.executable == "/usr/bin/printf"
    assert validated.static_args == ("--",)

    resolved = registry.materialize(validated)
    assert calls == 1
    assert resolved.argv == ("/usr/bin/printf", "--", "x", "2")


def test_record_detects_argument_and_provenance_tampering_before_builder() -> None:
    calls = 0

    def build(arguments: BaseModel) -> Sequence[str]:
        nonlocal calls
        calls += 1
        return ("unused",)

    registry = CommandCapabilityRegistry([_capability(builder=build)])
    record = registry.validate_plan(
        {"capability_id": "text.count", "arguments": {"text": "x", "repeat": 1}}
    )
    dumped = record.model_dump(mode="json")
    dumped["arguments"]["repeat"] = 2
    with pytest.raises(ValidationError, match="arguments_sha256"):
        registry.resolve(dumped)
    assert calls == 0

    stale_fingerprint = record.model_copy(update={"capability_fingerprint": "0" * 64})
    with pytest.raises(ValueError, match="fingerprint"):
        registry.resolve(stale_fingerprint)
    assert calls == 0


@pytest.mark.parametrize(
    ("builder_result", "message"),
    [
        ("one string", "sequence of strings"),
        (iter(["generator"]), "sequence of strings"),
        ([1], "must be a string"),
        (["bad\x00arg"], "must not contain NUL"),
        (["\ud800"], "valid UTF-8"),
    ],
)
def test_resolve_rejects_invalid_builder_results(builder_result: Any, message: str) -> None:
    calls = 0

    def build(_: BaseModel) -> Any:
        nonlocal calls
        calls += 1
        return builder_result

    registry = CommandCapabilityRegistry([_capability(builder=build)])
    record = registry.validate_plan(
        {"capability_id": "text.count", "arguments": {"text": "x", "repeat": 1}}
    )
    with pytest.raises((TypeError, ValueError), match=message):
        registry.resolve(record)
    assert calls == 1


def test_wire_models_forbid_extra_fields_and_non_json_arguments() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        CommandPlanEnvelope.model_validate(
            {"capability_id": "text.count", "arguments": {}, "command": ["/bin/sh"]},
            strict=True,
        )
    with pytest.raises(ValidationError, match="valid JSON"):
        CommandPlanEnvelope(capability_id="text.count", arguments={"not_json": object()})

    registry = CommandCapabilityRegistry([_capability()])
    record = registry.validate_plan(
        {"capability_id": "text.count", "arguments": {"text": "x", "repeat": 1}}
    )
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        CommandPlanRecord.model_validate({**record.model_dump(), "executable": "/bin/sh"}, strict=True)
