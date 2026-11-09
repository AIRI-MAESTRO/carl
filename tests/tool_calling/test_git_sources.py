"""Regression tests for loading tools and skills from git by commit with SHA256 pinning."""

import errno
import hashlib
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from threading import Barrier
from unittest.mock import MagicMock

import pytest

from mmar_carl.models.agent_skill import AgentSkillSource
from mmar_carl.models.config import GitToolSource, ToolDiscoveryStepConfig, ToolStepConfig
from mmar_carl.models.context import ReasoningContext
from mmar_carl.models.steps import ToolDiscoveryStepDescription, ToolStepDescription
from mmar_carl.skill_loader import SkillLoader
from mmar_carl.step_executors import ToolDiscoveryStepExecutor, ToolStepExecutor


@pytest.fixture
def git_artifacts(tmp_path, monkeypatch):
    from mmar_carl import skill_loader

    monkeypatch.setattr(
        skill_loader, "_resolve_git_directory",
        partial(skill_loader._resolve_git_directory, cache_dir=str(tmp_path / "cache")),
    )
    repo = tmp_path / "repo"
    tools = repo / "tools"
    skill = repo / "skills" / "sample"
    tools.mkdir(parents=True)
    skill.mkdir(parents=True)
    module = tools / "tool.py"
    module.write_text(
        "from mmar_carl import carl_tool\n"
        "@carl_tool(tags=['math'])\n"
        "def answer():\n"
        "    return 42\n"
        "def helper():\n"
        "    return 0\n",
        encoding="utf-8",
    )
    skill_md = skill / "SKILL.md"
    (tools / "dataclass_tool.py").write_text(
        "from __future__ import annotations\n"
        "from dataclasses import dataclass\n"
        "from mmar_carl import carl_tool\n"
        "@dataclass\n"
        "class Answer:\n"
        "    value: int\n"
        "@carl_tool\n"
        "def answer():\n"
        "    return Answer(42).value\n",
        encoding="utf-8",
    )
    (tools / "broken.py").write_text("raise RuntimeError('broken import')\n", encoding="utf-8")
    skill_md.write_text(
        "---\nname: sample\ndescription: Sample skill\n---\nPinned instructions.\n",
        encoding="utf-8",
    )
    hashes = {
        "tool.py": hashlib.sha256(module.read_bytes()).hexdigest(),
        "SKILL.md": hashlib.sha256(skill_md.read_bytes()).hexdigest(),
    }

    def git(*arguments):
        return subprocess.run(
            ["git", "-C", str(repo), *arguments],
            check=True, capture_output=True, text=True,
        ).stdout.strip()

    git("init")
    git("add", ".")
    git("-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-m", "Artifacts")
    commit = git("rev-parse", "HEAD")
    module.write_text(module.read_text(encoding="utf-8").replace("42", "99"), encoding="utf-8")
    skill_md.write_text(
        skill_md.read_text(encoding="utf-8").replace("Pinned", "Latest"), encoding="utf-8",
    )
    git("add", ".")
    git("-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-m", "New versions")
    return repo.as_uri(), commit, hashes


def tool_step(url, commit, sha256, function=None):
    return ToolDiscoveryStepDescription(
        number=1,
        title="Load git tools",
        config=ToolDiscoveryStepConfig.model_validate({
            "source": {
                "git_url": url,
                "git_ref": commit,
                "git_subdirectory": "tools",
                "module_file": "tool.py",
                "function": function,
                "sha256": sha256,
            },
        }),
    )


@pytest.mark.asyncio
async def test_git_tool_by_commit_registers_and_runs(git_artifacts):
    url, commit, hashes = git_artifacts
    context = ReasoningContext(outer_context="", api=MagicMock())
    step = tool_step(url, commit, {"tool.py": hashes["tool.py"]})
    assert isinstance(step.config.source, GitToolSource)
    discovered = await ToolDiscoveryStepExecutor().execute(step, context)
    assert discovered.success, discovered.error_message
    assert discovered.result_data["tool_names"] == ["answer"]
    assert context.get_tool_tags("answer") == {"math"}
    called = await ToolStepExecutor().execute(
        ToolStepDescription(
            number=2, title="Call git tool", config=ToolStepConfig(tool_name="answer"),
        ),
        context,
    )
    assert called.success, called.error_message
    assert called.result_data == 42


@pytest.mark.asyncio
async def test_git_tool_function_without_decorator(git_artifacts):
    url, commit, hashes = git_artifacts
    context = ReasoningContext(outer_context="", api=MagicMock())
    discovered = await ToolDiscoveryStepExecutor().execute(
        tool_step(url, commit, {"tool.py": hashes["tool.py"]}, function="helper"), context,
    )
    assert discovered.success, discovered.error_message
    assert discovered.result_data["tool_names"] == ["helper"]


@pytest.mark.asyncio
async def test_git_tool_hash_mismatch_does_not_register(git_artifacts):
    url, commit, hashes = git_artifacts
    executor = ToolDiscoveryStepExecutor()
    cached = await executor.execute(
        tool_step(url, commit, {"tool.py": hashes["tool.py"]}),
        ReasoningContext(outer_context="", api=MagicMock()),
    )
    assert cached.success, cached.error_message
    context = ReasoningContext(outer_context="", api=MagicMock())
    result = await executor.execute(tool_step(url, commit, {"tool.py": "0" * 64}), context)
    assert not result.success
    assert "SHA256 mismatch" in result.error_message
    assert not context.has_tool("answer")


def test_git_skill_by_commit_loads(git_artifacts, tmp_path):
    url, commit, hashes = git_artifacts
    loader = SkillLoader(cache_dir=str(tmp_path / "skill-cache"))
    manifest = loader.load_sync(AgentSkillSource(
        git_url=url,
        git_ref=commit,
        git_subdirectory="skills/sample",
        sha256={"SKILL.md": hashes["SKILL.md"]},
    ))
    assert manifest.name == "sample"
    assert manifest.instructions == "Pinned instructions."


@pytest.mark.asyncio
async def test_named_git_function_preserves_tags_for_restricted_call(git_artifacts):
    url, commit, hashes = git_artifacts
    context = ReasoningContext(outer_context="", api=MagicMock())
    discovered = await ToolDiscoveryStepExecutor().execute(
        tool_step(url, commit, {"tool.py": hashes["tool.py"]}, function="answer"), context,
    )
    assert discovered.success, discovered.error_message
    assert context.get_tool_tags("answer") == {"math"}
    called = await ToolStepExecutor().execute(
        ToolStepDescription(number=2, title="Restricted call", config=ToolStepConfig(
            tool_name="answer", allowed_tool_tags=["math"],
        )), context,
    )
    assert called.success, called.error_message
    assert called.result_data == 42


@pytest.mark.asyncio
async def test_git_module_with_postponed_dataclass_annotations(git_artifacts):
    url, commit, _ = git_artifacts
    step = tool_step(url, commit, None)
    step.config.source.module_file = "dataclass_tool.py"
    context = ReasoningContext(outer_context="", api=MagicMock())
    discovered = await ToolDiscoveryStepExecutor().execute(step, context)
    assert discovered.success, discovered.error_message
    called = await ToolStepExecutor().execute(
        ToolStepDescription(number=2, title="Call dataclass tool", config=ToolStepConfig(tool_name="answer")),
        context,
    )
    assert called.success, called.error_message
    assert called.result_data == 42


@pytest.mark.asyncio
@pytest.mark.parametrize("previously_loaded", [False, True])
async def test_failed_git_import_restores_module_registry(git_artifacts, monkeypatch, previously_loaded):
    from types import ModuleType
    from mmar_carl import skill_loader

    url, commit, _ = git_artifacts
    directory = skill_loader._resolve_git_directory(url, commit, "tools")
    path = (directory / "broken.py").resolve()
    name = f"_carl_tools_{hashlib.sha256(str(path).encode()).hexdigest()}"
    previous = ModuleType(name)
    if previously_loaded:
        monkeypatch.setitem(sys.modules, name, previous)
    else:
        monkeypatch.delitem(sys.modules, name, raising=False)
    step = tool_step(url, commit, None)
    step.config.source.module_file = "broken.py"
    result = await ToolDiscoveryStepExecutor().execute(step, ReasoningContext(outer_context="", api=MagicMock()))
    assert not result.success
    assert "broken import" in result.error_message
    if previously_loaded:
        assert sys.modules[name] is previous
    else:
        assert name not in sys.modules


def test_concurrent_cold_git_cache_publishes_once_and_checks_each_hash(git_artifacts, monkeypatch):
    from mmar_carl import skill_loader
    from mmar_carl.skill_resolver import SkillIntegrityError

    url, commit, hashes = git_artifacts
    barrier = Barrier(4, timeout=15)
    rename = Path.rename

    def simultaneous_publish(path, target):
        # Force all four real Git checkouts to finish before any publishes.
        if path.name == "repo":
            barrier.wait()
        return rename(path, target)

    monkeypatch.setattr(Path, "rename", simultaneous_publish)

    def load(index):
        expected = hashes["tool.py"] if index < 3 else "0" * 64
        try:
            return skill_loader._resolve_git_directory(url, commit, "tools", {"tool.py": expected})
        except SkillIntegrityError:
            return "hash rejected"

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(load, range(4)))
    assert results[0] == results[1] == results[2]
    assert (results[0] / "tool.py").is_file()
    assert results[3] == "hash rejected"
    # Temporary checkouts, including losing publishers, are removed.
    assert list(results[0].parent.parent.iterdir()) == [results[0].parent]


def test_git_cache_publication_does_not_swallow_other_errors(git_artifacts, monkeypatch):
    from mmar_carl import skill_loader

    url, commit, _ = git_artifacts

    def denied(path, target):
        raise PermissionError(errno.EACCES, "publication denied")

    monkeypatch.setattr(Path, "rename", denied)
    with pytest.raises(PermissionError, match="publication denied"):
        skill_loader._resolve_git_directory(url, commit, "tools")


@pytest.mark.live
@pytest.mark.asyncio
async def test_openrouter_free_runs_pinned_git_tool_and_structured_chain(git_artifacts):
    import os

    from mmar_carl import (
        OpenAIClientConfig,
        OpenAICompatibleClient,
        ReasoningChain,
        StructuredOutputStepConfig,
        StructuredOutputStepDescription,
    )

    model = os.environ.get("LIVE_TEST_MODEL", "")
    key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not model.endswith(":free") or not key:
        pytest.skip("set a free LIVE_TEST_MODEL and an OpenRouter API key")

    url, commit, hashes = git_artifacts
    steps = [
        tool_step(url, commit, {"tool.py": hashes["tool.py"]}),
        ToolStepDescription(
            number=2, title="Read pinned answer", dependencies=[1],
            config=ToolStepConfig(tool_name="answer", allowed_tool_tags=["math"]),
        ),
        StructuredOutputStepDescription(
            number=3, title="Echo the pinned answer", dependencies=[2],
            config=StructuredOutputStepConfig(
                input_mapping={"answer": "$steps.2.result_data", "label": "$outer_context.label"},
                instruction="Copy both input values exactly into the output JSON.",
                output_schema={
                    "type": "object",
                    "properties": {"answer": {"type": "integer"}, "label": {"type": "string"}},
                    "required": ["answer", "label"],
                    "additionalProperties": False,
                },
            ),
        ),
    ]
    chain = ReasoningChain.from_json(ReasoningChain(steps=steps, fail_fast=True).to_json())
    client = OpenAICompatibleClient(OpenAIClientConfig(
        base_url="https://openrouter.ai/api/v1", api_key=key, model=model,
        temperature=0, max_tokens=4096, timeout=30,
    ))
    context = ReasoningContext(outer_context="Copy the inputs.", runtime_input={"label": "CARL"}, api=client)
    try:
        result = await chain.execute_async(context)
    finally:
        await context.close()

    assert result.success, [(step.step_number, step.error_message) for step in result.get_failed_steps()]
    assert result.step_results[1].tool_name == "answer"
    assert result.step_results[2].result_data == {"answer": 42, "label": "CARL"}
    assert result.step_results[2].model == model
    assert result.token_usage["total"] > 0
