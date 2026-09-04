"""
Claude-Code-as-step tests.

All tests are hermetic: the ``claude`` CLI is replaced with a small fake
executable that echoes the ``-p`` prompt back inside a canned payload
matching the shapes Claude Code 2.x emits (``--output-format json`` single
object, or ``stream-json`` JSONL events). The fake's behaviour is switched
via the step's ``env`` field (``FAKE_CLAUDE_MODE``, ``FAKE_CLAUDE_RESULT``),
and it can dump its argv to a file (``FAKE_CLAUDE_ARGV_FILE``) so tests can
assert CLI flag construction.
"""

import json
import stat
from unittest.mock import MagicMock

import pytest

from mmar_carl import (
    ClaudeCodeStepConfig,
    ClaudeCodeStepDescription,
    ReasoningChain,
    ReasoningContext,
    StepCache,
    StepType,
    check_claude_code_cli,
)

FAKE_CLI_SOURCE = '''#!/usr/bin/env python3
import json, os, sys, time

argv = sys.argv[1:]
argv_file = os.environ.get("FAKE_CLAUDE_ARGV_FILE")
if argv_file:
    with open(argv_file, "a") as f:
        f.write(json.dumps(argv) + "\\n")

mode = os.environ.get("FAKE_CLAUDE_MODE", "success")
prompt = argv[argv.index("-p") + 1] if "-p" in argv else ""
stream = "stream-json" in argv

if "--version" in argv:
    if mode == "crash":
        sys.stderr.write("boom: CLI exploded")
        sys.exit(2)
    print("2.1.119 (Claude Code)")
    sys.exit(0)

if mode == "sleep":
    time.sleep(10)
if mode == "garbage":
    print("this is not json")
    sys.exit(0)
if mode == "crash":
    sys.stderr.write("boom: CLI exploded")
    sys.exit(2)

is_error = mode == "agent_error"
result_text = os.environ.get("FAKE_CLAUDE_RESULT")
if result_text is None:
    result_text = "agent failed mid-run" if is_error else "ECHO: " + prompt
payload = {
    "type": "result",
    "subtype": "success" if not is_error else "error_during_execution",
    "is_error": is_error,
    "result": result_text,
    "session_id": "sess-1234",
    "num_turns": 3,
    "duration_ms": 42,
    "total_cost_usd": 0.01,
    "usage": {
        "input_tokens": 10,
        "cache_creation_input_tokens": 100,
        "cache_read_input_tokens": 5,
        "output_tokens": 7,
    },
    "modelUsage": {"claude-opus-4-7": {"costUSD": 0.01}},
}

if stream:
    print(json.dumps({"type": "system", "subtype": "init", "session_id": "sess-1234"}))
    print(json.dumps({
        "type": "assistant",
        "message": {"content": [
            {"type": "text", "text": "Analyzing the task."},
            {"type": "tool_use", "name": "Read", "input": {}},
        ]},
    }))
    print(json.dumps({
        "type": "assistant",
        "message": {"content": [{"type": "text", "text": "All done."}]},
    }))
    print(json.dumps(payload))
else:
    print(json.dumps(payload))
'''


@pytest.fixture
def fake_cli(tmp_path):
    path = tmp_path / "fake_claude"
    path.write_text(FAKE_CLI_SOURCE)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return str(path)


def make_context(outer_context: str = "test input", **kwargs) -> ReasoningContext:
    return ReasoningContext(outer_context=outer_context, api=MagicMock(), **kwargs)


def make_step(number: int, fake_cli: str, **config_kwargs) -> ClaudeCodeStepDescription:
    config_kwargs.setdefault("cli_path", fake_cli)
    config_kwargs.setdefault("timeout", 15.0)
    return ClaudeCodeStepDescription(
        number=number,
        title=f"Claude Code step {number}",
        dependencies=config_kwargs.pop("dependencies", []),
        config=ClaudeCodeStepConfig(**config_kwargs),
    )


class TestClaudeCodeStepSuccess:
    async def test_full_chain_run_collects_result_and_session(self, fake_cli):
        step = make_step(1, fake_cli, task="Summarise the repo")
        chain = ReasoningChain(steps=[step])
        ctx = make_context()

        result = await chain.execute_async(ctx)

        assert result.success
        step_result = result.step_results[0]
        assert step_result.step_type == StepType.CLAUDE_CODE
        assert step_result.result == "ECHO: Summarise the repo"
        assert step_result.result_data["session_id"] == "sess-1234"
        assert step_result.result_data["total_cost_usd"] == 0.01
        assert step_result.result_data["num_turns"] == 3
        # 10 + 100 + 5 prompt-side, 7 completion
        assert step_result.token_usage == {"prompt": 115, "completion": 7, "total": 122}
        assert step_result.model == "claude-opus-4-7"
        # history entry appended in the standard format
        assert any("[CLAUDE CODE]" in entry for entry in step_result.updated_history)
        # session id persisted for later resume steps
        assert ctx.memory_read("step_1", namespace="claude_code") == "sess-1234"

    async def test_input_mapping_renders_task_template(self, fake_cli):
        step = make_step(
            1,
            fake_cli,
            task="Analyse: {question} using {source}",
            input_mapping={"question": "$memory.inputs.q", "source": "$outer_context"},
        )
        ctx = make_context(outer_context="repo docs")
        ctx.memory_write("q", "why is CI red?", namespace="inputs")

        result = await ReasoningChain(steps=[step]).execute_async(ctx)

        assert result.success
        assert result.step_results[0].result == "ECHO: Analyse: why is CI red? using repo docs"

    async def test_argv_flags_are_constructed(self, fake_cli, tmp_path):
        argv_file = tmp_path / "argv.jsonl"
        step = make_step(
            1,
            fake_cli,
            task="do things",
            model="sonnet",
            max_turns=4,
            allowed_tools=["Read", "Grep", "Bash(git log:*)"],
            disallowed_tools=["WebSearch"],
            permission_mode="acceptEdits",
            system_prompt="You are a terse reviewer.",
            append_system_prompt="Be terse.",
            add_dirs=["/tmp/extra"],
            resume_session="sess-literal",
            env={"FAKE_CLAUDE_ARGV_FILE": str(argv_file)},
        )

        result = await ReasoningChain(steps=[step]).execute_async(make_context())

        assert result.success
        argv = json.loads(argv_file.read_text().strip())
        assert argv[:4] == ["-p", "do things", "--output-format", "json"]
        assert argv[argv.index("--model") + 1] == "sonnet"
        assert argv[argv.index("--max-turns") + 1] == "4"
        assert argv[argv.index("--allowedTools") + 1] == "Read,Grep,Bash(git log:*)"
        assert argv[argv.index("--disallowedTools") + 1] == "WebSearch"
        assert argv[argv.index("--permission-mode") + 1] == "acceptEdits"
        assert argv[argv.index("--system-prompt") + 1] == "You are a terse reviewer."
        assert argv[argv.index("--append-system-prompt") + 1] == "Be terse."
        assert argv[argv.index("--add-dir") + 1] == "/tmp/extra"
        assert argv[argv.index("--resume") + 1] == "sess-literal"

    async def test_second_step_resumes_session_from_memory(self, fake_cli, tmp_path):
        argv_file = tmp_path / "argv2.jsonl"
        step1 = make_step(1, fake_cli, task="start work")
        step2 = make_step(
            2,
            fake_cli,
            dependencies=[1],
            task="continue work",
            resume_session="$memory.claude_code.step_1",
            env={"FAKE_CLAUDE_ARGV_FILE": str(argv_file)},
        )

        result = await ReasoningChain(steps=[step1, step2]).execute_async(make_context())

        assert result.success
        argv = json.loads(argv_file.read_text().strip())
        assert argv[argv.index("--resume") + 1] == "sess-1234"

    async def test_long_result_is_truncated(self, fake_cli):
        step = make_step(1, fake_cli, task="x" * 500, max_output_chars=50)

        result = await ReasoningChain(steps=[step]).execute_async(make_context())

        step_result = result.step_results[0]
        assert step_result.result.endswith("…[truncated]")
        assert len(step_result.result) < 100
        # full text still available in the raw payload
        assert len(step_result.result_data["payload"]["result"]) > 500


class TestClaudeCodeStepStreaming:
    async def test_streaming_forwards_chunks_and_result(self, fake_cli, tmp_path):
        argv_file = tmp_path / "argv.jsonl"
        chunks: list[str] = []
        step = make_step(
            1,
            fake_cli,
            task="stream task",
            stream=True,
            env={"FAKE_CLAUDE_ARGV_FILE": str(argv_file)},
        )
        ctx = make_context(on_llm_chunk=chunks.append)

        result = await ReasoningChain(steps=[step]).execute_async(ctx)

        assert result.success
        step_result = result.step_results[0]
        # only text blocks are forwarded — the tool_use block is not
        assert chunks == ["Analyzing the task.", "All done."]
        assert step_result.result == "ECHO: stream task"
        assert step_result.result_data["session_id"] == "sess-1234"
        assert step_result.token_usage["total"] == 122
        # stream mode switches the output format and adds --verbose
        argv = json.loads(argv_file.read_text().strip())
        assert argv[argv.index("--output-format") + 1] == "stream-json"
        assert "--verbose" in argv

    async def test_streaming_timeout_kills_process(self, fake_cli):
        step = make_step(
            1, fake_cli, task="slow", stream=True, timeout=0.5, env={"FAKE_CLAUDE_MODE": "sleep"}
        )

        result = await ReasoningChain(steps=[step]).execute_async(make_context())

        step_result = result.step_results[0]
        assert not step_result.success
        assert "timed out" in step_result.error_message

    async def test_streaming_agent_error(self, fake_cli):
        step = make_step(
            1, fake_cli, task="fail", stream=True, env={"FAKE_CLAUDE_MODE": "agent_error"}
        )

        result = await ReasoningChain(steps=[step]).execute_async(make_context())

        step_result = result.step_results[0]
        assert not step_result.success
        assert "error_during_execution" in step_result.error_message


class TestClaudeCodeStepOutputs:
    async def test_output_memory_key_writes_answer_text(self, fake_cli):
        step = make_step(1, fake_cli, task="compute", output_memory_key="answer")
        ctx = make_context()

        result = await ReasoningChain(steps=[step]).execute_async(ctx)

        assert result.success
        assert ctx.memory_read("answer", namespace="claude_code") == "ECHO: compute"

    async def test_output_memory_custom_namespace(self, fake_cli):
        step = make_step(
            1, fake_cli, task="compute", output_memory_key="answer", output_namespace="results"
        )
        ctx = make_context()

        await ReasoningChain(steps=[step]).execute_async(ctx)

        assert ctx.memory_read("answer", namespace="results") == "ECHO: compute"

    async def test_output_schema_parses_and_stores_object(self, fake_cli):
        schema = {
            "type": "object",
            "properties": {"answer": {"type": "number"}},
            "required": ["answer"],
        }
        step = make_step(
            1,
            fake_cli,
            task="compute",
            output_schema=schema,
            output_memory_key="answer",
            env={"FAKE_CLAUDE_RESULT": '{"answer": 42}'},
        )
        ctx = make_context()

        result = await ReasoningChain(steps=[step]).execute_async(ctx)

        assert result.success
        step_result = result.step_results[0]
        assert step_result.result_data["structured_output"] == {"answer": 42}
        # memory receives the parsed object, not the raw text
        assert ctx.memory_read("answer", namespace="claude_code") == {"answer": 42}

    async def test_output_schema_violation_fails_step(self, fake_cli):
        schema = {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
        }
        step = make_step(
            1,
            fake_cli,
            task="compute",
            output_schema=schema,
            env={"FAKE_CLAUDE_RESULT": '{"answer": 42}'},
        )

        result = await ReasoningChain(steps=[step]).execute_async(make_context())

        step_result = result.step_results[0]
        assert not step_result.success
        assert "does not match JSON Schema" in step_result.error_message

    async def test_output_schema_without_json_answer_fails_step(self, fake_cli):
        step = make_step(
            1,
            fake_cli,
            task="compute",
            output_schema={"type": "object"},
        )

        result = await ReasoningChain(steps=[step]).execute_async(make_context())

        step_result = result.step_results[0]
        assert not step_result.success
        assert "no JSON payload" in step_result.error_message


class TestClaudeCodeStepFailures:
    async def test_agent_reported_error(self, fake_cli):
        step = make_step(1, fake_cli, task="fail please", env={"FAKE_CLAUDE_MODE": "agent_error"})

        result = await ReasoningChain(steps=[step]).execute_async(make_context())

        step_result = result.step_results[0]
        assert not step_result.success
        assert "error_during_execution" in step_result.error_message
        assert "agent failed mid-run" in step_result.error_message

    async def test_non_json_output(self, fake_cli):
        step = make_step(1, fake_cli, task="garbage", env={"FAKE_CLAUDE_MODE": "garbage"})

        result = await ReasoningChain(steps=[step]).execute_async(make_context())

        step_result = result.step_results[0]
        assert not step_result.success
        assert "did not return parseable JSON" in step_result.error_message

    async def test_nonzero_exit_code(self, fake_cli):
        step = make_step(1, fake_cli, task="crash", env={"FAKE_CLAUDE_MODE": "crash"})

        result = await ReasoningChain(steps=[step]).execute_async(make_context())

        step_result = result.step_results[0]
        assert not step_result.success
        assert "exited with code 2" in step_result.error_message
        assert "boom" in step_result.error_message

    async def test_timeout_kills_process(self, fake_cli):
        step = make_step(1, fake_cli, task="slow", timeout=0.5, env={"FAKE_CLAUDE_MODE": "sleep"})

        result = await ReasoningChain(steps=[step]).execute_async(make_context())

        step_result = result.step_results[0]
        assert not step_result.success
        assert "timed out" in step_result.error_message

    async def test_missing_cli_binary(self):
        step = ClaudeCodeStepDescription(
            number=1,
            title="No CLI",
            config=ClaudeCodeStepConfig(task="hi", cli_path="/nonexistent/claude-xyz"),
        )

        result = await ReasoningChain(steps=[step]).execute_async(make_context())

        step_result = result.step_results[0]
        assert not step_result.success
        assert "not found" in step_result.error_message

    async def test_missing_placeholder_input(self, fake_cli):
        step = make_step(1, fake_cli, task="Use {missing}", input_mapping={"other": "'x'"})

        result = await ReasoningChain(steps=[step]).execute_async(make_context())

        step_result = result.step_results[0]
        assert not step_result.success
        assert "input_mapping" in step_result.error_message


class TestCliAvailabilityCheck:
    def test_available_cli_with_version_probe(self, fake_cli):
        status = check_claude_code_cli(fake_cli)

        assert status.available
        assert status.resolved_path == fake_cli
        assert status.version == "2.1.119 (Claude Code)"
        assert status.error is None

    def test_available_cli_without_probe(self, fake_cli):
        status = check_claude_code_cli(fake_cli, probe_version=False)

        assert status.available
        assert status.resolved_path == fake_cli
        assert status.version is None

    def test_missing_cli(self):
        status = check_claude_code_cli("/nonexistent/claude-xyz")

        assert not status.available
        assert status.resolved_path is None
        assert "not found" in status.error

    def test_broken_cli_fails_version_probe(self, fake_cli, monkeypatch):
        monkeypatch.setenv("FAKE_CLAUDE_MODE", "crash")

        status = check_claude_code_cli(fake_cli)

        assert not status.available
        assert status.resolved_path == fake_cli
        assert "exited with code 2" in status.error


class TestChainPreflight:
    def test_preflight_reports_available_cli(self, fake_cli):
        chain = ReasoningChain(steps=[
            make_step(1, fake_cli, task="a"),
            make_step(2, fake_cli, dependencies=[1], task="b"),  # same CLI — deduplicated
        ])

        report = chain.preflight(make_context())

        assert report.required_claude_code_clis == [fake_cli]
        assert report.missing_claude_code_clis == []
        assert report.all_present
        assert "claude code cli" in report.format_text()

    def test_preflight_flags_missing_cli(self):
        chain = ReasoningChain(steps=[
            ClaudeCodeStepDescription(
                number=1,
                title="No CLI",
                config=ClaudeCodeStepConfig(task="hi", cli_path="/nonexistent/claude-xyz"),
            ),
        ])

        report = chain.preflight(make_context())

        assert report.required_claude_code_clis == ["/nonexistent/claude-xyz"]
        assert report.missing_claude_code_clis == ["/nonexistent/claude-xyz"]
        assert not report.all_present
        assert "claude code cli: /nonexistent/claude-xyz" in report.format_text()


class TestClaudeCodeStepValidation:
    def test_cache_is_rejected(self):
        with pytest.raises(ValueError, match="cannot be cached"):
            ClaudeCodeStepDescription(
                number=1,
                title="Cached",
                config=ClaudeCodeStepConfig(task="hi"),
                cache=StepCache(),
            )

    def test_blank_output_memory_key_rejected(self):
        with pytest.raises(ValueError, match="output_memory_key"):
            ClaudeCodeStepConfig(task="hi", output_memory_key="   ")

    def test_blank_output_namespace_rejected(self):
        with pytest.raises(ValueError, match="output_namespace"):
            ClaudeCodeStepConfig(task="hi", output_namespace="   ")

    def test_step_type_and_dump(self):
        step = ClaudeCodeStepDescription(number=1, title="T", config=ClaudeCodeStepConfig(task="hi"))
        assert step.step_type == StepType.CLAUDE_CODE
        assert step.model_dump()["step_type"] == "claude_code"
