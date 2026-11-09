"""Tool steps expose JSON text results as structured result_data."""

from __future__ import annotations

import asyncio
import json

from mmar_carl import ReasoningChain, ReasoningContext
from mmar_carl.step_executors import _mcp_result_data, _structured_tool_result


class _NoLLM:
    async def get_response(self, prompt: str, *args, **kwargs) -> str:  # pragma: no cover - never called
        raise AssertionError("no LLM step in this chain")


def _two_tool_chain() -> dict:
    return {
        "max_workers": 1,
        "steps": [
            {
                "number": 1,
                "title": "lookup",
                "step_type": "tool",
                "dependencies": [],
                "step_config": {
                    "tool_name": "lookup",
                    "parameters": [{"name": "key", "type": "string"}],
                    "input_mapping": {"key": '"S4"'},
                },
            },
            {
                "number": 2,
                "title": "timezone",
                "step_type": "tool",
                "dependencies": [1],
                "step_config": {
                    "tool_name": "timezone",
                    "parameters": [{"name": "lat", "type": "float"}, {"name": "lon", "type": "float"}],
                    "input_mapping": {"lat": "$steps.1.result_data.latitude", "lon": "$steps.1.result_data.longitude"},
                },
            },
        ],
    }


def test_json_text_tool_result_feeds_next_tool() -> None:
    seen: dict = {}

    def lookup(key: str) -> str:
        return json.dumps({"latitude": 60.0, "longitude": 30.0})

    def timezone(lat: float, lon: float) -> str:
        seen["args"] = (lat, lon)
        return "Europe/Helsinki"

    chain = ReasoningChain.from_dict_typed(_two_tool_chain())
    context = ReasoningContext(outer_context="", api=_NoLLM())
    context.register_tool("lookup", lookup)
    context.register_tool("timezone", timezone)
    result = asyncio.run(chain.execute_async(context))
    assert result.success, [r.error for r in result.step_results]
    assert seen["args"] == (60.0, 30.0)
    first = result.step_results[0]
    assert first.result_data == {"latitude": 60.0, "longitude": 30.0}
    assert first.result == '{"latitude": 60.0, "longitude": 30.0}'
    assert result.step_results[1].result_data == "Europe/Helsinki"


def test_structured_tool_result_only_decodes_whole_json_containers() -> None:
    assert _structured_tool_result('{"a": 1}') == {"a": 1}
    assert _structured_tool_result("  [1, 2] ") == [1, 2]
    assert _structured_tool_result("plain text") == "plain text"
    assert _structured_tool_result('"quoted"') == '"quoted"'
    assert _structured_tool_result("42") == "42"
    assert _structured_tool_result("{not json") == "{not json"
    payload = {"k": "v"}
    assert _structured_tool_result(payload) is payload


def test_mcp_result_data_decodes_json_text_and_keeps_blocks_otherwise() -> None:
    blocks = [{"type": "text", "text": '{"temp": 4}'}]
    assert _mcp_result_data(blocks, '{"temp": 4}') == {"temp": 4}
    plain = [{"type": "text", "text": "sunny"}]
    assert _mcp_result_data(plain, "sunny") is plain
    structured = {"temp": 4}
    assert _mcp_result_data(structured, '{"temp": 4}') is structured
