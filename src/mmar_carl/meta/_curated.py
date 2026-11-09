"""Curated, editorial metadata for CARL step types — the small layer that
runtime introspection *cannot* infer.

Everything structural (field names, types, required/optional, defaults,
descriptions) is derived live from the pydantic models by
:mod:`mmar_carl.meta._introspect`. This module supplies only the things a
model-reflection pass can't know:

- ``purpose`` — a one-line, planner-facing summary of what the step is *for*
  (docstrings are verbose and inconsistent; these are tuned for token budget).
- ``example`` — a known-good minimal spec payload so generation/mutation can
  splice a valid step without a round-trip to an LLM.
- ``mutable_fields`` — which fields an evolutionary mutator may safely change
  without breaking the step's structure (editorial; *not* derivable).
- ``required_overrides`` — fields that a validator enforces as required even
  though pydantic marks them optional (e.g. ``LLMStepDescription.aim`` has
  ``default=""`` but a ``@model_validator`` rejects an empty value).

The catalog degrades gracefully when a step type is missing here: ``purpose``
falls back to the class docstring's first line, ``example`` to a required-only
skeleton, and ``mutable_fields`` to an empty list. A drift test
(:func:`mmar_carl.meta.catalog.check_drift`) warns when a registered step type
has no curated entry, so new types get a nudge without breaking the build.

Field-path convention used by ``mutable_fields`` and elsewhere:
``"step_config.<field>"`` targets a field inside the embedded config model;
a bare ``"<field>"`` targets an inlined/base field on the step description
(this is how ``llm`` steps and common fields like ``aim`` are addressed).
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Per-step-type curated metadata, keyed by the ``StepType`` enum *value*.
# ---------------------------------------------------------------------------

CURATED_STEP_META: dict[str, dict[str, Any]] = {
    "agent": {
        "purpose": "Use allowed tools in a bounded reasoning loop.",
        "example": {"step_config": {"goal": "Look up the answer.", "tools": ["lookup"]}},
        "mutable_fields": ["step_config.goal", "step_config.input_mapping"],
    },
    "code": {
        "purpose": "Run an exact Python source through a host-approved runtime profile.",
        "example": {"step_config": {
            "source": "$metadata.code_source", "runtime_profile": "python-safe-v1",
            "input_schema": {"type": "object"}, "output_schema": {"type": "number"},
        }},
    },
    "wait": {
        "purpose": "Wait for a duration, deadline, or named event.",
        "example": {"step_config": {"condition": {"type": "after", "seconds": 0.25}}},
    },
    "map": {
        "purpose": "Call a registered tool for each array element with bounded concurrency.",
        "example": {"step_config": {
            "tool_name": "enrich", "items_source": "$outer_context", "item_parameter": "item",
        }},
    },
    "command_plan": {
        "purpose": "Choose arguments for a host-registered command capability.",
        "example": {"step_config": {
            "instruction": "Inspect the supplied text.", "capability_ids": ["text.grep"],
        }},
    },
    "command": {
        "purpose": "Run an argv command under host execution policy.",
        "example": {"step_config": {"command": ["/usr/bin/printf", "%s", "hello"]}},
    },
    "shell_session": {
        "purpose": "Run a static shell script under host execution policy.",
        "example": {"step_config": {"commands": ["printf '%s' hello"]}},
    },
    "claude_code": {
        "purpose": "Delegate a task to the Claude Code CLI.",
        "example": {"step_config": {"task": "Summarize the input.", "allowed_tools": ["Read"]}},
    },
    "codex": {
        "purpose": "Delegate a task to a Codex SDK agent.",
        "example": {"step_config": {"task": "Review the supplied change.", "sandbox": "read-only"}},
    },
    "planner": {
        "purpose": "Select an ordered sequence of declared tool actions at runtime.",
        "example": {"step_config": {
            "instruction": "Select a lookup if needed.", "actions": {"lookup": {"tool_name": "lookup"}},
        }},
    },
    "llm": {
        "purpose": "Chain-of-thought reasoning with an LLM. The default building block.",
        "example": {"aim": "Analyse the input and produce structured findings."},
        "mutable_fields": [
            "aim",
            "reasoning_questions",
            "stage_action",
            "example_reasoning",
            "step_context_queries",
            "llm_config",
        ],
        "required_overrides": ["aim"],
    },
    "tool": {
        "purpose": "Call a registered Python function/tool with mapped inputs.",
        "example": {
            "step_config": {
                "tool_name": "web_search",
                "input_mapping": {"query": "$history[-1]"},
            }
        },
        "mutable_fields": [
            "step_config.tool_description",
            "step_config.input_mapping",
            "step_config.timeout",
            "step_config.error_recovery",
        ],
    },
    "mcp": {
        "purpose": "Invoke a tool on a Model Context Protocol server.",
        "example": {
            "step_config": {
                "server": {"server_name": "docs", "transport": "stdio", "command": "docs-server"},
                "tool_name": "search",
                "argument_mapping": {"q": "$history[-1]"},
            }
        },
        "mutable_fields": [
            "step_config.arguments",
            "step_config.argument_mapping",
            "step_config.timeout",
        ],
    },
    "mcp_resource": {
        "purpose": "Read a named read-only MCP resource into memory/history.",
        "example": {
            "step_config": {
                "server": {"server_name": "docs", "transport": "sse", "url": "http://docs/sse"},
                "resource_uri": "docs://api/reference.md",
                "output_memory_key": "api_docs",
            }
        },
        "mutable_fields": [
            "step_config.output_memory_key",
            "step_config.output_namespace",
            "step_config.timeout",
        ],
    },
    "memory": {
        "purpose": "Read / write / append / delete / list namespaced shared memory.",
        "example": {
            "step_config": {
                "operation": "write",
                "memory_key": "draft",
                "value_source": "$history[-1]",
                "namespace": "default",
            }
        },
        "mutable_fields": [
            "step_config.value_source",
            "step_config.default_value",
            "step_config.namespace",
        ],
    },
    "transform": {
        "purpose": "Deterministic data transformation (no LLM call).",
        "example": {
            "step_config": {
                "transform_type": "format",
                "input_key": "$history[-1]",
                # `{input}` is the only name bound here: the executor renders
                # this template with `output_format.format(input=input_data)`.
                # The example said `{value}`, so every `transform` step built
                # from this catalog raised KeyError the moment it ran — in
                # 0.00s, and *after* the upstream steps had already been paid
                # for. Nothing caught it because the tests using this example
                # validate and render specs without executing them; see
                # `test_curated_examples_execute`, which does execute them.
                "output_format": "Summary: {input}",
            }
        },
        "mutable_fields": [
            "step_config.transform_type",
            "step_config.output_format",
            "step_config.expression",
            "step_config.map_template",
        ],
    },
    "selection": {
        "purpose": "Select item(s) from a collection by a strategy (top-k / weighted / random / filter / all).",
        "example": {
            "step_config": {
                "source": "$memory.pool.candidates",
                "strategy": "top_k",
                "count": 1,
                "score_key": "fitness",
            }
        },
        "mutable_fields": [
            "step_config.strategy",
            "step_config.count",
            "step_config.score_key",
            "step_config.predicate",
        ],
    },
    "conditional": {
        "purpose": "Branch the DAG: route to a step whose condition matches.",
        "example": {
            "step_config": {
                "branches": [{"condition": "len($history[-1]) > 100", "next_step": 3}],
                "default_step": 4,
                "condition_context_key": "$history[-1]",
            }
        },
        "mutable_fields": [
            "step_config.branches",
            "step_config.default_step",
            "step_config.condition_context_key",
        ],
    },
    "structured_output": {
        "purpose": "LLM step whose output must match a JSON schema.",
        "example": {
            "step_config": {
                "output_schema": {"type": "object", "properties": {"sentiment": {"type": "string"}}},
                "input_source": "$history[-1]",
            }
        },
        "mutable_fields": [
            "step_config.instruction",
            "step_config.input_source",
            "step_config.strict_json",
        ],
    },
    "agent_skill": {
        "purpose": "Execute a portable AgentSkill (SKILL.md folder) in one of several modes.",
        "example": {
            "step_config": {
                "skill": "pdf",
                "task": "Extract the text from {pdf_path}.",
                "input_mapping": {"pdf_path": "$memory.input.pdf_path"},
            }
        },
        "mutable_fields": [
            "step_config.task",
            "step_config.execution_mode",
            "step_config.llm_config",
            "step_config.input_mapping",
        ],
    },
    "evaluation": {
        "purpose": "Inline quality gate: score a prior step's output; can fail/gate downstream.",
        "example": {
            "dependencies": [1],
            "step_config": {
                "evaluates_step": 1,
                "criteria": ["min_words:50"],
                "evaluation_method": "rule",
            },
        },
        "mutable_fields": [
            "step_config.criteria",
            "step_config.evaluation_method",
            "step_config.on_fail",
        ],
    },
    "agent_handoff": {
        "purpose": "Delegate to a complete sub-chain with input/output mapping.",
        "example": {
            "step_config": {
                "input_mapping": {"input.topic": "$memory.input.topic"},
                "output_memory_key": "research",
            }
        },
        "mutable_fields": [
            "step_config.input_mapping",
            "step_config.output_memory_key",
            "step_config.propagate_failure",
            "step_config.share_memory",
        ],
        # ``sub_chain`` is runtime-only (exclude=True); CARE rebinds it from a
        # gigaevo-memory entity ref before execution.
    },
    "supervisor": {
        "purpose": "LLM routes a task to one of N registered specialist sub-chains.",
        "example": {
            "step_config": {
                "routing_prompt": (
                    "Pick ONE specialist for the task. Reply with just the name.\nSpecialists: {agents}\n\nTask: {task}"
                ),
                "output_memory_key": "specialist_result",
            }
        },
        "mutable_fields": [
            "step_config.routing_prompt",
            "step_config.fallback_agent",
            "step_config.llm_config",
        ],
        # ``agents`` is runtime-only (exclude=True).
    },
    "debate": {
        "purpose": "Round-robin multi-role debate, then an LLM judge synthesis.",
        "example": {
            "step_config": {
                "roles": ["optimist", "skeptic"],
                "rounds": 2,
                "judge_prompt": "Given the debate transcript, synthesise the best answer.\n{transcript}",
            }
        },
        "mutable_fields": [
            "step_config.role_prompts",
            "step_config.rounds",
            "step_config.judge_prompt",
            "step_config.llm_config",
        ],
    },
    "parallel_sampling": {
        "purpose": "Draw N independent LLM samples and aggregate (vote / best-of-N / judge).",
        "example": {"step_config": {"n_samples": 5, "aggregation": "majority_vote"}},
        "mutable_fields": [
            "step_config.n_samples",
            "step_config.aggregation",
            "step_config.judge_prompt",
        ],
        # ``base_step`` is runtime-only (exclude=True).
    },
    "tool_discovery": {
        "purpose": "Discover and register tools at runtime from a module/callable/dict source.",
        "example": {"step_config": {"source": {"module": "my_pkg.tools", "name_prefix": "tool_"}}},
        "mutable_fields": [
            "step_config.output_memory_key",
            "step_config.tool_timeout",
        ],
    },
    "human_input": {
        "purpose": "Pause execution and wait for human input (callable or webhook).",
        "example": {"step_config": {"prompt": "Approve the draft? (yes/no)", "output_memory_key": "approval"}},
        "mutable_fields": [
            "step_config.prompt",
            "step_config.fallback_value",
            "step_config.timeout",
        ],
    },
}


# ---------------------------------------------------------------------------
# Dynamic-reference grammar — the ``$...`` syntax CARL resolves at runtime.
# Mirrors resolve_context_reference() in step_executors.py. Exposed as data so
# external generators/mutators can learn the grammar without reading source.
# ---------------------------------------------------------------------------

REFERENCE_GRAMMAR: list[dict[str, str]] = [
    {
        "prefix": "$history",
        "syntax": "$history[-1] | $history[0] | $history",
        "description": "Output of a previous step by index (negative allowed); bare form is the whole list.",
        "returns": "str | list[str]",
    },
    {
        "prefix": "$memory",
        "syntax": "$memory.<namespace>.<key> | $memory.<key>",
        "description": "Read namespaced shared memory. Namespace defaults to 'default' when omitted.",
        "returns": "Any",
    },
    {
        "prefix": "$ltm",
        "syntax": "$ltm.<key>",
        "description": "Read long-term memory (requires an LTMBase backend wired into the context).",
        "returns": "Any",
    },
    {
        "prefix": "$event",
        "syntax": "$event.<name>",
        "description": "Most-recent payload emitted for an event via context.emit_event(name, payload).",
        "returns": "Any",
    },
    {
        "prefix": "$metadata",
        "syntax": "$metadata.<path>.<to>.<key>",
        "description": "Nested dotted-path traversal of context.metadata.",
        "returns": "Any",
    },
    {
        "prefix": "$steps",
        "syntax": "$steps.<N>.result",
        "description": (
            "Structured per-step metadata stored under context.metadata['step_results'][N]. "
            "Declare step N as a direct or transitive dependency of every consumer."
        ),
        "returns": "Any",
    },
    {
        "prefix": "$outer_context",
        "syntax": "$outer_context",
        "description": "The chain's original input. Parsed as JSON when it looks like JSON, else the raw string.",
        "returns": "str | dict | list",
    },
    {
        "prefix": "literal",
        "syntax": "\"some text\" | 'some text'",
        "description": "A quoted string literal, passed through verbatim (outer quotes stripped).",
        "returns": "str",
    },
    {
        "prefix": "bare",
        "syntax": "<key>",
        "description": "Fallback: an unprefixed key is looked up in context.metadata.",
        "returns": "Any",
    },
]

# Config fields that accept ``$...`` reference strings, keyed by step type value.
# Drives both validate_spec reference scanning and the generation prompt.
REFERENCE_FIELDS: dict[str, list[str]] = {
    "agent": ["step_config.input_mapping.*"],
    "code": ["step_config.source", "step_config.input_mapping.*"],
    "map": ["step_config.items_source", "step_config.input_mapping.*"],
    "command_plan": ["step_config.input_mapping.*"],
    "command": [
        "step_config.input_mapping.*", "step_config.plan_source", "step_config.stdin_source",
        "step_config.artifact_inputs.*.source",
    ],
    "shell_session": ["step_config.input_mapping.*", "step_config.artifact_inputs.*.source"],
    "human_input": ["step_config.prompt_input_mapping.*"],
    "claude_code": ["step_config.input_mapping.*", "step_config.resume_session"],
    "codex": ["step_config.input_mapping.*", "step_config.resume_session"],
    "tool": ["step_config.input_mapping.*"],
    "agent_skill": ["step_config.input_mapping.*"],
    "mcp": ["step_config.argument_mapping.*"],
    "memory": ["step_config.value_source"],
    "transform": ["step_config.input_key"],
    "selection": ["step_config.source"],
    "conditional": ["step_config.condition_context_key", "step_config.branches.*.condition"],
    "structured_output": ["step_config.input_source", "step_config.input_mapping.*"],
    "agent_handoff": ["step_config.input_mapping.*"],
    "supervisor": ["step_config.task_source", "step_config.input_mapping.*"],
    "debate": ["step_config.task_source"],
    "evaluation": ["step_config.input_source"],
}


# ---------------------------------------------------------------------------
# Validation rules — what validate_spec() / the chain validator enforce.
# Structured so a TUI/CLI can render them and a planner can be reminded of them.
# ---------------------------------------------------------------------------

VALIDATION_RULES: list[dict[str, str]] = [
    {"code": "non_empty", "severity": "error", "rule": "A chain must contain at least one step."},
    {"code": "unique_numbers", "severity": "error", "rule": "Each step.number must be unique within the chain."},
    {
        "code": "deps_exist",
        "severity": "error",
        "rule": "Every value in step.dependencies must reference an existing step.number.",
    },
    {"code": "acyclic", "severity": "error", "rule": "The dependency graph must be acyclic (no cycles)."},
    {
        "code": "step_result_dependencies",
        "severity": "error",
        "rule": "A catalogued $steps.N read must name an existing direct or transitive dependency.",
    },
    {
        "code": "known_step_type",
        "severity": "error",
        "rule": "step_type must be one of the registered StepType values.",
    },
    {
        "code": "required_fields",
        "severity": "error",
        "rule": "Each step must provide the required fields for its step_type (see the catalog card).",
    },
    {
        "code": "memory_writer",
        "severity": "warning",
        "rule": "A $memory.ns.key read should be written by a prior step or pre-populated before execution.",
    },
    {
        "code": "history_depth",
        "severity": "warning",
        "rule": "A $history[N] read with N >= the history depth at that point resolves to None.",
    },
]


__all__ = [
    "CURATED_STEP_META",
    "REFERENCE_GRAMMAR",
    "REFERENCE_FIELDS",
    "VALIDATION_RULES",
]
