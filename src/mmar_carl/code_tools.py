"""Code tools — tool implementations as *data*, so they can be evolved.

A :class:`ToolStepDescription` normally references a tool by name; the callable
is registered on the :class:`~mmar_carl.models.context.ReasoningContext` at
runtime and is invisible to spec-level tooling (serialization, validation,
mutation). That makes tool *behaviour* a fixed point: an evolution engine can
rewire or retype the steps around a tool, but never change what the tool does.

This module makes tool implementations part of the spec substrate:

- :class:`CodeTool` — one versioned tool implementation: a ``name`` plus the
  Python ``code`` (source text) that defines a function of that name.
- :class:`CodeToolLibrary` — a named, versioned collection of :class:`CodeTool`s.
  Serializable; travels inside a chain spec under
  ``spec["metadata"]["tool_library"]`` (:data:`TOOL_LIBRARY_KEY`), so the chain
  *and its tools* form one genome that survives ``to_dict → from_dict`` — and
  therefore mutation, persistence, and replay.
- :func:`validate_code_tool` — no-execution validation (parses, defines the
  right function, no imports / dunder access), used by ``validate_spec``.
- :func:`compile_code_tool` — build the callable by executing the source in a
  **restricted namespace** (:data:`SAFE_BUILTINS` whitelist + ``math``).

Execution is **opt-in**: :meth:`ReasoningContext.bind_tool_library` refuses to
register code tools unless the context was created with
``allow_code_tools=True``. Binding never overwrites tools that are already
registered on the context (host tools win), so an evolved spec can't hijack a
host primitive by shadowing its name.

Version pinning: binding registers each tool's *latest* version under its
plain ``name``, and every version under ``"{name}@v{version}"`` — a step whose
``tool_name`` is ``"scale@v2"`` stays pinned while ``"scale"`` tracks evolution.
"""

from __future__ import annotations

import ast
import math
from typing import Any, Callable, Optional

from pydantic import BaseModel, Field

# Where a chain spec carries its tool library (``spec["metadata"][TOOL_LIBRARY_KEY]``).
TOOL_LIBRARY_KEY = "tool_library"

# Builtins available to code tools. Deliberately excludes import machinery,
# I/O, reflection (getattr/setattr/vars/globals) and eval/exec/compile.
SAFE_BUILTINS: dict[str, Any] = {
    name: __builtins__[name] if isinstance(__builtins__, dict) else getattr(__builtins__, name)
    for name in (
        "abs", "all", "any", "bool", "dict", "divmod", "enumerate", "filter",
        "float", "frozenset", "int", "isinstance", "issubclass", "iter", "len",
        "list", "map", "max", "min", "next", "pow", "range", "repr", "reversed",
        "round", "set", "sorted", "str", "sum", "tuple", "zip",
        "ArithmeticError", "AttributeError", "Exception", "IndexError",
        "KeyError", "StopIteration", "TypeError", "ValueError",
        "ZeroDivisionError",
    )
}


class CodeToolError(ValueError):
    """A code tool failed validation or compilation."""


class CodeTool(BaseModel):
    """One versioned tool implementation, as data.

    ``code`` is Python source that must define ``def {name}(...)`` at module
    level. ``version`` orders implementations of the same tool;
    ``parent_version`` records lineage across mutations.
    """

    name: str = Field(..., min_length=1, description="Tool name; the code must define a function of this name.")
    code: str = Field(..., min_length=1, description="Python source defining `def {name}(...)` at top level.")
    description: str = Field(default="", description="What the tool does (surfaced to mutation operators).")
    version: int = Field(default=1, ge=1, description="Implementation version (1-based, monotonic per name).")
    parent_version: Optional[int] = Field(default=None, description="Version this one was mutated from.")
    tags: list[str] = Field(default_factory=list, description="Registry tags (see register_tool).")


# Names whose *use* in code-tool source is rejected outright (mirrors
# GigaEvo's DANGEROUS_PATTERNS screen, but AST-based instead of regex so
# comments/strings don't false-positive). SAFE_BUILTINS already omits them,
# so this is defence in depth: fail at validation, not at call time.
BANNED_NAMES: frozenset = frozenset(
    {
        "eval", "exec", "compile", "open", "input", "breakpoint",
        "__import__", "getattr", "setattr", "delattr", "globals", "locals", "vars",
    }
)

# Upper bound on one tool's source (same idea as GigaEvo's max_code_length —
# keeps an LLM mutation from ballooning the genome).
MAX_CODE_LENGTH = 10_000


def validate_code_tool(tool: CodeTool) -> list[str]:
    """No-execution validation of one :class:`CodeTool` → list of error strings.

    Checks: length bound; the source parses; it defines a top-level
    ``def {name}``; no ``import`` statements, no dunder attribute access, and no
    use of :data:`BANNED_NAMES` (the cheap escape hatches out of the restricted
    namespace). Complements — not replaces — the runtime restriction of
    :data:`SAFE_BUILTINS`.
    """
    if len(tool.code) > MAX_CODE_LENGTH:
        return [f"code too long: {len(tool.code)} > {MAX_CODE_LENGTH}"]
    errors: list[str] = []
    try:
        tree = ast.parse(tool.code)
    except SyntaxError as exc:
        return [f"code does not parse: {exc.msg} (line {exc.lineno})"]
    top_defs = [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if tool.name not in top_defs:
        errors.append(f"code must define a top-level function named {tool.name!r} (found: {top_defs or 'none'})")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            errors.append(f"import statements are not allowed in code tools (line {node.lineno})")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            errors.append(f"dunder attribute access is not allowed in code tools (line {node.lineno})")
        elif isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            errors.append(f"use of {node.id!r} is not allowed in code tools (line {node.lineno})")
    return errors


def compile_code_tool(tool: CodeTool) -> Callable[..., Any]:
    """Build the callable for ``tool`` by exec-ing its source in a restricted
    namespace (:data:`SAFE_BUILTINS` + ``math``). Raises :class:`CodeToolError`
    on validation failure or if the function doesn't materialise."""
    errors = validate_code_tool(tool)
    if errors:
        raise CodeToolError(f"code tool {tool.name!r} v{tool.version}: " + "; ".join(errors))
    namespace: dict[str, Any] = {"__builtins__": dict(SAFE_BUILTINS), "math": math}
    try:
        exec(compile(tool.code, f"<code_tool:{tool.name}@v{tool.version}>", "exec"), namespace)  # noqa: S102
    except Exception as exc:
        raise CodeToolError(f"code tool {tool.name!r} v{tool.version} failed to load: {exc}") from exc
    fn = namespace.get(tool.name)
    if not callable(fn):
        raise CodeToolError(f"code tool {tool.name!r} v{tool.version} did not define a callable {tool.name!r}")
    return fn


class CodeToolLibrary(BaseModel):
    """A named, versioned collection of :class:`CodeTool`s.

    ``tools`` maps tool name → versions in ascending version order. The library
    serializes to/from a plain dict so it can ride inside a chain spec's
    ``metadata`` (:data:`TOOL_LIBRARY_KEY`) and be mutated as spec data.
    """

    tools: dict[str, list[CodeTool]] = Field(default_factory=dict)

    # -- content -----------------------------------------------------------
    def names(self) -> list[str]:
        return sorted(self.tools)

    def versions(self, name: str) -> list[CodeTool]:
        return list(self.tools.get(name, []))

    def get(self, name: str, version: Optional[int] = None) -> Optional[CodeTool]:
        """The requested version of ``name`` (latest when ``version`` is None)."""
        versions = self.tools.get(name)
        if not versions:
            return None
        if version is None:
            return versions[-1]
        return next((t for t in versions if t.version == version), None)

    def add(
        self,
        name: str,
        code: str,
        *,
        description: str = "",
        tags: Optional[list[str]] = None,
        parent_version: Optional[int] = None,
        validate: bool = True,
    ) -> CodeTool:
        """Append a new version of ``name`` (creating the tool on first add).

        The new version number is ``latest + 1``; ``parent_version`` defaults to
        the current latest. With ``validate=True`` (default) the code must pass
        :func:`validate_code_tool`.
        """
        latest = self.get(name)
        tool = CodeTool(
            name=name,
            code=code,
            description=description or (latest.description if latest else ""),
            version=(latest.version + 1) if latest else 1,
            parent_version=parent_version if parent_version is not None else (latest.version if latest else None),
            tags=list(tags) if tags is not None else (list(latest.tags) if latest else []),
        )
        if validate:
            errors = validate_code_tool(tool)
            if errors:
                raise CodeToolError(f"code tool {name!r}: " + "; ".join(errors))
        self.tools.setdefault(name, []).append(tool)
        return tool

    def merge(self, other: "CodeToolLibrary") -> "CodeToolLibrary":
        """A new library with this library's tools plus ``other``'s tools for
        names this one doesn't have (self wins on name conflicts) — the
        crossover-friendly union."""
        merged = CodeToolLibrary(tools={k: [t.model_copy(deep=True) for t in v] for k, v in self.tools.items()})
        for name, versions in other.tools.items():
            if name not in merged.tools:
                merged.tools[name] = [t.model_copy(deep=True) for t in versions]
        return merged

    def validate_all(self) -> list[str]:
        """Validation errors across every tool version (empty = valid)."""
        errors: list[str] = []
        for name, versions in self.tools.items():
            for tool in versions:
                for err in validate_code_tool(tool):
                    errors.append(f"{name}@v{tool.version}: {err}")
        return errors

    # -- (de)serialization --------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {name: [t.model_dump(mode="json") for t in versions] for name, versions in self.tools.items()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CodeToolLibrary":
        tools = {
            name: sorted((CodeTool.model_validate(t) for t in versions), key=lambda t: t.version)
            for name, versions in (data or {}).items()
        }
        return cls(tools=tools)

    # -- spec attachment ----------------------------------------------------
    @classmethod
    def from_spec(cls, spec: dict[str, Any]) -> Optional["CodeToolLibrary"]:
        """The library embedded in a chain spec dict, or None."""
        data = (spec.get("metadata") or {}).get(TOOL_LIBRARY_KEY)
        return cls.from_dict(data) if data else None

    def attach_to(self, spec: dict[str, Any]) -> dict[str, Any]:
        """Write this library into ``spec["metadata"]`` (in place; returns spec)."""
        spec.setdefault("metadata", {})[TOOL_LIBRARY_KEY] = self.to_dict()
        return spec


__all__ = [
    "TOOL_LIBRARY_KEY",
    "SAFE_BUILTINS",
    "BANNED_NAMES",
    "MAX_CODE_LENGTH",
    "CodeTool",
    "CodeToolError",
    "CodeToolLibrary",
    "compile_code_tool",
    "validate_code_tool",
]
