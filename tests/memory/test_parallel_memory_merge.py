"""
Tests for the parallel batch memory merge — dirty-key tracking in COWMemoryStore.

These pin the semantics of what a parallel step contributes to parent memory
when the batch finishes:

- A step that only *reads* a namespace contributes nothing.  It can no longer
  revert a sibling's write to that namespace (the winner used to depend purely
  on step declaration order).
- Two steps writing different keys of one namespace both survive, whether or
  not those keys existed before the batch.
- Two steps writing the *same* key resolve last-write-wins in step declaration
  order: the highest-numbered step in the batch merges last and wins.
- A key deleted by one step stays deleted; a sibling's untouched snapshot of
  that namespace does not resurrect it.
- Nested mutable values (lists/dicts) are isolated per step, so an in-place
  mutation reaches the parent only through the merge — never during execution,
  and never at all if the step fails.
- Namespaces a step never touches are not copied at all (zero-copy fast path).
- Values merged back are validated against ``ReasoningContext.memory_schema``,
  which the raw merge used to bypass.
"""

from typing import Any, Optional

import pytest

from mmar_carl import Language, LLMClientBase, ReasoningChain, ReasoningContext
from mmar_carl.cow_memory import COWMemoryStore
from mmar_carl.models.config import MemoryStepConfig
from mmar_carl.models.enums import MemoryOperation, StepType
from mmar_carl.models.results import StepExecutionResult
from mmar_carl.models.steps import LLMStepDescription, MemoryStepDescription
from mmar_carl.step_executors import (
    StepExecutorBase,
    get_executor,
    register_executor,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _MockLLMClient(LLMClientBase):
    async def get_response(self, prompt: str) -> str:
        return "llm ok"

    async def get_response_with_retries(self, prompt: str, retries: int = 3) -> str:
        return await self.get_response(prompt)


def _context(memory: Optional[dict] = None, **kwargs) -> ReasoningContext:
    ctx = ReasoningContext(
        outer_context="test",
        api=_MockLLMClient(),
        model="test",
        language=Language.ENGLISH,
        **kwargs,
    )
    for namespace, pairs in (memory or {}).items():
        for key, value in pairs.items():
            ctx.memory_write(key, value, namespace=namespace)
    return ctx


def _write(number: int, key: str, value: str, namespace: str = "ns") -> MemoryStepDescription:
    return MemoryStepDescription(
        number=number,
        title=f"write {key}",
        config=MemoryStepConfig(
            operation=MemoryOperation.WRITE,
            memory_key=key,
            value_source=f"'{value}'",
            namespace=namespace,
        ),
    )


def _read(number: int, key: str, namespace: str = "ns") -> MemoryStepDescription:
    return MemoryStepDescription(
        number=number,
        title=f"read {key}",
        config=MemoryStepConfig(
            operation=MemoryOperation.READ,
            memory_key=key,
            namespace=namespace,
        ),
    )


def _append(number: int, key: str, value: str, namespace: str = "ns") -> MemoryStepDescription:
    return MemoryStepDescription(
        number=number,
        title=f"append {key}",
        config=MemoryStepConfig(
            operation=MemoryOperation.APPEND,
            memory_key=key,
            value_source=f"'{value}'",
            namespace=namespace,
        ),
    )


def _delete(number: int, key: str, namespace: str = "ns") -> MemoryStepDescription:
    return MemoryStepDescription(
        number=number,
        title=f"delete {key}",
        config=MemoryStepConfig(
            operation=MemoryOperation.DELETE,
            memory_key=key,
            namespace=namespace,
        ),
    )


async def _run(steps, ctx: ReasoningContext, max_workers: Any = 2):
    chain = ReasoningChain(steps=steps, max_workers=max_workers)
    return await chain.execute_async(ctx)


# ---------------------------------------------------------------------------
# 1. A read-only sibling must not revert a write
# ---------------------------------------------------------------------------


class TestReadOnlyStepDoesNotRevertWrite:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("writer_first", [True, False], ids=["writer-first", "reader-first"])
    async def test_reader_does_not_clobber_writer(self, writer_first: bool):
        """A READ step and a WRITE step on the same key run in one batch.

        The read must contribute nothing to the merge, so the write survives
        regardless of which step is declared first.  Before dirty-key tracking
        the reader's stale whole-namespace snapshot won whenever it was
        declared second.
        """
        ctx = _context(memory={"ns": {"k": "OLD"}})
        steps = [_write(1, "k", "NEW"), _read(2, "k")] if writer_first else [_read(1, "k"), _write(2, "k", "NEW")]

        result = await _run(steps, ctx)

        assert result.success
        assert ctx.memory["ns"]["k"] == "NEW"

    @pytest.mark.asyncio
    async def test_reader_of_other_key_does_not_clobber_write(self):
        """Reading key ``a`` must not revert a sibling's write to key ``b``."""
        ctx = _context(memory={"ns": {"a": "A", "b": "OLD"}})

        result = await _run([_read(1, "a"), _write(2, "b", "NEW")], ctx)

        assert result.success
        assert ctx.memory["ns"] == {"a": "A", "b": "NEW"}


# ---------------------------------------------------------------------------
# 2 & 3. Concurrent writes
# ---------------------------------------------------------------------------


class TestConcurrentWrites:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "seed",
        [
            {},
            {"ns": {"a": "A0"}},
            {"ns": {"a": "A0", "b": "B0"}},
        ],
        ids=["namespace-absent", "one-key-exists", "both-keys-exist"],
    )
    async def test_two_writers_different_keys_both_survive(self, seed: dict):
        """Two parallel writers to different keys of one namespace both persist.

        This is the whole trigger matrix of the lost-update bug: it used to be
        safe only while the target key did not already exist.
        """
        ctx = _context(memory=seed)

        result = await _run([_write(1, "a", "A1"), _write(2, "b", "B1")], ctx)

        assert result.success
        assert ctx.memory["ns"]["a"] == "A1"
        assert ctx.memory["ns"]["b"] == "B1"

    @pytest.mark.asyncio
    async def test_writer_does_not_erase_untouched_pre_existing_key(self):
        """Keys nobody in the batch touches keep their pre-batch value."""
        ctx = _context(memory={"ns": {"keep": "KEEP", "a": "A0"}})

        result = await _run([_write(1, "a", "A1"), _read(2, "keep")], ctx)

        assert result.success
        assert ctx.memory["ns"] == {"keep": "KEEP", "a": "A1"}

    @pytest.mark.asyncio
    @pytest.mark.parametrize("max_workers", [1, 2, 8, "auto"])
    async def test_two_writers_same_key_last_declared_step_wins(self, max_workers):
        """PINNED SEMANTICS: same-key conflicts resolve last-write-wins, where
        "last" is the highest-numbered step of the batch.

        Merge order follows step declaration order, not completion order, so
        this is deterministic and independent of ``max_workers``.
        """
        ctx = _context(memory={"ns": {"k": "OLD"}})

        result = await _run(
            [_write(1, "k", "S1"), _write(2, "k", "S2"), _write(3, "k", "S3")],
            ctx,
            max_workers=max_workers,
        )

        assert result.success
        assert ctx.memory["ns"]["k"] == "S3"


# ---------------------------------------------------------------------------
# 4. Deletes
# ---------------------------------------------------------------------------


class TestConcurrentDeletes:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("delete_first", [True, False], ids=["delete-first", "write-first"])
    async def test_delete_sticks_while_sibling_writes_another_key(self, delete_first: bool):
        """A deleted key must not be resurrected by a sibling's snapshot.

        The sibling still holds the pre-batch copy of the namespace (including
        the deleted key); only its own written key may be merged.
        """
        ctx = _context(memory={"ns": {"doomed": "X", "other": "O0"}})
        steps = (
            [_delete(1, "doomed"), _write(2, "other", "O1")]
            if delete_first
            else [_write(1, "other", "O1"), _delete(2, "doomed")]
        )

        result = await _run(steps, ctx)

        assert result.success
        assert "doomed" not in ctx.memory["ns"]
        assert ctx.memory["ns"]["other"] == "O1"

    @pytest.mark.asyncio
    async def test_delete_is_not_undone_by_read_only_sibling(self):
        ctx = _context(memory={"ns": {"doomed": "X", "other": "O"}})

        result = await _run([_delete(1, "doomed"), _read(2, "other")], ctx)

        assert result.success
        assert ctx.memory["ns"] == {"other": "O"}

    @pytest.mark.asyncio
    async def test_delete_vs_write_same_key_last_declared_wins(self):
        """PINNED SEMANTICS: a delete racing a write on one key follows the
        same last-declared-step-wins rule as two writes."""
        ctx = _context(memory={"ns": {"k": "OLD"}})
        result = await _run([_delete(1, "k"), _write(2, "k", "NEW")], ctx)
        assert result.success
        assert ctx.memory["ns"]["k"] == "NEW"

        ctx = _context(memory={"ns": {"k": "OLD"}})
        result = await _run([_write(1, "k", "NEW"), _delete(2, "k")], ctx)
        assert result.success
        assert "k" not in ctx.memory["ns"]

    @pytest.mark.asyncio
    async def test_failed_step_delete_is_not_applied(self):
        """A step that fails must not take its deletions with it."""
        ctx = _context(memory={"ns": {"doomed": "X"}})

        class _DeleteThenFail(StepExecutorBase):
            async def execute(self, step, context, prompt_template=None):
                context.memory_delete("doomed", namespace="ns")
                raise RuntimeError("boom")

        original = get_executor(StepType.LLM)
        register_executor(StepType.LLM, _DeleteThenFail())
        try:
            # max_workers > 1 routes the batch through the gather path, which
            # turns a raising step into a failed result rather than propagating.
            result = await _run(
                [LLMStepDescription(number=1, title="fail", aim="x", prompt_template="x")],
                ctx,
                max_workers=2,
            )
        finally:
            register_executor(StepType.LLM, original)

        assert not result.success
        assert ctx.memory["ns"]["doomed"] == "X"


# ---------------------------------------------------------------------------
# 5. Nested mutable values
# ---------------------------------------------------------------------------


class TestNestedValueIsolation:
    def test_nested_list_read_is_isolated_from_base(self):
        """Reading a list out of the store must not hand out the parent's object."""
        base = {"ns": {"mylist": [1, 2]}}
        cow = COWMemoryStore(base=base)

        cow["ns"]["mylist"].append(3)

        assert base["ns"]["mylist"] == [1, 2], "parent list mutated in place"
        assert cow["ns"]["mylist"] is not base["ns"]["mylist"]
        assert cow["ns"]["mylist"] == [1, 2, 3]

    def test_nested_dict_read_is_isolated_from_base(self):
        base = {"ns": {"cfg": {"a": 1}}}
        cow = COWMemoryStore(base=base)

        cow["ns"]["cfg"]["b"] = 2

        assert base["ns"]["cfg"] == {"a": 1}
        assert cow["ns"]["cfg"] == {"a": 1, "b": 2}

    def test_deeply_nested_value_is_isolated(self):
        base = {"ns": {"tree": {"branch": [{"leaf": 1}]}}}
        cow = COWMemoryStore(base=base)

        cow["ns"]["tree"]["branch"][0]["leaf"] = 99

        assert base["ns"]["tree"]["branch"][0]["leaf"] == 1

    def test_in_place_mutation_is_reported_as_a_pending_write(self):
        """An in-place mutation must still reach the parent through the merge."""
        base = {"ns": {"mylist": [1, 2]}}
        cow = COWMemoryStore(base=base)

        cow["ns"]["mylist"].append(3)

        assert cow.pending_writes == {"ns": {"mylist": [1, 2, 3]}}

    def test_reading_without_mutating_is_not_a_pending_write(self):
        base = {"ns": {"mylist": [1, 2], "cfg": {"a": 1}}}
        cow = COWMemoryStore(base=base)

        assert cow["ns"]["mylist"] == [1, 2]
        assert cow["ns"]["cfg"] == {"a": 1}
        assert cow.pending_writes == {}

    @pytest.mark.asyncio
    async def test_failed_step_append_does_not_reach_parent(self):
        """Proof that the mutation is isolated rather than applied in place:
        a step that appends and then fails leaves the parent list untouched."""
        ctx = _context(memory={"ns": {"items": ["base"]}})

        class _AppendThenFail(StepExecutorBase):
            async def execute(self, step, context, prompt_template=None):
                context.memory_append("items", "leaked", namespace="ns")
                raise RuntimeError("boom")

        original = get_executor(StepType.LLM)
        register_executor(StepType.LLM, _AppendThenFail())
        try:
            # max_workers > 1 routes the batch through the gather path, which
            # turns a raising step into a failed result rather than propagating.
            result = await _run(
                [LLMStepDescription(number=1, title="fail", aim="x", prompt_template="x")],
                ctx,
                max_workers=2,
            )
        finally:
            register_executor(StepType.LLM, original)

        assert not result.success
        assert ctx.memory["ns"]["items"] == ["base"]

    @pytest.mark.asyncio
    async def test_append_survives_alongside_read_only_sibling(self):
        ctx = _context(memory={"ns": {"items": ["base"], "other": "O"}})

        result = await _run([_append(1, "items", "A"), _read(2, "other")], ctx)

        assert result.success
        assert ctx.memory["ns"]["items"] == ["base", "A"]

    @pytest.mark.asyncio
    async def test_parallel_appends_to_different_keys_both_survive(self):
        ctx = _context(memory={"ns": {"left": ["base"], "right": ["base"]}})

        result = await _run([_append(1, "left", "A"), _append(2, "right", "B")], ctx)

        assert result.success
        assert ctx.memory["ns"]["left"] == ["base", "A"]
        assert ctx.memory["ns"]["right"] == ["base", "B"]

    @pytest.mark.asyncio
    async def test_parallel_appends_to_same_key_are_last_write_wins(self):
        """PINNED SEMANTICS: appends are writes of the whole list, not a union.

        Both steps observe the pre-batch list, so the highest-declared step's
        version wins.  Union semantics would require an operation log; use
        separate keys (or separate batches) when both appends must survive.
        """
        ctx = _context(memory={"ns": {"items": ["base"]}})

        result = await _run([_append(1, "items", "A"), _append(2, "items", "B")], ctx)

        assert result.success
        assert ctx.memory["ns"]["items"] == ["base", "B"]

    @pytest.mark.asyncio
    async def test_sequential_appends_across_batches_accumulate(self):
        """Appends in different batches see each other and both survive."""
        ctx = _context(memory={"ns": {"items": ["base"]}})

        step_two = _append(2, "items", "B")
        step_two.dependencies = [1]
        result = await _run([_append(1, "items", "A"), step_two], ctx)

        assert result.success
        assert ctx.memory["ns"]["items"] == ["base", "A", "B"]


# ---------------------------------------------------------------------------
# 6. Zero-copy regression
# ---------------------------------------------------------------------------


class TestZeroCopyPreserved:
    def test_untouched_namespace_is_never_materialised(self):
        base = {"touched": {"k": 1}, "untouched": {"k": 2}}
        cow = COWMemoryStore(base=base)

        cow["touched"]["k"] = 99

        # "untouched" is still served straight from the base, uncopied.
        assert not dict.__contains__(cow, "untouched")
        assert cow.pending_writes == {"touched": {"k": 99}}
        assert set(cow.keys()) == {"touched", "untouched"}

    def test_untouched_namespace_values_are_not_copied(self):
        payload = ["big"] * 100
        base = {"untouched": {"payload": payload}}
        cow = COWMemoryStore(base=base)

        assert "untouched" in cow  # membership must not materialise it
        assert not dict.__contains__(cow, "untouched")
        assert base["untouched"]["payload"] is payload

    def test_unread_value_in_a_touched_namespace_is_not_copied(self):
        """Materialising a namespace copies the mapping, not every value."""
        payload = {"deep": ["data"]}
        base = {"ns": {"payload": payload, "flag": "x"}}
        cow = COWMemoryStore(base=base)

        cow["ns"]["flag"] = "y"  # touches the namespace, not `payload`

        assert dict.__getitem__(dict.__getitem__(cow, "ns"), "payload") is payload

    @pytest.mark.asyncio
    async def test_parent_namespace_objects_survive_an_unrelated_write(self):
        ctx = _context(memory={"alpha": {"k": "OLD"}, "beta": {"payload": ["data"]}})
        beta_dict = ctx.memory["beta"]
        beta_payload = beta_dict["payload"]

        result = await _run([_write(1, "k", "NEW", namespace="alpha")], ctx, max_workers=1)

        assert result.success
        assert ctx.memory["alpha"]["k"] == "NEW"
        assert ctx.memory["beta"] is beta_dict
        assert ctx.memory["beta"]["payload"] is beta_payload


# ---------------------------------------------------------------------------
# Schema validation on merge
# ---------------------------------------------------------------------------


class _RawMemoryWriter(StepExecutorBase):
    """Writes straight into ``context.memory``, bypassing ``memory_write``.

    This is what a custom step executor (the documented extension point) does
    when it does not use the ``ReasoningContext`` memory helpers — the one path
    into parent memory that the batch merge, not ``memory_write``, guards.
    """

    def __init__(self, namespace: str, key: str, value: Any) -> None:
        self._namespace = namespace
        self._key = key
        self._value = value

    async def execute(self, step, context, prompt_template=None) -> StepExecutionResult:
        if self._namespace not in context.memory:
            context.memory[self._namespace] = {}
        context.memory[self._namespace][self._key] = self._value
        return StepExecutionResult(
            step_number=step.number,
            step_title=step.title,
            step_type=StepType.LLM,
            result="raw write",
            success=True,
            updated_history=list(context.history),
        )


async def _run_raw_writer(ctx: ReasoningContext, namespace: str, key: str, value: Any):
    original = get_executor(StepType.LLM)
    register_executor(StepType.LLM, _RawMemoryWriter(namespace, key, value))
    try:
        return await _run(
            [LLMStepDescription(number=1, title="raw", aim="x", prompt_template="x")],
            ctx,
            max_workers=1,
        )
    finally:
        register_executor(StepType.LLM, original)


class TestSchemaValidationOnMerge:
    @pytest.mark.asyncio
    async def test_schema_violating_raw_write_is_rejected_at_merge(self):
        ctx = _context(memory_schema={"ns": {"n": int}})

        result = await _run_raw_writer(ctx, "ns", "n", "not-an-int")

        assert not result.success
        failed = result.get_failed_steps()
        assert failed and "expected int" in (failed[0].error_message or "")
        assert "n" not in ctx.memory.get("ns", {})

    @pytest.mark.asyncio
    async def test_schema_conforming_raw_write_is_merged(self):
        ctx = _context(memory_schema={"ns": {"n": int}})

        result = await _run_raw_writer(ctx, "ns", "n", 42)

        assert result.success
        assert ctx.memory["ns"]["n"] == 42

    @pytest.mark.asyncio
    async def test_undeclared_key_is_merged_unchecked(self):
        """Schemas are additive: undeclared (namespace, key) pairs pass through."""
        ctx = _context(memory_schema={"ns": {"n": int}})

        result = await _run_raw_writer(ctx, "ns", "free", object())

        assert result.success
        assert "free" in ctx.memory["ns"]


# ---------------------------------------------------------------------------
# Namespace-level removal
# ---------------------------------------------------------------------------


class TestNamespaceRemoval:
    def test_deleted_namespace_is_reported_and_hidden(self):
        base = {"gone": {"k": 1}, "kept": {"k": 2}}
        cow = COWMemoryStore(base=base)

        del cow["gone"]

        assert "gone" not in cow
        assert cow.deleted_namespaces == {"gone"}
        assert set(cow.keys()) == {"kept"}
        assert base["gone"] == {"k": 1}  # parent untouched until merge

    def test_pop_removes_namespace(self):
        cow = COWMemoryStore(base={"ns": {"k": 1}})

        assert cow.pop("ns") == {"k": 1}
        assert "ns" not in cow
        assert cow.pop("ns", "fallback") == "fallback"

    @pytest.mark.asyncio
    async def test_namespace_delete_is_committed_on_merge(self):
        ctx = _context(memory={"gone": {"k": "v"}, "kept": {"k": "v"}})

        class _DropNamespace(StepExecutorBase):
            async def execute(self, step, context, prompt_template=None):
                del context.memory["gone"]
                return StepExecutionResult(
                    step_number=step.number,
                    step_title=step.title,
                    step_type=StepType.LLM,
                    result="dropped",
                    success=True,
                    updated_history=list(context.history),
                )

        original = get_executor(StepType.LLM)
        register_executor(StepType.LLM, _DropNamespace())
        try:
            result = await _run(
                [LLMStepDescription(number=1, title="drop", aim="x", prompt_template="x")],
                ctx,
                max_workers=1,
            )
        finally:
            register_executor(StepType.LLM, original)

        assert result.success
        assert "gone" not in ctx.memory
        assert ctx.memory["kept"]["k"] == "v"
