"""
Copy-on-write (COW) memory store for parallel step execution isolation.

Instead of ``copy.deepcopy(context.memory)`` before each parallel batch,
the executor creates a ``COWMemoryStore`` that shares read access to the
parent's memory dict without copying.  Namespaces are materialised into a
local view on first access, individual values are isolated lazily on first
read, and **only the keys a step actually wrote** are merged back afterwards.

Key properties
--------------
* **Reads**: zero-copy for namespaces the step never touches.  Touching a
  namespace costs one shallow dict copy; reading a *mutable* value inside it
  additionally deep-copies that one value so in-place mutation cannot reach
  the parent (or a sibling step).  Immutable scalars are never copied.
* **Writes**: tracked per key.  ``view[key] = v`` and ``del view[key]``
  record the key as dirty / deleted; a plain read records nothing.
* **Merge**: the executor merges only :attr:`COWMemoryStore.pending_writes`
  and applies :attr:`COWMemoryStore.pending_deletes`.  A step that only
  *reads* a namespace contributes nothing, so it can no longer revert a
  sibling's write to that namespace (the historical "read reverts write"
  bug, whose winner depended purely on step declaration order).
* **API compatibility**: ``COWMemoryStore`` is a ``dict`` subclass and the
  per-namespace views are ``dict`` subclasses, so all existing code that
  treats ``context.memory`` as ``dict[str, dict[str, Any]]`` keeps working.

Conflict semantics (deliberate, see ``tests/memory/``)
-----------------------------------------------------
Parallel steps in one batch each observe the *pre-batch* state.  When two of
them write the same ``(namespace, key)``, the merge is **last-write-wins in
step declaration order** — the highest-declared step's value survives.  The
same holds for a delete racing a write on one key.  Distinct keys, distinct
namespaces, and delete-vs-write on *different* keys all survive
independently.

In-place mutation of a mutable value (``memory[ns]["items"].append(x)``,
``context.memory_append(...)``) is detected by comparing the step's isolated
copy against the parent value at merge time, so such mutations are merged
back — but two steps appending to the same list in one batch still resolve
last-write-wins, not by union.

Assigning a whole namespace (``memory[ns] = {...}``) replaces it *inside the
step's view*, but merges as writes of the assigned keys only: keys the parent
holds that the new dict omits are left alone.  Use ``del memory[ns]`` to
actually drop a namespace.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Iterator

logger = logging.getLogger(__name__)

# Values of these types can be shared between parent and step without risk:
# they cannot be mutated in place.
_IMMUTABLE_SCALARS = (str, bytes, int, float, bool, complex, type(None))

# How deep to look inside tuples/frozensets before giving up and copying.
_IMMUTABLE_SCAN_DEPTH = 4


def _is_immutable(value: Any, _depth: int = 0) -> bool:
    """True when *value* provably cannot be mutated in place."""
    if isinstance(value, _IMMUTABLE_SCALARS):
        return True
    if _depth < _IMMUTABLE_SCAN_DEPTH and isinstance(value, (tuple, frozenset)):
        return all(_is_immutable(item, _depth + 1) for item in value)
    return False


def _isolate(value: Any) -> "tuple[Any, bool]":
    """Return ``(isolated_value, was_copied)`` for a value read from the base.

    Mutable values are deep-copied so the step cannot reach parent state
    through them.  Objects that refuse to deep-copy (open clients, locks,
    file handles) are returned as-is: isolation is best-effort and must never
    turn a working chain into a crash.
    """
    if _is_immutable(value):
        return value, False
    try:
        return copy.deepcopy(value), True
    except Exception:  # pragma: no cover - depends on user payloads
        logger.debug(
            "COW isolation: value of type %s could not be deep-copied; "
            "in-place mutations of it will leak into parent memory",
            type(value).__name__,
        )
        return value, False


def _values_equal(a: Any, b: Any) -> bool:
    """Best-effort equality used to detect in-place mutation.

    Returns ``False`` when the comparison itself is not a clean boolean
    (numpy arrays, pandas frames, ...).  Erring towards "changed" merges a
    possibly-unmodified value back, which matches the pre-fix behaviour for
    that key; erring the other way would silently drop a real write.
    """
    if a is b:
        return True
    try:
        return bool(a == b)
    except Exception:
        return False


class _NamespaceView(dict):
    """A single memory namespace, copy-on-write over a parent namespace dict.

    The full parent namespace is shallow-copied into the view on creation so
    every dict access path (including C-level ones) sees the complete set of
    keys.  Mutable *values* are isolated lazily on first read via
    :meth:`__getitem__`.

    Tracked state:

    ``_dirty``
        keys explicitly assigned by the step — always merged back.
    ``_deleted``
        keys explicitly removed by the step — deleted from the parent.
    ``_materialized``
        keys whose value was isolated on read; merged back only if the
        isolated copy differs from the parent value (in-place mutation).
    """

    def __init__(
        self,
        initial: "dict[str, Any] | None" = None,
        *,
        base: "dict[str, Any] | None" = None,
        all_dirty: bool = False,
    ) -> None:
        super().__init__(initial if initial is not None else {})
        self._cow_base: dict[str, Any] = base if base is not None else {}
        self._dirty: set[str] = set(dict.keys(self)) if all_dirty else set()
        self._deleted: set[str] = set()
        self._materialized: set[str] = set()

    # ------------------------------------------------------------------
    # Change tracking
    # ------------------------------------------------------------------

    @property
    def written_keys(self) -> "set[str]":
        """Keys this view must merge back: explicit writes + detected mutations."""
        written = set(self._dirty)
        for key in self._materialized:
            if key in self._dirty or key in self._deleted:
                continue
            if not dict.__contains__(self, key):
                continue
            if not _values_equal(dict.__getitem__(self, key), self._cow_base.get(key)):
                written.add(key)
        return written

    @property
    def deleted_keys(self) -> "set[str]":
        """Keys this view removed and that therefore must be dropped from the parent."""
        return set(self._deleted)

    def pending_writes(self) -> "dict[str, Any]":
        """``{key: value}`` for every key this view wrote (or mutated in place)."""
        return {key: dict.__getitem__(self, key) for key in self.written_keys}

    # ------------------------------------------------------------------
    # dict protocol
    # ------------------------------------------------------------------

    def __getitem__(self, key: str) -> Any:
        value = dict.__getitem__(self, key)
        if key in self._dirty or key in self._materialized:
            return value
        isolated, copied = _isolate(value)
        if copied:
            dict.__setitem__(self, key, isolated)
        if not _is_immutable(value):
            # Track even a failed copy: the merge comparison then sees the
            # identical object and correctly reports "unchanged", and we avoid
            # retrying an impossible deepcopy on every read.
            self._materialized.add(key)
        return isolated if copied else value

    def __setitem__(self, key: str, value: Any) -> None:
        dict.__setitem__(self, key, value)
        self._dirty.add(key)
        self._deleted.discard(key)
        self._materialized.discard(key)

    def __delitem__(self, key: str) -> None:
        dict.__delitem__(self, key)
        self._deleted.add(key)
        self._dirty.discard(key)
        self._materialized.discard(key)

    def __iter__(self) -> "Iterator[str]":
        # Defining __iter__ makes CPython take the generic mapping path in
        # ``dict(view)`` / ``{**view}`` / ``dict.update(view)``, so those go
        # through __getitem__ and receive isolated values rather than
        # aliases into parent memory.
        return iter(list(dict.keys(self)))

    def get(self, key: str, default: Any = None) -> Any:  # type: ignore[override]
        try:
            return self[key]
        except KeyError:
            return default

    def items(self):  # type: ignore[override]
        return [(key, self[key]) for key in list(dict.keys(self))]

    def values(self):  # type: ignore[override]
        return [self[key] for key in list(dict.keys(self))]

    def copy(self) -> "dict[str, Any]":  # type: ignore[override]
        return {key: self[key] for key in list(dict.keys(self))}

    def setdefault(self, key: str, default: Any = None) -> Any:  # type: ignore[override]
        if dict.__contains__(self, key):
            return self[key]
        self[key] = default
        return default

    def update(self, other=(), **kwargs) -> None:  # type: ignore[override]
        if hasattr(other, "keys"):
            for key in other.keys():
                self[key] = other[key]
        else:
            for key, value in other:
                self[key] = value
        for key, value in kwargs.items():
            self[key] = value

    def pop(self, key: str, *default: Any) -> Any:  # type: ignore[override]
        if not dict.__contains__(self, key):
            if default:
                return default[0]
            raise KeyError(key)
        value = self[key]
        del self[key]
        return value

    def popitem(self):  # type: ignore[override]
        keys = list(dict.keys(self))
        if not keys:
            raise KeyError("popitem(): dictionary is empty")
        key = keys[-1]  # LIFO, matching dict.popitem
        value = self[key]
        del self[key]
        return key, value

    def clear(self) -> None:  # type: ignore[override]
        for key in list(dict.keys(self)):
            del self[key]


class COWMemoryStore(dict):
    """
    Copy-on-write namespace-level memory store.

    Drop-in replacement for ``dict[str, dict[str, Any]]`` in
    :class:`~mmar_carl.models.context.ReasoningContext`.

    Usage in the executor::

        cow = COWMemoryStore(base=context.memory)
        snapshot.memory = cow

        # After step execution, merge ONLY what the step actually changed:
        for namespace, keys in cow.pending_deletes.items():
            for key in keys:
                context.memory.get(namespace, {}).pop(key, None)
        for namespace, data in cow.pending_writes.items():
            context.memory.setdefault(namespace, {}).update(data)

    Attributes
    ----------
    _base : dict
        Reference to the parent memory dict.  Never mutated.
    """

    def __init__(self, base: "dict[str, dict[str, Any]]") -> None:
        super().__init__()
        # Store base as a plain attribute (not in the dict itself)
        object.__setattr__(self, "_cow_base", base)
        object.__setattr__(self, "_cow_deleted_namespaces", set())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def _base(self) -> "dict[str, dict[str, Any]]":
        return object.__getattribute__(self, "_cow_base")

    @property
    def _deleted_namespaces(self) -> "set[str]":
        return object.__getattribute__(self, "_cow_deleted_namespaces")

    def _ensure_local(self, namespace: str) -> "_NamespaceView":
        """Materialise *namespace* as a local view if not already local."""
        if not dict.__contains__(self, namespace):
            base_ns = self._base.get(namespace)
            if base_ns is None or namespace in self._deleted_namespaces:
                view = _NamespaceView()
            else:
                view = _NamespaceView(base_ns, base=base_ns)
            dict.__setitem__(self, namespace, view)
        return dict.__getitem__(self, namespace)

    # ------------------------------------------------------------------
    # Change tracking — what the executor merges back
    # ------------------------------------------------------------------

    @property
    def pending_writes(self) -> "dict[str, dict[str, Any]]":
        """``{namespace: {key: value}}`` for keys this store actually wrote.

        Namespaces that were only read contribute nothing, which is what
        keeps a read-only step from reverting a sibling's write.
        """
        changes: dict[str, dict[str, Any]] = {}
        for namespace, view in dict.items(self):
            if isinstance(view, _NamespaceView):
                data = view.pending_writes()
            else:  # a plain dict assigned directly — treat every key as written
                data = dict(view)
            if data:
                changes[namespace] = data
        return changes

    @property
    def pending_deletes(self) -> "dict[str, set[str]]":
        """``{namespace: {key, ...}}`` for keys this store deleted."""
        deletes: dict[str, set[str]] = {}
        for namespace, view in dict.items(self):
            if not isinstance(view, _NamespaceView):
                continue
            keys = view.deleted_keys
            if keys:
                deletes[namespace] = keys
        return deletes

    @property
    def deleted_namespaces(self) -> "set[str]":
        """Namespaces removed wholesale via ``del store[namespace]``."""
        return {ns for ns in self._deleted_namespaces if not dict.__contains__(self, ns)}

    @property
    def overlay(self) -> "dict[str, dict[str, Any]]":
        """Deprecated alias for :attr:`pending_writes`.

        Historically this returned every *touched* namespace (reads
        included), which is precisely what made a read-only step clobber a
        sibling's write on merge.  It now returns only written keys; use
        :attr:`pending_writes` / :attr:`pending_deletes` in new code.
        """
        return self.pending_writes

    # ------------------------------------------------------------------
    # dict protocol overrides
    # ------------------------------------------------------------------

    def __getitem__(self, namespace: str) -> "dict[str, Any]":
        if dict.__contains__(self, namespace):
            return dict.__getitem__(self, namespace)
        if namespace in self._base and namespace not in self._deleted_namespaces:
            return self._ensure_local(namespace)
        raise KeyError(namespace)

    def __setitem__(self, namespace: str, value: "dict[str, Any]") -> None:
        self._deleted_namespaces.discard(namespace)
        if isinstance(value, _NamespaceView):
            dict.__setitem__(self, namespace, value)
            return
        # An explicit namespace assignment is a write of every key it carries.
        dict.__setitem__(self, namespace, _NamespaceView(value, all_dirty=True))

    def __delitem__(self, namespace: str) -> None:
        if not self.__contains__(namespace):
            raise KeyError(namespace)
        if dict.__contains__(self, namespace):
            dict.__delitem__(self, namespace)
        self._deleted_namespaces.add(namespace)

    def __contains__(self, item: object) -> bool:
        if dict.__contains__(self, item):
            return True
        return item in self._base and item not in self._deleted_namespaces

    def get(self, namespace: str, default: Any = None) -> Any:  # type: ignore[override]
        try:
            return self[namespace]
        except KeyError:
            return default

    def pop(self, namespace: str, *default: Any) -> Any:  # type: ignore[override]
        if not self.__contains__(namespace):
            if default:
                return default[0]
            raise KeyError(namespace)
        value = self[namespace]
        del self[namespace]
        return value

    def __iter__(self) -> "Iterator[str]":
        seen: set[str] = set(dict.keys(self))
        yield from seen
        for ns in self._base:
            if ns not in seen and ns not in self._deleted_namespaces:
                yield ns

    def __len__(self) -> int:
        return len(self.keys())

    def items(self):  # type: ignore[override]
        # Route through __getitem__ so callers receive isolated namespace
        # views rather than the parent's own namespace dicts.
        return [(ns, self[ns]) for ns in list(self)]

    def keys(self):  # type: ignore[override]
        base_keys = {ns for ns in self._base if ns not in self._deleted_namespaces}
        return set(dict.keys(self)) | base_keys

    def values(self):  # type: ignore[override]
        return [value for _, value in self.items()]

    def setdefault(self, namespace: str, default: Any = None) -> Any:  # type: ignore[override]
        if namespace not in self:
            self[namespace] = default if default is not None else {}
        return self[namespace]

    def update(self, other=(), **kwargs):  # type: ignore[override]
        if hasattr(other, "items"):
            for k, v in other.items():
                self[k] = v
        else:
            for k, v in other:
                self[k] = v
        for k, v in kwargs.items():
            self[k] = v

    def __repr__(self) -> str:  # pragma: no cover
        overlay_keys = set(dict.keys(self))
        base_keys = set(self._base.keys())
        return (
            f"COWMemoryStore(local_namespaces={sorted(overlay_keys)}, "
            f"base_namespaces={sorted(base_keys - overlay_keys)}, "
            f"pending_writes={ {ns: sorted(d) for ns, d in self.pending_writes.items()} })"
        )
