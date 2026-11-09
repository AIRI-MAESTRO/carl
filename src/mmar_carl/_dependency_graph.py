"""Shared traversal for chain dependency validation."""

from collections.abc import Iterable, Iterator, Mapping


def dependency_back_edges[Node](
    dependencies: Mapping[Node, Iterable[Node]],
) -> Iterator[tuple[Node, Node]]:
    """Yield the DFS root and repeated node for each dependency back edge.

    Follow the input's node and dependency order. Missing nodes are checked
    separately by the caller.
    """
    visited: set[Node] = set()
    active: set[Node] = set()
    for root in dependencies:
        if root in visited:
            continue
        visited.add(root)
        active.add(root)
        stack = [(root, iter(dependencies[root]))]
        while stack:
            node, edges = stack[-1]
            for dependency in edges:
                if dependency in active:
                    yield root, dependency
                elif dependency in dependencies and dependency not in visited:
                    visited.add(dependency)
                    active.add(dependency)
                    stack.append((dependency, iter(dependencies[dependency])))
                    break
            else:
                active.remove(node)
                stack.pop()
