from typing import TypeVar

T = TypeVar("T")


def sort_graph(graph: dict[T, list[T]]) -> list[T]:
    visited: set[T] = set([])

    result = []

    def helper(vertix: T) -> None:
        visited.add(vertix)
        for dep in graph[vertix]:
            if dep not in visited:
                helper(dep)
        result.append(vertix)

    for v in graph.keys():
        if v not in visited:
            helper(v)

    return result
