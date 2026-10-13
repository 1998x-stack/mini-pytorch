"""Graph analysis utilities for mini-pytorch v0.14 lowering."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .ir import Graph, ValueRef


def iter_value_ids(spec: Any):
    """Yield Value ids referenced by a structured graph spec, preserving occurrences."""
    if isinstance(spec, ValueRef):
        yield int(spec.value_id)
        return
    if isinstance(spec, (tuple, list)):
        for item in spec:
            yield from iter_value_ids(item)
        return
    if isinstance(spec, dict):
        for key, value in spec.items():
            yield from iter_value_ids(key)
            yield from iter_value_ids(value)


@dataclass(frozen=True)
class ValueAnalysis:
    value_id: int
    producer_node_id: int | None
    user_node_ids: tuple[int, ...]
    use_count: int
    is_graph_output: bool
    is_live: bool


@dataclass(frozen=True)
class GraphAnalysis:
    values: dict[int, ValueAnalysis]
    live_value_ids: frozenset[int]
    live_node_ids: frozenset[int]

    def value(self, value_id: int) -> ValueAnalysis:
        try:
            return self.values[int(value_id)]
        except KeyError as exc:
            raise KeyError(f"no analysis for graph value %v{value_id}") from exc

    def format(self) -> str:
        lines = ["GraphAnalysis:"]
        for value_id in sorted(self.values):
            info = self.values[value_id]
            producer = "-" if info.producer_node_id is None else f"node{info.producer_node_id}"
            users = ", ".join(f"node{x}" for x in info.user_node_ids) or "-"
            lines.append(
                f"  %v{value_id}: producer={producer}, users=[{users}], "
                f"use_count={info.use_count}, output={info.is_graph_output}, live={info.is_live}"
            )
        return "\n".join(lines)


def analyze_graph(graph: Graph) -> GraphAnalysis:
    if not isinstance(graph, Graph):
        raise TypeError("analyze_graph expects a Graph")
    graph.validate()

    producers: dict[int, int] = {}
    users: dict[int, list[int]] = {value_id: [] for value_id in graph.values}
    node_by_id = {node.id: node for node in graph.nodes}

    for node in graph.nodes:
        for value_id in iter_value_ids(node.outputs):
            if value_id in producers:
                raise ValueError(f"graph value %v{value_id} has multiple producer nodes")
            producers[value_id] = node.id
        for value_id in iter_value_ids(node.args):
            users[value_id].append(node.id)
        for value_id in iter_value_ids(node.kwargs):
            users[value_id].append(node.id)

    graph_outputs = frozenset(iter_value_ids(graph.outputs))
    live_values: set[int] = set(graph_outputs)
    live_nodes: set[int] = set()
    stack = list(graph_outputs)
    while stack:
        value_id = stack.pop()
        producer_id = producers.get(value_id)
        if producer_id is None or producer_id in live_nodes:
            continue
        live_nodes.add(producer_id)
        node = node_by_id[producer_id]
        for dependency in list(iter_value_ids(node.args)) + list(iter_value_ids(node.kwargs)):
            if dependency not in live_values:
                live_values.add(dependency)
                stack.append(dependency)

    infos: dict[int, ValueAnalysis] = {}
    for value_id in sorted(graph.values):
        user_occurrences = users[value_id]
        infos[value_id] = ValueAnalysis(
            value_id=value_id,
            producer_node_id=producers.get(value_id),
            user_node_ids=tuple(sorted(set(user_occurrences))),
            use_count=len(user_occurrences),
            is_graph_output=value_id in graph_outputs,
            is_live=value_id in live_values,
        )

    return GraphAnalysis(
        values=infos,
        live_value_ids=frozenset(live_values),
        live_node_ids=frozenset(live_nodes),
    )
