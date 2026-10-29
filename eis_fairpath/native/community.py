"""Deterministic all-type historical communities for one causal query."""

from collections import defaultdict
from collections.abc import Mapping
import copy

import networkx as nx

from ..history import EdgeRef, history_edges

from .types import CommunityView, Node


_COMMUNITY_CACHE = {}


def clear_community_cache():
    """Clear the process-local cache used by repeated query explanations."""
    _COMMUNITY_CACHE.clear()


def build_temporal_degrees(edges) -> dict[int, dict[Node, float]]:
    """Build per-snapshot physical incidence degrees for path scoring.

    ``edges`` is the final query candidate domain (normally ``ctx.edges``),
    so events removed by the causal/domain selector are never counted.  Each
    physical event contributes one incidence to each typed endpoint and is
    accumulated only at its own timestamp; repeated event IDs are rejected
    instead of silently inflating a degree.
    """
    if not isinstance(edges, (list, tuple)):
        raise ValueError("edges must be a list or tuple of EdgeRef instances")
    by_timestamp = defaultdict(lambda: defaultdict(float))
    seen = set()
    for edge in edges:
        if not isinstance(edge, EdgeRef):
            raise ValueError("edges must contain EdgeRef instances")
        if edge.event_id in seen:
            raise ValueError(f"duplicate physical event ID: {edge.event_id}")
        seen.add(edge.event_id)
        left = edge.relation[0], edge.src
        right = edge.relation[2], edge.dst
        by_timestamp[edge.timestamp][left] += 1.0
        by_timestamp[edge.timestamp][right] += 1.0
    return {
        int(timestamp): {
            node: float(value)
            for node, value in sorted(degrees.items())
        }
        for timestamp, degrees in sorted(by_timestamp.items())
    }


def build_temporal_coverage_labels(edges, labels) -> dict[int, dict[Node, int]]:
    """Restrict the fixed query partition to nodes present in each snapshot.

    Coverage is evaluated within a snapshot, so a community identifier from
    one year is never compared with an identifier from another year merely by
    its integer value.  The labels themselves remain the fixed query labels;
    this helper only restricts their node domain to the final ``ctx.edges``.
    """
    if not isinstance(labels, Mapping):
        raise ValueError("community labels must be a mapping")
    degrees = build_temporal_degrees(edges)
    result = {}
    for timestamp, nodes in sorted(degrees.items()):
        missing = sorted(node for node in nodes if node not in labels)
        if missing:
            raise ValueError(f"community label missing for {missing[0]}")
        result[timestamp] = {
            node: int(labels[node])
            for node in sorted(nodes)
        }
    return result


def build_temporal_strengths(edges, labels) -> dict[int, dict[Node, float]]:
    """Build snapshot-local community strengths for the final path domain.

    The community labels are the query's fixed labels, while every quantity in
    the strength formula is recomputed on the weighted graph induced by that
    timestamp's events.  In particular, nodes that have no incident event in
    a snapshot are excluded from its PageRank, volume, and community-size
    terms.  This keeps path scoring from mixing historical aggregates with a
    single event's snapshot.
    """
    if not isinstance(edges, (list, tuple)):
        raise ValueError("edges must be a list or tuple of EdgeRef instances")
    if not isinstance(labels, Mapping):
        raise ValueError("community labels must be a mapping")

    by_timestamp = defaultdict(list)
    seen = set()
    for edge in edges:
        if not isinstance(edge, EdgeRef):
            raise ValueError("edges must contain EdgeRef instances")
        if edge.event_id in seen:
            raise ValueError(f"duplicate physical event ID: {edge.event_id}")
        seen.add(edge.event_id)
        left = edge.relation[0], edge.src
        right = edge.relation[2], edge.dst
        if left not in labels:
            raise ValueError(f"community label missing for {left}")
        if right not in labels:
            raise ValueError(f"community label missing for {right}")
        by_timestamp[int(edge.timestamp)].append((left, right))

    result = {}
    for timestamp, routes in sorted(by_timestamp.items()):
        graph = nx.Graph()
        nodes = sorted({node for route in routes for node in route})
        graph.add_nodes_from(nodes)
        weights = defaultdict(float)
        for left, right in routes:
            weights[tuple(sorted((left, right)))] += 1.0
        graph.add_weighted_edges_from(
            (left, right, weight)
            for (left, right), weight in sorted(weights.items())
        )
        total_weight = float(graph.size(weight="weight"))
        pagerank = nx.pagerank(graph, weight="weight") if total_weight else {
            node: 0.0 for node in nodes
        }
        degrees = {
            node: float(value)
            for node, value in graph.degree(weight="weight")
        }
        grouped = defaultdict(list)
        for node in nodes:
            grouped[labels[node]].append(node)

        snapshot_strengths = {}
        for community_id, group in sorted(grouped.items(), key=lambda item: item[0]):
            group = sorted(group)
            internal_weight = float(graph.subgraph(group).size(weight="weight"))
            volume = sum(degrees[node] for node in group)
            strength = (
                internal_weight / total_weight
                - volume ** 2 / (4 * total_weight ** 2)
                + sum(pagerank[node] for node in group) / len(group)
            ) if total_weight else 0.0
            for node in group:
                snapshot_strengths[node] = float(strength)
        result[timestamp] = snapshot_strengths
    return result


def _active_nodes(history) -> list[Node]:
    nodes = set()
    for snapshot in history:
        if set(snapshot.x_dict) != set(snapshot.active):
            raise ValueError("Snapshot features and active node maps must agree")
        for typ, active in snapshot.active.items():
            if active.ndim != 1 or len(active) != len(snapshot.x_dict[typ]):
                raise ValueError("Active masks must have one entry per typed node")
            nodes.update((typ, index) for index in active.detach().cpu().nonzero().flatten().tolist())
    return sorted(nodes)


def _ordered_labels(communities) -> tuple[dict[Node, int], dict[int, frozenset[Node]]]:
    ordered = sorted(
        (frozenset(group) for group in communities),
        key=lambda group: (-len(group), tuple(sorted(group))),
    )
    members = {index: group for index, group in enumerate(ordered)}
    labels = {
        node: index
        for index, group in members.items()
        for node in sorted(group)
    }
    return labels, members


def build_communities(history, *, seed=42, assignments=None) -> CommunityView:
    """Build the fixed weighted Louvain/PageRank view for a legal history."""
    if not isinstance(history, list) or not history:
        raise ValueError("history must be a nonempty list")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")

    nodes = _active_nodes(history)
    edges = history_edges(history)
    assignment_key = None if assignments is None else tuple(sorted(assignments.items()))
    key = (seed, tuple(snapshot.year for snapshot in history), tuple(nodes), tuple(edges), assignment_key)
    cached = _COMMUNITY_CACHE.get(key)
    if cached is not None:
        # CommunityView is intentionally mutable in query contexts (tests and
        # downstream adapters may annotate it). Never hand out the cached
        # instance directly or one query can corrupt another query's view.
        return copy.deepcopy(cached)
    graph = nx.Graph()
    graph.add_nodes_from(nodes)

    weights = defaultdict(float)
    legal = set(nodes)
    for edge in edges:
        left = edge.relation[0], edge.src
        right = edge.relation[2], edge.dst
        if left not in legal or right not in legal:
            raise ValueError("Historical events must connect legal active nodes")
        pair = tuple(sorted((left, right)))
        weights[pair] += 1.0
    graph.add_weighted_edges_from(
        (left, right, weight) for (left, right), weight in sorted(weights.items())
    )

    total_weight = float(graph.size(weight="weight"))
    metadata = {
        "algorithm": "louvain" if assignments is None else "frozen_assignments",
        "seed": seed,
        "node_count": len(nodes),
        "physical_event_count": len(edges),
        "weighted_edge_count": graph.number_of_edges(),
        "total_edge_weight": total_weight,
    }
    if not nodes:
        metadata.update(status="no_nodes", community_count=0, pagerank=[])
        return CommunityView({}, {}, {}, {}, metadata)

    degrees = {node: float(value) for node, value in graph.degree(weight="weight")}
    if assignments is not None:
        if not isinstance(assignments, dict):
            raise ValueError("frozen community assignments must be a mapping")
        missing = sorted(set(nodes) - set(assignments))
        incident_missing = [node for node in missing if degrees[node] > 0]
        if incident_missing:
            raise ValueError(f"frozen community assignments miss incident nodes: {incident_missing[:3]}")
        labels = {node: int(assignments[node]) for node in nodes if node in assignments}
        next_label = max(assignments.values(), default=-1) + 1
        # Census labels cover nodes incident within the historical window.
        # Previously active isolated nodes still occupy teacher tensor rows;
        # singleton labels cannot change any event's Internal/Cross category.
        labels.update({node: next_label + i for i, node in enumerate(missing)})
        grouped = defaultdict(set)
        for node, label in labels.items():
            grouped[label].add(node)
        members = {label: frozenset(group) for label, group in sorted(grouped.items())}
        pagerank = nx.pagerank(graph, weight="weight") if total_weight else {node: 0.0 for node in nodes}
        metadata.update(status="frozen_assignments", community_count=len(members),
                        assignment_count=len(assignments), isolated_unassigned=len(missing), pagerank=[])
    elif total_weight == 0:
        labels, members = _ordered_labels([{node} for node in nodes])
        metadata.update(status="no_edges", community_count=len(members), pagerank=[])
        result = CommunityView(labels, {node: 0.0 for node in nodes}, degrees, members, metadata)
        _COMMUNITY_CACHE[key] = copy.deepcopy(result)
        return result
    else:
        detected = nx.community.louvain_communities(graph, weight="weight", seed=seed)
        labels, members = _ordered_labels(detected)
        pagerank = nx.pagerank(graph, weight="weight")
    strengths = {}
    for community_id, group in members.items():
        internal_weight = float(graph.subgraph(group).size(weight="weight"))
        volume = sum(degrees[node] for node in group)
        strength = (
            internal_weight / total_weight
            - volume ** 2 / (4 * total_weight ** 2)
            + sum(pagerank[node] for node in group) / len(group)
        ) if total_weight else 0.0
        for node in group:
            strengths[node] = float(strength)
    metadata.update(
        status="ok",
        community_count=len(members),
        pagerank=[
            {"node_type": node[0], "node_id": node[1], "value": float(pagerank[node])}
            for node in nodes
        ],
    )
    result = CommunityView(labels, strengths, degrees, members, metadata)
    _COMMUNITY_CACHE[key] = copy.deepcopy(result)
    return result
