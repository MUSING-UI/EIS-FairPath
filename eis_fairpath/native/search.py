"""Exact current-mask H4 search with native linear temporal path scoring.

Distinct physical routes and node-simple paths are retained. Snapshot coverage
is computed separately at each event timestamp. Residual reachability removes
only impossible completions after the original numeric branch bounds.
"""
from __future__ import annotations

import math
import time
from collections import defaultdict, deque, OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral

import numpy as np

from ..history import EdgeRef
from .types import Node

LINEAR_TEMPORAL_SCORE = "temporal_additive_linear"
_REVERSE = {
    ("author", "writes", "paper"): ("paper", "written_by", "author"),
    ("paper", "written_by", "author"): ("author", "writes", "paper"),
    ("paper", "published_in", "venue"): ("venue", "publishes", "paper"),
    ("venue", "publishes", "paper"): ("paper", "published_in", "venue"),
    ("author", "collaborates", "author"): ("author", "collaborates", "author"),
}


class ResidualReachability:
    """One search's static directed topology; cache only visited-node sets.

    Each result excludes every visited node as an intermediate predecessor,
    exactly as the existing native check before K paths exist. The fixed
    graph/target live on this instance, so cache keys never mix queries,
    weights, masks, score modes, communities or separate searches.
    """

    def __init__(self, reverse_nodes, target, *, limit=128):
        if type(limit) is not int or limit <= 0:
            raise ValueError('Positive integer bounded cache limit required')
        self._reverse = {node: tuple(rows) for node, rows in reverse_nodes.items()}
        self._target = target
        self.limit = limit
        self._cache = OrderedDict()

    def __len__(self):
        return len(self._cache)

    def reachable(self, visited):
        token = frozenset(visited)
        if token in self._cache:
            result = self._cache.pop(token)
            self._cache[token] = result
            return result
        reachable = {self._target}
        pending = deque([self._target])
        while pending:
            node = pending.popleft()
            for predecessor in self._reverse.get(node, ()):
                if predecessor not in token and predecessor not in reachable:
                    reachable.add(predecessor)
                    pending.append(predecessor)
        result = frozenset(reachable)
        self._cache[token] = result
        if len(self._cache) > self.limit:
            self._cache.popitem(last=False)
        return result


@dataclass(frozen=True)
class RankedPath:
    edges: tuple[EdgeRef, ...]
    score: float
    terms: dict

    @property
    def event_ids(self) -> tuple[str, ...]:
        return tuple(edge.event_id for edge in self.edges)


@dataclass
class SearchResult:
    paths: list[RankedPath]
    stats: dict


@dataclass(frozen=True)
class _ExactSearchTopology:
    """Query-static topology for repeated exact feedback searches.

    The optimizer changes event weights on every step, but it does not change
    the event routes, typed nodes, or which nodes can reach the query target.
    Keeping those facts outside ``_exact_topk`` avoids rebuilding the same
    adjacency and reachability structures hundreds of times per query.
    """

    event_ids: tuple[str, ...]
    source: Node
    target: Node
    adjacency: dict[Node, tuple[EdgeRef, ...]]
    nodes: tuple[Node, ...]
    node_index: dict[Node, int]
    route_dest: dict[EdgeRef, Node]
    transition_sources: np.ndarray
    transition_destinations: np.ndarray
    transition_routes: tuple[EdgeRef, ...]
    target_index: int
    pruned_routes: int



def prepare_exact_topology(edges: Sequence[EdgeRef], source, target):
    """Prepare a reusable, semantics-preserving exact-search topology.

    Only routes that are reachable from ``source`` and can still reach
    ``target`` are retained.  Such a route cannot occur in any legal source to
    target path, so this reduction does not change the complete path set or
    its ranking.  Event weights are deliberately absent from the object
    because they change during mask optimization.
    """
    source = _node(source, "source")
    target = _node(target, "target")
    edges = tuple(edges)
    if any(not isinstance(edge, EdgeRef) for edge in edges):
        raise ValueError("edges must contain EdgeRef instances")
    event_ids = tuple(edge.event_id for edge in edges)
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("edges must contain unique physical event IDs")

    raw = _adjacency(edges)
    forward = {source}
    pending = deque([source])
    while pending:
        node = pending.popleft()
        for route in raw.get(node, ()):
            neighbor = _route_nodes(route)[1]
            if neighbor not in forward:
                forward.add(neighbor)
                pending.append(neighbor)

    reverse = defaultdict(list)
    for routes in raw.values():
        for route in routes:
            predecessor, neighbor = _route_nodes(route)
            reverse[neighbor].append(predecessor)
    can_reach_target = {target}
    pending = deque([target])
    while pending:
        node = pending.popleft()
        for predecessor in reverse.get(node, ()):
            if predecessor not in can_reach_target:
                can_reach_target.add(predecessor)
                pending.append(predecessor)

    relevant = forward & can_reach_target
    filtered = defaultdict(list)
    raw_route_count = sum(len(routes) for routes in raw.values())
    for node in relevant:
        for route in raw.get(node, ()):
            neighbor = _route_nodes(route)[1]
            if neighbor in relevant:
                filtered[node].append(route)
    adjacency = {
        node: tuple(routes) for node, routes in filtered.items()
        if routes
    }
    nodes = tuple(sorted(
        {node for routes in adjacency.values() for route in routes
         for node in _route_nodes(route)} | {source, target}
    ))
    node_index = {node: index for index, node in enumerate(nodes)}
    route_dest = {
        route: _route_nodes(route)[1]
        for routes in adjacency.values() for route in routes
    }

    transition_sources = []
    transition_destinations = []
    transition_routes = []
    for node in sorted(adjacency):
        source_index = node_index[node]
        for route in adjacency[node]:
            transition_sources.append(source_index)
            transition_destinations.append(node_index[route_dest[route]])
            transition_routes.append(route)
    transitions = (
        np.asarray(transition_sources, dtype=np.int64),
        np.asarray(transition_destinations, dtype=np.int64),
    )
    return _ExactSearchTopology(
        event_ids=event_ids,
        source=source,
        target=target,
        adjacency=adjacency,
        nodes=nodes,
        node_index=node_index,
        route_dest=route_dest,
        transition_sources=transitions[0],
        transition_destinations=transitions[1],
        transition_routes=tuple(transition_routes),
        target_index=node_index[target],
        pruned_routes=raw_route_count - len(transition_routes),
    )


def _node(value, name: str) -> Node:
    if (not isinstance(value, tuple) or len(value) != 2
            or not isinstance(value[0], str) or not value[0]
            or isinstance(value[1], bool) or not isinstance(value[1], Integral)
            or value[1] < 0):
        raise ValueError(f"{name} must be a typed node (type, nonnegative index)")
    return value[0], int(value[1])


def _limit(value, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _finite(value, name: str) -> float:
    try:
        if hasattr(value, "detach"):
            value = value.detach().cpu().item()
        result = float(value)
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError(f"{name} must be a finite scalar") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite scalar")
    return result


def _route_key(edge: EdgeRef):
    return (edge.event_id, edge.timestamp, edge.relation, edge.src, edge.dst)


def _path_key(edges: tuple[EdgeRef, ...]):
    return tuple(_route_key(edge) for edge in edges)


def _reverse(edge: EdgeRef) -> EdgeRef:
    return EdgeRef(edge.dst, edge.src, _REVERSE[edge.relation],
                   edge.timestamp, edge.event_id)


def _route_nodes(edge: EdgeRef) -> tuple[Node, Node]:
    return (edge.relation[0], edge.src), (edge.relation[2], edge.dst)


def _strength(community, node, timestamp=None):
    """Return the strength for ``node`` in the event's snapshot.

    Standalone callers that provide only the historical ``CommunityView``
    continue to use ``strengths``.  Query contexts attach a temporal map; once
    an event timestamp is supplied, a missing snapshot or endpoint is an
    invalid context rather than an implicit fallback to historical values.
    """
    temporal = getattr(community, "path_strengths_by_timestamp", None)
    if temporal is not None and timestamp is not None:
        try:
            value = temporal[timestamp][node]
        except (KeyError, TypeError) as error:
            raise ValueError(
                f"path strength missing for {node} at timestamp {timestamp}"
            ) from error
        return _finite(value, f"path strength for {node} at {timestamp}")
    return _finite(community.strengths[node], f"strength for {node}")


def _degree(community, node, timestamp=None):
    """Return the path degree for ``node`` at an event timestamp.

    Query contexts provide ``path_degrees_by_timestamp``.  Standalone callers
    that construct the historical ``CommunityView`` shape still use the
    legacy aggregate degree when no temporal map (or timestamp) is supplied.
    """
    temporal = getattr(community, "path_degrees_by_timestamp", None)
    if temporal is not None and timestamp is not None:
        try:
            value = temporal[timestamp][node]
        except (KeyError, TypeError) as error:
            raise ValueError(
                f"path degree missing for {node} at timestamp {timestamp}"
            ) from error
        return _finite(value, f"path degree for {node} at {timestamp}")
    return _finite(community.degrees[node], f"degree for {node}")


def _coverage_label(community, node, timestamp=None):
    """Return the community identifier used by snapshot-local coverage."""
    temporal = getattr(community, "path_community_labels_by_timestamp", None)
    if temporal is not None and timestamp is not None:
        try:
            value = temporal[timestamp][node]
        except (KeyError, TypeError) as error:
            raise ValueError(
                f"path coverage community missing for {node} at timestamp {timestamp}"
            ) from error
    else:
        value = community.labels[node]
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError("path coverage community labels must be integers")
    return int(value)


def _validate_inputs(edges, source, target, weights, community, *, allow_weights_above_one=False):
    source = _node(source, "source")
    target = _node(target, "target")
    edges = list(edges)
    if any(not isinstance(edge, EdgeRef) for edge in edges):
        raise ValueError("edges must contain EdgeRef instances")
    event_ids = [edge.event_id for edge in edges]
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("edges must contain unique physical event IDs")
    if not isinstance(weights, Mapping):
        raise ValueError("weights must map every event ID to one scalar")
    expected, supplied = set(event_ids), set(weights)
    missing, unknown = expected - supplied, supplied - expected
    if missing:
        raise ValueError(f"missing event weights: {sorted(missing)}")
    if unknown:
        raise ValueError(f"unknown event weights: {sorted(unknown)}")
    parsed_weights = {}
    for event_id in event_ids:
        value = _finite(weights[event_id], f"weight for {event_id}")
        if value <= 0 or (value > 1 and not allow_weights_above_one):
            interval = "positive" if allow_weights_above_one else "in (0, 1]"
            raise ValueError(f"event weights must be {interval}")
        parsed_weights[event_id] = value

    for attribute in ("labels", "strengths", "degrees"):
        if not isinstance(getattr(community, attribute, None), Mapping):
            raise ValueError(f"community.{attribute} must be a mapping")
    temporal_degrees = getattr(community, "path_degrees_by_timestamp", None)
    if temporal_degrees is not None and not isinstance(temporal_degrees, Mapping):
        raise ValueError("community.path_degrees_by_timestamp must be a mapping")
    temporal_strengths = getattr(community, "path_strengths_by_timestamp", None)
    if temporal_strengths is not None and not isinstance(temporal_strengths, Mapping):
        raise ValueError("community.path_strengths_by_timestamp must be a mapping")
    temporal_coverage = getattr(community, "path_community_labels_by_timestamp", None)
    if temporal_coverage is not None and not isinstance(temporal_coverage, Mapping):
        raise ValueError(
            "community.path_community_labels_by_timestamp must be a mapping"
        )
    graph_nodes = {node for edge in edges for node in _route_nodes(edge)}
    for node in graph_nodes:
        if node not in community.labels:
            raise ValueError(f"community label missing for {node}")
        if node not in community.strengths:
            raise ValueError(f"community strength missing for {node}")
        if node not in community.degrees:
            raise ValueError(f"community degree missing for {node}")
        _strength(community, node)
        if (isinstance(community.labels[node], bool)
                or not isinstance(community.labels[node], Integral)):
            raise ValueError("community labels must be integers")
    for edge in edges:
        for node in _route_nodes(edge):
            _strength(community, node, edge.timestamp)
            _coverage_label(community, node, edge.timestamp)
            degree = _degree(community, node, edge.timestamp)
            if degree <= 0:
                raise ValueError("community degrees for event endpoints must be positive")
    return edges, source, target, parsed_weights


def _adjacency(edges: Sequence[EdgeRef]):
    adjacency = defaultdict(list)
    for edge in edges:
        source, _ = _route_nodes(edge)
        adjacency[source].append(edge)
        reverse = _reverse(edge)
        if reverse != edge:
            reverse_source, _ = _route_nodes(reverse)
            adjacency[reverse_source].append(reverse)
    for routes in adjacency.values():
        routes.sort(key=_route_key)
    return adjacency


def _linear_edge_terms(edge, weights, community, strength_weight, degree_weight,
                       length_weight):
    """Score phase-specific w directly; the caller supplies the author offset once."""
    _require_snapshot_scoring(community, LINEAR_TEMPORAL_SCORE)
    source, target = _route_nodes(edge)
    degree = _degree(community, target, edge.timestamp)
    if degree < 1.:
        raise ValueError("snapshot incidence degree must be >= 1 for a path endpoint")
    weight = _finite(weights[edge.event_id], "phase-specific weight")
    if weight <= 0.:
        raise ValueError("phase-specific weight must be positive")
    coefficients = [
        _finite(value, name) for value, name in (
            (strength_weight, "strength_weight"), (degree_weight, "degree_weight"),
            (length_weight, "length_weight"))]
    if min(coefficients) < 0.:
        raise ValueError("score weights must be nonnegative")
    strength_weight, degree_weight, length_weight = coefficients
    gap = abs(_strength(community, source, edge.timestamp)
              - _strength(community, target, edge.timestamp))
    return {"log_weight": math.log(weight), "strength": strength_weight * gap,
            "degree": -degree_weight * math.log(degree), "length": -length_weight}


@dataclass(frozen=True)
class _LengthUpperBounds:
    values: np.ndarray
    peak: int

    def __getitem__(self, interval):
        minimum, maximum = interval
        if minimum < 1 or maximum < minimum:
            return -math.inf
        return self.values[min(max(self.peak, minimum), maximum)]


def _length_upper_bounds(route_terms, best_target_incoming, node_count):
    """Exact interval maxima in O(V) space instead of an O(V**2) table.

    Prefix sums of descending route terms increase, then decrease. Adding
    the fixed final-edge term preserves that order, so an interval maximum
    is its endpoint nearest the global peak (or the peak itself). Preserve
    the original sequential floating-point sums and additions verbatim.
    """
    values = np.full(node_count + 1, -math.inf, dtype=np.float64)
    if math.isfinite(best_target_incoming):
        prefix = [0.0]
        for value in sorted(route_terms, reverse=True)[:node_count]:
            prefix.append(prefix[-1] + value)
        for length in range(1, node_count + 1):
            values[length] = best_target_incoming + prefix[min(length - 1, len(prefix) - 1)]
    return _LengthUpperBounds(values, int(np.argmax(values)))


def _exact_topk(edges, source, target, weights, community, *, feedback_max_hops,
                k, strength_weight, degree_weight, coverage_weight,
                length_weight=0.0, exact_topology=None):
    """Rank complete node-simple paths using admissible native score bounds."""
    started = time.perf_counter()
    if exact_topology is None:
        exact_topology = prepare_exact_topology(edges, source, target)
        topology_reused = False
    else:
        expected_ids = tuple(edge.event_id for edge in edges)
        if (not isinstance(exact_topology, _ExactSearchTopology)
                or exact_topology.event_ids != expected_ids
                or exact_topology.source != source
                or exact_topology.target != target):
            raise ValueError("exact_topology does not match the search inputs")
        topology_reused = True
    adjacency = exact_topology.adjacency
    # The topology has already removed routes that are unreachable from the
    # source or cannot reach the target.  Keeping this set preserves the old
    # safety check while making the per-search loop use only relevant nodes.
    can_reach_target = set(exact_topology.nodes)

    # For a finite feedback K, an admissible branch bound avoids scoring every
    # hopeless long prefix while preserving the global top-K guarantee.
    # Additive, linear and bounded temporal coverage penalties are <= 0;
    # omitting them from the prefix/suffix bound is therefore admissible. The
    # relaxation ignores node simplicity and permits at most |V| remaining
    # steps, so it is an upper bound on every legal completion. It is never
    # used for k=None (the final output scan), which remains a complete
    # enumeration of the legal path set.
    route_terms = {
        route: sum(_linear_edge_terms(
            route, weights, community, strength_weight, degree_weight,
            length_weight).values())
        for routes in adjacency.values() for route in routes
    }
    nodes = set(exact_topology.nodes)
    future_bounds = None
    positive_suffix = None
    length_upper = None
    hop_distance = None
    bound_started = time.perf_counter()
    if k is not None:
        node_count = len(nodes)
        previous = np.full(node_count, -math.inf, dtype=np.float64)
        previous[exact_topology.target_index] = 0.0
        future_bounds = [previous]
        route_terms_array = np.asarray(
            [route_terms[route] for route in exact_topology.transition_routes],
            dtype=np.float64,
        )
        source_indices = exact_topology.transition_sources
        destination_indices = exact_topology.transition_destinations
        target_index = exact_topology.target_index
        for _ in range(node_count):
            current = previous.copy()
            if len(route_terms_array):
                candidates = route_terms_array + previous[destination_indices]
                np.maximum.at(current, source_indices, candidates)
            # Paths terminate at the query target. The scalar recurrence
            # skipped outgoing routes from target, so restore that boundary.
            current[target_index] = 0.0
            # Relax upward by one ULP to keep the bound admissible under
            # floating-point roundoff. This may reduce pruning slightly, but
            # can never remove a path that the scalar recurrence would retain.
            finite = np.isfinite(current)
            current[finite] = np.nextafter(current[finite], np.inf)
            current[target_index] = 0.0
            future_bounds.append(current)
            previous = current
        # The max-plus recurrence ignores node simplicity, so a positive
        # cycle can make its bound much looser than any legal simple path.
        # The sum of the largest positive route terms that fit in the
        # remaining depth is another admissible bound.  Taking the minimum
        # keeps exactness while avoiding pathological optimistic branches.
        positive_terms = sorted(
            (max(0.0, float(value)) for value in route_terms.values()),
            reverse=True,
        )
        positive_suffix = [0.0]
        for value in positive_terms[:node_count]:
            positive_suffix.append(positive_suffix[-1] + value)

        # Every completion has one final route entering the target.  Combine
        # the best such route with the globally largest remaining route terms
        # and the reverse hop distance.  This remains an upper bound (it may
        # reuse a route and ignores node types), but unlike a zero-positive
        # bound it accounts for the negative edges that a target-reaching
        # path is forced to pay.
        reverse_nodes = defaultdict(list)
        target_incoming_terms = []
        for routes in adjacency.values():
            for route in routes:
                source_node, neighbor = _route_nodes(route)
                reverse_nodes[neighbor].append(source_node)
                if neighbor == target:
                    target_incoming_terms.append(route_terms[route])
        hop_distance = {target: 0}
        pending = deque([target])
        while pending:
            node = pending.popleft()
            for predecessor in reverse_nodes.get(node, ()):
                if predecessor not in hop_distance:
                    hop_distance[predecessor] = hop_distance[node] + 1
                    pending.append(predecessor)
        best_target_incoming = max(target_incoming_terms, default=-math.inf)
        length_upper = _length_upper_bounds(
            route_terms.values(), best_target_incoming, node_count)
        residual_reachability = ResidualReachability(reverse_nodes, target, limit=128)
    bound_elapsed = time.perf_counter() - bound_started
    enumeration_started = time.perf_counter()
    best = []
    candidate_paths = 0
    expansions = 0
    unreachable_pruned = exact_topology.pruned_routes
    bound_pruned = 0
    # Iterator frames keep working memory proportional to path depth, not to
    # the number of pending path prefixes. No recursion or expansion cap.
    # The top-k and complete-audit branches both retain depth-first stacks.

    def keep(path):
        nonlocal candidate_paths
        candidate_paths += 1
        ranked = _rank_path(
            path, source, weights, community, strength_weight, degree_weight,
            coverage_weight, length_weight=length_weight)
        best.append(ranked)
        if k is not None:
            best.sort(key=lambda item: (-item.score, _path_key(item.edges)))
            if len(best) > k:
                best.pop()

    if k is not None:
        # Depth-first exact top-k search.  The previous best-first frontier
        # retained every unresolved prefix, which can exhaust memory on a
        # dense causal domain even though the score bound is admissible.  A
        # depth-first traversal keeps only one path stack while sorting each
        # local branch by its upper bound so good paths still arrive early.
        def upper_bound_children(path, node, visited, prefix_score):
            nonlocal expansions, unreachable_pruned, bound_pruned
            if (feedback_max_hops is not None
                    and len(path) >= feedback_max_hops):
                return []
            reachable_remaining = can_reach_target
            if len(best) < k:
                # Before k complete paths exist, score bounds cannot prune.
                # Exclude already visited nodes from reverse reachability:
                # a legal node-simple completion cannot pass through them.
                # This avoids enumerating dense dead ends while waiting for
                # the kth path and never removes a legal completion.
                reachable_remaining = {target}
                pending = deque([target])
                while pending:
                    reachable_node = pending.popleft()
                    for predecessor in reverse_nodes.get(reachable_node, ()):
                        if (predecessor not in visited
                                and predecessor not in reachable_remaining):
                            reachable_remaining.add(predecessor)
                            pending.append(predecessor)
            children = []
            for route in adjacency.get(node, ()):
                neighbor = exact_topology.route_dest.get(route)
                if neighbor is None:
                    neighbor = _route_nodes(route)[1]
                if neighbor in visited:
                    continue
                if neighbor not in reachable_remaining:
                    unreachable_pruned += 1
                    continue
                expansions += 1
                next_score = prefix_score + route_terms[route]
                potential_bridge = 0.0
                if neighbor == target:
                    child_bound = next_score + potential_bridge
                else:
                    remaining = min(
                        len(nodes) - len(visited), len(future_bounds) - 1)
                    future = float(future_bounds[remaining][
                        exact_topology.node_index[neighbor]])
                    if future == -math.inf:
                        bound_pruned += 1
                        continue
                    future = min(future, positive_suffix[min(
                        remaining, len(positive_suffix) - 1)])
                    minimum_length = hop_distance.get(neighbor)
                    if (minimum_length is None
                            or minimum_length > remaining):
                        bound_pruned += 1
                        continue
                    if length_upper is not None:
                        future = min(future, float(length_upper[
                            minimum_length, remaining]))
                        if future == -math.inf:
                            bound_pruned += 1
                            continue
                    child_bound = next_score + future + potential_bridge
                if len(best) >= k and child_bound < best[-1].score:
                    bound_pruned += 1
                    continue
                if len(best) >= k:
                    if neighbor not in residual_reachability.reachable(visited):
                        unreachable_pruned += 1
                        continue
                children.append((
                    child_bound, _route_key(route), route, neighbor,
                    next_score,
                ))
            children.sort(key=lambda item: (-item[0], item[1]))
            return children

        def visit(path, node, visited, prefix_score):
            nonlocal bound_pruned
            if node == target:
                keep(path)
                return
            for (child_bound, _, route, neighbor, next_score) in upper_bound_children(
                    path, node, visited, prefix_score):
                if len(best) >= k and child_bound < best[-1].score:
                    bound_pruned += 1
                    continue
                visit(path + (route,), neighbor, visited | {neighbor},
                      next_score)

        if source != target and source in can_reach_target:
            visit(tuple(), source, frozenset({source}), 0.0)
        algorithm = "depth_first_complete_node_simple_path_branch_and_bound"
    else:
        # ``feedback_k=None`` is retained for complete-path audits. It must
        # materialize every legal path and therefore cannot use early top-k
        # termination.
        path, visited = [], {source}
        prefix_scores = [0.0]
        stack = ([iter(adjacency.get(source, ()))]
                 if source != target and source in can_reach_target else [])
        while stack:
            route = next(stack[-1], None)
            if route is None:
                stack.pop()
                if path:
                    popped = path.pop()
                    neighbor = exact_topology.route_dest.get(popped)
                    if neighbor is None:
                        neighbor = _route_nodes(popped)[1]
                    visited.remove(neighbor)
                    prefix_scores.pop()
                continue
            neighbor = exact_topology.route_dest.get(route)
            if neighbor is None:
                neighbor = _route_nodes(route)[1]
            if neighbor in visited:
                continue
            if neighbor not in can_reach_target:
                unreachable_pruned += 1
                continue
            if feedback_max_hops is not None and len(path) >= feedback_max_hops:
                continue
            expansions += 1
            next_score = prefix_scores[-1] + route_terms[route]
            path.append(route)
            if neighbor == target:
                keep(tuple(path))
                path.pop()
            else:
                visited.add(neighbor)
                prefix_scores.append(next_score)
                stack.append(iter(adjacency.get(neighbor, ())))
        algorithm = "complete_node_simple_path_branch_and_bound"

    best.sort(key=lambda item: (-item.score, _path_key(item.edges)))
    enumeration_elapsed = time.perf_counter() - enumeration_started
    elapsed = time.perf_counter() - started

    ranking_guarantee = (
        "global_complete_path_score"
        if feedback_max_hops is None
        else "bounded_complete_path_score"
    )
    return SearchResult(best, {
        "mode": "exact_topk",
        "algorithm": algorithm,
        "complete": True,
        "search_complete": True,
        "optimal": True,
        "ranking_guarantee": ranking_guarantee,
        "truncated": False,
        "truncated_states": 0,
        "expansion_limit_reached": False,
        "expansions": expansions,
        "unreachable_pruned": unreachable_pruned,
        "bound_pruned": bound_pruned,
        "max_expansions": None,
        "path_limit": None,
        "candidate_pool": None,
        "candidate_paths": candidate_paths,
        "candidate_paths_scored": candidate_paths,
        "candidate_paths_exact": k is None,
        "returned_paths": len(best),
        "requested_paths": k,
        "feedback_k": k,
        "feedback_max_hops": feedback_max_hops,
        "hop_limit": feedback_max_hops,
        "event_budget_applied": False,
        "feedback_path_lengths": [len(path.edges) for path in best],
        "feedback_path_scores": [float(path.score) for path in best],
        "feedback_path_terms": [dict(path.terms) for path in best],
        "score_mode": LINEAR_TEMPORAL_SCORE,
        "topology_reused": topology_reused,
        "topology_nodes": len(nodes),
        "topology_routes": len(exact_topology.transition_routes),
        "topology_pruned_routes": exact_topology.pruned_routes,
        "bound_backend": "numpy_maxplus",
        "bound_seconds": bound_elapsed,
        "enumeration_seconds": enumeration_elapsed,
        "elapsed_seconds": elapsed,
    })


def extract_paths(paths, *, event_budget=15, max_paths=3,
                  output_max_hops=None) -> list[RankedPath]:
    """Greedily select ranked paths under explicit output constraints."""
    event_budget = _limit(event_budget, "event_budget")
    max_paths = _limit(max_paths, "max_paths")
    if output_max_hops is not None:
        output_max_hops = _limit(output_max_hops, "output_max_hops")
    if max_paths == 0:
        return []
    selected = []
    selected_events = set()
    for path in paths:
        if not isinstance(path, RankedPath):
            raise ValueError("paths must contain RankedPath instances")
        if output_max_hops is not None and len(path.edges) > output_max_hops:
            continue
        path_events = set(path.event_ids)
        if len(selected_events | path_events) > event_budget:
            continue
        selected.append(path)
        selected_events.update(path_events)
        if len(selected) == max_paths:
            break
    return selected


def _require_snapshot_scoring(community, score_mode=LINEAR_TEMPORAL_SCORE):
    for name in ("path_degrees_by_timestamp", "path_strengths_by_timestamp",
                 "path_community_labels_by_timestamp"):
        if not isinstance(getattr(community, name, None), Mapping):
            raise ValueError(f"Linear temporal score requires snapshot map {name}")


def _rank_path(edges, source, weights, community, strength_weight, degree_weight,
               coverage_weight, *, length_weight=0.0):
    """Native log-weight, strength, degree and per-snapshot coverage score."""
    edge_terms = [
        _linear_edge_terms(edge, weights, community, strength_weight,
                           degree_weight, length_weight)
        for edge in edges
    ]
    weight_term = sum(term["log_weight"] for term in edge_terms)
    strength_term = sum(term["strength"] for term in edge_terms)
    degree_term = sum(term["degree"] for term in edge_terms)
    nodes = [source] + [_route_nodes(edge)[1] for edge in edges]
    communities = {_coverage_label(community, node) for node in nodes}
    snapshot_nodes = defaultdict(set)
    for edge in edges:
        left, right = _route_nodes(edge)
        snapshot_nodes[edge.timestamp].update((left, right))
    snapshot_counts = [
        len({_coverage_label(community, node, timestamp) for node in nodes_at_time})
        for timestamp, nodes_at_time in sorted(snapshot_nodes.items())
    ]
    coverage_term = -coverage_weight * sum(
        1.0 / count for count in snapshot_counts
    ) / len(snapshot_counts)
    terms = dict(log_weight=weight_term, strength=strength_term,
                 degree=degree_term, coverage=coverage_term,
                 community_count=len(communities))
    if snapshot_counts:
        terms["snapshot_community_counts"] = snapshot_counts
        terms["coverage_snapshot_count"] = len(snapshot_counts)
    score = weight_term + strength_term + degree_term + coverage_term
    length_term = -length_weight * len(edges)
    score += length_term
    terms.update(length=length_term, path_length=len(edges),
                 score_mode=LINEAR_TEMPORAL_SCORE, path_score=score)
    return RankedPath(tuple(edges), score, terms)


def search_paths(edges, source, target, weights, community, *, max_hops=4,
                 feedback_k=5, exact_topology=None, allow_weights_above_one=False):
    """Exact top-five current-weight H4 search over fixed physical topology."""
    _require_snapshot_scoring(community)
    edges, source, target, weights = _validate_inputs(
        edges, source, target, weights, community,
        allow_weights_above_one=allow_weights_above_one)
    return _exact_topk(
        edges, source, target, weights, community,
        feedback_max_hops=_limit(max_hops, "max_hops"),
        k=_limit(feedback_k, "feedback_k"),
        strength_weight=1., degree_weight=1., coverage_weight=1.,
        exact_topology=exact_topology)
