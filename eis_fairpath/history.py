"""Physical historical-event utilities for generating fixed rule-reference data.

This preserves the existing rule-data neighborhood contract. It contains no
learned masks, explainer path ranking, or explanation selection.
"""
from collections import defaultdict, deque
from dataclasses import dataclass
from numbers import Integral


_REVERSE = {
    ("author", "writes", "paper"): ("paper", "written_by", "author"),
    ("paper", "written_by", "author"): ("author", "writes", "paper"),
    ("paper", "published_in", "venue"): ("venue", "publishes", "paper"),
    ("venue", "publishes", "paper"): ("paper", "published_in", "venue"),
    ("author", "collaborates", "author"): ("author", "collaborates", "author"),
}


@dataclass(frozen=True)
class EdgeRef:
    src: int
    dst: int
    relation: tuple[str, str, str]
    timestamp: int
    event_id: str

    def __post_init__(self):
        for name in ("src", "dst", "timestamp"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError(f"{name} must be an integer")
            if name != "timestamp" and value < 0:
                raise ValueError(f"{name} must be nonnegative")
            object.__setattr__(self, name, int(value))
        if not isinstance(self.relation, tuple) or self.relation not in _REVERSE:
            raise ValueError("relation must be a supported typed relation tuple")
        if not isinstance(self.event_id, str) or not self.event_id:
            raise ValueError("event_id must be a nonempty string")

    def as_dict(self):
        return {"src": self.src, "dst": self.dst, "relation": self.relation,
                "timestamp": self.timestamp, "event_id": self.event_id}


def _reverse(edge):
    return EdgeRef(edge.dst, edge.src, _REVERSE[edge.relation], edge.timestamp, edge.event_id)


def _canonical(edge):
    if edge.relation[1] in ("written_by", "publishes"):
        return _reverse(edge)
    if edge.relation[1] == "collaborates" and edge.src > edge.dst:
        return _reverse(edge)
    return edge


def _edge_key(edge):
    return (edge.timestamp, edge.relation, edge.src, edge.dst, edge.event_id)


def _unique_edges(edges):
    by_event = {}
    for edge in edges:
        if not isinstance(edge, EdgeRef):
            raise ValueError("edges must contain EdgeRef instances")
        edge = _canonical(edge)
        if edge.event_id in by_event and by_event[edge.event_id] != edge:
            raise ValueError(f"Conflicting references for physical event {edge.event_id}")
        by_event[edge.event_id] = edge
    return sorted(by_event.values(), key=_edge_key)


def history_edges(history):
    """Read Snapshot.edge_event_index; return one canonical edge per event ID.

    Repeated snapshots are harmless, but inconsistent IDs, missing physical
    event routes and malformed event-index maps raise rather than lose evidence.
    """
    references = []
    for snapshot in history:
        if len(set(snapshot.event_ids)) != len(snapshot.event_ids):
            raise ValueError("Snapshot event IDs must be unique")
        if set(snapshot.edge_index_dict) != set(snapshot.edge_event_index):
            raise ValueError("Snapshot edge routes and physical event maps must agree")
        seen = set()
        for relation, indices in snapshot.edge_index_dict.items():
            event_indices = snapshot.edge_event_index[relation]
            if indices.ndim != 2 or indices.shape[0] != 2 or event_indices.ndim != 1:
                raise ValueError("Expected [2, edges] routes and [edges] physical event indices")
            if indices.shape[1] != event_indices.shape[0]:
                raise ValueError("Each edge route needs exactly one physical event index")
            pairs = indices.detach().cpu().T.tolist()
            for (src, dst), index in zip(pairs, event_indices.detach().cpu().tolist()):
                if not isinstance(index, int) or index < 0 or index >= len(snapshot.event_ids):
                    raise ValueError("Physical event index is outside Snapshot.event_ids")
                seen.add(index)
                references.append(EdgeRef(src, dst, relation, snapshot.year, snapshot.event_ids[index]))
        if len(seen) != len(snapshot.event_ids):
            raise ValueError("Every physical event must have an edge route")
    return _unique_edges(references)


def _node(node):
    if (not isinstance(node, tuple) or len(node) != 2 or
            not isinstance(node[0], str) or not node[0] or
            isinstance(node[1], bool) or not isinstance(node[1], Integral) or node[1] < 0):
        raise ValueError("Endpoints must be (node_type, nonnegative_integer_index)")
    return node[0], int(node[1])


def _limit(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _adjacency(edges):
    adjacent = defaultdict(list)
    for edge in edges:
        adjacent[(edge.relation[0], edge.src)].append(edge)
        reverse = _reverse(edge)
        if reverse != edge:
            adjacent[(reverse.relation[0], reverse.src)].append(reverse)
    for routes in adjacent.values():
        routes.sort(key=_edge_key)
    return adjacent


def candidate_edges(edges, src, dst, hops=2, max_edges=256):
    """Expand both typed endpoints for at most ``hops``, stopping at the cap.

    Index construction scans all input edges; hop expansion itself retains at
    most ``max_edges`` physical events and never enumerates paths. Round-robin
    frontier expansion gives both query endpoints access to the finite budget.
    """
    src, dst = _node(src), _node(dst)
    _limit(hops, "hops")
    _limit(max_edges, "max_edges")
    canonical = _unique_edges(edges)
    if not hops or not max_edges:
        return []
    adjacent = _adjacency(canonical)
    selected, seen = {}, {src, dst}
    frontier = {src, dst}
    for _ in range(hops):
        pending = deque(iter(adjacent.get(node, ())) for node in sorted(frontier))
        next_frontier = set()
        while pending and len(selected) < max_edges:
            iterator = pending.popleft()
            edge = next(iterator, None)
            if edge is None:
                continue
            pending.append(iterator)
            selected[edge.event_id] = _canonical(edge)
            neighbor = edge.relation[2], edge.dst
            if neighbor not in seen:
                next_frontier.add(neighbor)
        if len(selected) >= max_edges or not next_frontier:
            break
        seen.update(next_frontier)
        frontier = next_frontier
    return sorted(selected.values(), key=_edge_key)
