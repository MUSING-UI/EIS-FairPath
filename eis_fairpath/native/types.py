"""Typed query and temporal community records."""

from dataclasses import dataclass


Node = tuple[str, int]


@dataclass(frozen=True)
class Query:
    query_id: str
    src: int
    dst: int
    time: int
    relation: str = "collaborates"

    def __post_init__(self):
        if not isinstance(self.query_id, str) or not self.query_id:
            raise ValueError("query_id must be a nonempty string")
        for name in ("src", "dst", "time"):
            value = getattr(self, name)
            if type(value) is not int:
                raise ValueError(f"{name} must be an integer")
        if self.src < 0 or self.dst < 0:
            raise ValueError("query endpoints must be nonnegative")
        if self.src == self.dst:
            raise ValueError("query endpoints must be distinct")
        if not isinstance(self.relation, str) or not self.relation:
            raise ValueError("relation must be a nonempty string")

    @property
    def source(self) -> Node:
        return "author", self.src

    @property
    def target(self) -> Node:
        return "author", self.dst


@dataclass
class CommunityView:
    labels: dict[Node, int]
    strengths: dict[Node, float]
    degrees: dict[Node, float]
    members: dict[int, frozenset[Node]]
    metadata: dict
    # Path scoring uses a separate temporal incidence degree map.  The legacy
    # ``degrees`` field remains the historical community statistic.
    path_degrees_by_timestamp: dict[int, dict[Node, float]] | None = None
    # Path scoring uses snapshot-local strengths with the query's fixed labels.
    # ``strengths`` remains the historical community statistic for compatibility.
    path_strengths_by_timestamp: dict[int, dict[Node, float]] | None = None
    # Fixed query labels restricted to the nodes present in each path snapshot.
    path_community_labels_by_timestamp: dict[int, dict[Node, int]] | None = None


@dataclass
class QueryContext:
    query: Query
    history: list
    edges: list
    all_edges: list
    community: CommunityView
    metadata: dict

    @property
    def event_ids(self) -> list[str]:
        return [edge.event_id for edge in self.edges]
