"""Deterministic finite physical-event input without reference or score access."""
from collections import defaultdict, deque
from dataclasses import asdict
import hashlib
import json

import torch

from ..graph import RELATIONS
from .types import NODE_TYPES, SampledQuery, SamplerConfig, SamplingLimitError, StateKey, support_digest


def _digest(value):
    return hashlib.sha256(value.encode()).digest()


class HistoryIndex:
    """Reusable CPU incoming routes; the model never receives this full index."""

    def __init__(self, history, *, dataset=""):
        self.dataset = dataset
        self.snapshots = tuple(sorted(history, key=lambda snap: snap.year))
        if not self.snapshots:
            raise ValueError("HistoryIndex requires at least one snapshot")
        if len({s.year for s in self.snapshots}) != len(self.snapshots):
            raise ValueError("HistoryIndex snapshot years must be unique")
        self.features_by_year = {}
        self.incoming = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        self.events = {}
        self._ordered = {}
        self._snapshot_hashes = {}
        self._window_hashes = {}
        feature_dim = None
        for snap in self.snapshots:
            h = hashlib.sha256()
            if set(snap.x_dict) != set(NODE_TYPES):
                raise ValueError("Snapshot must contain author, paper and venue features")
            features = {typ: snap.x_dict[typ].detach().cpu().contiguous() for typ in NODE_TYPES}
            for typ, tensor in features.items():
                if tensor.ndim != 2:
                    raise ValueError("Snapshot features must be two-dimensional")
                feature_dim = tensor.shape[1] if feature_dim is None else feature_dim
                if tensor.shape[1] != feature_dim:
                    raise ValueError("All node types need the same fixed feature dimension")
                h.update(str((snap.year, typ, tuple(tensor.shape))).encode())
                h.update(tensor.numpy().tobytes())
            self.features_by_year[snap.year] = features
            if len(set(snap.event_ids)) != len(snap.event_ids):
                raise ValueError("Snapshot physical event IDs must be unique")
            if set(snap.edge_index_dict) != set(snap.edge_event_index):
                raise ValueError("Snapshot routes and physical event maps must agree")
            seen_events, seen_routes = set(), set()
            for relation, tensor in snap.edge_index_dict.items():
                if relation not in RELATIONS:
                    raise ValueError("Unsupported typed relation")
                event_map = snap.edge_event_index[relation]
                if tensor.ndim != 2 or tensor.shape[0] != 2 or event_map.ndim != 1 or tensor.shape[1] != len(event_map):
                    raise ValueError("Malformed routes or physical event map")
                for (src, dst), index in zip(tensor.detach().cpu().T.tolist(), event_map.detach().cpu().tolist()):
                    if type(index) is not int or not 0 <= index < len(snap.event_ids):
                        raise ValueError("Route physical event index is outside event catalog")
                    if not (0 <= src < len(features[relation[0]]) and 0 <= dst < len(features[relation[2]])):
                        raise ValueError("Route node index is outside fixed feature catalog")
                    event_id = snap.event_ids[index]
                    seen_events.add(index)
                    canonical = self._canonical(relation, src, dst, snap.year)
                    if event_id in self.events and self.events[event_id] != canonical:
                        raise ValueError("Conflicting aliases for a physical event ID")
                    self.events[event_id] = canonical
                    route = (relation, src, dst, event_id)
                    if route not in seen_routes:
                        seen_routes.add(route)
                        self.incoming[(relation[2], dst)][relation][snap.year].append((src, event_id))
            if len(seen_events) != len(snap.event_ids):
                raise ValueError("Every physical event needs at least one route")
            physical = sorted((event, self.events[event]) for event in snap.event_ids)
            h.update(json.dumps((physical, sorted(seen_routes)), separators=(",", ":")).encode())
            self._snapshot_hashes[snap.year] = h.hexdigest()
        self.input_hash = self.window_input_hash(tuple(s.year for s in self.snapshots))
        self.feature_dim = feature_dim

    def window_input_hash(self, years):
        """Hash only legal-window features/events/routes, before sampling order."""
        years = tuple(sorted(set(years)))
        if any(year not in self._snapshot_hashes for year in years):
            raise ValueError("Window includes a year outside HistoryIndex")
        if years not in self._window_hashes:
            h = hashlib.sha256(self.dataset.encode())
            for year in years:
                h.update(f"{year}|{self._snapshot_hashes[year]}".encode())
            self._window_hashes[years] = h.hexdigest()
        return self._window_hashes[years]

    @staticmethod
    def _canonical(relation, src, dst, year):
        if relation[1] == "written_by":
            return year, RELATIONS[0], dst, src
        if relation[1] == "publishes":
            return year, RELATIONS[2], dst, src
        if relation[1] == "collaborates":
            src, dst = sorted((src, dst))
        return year, relation, src, dst

    def ordered_bucket(self, node, relation, year, seed):
        """Hash order is indexed once per seed, rather than sorting per query."""
        key = node, relation, year, seed
        if key not in self._ordered:
            values = self.incoming.get(node, {}).get(relation, {}).get(year, ())
            self._ordered[key] = tuple(sorted(values, key=lambda item: (_digest(f"{seed}|{item[1]}"), item)))
        return self._ordered[key]


def _context_candidates(index, state, years, config, root_key):
    """Round-robin typed relations and event years, with a per-relation fanout."""
    node = state.node_type, state.node_id
    available = index.incoming.get(node, {})
    relations = sorted(available, key=lambda relation: (_digest(f"{root_key}|relation|{relation}"), relation))
    streams = deque()
    for relation in relations:
        eligible = [year for year in available[relation] if year in years and year <= state.cutoff]
        eligible.sort(key=lambda year: (_digest(f"{root_key}|year|{year}"), year))
        buckets = deque((year, iter(index.ordered_bucket(node, relation, year, config.seed))) for year in eligible)

        def per_relation(buckets=buckets, relation=relation):
            count = 0
            while buckets and count < config.fanout:
                year, iterator = buckets.popleft()
                value = next(iterator, None)
                if value is None:
                    continue
                buckets.append((year, iterator))
                count += 1
                yield relation, year, value[0], value[1]

        streams.append(iter(per_relation()))
    while streams:
        iterator = streams.popleft()
        candidate = next(iterator, None)
        if candidate is not None:
            streams.append(iterator)
            yield candidate


def sample_query(history_or_index, query, *, dataset="", config=SamplerConfig()):
    """Sample two unordered roots with shared physical budget and causal cutoffs."""
    index = history_or_index if isinstance(history_or_index, HistoryIndex) else HistoryIndex(history_or_index, dataset=dataset)
    if dataset and index.dataset and dataset != index.dataset:
        raise ValueError("Dataset does not match HistoryIndex")
    dataset = dataset or index.dataset
    years = tuple(s.year for s in index.snapshots if s.year < query.time)[-config.history_window:]
    history_input_hash = index.window_input_hash(years)
    root_nodes = tuple(sorted((query.source, query.target)))
    root_key = json.dumps((dataset, history_input_hash, query.time, root_nodes, config.seed), separators=(",", ":"))
    if not years:
        # Catalog dimensions permit endpoint indexing, but future values cannot
        # become model inputs when there is no legal historical snapshot.
        catalog = index.features_by_year[index.snapshots[0].year]
        features = {typ: torch.zeros_like(tensor) for typ, tensor in catalog.items()}
    else:
        features = index.features_by_year[years[-1]]
    states, depths, positions = [], [], {}
    occurrence_count = 0

    def add_state(state, depth):
        nonlocal occurrence_count
        if state not in positions:
            if state.node_type not in NODE_TYPES or not 0 <= state.node_id < len(features[state.node_type]):
                raise ValueError("Query/state node is outside fixed identity catalog")
            occurrence_count += config.layers - depth + 1
            if occurrence_count > config.occurrence_cap:
                raise SamplingLimitError(f"occurrence_cap exceeded for {query.query_id}: {occurrence_count}>{config.occurrence_cap}")
            positions[state] = len(states)
            states.append(state)
            depths.append(depth)
        return positions[state]

    roots = tuple(add_state(StateKey(typ, node, query.time), 0) for typ, node in root_nodes)
    frontier = list(roots)
    admitted, admission_counts = {}, [0] * config.layers
    routes, seen_routes = [], set()
    route_layer_count = 0
    for depth in range(1, config.layers + 1):
        pending = deque((target, iter(_context_candidates(index, states[target], years, config, root_key)))
                        for target in sorted(frontier, key=lambda i: (_digest(f"{root_key}|context|{states[i]}"), states[i])))
        next_frontier = set()
        while pending:
            target, iterator = pending.popleft()
            candidate = next(iterator, None)
            if candidate is None:
                continue
            pending.append((target, iterator))
            relation, year, source_node, event_id = candidate
            if year >= query.time or year > states[target].cutoff:
                raise AssertionError("Internal sampler causal cutoff violation")
            if event_id not in admitted:
                if len(admitted) >= config.event_budget or admission_counts[depth - 1] >= config.depth_quotas[depth - 1]:
                    continue
                admitted[event_id] = depth
                admission_counts[depth - 1] += 1
            source_state = StateKey(relation[0], source_node, year)
            is_new = source_state not in positions
            source = add_state(source_state, depth)
            route = source, target, event_id, RELATIONS.index(relation), states[target].cutoff - year
            if route not in seen_routes:
                seen_routes.add(route)
                routes.append(route)
                route_layer_count += config.layers - depths[target]
                if route_layer_count > config.route_cap:
                    raise SamplingLimitError(f"route_cap exceeded for {query.query_id}: {route_layer_count}>{config.route_cap}")
            if is_new and depth < config.layers:
                next_frontier.add(source)
        frontier = list(next_frontier)
        if not frontier:
            break
    event_ids = tuple(sorted(admitted))
    event_positions = {event: i for i, event in enumerate(event_ids)}
    routes.sort()
    long = lambda values: torch.tensor(values, dtype=torch.long)
    metadata = {
        "dataset": dataset, "query_id": query.query_id, "query_time": query.time,
        "canonical_root_nodes": root_nodes, "history_years": list(years), "history_input_hash": history_input_hash,
        "admission_counts_by_depth": admission_counts, "admission_depth": dict(sorted(admitted.items())),
        "event_years": {event: index.events[event][0] for event in event_ids},
        "num_physical_events": len(event_ids), "num_states": len(states), "num_occurrences": occurrence_count,
        "num_routes": len(routes), "num_route_layers": route_layer_count,
        "node_type_order": list(NODE_TYPES), "relation_order": [list(relation) for relation in RELATIONS],
        "sampler_config": asdict(config),
    }
    sample = SampledQuery(tuple(states), roots, event_ids,
                          torch.stack([features[state.node_type][state.node_id] for state in states]),
                          long([NODE_TYPES.index(state.node_type) for state in states]), long(depths),
                          long([r[0] for r in routes]), long([r[1] for r in routes]),
                          long([event_positions[r[2]] for r in routes]), long([r[3] for r in routes]),
                          torch.tensor([r[4] for r in routes], dtype=torch.float32), config, metadata, "")
    sample.support_hash = support_digest(sample)
    return sample
