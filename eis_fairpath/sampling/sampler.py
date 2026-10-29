"""Native fixed support, or forward-only transfer of unused admission quotas."""
from collections import deque
from dataclasses import asdict, replace
import json

import torch

from ..graph import RELATIONS
from .sampler_v1 import (
    HistoryIndex, _context_candidates, _digest, sample_query as native_sample_query,
)
from ..sampling.types import (
    NODE_TYPES, SampledQuery, SamplerConfig, SamplingLimitError, StateKey, support_digest,
)


def sample_query(history_or_index, query, *, dataset='', config=SamplerConfig(), quota_mode='fixed'):
    if quota_mode == 'fixed':
        return native_sample_query(history_or_index, query, dataset=dataset, config=config)
    if quota_mode != 'carry':
        raise ValueError(f'Unknown quota_mode: {quota_mode}')
    index = history_or_index if isinstance(history_or_index, HistoryIndex) else HistoryIndex(history_or_index, dataset=dataset)
    if dataset and index.dataset and dataset != index.dataset:
        raise ValueError('Dataset does not match HistoryIndex')
    dataset = dataset or index.dataset
    years = tuple(s.year for s in index.snapshots if s.year < query.time)[-config.history_window:]
    history_input_hash = index.window_input_hash(years)
    root_nodes = tuple(sorted((query.source, query.target)))
    root_key = json.dumps((dataset, history_input_hash, query.time, root_nodes, config.seed), separators=(',', ':'))
    if years:
        features = index.features_by_year[years[-1]]
    else:
        catalog = index.features_by_year[index.snapshots[0].year]
        features = {typ: torch.zeros_like(tensor) for typ, tensor in catalog.items()}
    states, depths, positions = [], [], {}
    occurrence_count = 0

    def add_state(state, depth):
        nonlocal occurrence_count
        if state not in positions:
            if state.node_type not in NODE_TYPES or not 0 <= state.node_id < len(features[state.node_type]):
                raise ValueError('Query/state node is outside fixed identity catalog')
            occurrence_count += config.layers-depth+1
            if occurrence_count > config.occurrence_cap:
                raise SamplingLimitError(f'occurrence_cap exceeded for {query.query_id}: {occurrence_count}>{config.occurrence_cap}')
            positions[state] = len(states)
            states.append(state)
            depths.append(depth)
        return positions[state]

    roots = tuple(add_state(StateKey(typ, node, query.time), 0) for typ, node in root_nodes)
    frontier = list(roots)
    admitted, admission_counts = {}, [0]*config.layers
    routes, seen_routes = [], set()
    route_layer_count, carry = 0, 0
    effective_quotas, unused_quotas = [], []
    for depth in range(1, config.layers+1):
        quota = config.depth_quotas[depth-1]+carry
        effective_quotas.append(quota)
        pending = deque((target, iter(_context_candidates(index, states[target], years, config, root_key)))
                        for target in sorted(frontier, key=lambda i: (_digest(f'{root_key}|context|{states[i]}'), states[i])))
        next_frontier = set()
        while pending:
            target, iterator = pending.popleft()
            candidate = next(iterator, None)
            if candidate is None:
                continue
            pending.append((target, iterator))
            relation, year, source_node, event_id = candidate
            if year >= query.time or year > states[target].cutoff:
                raise AssertionError('Internal sampler causal cutoff violation')
            if event_id not in admitted:
                if len(admitted) >= config.event_budget or admission_counts[depth-1] >= quota:
                    continue
                admitted[event_id] = depth
                admission_counts[depth-1] += 1
            source_state = StateKey(relation[0], source_node, year)
            is_new = source_state not in positions
            source = add_state(source_state, depth)
            route = source, target, event_id, RELATIONS.index(relation), states[target].cutoff-year
            if route not in seen_routes:
                seen_routes.add(route)
                routes.append(route)
                route_layer_count += config.layers-depths[target]
                if route_layer_count > config.route_cap:
                    raise SamplingLimitError(f'route_cap exceeded for {query.query_id}: {route_layer_count}>{config.route_cap}')
            if is_new and depth < config.layers:
                next_frontier.add(source)
        frontier = list(next_frontier)
        carry = quota-admission_counts[depth-1]
        unused_quotas.append(carry)
        # Even an empty frontier needs its remaining effective config recorded.
    effective_config = replace(config, depth_quotas=tuple(effective_quotas))
    event_ids = tuple(sorted(admitted))
    event_positions = {event: i for i, event in enumerate(event_ids)}
    routes.sort()
    long = lambda values: torch.tensor(values, dtype=torch.long)
    metadata = {
        'dataset': dataset, 'query_id': query.query_id, 'query_time': query.time,
        'canonical_root_nodes': root_nodes, 'history_years': list(years), 'history_input_hash': history_input_hash,
        'admission_counts_by_depth': admission_counts, 'admission_depth': dict(sorted(admitted.items())),
        'event_years': {event: index.events[event][0] for event in event_ids},
        'num_physical_events': len(event_ids), 'num_states': len(states), 'num_occurrences': occurrence_count,
        'num_routes': len(routes), 'num_route_layers': route_layer_count,
        'node_type_order': list(NODE_TYPES), 'relation_order': [list(relation) for relation in RELATIONS],
        'sampler_config': asdict(effective_config), 'quota_mode': quota_mode,
        'base_depth_quotas': list(config.depth_quotas), 'effective_depth_quotas': effective_quotas,
        'unused_quota_by_depth': unused_quotas,
    }
    sample = SampledQuery(tuple(states), roots, event_ids,
        torch.stack([features[state.node_type][state.node_id] for state in states]),
        long([NODE_TYPES.index(state.node_type) for state in states]), long(depths),
        long([r[0] for r in routes]), long([r[1] for r in routes]),
        long([event_positions[r[2]] for r in routes]), long([r[3] for r in routes]),
        torch.tensor([r[4] for r in routes], dtype=torch.float32), effective_config, metadata, '')
    sample.support_hash = support_digest(sample)
    return sample
