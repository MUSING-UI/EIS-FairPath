"""Public KL_H4 preparation and fresh per-query mask optimization."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import math
import time

import torch

from .evidence import build_evidence
from .history import history_edges
from .losses import MeanRoleKL, author_path_loss, mask_weights
from .models.adapter import FixedSupportTeacher
from .native.author_path import author_initialization, author_path_weights, author_strength_differences
from .native.community import (build_communities, build_temporal_degrees,
    build_temporal_strengths, build_temporal_coverage_labels)
from .native.search import prepare_exact_topology, search_paths, extract_paths
from .native.types import Query, QueryContext
from .sampling.sampler import sample_query
from .sampling.types import SamplerConfig, support_digest


DEFAULT_SAMPLER = SamplerConfig(event_budget=1024, depth_quotas=(128, 256, 320, 320))


def _model_hash(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        cpu = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(cpu.dtype).encode())
        digest.update(str(tuple(cpu.shape)).encode())
        digest.update(cpu.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class NativeSampleTeacher:
    """Frozen model with one sampled physical support for every intervention."""

    def __init__(self, model, sample):
        self.model = model.eval().requires_grad_(False)
        for parameter in model.parameters():
            parameter.grad = None
        self.fixed = FixedSupportTeacher(model, sample)
        self.state_hash = _model_hash(model)
        self.forward_calls = 0

    def predict(self, event_ids, gates, *, outside=1.):
        self.forward_calls += 1
        return self.fixed.masked_logit(event_ids, gates, outside=outside).reshape(1)

    def verify_frozen(self):
        if (_model_hash(self.model) != self.state_hash
                or any(p.grad is not None or p.requires_grad for p in self.model.parameters())):
            raise RuntimeError('The frozen teacher changed during explanation')


def bounded_context(graph, query, sample, *, community_seed=42, history_window=5):
    """Full historical community partition restricted to the sampled root DAG."""
    if support_digest(sample) != sample.support_hash:
        raise ValueError('Stale sampled support')
    history = graph.history(query.time, history_window, 'cpu')
    all_edges = history_edges(history)
    domain = set(sample.computation_event_ids)
    edges = [edge for edge in all_edges if edge.event_id in domain]
    if {edge.event_id for edge in edges} != domain:
        raise ValueError('Sampled computation events differ from query history')
    frozen = getattr(graph, 'community_assignments', None)
    assignments = frozen[max(s.year for s in history)] if frozen is not None else None
    community = build_communities(history, seed=community_seed, assignments=assignments)
    community.path_degrees_by_timestamp = build_temporal_degrees(edges)
    community.path_strengths_by_timestamp = build_temporal_strengths(edges, community.labels)
    community.path_community_labels_by_timestamp = build_temporal_coverage_labels(edges, community.labels)
    metadata = dict(history_years=[s.year for s in history], history_window=history_window,
        learning_domain='sampled root DAG', native_support_hash=sample.support_hash,
        learnable_event_count=len(edges), outside_event_count=len(all_edges)-len(edges),
        sampled_physical_event_count=len(sample.event_ids),
        path_strength_partition='fixed query community partition',
        path_degree_source='candidate events by timestamp', path_strength_source='candidate events by timestamp',
        path_coverage_source='candidate events by timestamp')
    return QueryContext(query, history, edges, all_edges, community, metadata)


@dataclass
class PreparedQuery:
    ctx: QueryContext
    sample: object
    teacher: NativeSampleTeacher
    full_logit: float
    full_score: float

    @property
    def event_ids(self):
        return self.ctx.event_ids

    @property
    def groups(self):
        labels = self.ctx.community.labels
        return {edge.event_id: 'internal' if labels[edge.relation[0], edge.src]
            == labels[edge.relation[2], edge.dst] else 'cross' for edge in self.ctx.edges}

    def predict(self, gates, *, outside=1.):
        if len(gates) != len(self.event_ids):
            raise ValueError('A full ordered physical-domain gate vector is required')
        return self.teacher.predict(self.event_ids, gates, outside=outside)


def prepare_query(graph, model, author_pair, year, qid=None, sampler_config=None):
    """Prepare external author IDs (or internal indices) without labels or paths."""
    if len(author_pair) != 2 or type(year) is not int:
        raise ValueError('An author pair and integer target year are required')
    config = DEFAULT_SAMPLER if sampler_config is None else sampler_config
    if isinstance(config, dict):
        config = dict(config)
        # Current native completion and identity policies are fixed by sampling.
        if config.pop('completion_mode', 'native') != 'native' or config.pop('identity_mode', 'preserve') != 'preserve':
            raise ValueError('Native completion with preserved identity is required')
        config = SamplerConfig(**config)
    if not isinstance(config, SamplerConfig) or config.layers != 4 or config.history_window != 5:
        raise ValueError('KL_H4 requires a four-layer sampler and five-year history')
    if year not in graph.years or not set(range(year-5, year)) <= set(graph.years):
        raise ValueError('Target year requires the complete five preceding calendar years')
    catalog = graph.node_index['author']
    endpoints = []
    for author in author_pair:
        if isinstance(author, str):
            if author not in catalog:
                raise ValueError(f'Unknown external author ID: {author}')
            endpoints.append(catalog[author])
        elif type(author) is int and 0 <= author < len(catalog):
            endpoints.append(author)
        else:
            raise ValueError('Author identifiers must be known external strings or internal indices')
    src, dst = sorted(endpoints)
    if qid is None:
        qid = f'{graph.dataset}|{year}|{src}|{dst}'
    query = Query(qid, src, dst, year, relation='rule_link')
    history = graph.history(year, 5, 'cpu')
    sample = sample_query(history, query, dataset=graph.dataset, config=config, quota_mode='fixed')
    parameter = next(model.parameters())
    if parameter.device.type != 'cpu':
        raise ValueError('The publication numerical kernel uses a CPU teacher')
    if sample.features.dtype != parameter.dtype or sample.delta_time.dtype != parameter.dtype:
        sample = replace(sample, features=sample.features.to(parameter.dtype),
            delta_time=sample.delta_time.to(parameter.dtype), support_hash='')
        sample.support_hash = support_digest(sample)
    ctx = bounded_context(graph, query, sample)
    teacher = NativeSampleTeacher(model, sample)
    with torch.no_grad():
        full = teacher.predict(ctx.event_ids, sample.features.new_ones(len(ctx.edges)), outside=1.)
    if not bool(torch.isfinite(full).all()):
        raise FloatingPointError('Nonfinite original teacher logit')
    return PreparedQuery(ctx, sample, teacher, float(full.item()), float(full.sigmoid().item()))


def explain(prepared, seed=42, evidence=None):
    """Run 40 fresh Adam updates and 41 exact searches using current mask M.

    Evidence is fixed across mask seeds. Supplying the prior result's evidence
    reuses its four probes on this same ordered event/role domain.
    """
    if type(seed) is not int:
        raise ValueError('Mask seed must be an integer')
    started = time.perf_counter()
    ctx, ids, groups = prepared.ctx, prepared.event_ids, prepared.groups
    if not ids:
        raise ValueError('A nonempty sampled physical event domain is required')
    if prepared.full_score <= .5:
        raise ValueError('KL_H4 explains positive teacher predictions (original score must exceed .5)')
    prepared.teacher.verify_frozen()
    if support_digest(prepared.sample) != prepared.sample.support_hash:
        raise ValueError('Sampled support changed after preparation')
    source_binding = dict(query_id=ctx.query.query_id,
        teacher_state_hash=prepared.teacher.state_hash,
        support_hash=prepared.sample.support_hash, probe_count=4)
    if evidence is None:
        evidence = build_evidence(ids, ctx.query.query_id, prepared.predict)
        evidence['groups'] = dict(groups)
        evidence['source'].update(source_binding)
    else:
        source = evidence.get('source') if isinstance(evidence, dict) else None
        if (not isinstance(source, dict) or evidence.get('event_ids') != ids
                or evidence.get('groups') != groups or source.get('channel') != 'RANDOM_MAGNITUDE'
                or type(source.get('probe_count')) is not int
                or any(source.get(key) != value for key, value in source_binding.items())):
            raise ValueError('Evidence must match the query, frozen teacher, support, ordered role domain, and four probes')
    objective = MeanRoleKL(ids, groups, evidence['scores'])
    raw, initialization = author_initialization(ctx, seed=seed, device='cpu')
    initial = raw.detach().cpu().tolist()
    differences = raw.new_tensor([abs(ctx.community.strengths[e.relation[0], e.src]
        - ctx.community.strengths[e.relation[2], e.dst]) for e in ctx.edges])
    path_differences = raw.new_tensor(author_strength_differences(ctx.edges, ctx.community))
    topology = prepare_exact_topology(ctx.edges, ctx.query.source, ctx.query.target)
    optimizer = torch.optim.Adam([raw], lr=.2)
    diagnostic_calls_before = prepared.teacher.forward_calls
    trace = []

    def search(weights, step):
        found = search_paths(ctx.edges, ctx.query.source, ctx.query.target,
            dict(zip(ids, weights.detach().cpu().tolist())), ctx.community,
            max_hops=4, feedback_k=5, exact_topology=topology, allow_weights_above_one=step < 40)
        return found, dict(step=step, raw_mask=raw.detach().cpu().tolist(),
            search_weights=weights.detach().cpu().tolist(), paths=[list(p.event_ids) for p in found.paths],
            search=dict(found.stats))

    for step in range(40):
        _, gates = mask_weights(raw, differences)
        weights = author_path_weights(raw, path_differences, theta=.5)
        found, row = search(weights, step)
        paths = found.paths[:5]
        # These two forwards report retention/complement; no prediction BCE is constructed.
        with torch.no_grad():
            kept = prepared.predict(gates, outside=1.)
            complement = prepared.predict(1.-gates, outside=0.)
        terms = author_path_loss(weights, ids, paths)
        prediction = objective.prediction_loss(raw)
        total = terms['path'] + prediction
        if not bool(torch.isfinite(total)):
            raise FloatingPointError('Nonfinite KL_H4 objective')
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        if raw.grad is None or not bool(torch.isfinite(raw.grad).all()):
            raise FloatingPointError('Finite full-domain mask gradient required')
        row.update(losses=dict(prediction=float(prediction.detach()), path=float(terms['path'].detach()),
            total=float(total.detach())), path_terms={k: float(v.detach()) for k, v in terms.items()},
            gradient=raw.grad.detach().cpu().tolist(), soft_keep_logit=float(kept.item()),
            soft_complement_logit=float(complement.item()))
        optimizer.step()
        if not bool(torch.isfinite(raw).all()):
            raise FloatingPointError('Nonfinite optimized mask')
        row['raw_mask_after'] = raw.detach().cpu().tolist()
        trace.append(row)
    with torch.no_grad():
        _, gates = mask_weights(raw, differences)
        found, final = search(raw.sigmoid(), 40)
        selected = extract_paths(found.paths, event_budget=15, max_paths=3, output_max_hops=4)
        kept = prepared.predict(gates, outside=1.)
        complement = prepared.predict(1.-gates, outside=0.)
    prepared.teacher.verify_frozen()
    paths = [list(path.event_ids) for path in selected]
    final_values = raw.detach().cpu().tolist()
    return dict(method='MEAN_PATH_V1', display_name='KL_H4', seed=seed, query=asdict(ctx.query),
        learnable_event_ids=ids, groups=groups, initial_M=initial, initialization=initialization,
        raw_logits=dict(zip(ids, final_values)), final_M=final_values,
        scores=dict(zip(ids, raw.detach().sigmoid().cpu().tolist())),
        prediction_gates=dict(zip(ids, gates.cpu().tolist())),
        paths=paths, selected_events=sorted({event for path in paths for event in path}),
        path_scores=[path.score for path in selected], output_type='paths' if paths else 'no_path',
        full_logit=prepared.full_logit, full_score=prepared.full_score,
        soft_keep_logit=float(kept.item()), soft_complement_logit=float(complement.item()),
        optimizer_updates=40, search_calls=41,
        teacher_forward_calls=prepared.teacher.forward_calls-diagnostic_calls_before,
        evidence=evidence, prediction_objective=objective.metadata(), trace=trace, final_search=final,
        effective_config=dict(lr=.2, updates=40, max_hops=4, feedback_paths=5,
            max_output_paths=3, event_budget=15, theta=.5, delta=1., path_margin=.1),
        teacher_state_hash=prepared.teacher.state_hash, support_hash=prepared.sample.support_hash,
        wall_seconds=time.perf_counter()-started)
