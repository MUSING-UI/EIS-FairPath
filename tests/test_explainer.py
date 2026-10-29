import json
from copy import deepcopy

import pytest
import torch


def test_exact_search_preserves_temporal_physical_events_and_current_weights():
    from eis_fairpath.history import EdgeRef
    from eis_fairpath.native import search
    from eis_fairpath.native.types import CommunityView
    # Two distinct physical two-hop routes, including nonmonotone timestamps.
    edges = [EdgeRef(0, 0, ('author', 'writes', 'paper'), 2007, 'a'),
        EdgeRef(1, 0, ('author', 'writes', 'paper'), 2006, 'b'),
        EdgeRef(0, 1, ('author', 'writes', 'paper'), 2006, 'c'),
        EdgeRef(1, 1, ('author', 'writes', 'paper'), 2007, 'd')]
    nodes = [('author', 0), ('author', 1), ('paper', 0), ('paper', 1)]
    labels = {node: 0 for node in nodes}
    view = CommunityView(labels, {n: 0. for n in nodes}, {n: 1. for n in nodes}, {}, {})
    view.path_degrees_by_timestamp = {y: {n: 1. for n in nodes} for y in (2006, 2007)}
    view.path_strengths_by_timestamp = {y: {n: 0. for n in nodes} for y in (2006, 2007)}
    view.path_community_labels_by_timestamp = {y: labels for y in (2006, 2007)}
    topology = search.prepare_exact_topology(edges, ('author', 0), ('author', 1))
    one = search.search_paths(edges, ('author', 0), ('author', 1),
        dict(a=.9, b=.9, c=.1, d=.1), view, exact_topology=topology)
    two = search.search_paths(edges, ('author', 0), ('author', 1),
        dict(a=.1, b=.1, c=.9, d=.9), view, exact_topology=topology)
    assert one.paths[0].event_ids == ('a', 'b')
    assert two.paths[0].event_ids == ('c', 'd')
    assert one.stats['complete'] and not one.stats['truncated']


def test_real_teacher_runs_forty_updates_and_forty_one_fresh_searches(tmp_path, monkeypatch):
    from eis_fairpath.graph import load_graph
    from eis_fairpath.models.structure import StructureAwareHGTTeacher
    from eis_fairpath.explainer import prepare_query, explain
    torch.set_num_threads(1)
    torch.manual_seed(42)
    data = dict(dataset='cs_dblp_mining_ir', years=list(range(2003, 2009)), papers=[
        dict(id='pA', year=2007, authors=['a', 'b'], venue='shared'),
        dict(id='pB', year=2007, authors=['c', 'd'], venue='shared'),
        dict(id='pC', year=2007, authors=['a', 'c'], venue='bridge')])
    graph_file = tmp_path / 'graph.json'
    graph_file.write_text(json.dumps(data), encoding='utf8')
    graph = load_graph(graph_file)
    model = StructureAwareHGTTeacher(input_dim=16, hidden_dim=128, heads=2, time_dim=32,
        layers=4, role_mode='membership', head_mode='structure', head_hidden=64)
    with torch.no_grad():
        model.output_bias.fill_(3.)
    query = prepare_query(graph, model, ('b', 'd'), 2008, qid='tiny')
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    def forbidden(*args, **kwargs):
        raise AssertionError('KL_H4 must not construct BCE')
    monkeypatch.setattr(torch.nn.functional, 'binary_cross_entropy_with_logits', forbidden)
    result = explain(query, seed=17)
    assert result['optimizer_updates'] == 40
    assert result['search_calls'] == 41
    assert result['evidence']['source']['probe_count'] == 4
    assert result['teacher_forward_calls'] == 82
    assert len(result['trace']) == 40
    assert len(result['paths']) <= 3
    assert len(result['selected_events']) <= 12
    assert result['final_search']['raw_mask'] == result['final_M']
    for row in result['trace']:
        assert set(row['losses']) == {'prediction', 'path', 'total'}
        assert row['raw_mask'] != row['raw_mask_after']
        assert row['losses']['total'] == pytest.approx(row['losses']['prediction'] + row['losses']['path'])
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    assert all(p.grad is None for p in model.parameters())
    json.dumps(result, allow_nan=False)
    repeated = explain(query, seed=17, evidence=result['evidence'])
    assert repeated['final_M'] == result['final_M']


def test_cached_evidence_requires_query_teacher_support_and_four_probe_binding():
    from eis_fairpath.graph import ExperimentGraph
    from eis_fairpath.models.structure import StructureAwareHGTTeacher
    from eis_fairpath.explainer import prepare_query, explain
    torch.set_num_threads(1)
    torch.manual_seed(42)
    data = dict(dataset='tiny', years=list(range(2003, 2009)), papers=[
        dict(id='pA', year=2007, authors=['a', 'b'], venue='shared'),
        dict(id='pB', year=2007, authors=['c', 'd'], venue='shared'),
        dict(id='pC', year=2007, authors=['a', 'c'], venue='bridge')])
    graph = ExperimentGraph(data)
    model = StructureAwareHGTTeacher(input_dim=16, hidden_dim=8, heads=2,
        time_dim=4, layers=4, head_mode='structure', head_hidden=4)
    with torch.no_grad():
        model.output_bias.fill_(3.)
    first = prepare_query(graph, model, ('b', 'd'), 2008, qid='first')
    evidence = explain(first, seed=17)['evidence']
    second = prepare_query(graph, model, ('b', 'd'), 2008, qid='second')
    with pytest.raises(ValueError, match='Evidence'):
        explain(second, seed=17, evidence=evidence)
    expected = dict(query_id='first', teacher_state_hash=first.teacher.state_hash,
        support_hash=first.sample.support_hash, probe_count=4)
    for field, value in expected.items():
        assert evidence['source'][field] == value
        missing = deepcopy(evidence)
        missing['source'].pop(field)
        with pytest.raises(ValueError, match='Evidence'):
            explain(first, seed=17, evidence=missing)
        mismatched = deepcopy(evidence)
        mismatched['source'][field] = 16 if field == 'probe_count' else 'different'
        with pytest.raises(ValueError, match='Evidence'):
            explain(first, seed=17, evidence=mismatched)
