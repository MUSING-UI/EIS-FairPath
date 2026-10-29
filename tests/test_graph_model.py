"""Behavior checks for physical identity, causal support, and teacher gradients."""
import json
import gzip
import pytest
import torch


def tiny_data():
    return {"dataset": "tiny", "years": list(range(2000, 2006)), "papers": [
        {"id": "p0", "year": 2000, "authors": ["a", "b"], "venue": "V"},
        {"id": "p1", "year": 2001, "authors": ["b", "c"], "venue": "V"},
        {"id": "future", "year": 2005, "authors": ["a", "c"], "venue": "V"}]}


def test_graph_physical_events_have_one_reverse_alias_identity():
    from eis_fairpath.graph import ExperimentGraph
    from eis_fairpath.history import history_edges
    graph = ExperimentGraph(tiny_data())
    assert {kind: len(ids) for kind, ids in graph.node_index.items()} == {"author": 3, "paper": 3, "venue": 1}
    snap = graph.snapshot(2000)
    assert snap.num_events == 4
    assert sum(routes.shape[1] for routes in snap.edge_index_dict.values()) == 8
    assert len(history_edges([snap, snap])) == 4
    assert len(history_edges(graph.history(2005))) == 8
    assert not graph.empty_snapshot(2000).active["paper"][graph.node_index["paper"]["future"]]


def support():
    from eis_fairpath.graph import ExperimentGraph
    from eis_fairpath.sampling import sample_pair
    graph = ExperimentGraph(tiny_data())
    return sample_pair(graph.history(2005), (0, 2), 2005, dataset="tiny")


def teacher():
    from eis_fairpath.models import StructureAwareHGTTeacher
    torch.manual_seed(42)
    return StructureAwareHGTTeacher(input_dim=16, hidden_dim=8, heads=2, time_dim=4, layers=4,
                                    head_mode="structure", head_hidden=4)


def test_sampling_excludes_future_and_intervention_has_live_gradients():
    from eis_fairpath.models.adapter import FixedSupportTeacher
    sample = support()
    assert sample.metadata["history_years"] == [2000, 2001, 2002, 2003, 2004]
    assert set(sample.metadata["event_years"].values()) == {2000, 2001}
    assert len(sample.event_ids) == 8
    model = teacher().requires_grad_(False)
    fixed = FixedSupportTeacher(model, sample)
    weights = torch.full((len(fixed.domain_event_ids),), .5, requires_grad=True)
    score = fixed.masked_logit(fixed.domain_event_ids, weights, outside=1.)
    gradient, = torch.autograd.grad(score, weights)
    assert torch.isfinite(gradient).all()
    assert gradient.abs().max() > 1e-8
    one = fixed.masked_logit(fixed.domain_event_ids, torch.ones_like(weights), outside=1.)
    zero = fixed.masked_logit(fixed.domain_event_ids, torch.zeros_like(weights), outside=1.)
    assert abs(float(one-zero)) > 1e-6
    assert sample.support_hash == support().support_hash


def test_teacher_backbone_can_be_trained_and_rejects_tampered_support():
    sample, model = support(), teacher()
    torch.nn.functional.binary_cross_entropy_with_logits(model(sample), torch.ones(())).backward()
    assert any(p.grad is not None and bool(p.grad.abs().max() > 0) for p in model.convs.parameters())
    sample.features[0, 0] += 1
    with pytest.raises(ValueError, match="support"):
        model(sample)


def test_dblp_builder_preserves_complete_matching_records(tmp_path):
    from eis_fairpath.datasets import build_dblp, write_dataset
    source = tmp_path / "raw.json.gz"
    rows = [["conf/acl/a", [" Alice ", "BOB", "Alice"], 2000],
            ["conf/acl/b", ["Carol"], 2001], ["conf/cvpr/x", ["Dan"], 2001]]
    with gzip.open(source, "wt", encoding="utf8") as stream:
        stream.write("[\n" + ",\n".join(json.dumps(row) for row in rows) + "\n]\n")
    data = build_dblp(source, "dblp_nlp", verify=False)
    assert [p["id"] for p in data["papers"]] == ["dblp:conf/acl/a", "dblp:conf/acl/b"]
    assert len(data["papers"][0]["authors"]) == 2
    destination = write_dataset(data, tmp_path / "processed.json")
    assert json.loads(destination.read_text(encoding="utf8")) == data
    with pytest.raises(ValueError, match="SHA-256"):
        build_dblp(source, "dblp_nlp")
