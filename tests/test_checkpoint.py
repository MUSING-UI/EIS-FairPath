"""The portable loader must retain calibrated weights and reject wrong-domain use."""
import torch
import pytest

from eis_fairpath.checkpoints import load_teacher
from eis_fairpath.models import StructureAwareHGTTeacher


def checkpoint_file(tmp_path):
    architecture = dict(input_dim=16, hidden_dim=8, heads=2, time_dim=4,
                        layers=4, role_mode="membership", head_mode="structure", head_hidden=4)
    model = StructureAwareHGTTeacher(**architecture).double()
    with torch.no_grad():
        model.output_scale.fill_(1.75)
        model.output_bias.fill_(-0.3)
    path = tmp_path / "teacher.pt"
    torch.save(dict(format="eis-fairpath-teacher-v1", dataset="tiny", architecture=architecture,
                    sampler=dict(layers=4, event_budget=32, depth_quotas=[8, 8, 8, 8],
                                 fanout=4, history_window=5, seed=42), state_dict=model.state_dict()), path)
    return path, model


def test_load_preserves_calibration_dtype_and_freezes_teacher(tmp_path):
    path, original = checkpoint_file(tmp_path)
    loaded = load_teacher(path, dataset="tiny")
    assert not loaded.model.training
    assert all(not parameter.requires_grad for parameter in loaded.model.parameters())
    assert loaded.sampler.event_budget == 32
    for name, value in original.state_dict().items():
        assert torch.equal(loaded.model.state_dict()[name], value)
        assert loaded.model.state_dict()[name].dtype == value.dtype


def test_wrong_domain_checkpoint_is_rejected(tmp_path):
    path, _ = checkpoint_file(tmp_path)
    with pytest.raises(ValueError, match="dataset"):
        load_teacher(path, dataset="another_domain")
