from types import SimpleNamespace

import pytest
import torch


def test_equal_role_kl_matches_closed_form_and_raw_mask_gradient():
    from eis_fairpath.losses import MeanRoleKL
    ids = ['a', 'b', 'c', 'd', 'e']
    groups = dict(zip(ids, ['internal', 'internal', 'cross', 'cross', 'cross']))
    objective = MeanRoleKL(ids, groups, dict(zip(ids, [1., 3., 0., 2., 4.])))
    raw = torch.tensor([-.3, .7, 1., -.5, .2], requires_grad=True)
    actual = objective.prediction_loss(raw)
    p1 = torch.tensor([.25, .75], dtype=torch.float64)
    p2 = torch.tensor([0., 1./3., 2./3.], dtype=torch.float64)
    expected = .5 * ((p1 * (p1.log() - raw[:2].double().log_softmax(0))).sum()
        + (p2[1:] * (p2[1:].log() - raw[2:].double().log_softmax(0)[1:])).sum())
    assert actual.item() == pytest.approx(expected.item(), abs=1e-14)
    actual.backward()
    target = torch.cat([p1, p2])
    expected_gradient = .5 * (torch.cat([raw[:2].double().softmax(0), raw[2:].double().softmax(0)]) - target)
    assert torch.allclose(raw.grad.double(), expected_gradient, atol=2e-8, rtol=0)


def test_constant_role_disables_entire_kl_with_connected_zero():
    from eis_fairpath.losses import MeanRoleKL
    ids = ['a', 'b', 'c', 'd']
    objective = MeanRoleKL(ids, dict(zip(ids, ['internal']*2 + ['cross']*2)),
        dict(zip(ids, [1., 1., 1., 3.])))
    raw = torch.tensor([.2, -.3, 1., 2.], requires_grad=True)
    value = objective.prediction_loss(raw)
    value.backward()
    assert value.item() == 0.
    assert torch.equal(raw.grad, torch.zeros_like(raw))


def test_native_path_loss_averages_occurrences_and_unique_off_complement():
    from eis_fairpath.losses import author_path_loss
    weights = torch.tensor([.1, .3, .6, .8], dtype=torch.float64, requires_grad=True)
    terms = author_path_loss(weights, ['a', 'b', 'c', 'd'],
        [SimpleNamespace(event_ids=('a', 'b')), SimpleNamespace(event_ids=('a', 'c'))])
    assert terms['on'].item() == pytest.approx(-.275)
    assert terms['off'].item() == pytest.approx(.8)
    assert terms['path'].item() == pytest.approx(1.15)
    terms['path'].backward()
    assert weights.grad.tolist() == pytest.approx([-1., -.5, -.5, 2.])


def test_path_hinge_has_zero_subgradient_at_boundary():
    from eis_fairpath.losses import author_path_loss
    values = torch.tensor([.5, .4], dtype=torch.float64, requires_grad=True)
    terms = author_path_loss(values, ['a', 'b'], [SimpleNamespace(event_ids=('a',))])
    assert terms['hinge'].item() == 0.
    terms['hinge'].backward()
    assert values.grad.tolist() == [0., 0.]
