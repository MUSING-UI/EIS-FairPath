"""Symmetric readouts over the unchanged event-gated native HGT support."""
import math

import torch
from torch import nn

from .count_aware import CountAwareHGTTeacher
from .teacher_v1 import NODE_TYPES


def _endpoint_reach(sample, event_mask, root):
    """Max-product reach through the actual layer DAG, with residual carry."""
    reach = event_mask.new_zeros(len(sample.states))
    reach[root] = 1.
    for layer in range(sample.config.layers, 0, -1):
        routes = sample.routes_for_layer(layer)
        source = sample.source_index[routes]
        target = sample.target_index[routes]
        if bool((sample.min_depth[source] > sample.config.layers - layer + 1).any()):
            raise ValueError('A route source has no valid previous-layer occurrence')
        candidates = reach[target] * event_mask[sample.event_index[routes]]
        reach = reach.scatter_reduce(0, source, candidates, reduce='amax', include_self=True)
    return reach


def pair_structure_features(sample, hidden, event_mask):
    """Pool same-occurrence common reach, once per non-endpoint physical node.

    This uses only the sampled occurrence DAG. Max-product reach is bounded by
    the native layer budget. Physical-event aliases read the same scalar gate;
    duplicate routes cannot inflate a maximum. A physical node's largest common
    reach supplies its weight, with the mean hidden state across tied aliases.
    All cutoff aliases of the two endpoints are excluded. Empty common support
    means zero features, rather than a claim that no legal graph path exists.
    """
    left = _endpoint_reach(sample, event_mask, sample.roots[0])
    right = _endpoint_reach(sample, event_mask, sample.roots[1])
    common = torch.minimum(left, right)
    endpoints = {(sample.states[index].node_type, sample.states[index].node_id) for index in sample.roots}
    physical_ids, physical_types, state_groups, state_indices = {}, [], [], []
    for index, state in enumerate(sample.states):
        key = (state.node_type, state.node_id)
        if key in endpoints:
            continue
        if key not in physical_ids:
            physical_ids[key] = len(physical_ids)
            physical_types.append(NODE_TYPES.index(state.node_type))
        state_groups.append(physical_ids[key])
        state_indices.append(index)
    width = hidden.shape[-1]
    if not state_indices:
        return hidden.new_zeros(len(NODE_TYPES) * width + len(NODE_TYPES))
    groups = torch.tensor(state_groups, dtype=torch.long, device=hidden.device)
    indices = torch.tensor(state_indices, dtype=torch.long, device=hidden.device)
    types = torch.tensor(physical_types, dtype=torch.long, device=hidden.device)
    weights = common[indices]
    physical_weights = weights.new_zeros(len(physical_ids)).scatter_reduce(
        0, groups, weights, reduce='amax', include_self=True)
    selected = ((weights == physical_weights[groups]) & (weights > 0)).to(hidden.dtype)
    tied = weights.new_zeros(len(physical_ids)).index_add(0, groups, selected)
    vectors = hidden.new_zeros((len(physical_ids), width)).index_add(
        0, groups, hidden[indices] * selected[:, None]) / tied.clamp_min(1)[:, None]
    counts = weights.new_zeros(len(NODE_TYPES)).index_add(0, types, physical_weights)
    sums = hidden.new_zeros((len(NODE_TYPES), width)).index_add(
        0, types, vectors * physical_weights[:, None])
    means = sums / counts.clamp_min(1)[:, None]
    return torch.cat((means.flatten(), counts.log1p()))


class StructureAwareHGTTeacher(CountAwareHGTTeacher):
    """Native CountAware HGT plus zero-initialized symmetric residual heads."""

    def __init__(self, *, head_mode='nonlinear', head_hidden=64, **architecture):
        if head_mode not in ('nonlinear', 'structure'):
            raise ValueError('head_mode must be nonlinear or structure')
        if type(head_hidden) is not int or head_hidden < 1:
            raise ValueError('head_hidden must be a positive integer')
        super().__init__(**architecture)
        self.head_mode, self.head_hidden = head_mode, head_hidden
        width = self.architecture['hidden_dim']
        self.residual_head = nn.Sequential(nn.Linear(3 * width, head_hidden), nn.GELU(), nn.Linear(head_hidden, 1))
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)
        if head_mode == 'structure':
            self.structure_head = nn.Sequential(nn.Linear(3 * width + 3, head_hidden, bias=False),
                nn.GELU(), nn.Linear(head_hidden, 1, bias=False))
            nn.init.zeros_(self.structure_head[-1].weight)
        self.register_buffer('output_scale', torch.tensor(1.))
        self.register_buffer('output_bias', torch.tensor(0.))

    @classmethod
    def from_base_state(cls, architecture, state, head_mode='nonlinear', head_hidden=64):
        """Require an exact CountAware backbone state; allow only new readouts."""
        model = cls(**architecture, head_mode=head_mode, head_hidden=head_hidden)
        current = model.state_dict()
        added = {key for key in current if key.startswith(('residual_head.', 'structure_head.'))}
        added.update(('output_scale', 'output_bias'))
        if set(state) != set(current) - added:
            raise ValueError('Base checkpoint must contain exactly the unchanged CountAware backbone')
        if any(state[key].shape != current[key].shape or state[key].dtype != current[key].dtype for key in state):
            raise ValueError('Base checkpoint tensor shape or dtype differs from the backbone')
        incompatible = model.load_state_dict(state, strict=False)
        if set(incompatible.missing_keys) != added or incompatible.unexpected_keys:
            raise ValueError('Base migration differs from the new readout additions')
        return model

    def feature_tensors(self, sample, event_mask=None):
        """Encode once; return differentiable features for live or frozen fits."""
        hidden = self.encode(sample, event_mask)
        left, right = hidden[sample.roots[0]], hidden[sample.roots[1]]
        base = (.5 * (self.decoder(left, right) + self.decoder(right, left))).squeeze(-1)
        endpoint = torch.cat((left + right, (left - right).abs(), left * right))
        if self.head_mode == 'structure':
            structure = pair_structure_features(sample, hidden, self._event_mask(sample, event_mask))
        else:
            structure = hidden.new_zeros(3 * self.architecture['hidden_dim'] + 3)
        return dict(base_logit=base, endpoint_features=endpoint, structure_features=structure)

    def forward_features(self, features):
        """The same complete readout for individual or stacked cached features."""
        result = features['base_logit'] + self.residual_head(features['endpoint_features']).squeeze(-1)
        if self.head_mode == 'structure':
            result = result + self.structure_head(features['structure_features']).squeeze(-1)
        return result * self.output_scale + self.output_bias

    def forward(self, sample, event_mask=None):
        return self.forward_features(self.feature_tensors(sample, event_mask))

    def set_calibration(self, scale, bias):
        scale, bias = float(scale), float(bias)
        if not math.isfinite(scale) or scale <= 0 or not math.isfinite(bias):
            raise ValueError('Calibration requires a finite positive scale and finite bias')
        with torch.no_grad():
            self.output_scale.fill_(scale)
            self.output_bias.fill_(bias)
