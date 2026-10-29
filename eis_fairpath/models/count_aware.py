"""HGT residual counts of distinct sampled physical events, with gate gradients."""
import torch
from torch import nn

from .teacher_v1 import GatedHGTLayer, NODE_TYPES
from .teacher_v2 import CausalHGTTeacher


def weighted_physical_counts(sample, route_ids, event_mask, num_states):
    """Count each sampled physical event once for each incoming target state."""
    counts = event_mask.new_zeros(num_states)
    if not route_ids.numel():
        return counts
    pairs = torch.stack((sample.target_index[route_ids], sample.event_index[route_ids]), dim=1)
    unique = torch.unique(pairs, dim=0)
    # Uniqueness uses integer support only; the shared event gate stays in autograd.
    counts.index_add_(0, unique[:, 0], event_mask[unique[:, 1]])
    return counts


class CountAwareHGTLayer(GatedHGTLayer):
    """Preserve the native gated HGT and add a typed count residual."""

    def __init__(self, hidden_dim, heads, time_dim):
        super().__init__(hidden_dim, heads, time_dim)
        self.count_projection = nn.ModuleList(
            nn.Linear(1, hidden_dim, bias=False) for _ in NODE_TYPES)
        for projection in self.count_projection:
            nn.init.zeros_(projection.weight)

    def forward(self, previous, sample, route_ids, event_mask, max_depth):
        result = super().forward(previous, sample, route_ids, event_mask, max_depth)
        counts = weighted_physical_counts(sample, route_ids, event_mask, len(previous)).log1p()
        active = sample.min_depth <= max_depth
        for node_type, projection in enumerate(self.count_projection):
            index = torch.where(active & (sample.node_types == node_type))[0]
            if index.numel():
                result[index] = result[index] + projection(counts[index, None])
        return result


class CountAwareHGTTeacher(CausalHGTTeacher):
    """Same occurrence support, identity, relative time, and decoder as v2 HGT."""

    def __init__(self, **architecture):
        super().__init__(**architecture)
        layers = []
        for old in self.convs:
            new = CountAwareHGTLayer(self.architecture['hidden_dim'], self.architecture['heads'],
                self.architecture['time_dim'])
            incompatible = new.load_state_dict(old.state_dict(), strict=False)
            expected = {f'count_projection.{i}.weight' for i in range(len(NODE_TYPES))}
            if set(incompatible.missing_keys) != expected or incompatible.unexpected_keys:
                raise ValueError('Native HGT layer migration differs from count residual addition')
            layers.append(new)
        self.convs = nn.ModuleList(layers)

    @classmethod
    def from_base_state(cls, architecture, state):
        model = cls(**architecture)
        incompatible = model.load_state_dict(state, strict=False)
        expected = {f'convs.{layer}.count_projection.{node_type}.weight'
            for layer in range(model.architecture['layers']) for node_type in range(len(NODE_TYPES))}
        if set(incompatible.missing_keys) != expected or incompatible.unexpected_keys:
            raise ValueError('Base checkpoint must differ only by the new zero count projections')
        return model
