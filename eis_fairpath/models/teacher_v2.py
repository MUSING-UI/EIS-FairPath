"""A symmetric query endpoint signal in the existing four-layer HGT."""
import torch
from torch import nn

from .teacher_v1 import CausalHGTTeacher as NativeTeacher
from ..sampling.types import support_digest


def endpoint_membership(sample):
    roots = {(sample.states[i].node_type, sample.states[i].node_id) for i in sample.roots}
    return sample.features.new_tensor([[float((s.node_type, s.node_id) in roots)] for s in sample.states])


class CausalHGTTeacher(NativeTeacher):
    def __init__(self, *, input_dim=16, hidden_dim=128, heads=2, time_dim=32, layers=4, role_mode='membership'):
        if role_mode not in ('none', 'membership'):
            raise ValueError(f'Unknown role_mode: {role_mode}')
        super().__init__(input_dim=input_dim, hidden_dim=hidden_dim, heads=heads, time_dim=time_dim, layers=layers)
        self.architecture['role_mode'] = role_mode
        # The new projection is initialized after all original shared parameters.
        self.role_projection = nn.Linear(1, hidden_dim, bias=False)
        nn.init.zeros_(self.role_projection.weight)

    def encode(self, sample, event_mask=None):
        # SampledQuery is mutable. Validate at the model boundary so even the
        # reused generic adapter cannot accept stale features/routes, including
        # in-place changes made after a previous successful forward.
        if support_digest(sample) != sample.support_hash:
            raise ValueError('Actual support differs from its support hash')
        if self.architecture['role_mode'] == 'none':
            return super().encode(sample, event_mask)
        event_mask = self._event_mask(sample, event_mask)
        previous = sample.features.new_empty((len(sample.states), self.architecture['hidden_dim']))
        for node_type, project in enumerate(self.project):
            index = torch.where(sample.node_types == node_type)[0]
            previous[index] = project(sample.features[index])
        previous = previous+self.role_projection(endpoint_membership(sample))
        for layer, conv in enumerate(self.convs, start=1):
            previous = conv(previous, sample, sample.routes_for_layer(layer), event_mask,
                            self.architecture['layers']-layer)
        return previous
