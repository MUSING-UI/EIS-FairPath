"""A frozen teacher over one sampled support; intervention never resamples."""
import torch


class FixedSupportTeacher:
    def __init__(self, model, sample):
        if any(p.requires_grad for p in model.parameters()):
            raise ValueError('Freeze teacher parameters before explanation')
        self.model = model.eval()
        self.sample = sample
        self.domain_event_ids = sample.computation_event_ids
        self.support_hash = sample.support_hash

    def native_logit(self):
        return self.model(self.sample)

    def masked_logit(self, event_ids, weights, *, outside):
        if outside not in (0., 1.):
            raise ValueError('outside must be explicit zero or one')
        if len(event_ids) != len(weights) or len(set(event_ids)) != len(event_ids):
            raise ValueError('One weight per unique event is required')
        unknown = set(event_ids)-set(self.domain_event_ids)
        if unknown:
            raise ValueError('Mask events outside the certified sampled domain')
        weights = torch.as_tensor(weights, dtype=self.sample.features.dtype,
                                  device=self.sample.features.device)
        positions = {event:i for i,event in enumerate(self.sample.event_ids)}
        indices = torch.tensor([positions[e] for e in event_ids], dtype=torch.long, device=weights.device)
        gates = weights.new_full((len(positions),), outside).index_copy(0, indices, weights)
        return self.model(self.sample, gates)
