"""KL_H4 objective: equal-role teacher attribution KL plus native path loss."""
from __future__ import annotations

import math

import torch


class MeanRoleKL:
    """Mean of Internal/Cross KL(S/sum(S) || softmax(raw M)), in float64.

    Constant evidence in either role disables the entire prediction term using
    a differentiable zero. The two role probabilities stay fixed across updates.
    """

    roles = ('internal', 'cross')

    def __init__(self, event_ids, groups, scores):
        self.event_ids = list(event_ids)
        if (not self.event_ids or len(set(self.event_ids)) != len(self.event_ids)
                or any(not isinstance(e, str) or not e for e in self.event_ids)
                or set(groups) != set(self.event_ids) or set(scores) != set(self.event_ids)
                or set(groups.values()) != set(self.roles)):
            raise ValueError('Unique full event domain with both physical roles and all scores required')
        if any(type(v) not in (float, int) or not math.isfinite(v) or v < 0 for v in scores.values()):
            raise ValueError('Evidence scores must be finite and nonnegative')
        self.groups, self.scores = dict(groups), dict(scores)
        self.indices, self.probabilities, self.available = [], [], []
        for role in self.roles:
            indices = [i for i, event in enumerate(self.event_ids) if groups[event] == role]
            values = [scores[self.event_ids[i]] for i in indices]
            maximum = max(values)
            # Rescaling preserves the original arithmetic and avoids overflow.
            scaled = [v / maximum for v in values] if maximum > 0 else [0.] * len(values)
            denominator = math.fsum(scaled)
            probabilities = [v / denominator for v in scaled] if denominator else scaled
            self.indices.append(indices)
            self.probabilities.append(torch.tensor(probabilities, dtype=torch.float64))
            self.available.append(min(values) != maximum)

    def observe(self, raw):
        if (not isinstance(raw, torch.Tensor) or raw.ndim != 1
                or len(raw) != len(self.event_ids) or not raw.is_floating_point()
                or not bool(torch.isfinite(raw).all())):
            raise ValueError('Finite floating raw mask vector on the full domain required')
        losses = []
        for indices, probabilities, available in zip(self.indices, self.probabilities, self.available):
            if not available:
                losses.append(raw.double().sum() * 0.)
                continue
            target = probabilities.to(raw.device)
            positive = target > 0
            logq = torch.log_softmax(raw[indices].double(), dim=0)
            losses.append((target[positive] * (target[positive].log() - logq[positive])).sum())
        return torch.stack(losses)

    def prediction_loss(self, raw):
        losses = self.observe(raw)
        return losses.mean() if all(self.available) else losses.sum() * 0.

    def metadata(self):
        return dict(roles=list(self.roles), available=list(self.available),
            target_probabilities=[p.tolist() for p in self.probabilities],
            aggregation='arithmetic mean', temperature=1., coefficient=1., arithmetic_dtype='float64',
            constant_role_policy='any constant role: connected zero for entire prediction loss')


def author_path_loss(weights, event_ids, paths, *, margin=.1, alpha=1., beta=1.):
    """Average on-path occurrences; average the unique off-path complement once."""
    index = {event: i for i, event in enumerate(event_ids)}
    occurrences = [index[event] for path in paths for event in path.event_ids]
    on_indices = torch.tensor(occurrences, dtype=torch.long, device=weights.device)
    off_mask = torch.ones(len(event_ids), dtype=torch.bool, device=weights.device)
    off_mask[on_indices] = False
    zero = weights.sum() * 0.
    on = -weights[on_indices].mean() if occurrences else zero
    off = weights[off_mask].mean() if off_mask.any() else zero
    hinge = torch.where(off + margin + on > 0., off + margin + on, zero)
    return dict(path=alpha * on + beta * off + beta * hinge, on=on, off=off, hinge=hinge)


def mask_weights(raw, strength_differences, *, delta=1., theta=.5):
    """Diagnostic prediction gates from historical community strengths."""
    base = torch.sigmoid(raw + delta)
    return base, torch.clamp(base + theta * strength_differences.abs(), min=0., max=1.)
