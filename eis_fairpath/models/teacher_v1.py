"""Stateless, bounded HGT on an explicit causal node-time occurrence DAG.

Only the sample's tensors are consumed. Relative time is added to each source
before its typed key/value projections; no history state or degree is read.
"""
import math
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from ..graph import RELATIONS

from ..sampling.types import SampledQuery


NODE_TYPES = ("author", "paper", "venue")


def relation_transform(values, relations, matrices):
    """Apply a matrix per relation without allocating one matrix per route."""
    result = torch.empty_like(values)
    for relation in range(len(matrices)):
        index = torch.where(relations == relation)[0]
        result[index] = torch.einsum("rhd,hde->rhe", values[index], matrices[relation])
    return result


def gated_aggregate(logits, values, targets, gates, num_states):
    """Local stable attention: normalize m*exp(logit), then attenuate value by m.

    ``logits`` is [routes, heads], ``values`` [routes, heads, head_dim]. Zero
    gates are excluded before the local maximum and denominator are computed.
    Empty gated neighborhoods return a zero aggregate with finite derivatives.
    """
    heads, head_dim = values.shape[1:]
    if not logits.shape[0]:
        return values.new_zeros((num_states, heads, head_dim))
    positive = gates > 0
    expanded_targets = targets[:, None].expand(-1, heads)
    # Center raw logits first: adding log(m) directly to a very large logit
    # loses mask precision even though its later exponential stays finite.
    raw_logits = logits.masked_fill(~positive[:, None], -torch.inf)
    raw_maxima = logits.new_full((num_states, heads), -torch.inf)
    raw_maxima.scatter_reduce_(0, expanded_targets, raw_logits.detach(), reduce="amax", include_self=True)
    raw_maxima = torch.where(torch.isfinite(raw_maxima), raw_maxima, torch.zeros_like(raw_maxima))
    safe_gates = torch.where(positive, gates, torch.ones_like(gates))
    log_weight = logits - raw_maxima[targets] + safe_gates.log()[:, None]
    log_weight = log_weight.masked_fill(~positive[:, None], -torch.inf)
    maxima = logits.new_full((num_states, heads), -torch.inf)
    maxima.scatter_reduce_(0, expanded_targets, log_weight.detach(), reduce="amax", include_self=True)
    maxima = torch.where(torch.isfinite(maxima), maxima, torch.zeros_like(maxima))
    weights = (log_weight - maxima[targets]).exp()
    denominator = logits.new_zeros((num_states, heads))
    denominator.index_add_(0, targets, weights)
    denominator = torch.where(denominator > 0, denominator, torch.ones_like(denominator))
    attention = weights / denominator[targets]
    messages = attention[:, :, None] * gates[:, None, None] * values
    aggregate = values.new_zeros((num_states, heads, head_dim))
    aggregate.index_add_(0, targets, messages)
    return aggregate


class RelativeTimeEncoding(nn.Module):
    def __init__(self, time_dim, hidden_dim):
        super().__init__()
        frequencies = torch.exp(-math.log(10000.) * torch.arange(time_dim // 2) / (time_dim // 2))
        self.register_buffer("frequencies", frequencies)
        self.project = nn.Linear(time_dim, hidden_dim, bias=False)

    def forward(self, delta):
        angle = delta[:, None] * self.frequencies[None, :]
        return self.project(torch.cat((angle.cos(), angle.sin()), dim=-1))


class GatedHGTLayer(nn.Module):
    """One synchronous occurrence layer with typed maps and relation matrices."""

    def __init__(self, hidden_dim, heads, time_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.query = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim) for _ in NODE_TYPES)
        self.key = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim) for _ in NODE_TYPES)
        self.value = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim) for _ in NODE_TYPES)
        self.output = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim) for _ in NODE_TYPES)
        self.norm = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in NODE_TYPES)
        self.skip = nn.Parameter(torch.ones(len(NODE_TYPES)))
        matrix_shape = (len(RELATIONS), heads, self.head_dim, self.head_dim)
        self.relation_attention = nn.Parameter(torch.empty(matrix_shape))
        self.relation_message = nn.Parameter(torch.empty(matrix_shape))
        self.relation_priority = nn.Parameter(torch.ones((len(RELATIONS), heads)))
        self.time = RelativeTimeEncoding(time_dim, hidden_dim)
        for relation in range(len(RELATIONS)):
            for head in range(heads):
                nn.init.xavier_uniform_(self.relation_attention[relation, head])
                nn.init.xavier_uniform_(self.relation_message[relation, head])

    @staticmethod
    def typed_projection(tensor, node_types, modules):
        result = torch.empty_like(tensor)
        for node_type, module in enumerate(modules):
            index = torch.where(node_types == node_type)[0]
            result[index] = module(tensor[index])
        return result

    def forward(self, previous, sample, route_ids, event_mask, max_depth):
        source = sample.source_index[route_ids]
        target = sample.target_index[route_ids]
        if bool((sample.min_depth[source] > max_depth + 1).any()):
            raise ValueError("A route source has no valid previous-layer occurrence")
        if bool((sample.min_depth[target] > max_depth).any()):
            raise ValueError("A route target is outside the current-layer occurrence DAG")
        active = sample.min_depth <= max_depth
        active_index = torch.where(active)[0]
        relation = sample.relation_index[route_ids]
        source_types = sample.node_types[source]
        source_input = previous[source] + self.time(sample.delta_time[route_ids])
        query = previous.new_zeros(previous.shape)
        query[active_index] = self.typed_projection(previous[active_index], sample.node_types[active_index], self.query)
        key = self.typed_projection(source_input, source_types, self.key)
        value = self.typed_projection(source_input, source_types, self.value)
        key = key.reshape(-1, self.heads, self.head_dim)
        value = value.reshape(-1, self.heads, self.head_dim)
        key = relation_transform(key, relation, self.relation_attention)
        value = relation_transform(value, relation, self.relation_message)
        query = query[target].reshape(-1, self.heads, self.head_dim)
        logits = (query * key).sum(-1) * self.relation_priority[relation] / math.sqrt(self.head_dim)
        aggregate = gated_aggregate(
            logits, value, target, event_mask[sample.event_index[route_ids]], len(previous)
        ).reshape(len(previous), self.hidden_dim)
        # An inactive state's last computed layer is retained only as storage;
        # the next valid layer cannot read it (checked above at its source).
        result = previous.clone()
        for node_type in range(len(NODE_TYPES)):
            index = torch.where(active & (sample.node_types == node_type))[0]
            if not index.numel():
                continue
            neighbor = F.gelu(self.output[node_type](aggregate[index]))
            alpha = self.skip[node_type].sigmoid()
            result[index] = self.norm[node_type](alpha * neighbor + (1. - alpha) * previous[index])
        return result


class CausalHGTTeacher(nn.Module):
    """Finite-depth event-gated HGT with symmetric author-pair prediction."""

    def __init__(self, input_dim=16, hidden_dim=128, heads=2, time_dim=32, layers=4):
        super().__init__()
        if input_dim < 1 or hidden_dim < 1 or heads < 1 or hidden_dim % heads:
            raise ValueError("Positive dimensions and hidden_dim divisible by heads are required")
        if time_dim < 2 or time_dim % 2 or layers < 1:
            raise ValueError("Even positive time_dim and positive layers are required")
        self.architecture = dict(input_dim=input_dim, hidden_dim=hidden_dim, heads=heads,
                                 time_dim=time_dim, layers=layers)
        self.metadata = (list(NODE_TYPES), list(RELATIONS))
        self.project = nn.ModuleList(nn.Linear(input_dim, hidden_dim) for _ in NODE_TYPES)
        self.convs = nn.ModuleList(GatedHGTLayer(hidden_dim, heads, time_dim) for _ in range(layers))
        self.decoder = nn.Bilinear(hidden_dim, hidden_dim, 1)

    def _event_mask(self, sample, event_mask):
        if sample.config.layers != self.architecture["layers"]:
            raise ValueError("Sample layer budget and model layer count differ")
        if sample.features.ndim != 2 or sample.features.shape[1] != self.architecture["input_dim"]:
            raise ValueError("Sample input feature dimension differs from the model")
        if event_mask is None:
            return sample.features.new_ones(len(sample.event_ids))
        event_mask = torch.as_tensor(event_mask, dtype=sample.features.dtype, device=sample.features.device)
        if event_mask.ndim != 1 or event_mask.numel() != len(sample.event_ids):
            raise ValueError("One scalar gate per physical event is required")
        if not bool(torch.isfinite(event_mask).all()) or bool((event_mask < 0).any()) or bool((event_mask > 1).any()):
            raise ValueError("Physical event gates must be finite probabilities in [0,1]")
        return event_mask

    def encode(self, sample: SampledQuery, event_mask=None):
        event_mask = self._event_mask(sample, event_mask)
        hidden_dim = self.architecture["hidden_dim"]
        previous = sample.features.new_empty((len(sample.states), hidden_dim))
        for node_type, project in enumerate(self.project):
            index = torch.where(sample.node_types == node_type)[0]
            previous[index] = project(sample.features[index])
        for layer, conv in enumerate(self.convs, start=1):
            previous = conv(previous, sample, sample.routes_for_layer(layer), event_mask,
                            self.architecture["layers"] - layer)
        return previous

    def forward(self, sample: SampledQuery, event_mask=None):
        hidden = self.encode(sample, event_mask)
        left, right = hidden[sample.roots[0]], hidden[sample.roots[1]]
        return (.5 * (self.decoder(left, right) + self.decoder(right, left))).squeeze(-1)

    def predict(self, samples: Sequence[SampledQuery], event_masks=None):
        if event_masks is None:
            event_masks = [None] * len(samples)
        if len(event_masks) != len(samples):
            raise ValueError("One event mask per query is required")
        if not samples:
            return next(self.parameters()).new_empty(0)
        return torch.stack([self(sample, mask) for sample, mask in zip(samples, event_masks)])
