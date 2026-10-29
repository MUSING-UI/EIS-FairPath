"""Explicit tensor support for a finite-depth, physical-event teacher."""
from dataclasses import asdict, dataclass, replace
import hashlib
import json

import torch


NODE_TYPES = ("author", "paper", "venue")


class SamplingLimitError(RuntimeError):
    """A declared occurrence/route guard was exceeded; the query is not dropped."""


@dataclass(frozen=True)
class SamplerConfig:
    layers: int = 4
    event_budget: int = 512
    depth_quotas: tuple[int, ...] = (64, 128, 160, 160)
    fanout: int = 32
    occurrence_cap: int = 8192
    route_cap: int = 32768
    seed: int = 42
    history_window: int = 5

    def __post_init__(self):
        for field in ("layers", "event_budget", "fanout", "occurrence_cap", "route_cap", "history_window"):
            if type(getattr(self, field)) is not int or getattr(self, field) <= 0:
                raise ValueError(f"{field} must be a positive integer")
        if type(self.seed) is not int:
            raise ValueError("seed must be an integer")
        quotas = tuple(self.depth_quotas)
        if len(quotas) != self.layers or any(type(q) is not int or q < 0 for q in quotas):
            raise ValueError("depth_quotas must have one nonnegative integer per layer")
        object.__setattr__(self, "depth_quotas", quotas)


@dataclass(frozen=True, order=True)
class StateKey:
    node_type: str
    node_id: int
    cutoff: int


def support_digest(sample):
    """Hash the actual fixed support, including feature values and route aliases."""
    h = hashlib.sha256()
    descriptor = {
        "states": [(s.node_type, s.node_id, s.cutoff) for s in sample.states],
        "roots": sample.roots, "events": sample.event_ids, "config": asdict(sample.config),
        "dataset": sample.metadata.get("dataset", ""),
        "history_input_hash": sample.metadata.get("history_input_hash", ""),
    }
    h.update(json.dumps(descriptor, sort_keys=True, separators=(",", ":")).encode())
    for tensor in (sample.features, sample.node_types, sample.min_depth, sample.source_index,
                   sample.target_index, sample.event_index, sample.relation_index, sample.delta_time):
        cpu = tensor.detach().cpu().contiguous()
        h.update(str(cpu.dtype).encode())
        h.update(str(tuple(cpu.shape)).encode())
        h.update(cpu.numpy().tobytes())
    return h.hexdigest()


@dataclass
class SampledQuery:
    states: tuple[StateKey, ...]
    roots: tuple[int, int]
    event_ids: tuple[str, ...]
    features: torch.Tensor
    node_types: torch.Tensor
    min_depth: torch.Tensor
    source_index: torch.Tensor
    target_index: torch.Tensor
    event_index: torch.Tensor
    relation_index: torch.Tensor
    delta_time: torch.Tensor
    config: SamplerConfig
    metadata: dict
    support_hash: str

    def routes_for_layer(self, layer):
        if type(layer) is not int or not 1 <= layer <= self.config.layers:
            raise ValueError("layer must be between 1 and configured layers")
        eligible = self.min_depth[self.target_index] <= self.config.layers - layer
        return torch.nonzero(eligible, as_tuple=False).flatten()

    @property
    def computation_event_ids(self):
        """Unique events in the structural root DAG, independent of numeric gradients."""
        current = set(self.roots)
        used = set()
        sources = self.source_index.detach().cpu().tolist()
        targets = self.target_index.detach().cpu().tolist()
        event_indices = self.event_index.detach().cpu().tolist()
        for layer in range(self.config.layers, 0, -1):
            next_states = set(current)  # Pointwise previous-layer residual dependency.
            for route in self.routes_for_layer(layer).detach().cpu().tolist():
                if targets[route] in current:
                    used.add(self.event_ids[event_indices[route]])
                    next_states.add(sources[route])
            current = next_states
        return tuple(event for event in self.event_ids if event in used)

    def to(self, device):
        fields = ("features", "node_types", "min_depth", "source_index", "target_index",
                  "event_index", "relation_index", "delta_time")
        return replace(self, **{field: getattr(self, field).to(device) for field in fields},
                       metadata=dict(self.metadata))

    def without_events(self, ids):
        """Delete routes only; retain the fixed sampled node/event catalog and features."""
        ids = set(ids)
        unknown = ids - set(self.event_ids)
        if unknown:
            raise ValueError(f"Unknown sampled physical event IDs: {sorted(unknown)}")
        retained = torch.tensor([event not in ids for event in self.event_ids],
                                device=self.event_index.device, dtype=torch.bool)
        keep = retained[self.event_index]
        fields = ("source_index", "target_index", "event_index", "relation_index", "delta_time")
        metadata = dict(self.metadata)
        metadata.update(fixed_support_parent_hash=self.support_hash,
                        removed_event_ids=sorted(ids), num_routes=int(keep.sum()),
                        num_route_layers=int((self.config.layers - self.min_depth[self.target_index[keep]]).sum()))
        sample = replace(self, **{field: getattr(self, field)[keep] for field in fields},
                         metadata=metadata, support_hash="")
        sample.support_hash = support_digest(sample)
        return sample
