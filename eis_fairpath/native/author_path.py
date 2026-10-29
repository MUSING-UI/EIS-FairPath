"""Native occurrence path weights and per-relation Gaussian initialization."""
from collections import Counter, defaultdict
import math
import torch
from .search import _strength

def author_path_weights(logits, strength_differences, *, theta=.5):
    """Training weights; the author's additive bias is neither shifted nor clipped."""
    return torch.sigmoid(logits) + theta * strength_differences.abs()

def author_strength_differences(edges, community):
    """Use the exact scorer's endpoint strengths at each event's timestamp."""
    if edges and getattr(community, "path_strengths_by_timestamp", None) is None:
        raise ValueError("Author temporal path weights require snapshot strengths")
    return [abs(_strength(community, (edge.relation[0], edge.src), edge.timestamp)
                - _strength(community, (edge.relation[2], edge.dst), edge.timestamp))
            for edge in edges]

def relation_node_counts(edges, nodes):
    """DGL edge_type_subgraph counts every node of each retained node type."""
    counts = Counter(node_type for node_type, _ in set(nodes))
    return {relation: sum(counts[t] for t in {relation[0], relation[2]})
            for relation in sorted({edge.relation for edge in edges})}

def initialize_author_logits(edges, nodes, *, seed=42, device="cpu"):
    """Per-relation N(0, 2/N_r); sample reproducibly without changing global RNG."""
    edges = list(edges)
    counts = relation_node_counts(edges, nodes)
    indices = defaultdict(list)
    for i, edge in enumerate(edges):
        indices[edge.relation].append(i)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    values = torch.empty(len(edges), dtype=torch.float32)
    for relation in sorted(indices):
        n = counts[relation]
        std = math.sqrt(2. / n) if n else 1.
        values[indices[relation]] = torch.randn(len(indices[relation]), generator=generator) * std
    return values.to(device).requires_grad_()

def author_initialization(ctx, *, seed, device):
    # The temporal learning graph consists of ctx.edges plus the query nodes.
    nodes = {ctx.query.source, ctx.query.target}
    for edge in ctx.edges:
        nodes.update(((edge.relation[0], edge.src), (edge.relation[2], edge.dst)))
    counts = relation_node_counts(ctx.edges, nodes)
    return initialize_author_logits(ctx.edges, nodes, seed=seed, device=device), {
        "relation_node_counts": {"|".join(r): n for r, n in counts.items()},
        "node_count_scope": "ctx.edges endpoints plus query endpoints",
        "distribution": "normal(0, sqrt(2 / N_relation))",
        "rng": "local_cpu_generator_per_query_seed",
    }
