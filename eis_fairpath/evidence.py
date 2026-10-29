"""Four query-seeded antithetic probes of positive-class logit gate gradients."""
from __future__ import annotations

import hashlib
import math
import random
import struct

import torch


def _f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def random_plan(event_ids, qid):
    ids = list(event_ids)
    if (not ids or len(set(ids)) != len(ids)
            or any(not isinstance(e, str) or not e for e in ids)):
        raise ValueError('Unique ordered nonempty event domain required')
    if not isinstance(qid, str) or not qid:
        raise ValueError('Nonempty query identity required')
    seed = int.from_bytes(hashlib.sha256(
        f'pathfree-input-gradient-v1|20261027|{qid}'.encode()).digest()[:8], 'big')
    rng = random.Random(seed)
    rows = []
    for pair in range(2):
        gates = [_f32(rng.random()) for _ in ids]
        for index, values in ((pair*2, gates), (pair*2+1, [_f32(1.-x) for x in gates])):
            rows.append(dict(probe_id=f'RANDOM:all:{index}', index=index, query_seed=seed, gates=values))
    return rows


def capture_gradient(predict, gates, *, device='cpu'):
    values = torch.tensor(gates, dtype=torch.float32, device=device, requires_grad=True)
    with torch.enable_grad():
        output = predict(values, outside=1.)
        z = output.logits if hasattr(output, 'logits') else output
        if z.numel() != 1 or not bool(torch.isfinite(z).all()) or not z.requires_grad:
            raise ValueError('One finite differentiable frozen prediction required')
        gradient, = torch.autograd.grad(z.reshape(()), values, allow_unused=True)
    if gradient is None or gradient.shape != values.shape or not bool(torch.isfinite(gradient).all()):
        raise ValueError('Finite full-domain input-gate gradient required')
    return dict(logit=float(z.detach().reshape(())), gradient=gradient.detach().cpu().tolist())


def aggregate(rows):
    if len(rows) != 4:
        raise ValueError('Exactly four gradient probes required')
    size = len(rows[0]['gradient'])
    if not size or any(len(r['gradient']) != size for r in rows):
        raise ValueError('Four aligned full-domain gradients required')
    return [_f32(math.fsum(abs(row['gradient'][i]) for row in rows) / 4.) for i in range(size)]


def build_evidence(event_ids, qid, predict, *, device='cpu'):
    ids = list(event_ids)
    rows = random_plan(ids, qid)
    for row in rows:
        row.update(capture_gradient(predict, row['gates'], device=device))
    return dict(event_ids=ids, scores=dict(zip(ids, aggregate(rows))), probes=rows,
        source=dict(channel='RANDOM_MAGNITUDE', query_id=qid, probe_count=4,
            outside_gate=1., statistic='float32 mean absolute positive-class logit gate derivative'))
