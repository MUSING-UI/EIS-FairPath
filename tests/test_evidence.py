import hashlib
import math
import random
import struct

import torch


def f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def test_four_probes_use_qid_seed_and_float32_antithetic_pairs():
    from eis_fairpath.evidence import random_plan
    ids = ['a', 'b', 'c']
    rows = random_plan(ids, 'new-query')
    seed = int.from_bytes(hashlib.sha256(b'pathfree-input-gradient-v1|20261027|new-query').digest()[:8], 'big')
    rng = random.Random(seed)
    expected = []
    for _ in range(2):
        draw = [f32(rng.random()) for _ in ids]
        expected.extend([draw, [f32(1.-x) for x in draw]])
    assert [row['gates'] for row in rows] == expected
    assert len(rows) == 4


def test_evidence_is_mean_absolute_positive_class_logit_gate_gradient():
    from eis_fairpath.evidence import build_evidence, random_plan
    ids = ['a', 'b', 'c']
    calls = []
    def predict(gates, outside=1.):
        calls.append(outside)
        return gates[0]**2 - 3.*gates[1] + gates[2]**3
    result = build_evidence(ids, 'new-query', predict)
    rows = random_plan(ids, 'new-query')
    expected = [f32(math.fsum([2.*r['gates'][0] for r in rows])/4.), 3.,
        f32(math.fsum([f32(3.*f32(r['gates'][2]**2)) for r in rows])/4.)]
    assert result['scores'] == dict(zip(ids, expected))
    assert calls == [1.] * 4
    assert all(p.grad is None for p in [])
