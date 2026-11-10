"""Offline checks of ur_kin (no Isaac): FK/IK round trips and ik_best's limits, clearance and unwinding.

    python sim/shoe_return/test_ur_kin.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ur_kin  # noqa: E402

rng = np.random.default_rng(0)
home = np.array([0.0, -1.9, 1.9, -1.57, -1.57, 0.0])
ok = 0
for _ in range(100):
    q = home + rng.uniform(-0.8, 0.8, 6)
    T = ur_kin.fk(q)
    q2, res = ur_kin.ik(T, q + rng.normal(0, 0.1, 6))
    ok += res < 1e-4 and np.abs(ur_kin.fk(q2)[:3, 3] - T[:3, 3]).max() < 1e-4
assert ok == 100, ok
for _ in range(50):
    q = home + rng.uniform(-0.6, 0.6, 6)
    qb, res = ur_kin.ik_best(ur_kin.fk(q), home, home)
    assert res < 1e-3 and ur_kin.within_limits(qb)
    assert min(T[2, 3] for T in ur_kin.frames(qb)[1:5]) >= 0.06
# Solutions found from home keep the wrists within +-180 deg (no accumulated wind-up).
for _ in range(50):
    q = home + rng.uniform(-1.0, 1.0, 6)
    qb, res = ur_kin.ik_best(ur_kin.fk(q), home, home)
    assert res > 1e-3 or not ur_kin.in_family(qb) or np.all(np.abs(qb[3:]) <= np.pi + 1e-6), qb
print("ALL OK")
