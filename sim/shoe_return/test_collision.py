"""Offline checks of the transit collision model (no Isaac): sphere model, box hits, detour planning.

    python sim/shoe_return/test_collision.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import collision  # noqa: E402

T, I = np.eye(4), np.eye(4)
spheres = lambda q: collision.arm_spheres(T, q, I)
q = np.array([0, -1.9, 1.9, -1.57, -1.57, 0.0])
c, r = spheres(q)
assert len(r) > 20 and c[:, 2].min() >= 0.0

tip = np.eye(4)
tip[:3, 3] = c[-1]
assert collision.Scene(boxes=[(tip, np.full(3, 0.05))]).hits(c, r) == "box0"
assert collision.Scene(table_z=-0.5).hits(c, r) is None
assert collision.Scene(table_z=0.2).hits(c, r) == "table"
assert collision.Scene(others=[(c + [0.05, 0, 0], r)]).hits(c, r) == "arm"

q2 = q + [1.5, 0.3, -0.3, 0, 0, 0]
mid = np.eye(4)
mid[:3, 3] = (c[-1] + spheres(q2)[0][-1]) / 2
scene = collision.Scene(boxes=[(mid, np.full(3, 0.05))])
assert collision.path_hit(spheres, scene, q, q2) == "box0"
path = collision.plan(spheres, scene, q, q2, vias=[q + [0.75, 0.6, -0.9, 0, 0, 0]])
assert path is not None and np.allclose(path[-1], q2)
pts = [q, *path]
assert all(collision.path_hit(spheres, scene, a, b) is None for a, b in zip(pts, pts[1:]))
# A path in contact all the way (e.g. a held object dragged along the table) is not free.
low = collision.Scene(table_z=10.0)
assert collision.path_hit(spheres, low, q, q2) == "table"
print("ALL OK")
