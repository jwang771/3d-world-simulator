"""Collision checking for joint-space transits (numpy only).

Each UR10e + 2F-85 is a chain of spheres placed along its DH links; obstacles are oriented boxes (the shoebox base
and the lid at its current hinge angle), the table plane and the other arm's spheres. `plan` returns joint-space
waypoints from q0 to q1 whose straight segments are collision free: the direct path if it is, else a path through
one or two via configurations (raised "carry" poses first, then random samples).
"""
import numpy as np

import ur_kin

# UR10e link geometry beyond the DH parameters (from the UR10e datasheet drawings): the upper arm tube sits
# 0.176 m along the joint-2 axis from the DH plane, the forearm 0.040 m.
UPPER_OFFSET, FORE_OFFSET = 0.176, 0.040
RADII = dict(shoulder=0.085, upper=0.065, fore=0.055, wrist=0.058, gripper=0.055, fingers=0.045)
SPACING = 0.07


def _segment(a, b, r, out):
    n = max(1, int(np.ceil(np.linalg.norm(b - a) / SPACING)))
    for s in np.linspace(0.0, 1.0, n + 1):
        out.append((a + (b - a) * s, r))


def arm_spheres(T_world_dh, q, T_flange_tcp, model="ur10e", attached=None):
    """(centres (N, 3), radii (N,)) of the arm at q. T_world_dh: world <- DH base; T_flange_tcp: DH flange -> TCP.
    `attached`: (points (M, 3) in the TCP frame, radius) of a held object, which moves rigidly with the hand."""
    F = [T_world_dh @ T for T in ur_kin.frames(q, model)]
    P = [T_world_dh[:3, 3]] + [T[:3, 3] for T in F]
    z1 = F[0][:3, 2]  # joint-2 axis (joints 2-4 are parallel)
    out = []
    _segment(P[0], P[1], RADII["shoulder"], out)
    _segment(P[1] + z1 * UPPER_OFFSET, P[2] + z1 * UPPER_OFFSET, RADII["upper"], out)
    _segment(P[2] + z1 * FORE_OFFSET, P[3] + z1 * FORE_OFFSET, RADII["fore"], out)
    _segment(P[3], P[4], RADII["wrist"], out)
    _segment(P[4], P[5], RADII["wrist"], out)
    _segment(P[5], P[6], RADII["wrist"], out)
    tcp = F[5] @ T_flange_tcp
    _segment(P[6], tcp[:3, 3] - tcp[:3, 2] * 0.03, RADII["gripper"], out)
    out.append((tcp[:3, 3], RADII["fingers"]))
    if attached is not None:
        pts, rad = attached
        out += [(tcp[:3, :3] @ p + tcp[:3, 3], rad) for p in pts]
    c = np.array([o[0] for o in out])
    r = np.array([o[1] for o in out])
    return c, r


N_FIXED = 2  # base/shoulder spheres: they never move off the robot's mount, so skip them in table checks


class Scene:
    """Obstacles for one arm: oriented boxes [(T_world_box, half_extents)], a table height, other arms' spheres."""

    def __init__(self, boxes=(), table_z=-np.inf, others=(), margin=0.015):
        self.boxes, self.table_z, self.others, self.margin = list(boxes), table_z, list(others), margin

    def hits(self, c, r):
        m = self.margin
        if np.any(c[N_FIXED:, 2] - r[N_FIXED:] < self.table_z + m):
            return "table"
        for i, (T, half) in enumerate(self.boxes):
            local = (c - T[:3, 3]) @ T[:3, :3]
            d = np.linalg.norm(local - np.clip(local, -half, half), axis=1)
            if np.any(d < r + m):
                return f"box{i}"
        for oc, orad in self.others:
            d = np.linalg.norm(c[:, None, :] - oc[None, :, :], axis=-1)
            if np.any(d < r[:, None] + orad[None, :] + m):
                return "arm"
        return None


def path_hit(spheres_at, scene, q0, q1, step=np.deg2rad(3.0)):
    """First obstacle hit along the straight joint path q0 -> q1 (None if free). If an endpoint is itself in
    contact (e.g. a hand leaving a grasp), contact is allowed next to it, but only within the first/last quarter of
    the path: a path that is in contact all the way is not free."""
    n = max(1, int(np.ceil(np.abs(q1 - q0).max() / step)))
    hits = [scene.hits(*spheres_at(q0 + (q1 - q0) * s)) for s in np.linspace(0.0, 1.0, n + 1)]
    allow = max(1, (n + 1) // 4)
    lo, hi = 0, len(hits)
    while lo < allow and hits[lo]:
        lo += 1
    while len(hits) - hi < allow and hi > lo and hits[hi - 1]:
        hi -= 1
    return next((h for h in hits[lo:hi] if h), None)


def plan(spheres_at, scene, q0, q1, vias=(), rng=None, samples=300):
    """Joint waypoints after q0 ending at q1 with collision-free straight segments, or None.
    `vias`: preferred intermediate configurations (tried alone and in pairs, in order) before random samples."""
    if path_hit(spheres_at, scene, q0, q1) is None:
        return [q1]
    free = lambda a, b: path_hit(spheres_at, scene, a, b) is None
    vias = [v for v in vias if scene.hits(*spheres_at(v)) is None]
    # Joint travel, plus a charge per via (each one is a stop-and-go that swings a held object).
    cost = lambda path: sum(np.abs(b - a).max() for a, b in zip([q0, *path[:-1]], path)) + 0.3 * (len(path) - 1)
    best = None
    for v in vias:
        if free(q0, v) and free(v, q1):
            best = min([best, [v, q1]], key=lambda p: np.inf if p is None else cost(p))
    if best is None:
        for a in vias:
            if not free(q0, a):
                continue
            for b in vias:
                if b is not a and free(a, b) and free(b, q1):
                    best = min([best, [a, b, q1]], key=lambda p: np.inf if p is None else cost(p))
    if best is not None:
        return best
    rng = rng or np.random.default_rng(0)
    mid = (q0 + q1) / 2
    for _ in range(samples):
        v = mid + rng.normal(0.0, 0.6, 6) * [1.0, 0.6, 0.6, 0.8, 0.5, 0.3]
        v[2] = np.clip(v[2], -np.pi + 0.05, np.pi - 0.05)
        if scene.hits(*spheres_at(v)) is None and free(q0, v) and free(v, q1):
            return [v, q1]
    return None
