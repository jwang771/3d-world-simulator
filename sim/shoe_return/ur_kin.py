"""Universal Robots kinematics (numpy): DH forward kinematics and damped-least-squares IK seeded from a preferred
configuration, so transits can move in joint space (UR "movej") instead of through Cartesian paths that pass near
singularities. Frames: DH base -> DH tool flange; run_sim calibrates them against the simulated robot.
"""
import numpy as np

# Official UR DH parameters (https://www.universal-robots.com/articles/ur/application-installation/dh-parameters-for-calculations-of-kinematics-and-dynamics/)
DH = {
    "ur10e": dict(d=[0.1807, 0, 0, 0.17415, 0.11985, 0.11655], a=[0, -0.6127, -0.57155, 0, 0, 0]),
    "ur5e": dict(d=[0.1625, 0, 0, 0.1333, 0.0997, 0.0996], a=[0, -0.425, -0.3922, 0, 0, 0]),
}
ALPHA = [np.pi / 2, 0, 0, np.pi / 2, -np.pi / 2, 0]


def _link(theta, d, a, alpha):
    ct, st, ca, sa = np.cos(theta), np.sin(theta), np.cos(alpha), np.sin(alpha)
    return np.array([[ct, -st * ca, st * sa, a * ct], [st, ct * ca, -ct * sa, a * st], [0, sa, ca, d], [0, 0, 0, 1.0]])


def fk(q, model="ur10e"):
    p = DH[model]
    T = np.eye(4)
    for i in range(6):
        T = T @ _link(q[i], p["d"][i], p["a"][i], ALPHA[i])
    return T


def frames(q, model="ur10e"):
    """World-from-base transform of every joint frame 1..6 (for clearance checks)."""
    p = DH[model]
    T, out = np.eye(4), []
    for i in range(6):
        T = T @ _link(q[i], p["d"][i], p["a"][i], ALPHA[i])
        out.append(T)
    return out


def _err(T, Tg):
    dp = Tg[:3, 3] - T[:3, 3]
    R = Tg[:3, :3] @ T[:3, :3].T
    angle = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if angle < 1e-9:
        w = np.zeros(3)
    else:
        w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) * angle / (2 * np.sin(angle))
    return np.concatenate([dp, w])


def jacobian(q, model="ur10e", eps=1e-6):
    T0 = fk(q, model)
    J = np.zeros((6, 6))
    for i in range(6):
        dq = np.array(q, float)
        dq[i] += eps
        J[:, i] = _err(T0, fk(dq, model)) / eps
    return J


def ik(Tg, seed, model="ur10e", iters=200, tol=1e-5, damping=1e-3):
    """Joint angles reaching Tg (DH base -> flange), found by DLS from `seed`; returns (q, residual norm)."""
    q = np.array(seed, float)
    for _ in range(iters):
        e = _err(fk(q, model), Tg)
        n = np.linalg.norm(e)
        if n < tol:
            break
        J = jacobian(q, model)
        dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(6), e)
        step = np.linalg.norm(dq)
        q = q + (dq if step < 0.3 else dq * 0.3 / step)
    return q, float(np.linalg.norm(_err(fk(q, model), Tg)))


def nearest_equivalent(q, ref):
    """Shift each joint by multiples of 2*pi toward ref (UR joints span +-2*pi)."""
    q = np.array(q, float)
    return q + 2 * np.pi * np.round((np.asarray(ref) - q) / (2 * np.pi))


# UR e-Series joint limits: +-360 deg on every joint except the elbow (+-180 deg).
LIMITS = np.array([2 * np.pi, 2 * np.pi, np.pi, 2 * np.pi, 2 * np.pi, 2 * np.pi])


def within_limits(q, margin=0.03):
    return bool(np.all(np.abs(q) <= LIMITS - margin))


def seeds_around(home):
    """Starting points covering the usual UR branches: elbow up/down and both wrist flips."""
    out = []
    for lift, elbow in ((-1.9, 1.9), (-1.2, 1.4), (-2.4, 2.4), (-0.6, -1.9), (-1.5, -1.5)):
        for w2 in (-np.pi / 2, np.pi / 2):
            q = np.array(home, float)
            q[1], q[2], q[4] = lift, elbow, w2
            out.append(q)
    return out


def in_family(q):
    """The configuration family UR cells normally work in: elbow up (shoulder lift in (-pi, 0), elbow in
    (0, pi)) and wrist down (wrist_2 in (-pi, 0) modulo 2 pi)."""
    w2 = (q[4] + np.pi) % (2 * np.pi) - np.pi
    return -np.pi < q[1] < 0 and 0 < q[2] < np.pi and -np.pi < w2 < 0


def ik_best(Tg, current, home, model="ur10e", tol=1e-3, min_link_z=0.06):
    """Joint solution for Tg inside the joint limits, with the elbow and wrist joints at least `min_link_z` above
    the robot's mounting surface (z=0 of the DH base), that needs the least motion from `current`.
    Returns (q, residual); residual > tol means no valid solution was found (q is then the best effort)."""
    best, fallback = None, None
    for seed in [current, home, *seeds_around(home)]:
        q, res = ik(Tg, seed, model)
        if fallback is None or res < fallback[1]:
            fallback = (q, res)
        if res > tol:
            continue
        near = nearest_equivalent(q, current)
        principal = (q + np.pi) % (2 * np.pi) - np.pi
        cands = [near, np.where(np.arange(6) == 2, principal, near), principal]  # elbow principal; all principal
        for c in cands:
            if not within_limits(c) or min(T[2, 3] for T in frames(c, model)[1:5]) < min_link_z:
                continue
            # Least motion, staying in the usual family; the small pull toward home unwinds joints that a chain of
            # nearest-equivalent solutions left a full turn away (a wrist at 269 deg cost a 434 deg move later).
            cost = (np.abs(c - current) @ [1.0, 1.0, 1.0, 0.6, 0.6, 0.3] + (0.0 if in_family(c) else 100.0)
                    + 0.15 * np.abs(c - np.asarray(home)).sum())
            if best is None or cost < best[1]:
                best = (c, cost, res)
    if best is None:
        return fallback
    return best[0], best[2]
