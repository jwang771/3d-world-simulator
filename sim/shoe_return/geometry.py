"""Small numpy pose helpers. Quaternions are (w, x, y, z), matching Isaac Sim."""
import numpy as np


def quat_to_mat(q):
    w, x, y, z = np.asarray(q, np.float64) / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def mat_to_quat(R):
    R = np.asarray(R, np.float64)
    tr = np.trace(R)
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    return q / np.linalg.norm(q) * (1 if q[0] >= 0 else -1)


def mat_to_rotvec(R):
    q = mat_to_quat(R)
    angle = 2.0 * np.arctan2(np.linalg.norm(q[1:]), q[0])
    n = np.linalg.norm(q[1:])
    return np.zeros(3) if n < 1e-9 else q[1:] / n * angle


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def rot_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1.0, 0], [-s, 0, c]])


def pose(R, t):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


def pose_from_pq(p, q):
    return pose(quat_to_mat(q), p)


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def yaw_of(R):
    return float(np.arctan2(R[1, 0], R[0, 0]))


def look_at_ros(eye, target, up=(0.0, 0.0, 1.0)):
    """Camera-to-world rotation in the ROS/OpenCV optical convention (+z forward, +x right, +y down)."""
    f = np.asarray(target, np.float64) - np.asarray(eye, np.float64)
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    if np.linalg.norm(r) < 1e-6:
        r = np.cross(f, (0.0, 1.0, 0.0))
    r /= np.linalg.norm(r)
    d = np.cross(f, r)
    return np.stack([r, d, f], axis=1)


def top_down(yaw):
    """Gripper pointing straight down (hand +z = world -z), fingers closing along world Rz(yaw) @ +y."""
    return rot_z(yaw) @ rot_y(np.pi)


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0, 0], [0, c, -s], [0, s, c]])


def rotvec_to_mat(v):
    angle = np.linalg.norm(v)
    if angle < 1e-12:
        return np.eye(3)
    k = np.asarray(v) / angle
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle) * Kx + (1 - np.cos(angle)) * Kx @ Kx


def slerp_mat(R0, R1, s):
    return R0 @ rotvec_to_mat(s * mat_to_rotvec(R0.T @ R1))


def tcp_yaw(R):
    """Yaw of a top-down gripper frame: inverse of top_down()."""
    return float(np.arctan2(R[1, 1], R[0, 1]) - np.pi / 2)
