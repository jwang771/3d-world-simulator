"""Shoe-return episodes -> (camera images, proprio, action chunk) samples for imitation learning.

Actions are the recorded per-arm TCP targets (position, rotation vector, gripper) of `action/cartesian_action`. A
top-down gripper sits at a rotation-vector magnitude of ~pi, where the vector flips sign, so policies learn the
continuous 6-D rotation representation (first two columns of the rotation matrix) instead: 2 x (3 + 6 + 1) = 20.
Proprio is the previous action (the commanded pose), which the policy also gets at test time.

Decoded, downsized frames are cached per episode as .npz next to the dataset (<root>/_policy_cache/).
"""
import json
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch

CAMS = (0, 3, 4)  # overhead, left wrist, right wrist
IMG_HW = (96, 128)
STRIDE = 2  # recorded frames per policy step (15 Hz recording -> 7.5 Hz policy)


def rotvec_to_mat(v):
    v = np.asarray(v, float)
    th = np.linalg.norm(v, axis=-1, keepdims=True)
    k = np.where(th > 1e-9, v / np.maximum(th, 1e-12), [1.0, 0, 0])
    K = np.zeros(v.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 2] = -k[..., 2], k[..., 1], -k[..., 0]
    K[..., 1, 0], K[..., 2, 0], K[..., 2, 1] = k[..., 2], -k[..., 1], k[..., 0]
    th = th[..., None]
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def mat_to_6d(R):
    return np.concatenate([R[..., :, 0], R[..., :, 1]], axis=-1)


def sixd_to_mat(x):
    a, b = x[..., :3], x[..., 3:6]
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b - (a * b).sum(-1, keepdims=True) * a
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return np.stack([a, b, np.cross(a, b)], axis=-1)


def to_policy_action(a14):
    """(T, 14) [p, rotvec, grip] x 2 -> (T, 20) [p, rot6, grip] x 2."""
    out = []
    for arm in range(2):
        a = a14[:, arm * 7:(arm + 1) * 7]
        out += [a[:, :3], mat_to_6d(rotvec_to_mat(a[:, 3:6])), a[:, 6:7]]
    return np.concatenate(out, axis=1).astype(np.float32)


def from_policy_action(a20):
    """(20,) -> per arm (p, R, grip)."""
    arms = []
    for arm in range(2):
        a = a20[arm * 10:(arm + 1) * 10]
        arms.append((a[:3], sixd_to_mat(a[3:9]), float(a[9])))
    return arms


def decode(jpg, hw=IMG_HW):
    im = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)[..., ::-1]
    return cv2.resize(im, (hw[1], hw[0]), interpolation=cv2.INTER_AREA)


def cache_episode(path, cache_dir):
    """Decode one episode to <cache_dir>/<split>_<stem>.npz (images (N, C, H, W, 3) uint8, actions (N, 20))."""
    path = Path(path)
    out = Path(cache_dir) / f"{path.parent.name}_{path.stem}.npz"
    if out.exists():
        return out
    with h5py.File(path, "r") as f:
        if not bool(f.attrs.get("success", False)):
            return None
        a = to_policy_action(f["action/cartesian_action"][:])[::STRIDE]
        n = len(a)
        imgs = np.stack([np.stack([decode(f[f"render/camera_{c}/jpg"][i * STRIDE]) for c in CAMS]) for i in range(n)])
        meta = dict(shoe=str(f.attrs.get("shoe", "")), seed=int(f.attrs.get("seed", -1)))
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, imgs=imgs, actions=a, meta=json.dumps(meta))
    return out


class ChunkDataset(torch.utils.data.Dataset):
    def __init__(self, files, chunk=16, stats=None):
        self.eps = []
        for f in files:
            d = np.load(f)
            self.eps.append((d["imgs"], d["actions"]))
        self.index = [(e, t) for e, (_, a) in enumerate(self.eps) for t in range(len(a))]
        self.chunk = chunk
        allact = np.concatenate([a for _, a in self.eps])
        self.stats = stats or dict(lo=allact.min(0).tolist(), hi=allact.max(0).tolist())
        self.lo, self.hi = np.array(self.stats["lo"], np.float32), np.array(self.stats["hi"], np.float32)

    def norm(self, a):
        return 2 * (a - self.lo) / np.maximum(self.hi - self.lo, 1e-4) - 1

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        e, t = self.index[i]
        imgs, act = self.eps[e]
        prev = act[max(t - 1, 0)]
        idx = np.minimum(np.arange(t, t + self.chunk), len(act) - 1)
        pad = (np.arange(t, t + self.chunk) >= len(act))
        return (torch.from_numpy(imgs[t]).permute(0, 3, 1, 2).float() / 255.0,
                torch.from_numpy(self.norm(prev)), torch.from_numpy(self.norm(act[idx])), torch.from_numpy(pad))
