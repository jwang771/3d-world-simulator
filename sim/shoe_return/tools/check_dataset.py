"""Validates a generated shoe_return dataset: schema, frame counts, NaNs, and loading through the repo's readers.

    python sim/shoe_return/tools/check_dataset.py --root data/shoe_return_sim [--frames 3]

Exits non-zero if any episode fails; prints one line per episode.
"""
import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from datasets.aloha_packed import AlohaPackedPCDataset  # noqa: E402
from datasets.frames import Frames  # noqa: E402
from models.cached_robot_labels import load_frame  # noqa: E402
from tasks import TASKS  # noqa: E402

REQUIRED_ATTRS = ["episode", "num_frames", "H", "W", "cam_ids", "bbox", "has_obj_mask", "depth_semantics"]
PER_CAM = ["cam/camera_{c}/K_ir", "cam/camera_{c}/E_cam2world", "img/camera_{c}/depth", "img/camera_{c}/valid_mask", "img/camera_{c}/rgb_in_ir", "img/camera_{c}/obj_mask", "render/camera_{c}/jpg", "render/camera_{c}/K", "render/camera_{c}/viewmat"]


def check_episode(path, task, n_frames):
    problems = []
    root = path.parent.parent
    with h5py.File(path, "r") as f:
        for a in REQUIRED_ATTRS:
            if a not in f.attrs:
                problems.append(f"missing attr {a}")
        T = int(f.attrs["num_frames"])
        ep = int(f.attrs["episode"])
        cams = [int(c) for c in f.attrs["cam_ids"]]
        if f["action/cartesian_action"].shape != (T, 14):
            problems.append(f"action shape {f['action/cartesian_action'].shape}")
        if not np.isfinite(f["action/cartesian_action"][:]).all():
            problems.append("NaN/inf in actions")
        for c in cams:
            for key in PER_CAM:
                k = key.format(c=c)
                if k not in f:
                    problems.append(f"missing {k}")
                elif f[k].shape[0] != T:
                    problems.append(f"{k} has {f[k].shape[0]} frames, expected {T}")
            d = f[f"img/camera_{c}/depth"]
            for t in (0, T // 2, T - 1):
                if not np.isfinite(d[t].astype(np.float32)).all():
                    problems.append(f"camera {c} depth not finite at frame {t}")
            for name in ("K_ir", "E_cam2world"):
                if not np.isfinite(f[f"cam/camera_{c}/{name}"][:]).all():
                    problems.append(f"camera {c} {name} not finite")
    for side, rel in (("fs_disp", f"fs_disp/disp_episode_{ep}.hdf5"), ("gripper", f"gripper_supervision/masks_halo6_surface/episode_{ep}.h5"), ("robot", f"robot_label_cache/episode_{ep}.h5")):
        if not (root / rel).exists():
            problems.append(f"missing side file {rel}")
    if problems:
        return problems
    bounds = [[task["x_range"][0], task["y_range"][0], task["z_range"][0]], [task["x_range"][1], task["y_range"][1], task["z_range"][1]]]
    ds = AlohaPackedPCDataset([path.parent], n_points=20000, point_bounds=bounds)
    fi = [i for i, fp in enumerate(ds.files) if Path(fp).resolve() == path.resolve()][0]
    idx = [i for i, (e, _) in enumerate(ds.index) if e == fi]
    picks = [idx[int(j)] for j in np.linspace(0, len(idx) - 1, n_frames)]
    frames = Frames(ds.files, "cpu")
    nx, ny, nz = task["grid"]
    shapes = [(nx, ny, nz), (2 * nx, 2 * ny, 2 * nz)]
    for gi in picks:
        s = ds[gi]
        t = int(s["frame_idx"])
        if not torch.isfinite(s["xyz"]).all():
            problems.append(f"non-finite points at frame {t}")
        if int(s["obj_label"].sum()) == 0:
            problems.append(f"no shoe points at frame {t}")
        frames.render(fi, t)
        frames.depth(fi, t)
        load_frame(root / "robot_label_cache", ep, t, shapes, torch.tensor(bounds[0]), torch.tensor(bounds[1]), "cpu")
    return problems


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--task", default="shoe_return")
    p.add_argument("--frames", type=int, default=3, help="frames per episode loaded through the readers")
    a = p.parse_args()
    task = TASKS[a.task]
    files = sorted(Path(a.root).glob("*/episode_*.hdf5"))
    if not files:
        sys.exit(f"no episodes under {a.root}")
    bad = 0
    for fp in files:
        try:
            problems = check_episode(fp, task, a.frames)
        except Exception as e:  # a loader crash is a failed check, not a checker crash
            problems = [f"{type(e).__name__}: {e}"]
        bad += bool(problems)
        print(f"{'FAIL' if problems else 'ok  '} {fp.relative_to(a.root)}" + ("" if not problems else ": " + "; ".join(problems[:5])))
    print(f"{len(files) - bad}/{len(files)} episodes pass")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
