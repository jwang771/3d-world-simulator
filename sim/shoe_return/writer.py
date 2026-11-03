"""Writes simulated episodes in the packed format of docs/data_format.md.

No Isaac dependency: it takes numpy frames, so it can be tested on its own.
Per episode it writes
  <root>/<split>/episode_<E>.hdf5
  <root>/fs_disp/disp_episode_<E>.hdf5                    (synthetic stereo from rendered depth)
  <root>/gripper_supervision/masks_halo6_surface/episode_<E>.h5
  <root>/robot_label_cache/episode_<E>.h5
"""
import os
from pathlib import Path

import cv2
import h5py
import numpy as np

GRIPPER_HALO_PX = 6
DEPTH_RANGE = (0.1, 3.0)
EDGE_JUMP = 0.03


def pack(mask):
    return np.packbits(np.asarray(mask, bool).reshape(-1))


def level_shapes(grid):
    """zyx grid shapes for each decoder level (models/build.py: two levels, x2 upsampling)."""
    nx, ny, nz = grid
    return [(nz, ny, nx), (2 * nz, 2 * ny, 2 * nx)]


def voxelize_boxes(boxes, bounds, shape_zyx):
    """Boolean z, y, x grid of the cells whose centres fall inside any oriented box.

    boxes: iterable of (T_world_box 4x4, half_extents 3). bounds: 2 x 3 (min, max).
    """
    bounds = np.asarray(bounds, np.float64)
    nz, ny, nx = shape_zyx
    n = np.array([nx, ny, nz])
    vs = (bounds[1] - bounds[0]) / n
    grid = np.zeros(shape_zyx, bool)
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], np.float64)
    for T, half in boxes:
        T = np.asarray(T, np.float64)
        half = np.asarray(half, np.float64)
        corners = (signs * half) @ T[:3, :3].T + T[:3, 3]
        lo = np.clip(np.floor((corners.min(0) - bounds[0]) / vs).astype(int), 0, n)
        hi = np.clip(np.ceil((corners.max(0) - bounds[0]) / vs).astype(int), 0, n)
        if np.any(hi <= lo):
            continue
        ix, iy, iz = (np.arange(lo[a], hi[a]) for a in range(3))
        X, Y, Z = np.meshgrid(ix, iy, iz, indexing="ij")
        idx = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
        centres = bounds[0] + (idx + 0.5) * vs
        local = (centres - T[:3, 3]) @ T[:3, :3]
        inside = np.all(np.abs(local) <= half + 1e-9, axis=1)
        idx = idx[inside]
        grid[idx[:, 2], idx[:, 1], idx[:, 0]] = True
    return grid


def downsample(grid_zyx):
    z, y, x = grid_zyx.shape
    return grid_zyx.reshape(z // 2, 2, y // 2, 2, x // 2, 2).any(axis=(1, 3, 5))


def filter_mask(depth, valid):
    """valid_mask minus depth-discontinuity pixels (the sim's stand-in for outlier removal)."""
    d = np.where(valid, depth, 0).astype(np.float32)
    k = np.ones((3, 3), np.uint8)
    hi = cv2.dilate(d, k)
    lo = cv2.erode(np.where(valid, d, np.float32(1e3)), k)
    return valid & ((hi - lo) < EDGE_JUMP)


class _Appender:
    def __init__(self, h5):
        self.h5, self.ds = h5, {}

    def __call__(self, name, arr, compress=True):
        arr = np.asarray(arr)
        ds = self.ds.get(name)
        if ds is None:
            kw = dict(compression="gzip", compression_opts=4) if compress and arr.size > 64 else {}
            ds = self.ds[name] = self.h5.create_dataset(
                name, shape=(0, *arr.shape), maxshape=(None, *arr.shape), dtype=arr.dtype, chunks=(1, *arr.shape), **kw
            )
        t = ds.shape[0]
        ds.resize(t + 1, axis=0)
        ds[t] = arr

    def bytes(self, name, buf):
        ds = self.ds.get(name)
        if ds is None:
            ds = self.ds[name] = self.h5.create_dataset(name, shape=(0,), maxshape=(None,), dtype=h5py.vlen_dtype(np.uint8))
        t = ds.shape[0]
        ds.resize(t + 1, axis=0)
        ds[t] = np.frombuffer(buf, np.uint8)


class EpisodeWriter:
    def __init__(self, root, split, episode, cam_ids, task, render_cam_ids=None, stereo_baseline=0.05, jpeg_quality=92):
        self.root, self.episode = Path(root), int(episode)
        self.cam_ids = [int(c) for c in cam_ids]
        self.render_cam_ids = [int(c) for c in (render_cam_ids or cam_ids)]
        self.bounds = np.array(
            [[task["x_range"][0], task["y_range"][0], task["z_range"][0]], [task["x_range"][1], task["y_range"][1], task["z_range"][1]]],
            np.float32,
        )
        self.shapes = level_shapes(task["grid"])
        self.baseline, self.jpeg_quality = float(stereo_baseline), int(jpeg_quality)
        name = f"episode_{self.episode}"
        self.paths = {
            "main": self.root / split / f"{name}.hdf5",
            "disp": self.root / "fs_disp" / f"disp_{name}.hdf5",
            "grip": self.root / "gripper_supervision" / "masks_halo6_surface" / f"{name}.h5",
            "robot": self.root / "robot_label_cache" / f"{name}.h5",
        }
        self.files = {}
        for k, p in self.paths.items():
            p.parent.mkdir(parents=True, exist_ok=True)
            self.files[k] = h5py.File(str(p) + ".tmp", "w")
        self.app = {k: _Appender(f) for k, f in self.files.items()}
        self.T, self.HW = 0, None

    def _jpg(self, rgb):
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        assert ok
        return buf.tobytes()

    def add_frame(self, action, cams, robot_boxes, gripper_boxes):
        """action: (14,). cams: {cid: dict(rgb HxWx3 uint8, depth HxW metres, obj_mask HxW uint8,
        gripper_mask HxW bool, K 3x3, E_cam2world 4x4)}. *_boxes: [(T_world_box, half_extents)]."""
        m, d, g, r = self.app["main"], self.app["disp"], self.app["grip"], self.app["robot"]
        m("action/cartesian_action", np.asarray(action, np.float32).reshape(14), compress=False)
        for c in self.cam_ids:
            cam = cams[c]
            depth = np.nan_to_num(np.asarray(cam["depth"], np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            H, W = depth.shape
            self.HW = self.HW or (H, W)
            assert (H, W) == self.HW, "all cameras must share one depth size"
            valid = (depth > DEPTH_RANGE[0]) & (depth < DEPTH_RANGE[1])
            depth[~valid] = 0.0
            K = np.asarray(cam["K"], np.float32)
            E = np.asarray(cam["E_cam2world"], np.float32)
            obj = np.asarray(cam["obj_mask"], np.uint8)
            jpg = self._jpg(np.asarray(cam["rgb"], np.uint8))
            m(f"cam/camera_{c}/K_ir", K, compress=False)
            m(f"cam/camera_{c}/E_cam2world", E, compress=False)
            m(f"img/camera_{c}/depth", depth.astype(np.float16))
            m(f"img/camera_{c}/valid_mask", pack(valid))
            m(f"img/camera_{c}/filter_mask", pack(filter_mask(depth, valid)))
            m.bytes(f"img/camera_{c}/rgb_in_ir", jpg)
            m(f"img/camera_{c}/obj_mask", obj)
            if c in self.render_cam_ids:
                # The render crop is the full image; colour and depth come from one simulated sensor.
                m.bytes(f"render/camera_{c}/jpg", jpg)
                m(f"render/camera_{c}/K", K, compress=False)
                m(f"render/camera_{c}/viewmat", np.linalg.inv(E.astype(np.float64)).astype(np.float32), compress=False)
                m(f"render/camera_{c}/obj_mask", pack(obj > 0))
                m(f"render/camera_{c}/valid_mask", pack(valid))
                grip = cv2.dilate(np.asarray(cam["gripper_mask"], np.uint8), np.ones((2 * GRIPPER_HALO_PX + 1,) * 2, np.uint8)) > 0
                g(f"camera_{c}/gripper_mask", pack(grip))
            disp = np.where(valid, float(K[0, 0]) * self.baseline / np.maximum(depth, 1e-6), 0.0).astype(np.float32)
            d(f"disp/camera_{c}", disp)
            d(f"meta/camera_{c}/baseline", np.float32(self.baseline), compress=False)
        for kind, boxes in (("robot", robot_boxes), ("gripper", gripper_boxes)):
            fine = voxelize_boxes(boxes, self.bounds, self.shapes[-1])
            grids = [fine]
            for _ in self.shapes[:-1]:
                grids.insert(0, downsample(grids[0]))
            for lv, grid in enumerate(grids):
                r(f"{kind}/{lv}", pack(grid))
        self.T += 1

    def close(self, extra_attrs=None):
        H, W = self.HW
        f = self.files["main"]
        f.attrs.update(
            episode=self.episode,
            num_frames=self.T,
            H=H,
            W=W,
            cam_ids=np.array(self.cam_ids, np.int64),
            render_cam_ids=np.array(self.render_cam_ids, np.int64),
            bbox=np.array([self.bounds[0, 0], self.bounds[1, 0], self.bounds[0, 1], self.bounds[1, 1], self.bounds[0, 2], self.bounds[1, 2]], np.float32),
            has_obj_mask=True,
            filter_kind="sim_depth_edge",
            depth_semantics="full_scene_lrc",
            source="isaac_sim_shoe_return",
        )
        f.attrs.update(extra_attrs or {})
        for c in self.render_cam_ids:
            f[f"render/camera_{c}"].attrs.update(H=H, W=W, crop=np.array([0, 0, W, H], np.int64))
        rf = self.files["robot"]
        rf.attrs["bounds"] = self.bounds
        for kind in ("robot", "gripper"):
            for lv, shape in enumerate(self.shapes):
                rf[f"{kind}/{lv}"].attrs["shape"] = np.array(shape, np.int64)
        for k, fh in self.files.items():
            fh.close()
            os.replace(str(self.paths[k]) + ".tmp", self.paths[k])

    def discard(self):
        for k, fh in self.files.items():
            fh.close()
            Path(str(self.paths[k]) + ".tmp").unlink(missing_ok=True)
