"""MP4 of a recorded episode from its stored JPEG frames, cameras side by side (imageio + its bundled FFmpeg).

    python sim/shoe_return/tools/episode_video.py <episode.hdf5> <out.mp4> [--cams 1 0 4] [--fps 15] [--height 360]
"""
import argparse

import cv2
import h5py
import imageio.v2 as imageio
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episode")
    ap.add_argument("out")
    ap.add_argument("--cams", type=int, nargs="+", default=[1, 0, 4])
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--height", type=int, default=360)
    ap.add_argument("--stride", type=int, default=1, help="use every n-th recorded frame")
    args = ap.parse_args()
    f = h5py.File(args.episode, "r")
    jpgs = [f[f"render/camera_{c}/jpg"] for c in args.cams]
    with imageio.get_writer(args.out, fps=args.fps, codec="libx264", quality=7, macro_block_size=8) as w:
        for i in range(0, len(jpgs[0]), args.stride):
            tiles = []
            for j in jpgs:
                im = cv2.imdecode(np.frombuffer(j[i], np.uint8), cv2.IMREAD_COLOR)[..., ::-1]
                s = args.height / im.shape[0]
                tiles.append(cv2.resize(im, (int(round(im.shape[1] * s / 8)) * 8, args.height)))
            w.append_data(np.hstack(tiles))
    print(f"wrote {args.out}: {len(range(0, len(jpgs[0]), args.stride))} frames from cameras {args.cams}, success={f.attrs.get('success')}")


if __name__ == "__main__":
    main()
