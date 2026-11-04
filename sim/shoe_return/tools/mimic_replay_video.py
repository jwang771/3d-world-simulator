"""Replay demos from a Mimic dataset in the shoe-return env with cameras on, and write an MP4 per demo
(front, overhead and right-wrist cameras side by side).

    cd ../IsaacLab && OMNI_KIT_ACCEPT_EULA=YES uv run python ../3d-world-simulator/sim/shoe_return/tools/mimic_replay_video.py \
        --dataset runs/mimic/gen_superstar.hdf5 --shoe SUPERSTAR_CLR --out runs/mimic/videos [--demos demo_0 demo_1]
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "mimic"))

from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True)
parser.add_argument("--shoe", default="SUPERSTAR_CLR")
parser.add_argument("--out", required=True, help="output directory for the MP4 files")
parser.add_argument("--demos", nargs="*", default=None, help="demo names (default: all)")
parser.add_argument("--every", type=int, default=8, help="physics steps per video frame")
parser.add_argument("--fps", type=int, default=30)
add_launcher_args(parser)
args = parser.parse_args()


def main():
    import isaaclab.sim as sim_utils

    args.require_kit, args.enable_cameras, args.physics = True, True, "isaacsim_physx"
    with launch_simulation(sim_utils.SimulationCfg(dt=1 / 120, device=args.device), args):
        import cv2
        import gymnasium as gym
        import imageio.v2 as imageio
        import numpy as np
        import torch

        import shoe_return_mimic_env as M
        from isaaclab.utils.datasets import HDF5DatasetFileHandler

        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        M.configure(shoe=args.shoe, cache=str(out / "_mimic_assets"), cameras=True, width=640, height=480)
        M.register()
        cfg = M.ShoeReturnMimicEnvCfg()
        cfg.scene.num_envs = 1
        cfg.sim.device = args.device or "cuda:0"
        cfg.terminations.time_out = None
        cfg.terminations.success = None
        env = gym.make(M.TASK_ID, cfg=cfg).unwrapped
        env.reset()
        h = HDF5DatasetFileHandler()
        h.open(args.dataset)
        names = args.demos or list(h.get_episode_names())
        for name in names:
            ep = h.load_episode(name, env.device)
            env.reset_to(ep.data["initial_state"], torch.tensor([0], device=env.device), is_relative=True)
            actions = ep.data["actions"]
            path = out / f"{Path(args.dataset).stem}_{name}.mp4"
            with imageio.get_writer(str(path), fps=args.fps, codec="libx264", quality=7, macro_block_size=8) as w:
                for i in range(len(actions)):
                    env.step(actions[i][None])
                    if i % args.every == 0:
                        tiles = []
                        for cam in ("camera_1", "camera_0", "camera_4"):
                            rgb = env.scene[cam].data.output["rgb"]
                            rgb = (rgb.torch if hasattr(rgb, "torch") else rgb)[0, ..., :3].cpu().numpy()
                            tiles.append(cv2.resize(rgb, (424, 320)))
                        w.append_data(np.hstack(tiles))
            ok = bool(M.task_success(env)[0])
            print(f"[replay] {name}: {len(actions)} steps, task success at the end: {ok} -> {path}", flush=True)
        env.close()


if __name__ == "__main__":
    main()
