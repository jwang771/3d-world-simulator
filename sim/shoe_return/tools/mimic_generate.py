"""Generate shoe-return demonstrations with Isaac Lab Mimic from scripted source demos.

Same pipeline as Isaac Lab's scripts/imitation_learning/isaaclab_mimic/generate_dataset.py (setup_env_config,
setup_async_generation, env_loop), but Kit is launched before the env config is built: the config writes the shoe
box USD, and pxr may only be imported once Kit is up.

    cd ../IsaacLab && OMNI_KIT_ACCEPT_EULA=YES uv run python ../3d-world-simulator/sim/shoe_return/tools/mimic_generate.py \
        --input runs/mimic/src_superstar.hdf5 --output runs/mimic/gen_superstar.hdf5 --shoe SUPERSTAR_CLR --trials 10
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE / "mimic"))

from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True, help="source demos (run_sim.py --mimic-record)")
parser.add_argument("--output", required=True, help="generated dataset (.hdf5); successful episodes only")
parser.add_argument("--shoe", default="SUPERSTAR_CLR", help="shoe model (must match the source demos)")
parser.add_argument("--trials", type=int, default=10, help="number of successful demos to generate")
parser.add_argument("--max-failures", type=int, default=50)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--keep-failed", action="store_true", help="also save failed attempts (to <output>_failed.hdf5)")
add_launcher_args(parser)
args = parser.parse_args()


def main():
    import isaaclab.sim as sim_utils

    args.require_kit, args.enable_cameras, args.physics = True, True, "isaacsim_physx"
    with launch_simulation(sim_utils.SimulationCfg(dt=1 / 120, device=args.device), args):
        import asyncio
        import random

        import gymnasium as gym
        import numpy as np
        import torch

        import shoe_return_mimic_env as M
        from isaaclab_mimic.datagen.generation import env_loop, setup_async_generation, setup_env_config
        from isaaclab_mimic.datagen.utils import setup_output_paths

        out = Path(args.output).resolve()
        M.configure(shoe=args.shoe, cache=str(out.parent / "_mimic_assets"), cameras=False)
        M.register()
        output_dir, output_name = setup_output_paths(str(out))
        env_cfg, success_term = setup_env_config(env_name=M.TASK_ID, output_dir=output_dir, output_file_name=output_name, num_envs=1,
                                                 device=args.device or "cuda:0", generation_num_trials=args.trials)
        if args.keep_failed:
            from isaaclab.managers import DatasetExportMode

            env_cfg.datagen_config.generation_keep_failed = True
            env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_FAILED_IN_SEPARATE_FILES
        env_cfg.datagen_config.max_num_failures = args.max_failures
        env_cfg.datagen_config.seed = args.seed
        env = gym.make(M.TASK_ID, cfg=env_cfg).unwrapped
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        env.reset()
        comps = setup_async_generation(env=env, num_envs=1, input_file=str(Path(args.input).resolve()), success_term=success_term,
                                       pause_subtask=False, motion_planners=None)
        tasks = asyncio.ensure_future(asyncio.gather(*comps["tasks"]))
        try:
            env_loop(env, comps["reset_queue"], comps["action_queue"], comps["info_pool"], comps["event_loop"], data_gen_tasks=tasks)
        finally:
            tasks.cancel()
            try:
                comps["event_loop"].run_until_complete(tasks)
            except (asyncio.CancelledError, Exception):
                pass
        env.close()


if __name__ == "__main__":
    main()
