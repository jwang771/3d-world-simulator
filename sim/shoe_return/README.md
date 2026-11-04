# Shoe returns simulation (Isaac Lab 3.0 / Isaac Sim 6.2)

A returns-handling cell that generates training data for this repo's 3D world model (`--task shoe_return`), plus imitation-learning baselines trained on that data. Status, measured success rates and history are in [../../PROGRESS.md](../../PROGRESS.md). **The Phase 0 gate (18/20 scripted successes) is not met yet; see the progress log for the measured rate.**

## Cell

| Component | What it is |
|---|---|
| Robots | Two UR10e arms, each with a Robotiq 2F-85 gripper and silicone pads (friction 1.5). The spec allows any UR arm. The Isaac Sim 6.2 UR5e gripper variant is unstable (see `PROGRESS.md`). Both 2F-85 knuckles are driven: driving only one, as the stock config does, twisted the wrist whenever the gripper squeezed. |
| Box | A shoebox articulation sized to the pair, with 6 cm length play, 4 cm width play and 2.5 cm headroom. It sits fixed in a nest, with the hinge facing the left arm. The lid hinge is creased paperboard: static friction 0.06 N m holds the lid where it is left above ~70 deg. |
| Shoes | A true left/right pair from 63 scanned shoes (Google Scanned Objects, CC-BY 4.0). Each has photographic textures, CoACD convex colliders and per-shoe grasp metadata. 51 shoes are used for training and 12 are held out. |
| Cameras | Four RTX RGB-D cameras: overhead (0), front (1), left wrist (3) and right wrist (4). |

## Scripted expert (`run_sim.py`)

1. The left arm pinches the lid tab and swings the lid open past vertical. It lets go and clears the box.
2. The right arm picks shoe 1 by a collar pinch: one finger inside the collar, one behind the heel. It carries the shoe high and slowly in joint space, using collision-checked transits.
3. It sets the shoe down flat in its slot, steering closed-loop on the measured heel. It then opens fully, centres the inside finger in the collar and lifts. A slipped, hooked or propped shoe is re-picked once.
4. The same for shoe 2, heel-to-toe.
5. The left arm regrasps the tab, swings the lid shut, and presses it down if it does not seat.

| Part | How it works |
|---|---|
| Kinematics | `ur_kin.py`: DH model calibrated against the simulator (0.00 mm). IK is limit-, clearance- and branch-aware, and unwinds wrists. |
| Transits | `collision.py`: sphere model of both arms and the held shoe, checked against the table, box, lid and other arm. Detours go via low or raised via-points. |

## Run

```bash
cd ../IsaacLab                                    # Isaac Lab v3.0.0 checkout, `uv sync --extra isaacsim --extra rsl-rl --extra video`
OMNI_KIT_ACCEPT_EULA=YES uv run python ../3d-world-simulator/sim/shoe_return/run_sim.py \
    --out ../3d-world-simulator/data/shoe_return_sim --episodes 1 --shoe random [--debug] [--keep-failures]
# many episodes, one Isaac process each (each gets its own shoe pair):
uv run --no-sync python ../3d-world-simulator/sim/shoe_return/tools/run_batch.py --out <root> --episodes 20 --seed 1000 -- --width 320 --height 240
```

Headless is the default; add `--viz kit` for the GUI. Running several Isaac processes at once thrashes the CPU: on 32 threads, one process runs ~53 physics steps/s, and two together are the useful maximum.

## Policies (`policy/`)

- **Models:** ACT and Diffusion Policy in plain PyTorch, with a torchvision ResNet-18 backbone.
- **Data:** they learn from the overhead and both wrist cameras plus the previous command. Each predicts 16-step chunks of both arms' TCP targets (position, 6-D rotation, gripper) at 7.5 Hz.

```bash
uv run --no-sync python ../3d-world-simulator/sim/shoe_return/policy/train.py --config ../3d-world-simulator/sim/shoe_return/policy/configs/act.json --data <root> --out runs/policy/act
# closed-loop evaluation on held-out shoes, same IK and success check as the expert:
uv run --no-sync python ../3d-world-simulator/sim/shoe_return/tools/run_batch.py --out runs/eval_act --episodes 10 --seed 9000 -- --policy runs/policy/act/model.pt --shoe random:test --no-write --width 320 --height 240
```

## Build the shoe library

The library is git-ignored, about 0.7 GB.

```bash
python sim/shoe_return/tools/build_shoe_assets.py --src <folder of GSO model folders> --out sim/shoe_return/assets/shoes
```

## Tests and tools

| File | What it does | Needs |
|---|---|---|
| `test_writer_compat.py` | Checks that writer output loads through the repo's readers | No Isaac |
| `test_scene_geometry.py` | Checks box, lid, tab, spawn geometry and task bounds | No Isaac |
| `test_ur_kin.py` | FK/IK round trips, limits, clearance, unwinding | No Isaac |
| `test_collision.py` | Sphere model, hits, detour planning | No Isaac |
| `tools/check_dataset.py` | Validates a generated dataset through the repo loaders | |
| `tools/episode_video.py` | MP4 of an episode from its stored frames | |
| `tools/run_batch.py` | Parallel or sequential single-episode runs, merged into one dataset | Isaac |
| `tools/*_probe.py` | Isaac probes used to debug the IK, the lid and the gripper | Isaac |

## Files

| File | Role |
|---|---|
| `run_sim.py` | Expert, policy evaluation and episode loop |
| `envs/repack_scene.py` | Isaac Lab scene config |
| `scene.py` | Layout and USD asset builders |
| `ur_kin.py` | UR kinematics |
| `collision.py` | Transit collision model |
| `writer.py` | Episode writer |
| `geometry.py` | Pose maths |
| `policy/` | ACT, Diffusion Policy, data loading, training |
