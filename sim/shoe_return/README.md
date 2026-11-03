# Shoe returns simulation (Isaac Lab 3.0 / Isaac Sim 6.2)

A returns-handling cell that generates training data for this repo's 3D world model (`--task shoe_return`). Status and history are tracked in [../../PROGRESS.md](../../PROGRESS.md). **The full T1 task does not yet succeed end to end; see the progress log for what works.**

## Cell

| Component | What it is |
|---|---|
| Robots | Two UR10e arms, each with a Robotiq 2F-85 gripper. The spec allows any UR arm. The Isaac Sim 6.2 UR5e asset's gripper variant is unstable (see `PROGRESS.md`). |
| Box | A shoebox articulation with a damped hinged lid and a pull tab. It sits fixed in a nest on the table and is sized to the shoe pair. |
| Shoes | A true left/right pair from a library of 63 scanned shoes (Google Scanned Objects, CC-BY 4.0). Each has photographic textures, CoACD convex colliders and per-shoe grasp metadata. |
| Cameras | Four RTX RGB-D cameras: overhead (0), front (1), left wrist (3) and right wrist (4). |

## Expert (scripted)

| Part | How it works |
|---|---|
| Kinematics | Joint-space transits and Cartesian approaches run on a calibrated UR model in `ur_kin.py`. IK is limit-aware and branch-aware. |
| Lid | Closed-loop on the measured hinge angle, world-anchored at grasp time. |
| Placement | Closed-loop on the measured heel pose. Grasps that slip are retried. |

## Run

```bash
cd ../IsaacLab                                    # Isaac Lab v3.0.0 checkout, `uv sync --extra isaacsim --extra rsl-rl --extra video`
OMNI_KIT_ACCEPT_EULA=YES uv run python ../3d-world-simulator/sim/shoe_return/run_sim.py \
    --out ../3d-world-simulator/data/shoe_return_sim --episodes 20 --shoe random [--debug] [--keep-failures]
```

Headless is the default; add `--viz kit` for the GUI.

## Build the shoe library

The library is git-ignored, about 0.7 GB.

```bash
python sim/shoe_return/tools/build_shoe_assets.py --src <folder of GSO model folders> --out sim/shoe_return/assets/shoes
```

## Tests and tools

| File | What it does | Needs |
|---|---|---|
| `test_writer_compat.py` | Checks that writer output loads through the repo's readers | No Isaac |
| `test_scene_geometry.py` | Checks box, lid, tab and spawn geometry | No Isaac |
| `tools/check_dataset.py` | Validates a generated dataset | |
| `tools/*_probe.py` | Isaac probes used to debug the IK, the lid kinematics and the gripper | Isaac |

## Files

| File | Role |
|---|---|
| `run_sim.py` | Expert and episode loop |
| `envs/repack_scene.py` | Isaac Lab scene config |
| `scene.py` | Layout and USD asset builders |
| `ur_kin.py` | UR kinematics |
| `writer.py` | Episode writer |
| `geometry.py` | Pose maths |
