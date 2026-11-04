# VR training world for the shoe-return cell

Goal: put a person in VR inside the Isaac Sim shoe-return cell, let them drive both UR10e arms with their hands, record what they do as demonstrations, multiply those with Isaac Lab Mimic, train policies on the result, and watch the trained policy run in the same world.

Status (2026-11-04): headset is a **Meta Quest 3**, running in its **web browser** (WebXR), not a native app or CloudXR. Steps marked **done** ran, but only with a scripted WebSocket client standing in for the headset.

## Why

| Source of demos | Result so far |
|---|---|
| Scripted expert | 13/20 successes (65%) on the 20-seed gate |
| ACT / Diffusion Policy trained on expert demos | 0/10 on held-out shoes |
| Isaac Lab Mimic from expert demos | Pipeline runs; 0 successful generated demos so far |

A human in VR handles the hard cases (slipped shoes, lid that does not seat) that the script gets wrong, and Mimic turns a few good human demos into hundreds.

## Loop

```
 Quest 3 browser (WebXR page, controllers)
          │  WebSocket: controller poses up, JPEG stereo frames down (Wi-Fi HTTPS :8443 or USB adb reverse :8080)
          ▼
 Isaac Sim shoe-return cell  ──► record_demos (HDF5, Mimic format)
          ▲                              │
          │                              ▼
 policy rollout in VR / on screen  ◄── Mimic generation ──► ACT / DP training
```

## What already exists

- **done** Mimic environment `Isaac-ShoeReturn-T1-Mimic-v0` (`sim/shoe_return/mimic/shoe_return_mimic_env.py`). Its action is per arm `[x, y, z, qx, qy, qz, qw, grip]`, an absolute TCP pose: the same kind of command a VR hand pose gives.
- **done** Recording into Mimic HDF5 (`run_sim.py --mimic-record`), generation (`tools/mimic_generate.py`), video replay (`tools/mimic_replay_video.py`).
- In Isaac Lab 3.0 (present in our checkout, not yet tried): `isaaclab.devices.openxr` (`OpenXRDevice`, `XrCfg`), retargeters `Se3AbsRetargeter`, `Se3RelRetargeter`, `GripperRetargeter`, and `scripts/tools/record_demos.py`.

## Steps

1. **done: Quest 3 browser teleop** (`sim/shoe_return/vr/`). Stereo video panel, clutch control of both arms, B saves a Mimic source demo if the task is done. Measured 0.29x real time. Next: try it on the real headset, then speed the sim up (fewer render passes, lower resolution, physics at 60 Hz if grasps stay stable).
2. **Desktop teleop first (no headset).** Wire keyboard / SpaceMouse to the Mimic env through `Se3AbsRetargeter`-style mapping, one arm at a time with a key to switch arms. This proves recording and replay with a human in the loop, and is the fallback when the headset is not around.
3. **XR anchor and scale.** Place the XR origin at standing height in front of the table (`XrCfg.anchor_pos`), 1:1 scale, so the box and shoes appear life size.
4. **Two-hand mapping.** Left hand → left arm TCP, right hand → right arm TCP (absolute pose, offset captured when the user "clutches"). Pinch → gripper close; the action term already accepts open / pinch-open / closed levels.
5. **Safety in the sim.** Keep the per-step joint clamp (±0.025 rad) and IK limits in `UrTcpPoseAction`, so a fast hand motion cannot fling the arm. Show the IK target as a ghost marker when it is unreachable.
6. **Record.** `record_demos.py`-style loop on the Mimic env, saving only successful episodes, plus the subtask term signals already defined (lid opened, shoe grasped, shoe placed).
7. **Generate.** Feed human demos to `mimic_generate.py`, aiming for 10x–50x as many generated demos.
8. **Train and evaluate.** Train ACT / DP on human + generated demos; evaluate on held-out shoes with the existing `run_batch.py --policy` path.
9. **Watch the policy in VR.** Run the trained policy in the same XR session so the user can see it, and take over (clutch) when it fails: those takeovers become new demos (DAgger-style).
10. **More of the world.** Additional tasks in the same cell (unpack a returned box, inspect a shoe, pick from a tote), randomised lighting and boxes, and the camera streams written in the repo's world-model HDF5 format.

## Open questions

- Can the Quest reach the Isaac machine over Wi-Fi (same subnet, ports 8080/8443 open)? Otherwise use USB with `adb reverse`.
- Hands (hand tracking) or controllers? Controllers give a cleaner gripper trigger; hand tracking feels more natural.
- How many human demos per shoe to aim for.
