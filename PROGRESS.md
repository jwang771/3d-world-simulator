# Shoe Returns Sim: progress log

Status words used below: **ran and passed**, **ran and failed**, **not run**. Nothing is ticked without evidence.

## Milestones

- [ ] M0 Isaac Sim and Isaac Lab installed and pinned; 18 of 20 scripted T1 episodes succeed; video committed
- [ ] M1 Asset library of 30+ shoe models and 3 box types; T1 still at 90%+; asset gallery rendered
- [ ] M2 Tasks T1 to T7 at 85%+ scripted success; T8 at 60%+; failure labels and language instructions recorded
- [ ] M3 Data engine: scripted, teleop and Mimic-augmented data; medium dataset (10k episodes) validated with a dataset card
- [ ] M4 ACT and Diffusion Policy baselines; 70%+ success on T1 with held-out shoes; one-command evaluation report
- [ ] M5 World model trained on shoe-return data; rollout, consistency, counterfactual and policy-ranking results reported
- [ ] M6 Sensitivity sweep, sim-to-real checklist, frozen benchmark v1.0, reproducible from a fresh clone

## Pinned stack (Phase 0, step 2)

| Component | Version | Source |
|---|---|---|
| Isaac Lab | v3.0.0 (commit e3f7d9ff1f5d, released 2026-10-20) | github.com/isaac-sim/IsaacLab, "recommended baseline for new projects" |
| Isaac Sim | 6.2.0 (via Isaac Lab's `isaacsim` extra) | Isaac Lab 3.0 install guide: "Isaac Sim 5.1 and older are unsupported" |
| Python | 3.12 | required by Isaac Lab 3.0 |
| PyTorch / CUDA | 2.11 / 13.0 | Isaac Lab 3.0 release notes |
| Install method | `uv sync --extra isaacsim --extra rsl-rl --extra mimic --extra video` in an Isaac Lab checkout at `../IsaacLab` | official "automatic setup with uv" path |

The lockfile is Isaac Lab's own `uv.lock` at that commit; it gets copied into `sim/shoe_return/` once the install is verified.

## Session log

### 2026-11-03, session 1

**Machine (Phase 0, step 1): ran and passed.**

| Item | Found | Isaac Lab 3.0 / Isaac Sim 6.2 needs |
|---|---|---|
| OS | Ubuntu 24.04.5, GLIBC 2.39 | Ubuntu 22.04+, GLIBC 2.35+ |
| Driver | NVIDIA 580.178.04 | 580.65.06+ |
| GPU | RTX 4090, 24 GB VRAM | RTX, 16 GB+ |
| RAM | 62 GB | 32 GB+ |
| Disk | 823 GB free | n/a |
| CPU | 32 threads | n/a |

**What existed before this session:** `sim/shoe_return/` with the scene, scripted episode, writer and two offline tests, all written against the Isaac Sim 5.x API and never run in Isaac.

**Decisions:**
- Switched from the Isaac Sim 5.1 install I had started to Isaac Lab 3.0 + Isaac Sim 6.2. 5.1 is unsupported by current Isaac Lab, and 3.0 adds the Newton VBD deformables needed for tissue paper and soft uppers in Phase 1. The 5.1 install was stopped before it finished.
- Realistic shoes will come from Google Scanned Objects (CC-BY 4.0, 256 shoe scans). The scans are hollow with the collar opening captured, so the grasp can be the realistic one: one finger inside the collar, one behind the heel, pinching the heel counter. Real heels (about 9 cm) are wider than the 2F-85's 8.5 cm stroke, so a whole-heel grasp is not possible on real shoes.

**Ran:**
- Offline tests (`test_writer_compat.py`, `test_scene_geometry.py`): ran and passed (before this session's changes).
- `sim/shoe_return/tools/build_shoe_assets.py` (scan -> oriented, CoACD collision, heel-pinch grasp, textured USD, mirrored right shoe): running on a sample.

**Not run:** anything in Isaac.

**Install (Phase 0, step 2): ran and passed.** `uv sync --extra isaacsim --extra rsl-rl --extra video` in `../IsaacLab` (v3.0.0): isaacsim 6.2.0.0, isaaclab 28.0.0, torch 2.11.0+cu130, warp 1.17.0. Isaac Sim requires accepting the NVIDIA Omniverse EULA (`OMNI_KIT_ACCEPT_EULA=YES`); it was accepted on the user's instruction to install and run Isaac. The `mimic` extra is **not installed**: its `robomimic -> egl-probe` dependency builds from source and this machine has no C compiler (`make`/`gcc` absent, `sudo` needs a password). Needed for Phase 3; fix with `sudo apt install build-essential`.

**Port (Phase 0, step 3): ran.** `scene.py` is now pure USD/numpy (asset builders: table, shoebox articulation, procedural shoes). `envs/repack_scene.py` is the Isaac Lab scene. `run_sim.py` drives the arms with Isaac Lab differential IK. API facts learned the hard way (all from running, not memory):
- Isaac Lab 3.0 quaternions are (x, y, z, w); asset data is read through `.torch`; there is no `--headless` flag (headless is the default, `--viz kit` opens the GUI).
- `pxr` from pip `usd-core` must not be imported before Kit starts, or Kit's USD 25.11 libraries fail to load.
- A scene built after `launch_simulation` needs `require_kit`, `enable_cameras` set explicitly or Kit starts without the RTX renderer.
- Semantic segmentation comes back as packed RGBA uint32 with colour-string label keys even with colorize off.

**Robot (decision, with evidence):** switched from UR5e to **UR10e + Robotiq 2F-85** (the spec says "e.g. UR5e"). The Isaac Sim 6.2 UR5e asset's 2F-85 Gripper variant is physically unstable: with every arm joint held and no IK, closing the gripper spins wrist_3 without bound (to 1e22 deg in 600 steps). The UR10e asset with *identical* settings holds to 0.8 deg (`tools/gripper_hold_probe.py`). Joints, mimic settings, DoF order and link masses are identical between the two (`tools/gripper_diff_probe.py`); the difference is how the gripper is mounted (UR10e: through `ee_link`). Stripping the UR5e gripper's nested ArticulationRootAPI did not help.

**Physics parameters changed (step 5), and why:**
| Parameter | Value | Why |
|---|---|---|
| UR10e joint effort / speed | 330/150/56 Nm, 120-180 deg/s (datasheet) | Isaac Lab's UR10e config is uncapped; a collision launched the box |
| 2F-85 `finger_joint` drive | stiffness 40, damping 1, 10 Nm | Isaac Lab gear-assembly tuning; default inner-finger drive destabilised the linkage |
| 2F-85 inner fingers | stiffness 10, damping 0.05 | same source; keeps pads parallel |
| 2F-85 close target | 0.72 rad | stays off the 0.82 rad joint limit |
| Finger pad friction | 1.2 | silicone pads; none was bound after the port |
| Lights | dome 350, key 500 | 900/1800 washed out the RTX render |

**Episodes run so far (all single debug episodes, procedural shoes):** UR5e runs all failed (arm/box explosions, root-caused above). First UR10e run: **ran and failed** (`shoe_outside_box`): the lid opened to 105 deg and closed to 0 deg, shoe 1 ended in the box, shoe 2 slipped out of the heel pinch, and the open-loop lid arc dragged the box 25 cm and stood it on end while closing. Fixes in progress: closed-loop lid swing (target = measured hinge + 6 deg), regrasp when a lift fails, deeper heel pinch.

**Shoe library: ran and passed (asset build only, not yet in Isaac).** 63 of 255 Google Scanned Objects shoes built (rejections: 104 collar filled by convex hulls, 47 outside the size envelope, 17 collar too short, 14 heel wall, 9 collar closed, 1 crash).

**Not run yet:** 20-episode gate, video export (the bundled OpenCV has no FFmpeg; frames are in the HDF5), dataset checker on Isaac output.

**Later the same session: expert redesigns, each driven by an observed failure (all ran; none has yet produced a successful episode):**
| Observed failure (from recorded frames + debug log) | Change |
|---|---|
| Open-loop lid arc drags the box 25-35 cm and stands it on end | Lid motion became closed-loop on the measured hinge angle, world-anchored at contact (target-follows-box was self-reinforcing) |
| Rigid pinch on a pull tab: weak grip slips, strong grip lifts the whole box | **Pull tab removed; lid gets a front lip.** Left arm opens it with closed fingertips under the lip (contact only, compliant), holds it at 80 deg, lowers it at the end. This is also the spec's "left arm holds the lid open" |
| Lip unreachable (arm would point at its own base) | Box turned to face the robots (hinge far side, lip near side) and moved 12 cm further out; reach checked offline (wrist 0.36-0.96 m from shoulder) |
| Cartesian differential IK flips the UR10e near singularities on long transits | `ur_kin.py`: UR DH forward kinematics + numerical IK (unit-tested, 200/200 round trips); transits are joint-space `MoveJ`; DH model calibrated against the sim at start (0.00 mm error) |
| IK picked an elbow angle of 213 deg (UR elbow limit is 180) so the arm could not reach the lip | Branch-aware IK: 12 seeds, joint limits enforced, least joint motion (100/100 offline) |
| Placement solved once from the commanded pose; real hand lags 5 cm, shoe dropped beside the box | Placement closes the loop on the measured shoe pose before descending |
| Procedural box-shoes slip out of a heel pinch | **Switched to scanned GSO pairs** (`--shoe`), box sized per pair, mirrored right foot |
| Collar pinch slips on the convex-hull heel wall | Whole-heel clamp from outside, below the collar (median heel width 59 mm; 62/63 shoes fit the 85 mm stroke) |
| Tapered heel wedges the 2F-85 open on lift at 10 Nm | Finger drive raised to 30 Nm (real 2F-85 grips 20-235 N) |
| Lift check fooled by heel-held shoe pivoting toe-down | Check grasp-point height, not shoe origin |
| Contact-only lid lift under a lip: margins of a few mm, unreliable | **Box fixed in a nest** (spec allows "base fixed"); back to the top-down tab pinch with the world-anchored arc |
| Isaac's Cartesian DLS veers onto other IK branches near wrist singularities | Per-step IK on the calibrated UR model, seeded from the last command, rate-limited; elbow-up/wrist-down family preferred; links kept >= 6 cm above the table |
| Heel clamp: hanging toe's torque acts about the closing axis, shoe swings out | Collar pinch is the default again (`--grasp collar`), now with the 30 Nm drive |

**State at the end of session 1 (honest):**
- Ran and passed: install, the pinned stack, Isaac boot, scene spawn (two UR10e + 2F-85, fixed shoebox articulation, scanned shoe pair, 4 RTX RGB-D cameras), recording in the repo format (frames, labels, side files), the offline test suites.
- Ran and works in isolation: lid opening on a fixed box (opens to the 105 deg stop, box does not move); right arm pick, carry, align and release of a scanned shoe (seen in several runs); arm kinematics (DH model matches sim to 0.00 mm).
- **Ran and failed: no complete successful T1 episode yet. The Phase 0 gate (18/20) is not met.** The remaining failures are arm-arm and arm-lid collisions plus occasional wrist wind-up while both arms share the box workspace: the scripted expert has no collision-aware motion planning.
- Next session, in order: (1) add collision-aware transits (sphere model of both arms + box/lid as obstacles, checked along MoveJ paths, with re-planning via a high via-point); (2) re-run the single episode until it succeeds; (3) the 20-episode gate with `tools/check_dataset.py`; (4) install build tools (`sudo apt install build-essential`) so Isaac Lab Mimic and cuRobo (a proper GPU motion planner) can be used.
