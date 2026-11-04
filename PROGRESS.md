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
- Session 1 plan (superseded by session 2 below): (1) add collision-aware transits (sphere model of both arms + box/lid as obstacles, checked along MoveJ paths, with re-planning via a high via-point); (2) re-run the single episode until it succeeds; (3) the 20-episode gate with `tools/check_dataset.py`; (4) install build tools (`sudo apt install build-essential`) so Isaac Lab Mimic and cuRobo (a proper GPU motion planner) can be used.

### 2026-11-04, session 2

**Root cause of slipping grasps and arm "flailing" (ran, with evidence).** A grasp-only probe (`run_sim.py --skill pick`, low resolution, about 15 s per run) showed the arm's tracking error growing by about 4 deg every half second as soon as the gripper closed, even after the shoe had dropped. Findings:
- In the Isaac Sim 6.2 2F-85 (`Robotiq_2F_85_phyisics_mimic.usda`), all five linkage joints are PhysX mimic joints of `finger_joint`. Isaac Lab's config leaves `right_outer_knuckle_joint` undriven (stiffness 0), so the right finger is moved only through the mimic constraint. The two knuckles turn in mirrored directions, so the constraint's reaction on the gripper base does not cancel, and squeezing twisted the wrist. The constraint also stretched by 6-8 deg, and an empty gripper stalled at 32 deg (Isaac Lab's own config stalls at 24 deg) instead of reaching its 41 deg target (`tools/gripper_hold_probe.py --mode commanded`).
- Drives set on mimic joints have no effect (results identical to the decimal place).
- **Fix:** drive `right_outer_knuckle_joint` with the same PD as `finger_joint` (`envs/repack_scene.drive_both_knuckles`). The empty gripper now closes to its target with zero arm error. A scanned shoe (SUPERSTAR) is lifted 0.20 m by the collar pinch and held for 5 s (120 stiffness, 30 Nm). Before the fix the same shoe slipped in every setting tried (40/10, 120/30, 400/30).

**Lid knocked shut (ran, from recorded frames).** In the last session-1 episode the lid rested open on its stop, then an arm's joint-space transit swept through it. **Fix:** `collision.py`, a sphere model of both arms checked along every MoveJ against the table, the box, the lid (at the measured hinge angle) and the other arm. If the direct path is blocked, it detours via raised poses or random via-points. Offline test: `test_collision.py`, ran and passed. Isaac run: pending.

**Later in session 2 (all ran; results are from single debug episodes and 5-episode mini-gates, not yet the 20-episode gate):**

| Observed failure (frames + debug trace) | Change |
|---|---|
| 0.72 rad close target reached exactly with the knuckle fix, so a 6 mm lid tab (contact ~0.74 rad) was touched but not squeezed; the lid slid out at 53 deg | `GRIP_CLOSED` 0.80 rad (limit 0.82) |
| Shoe picked first and held ~7 s while the lid opened crept out of the pinch | right arm waits empty above shoe 1; picks once the lid is up |
| Placement steered by a heel error of 45 cm when the shoe was not in the hand: arm swept into the left arm | "still held?" check before placing; re-pick, or treat a shoe already in the box as placed; fine alignment capped at 8 cm |
| Released while hanging toe-down at rim height, the shoe stayed hooked on the finger inside the collar and came back out | shoes are lowered until they rest flat (heel 1 cm above the floor), released at pinch width, lifted slowly; if the heel still comes up, set back down, shift 1.5 cm toe-ward, lift again. Box length play 6 cm, shoe set 2 cm toe-ward |
| Hanging toe (up to ~0.3 m below the hand) struck the box corner in transit; the collision check missed it because a path in contact everywhere was trimmed to "free" | contact allowance limited to the first/last quarter of a path; held shoe modelled as spheres along its measured pose; lift and carry 0.35 m above the rim; carries at 0.6 rad/s; low inward detours preferred to high ones |
| Released lid near vertical knocked or crept shut | paperboard hinge damping 0.25 -> 2.0 N m s/rad; planner buffer 5 cm around the lid |
| Far-side hinge: opening pushed the lid away from the robots at the reach limit and stalled at 82-87 deg for some box poses; closing needed a tab regrasp at ~0.6 m reach; wrist wound to 269 deg | box turned so the hinge faces the left arm (`BOX_NOMINAL` yaw pi/2, 6 cm nearer); left arm now holds the lid open, compliant, during packing and swings it shut without letting go; press-to-close fallback; IK pulls toward home to unwind |
| 5 Isaac processes in parallel: load 98 on 32 threads, workers stalled; 2 thread-capped workers ~11 steps/s each | sequential runs (one process ~53 physics steps/s); `tools/run_batch.py` |

**Status at this point (honest):** 3 single-seed debug episodes succeeded end to end (SUPERSTAR, scripted expert). The first 5-episode mini-gate runs over random shoes and box poses had 0/5 successes with intermediate versions; each failure above was diagnosed from frames and fixed. The 20-episode Phase 0 gate has **not** been run on a stable version yet.

**Policy baselines (Phase 4 code, not yet trained on real data):** `policy/` has ACT and Diffusion Policy in plain PyTorch (ResNet-18 ImageNet backbone via torchvision; no new major library), 6-D rotation actions, receding-horizon closed-loop evaluation in `run_sim.py --policy`, held-out shoe split (`--shoe random:train|test`, 51/12). Smoke-trained 400 steps on 2 episodes and rolled out in Isaac: the pipeline runs; no performance claims.

**Session 2, continued: measured results (all ran).**

Further expert fixes, each from an observed failure: box turned so the hinge faces the left arm; creased-paperboard hinge with static friction (0.06 N m) so a released lid stays open; left arm opens, lets go, later regrasps and closes, with a press-to-close fallback; full-open release with the inside finger centred in the collar; a shoe not lying flat is re-placed once; tighter fine alignment (8 mm, 0.04 rad); box play 6 cm long x 6 cm across x 2.5 cm high; lid gap 6 mm and a 2 mm box contact offset (the lid skirt was landing on the wall top); 24000-step episode limit; gate runs from a frozen `git worktree` so mid-run edits cannot leak into a measurement.

| Run | Code | Result |
|---|---|---|
| Mini-gate 6 (seeds 1000-1004, all shoes) | before box/lid fit fixes | 2/5 |
| Gate 1 (seeds 1000-1019, all shoes) | `d70c650` (frozen) | **6/11 when stopped**; 4 of the 5 failures were the lid stopping on the wall top (fixed in `be473e9`), 1 a shoe dropped in transit. Stopped because it could no longer reach 18/20 |
| Data collection, training shoes | `d70c650`-`be473e9` | 7 successes of 13 episodes (failed episodes are not kept) |
| `tools/check_dataset.py` on mini-gate output | | 5/5 episodes load through `AlohaPackedPCDataset`/`Frames` (after allowing shoes hidden in a closed box at the end of a success) |
| Video | | `docs/media/t1_scripted_success.mp4` (front, overhead, right wrist; scripted success, seed 1000) |

**Phase 4, round 1 (ran; result: no success).** ACT and Diffusion Policy trained 20k steps each on 13 successful training-shoe episodes (8421 samples, seed 0; train L1 0.017 / denoising MSE 0.0037). Closed-loop on held-out shoes: ACT **0/3**, Diffusion Policy **0/2** (evaluations stopped early to free the machine for data collection). ACT barely moves from where it is (with the previous command as input it learned to repeat it, the "copycat" failure); Diffusion Policy did pinch the lid tab and lift the lid to 30 deg once, then lost it. Round 2 adds image augmentation and drops/noises the proprio input, and trains on more data.
