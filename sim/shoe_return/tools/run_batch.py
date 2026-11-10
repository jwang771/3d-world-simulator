"""Run many single-episode simulations in parallel (one Isaac process each, so every episode gets its own shoe pair
and asset cache) and merge them into one dataset root.

    cd ../IsaacLab && uv run --no-sync python ../3d-world-simulator/sim/shoe_return/tools/run_batch.py \
        --out ../3d-world-simulator/data/shoe_return_sim --episodes 20 --workers 5 --seed 1000 -- --width 320 --height 240

Episode e uses seed (--seed + e). Arguments after `--` go to run_sim.py unchanged. Writes <out>/batch_stats.json.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
RUN_SIM = HERE / "run_sim.py"


def run_one(e, args, extra):
    work = Path(args.out) / "_work" / f"ep{e:05d}"
    work.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(RUN_SIM), "--out", str(work), "--episodes", "1", "--start-episode", str(e),
           "--seed", str(args.seed + e), "--val-every", str(args.val_every), *extra]
    if args.keep_failures:
        cmd.append("--keep-failures")
    # Each Isaac process otherwise spreads over every core; several at once then thrash (load 98 on 32 threads).
    threads = str(args.threads)
    env = dict(os.environ, OMNI_KIT_ACCEPT_EULA="YES", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    if args.threads:
        cmd += ["--kit_args", f"--/plugins/carb.tasking.plugin/threadCount={threads}"]
    t0 = time.time()
    with open(work / "log.txt", "w") as log:
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, timeout=args.timeout).returncode
    stats = list(work.glob("run_stats_*.json"))
    rec = json.loads(stats[0].read_text())[0] if stats else dict(episode=e, success=False, failure="crash")
    shoe = next((ln.split()[2] for ln in (work / "log.txt").read_text().splitlines() if ln.startswith("[assets] shoe")), "?")
    rec.update(seed=args.seed + e, shoe=shoe, returncode=rc, wall_s=round(time.time() - t0))
    print(f"[batch] episode {e} shoe {shoe}: {'ok' if rec['success'] else 'FAILED ' + str(rec.get('failure'))} ({rec['wall_s']}s)", flush=True)
    return rec


def merge(work, out):
    """Move episode files from a worker dir into the dataset root (episode numbers are unique per worker)."""
    for f in [*work.rglob("*.hdf5"), *work.rglob("*.h5")]:
        rel = f.relative_to(work)
        if rel.parts[0] == "_assets":
            continue
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(f), dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--start-episode", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--val-every", type=int, default=10)
    ap.add_argument("--keep-failures", action="store_true")
    ap.add_argument("--keep-work", action="store_true", help="keep per-episode work dirs (logs, asset caches)")
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--threads", type=int, default=0, help="cap Kit task threads per Isaac process (0: no cap). Measured on a\n                    32-thread machine: one uncapped process ~53 physics steps/s; 2 capped at 8 ~11 steps/s each; 5 uncapped thrash")
    args, extra = ap.parse_known_args()
    extra = [a for a in extra if a != "--"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    eps = range(args.start_episode, args.start_episode + args.episodes)
    with ThreadPoolExecutor(args.workers) as pool:
        recs = list(pool.map(lambda e: run_one(e, args, extra), eps))
    for e in eps:
        merge(out / "_work" / f"ep{e:05d}", out)
    if not args.keep_work:
        for e in eps:
            w = out / "_work" / f"ep{e:05d}"
            (out / "logs").mkdir(exist_ok=True)
            if (w / "log.txt").exists():
                shutil.copy(w / "log.txt", out / "logs" / f"episode_{e}.txt")
            shutil.rmtree(w, ignore_errors=True)
    stats_path = out / f"batch_stats_ep{args.start_episode}.json"
    stats_path.write_text(json.dumps(recs, indent=1))
    n = sum(r["success"] for r in recs)
    print(f"[batch] success {n}/{len(recs)} -> {stats_path}", flush=True)


if __name__ == "__main__":
    main()
