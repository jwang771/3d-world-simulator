"""Train an ACT or Diffusion Policy baseline on successful shoe-return episodes.

    cd ../IsaacLab && uv run --no-sync python ../3d-world-simulator/sim/shoe_return/policy/train.py \
        --config ../3d-world-simulator/sim/shoe_return/policy/configs/act.json --data ../3d-world-simulator/data/shoe_return_sim \
        --out runs/policy/act

Writes <out>/model.pt (weights + config + action normalisation + git commit) and <out>/train_log.json.
"""
import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from data import ChunkDataset, cache_episode  # noqa: E402
from models import build  # noqa: E402


def git_commit():
    try:
        return subprocess.check_output(["git", "-C", str(HERE), "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", required=True, nargs="+", help="dataset root(s) with train/ (and val/) episode files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    random.seed(args.seed), np.random.seed(args.seed), torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    files = []
    for root in args.data:
        root = Path(root)
        for split in ("train", "val"):
            for ep in sorted((root / split).glob("episode_*.hdf5")):
                c = cache_episode(ep, root / "_policy_cache")
                if c is not None:
                    files.append(c)
    assert files, "no successful episodes found"
    ds = ChunkDataset(files, chunk=cfg["chunk"])
    print(f"[train] {cfg['kind']}: {len(files)} successful episodes, {len(ds)} samples", flush=True)
    dl = torch.utils.data.DataLoader(ds, batch_size=cfg["batch"], shuffle=True, num_workers=6, drop_last=True, persistent_workers=True)
    dev = "cuda"
    model = build(cfg["kind"], **cfg.get("model", {}), chunk=cfg["chunk"]).to(dev)
    backbone = [p for n, p in model.named_parameters() if "backbone" in n]
    rest = [p for n, p in model.named_parameters() if "backbone" not in n]
    opt = torch.optim.AdamW([dict(params=rest, lr=cfg["lr"]), dict(params=backbone, lr=cfg["lr_backbone"])], weight_decay=cfg.get("wd", 1e-4))
    total = cfg["steps"]
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / 500) * 0.5 * (1 + np.cos(np.pi * min(s, total) / total)))
    ema = torch.optim.swa_utils.AveragedModel(model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999)) if cfg.get("ema") else None
    log, step, t0 = [], 0, time.time()
    while step < total:
        for imgs, prop, act, pad in dl:
            imgs, prop, act, pad = imgs.to(dev, non_blocking=True), prop.to(dev), act.to(dev), pad.to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, parts = model.loss(imgs, prop, act, pad)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            sched.step()
            if ema is not None:
                ema.update_parameters(model)
            step += 1
            if step % 200 == 0 or step == total:
                rec = dict(step=step, loss=loss.item(), **parts, s=round(time.time() - t0))
                log.append(rec)
                print(f"[train] {rec}", flush=True)
            if step >= total:
                break
    final = ema.module if ema is not None else model
    torch.save(dict(state=final.state_dict(), cfg=cfg, stats=ds.stats, commit=git_commit(), episodes=[str(f) for f in files], seed=args.seed),
               out / "model.pt")
    (out / "train_log.json").write_text(json.dumps(log, indent=1))
    print(f"[train] saved {out / 'model.pt'}", flush=True)


if __name__ == "__main__":
    main()
