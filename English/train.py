"""

    python train.py --config config.yaml [--max_minutes 330]
"""
import argparse
import csv
import glob
import json
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import common
from model import FastSpeech2, fastspeech2_loss

KEYS = ["total", "mel", "postnet", "duration", "pitch", "energy"]


class TTSData(Dataset):
    def __init__(self, pre_dir, split, stats):
        self.dir, self.stats = os.path.join(pre_dir, split), stats
        self.ids = [l for l in open(os.path.join(pre_dir, f"{split}.txt")).read().split("\n") if l]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        z = np.load(os.path.join(self.dir, self.ids[i] + ".npz"))
        s = self.stats
        dur = z["duration"].astype(np.int64)
        pitch = (z["pitch"] - s["pitch_mean"]) / s["pitch_std"]
        energy = (z["energy"] - s["energy_mean"]) / s["energy_std"]
        pitch[dur == 0], energy[dur == 0] = 0.0, 0.0        # tokens without frames carry no target
        return dict(tokens=torch.from_numpy(z["tokens"].astype(np.int64)),
                    duration=torch.from_numpy(dur),
                    mel=torch.from_numpy(z["mel"].astype(np.float32)),
                    pitch=torch.from_numpy(pitch).float(), energy=torch.from_numpy(energy).float(),
                    speaker=int(z["speaker"]))


def collate(items):
    pad = torch.nn.utils.rnn.pad_sequence
    return dict(
        src=pad([x["tokens"] for x in items], batch_first=True),
        src_len=torch.tensor([len(x["tokens"]) for x in items]),
        mel=pad([x["mel"] for x in items], batch_first=True),
        d=pad([x["duration"] for x in items], batch_first=True),
        p=pad([x["pitch"] for x in items], batch_first=True),
        e=pad([x["energy"] for x in items], batch_first=True),
        speaker=torch.tensor([x["speaker"] for x in items]))


def lr_at(step, d_model, warmup, scale):
    return scale * d_model ** -0.5 * min(step ** -0.5, step * warmup ** -1.5)


def run_batch(model, b, dev):
    b = {k: v.to(dev) for k, v in b.items()}
    out = model(b["src"], b["src_len"], speaker=b["speaker"], d_target=b["d"], p_target=b["p"], e_target=b["e"])
    return fastspeech2_loss(out, b["mel"], b["d"], b["p"], b["e"])


@torch.no_grad()
def validate(model, loader, dev):
    model.eval()
    sums, n = dict.fromkeys(KEYS, 0.0), 0
    for b in loader:
        l = run_batch(model, b, dev)
        k = b["src"].size(0)
        for key in KEYS:
            sums[key] += l[key].item() * k
        n += k
    model.train()
    return {k: v / n for k, v in sums.items()}


def latest_ckpt(d):
    cks = sorted(glob.glob(os.path.join(d, "step_*.pt")), key=lambda p: int(p.split("_")[-1][:-3]))
    return cks[-1] if cks else None


def append_csv(path, step, vals):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["step"] + KEYS)
        w.writerow([step] + [f"{vals[k]:.5f}" for k in KEYS])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--max_minutes", type=float, default=None, help="stop (after saving) once this wall-time is used")
    args = ap.parse_args()
    cfg = common.load_config(args.config)
    t, p = cfg["train"], cfg["paths"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(t["seed"]); random.seed(t["seed"]); np.random.seed(t["seed"])
    os.makedirs(p["checkpoint_dir"], exist_ok=True)
    stats = json.load(open(os.path.join(p["preprocessed_dir"], "stats.json")))
    train_ds, val_ds = TTSData(p["preprocessed_dir"], "train", stats), TTSData(p["preprocessed_dir"], "val", stats)
    tl = DataLoader(train_ds, t["batch_size"], shuffle=True, collate_fn=collate, num_workers=t["num_workers"],
                    drop_last=True, persistent_workers=t["num_workers"] > 0)
    vl = DataLoader(val_ds, t["batch_size"], shuffle=False, collate_fn=collate, num_workers=t["num_workers"])
    model = FastSpeech2(cfg, stats["n_symbols"], stats, stats["n_speakers"]).to(dev)
    print(f"{sum(x.numel() for x in model.parameters()) / 1e6:.1f}M params | train {len(train_ds)} val {len(val_ds)} utts | {dev}")
    opt = torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9)
    step, best = 0, float("inf")
    ck = latest_ckpt(p["checkpoint_dir"])
    if ck:
        c = torch.load(ck, map_location=dev, weights_only=False)
        model.load_state_dict(c["model"]); opt.load_state_dict(c["opt"]); step, best = c["step"], c.get("best", best)
        print(f"resumed from {ck} (step {step})")
    t0, run, nrun = time.time(), dict.fromkeys(KEYS, 0.0), 0
    model.train()

    def save(tag=None):
        path = os.path.join(p["checkpoint_dir"], f"step_{step}.pt")
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), step=step, best=best, cfg=cfg, stats=stats), path)
        old = sorted(glob.glob(os.path.join(p["checkpoint_dir"], "step_*.pt")), key=lambda x: int(x.split("_")[-1][:-3]))
        for o in old[:-t["keep_last"]]:
            os.remove(o)

    while step < t["total_steps"]:
        for b in tl:
            step += 1
            for g in opt.param_groups:
                g["lr"] = lr_at(step, cfg["model"]["d_model"], t["warmup_steps"], t["lr_scale"])
            loss = run_batch(model, b, dev)
            opt.zero_grad(set_to_none=True)
            loss["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
            opt.step()
            for k in KEYS:
                run[k] += loss[k].item()
            nrun += 1
            if step % t["log_every"] == 0:
                avg = {k: v / nrun for k, v in run.items()}
                append_csv(os.path.join(p["checkpoint_dir"], "train_log.csv"), step, avg)
                print(f"step {step:6d} | " + " ".join(f"{k} {avg[k]:.3f}" for k in KEYS) +
                      f" | {(time.time() - t0) / 60:.1f} min", flush=True)
                run, nrun = dict.fromkeys(KEYS, 0.0), 0
            if step % t["val_every"] == 0:
                v = validate(model, vl, dev)
                append_csv(os.path.join(p["checkpoint_dir"], "val_log.csv"), step, v)
                print(f"  VAL step {step}: " + " ".join(f"{k} {v[k]:.3f}" for k in KEYS), flush=True)
                if v["total"] < best:
                    best = v["total"]
                    torch.save(dict(model=model.state_dict(), step=step, cfg=cfg, stats=stats, val=v),
                               os.path.join(p["checkpoint_dir"], "best.pt"))
            if step % t["save_every"] == 0:
                save()
            out_of_time = args.max_minutes and (time.time() - t0) / 60 > args.max_minutes
            if step >= t["total_steps"] or out_of_time:
                save()
                print("stopping at step", step, "(time limit)" if out_of_time else "(done)")
                common.plot_losses(p["results_dir"], p["checkpoint_dir"], cfg["language"])
                return
    save()
    common.plot_losses(p["results_dir"], p["checkpoint_dir"], cfg["language"])


if __name__ == "__main__":
    main()
