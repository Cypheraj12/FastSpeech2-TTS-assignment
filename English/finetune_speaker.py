"""Single-Speaker Target Adaptation on Speaker 74 (FastSpeech 2).

Fine-tunes the base multi-speaker checkpoint specifically on Speaker 74 to collapse
inter-speaker variance and eliminate multi-speaker averaging blur.

Usage:
    python finetune_speaker.py [--config config.yaml] [--steps 2500] [--batch_size 8]
"""
import argparse
import glob
import os
import time

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset

import common
import text as frontend
from model import FastSpeech2, fastspeech2_loss


class SingleSpeakerData(Dataset):
    def __init__(self, files, stats, target_speaker=74):
        self.files = files
        self.stats = stats
        self.target_speaker = target_speaker

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        z = np.load(self.files[i])
        s = self.stats
        dur = z["duration"].astype(np.int64)
        pitch = (z["pitch"] - s["pitch_mean"]) / s["pitch_std"]
        energy = (z["energy"] - s["energy_mean"]) / s["energy_std"]
        pitch[dur == 0], energy[dur == 0] = 0.0, 0.0
        return dict(
            tokens=torch.from_numpy(z["tokens"].astype(np.int64)),
            duration=torch.from_numpy(dur),
            mel=torch.from_numpy(z["mel"].astype(np.float32)),
            pitch=torch.from_numpy(pitch).float(),
            energy=torch.from_numpy(energy).float(),
            speaker=self.target_speaker,
        )


def collate_fn(items):
    pad = torch.nn.utils.rnn.pad_sequence
    return dict(
        src=pad([x["tokens"] for x in items], batch_first=True),
        src_len=torch.tensor([len(x["tokens"]) for x in items]),
        mel=pad([x["mel"] for x in items], batch_first=True),
        d=pad([x["duration"] for x in items], batch_first=True),
        p=pad([x["pitch"] for x in items], batch_first=True),
        e=pad([x["energy"] for x in items], batch_first=True),
        speaker=torch.tensor([x["speaker"] for x in items]),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--base_ckpt", default="checkpoints/best.pt")
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--speaker", type=int, default=74)
    args = ap.parse_args()

    cfg = common.load_config(args.config)
    p = cfg["paths"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    npz_files = sorted(glob.glob(os.path.join(p["preprocessed_dir"], "train", "*.npz")))
    spk_files = [path for path in npz_files if int(np.load(path)["speaker"]) == args.speaker]
    print(f"Loaded {len(spk_files)} utterances for target speaker {args.speaker}")

    c = torch.load(args.base_ckpt, map_location=dev, weights_only=False)
    stats = c["stats"]
    model = FastSpeech2(c["cfg"], stats["n_symbols"], stats, stats["n_speakers"]).to(dev)
    model.load_state_dict(c["model"])

    train_ds = SingleSpeakerData(spk_files, stats, target_speaker=args.speaker)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=2e-5)

    model.train()
    step = 0
    t0 = time.time()
    print(f"Starting adaptation to speaker {args.speaker} for {args.steps} steps...")

    while step < args.steps:
        for b in loader:
            step += 1
            b = {k: v.to(dev) for k, v in b.items()}
            out = model(b["src"], b["src_len"], speaker=b["speaker"],
                        d_target=b["d"], p_target=b["p"], e_target=b["e"])
            loss = fastspeech2_loss(out, b["mel"], b["d"], b["p"], b["e"])

            optimizer.zero_grad()
            loss["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if step % 100 == 0:
                elapsed = (time.time() - t0) / 60
                lr_cur = optimizer.param_groups[0]["lr"]
                print(f"Step {step:4d}/{args.steps} | Total: {loss['total'].item():.3f} | "
                      f"Mel: {loss['mel'].item():.3f} | Dur: {loss['duration'].item():.3f} | "
                      f"LR: {lr_cur:.1e} | ({elapsed:.1f} min)")

            if step in (1500, 2000):
                inter_path = os.path.join(p["checkpoint_dir"], f"ft_spk{args.speaker}_step{step}.pt")
                torch.save(dict(model=model.state_dict(), step=step, cfg=c["cfg"], stats=stats), inter_path)

            if step >= args.steps:
                break

    out_path = os.path.join(p["checkpoint_dir"], f"ft_spk{args.speaker}_best.pt")
    torch.save(dict(model=model.state_dict(), step=step, cfg=c["cfg"], stats=stats), out_path)
    print(f"Adaptation complete. Saved checkpoint to {out_path}")


if __name__ == "__main__":
    main()
