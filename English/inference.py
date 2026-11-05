"""Synthesise the five English test sentences and write audio + variance-adaptor plots.

    python inference.py --config config.yaml [--ckpt checkpoints/best.pt] [--text "any sentence"]
"""
import argparse
import json
import os

import numpy as np
import soundfile as sf
import torch

import common
import text as frontend
from model import FastSpeech2


def load_model(cfg, ckpt_path, dev):
    c = torch.load(ckpt_path, map_location=dev, weights_only=False)
    stats = c["stats"]
    model = FastSpeech2(c["cfg"], stats["n_symbols"], stats, stats["n_speakers"]).to(dev)
    model.load_state_dict(c["model"])
    return model.eval(), stats, c["step"]


@torch.no_grad()
def synthesize(model, stats, voc, text, cfg, dev, speaker=None, pitch_ctl=1.0, energy_ctl=1.0, dur_ctl=1.0):
    tokens = [t for t in frontend.text_to_tokens(text) if t in frontend.SYM2ID]
    ids = torch.tensor([frontend.tokens_to_ids(tokens)], device=dev)
    spk = torch.tensor([stats["default_speaker"] if speaker is None else speaker], device=dev)
    out = model(ids, torch.tensor([ids.size(1)], device=dev), speaker=spk,
                p_control=pitch_ctl, e_control=energy_ctl, d_control=dur_ctl)
    wav = common.vocode(voc, out["mel_post"])
    return dict(tokens=tokens, wav=wav, mel=out["mel_post"][0].cpu().numpy(),
                dur=out["d_rounded"][0].cpu().numpy().astype(int),
                pitch_hz=(out["p_pred"][0] * stats["pitch_std"] + stats["pitch_mean"]).cpu().numpy(),
                energy=(out["e_pred"][0] * stats["energy_std"] + stats["energy_mean"]).cpu().numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--text", default=None, help="synthesise this text instead of the test set")
    ap.add_argument("--speaker", type=int, default=None)
    args = ap.parse_args()
    cfg = common.load_config(args.config)
    p, a = cfg["paths"], cfg["audio"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = args.ckpt or os.path.join(p["checkpoint_dir"], "best.pt")
    model, stats, step = load_model(cfg, ckpt, dev)
    voc = common.load_vocoder(p["vocoder_repo"], dev)
    sents = {"custom": args.text} if args.text else json.load(open("test_sentences.json", encoding="utf-8"))
    res = p["results_dir"]
    dirs = {"audio": os.path.join(res, "test_generated_audios"),
            "duration": os.path.join(res, "variance_adaptor_plots", "duration"),
            "pitch": os.path.join(res, "variance_adaptor_plots", "pitch"),
            "energy": os.path.join(res, "variance_adaptor_plots", "energy")}
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    summary = {}
    for sid, text in sents.items():
        r = synthesize(model, stats, voc, text, cfg, dev, args.speaker)
        sf.write(os.path.join(dirs["audio"], f"{sid}.wav"), r["wav"], a["sampling_rate"], subtype="PCM_16")
        common.plot_variance(sid, r["tokens"], r["dur"], r["pitch_hz"], r["energy"], r["mel"], a, dirs)
        sec = a["hop_length"] / a["sampling_rate"]
        summary[sid] = dict(text=text, checkpoint_step=step, audio_sec=round(len(r["wav"]) / a["sampling_rate"], 2),
                            normalised=frontend.normalize(text), tokens=r["tokens"],
                            duration_ms=[round(float(d) * sec * 1000, 1) for d in r["dur"]],
                            pitch_hz=[round(float(x), 1) for x in r["pitch_hz"]],
                            energy=[round(float(x), 2) for x in r["energy"]])
        print(f"{sid}: {summary[sid]['audio_sec']} s -> {dirs['audio']}/{sid}.wav")
    with open(os.path.join(res, "test_predictions.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
