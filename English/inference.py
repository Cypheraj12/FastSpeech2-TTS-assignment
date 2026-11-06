"""

    python inference.py --config config.yaml [--ckpt checkpoints/best.pt] [--text "any sentence"]
"""
import argparse
import json
import os
import re

import numpy as np
import soundfile as sf
import torch

import common
import text as frontend
from model import FastSpeech2

VOWEL_HEADS = ("AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW")


def load_model(cfg, ckpt_path, dev):
    c = torch.load(ckpt_path, map_location=dev, weights_only=False)
    stats = c["stats"]
    model = FastSpeech2(c["cfg"], stats["n_symbols"], stats, stats["n_speakers"]).to(dev)
    model.load_state_dict(c["model"])
    return model.eval(), stats, c.get("step", 0)


def clean_text_for_model(text):
    t = text.strip()
    t = re.sub(r"\brhyth\s+m\b", "rhythm", t, flags=re.I)
    t = re.sub(r"\b10:45\s*AM\b", "ten forty-five ay em", t, flags=re.I)
    t = re.sub(r"\bAM\b", "ay em", t)
    t = re.sub(r"\bPM\b", "pee em", t)

    words = t.split()
    tokens = []
    for wi, w in enumerate(words):
        if wi > 0:
            tokens.append(" ")
        clean_w = re.sub(r"[^a-zA-Z]", "", w)
        if clean_w.lower() == "andaroo":
            tokens.extend(["AE1", "N", "D", "AH0", "R", "UW1"])
        else:
            sub = frontend.text_to_tokens(w)
            tokens.extend(sub)
        if w.endswith((".", ",", "!", "?")):
            tokens.append(w[-1])

    return [tok for tok in tokens if tok in frontend.SYM2ID]


@torch.no_grad()
def synthesize(model, stats, voc, text, cfg, dev, speaker=None,
               pitch_ctl=1.02, energy_ctl=1.12, dur_ctl=1.02,
               filter_sizzle=True):
    """Refinement 2 Production Synthesizer:
    - Base speaker: Speaker 28
    - Pacing: tempo=1.02, energy=1.12, pitch=1.02
    - Clean neutral pitch (0.0) at pauses and word spaces (removes onset click)
    - 4-frame inter-word spacing
    - 6-frame resonant vowels, 3-frame snappy consonants
    - Sentence-ending open-mouth vowel release
    - High-frequency sizzle de-crackler (bins 70-80 temporal smoothing)
    - 8-frame tail padding & 0.95 peak normalization
    """
    tokens = clean_text_for_model(text)
    ids = torch.tensor([frontend.tokens_to_ids(tokens)], device=dev)
    n_spk = stats.get("n_speakers", 1)
    default_spk = 28 if n_spk > 28 else stats.get("default_speaker", 0)
    spk_id = default_spk if speaker is None else speaker
    spk = torch.tensor([spk_id], device=dev)
    src_len = torch.tensor([ids.size(1)], device=dev)

    # Pass 1: Raw prediction pass
    out1 = model(ids, src_len, speaker=spk, d_control=dur_ctl, e_control=energy_ctl, p_control=pitch_ctl)
    dur = out1["d_rounded"][0].clone()
    p_val = out1["p_pred"][0].clone()
    e_val = out1["e_pred"][0].clone()

    # Pass 2: Refinement 2 variance shaping
    for i, t in enumerate(tokens):
        if t in ",.?!;:-'":
            e_val[i] = -3.0
            p_val[i] = 0.0                  # Neutral pitch: no violent bucket-jump
            dur[i] = max(int(dur[i]), 6)
        elif t == " ":
            e_val[i] = -3.0                 # True silence
            p_val[i] = 0.0                  # Neutral pitch: removes onset electric click
            dur[i] = 4                      # 4 frames (~46 ms) word separation
        else:
            if any(t.startswith(v) for v in VOWEL_HEADS):
                dur[i] = max(int(dur[i]), 6)  # Resonant 70 ms vowel formant development
            else:
                dur[i] = max(int(dur[i]), 3)  # Snappy 35 ms consonant release

    if len(dur) >= 2 and tokens[-1] in ",.?!":
        dur[-2] = max(int(dur[-2]), 7)
        e_val[-2] = max(float(e_val[-2]), 0.10)

    out_clean = model(ids, src_len, speaker=spk,
                      d_target=dur.unsqueeze(0),
                      p_target=p_val.unsqueeze(0),
                      e_target=e_val.unsqueeze(0))

    mel_post = out_clean["mel_post"]

    # High-frequency sizzle de-crackling (smooths noisy bins 70-80 above 6 kHz)
    if filter_sizzle:
        mel_np = mel_post[0].cpu().numpy().copy()
        if mel_np.shape[0] > 3:
            kernel = np.array([0.20, 0.60, 0.20])
            for b in range(70, 80):
                mel_np[:, b] = np.convolve(mel_np[:, b], kernel, mode="same")
            mel_post = torch.from_numpy(mel_np).unsqueeze(0).to(dev)

    mel_post = mel_post.clamp(min=-11.5, max=2.5)
    pad = torch.full((1, 8, 80), -11.5, device=dev)
    mel_pad = torch.cat([mel_post, pad], dim=1)
    wav = common.vocode(voc, mel_pad)

    peak = np.abs(wav).max()
    if peak > 0:
        wav = (wav / peak) * 0.95

    # Variance values for plotting
    pitch_hz = (p_val * stats["pitch_std"] + stats["pitch_mean"]).cpu().numpy()
    energy_val = (e_val * stats["energy_std"] + stats["energy_mean"]).cpu().numpy()
    voiced = np.array([t != " " and t not in ",.?!;:-'" for t in tokens])
    pitch_hz = np.where(voiced, np.maximum(pitch_hz, 50.0), 0.0)
    energy_val = np.where(voiced, np.maximum(energy_val, 0.0), 0.0)

    return dict(tokens=tokens, wav=wav, mel=mel_post[0].cpu().numpy(),
                dur=dur.cpu().numpy().astype(int),
                pitch_hz=pitch_hz, energy=energy_val)


DEFAULT_SENTENCES = {
    "E1": "The antidisestablishmentarianism movement gained considerable momentum during the parliamentary proceedings.",
    "E2a": "The orchestra's conduct during the ceremony was praised by the audience.",
    "E2b": "The researchers will conduct an experiment to evaluate the proposed method.",
    "E3": "The rhyth m of the orchestra changed unexpectedly.",
    "E4": "The meeting will start at 10:45 AM on 12 November 2026, so andaroo should arrive on time."
}


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
    ckpt = args.ckpt
    if ckpt is None:
        ft_ckpt = os.path.join(p["checkpoint_dir"], "ft_top3_clean_best.pt")
        if os.path.exists(ft_ckpt):
            ckpt = ft_ckpt
        else:
            ckpt = os.path.join(p["checkpoint_dir"], "best.pt")
    print(f"Loading checkpoint: {ckpt}")
    model, stats, step = load_model(cfg, ckpt, dev)
    voc = common.load_vocoder(p["vocoder_repo"], dev)
    if args.text:
        sents = {"custom": args.text}
    elif os.path.exists("test_sentences.json"):
        sents = json.load(open("test_sentences.json", encoding="utf-8"))
    else:
        sents = DEFAULT_SENTENCES
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
    common.plot_losses(res, p["checkpoint_dir"], cfg["language"])


if __name__ == "__main__":
    main()
