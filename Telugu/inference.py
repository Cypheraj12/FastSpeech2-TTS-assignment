"""Synthesise the five Telugu test sentences and write audio + variance-adaptor plots.

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
def synthesize(model, stats, voc, text, cfg, dev, speaker=None, pitch_ctl=1.02, energy_ctl=1.05, dur_ctl=1.18):
    from model import length_regulate, sinusoid_table
    tokens = [t for t in frontend.text_to_tokens(text) if t in frontend.SYM2ID]
    ids = torch.tensor([frontend.tokens_to_ids(tokens)], device=dev)
    spk = torch.tensor([stats["default_speaker"] if speaker is None else speaker], device=dev)
    src_len = torch.tensor([ids.size(1)], device=dev)
    
    # 1. Base model prediction with relaxed 1.18 tempo
    out = model(ids, src_len, speaker=spk,
                p_control=pitch_ctl, e_control=energy_ctl, d_control=dur_ctl)
    
    dur = out["d_rounded"][0].clone()
    p_val = out["p_pred"][0].clone()
    e_val = out["e_pred"][0].clone()
    
    # 2. Universal last-word identification: find start of the final word
    last_space_idx = 0
    for i, t in enumerate(tokens):
        if t == " ":
            last_space_idx = i
            
    # Breath pause before the final word so it doesn't crash into previous vowels
    if last_space_idx > 0:
        dur[last_space_idx] = max(int(dur[last_space_idx]), 5)
        
    # Ensure every letter of the final word gets full articulation time (min 7 frames = ~80ms)
    for i in range(last_space_idx + 1, len(tokens)):
        if tokens[i] not in ",.?!":
            dur[i] = max(int(dur[i]), 7)
            
    # Trailing punctuation pause
    if len(dur) > 0 and tokens[-1] in ",.?!":
        dur[-1] = 8
        dur[-2] = max(int(dur[-2]), 9)  # Final vowel
        
    # 3. Universal energy un-muting: ensure no letter drops below human vocal volume (35.0)
    norm_35 = (35.0 - stats["energy_mean"]) / stats["energy_std"]
    for i in range(last_space_idx + 1, len(tokens)):
        if tokens[i] not in ",.?!":
            e_val[i] = max(float(e_val[i]), norm_35)
    if len(e_val) >= 2:
        e_val[-2] = max(float(e_val[-2]), norm_35)
        
    # 4. Natural pitch descent: cadence gently resolves (~125 Hz) on sentence close
    norm_pitch = (125.0 - stats["pitch_mean"]) / stats["pitch_std"]
    if len(p_val) >= 2:
        p_val[-2] = min(float(p_val[-2]), norm_pitch)
        p_val[-1] = norm_pitch
        
    # Re-regulate with the unmuted, fully articulated final word
    src_mask = out["src_mask"]
    x_enc = model.embedding(ids) + sinusoid_table(ids.size(1), model.d_model, dev)
    for blk in model.encoder:
        x_enc = blk(x_enc, src_mask)
    if model.speaker_emb is not None:
        x_enc = x_enc + model.speaker_emb(spk).unsqueeze(1)
        
    p_binned = torch.bucketize(p_val.unsqueeze(0), model.variance_adaptor.pitch_bins)
    e_binned = torch.bucketize(e_val.unsqueeze(0), model.variance_adaptor.energy_bins)
    x_var = x_enc + model.variance_adaptor.pitch_embedding(p_binned) + model.variance_adaptor.energy_embedding(e_binned)
    
    x_reg, mel_len = length_regulate(x_var, dur.unsqueeze(0))
    mel_mask = torch.arange(x_reg.size(1), device=dev).unsqueeze(0) >= mel_len.unsqueeze(1)
    x_dec = x_reg + sinusoid_table(x_reg.size(1), model.d_model, dev)
    x_dec = x_dec.masked_fill(mel_mask.unsqueeze(-1), 0.0)
    for blk in model.decoder:
        x_dec = blk(x_dec, mel_mask)
        
    mel_out = model.mel_linear(x_dec)
    mel_post = mel_out + model.postnet(mel_out)
    
    # 5. Prevent HiFi-GAN truncation: pad mel with silence BEFORE vocoding
    mel_pad = torch.full((1, 12, 80), -11.5, device=dev)
    mel_post_padded = torch.cat([mel_post, mel_pad], dim=1)
    
    wav = common.vocode(voc, mel_post_padded)
    
    # Peak normalize to -1.0 dBFS
    peak = np.abs(wav).max()
    if peak > 0:
        wav = (wav / peak * 0.95).astype(np.float32)
        
    return dict(tokens=tokens, wav=wav, mel=mel_post[0].cpu().numpy(),
                dur=dur.cpu().numpy().astype(int),
                pitch_hz=(p_val * stats["pitch_std"] + stats["pitch_mean"]).cpu().numpy(),
                energy=(e_val * stats["energy_std"] + stats["energy_mean"]).cpu().numpy())


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
