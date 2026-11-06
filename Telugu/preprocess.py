"""Telugu audio preprocessing and feature extraction."""
import argparse
import csv
import io
import json
import os

import numpy as np
import soundfile as sf
from tqdm import tqdm

import common
import text as frontend


def to_mono_22k(wav, sr, target):
    import librosa
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != target:
        wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=target)
    peak = np.abs(wav).max()
    return (wav / peak * 0.95 if peak > 0 else wav).astype(np.float32)


def select(cfg):
    from datasets import Audio, load_dataset
    d, a = cfg["data"], cfg["audio"]
    out = cfg["paths"]["raw_dir"]
    os.makedirs(out, exist_ok=True)
    ds = load_dataset(d["hf_dataset"], split=d["hf_split"], streaming=True, token=os.environ.get("HF_TOKEN"))
    cols = list(ds.features.keys()) if ds.features else []
    audio_col = next((c for c in cols if "audio" in c.lower()), "audio")
    text_col = next((c for c in ("text", "sentence", "transcript", "transcription") if c in cols), "text")
    voice_col = next((c for c in ("gender", "speaker", "speaker_id", "speaker_name") if c in cols), None)
    print("columns:", cols, "| audio:", audio_col, "| text:", text_col, "| voice column:", voice_col)
    ds = ds.cast_column(audio_col, Audio(decode=False)).shuffle(seed=d["seed"], buffer_size=d["shuffle_buffer"])

    need = {"train": d["train_hours"] * 3600, "val": d["val_hours"] * 3600}
    acc, items, seen_text = {"train": 0.0, "val": 0.0}, {"train": [], "val": []}, set()
    dropped = {"other_voice": 0, "duration": 0, "empty_text": 0, "duplicate_text": 0}
    voice, cur = None, "train"
    for os_ in ("train", "val"):
        os.makedirs(os.path.join(out, os_), exist_ok=True)
    pbar = tqdm(total=sum(need.values()) / 3600, desc="hours selected", unit="h")
    for row in ds:
        v = str(row[voice_col]) if voice_col else "all"
        if voice is None:
            voice = v                                  # the first sampled row fixes the voice used
        if v != voice:
            dropped["other_voice"] += 1
            continue
        norm = frontend.normalize(row[text_col] or "")
        if len(norm) < 3:
            dropped["empty_text"] += 1
            continue
        if norm in seen_text:                          # no sentence can appear in both splits
            dropped["duplicate_text"] += 1
            continue
        wav, sr = sf.read(io.BytesIO(row[audio_col]["bytes"]), dtype="float32")
        dur = (len(wav) if wav.ndim == 1 else wav.shape[0]) / sr
        if not (d["min_duration"] <= dur <= d["max_duration"]):
            dropped["duration"] += 1
            continue
        seen_text.add(norm)
        wav = to_mono_22k(wav, sr, a["sampling_rate"])
        uid = f"{cur}_{len(items[cur]):05d}"
        sf.write(os.path.join(out, cur, uid + ".wav"), wav, a["sampling_rate"], subtype="PCM_16")
        items[cur].append(dict(id=uid, speaker=voice, duration=len(wav) / a["sampling_rate"],
                               text_raw=row[text_col], text_norm=norm))
        acc[cur] += items[cur][-1]["duration"]
        pbar.update(items[cur][-1]["duration"] / 3600)
        if acc[cur] >= need[cur]:
            if cur == "train":
                cur = "val"
            else:
                break
    pbar.close()
    stats = {"seed": d["seed"], "source": d["hf_dataset"], "voice_used": voice, "voice_column": voice_col,
             "filters": {"min_duration": d["min_duration"], "max_duration": d["max_duration"]},
             "dropped_while_streaming": dropped, "n_speakers": 1}
    for name, its in items.items():
        with open(os.path.join(out, f"{name}_manifest.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["id", "speaker", "duration_sec", "text_raw", "text_norm"])
            for c in its:
                w.writerow([c["id"], c["speaker"], f"{c['duration']:.3f}", c["text_raw"], c["text_norm"]])
        durs = np.array([c["duration"] for c in its])
        stats[name] = {"utterances": len(its), "hours": round(float(durs.sum() / 3600), 3),
                       "mean_dur_sec": round(float(durs.mean()), 2)}
    assert not ({c["text_norm"] for c in items["train"]} & {c["text_norm"] for c in items["val"]})
    with open(os.path.join(out, "speakers.json"), "w") as f:
        json.dump({voice: 0}, f)
    with open(os.path.join(out, "selection_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


def features(cfg, device):
    import torch
    import torchaudio
    a, p = cfg["audio"], cfg["paths"]
    raw, pre = p["raw_dir"], p["preprocessed_dir"]
    os.makedirs(pre, exist_ok=True)
    aligner = common.Aligner(device)
    spk2id = json.load(open(os.path.join(raw, "speakers.json")))
    resample16 = torchaudio.transforms.Resample(a["sampling_rate"], 16000)
    skipped, pitch_all, energy_all = [], [], []
    for split in ("train", "val"):
        rows = list(csv.DictReader(open(os.path.join(raw, f"{split}_manifest.csv"), encoding="utf-8")))
        os.makedirs(os.path.join(pre, split), exist_ok=True)
        kept = []
        for r in tqdm(rows, desc=f"features/{split}"):
            try:
                wav, sr = sf.read(os.path.join(raw, split, r["id"] + ".wav"), dtype="float32")
                wt = torch.from_numpy(wav)
                mel, energy = common.mel_and_energy(wt, a, device)
                n = mel.shape[0]
                tokens = frontend.text_to_tokens(r["text_raw"])
                roman = frontend.romanize_tokens(tokens)
                letters = "".join(roman)
                if not letters:
                    raise RuntimeError("no alignable letters")
                starts = aligner.letter_starts(resample16(wt).unsqueeze(0), letters)
                dur = common.durations_from_starts(roman, [frontend.is_separator(t) for t in tokens],
                                                   starts, n, a, cfg["align"]["word_final_cap"])
                f0 = common.interpolate_f0(common.extract_f0(wav, a, n))
                if f0 is None:
                    raise RuntimeError("no voiced frames")
                pitch, en = common.token_average(f0, dur), common.token_average(energy, dur)
                np.savez_compressed(os.path.join(pre, split, r["id"] + ".npz"),
                                    tokens=np.array(frontend.tokens_to_ids(tokens), np.int16),
                                    duration=np.array(dur, np.int16), mel=mel.astype(np.float16),
                                    pitch=pitch, energy=en, speaker=spk2id[r["speaker"]])
                kept.append(r["id"])
                if split == "train":
                    pitch_all.append(pitch[np.array(dur) > 0])
                    energy_all.append(en[np.array(dur) > 0])
            except Exception as e:                                     # noqa: BLE001
                skipped.append((r["id"], str(e)[:120]))
        with open(os.path.join(pre, f"{split}.txt"), "w") as f:
            f.write("\n".join(kept))
    P, E = np.concatenate(pitch_all), np.concatenate(energy_all)
    stats = {"pitch_mean": float(P.mean()), "pitch_std": float(P.std()),
             "energy_mean": float(E.mean()), "energy_std": float(E.std()),
             "pitch_min": float(((P - P.mean()) / P.std()).min()), "pitch_max": float(((P - P.mean()) / P.std()).max()),
             "energy_min": float(((E - E.mean()) / E.std()).min()), "energy_max": float(((E - E.mean()) / E.std()).max()),
             "n_speakers": len(spk2id), "n_symbols": frontend.N_SYMBOLS, "default_speaker": 0}
    with open(os.path.join(pre, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    with open(os.path.join(pre, "skipped.txt"), "w") as f:
        f.write("\n".join(f"{i}\t{e}" for i, e in skipped))
    print(f"done. skipped {len(skipped)} utterances (see skipped.txt). stats: {stats}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--stage", choices=["select", "features", "all"], default="all")
    args = ap.parse_args()
    cfg = common.load_config(args.config)
    if args.stage in ("select", "all"):
        select(cfg)
    if args.stage in ("features", "all"):
        import torch
        features(cfg, "cuda" if torch.cuda.is_available() else "cpu")
