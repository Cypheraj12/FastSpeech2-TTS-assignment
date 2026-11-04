"""English / Svarah preprocessing.

Stage 1 `select`   : pick ~5 h train + ~1 h validation from Svarah, write 22.05 kHz wavs + manifests.
Stage 2 `features` : mel, energy, F0 and forced-alignment durations -> preprocessed_dir/*.npz.

    python preprocess.py --stage select
    python preprocess.py --stage features
"""
import argparse
import csv
import io
import json
import os
import re

import numpy as np
import soundfile as sf
from tqdm import tqdm

import common
import text as frontend


# ------------------------------------------------------------------------------------ audio io
def to_mono_22k(wav, sr, target):
    import librosa
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != target:
        wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=target)
    peak = np.abs(wav).max()
    return (wav / peak * 0.95 if peak > 0 else wav).astype(np.float32)


# ------------------------------------------------------------------------------------ stage 1
def clean_transcript(t):
    t = re.sub(r"[\[<(][^\]>)]*[\]>)]", " ", t or "")            # drop [noise] <laugh> (tags)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def speaker_key(row, cols):
    """Speaker id column if the dataset has one; else the tuple of speaker-level metadata."""
    for c in ("speaker_id", "speaker", "spk_id", "spk", "speaker_name"):
        if c in row and row[c] not in (None, ""):
            return str(row[c])
    meta = [str(row[c]) for c in cols if c not in ("audio_filepath", "audio", "duration", "text")
            and not isinstance(row[c], (dict, list))]
    return "|".join(meta) if meta else "spk0"


def select(cfg):
    from datasets import Audio, load_dataset
    d, a = cfg["data"], cfg["audio"]
    out = cfg["paths"]["raw_dir"]
    os.makedirs(out, exist_ok=True)
    ds = load_dataset(d["hf_dataset"], split=d["hf_split"], token=os.environ.get("HF_TOKEN") or True)
    ds = ds.cast_column("audio_filepath", Audio(decode=False))
    cols = ds.column_names
    print("columns:", cols, "| rows:", len(ds))

    rng = np.random.RandomState(d["seed"])
    order = rng.permutation(len(ds))
    # ---- candidate filtering (seeded random order, first-come) -------------------------------
    cands, seen_text, dropped = [], set(), {"duration": 0, "empty_text": 0, "duplicate_text": 0}
    for idx in tqdm(order, desc="filtering"):
        row = ds[int(idx)]
        dur = float(row["duration"])
        if not (d["min_duration"] <= dur <= d["max_duration"]):
            dropped["duration"] += 1
            continue
        raw = clean_transcript(row["text"])
        norm = frontend.normalize(raw)
        if len(norm) < 4:
            dropped["empty_text"] += 1
            continue
        if norm in seen_text:                       # guarantees no sentence appears in both splits
            dropped["duplicate_text"] += 1
            continue
        seen_text.add(norm)
        cands.append(dict(index=int(idx), duration=dur, text_raw=raw, text_norm=norm,
                          speaker=speaker_key(row, cols)))
    total_h = sum(c["duration"] for c in cands) / 3600
    train_h, val_h = d["train_hours"], d["val_hours"]
    if total_h < train_h + val_h:                   # not enough audio -> keep the 5:1 ratio
        scale = total_h / (train_h + val_h)
        print(f"WARNING only {total_h:.2f} h available after filtering; scaling split by {scale:.2f}")
        train_h, val_h = train_h * scale, val_h * scale
    splits = {"train": [], "val": []}
    acc = {"train": 0.0, "val": 0.0}
    cur = "train"
    for c in cands:                                 # candidates are already in random order
        if cur == "train" and acc["train"] >= train_h * 3600:
            cur = "val"
        if cur == "val" and acc["val"] >= val_h * 3600:
            break
        splits[cur].append(c)
        acc[cur] += c["duration"]
    assert not ({c["index"] for c in splits["train"]} & {c["index"] for c in splits["val"]})
    assert not ({c["text_norm"] for c in splits["train"]} & {c["text_norm"] for c in splits["val"]})

    # ---- write audio + manifests ----------------------------------------------------------------
    speakers = sorted({c["speaker"] for s in splits.values() for c in s})
    spk2id = {s: i for i, s in enumerate(speakers)}
    stats = {"seed": d["seed"], "source": d["hf_dataset"], "rows_in_source": len(ds),
             "filters": {"min_duration": d["min_duration"], "max_duration": d["max_duration"]},
             "dropped": dropped, "n_speakers": len(speakers)}
    for name, items in splits.items():
        os.makedirs(os.path.join(out, name), exist_ok=True)
        with open(os.path.join(out, f"{name}_manifest.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["id", "speaker", "duration_sec", "text_raw", "text_norm", "source_index"])
            for n, c in enumerate(tqdm(items, desc=f"writing {name}")):
                row = ds[c["index"]]
                wav, sr = sf.read(io.BytesIO(row["audio_filepath"]["bytes"]), dtype="float32")
                wav = to_mono_22k(wav, sr, a["sampling_rate"])
                uid = f"{name}_{n:05d}"
                sf.write(os.path.join(out, name, uid + ".wav"), wav, a["sampling_rate"], subtype="PCM_16")
                c["id"] = uid
                w.writerow([uid, c["speaker"], f"{len(wav) / a['sampling_rate']:.3f}",
                            c["text_raw"], c["text_norm"], c["index"]])
        durs = np.array([c["duration"] for c in items])
        stats[name] = {"utterances": len(items), "hours": round(float(durs.sum() / 3600), 3),
                       "mean_dur_sec": round(float(durs.mean()), 2),
                       "speakers": len({c["speaker"] for c in items})}
    with open(os.path.join(out, "speakers.json"), "w") as f:
        json.dump(spk2id, f, indent=1)
    with open(os.path.join(out, "selection_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


# ------------------------------------------------------------------------------------ stage 2
def split_durations(groups, words, starts, n_frames, a, cap_s):
    """Word-level alignment -> phone-level durations (frames). Returns (tokens, durations)."""
    sr, hop = a["sampling_rate"], a["hop_length"]
    fr = lambda t: min(int(round(t * sr / hop)), n_frames)
    letters = [re.sub(r"[^a-z']", "", w) for w in words]
    # start frame of every alphabetic word and its (capped) end
    idx, wstart, wend = 0, {}, {}
    alpha = [i for i, l in enumerate(letters) if l]
    first = {i: idx_ for i, idx_ in zip(alpha, np.cumsum([0] + [len(letters[i]) for i in alpha[:-1]]))}
    for k, i in enumerate(alpha):
        s0 = fr(starts[first[i]])
        last_letter = first[i] + len(letters[i]) - 1
        nxt = fr(starts[first[alpha[k + 1]]]) if k + 1 < len(alpha) else n_frames
        wstart[i] = s0
        wend[i] = min(nxt, fr(starts[last_letter]) + int(round(cap_s * sr / hop)))
    tokens, dur = [], []
    weights = lambda p: 1.8 if p[0] in "AEIOU" else (0.6 if p in ("P", "T", "K", "B", "D", "G") else 1.0)
    for gi, g in enumerate(groups):
        if gi:
            tokens.append(" ")
            dur.append(0)
        if gi in wstart:
            w = np.array([weights(p) for p in g], dtype=float)
            span = max(wend[gi] - wstart[gi], 0)
            cum = np.round(np.cumsum(w) / w.sum() * span).astype(int)
            parts = np.diff(np.concatenate([[0], cum]))
        else:
            parts = np.zeros(len(g), dtype=int)
        tokens.extend(g)
        dur.extend(int(x) for x in parts)
    # pauses: the gap between a word's capped end and the next word's start goes to the
    # separator(s) between them
    pos_of_word, t_idx = {}, 0
    for gi, g in enumerate(groups):
        if gi:
            t_idx += 1
        pos_of_word[gi] = (t_idx, t_idx + len(g))
        t_idx += len(g)
    for k, i in enumerate(alpha):
        last = k + 1 == len(alpha)
        nxt_start = n_frames if last else wstart[alpha[k + 1]]
        gap = max(nxt_start - wend[i], 0)
        sep_range = range(pos_of_word[i][1], len(tokens) if last else pos_of_word[alpha[k + 1]][0])
        if gap:
            dur[sep_range[0] if len(sep_range) else pos_of_word[i][1] - 1] += gap
    dur[0] += wstart[alpha[0]]                                        # leading silence
    dur[int(np.argmax(dur))] += n_frames - sum(dur)                   # rounding repair
    assert min(dur) >= 0 and sum(dur) == n_frames, "bad durations"
    return tokens, dur


def features(cfg, device):
    import torch
    import torchaudio
    a, p = cfg["audio"], cfg["paths"]
    raw, pre = p["raw_dir"], p["preprocessed_dir"]
    os.makedirs(pre, exist_ok=True)
    aligner = common.Aligner(device)
    spk2id = json.load(open(os.path.join(raw, "speakers.json")))
    resample16 = torchaudio.transforms.Resample(a["sampling_rate"], 16000)
    skipped, pitch_all, energy_all, spk_train = [], [], [], []
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
                norm, words, groups = frontend.text_to_groups(r["text_raw"])
                assert len(words) == len(groups), "word/g2p mismatch"
                letters = "".join(re.sub(r"[^a-z']", "", w) for w in words)
                starts = aligner.letter_starts(resample16(wt).unsqueeze(0), letters)
                tokens, dur = split_durations(groups, words, starts, n, a, cfg["align"]["word_final_cap"])
                keep = [i for i, t in enumerate(tokens) if t in frontend.SYM2ID]
                tokens, dur = [tokens[i] for i in keep], [dur[i] for i in keep]
                if sum(dur) != n:                                       # a dropped token carried time
                    dur[int(np.argmax(dur))] += n - sum(dur)
                f0 = common.interpolate_f0(common.extract_f0(wav, a, n))
                if f0 is None:
                    raise RuntimeError("no voiced frames")
                pitch = common.token_average(f0, dur)
                en = common.token_average(energy, dur)
                np.savez_compressed(os.path.join(pre, split, r["id"] + ".npz"),
                                    tokens=np.array(frontend.tokens_to_ids(tokens), np.int16),
                                    duration=np.array(dur, np.int16), mel=mel.astype(np.float16),
                                    pitch=pitch, energy=en, speaker=spk2id[r["speaker"]])
                kept.append(r["id"])
                if split == "train":
                    spk_train.append(spk2id[r["speaker"]])
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
             "n_speakers": len(spk2id), "n_symbols": frontend.N_SYMBOLS,
             "default_speaker": int(np.bincount(spk_train).argmax())}
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
