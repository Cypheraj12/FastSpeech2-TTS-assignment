"""Shared utilities: config, mel/pitch/energy extraction, forced alignment, vocoder, plots."""
import json
import os
import sys
import unicodedata

import numpy as np
import torch
import yaml

_MEL_BASIS, _WINDOW = {}, {}


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ----------------------------------------------------------------------------- audio features
def mel_and_energy(wav, a, device="cpu"):
    """HiFi-GAN-compatible log-mel (natural log, magnitude) + frame energy (L2 norm of |STFT|).

    wav: 1-D float tensor in [-1, 1]. Returns mel (T, n_mels), energy (T,).
    Number of frames = len(wav) // hop.
    """
    import librosa
    key = (a["sampling_rate"], a["n_fft"], a["n_mels"], a["fmin"], a["fmax"], str(device))
    if key not in _MEL_BASIS:
        basis = librosa.filters.mel(sr=a["sampling_rate"], n_fft=a["n_fft"], n_mels=a["n_mels"],
                                    fmin=a["fmin"], fmax=a["fmax"])
        _MEL_BASIS[key] = torch.from_numpy(basis).float().to(device)
        _WINDOW[key] = torch.hann_window(a["win_length"]).to(device)
    y = wav.to(device).float().unsqueeze(0).unsqueeze(1)
    pad = int((a["n_fft"] - a["hop_length"]) / 2)
    y = torch.nn.functional.pad(y, (pad, pad), mode="reflect").squeeze(1)
    spec = torch.stft(y, a["n_fft"], hop_length=a["hop_length"], win_length=a["win_length"],
                      window=_WINDOW[key], center=False, return_complex=True)
    mag = torch.sqrt(spec.real ** 2 + spec.imag ** 2 + 1e-9)          # (1, F, T)
    mel = torch.log(torch.clamp(torch.matmul(_MEL_BASIS[key], mag), min=1e-5))
    energy = torch.norm(mag, dim=1)
    return mel[0].T.cpu().numpy(), energy[0].cpu().numpy()


def extract_f0(wav_np, a, n_frames):
    """Frame-level F0 in Hz (0 = unvoiced), length n_frames."""
    sr, hop = a["sampling_rate"], a["hop_length"]
    try:
        import pyworld as pw
        x = wav_np.astype(np.float64)
        f0, t = pw.dio(x, sr, frame_period=hop / sr * 1000)
        f0 = pw.stonemask(x, f0, t, sr)
    except ImportError:                                                # slow fallback
        import librosa
        f0, _, _ = librosa.pyin(wav_np, fmin=50, fmax=800, sr=sr, frame_length=1024, hop_length=hop)
        f0 = np.nan_to_num(f0)
    f0 = f0[:n_frames]
    if len(f0) < n_frames:
        f0 = np.pad(f0, (0, n_frames - len(f0)))
    return f0.astype(np.float32)


def interpolate_f0(f0):
    voiced = f0 > 0
    if voiced.sum() == 0:
        return None
    idx = np.arange(len(f0))
    return np.interp(idx, idx[voiced], f0[voiced]).astype(np.float32)


def token_average(frame_vals, durations):
    out, pos = np.zeros(len(durations), np.float32), 0
    for i, d in enumerate(durations):
        d = int(d)
        if d > 0:
            out[i] = frame_vals[pos:pos + d].mean()
        pos += d
    return out


# ----------------------------------------------------------------------------- forced alignment
def ctc_letter_starts(logp, tokens, blank=0):
    """Viterbi CTC forced alignment (numpy; no dependency on torchaudio's forced_align).

    logp: (T, C) log-probabilities, tokens: list[int]. Returns the frame at which each token is
    emitted (CTC posteriors are 'peaky', so this is where the letter is heard). Simplified
    topology: state j = "token j has been emitted"; frames either stay (blank or repeated token)
    or advance to the next token.
    """
    T, N = logp.shape[0], len(tokens)
    if T < N:
        raise RuntimeError("audio too short for transcript")
    tok = np.asarray(tokens)
    NEG = -1e30
    score = np.full(N + 1, NEG)                       # state k: tokens[:k] already emitted
    score[0] = 0.0
    entered = np.zeros((T, N + 1), dtype=bool)        # entered[t, k]: state k entered at frame t
    for t in range(T):
        stay = score.copy()
        stay[0] += logp[t, blank]
        stay[1:] += np.maximum(logp[t, blank], logp[t, tok])      # blank or repeat of tok[k-1]
        move = np.full(N + 1, NEG)
        move[1:] = score[:-1] + logp[t, tok]
        entered[t] = move > stay
        score = np.maximum(stay, move)
    if score[-1] <= NEG / 2:
        raise RuntimeError("no valid alignment path")
    starts = np.zeros(N, dtype=int)
    k = N
    for t in range(T - 1, -1, -1):
        if k >= 1 and entered[t, k]:
            starts[k - 1] = t
            k -= 1
    return starts


class Aligner:
    """MMS_FA (multilingual CTC model from torchaudio) + our own Viterbi; romanised letters in,
    letter start times (seconds) out."""

    def __init__(self, device):
        import torchaudio
        bundle = torchaudio.pipelines.MMS_FA
        self.device, self.sr = device, bundle.sample_rate
        self.model = bundle.get_model(with_star=False).to(device).eval()
        self.dict = bundle.get_dict(star=None)

    @torch.inference_mode()
    def letter_starts(self, wav_16k, letters):
        emission, _ = self.model(wav_16k.to(self.device))
        logp = torch.log_softmax(emission[0].float(), dim=-1).cpu().numpy()
        frames = ctc_letter_starts(logp, [self.dict[c] for c in letters], blank=0)
        return frames * (wav_16k.size(1) / logp.shape[0] / self.sr)


def durations_from_starts(roman, is_sep, starts, n_frames, a, cap_s=0.2):
    """Per-token durations (mel frames, summing to n_frames) from per-letter start times.

    roman[i]  : the a-z string aligned for token i ('' for tokens without sound, e.g. virama)
    is_sep[i] : True for space / punctuation tokens (they absorb pauses between words)
    A letter lasts until the next letter starts, except a word-final letter, which is capped at
    cap_s so the remaining time becomes a pause on the separator token(s) that follow it.
    """
    sr, hop = a["sampling_rate"], a["hop_length"]
    fr = lambda t: min(int(round(t * sr / hop)), n_frames)
    S = [fr(t) for t in starts]
    owner = [i for i, r in enumerate(roman) for _ in r]
    L = len(S)
    dur = [0] * len(roman)
    cap = int(round(cap_s * sr / hop))
    for j in range(L):
        nxt = S[j + 1] if j + 1 < L else n_frames
        word_final = j + 1 == L or owner[j + 1] != owner[j] and any(
            is_sep[k] for k in range(owner[j] + 1, owner[j + 1]))
        end = min(nxt, S[j] + cap) if word_final else nxt
        dur[owner[j]] += max(end - S[j], 0)
        if end < nxt:                                                  # pause after the word
            seps = [k for k in range(owner[j] + 1, owner[j + 1] if j + 1 < L else len(roman)) if is_sep[k]]
            dur[seps[0] if seps else owner[j]] += nxt - end
    dur[0] += S[0]                                                     # leading silence
    dur[int(np.argmax(dur))] += n_frames - sum(dur)                    # rounding repair
    if min(dur) < 0 or sum(dur) != n_frames:
        raise RuntimeError("invalid durations")
    return dur


# ----------------------------------------------------------------------------- vocoder
def load_vocoder(repo_dir, device):
    """Pretrained HiFi-GAN UNIVERSAL_V1 shipped inside ming024/FastSpeech2 (hifigan/ folder)."""
    sys.path.insert(0, repo_dir)
    import hifigan
    with open(os.path.join(repo_dir, "hifigan", "config.json")) as f:
        conf = hifigan.AttrDict(json.load(f))
    voc = hifigan.Generator(conf)
    ckpt = torch.load(os.path.join(repo_dir, "hifigan", "generator_universal.pth.tar"),
                      map_location=device, weights_only=False)
    voc.load_state_dict(ckpt["generator"])
    voc.eval()
    voc.remove_weight_norm()
    return voc.to(device)


@torch.no_grad()
def vocode(voc, mel):
    """mel: (1, T, n_mels) tensor -> float32 numpy waveform in [-1, 1]."""
    return voc(mel.transpose(1, 2)).squeeze().float().cpu().numpy()


# ----------------------------------------------------------------------------- plotting
def setup_fonts():
    """Pick a font that can draw Telugu if one is installed (apt install fonts-noto-core)."""
    import matplotlib
    from matplotlib import font_manager
    names = {f.name for f in font_manager.fontManager.ttflist}
    for cand in ["Noto Sans Telugu", "Noto Serif Telugu", "Lohit Telugu", "Gautami", "Nirmala UI"]:
        if cand in names:
            matplotlib.rcParams["font.family"] = [cand, "DejaVu Sans"]
            return cand
    return None


def plot_losses(results_dir, ckpt_dir, lang):
    """Reads train_log.csv / val_log.csv and writes training_loss.png and validation_loss.png."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    os.makedirs(results_dir, exist_ok=True)
    comps = ["mel", "postnet", "duration", "pitch", "energy"]
    tr_path, va_path = os.path.join(ckpt_dir, "train_log.csv"), os.path.join(ckpt_dir, "val_log.csv")
    if os.path.exists(tr_path):
        tr = pd.read_csv(tr_path)
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
        ax[0].plot(tr["step"], tr["total"], lw=1.2)
        ax[0].set(title=f"{lang}: training total loss", xlabel="step", ylabel="loss")
        for c in comps:
            ax[1].plot(tr["step"], tr[c], lw=1, label=c)
        ax[1].set(title=f"{lang}: training loss components", xlabel="step", yscale="log")
        ax[1].legend()
        for x in ax:
            x.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(results_dir, "training_loss.png"), dpi=150)
        plt.close(fig)
    if os.path.exists(va_path):
        va = pd.read_csv(va_path)
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
        ax[0].plot(va["step"], va["total"], marker="o", ms=3, lw=1.2, label="validation")
        if os.path.exists(tr_path):
            ax[0].plot(tr["step"], tr["total"], lw=0.8, alpha=0.5, label="training")
        ax[0].set(title=f"{lang}: validation total loss", xlabel="step", ylabel="loss")
        ax[0].legend()
        for c in comps:
            ax[1].plot(va["step"], va[c], marker="o", ms=2, lw=1, label=c)
        ax[1].set(title=f"{lang}: validation loss components", xlabel="step", yscale="log")
        ax[1].legend()
        for x in ax:
            x.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(results_dir, "validation_loss.png"), dpi=150)
        plt.close(fig)


def plot_variance(sent_id, tokens, dur, pitch_hz, energy, mel, a, out_dirs):
    """Writes duration/, pitch/, energy/ PNGs for one sentence.

    tokens: list[str]; dur: frames per token; pitch_hz, energy: per-token values; mel: (T, n_mels).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    setup_fonts()
    dur = np.asarray(dur)
    sec = a["hop_length"] / a["sampling_rate"]
    n = len(tokens)
    labels = [("␣" if t == " " else ("·" + t if unicodedata.category(t[0]).startswith("M") else t)) for t in tokens]
    w = max(9.0, 0.26 * n)

    # duration: bars per token
    fig, ax = plt.subplots(figsize=(w, 4))
    ax.bar(range(n), dur * sec * 1000, color="#3b6ea8")
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set(title=f"{sent_id}: predicted duration per token", ylabel="ms (frames x hop)")
    ax.set_xlim(-0.7, n - 0.3)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dirs["duration"], f"{sent_id}.png"), dpi=150)
    plt.close(fig)

    # pitch / energy: per-token values + frame-level contour over the predicted mel
    for name, vals, unit, color in [("pitch", pitch_hz, "F0 (Hz)", "#c0392b"),
                                    ("energy", energy, "energy (L2 norm)", "#27874f")]:
        fig, ax = plt.subplots(2, 1, figsize=(w, 7), gridspec_kw={"height_ratios": [1, 1]})
        frames = np.repeat(np.asarray(vals), dur)
        t = np.arange(len(frames)) * sec
        ax[0].imshow(mel.T, origin="lower", aspect="auto", extent=[0, len(mel) * sec, 0, mel.shape[1]],
                     cmap="magma")
        ax[0].set(title=f"{sent_id}: predicted mel-spectrogram", ylabel="mel bin", xlabel="time (s)")
        ax2 = ax[0].twinx()
        ax2.plot(t, frames, color="white" if name == "pitch" else "cyan", lw=1.2)
        ax2.set_ylabel(unit)
        ax[1].plot(range(n), vals, marker="o", ms=3, color=color, lw=1)
        ax[1].set_xticks(range(n))
        ax[1].set_xticklabels(labels, fontsize=8)
        ax[1].set(title=f"{sent_id}: predicted {name} per token", ylabel=unit)
        ax[1].set_xlim(-0.7, n - 0.3)
        ax[1].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dirs[name], f"{sent_id}.png"), dpi=150)
        plt.close(fig)
