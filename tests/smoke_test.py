"""Fast sanity check (a few seconds, no data needed). Run from inside English/ or Telugu/:

    cd English && python ../tests/smoke_test.py
"""
import json
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
import common
import text as frontend
from model import FastSpeech2, fastspeech2_loss

cfg = common.load_config("config.yaml")
print("language:", cfg["language"], "| symbols:", frontend.N_SYMBOLS)

# 1. frontend on the five test sentences
sents = json.load(open("test_sentences.json", encoding="utf-8"))
for sid, s in sents.items():
    toks = frontend.text_to_tokens(s)
    assert len(toks) > 5, sid
    print(f"{sid}: {len(toks)} tokens | normalised: {frontend.normalize(s)[:90]}")

# 2. model forward/backward with dummy stats, a batch of 3 and tiny variable lengths
stats = dict(pitch_min=-3., pitch_max=4., energy_min=-2., energy_max=5.)
n_spk = 5 if cfg["language"] == "English" else 1
m = FastSpeech2(cfg, frontend.N_SYMBOLS, stats, n_spk)
print(f"params: {sum(p.numel() for p in m.parameters()) / 1e6:.1f}M")
B, T = 3, 12
src = torch.randint(1, frontend.N_SYMBOLS, (B, T))
src_len = torch.tensor([12, 9, 7])
d = torch.randint(0, 6, (B, T))
for i, l in enumerate(src_len):
    d[i, l:] = 0
    d[i, 0] = max(int(d[i, 0]), 1)
mel_t = torch.randn(B, int(d.sum(1).max()), 80)
p, e = torch.randn(B, T), torch.randn(B, T)
out = m(src, src_len, speaker=torch.tensor([0, 1, 0]) if n_spk > 1 else None, d_target=d, p_target=p, e_target=e)
loss = fastspeech2_loss(out, mel_t, d, p, e)
loss["total"].backward()
assert all(torch.isfinite(v) for v in loss.values())
print("train step OK:", {k: round(v.item(), 3) for k, v in loss.items()})

# 3. inference path (no targets)
m.eval()
with torch.no_grad():
    o = m(src[:1], src_len[:1], speaker=torch.tensor([0]) if n_spk > 1 else None)
print("inference OK: mel", tuple(o["mel_post"].shape), "| total predicted frames", int(o["d_rounded"].sum()))

# 4. alignment -> durations (synthetic CTC posteriors)
rng = np.random.RandomState(0)
logp = np.log(rng.dirichlet(np.ones(6) * 0.3, size=60) * 0.05 + 1e-9)
logp[:, 0] = np.log(0.9)
for f, k in zip([5, 12, 30, 47], [1, 2, 3, 4]):
    logp[f, k], logp[f, 0] = np.log(0.9), np.log(0.05)
assert list(common.ctc_letter_starts(logp, [1, 2, 3, 4])) == [5, 12, 30, 47]
print("aligner OK")

# 5. vocoder + mel features, only if the reference repo is present
import os
repo = cfg["paths"]["vocoder_repo"]
if os.path.exists(os.path.join(repo, "hifigan", "generator_universal.pth.tar")):
    voc = common.load_vocoder(repo, "cpu")
    wav = torch.from_numpy(np.sin(2 * np.pi * 220 * np.arange(22050) / 22050).astype(np.float32) * 0.5)
    mel, en = common.mel_and_energy(wav, cfg["audio"])
    assert mel.shape == (22050 // 256, 80), mel.shape
    y = common.vocode(voc, torch.from_numpy(mel).unsqueeze(0))
    print("mel -> HiFi-GAN OK:", mel.shape, "->", y.shape)
else:
    print("(vocoder repo not found - skipped)")
print("ALL OK")
