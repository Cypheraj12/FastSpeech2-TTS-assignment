# FastSpeech 2: English and Telugu Text-to-Speech Synthesis
**Author:** Anant Joshi

An independent implementation and evaluation of the FastSpeech 2 non-autoregressive Text-to-Speech (TTS) architecture for English and Telugu. The system features Feed-Forward Transformer (FFT) encoder/decoder blocks, a Variance Adaptor predicting duration, pitch (F0), and energy, and a pre-trained universal HiFi-GAN neural vocoder.

---

## Project Structure

```
project/
├── English/
│   ├── train.py
│   ├── preprocess.py
│   ├── inference.py
│   ├── config.yaml
│   └── checkpoints/
├── Telugu/
│   ├── train.py
│   ├── preprocess.py
│   ├── inference.py
│   ├── config.yaml
│   └── checkpoints/
├── Results/
│   ├── English/
│   │   ├── training_loss.png
│   │   ├── validation_loss.png
│   │   ├── test_generated_audios/
│   │   └── variance_adaptor_plots/
│   │       ├── duration/
│   │       ├── pitch/
│   │       └── energy/
│   └── Telugu/
│       ├── training_loss.png
│       ├── validation_loss.png
│       ├── test_generated_audios/
│       └── variance_adaptor_plots/
│           ├── duration/
│           ├── pitch/
│           └── energy/
├── Report/
│   └── Technical_Report.pdf
├── README.md
├── daily_log/
│   ├── 4 nov.md
│   ├── 5 nov.md
│   ├── 6 nov.md
│   └── 7 nov.md
└── requirements.txt
```

---

## Architecture & Adaptations

The implementation follows the core architectural principles of FastSpeech 2 (Ren et al., 2021) with targeted adaptations per language:

| Component | Reference Implementation | This Project |
|:---|:---|:---|
| **Forced Alignment** | Montreal Forced Aligner (TextGrid) | Torchaudio **MMS_FA** (multilingual CTC) + custom Viterbi aligner (bypasses MFA setup and supports Telugu natively). |
| **English Frontend** | CMUdict dictionary lookup | `g2p_en` ARPAbet phonemes with lexical stress, paired with a hierarchical number, time, and ordinal verbalizer. |
| **Telugu Frontend** | — | Native Telugu Unicode character tokenization (U+0C00–U+0C7F) with a specialized Telugu number and date expander. |
| **Speaker Modeling** | Single speaker / LibriTTS multi | Learned 256-dim speaker embedding for Svarah (115 speakers); single-speaker modeling for IndicTTS. |
| **Vocoder** | Pretrained HiFi-GAN | Universal HiFi-GAN Generator (v1) operating on 80-bin log-mel spectrograms (22.05 kHz). |
| **Variance Adaptor** | Duration, Pitch, Energy | Explicit token-level duration, continuous pitch ($F_0$ via PyWorld DIO + StoneMask), and frame energy embeddings. |

---

## Dataset Selection & Preprocessing Strategy

1. **Audio Standardization:** All waveforms are resampled to 22.05 kHz mono and peak-normalized to $[-0.95, 0.95]$. Utterance durations are constrained to $[1.0, 10.0]$ seconds.
2. **Train/Validation Disjointness:** Seeded shuffling (seed 42) filters out duplicate transcripts. The first 5 hours are allocated to training and the subsequent 1 hour to validation, guaranteeing zero overlap across utterances and text.
3. **English (Svarah):** Sampled from `ai4bharat/Svarah` across 115 Indian-accented speakers. A learned speaker embedding conditions the acoustic encoder.
4. **Telugu (IndicTTS):** Streamed from `SPRINGLab/IndicTTS_Telugu`, filtered to a single consistent studio speaker voice.

---

## Execution & Reproduction

### 1. Installation
```bash
pip install -r requirements.txt
```

Download and unpack the pre-trained HiFi-GAN vocoder:
```bash
git clone --depth 1 https://github.com/ming024/FastSpeech2.git FastSpeech2_ref
cd FastSpeech2_ref/hifigan && unzip -o generator_universal.pth.tar.zip
```

### 2. Preprocessing & Alignment
Run inside `English/` or `Telugu/`:
```bash
# Stage 1: Dataset selection (~5h train / ~1h validation) and manifest generation
python preprocess.py --stage select

# Stage 2: Feature extraction (80-bin mel, PyWorld F0, energy, and MMS_FA durations)
python preprocess.py --stage features
```

### 3. Training
```bash
python train.py --config config.yaml
```
- Logs step-wise loss metrics (`mel`, `postnet`, `duration`, `pitch`, `energy`) every 100 steps.
- Evaluates on the validation set every 1,000 steps and saves the optimal model state to `checkpoints/best.pt`.

### 4. Inference & Variance Analysis
```bash
python inference.py --config config.yaml
```
Synthesizes speech for the benchmark test sentences and outputs:
- Generated waveforms into `Results/<Language>/test_generated_audios/`
- Duration, pitch contour, and energy plots into `Results/<Language>/variance_adaptor_plots/`

---

## Benchmark Test Sentences

The evaluation suite tests phonological edge cases, stress shifts, loanwords, and orthographic spacing:
* **English:** Includes extreme polysyllabic compounds (*antidisestablishmentarianism*), noun-verb heteronym stress shifts (*conduct* in E2a vs. E2b), spacing anomalies (*rhyth m*), and temporal code-mixing (*10:45 AM*, *12 November 2026*, *andaroo*).
* **Telugu:** Preserves exact conjuncts, split tokens (*అడ విలో*), polysemous nouns (*పట్టు* grip vs. silk), and Telugu temporal expressions (*2026 నవంబర్ 12వ తేదీ ఉదయం 10:45 గంటలకు*).

---

## Checkpoints

Due to GitHub's 100 MB file-size limit, trained model checkpoints (~128 MB each) are hosted on Google Drive:

* **English Checkpoints:** [Download English Checkpoints (Google Drive)](https://drive.google.com/drive/folders/1sH6JH3BGcYfuDA7JYUP3U6IBhGcGVe1S?usp=drive_link)
  * **Best / Used for Final Inference:** `ft_top3_clean_best.pt` (fine-tuned on clean speaker data; eliminates multi-speaker blurring).
  * Base multi-speaker model: `best.pt` (step 7,000, lowest validation loss across 115 speakers).

* **Telugu Checkpoints:** [Download Telugu Checkpoints (Google Drive)](https://drive.google.com/drive/folders/1754k2h3_iwyKGWcj_v0_TZbN0WACzoIt?usp=sharing)
  * **Best / Used for Final Inference:** `best.pt` (step 6,000, lowest validation loss of 2.1991).

To run inference, place the downloaded `.pt` files into their respective directory:
* `English/checkpoints/`
* `Telugu/checkpoints/`
