# Engineering Work Log & Daily Notes

**Project:** FastSpeech 2 TTS (English & Telugu)  
**Author:** Anant Joshi  
**Repo:** `FastSpeech2-TTS-assignment`

---

## November 4, 2026

### What got done today
- Setup base project structure, `.gitignore`, and minimal dependencies (`requirements.txt`).
- Implemented FastSpeech 2 architecture with multi-speaker embedding support (encoder FFT blocks, variance adaptor with duration/pitch/energy predictors and length regulator, decoder FFT blocks, linear mel projection, and 5-layer PostNet).
- Added English text frontend:
  - Text normalization: clock times (`10:45 AM` -> `ten forty five a m`), dates (`12 November 2026`), acronym spelling, currency, and number expansion.
  - CMU Pronouncing Dictionary + `g2p_en` phonemizer with ARPAbet token mapping.
- Added data selection and forced-alignment extraction for Svarah:
  - MMS_FA multilingual alignment with Viterbi CTC decoding.
  - 5.0 hours train (2,754 utts) and 1.0 hour validation disjoint sets.
- Fixed an `IndexError` bug in `English/text.py`: Svarah transcripts contain long numbers (like phone numbers/IDs >= 10 digits) which caused array overflow in `_ONES` inside `below_1000`. Patched it using 1000-modulo chunking and added `inflect` fallback.
- Trained English model on Colab (T4 GPU) up to step 8,500:
  - Training loss dropped from ~13 to 1.6.
  - Best validation loss reached 2.883 at step 7,000 (mel loss ~0.640).
- Trained Telugu model (IndicTTS) on Colab up to step 6,100:
  - Monotonic drop across all loss components.
  - Best validation loss reached 2.199 at step 6,000 (mel loss 0.567, duration loss down to 0.063).
- Fixed loss curves plotting: since training was stopped before the full 30,000-step loop, `common.plot_losses` didn't execute at the end of `train.py`. Wrote a standalone Python snippet to parse `train_log.csv` and `val_log.csv` directly from Drive and render the 4 loss curve plots.

---

### Issues faced & engineering strategy: English multi-speaker adaptation

#### The exact issue observed
When listening to the generated audio for the five English test sentences (`E1`, `E2a`, `E2b`, `E3`, `E4`):
- The Telugu sentences (`T1`–`T4`) came out very clear, natural, and easily understandable.
- The English sentences, however, came out heavily muffled, mumbled, and largely unintelligible — only a few words could be made out.
- Diagnosed the predicted durations: discovered that every single space token `' '` between words had predicted `0.0 ms` duration, causing adjacent words to slam into each other with zero breath or pause.
- Added explicit 4-frame (~46 ms) word boundary pauses on spaces and gave short consonants a 3-frame floor. This separated the words so individual words could now be identified, but revealed another serious issue: **a persistent shaking, vibrating disturbance / robotic flutter** in the voice itself.
- Verified the ground truth: tested raw Svarah audio from the dataset (`1_original_svarah_dataset_audio.wav`) and vocoded ground-truth Mel through HiFi-GAN (`2_ground_truth_mel_vocoded.wav`). Both were perfectly clear, confirming the issue is inside the model representation, not the dataset or the vocoder.

#### Root cause analysis: Studio Single-Speaker vs Crowdsourced Multi-Speaker
Comparing the two training regimes explained the contrast:
- **Telugu (IndicTTS)**: Single speaker in a sound booth. At 6,000 steps across 2,753 utterances, the model performed **~35 full passes** over that single clean voice, achieving rapid convergence.
- **English (Svarah)**: Crowdsourced across **115 different mobile phone speakers**. At 8,000 steps, each speaker had only been seen ~20–25 times. The multi-speaker decoder was attempting to average across 115 conflicting vocal tracts and room acoustics, creating an indistinct, blurred composite voice.

#### Engineering Decision & Strategy
Instead of blindly continuing multi-speaker training under GPU compute limits:
1. Pre-trained the model on the full 115-speaker Svarah corpus for 8,000 steps to build generalized English phoneme-to-acoustic representations.
2. Analyzed speaker distributions in the training set and identified **Speaker 74** (and Speaker 73) as having the highest data volume and cleanest recording profile.
3. Implemented a **Single-Speaker Fine-Tuning Adaptation (`finetune_speaker.py`)**: shifting training focus from 115 speakers directly onto Speaker 74 for an additional 2,000–2,500 steps. This allows the decoder to collapse its multi-speaker uncertainty onto a single vocal tract, sharpening phonetic clarity while preserving the generalized phoneme alignment learned during the 8,000-step pre-training phase.
4. Document this two-stage pre-training + speaker adaptation approach in the Technical Report as a principled response to resource and data heterogeneity constraints.

---

### Tomorrow's plan (Nov 5)
- Finalize Telugu commit (text processing, config, inference script, test sentences).
- Add the Colab notebook to the repository.
- Complete English voice tuning and document findings in the technical report.
