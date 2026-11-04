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

### Issues faced & current debugging: English test sentence quality

#### The exact issue observed right now
When listening to the generated audio for the five English test sentences (`E1`, `E2a`, `E2b`, `E3`, `E4`):
- The Telugu sentences (`T1`–`T4`) came out very clear, natural, and easily understandable.
- The English sentences, however, came out heavily muffled, mumbled, and largely unintelligible — only a few words could be barely made out.
- Tested slowing down duration (`dur_ctl=1.2`, `1.35`) and testing different speaker IDs (`0, 1, 2, 5, 10`). Words were still slurred.
- Diagnosed the predicted durations: discovered that every single space token `' '` between words had predicted `0.0 ms` duration, causing adjacent words to slam into each other with zero breath or pause.
- Added explicit 4-frame (~46 ms) word boundary pauses on spaces and gave short consonants a 3-frame floor (`fix_spaces.py`). This separated the words so individual words could now be identified, but revealed another serious issue: **a persistent shaking, vibrating, jittery disturbance / robotic flutter** in the voice itself.
- Tested eliminating pitch jitter and energy flutter with 3-tap moving-average smoothing (`A_smooth_pitch_energy_clean_voice.wav`) and even completely flat/neutral pitch and energy (`C_steady_neutral_pitch_zero_shake.wav`). Surprisingly, the shaking/vibrating disturbance was still clearly audible even with zero pitch or energy variation.
- **Current active experiment (`test_root_cause.py`)**: Isolating the root cause by comparing:
  1. `1_original_svarah_dataset_audio.wav`: Original audio recorded by the actual person in Svarah.
  2. `2_ground_truth_mel_vocoded.wav`: The ground-truth Svarah audio converted to Mel and reconstructed through HiFi-GAN.
  This will definitively tell us whether the jitter/shaking is an inherent acoustic artifact of Svarah's noisy mobile phone recordings, a Mel-scale/phase incompatibility with the HiFi-GAN universal vocoder, or an issue in the model's mel-linear/decoder representations.

---

### Tomorrow's plan (Nov 5)
- Finalize Telugu commit (text processing, config, inference script, test sentences).
- Add the Colab notebook to the repository.
- Complete English voice tuning and document findings in the technical report.
