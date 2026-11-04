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

### Issues faced & current debugging: English audio quality

#### The Problem
When running inference on the five English test sentences (`E1` to `E4`), the generated speech came out muffled, mumbled, and hard to understand, whereas the Telugu model produced very clear speech out of the box. Even after adjusting duration and volume, there is still an audible disturbance/flutter in the English voice.

#### Observations & Root-Cause Analysis
1. **Dataset difference (Studio vs Crowdsourced)**:
   - Telugu uses IndicTTS: a single speaker recorded in a professional sound booth with a studio condenser microphone. Clean acoustics, no ambient noise, very consistent timbre.
   - English uses Svarah: crowdsourced across ~115 speakers recording conversational sentences via mobile phones in varying real-world rooms. Background noise, phone mic compression, room reverberation, and wide accent variation are present directly in the training audio. FastSpeech 2 naturally learns this composite acoustic profile.
2. **Word spacing collapse (0 ms duration on spaces)**:
   - Inspected `test_predictions.json` for sentence `E2a`:
     ```
     tokens:    ['DH', 'AH0', ' ', 'AO1', 'R', 'K', 'AH0', 'S', 'T', 'R', 'AH0', 'Z', ' ', 'K', ...]
     durations: [139.3, 92.9, 0.0, 116.1, 58.0, 34.8, 116.1, 58.0, 46.4, 92.9, 0.0, ...]
     ```
   - Every `' '` (space token) predicted `0.0` duration frames. During inference, `length_regulate` discarded the space tokens entirely, smashing words into one continuous slur (`Theorchestrasconduct...`).
   - Gave space tokens an explicit 4-frame (~46 ms) word boundary and short consonants a minimum frame floor. Words became separated and recognizable, but the voice timbre still has an audible buzz/tremor.
3. **Pitch jitter & energy flutter**:
   - Inspected the predicted pitch values: rapid jumps between adjacent phonemes (e.g. 137 Hz -> 177 Hz -> 148 Hz on a single phoneme transition).
   - This rapid variation acts like a fast vibrato/tremolo effect.
   - Tested moving-average smoothing over the predicted pitch and energy vectors.
4. **Current Diagnostic**:
   - Testing ground truth audio directly from Svarah (`1_original_svarah_dataset_audio.wav`) and vocoding ground-truth Mel through HiFi-GAN (`2_ground_truth_mel_vocoded.wav`) to isolate how much of the blur is dataset acoustics versus model training artifacts.
   - Also noticed `common.load_vocoder` adds `/content/FastSpeech2_ref` to `sys.path`, which can shadow local `model.py` and `text.py` imports in the interactive kernel; cleaned up `sys.path` after loading.

---

### Tomorrow's plan (Nov 5)
- Finalize Telugu commit (text processing, config, inference script, test sentences).
- Add the Colab notebook to the repository.
- Complete English voice tuning and document findings in the technical report.
