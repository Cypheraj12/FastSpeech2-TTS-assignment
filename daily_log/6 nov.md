# 6 Nov 2026

## **DAILY LOG, errors,learning and decisions**

today worked on fine-tuning the english model on the 4 cleanest speakers for 6000 steps to collapse the multi-speaker blurriness and improved the audio output.


### what i done/implemented today:

1. selected the top 4 cleanest speakers from svarah and ran fine-tuning on colab t4 gpu for 6000 steps with initial learning rate 1.5e-4.
2. tracked loss convergence across the run: total loss dropped from 1.27 down to 0.952, and mel loss reached 0.435 at step 6000.
3. saved the checkpoint and generated initial test audios with hifigan vocoder.
4. fixed the electric click artifact at word starts by setting neutral pause pitch (0.0) instead of negative bucket values.
5. added 4-frame word spacing floor so words don't collide, and applied a moving average filter on upper mel bins 70-80 to remove high-frequency microphone crackle.


### what problem i am facing right now:

1. very long words like antidisestablishmentarianism in sentence e1 still have some slurring on middle syllables because phone audio in svarah lacks sharp consonant bursts.
2. need to finalize all 10 test audios and variance adaptor plots for english and telugu.


### what i am trying to solve/fix it:

1. adjusted vowel and consonant duration floors so individual phonemes stay distinct.
2. generated all final test audio files and variance plots for duration, pitch, and energy.


**all training runs, audio generation, and plots for both languages are now finished.**
