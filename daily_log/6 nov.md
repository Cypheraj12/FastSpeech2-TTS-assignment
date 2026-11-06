# 6 Nov 2026

## **DAILY LOG, errors,learning and decisions**

today worked on running and finishing the single speaker adaptation on the english model to solve the multi-speaker blurriness issue.


### what i done/implemented today:

1. launched the 2500 steps fine-tuning script on collab t4 gpu using speaker 28 and 74 clean utterances from the svarah training set.
2. monitored training loss during adaptation: total loss dropped from 1.59 down to 0.993, and mel spectral loss dropped down to 0.460 at step 2500, showing that fine-tuning on a single consistent voice successfully collapsed the 115-speaker median blurring.
3. saved the best fine-tuned checkpoint (`ft_top3_clean_best.pt`) directly to google drive.
4. tested synthesis on initial test sentences with the adapted model: voice clarity improved significantly compared to the base 8500 step checkpoint, sounds like a distinct individual rather than an averaged crowd.
5. inspected the initial test outputs and noticed some remaining issues: slight crackling/jitter on high frequencies, an electric click sound at the beginning of words, and ending syllables wrapping up too fast.


### what problem i am facing right now:

1. there is a noticeable high-frequency jitter/shaking on some words and an onset click at word starts when transitioning from silence to speech.
2. need to fine-tune the inference variance shaping (durations, pitch, and energy) to get crisp word separation and natural sentence endings before generating the final test set.


### what i am trying to solve/fix it:

1. experiment with pacing controls, pitch smoothing, and duration floors for vowels and consonants.
2. test different post-processing filters on the predicted mel-spectrogram to remove the high-frequency background crackle.
3. finalize the winning inference configuration and generate all benchmark test audios.


**thats the progress for nov 6, single speaker adaptation completed successfully and moving to final inference tuning.**
