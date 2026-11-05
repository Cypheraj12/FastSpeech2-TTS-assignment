# 5 Nov 2026

## **DAILY LOG, errors,learning and decisions**

got back to work today after collab gpu limit reset. main focus today was fixing the english audio blurriness and wrapping up the pipeline.


### what i done/implemented today:

1. connected to collab t4 gpu and restored preprocessed dataset from drive backup.
2. verified and finalized complete telugu codebase with common, config, model, preprocess, text, train, and inference scripts.
3. tested english base checkpoint on sentence e1 and e3 to inspect duration predictions. noticed tokens were getting rounded down to 0 frames because duration predictor was outputting small values, causing entire sentence to collapse into 0.14 seconds.
4. added word boundary duration clamping so spaces between words always have minimum 5 frames (58 ms) pause, preventing words from smashing together.
5. ran ground truth test on svarah dataset sample 0 and speaker 74 directly through hifigan vocoder, confirmed that audio came out perfectly clear with zero noise, proving that hifigan vocoder and mel extraction are completely solid.
6. tested multiple speaker ids on the base 8500 step model, speaker 74 was noticeably clearer than default speaker 73 and speaker 12.
7. wrote and launched single speaker adaptation script for speaker 74, filtering the 88 training utterances of speaker 74 and fine tuning for 2500 steps with cosine learning rate decay down to 2e-5.
8. observed rapid loss convergence during fine tuning, mel loss dropped sharply from 0.64 down to 0.48 within first 300 steps as the model collapsed the 115 speaker averaging noise.
9. committed clean telugu pipeline and today's work log to github.


### what problem i am facing right now:

1. multi speaker training from scratch on crowdsourced svarah data suffered from l1 loss averaging, because 115 speakers on mobile phones had conflicting background noise and room acoustics, making the decoder output smoothed blurry spectrograms.
2. needed to find the sweet spot for fine tuning steps so speaker 74 voice gets sharp without overfitting on the 88 sentences.


### what i am trying to solve/fix it:

1. chose 2500 steps with cosine learning rate decay and saved intermediate checkpoints at 1500 and 2000 steps so i can compare and pick the best sounding model.
2. once fine tuning is finished, will generate all five english test sentences e1 to e4 and inspect their variance adaptor plots.
3. will download the full results folder from drive, verify all wav files, and prepare the technical report tomorrow.


**thats the progress for nov 5, english training is currently finishing up and telugu is completely ready.**
