Daily Work Log

Date: 4 November 2026
Author: Anant Joshi

Received the assignment on 1st November and took 2-3 days to analyse the problem statement, go through the FastSpeech 2 reference papers and repository, understand the datasets, and plan out the architecture. Training FastSpeech 2 from scratch requires a heavy GPU which I do not have on my local laptop, so I am using Google Colab free T4 GPU until the daily usage limit reaches, and let us see how much I can train within these free limits.

What I done and implemented today:

1. Created the project repository, set up the directory structure, and wrote requirements file.
2. Implemented the complete FastSpeech 2 model architecture including encoder FFT blocks, variance adaptor with duration, pitch, and energy predictors, length regulator, decoder FFT blocks, linear projection, and 5-layer PostNet.
3. Built the English text processing frontend with number normalizer, date normalizer, time normalizer, acronym expander, and CMU dictionary with g2p_en phonemizer.
4. Set up dataset selection for Svarah and IndicTTS to extract 5 hours of training data and 1 hour of validation data with disjoint sets.
5. Used torchaudio MMS forced aligner to extract letter start times and compute phoneme durations.
6. Fixed a bug in the English number verbalizer where very large numbers like phone numbers and long ID strings caused an index error in below_1000.
7. Trained the English model on Colab T4 GPU up to step 8500 where total training loss dropped down to around 1.6.
8. Trained the Telugu model on Colab T4 GPU up to step 6100 where validation loss reached 2.19 and duration loss dropped down to 0.063.
9. Wrote a python script to parse the training and validation log files to generate loss curves for both languages because training was stopped before the full 30000 steps so the default plotting call was missed.
10. Tested inference on the Telugu model. Telugu words and sound came out very clear, but noticed the ending word of sentences like undi in T2a wraps up too fast and cuts off immediately without full decay. Also tested baseline pace versus slightly slower pace.

What problem I am facing right now:

1. English test audio quality is very blurry, muffled, and mumbled. It is hard to understand what words are being spoken, whereas Telugu audio came out clear.
2. When checking predicted durations for English sentence E2a, found out that the space token between words had duration 0, meaning all words were glued together without any pause.
3. Even after fixing the space duration to give words a natural pause, the English voice has a noticeable shaking, trembling, or vibrating disturbance in it.
4. The free GPU limit on my Google Colab account got exhausted for today, so the session got disconnected and switched to standard CPU.
5. In Telugu sentences, the last word of every sentence cuts off too abruptly as soon as the last letter finishes.

What I am trying to solve and fix it:

1. Tested ground truth audio from the Svarah dataset and vocoded it through HiFi-GAN. Both came out completely clear, proving that the vocoder and the dataset audio are fine, and the issue is inside the trained model.
2. Analyzed why Telugu converged so fast while English is muffled. Telugu has only 1 speaker in a quiet studio booth, so 6000 steps meant 35 full passes over that single voice. English has 115 different speakers on mobile phones, so at 8000 steps the model only saw each speaker around 20 times and is trying to average across 115 different voices at once.
3. Instead of trying to continue training across all 115 speakers under limited GPU time, decided on a fine-tuning strategy: use the 8000 steps as multi-speaker pre-training, and then fine-tune specifically on Speaker 74, who has around 100 utterances and the cleanest voice. Wrote the fine-tuning script to train on Speaker 74 for 2500 steps once GPU access is back.
4. For the Telugu audio ending cutoff, added trailing silence padding and extra duration frames to the final syllable so the ending words ring out and finish completely without sudden cutoff. Also tested slightly slower pace around 1.08 to 1.12 which sounds more relaxed and natural.

That is the wrap for today, now exhausted, will continue the remaining fixation and improvement tomorrow.
