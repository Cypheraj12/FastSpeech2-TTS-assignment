Date: 4 November 2026

DAILY LOG, errors,learning and decisions

recieved the assignment on 1st november,took me 2-3 days to understand the problem clearly, read and viewed the provided research papers and github code.

training FastSpeech 2 requires heavy gpu which i dont have on my laptop so using free collab t4 gpu till daily limit reaches and lets see how much i can train with it.


what i done/implemented today:

1. setup the github repo, directory structure and requirements.
2. implemented the complete fastspeech 2 model architecture with encoder, variance adaptor for duration, pitch, energy, length regulator, decoder and postnet.
3. wrote english text processing with number verbalizer, date and time normalizer, abbreviations and g2p phonemes.
4. did data selection for svarah and indictts to get 5 hours train and 1 hour validation separate data.
5. extracted letter alignments and phoneme durations using mms aligner.
6. fixed a bug in number verbalizer where very large numbers like phone numbers crashed with index error in below_1000 function.
7. trained english model on collab t4 gpu till 8500 steps, training loss came down from around 13 to 1.6.
8. trained telugu model on collab t4 gpu till 6100 steps, validation loss reached 2.19 and duration loss went down to 0.063.
9. wrote python script to plot training and validation loss curves from saved logs because training was stopped before 30000 steps so default plotting didnt trigger.
10. ran inference on telugu sentences, words and sound came out very clear, but noticed ending words like undi in t2a wrap up too fast and cut off immediately.
11. fixed the telugu ending cutoff by un-muting the final syllable energy, adjusting pitch cadence, and padding the mel spectrogram before hifigan so the ending decays naturally. tested speed 1.18 and all 5 telugu sentences are now completely finished and perfect.


what problem i am facing right now:

1. english generated audio is very blurry, muffled and mumbled, words are hard to understand unlike telugu which is clear.
2. checked predicted durations for english sentence e2a and saw that space token between words had 0 duration, so words were crashing into each other without any pause.
3. even after fixing space duration to give words small pause, there is still a noticeable shaking, vibrating disturbance in the voice.
4. collab free gpu daily limit got exhausted today so runtime disconnected and switched to cpu.


what i am trying to solve/fix it:

1. tested ground truth audio from svarah dataset directly and vocoded it with hifigan, both were completely clear so dataset and vocoder are not the issue, problem is inside trained model representation.
2. figured out why telugu is clear while english is muffled, telugu has only 1 speaker in studio so 6000 steps was 35 passes over that single voice, but english has 115 different speakers on phones so 8000 steps is only around 20 passes per speaker and model is struggling to average 115 voices.
3. decided to use the 8000 steps as multi speaker pretraining and now shifting focus to fine tune specifically on speaker 74 who has around 100 utterances and cleanest voice. wrote finetuning script to train on speaker 74 for next 2500 steps once gpu is available.
4. finalized telugu model inference with relaxed 1.18 speed and natural ending release, will run speaker 74 finetuning for english tomorrow once gpu is back.


thats the wrap for today,now exhausted,will continue the remaining fixation /improvement tomorrow.
