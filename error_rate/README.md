# Error rate

The error rate for each bird, algorithm (BD>LC>GS, LC>BD&GS, or LC&GS>BD&GS), and training set duration (2 or 8 minutes). Averaging them reproduces Table 1, and plotting them reproduces Fig. 6.

Each column shows algorithm name, training set duration, note error rate, timing error rate, and note & timing error rate.
+ Along with note error rates, the number of incorrectly identified notes and the number of total notes are provided. As written in the paper, a note error rate is defined as the quotient of them.
+ Along with timing and note & timing error rates, the number of incorrectly identified time frames and the number of total time frames are provided. As written in the paper, these error rates are defined as the quotients of them.

Because time frames are those in the spectrogram, they depend on the spectrogram parameters. The number of total time frames is defined as follows: *sum([ceil(wave_length - overlap_length / step_length) for each sequence])*, where *overlap_length* = *fft_length* - *step_length*, *fft_length* = 512, and *step_length* = 32 in our setting.

In this study, error rates were calculated by pooling all notes or time frames across cross-validation folds. For example, if, in each fold, incorrectly identified notes / total notes = 1/2, 3/4, and 2/5, the final error rate is (1+3+2)/(2+4+5), but not (1/2 + 3/4 + 2/5)/3. We did not calculate the cross-validation error by averaging error rates over cross-validation folds (although this may be a more standard way). 

Error rates provided in figshare along with wave files are NOT those reported in the paper. They are the expected results of running the code in this repository, in which several procedures (eg. data split of cross-validation, training hyperparameters, ...) are different from those reported in the paper.

## Test codes
+ [test/calc_average.py](error_rate/test/calc_average.py) calculates average error rates over birds.
+ [test/total_time_frames.py](error_rate/test/total_time_frames.py) shows how to calculate the number of total time frames.

