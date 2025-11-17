import torchaudio
import torchaudio.transforms as T





def extract_mfcc_pipeline(audio_path, sample_rate=16000, n_mfcc=13):
  # waveform:(channel,samples)  sr: original audio sample of loaded audio file
    waveform, sr = torchaudio.load(audio_path)

    # Resample if needed
    if sr != sample_rate:
        resampler = T.Resample(sr, sample_rate)
        waveform = resampler(waveform)

    # Extract MFCC
    mfcc_transform = T.MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,    # the number of MFCC coefficients to compute for each frame


        melkwargs={
            "n_fft": 512,                     # size of the Fast Fourier Transform (FFT) window, determines the frequency resolution
            "hop_length": 160,                # num of samples between successive frames, controls overlap between frames, 160 samples at 16 kHz means a hop length of 10 milliseconds (160/16000 = 0.01s).
            "n_mels": 26,                     # num of Mel bands (filters) to use in the Mel filterbank, More bands provide finer frequency resolution.
            "win_length": 400,                # len of window function applied to each frame, 400 samples at 16 kHz means a window length of 25 milliseconds (400/16000 = 0.025s).
            "window_fn": torch.hamming_window,# Specifies the window function to apply to each segment of the audio signal before performing the FFT,  helps reduce spectral leakage.
        }
    )

    mfcc = mfcc_transform(waveform) # computes mfcc feature, [1, 13, num_frames], for mono=1
    return mfcc.squeeze(0).transpose(0, 1)  # [Time, Features]

    '''
    mfcc.squeeze(0): If the mfcc tensor has a "channel" dimension (e.g., [1, 13, num_frames]), squeeze(0) removes this dimension if it's of size 1, resulting in [13, num_frames]. This is common for mono audio.
.transpose(0, 1): Swaps the first and second dimensions of the tensor. If the shape was [n_mfcc, num_frames], it becomes [num_frames, n_mfcc]. This is often desired because deep learning models for sequence data (like recurrent neural networks or lstm or transformers) typically expect the time dimension to be first ([Time, Features]).'''