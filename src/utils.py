import librosa
import numpy as np

def preprocess(file_path, sr=32000, mono=True, n_fft=1024, hop_length=192, n_mels=128, fmax=None, log_spec=False):
    if mono:
        sig, sr = librosa.load(file_path, sr=sr, mono=True)
        sig = sig[np.newaxis]
    else:
        sig, sr = librosa.load(file_path, sr=sr, mono=False)
        # sig, sf_sr = sf.read(file_path)
        # sig = np.transpose(sig, (1, 0))
        # sig = np.asarray([librosa.resample(s, sf_sr, sr) for s in sig])
    # sig = librosa.effects.pitch_shift(sig, sr, n_steps=pitch_shift)

    for y in sig:

        # compute stft
        stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=None, window='hann', center=True,
                            pad_mode='reflect')

        # keep only amplitures
        stft = np.abs(stft)

        # spectrogram weighting
        if log_spec:
            stft = np.log10(stft + 1)
        else:
            freqs = librosa.core.fft_frequencies(sr=sr, n_fft=n_fft)
            stft = librosa.perceptual_weighting(stft**2, freqs, ref=1.0, amin=1e-10, top_db=99.0)

        # apply mel filterbank
        spectrogram = librosa.feature.melspectrogram(S=stft, sr=sr, n_mels=n_mels, fmax=fmax)

    # print(sig.shape)
    # print(spectrogram.shape)
    return spectrogram
