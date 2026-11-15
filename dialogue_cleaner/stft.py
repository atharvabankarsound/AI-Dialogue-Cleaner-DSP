"""STFT / inverse STFT written from scratch (numpy only).

sqrt-Hann analysis + synthesis windows at 75% overlap give perfect
reconstruction via weighted overlap-add, which is also what the
real-time plugin version will use.
"""
import numpy as np

SR = 16000
N_FFT = 512
HOP = 128


def _window(n_fft=N_FFT):
    return np.sqrt(np.hanning(n_fft + 1)[:-1])


def stft(x, n_fft=N_FFT, hop=HOP):
    w = _window(n_fft)
    x = np.pad(x, (n_fft, n_fft))
    n_frames = 1 + (len(x) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    return np.fft.rfft(x[idx] * w, axis=1)  # (frames, bins)


def istft(X, length, n_fft=N_FFT, hop=HOP):
    w = _window(n_fft)
    frames = np.fft.irfft(X, n=n_fft, axis=1) * w
    out = np.zeros(n_fft + hop * (len(X) - 1))
    norm = np.zeros_like(out)
    for i, f in enumerate(frames):
        out[i * hop:i * hop + n_fft] += f
        norm[i * hop:i * hop + n_fft] += w ** 2
    out /= np.maximum(norm, 1e-8)
    return out[n_fft:n_fft + length]
