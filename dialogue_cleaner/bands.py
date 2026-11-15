"""Mel-spaced triangular band layout shared by the DSP and neural stages.

The network predicts one gain per band (RNNoise-style) instead of one per
FFT bin: 40 outputs instead of 257, which keeps it small enough for
real-time use inside a plugin.
"""
import numpy as np
from .stft import SR, N_FFT

N_BANDS = 40


def _mel(f):
    return 2595 * np.log10(1 + f / 700)


def _imel(m):
    return 700 * (10 ** (m / 2595) - 1)


def filterbank(n_bands=N_BANDS, sr=SR, n_fft=N_FFT):
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    edges = _imel(np.linspace(_mel(50), _mel(sr / 2), n_bands + 2))
    fb = np.zeros((n_bands, len(freqs)))
    for b in range(n_bands):
        lo, mid, hi = edges[b:b + 3]
        fb[b] = np.clip(np.minimum((freqs - lo) / (mid - lo), (hi - freqs) / (hi - mid)), 0, None)
    fb[0, freqs <= edges[1]] = 1.0  # cover DC..first centre
    return fb


FB = filterbank()
FB_NORM = FB / FB.sum(axis=0, keepdims=True).clip(1e-8)  # bands -> bins interpolation


def band_energy(power):            # (frames, bins) -> (frames, bands)
    return power @ FB.T


def bands_to_bins(gains):          # (frames, bands) -> (frames, bins)
    return gains @ FB_NORM
