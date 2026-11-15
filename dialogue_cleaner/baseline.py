"""Classic DSP baseline: spectral gating with a tracked noise floor.

1. Noise floor per bin = running low percentile of the magnitude
   (a simple form of minimum statistics; needs no "learn noise" button).
2. Wiener-style soft gain with over-subtraction and a gain floor.
3. Gain smoothing: fast attack / slow release in time, light blur in
   frequency, to reduce musical noise.
"""
import numpy as np
from scipy.ndimage import percentile_filter, uniform_filter1d
from .stft import stft, istft


def noise_floor(mag, win_frames=100, pct=10):
    floor = percentile_filter(mag, pct, size=(win_frames, 1), mode="nearest")
    return floor * 1.5  # bias compensation: a low percentile underestimates the mean


def smooth_gain(g, attack=0.6, release=0.92):
    out = np.empty_like(g)
    prev = g[0]
    for t in range(len(g)):
        coef = np.where(g[t] > prev, attack, release)
        prev = coef * prev + (1 - coef) * g[t]
        out[t] = prev
    return uniform_filter1d(out, size=3, axis=1)


def spectral_gate(x, over_sub=2.0, floor_db=-18.0):
    X = stft(x)
    mag = np.abs(X)
    nf = noise_floor(mag)
    snr_post = (mag ** 2) / (over_sub * nf ** 2 + 1e-12)
    g = np.clip(1 - 1 / np.maximum(snr_post, 1e-8), 0, 1)
    g = np.maximum(smooth_gain(g), 10 ** (floor_db / 20))
    return istft(X * g, len(x))
