"""Synthetic test material so the pipeline runs with no dataset.

NOT real speech: a glottal-pulse source through moving formant filters
with a syllable-rate envelope. Good enough to exercise the code; real
results need real dialogue (see README: swap in VCTK / DNS / your own).
"""
import numpy as np
from scipy.signal import lfilter, butter
from .stft import SR


def speechlike(dur, rng, sr=SR):
    n = int(dur * sr)
    t = np.arange(n) / sr
    f0 = 110 * 2 ** rng.uniform(0, 1) * (1 + 0.08 * np.sin(2 * np.pi * rng.uniform(0.3, 0.8) * t + rng.uniform(0, 6)))
    src = (np.mod(np.cumsum(f0) / sr, 1.0) < 0.12).astype(float) - 0.12
    out = np.zeros(n)
    seg = int(0.12 * sr)
    for s in range(0, n, seg):                      # new "vowel" every 120 ms
        chunk = src[s:s + seg]
        if rng.random() < 0.2:                      # occasional unvoiced consonant
            chunk = rng.standard_normal(len(chunk)) * 0.3
            fmts = [rng.uniform(3000, 6000)]
        else:
            fmts = [rng.uniform(300, 900), rng.uniform(1000, 2400), rng.uniform(2500, 3500)]
        y = np.zeros(len(chunk))
        for f in fmts:
            r = 0.97
            a = [1, -2 * r * np.cos(2 * np.pi * f / sr), r * r]
            y += lfilter([1 - r], a, chunk)
        out[s:s + seg] = y * np.hanning(len(chunk) + 2)[1:-1] ** 0.5
    env = np.repeat(rng.random(int(dur / 0.4) + 1) > 0.25, int(0.4 * sr))[:n].astype(float)  # pauses
    b, a = butter(1, 8 / (sr / 2)); env = lfilter(b, a, env)
    out *= env
    return out / (np.abs(out).max() + 1e-9) * 0.5


def noise(kind, n, rng, sr=SR):
    w = rng.standard_normal(n)
    if kind == "white":
        y = w
    elif kind == "pink":
        X = np.fft.rfft(w); f = np.arange(len(X)); f[0] = 1
        y = np.fft.irfft(X / np.sqrt(f), n)
    elif kind == "hum":                              # mains hum + harmonics + hiss
        t = np.arange(n) / sr
        y = sum(np.sin(2 * np.pi * 50 * k * t + rng.uniform(0, 6)) / k for k in range(1, 8)) + 0.15 * w
    elif kind == "aircon":                           # low rumble, slowly fluctuating level
        b, a = butter(2, 400 / (sr / 2)); y = lfilter(b, a, w)
        y *= 1 + 0.5 * np.sin(2 * np.pi * 0.3 * np.arange(n) / sr)
    elif kind == "babble":                           # several overlapping voices, far away
        y = sum(speechlike(n / sr, rng) for _ in range(5))
        b, a = butter(2, 2500 / (sr / 2)); y = lfilter(b, a, y)
    else:
        raise ValueError(kind)
    return y / (np.std(y) + 1e-9)


def mix(clean, nz, snr_db):
    gain = np.std(clean) / (10 ** (snr_db / 20))
    return clean + nz * gain
