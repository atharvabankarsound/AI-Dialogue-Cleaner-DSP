"""Noise-type identification: hand-built DSP features + a small neural classifier.

Features (84 values, all computed from the noisy clip itself):
  - spectral shape of the steady noise: 10th-percentile magnitude per bin -> 40 mel bands
  - spectral shape of the whole clip: mean power -> 40 mel bands
  - gustiness: how much the quietest low-frequency moments vary over time
  - tonality: narrow peaks below 500 Hz, and energy on a 50 / 60 Hz harmonic comb

The classifier is trained on SYNTHETIC noise only, so on real recordings it is a best guess.
"""
import numpy as np
from scipy.signal import butter, sosfilt
from .stft import stft, SR
from .bands import FB
from .synth import speechlike, noise, mix

LABELS = ["White noise / hiss", "Broadband room tone", "Mains hum", "Low rumble",
          "Wind", "Background voices", "Telephone-quality audio", "Little or no noise"]


def _welch_low(x, n=2048, hop=1024):
    w = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / n)
    if len(x) < n:
        x = np.pad(x, (0, n - len(x)))
    F = 1 + (len(x) - n) // hop
    idx = np.arange(n)[None, :] + hop * np.arange(F)[:, None]
    mag = np.abs(np.fft.rfft(x[idx] * w, axis=1))[:, :65]
    return np.sort(mag, axis=0)[int(0.2 * F)]


def features(x):
    mag = np.abs(stft(x)); T = len(mag)
    nf = np.sort(mag, axis=0)[int(0.1 * T)]
    A = np.log10(FB @ (nf ** 2) + 1e-10)
    B = np.log10(FB @ (mag ** 2).mean(axis=0) + 1e-10)
    ref = B.mean()
    L = np.log10((mag[4:T - 4, 1:10] ** 2).sum(axis=1) + 1e-10)   # ~30-280 Hz, skipping padded edges
    nb = len(L) // 25
    mins = L[:nb * 25].reshape(nb, 25).min(axis=1); s = np.sort(mins)
    gust = [mins.std(), s[int(0.9 * (nb - 1))] - s[int(0.1 * (nb - 1))]]
    P = _welch_low(x); band = P[3:65]; med = np.median(band) + 1e-10
    d1 = np.log10(band.max() / med + 1e-10)
    d2 = max(np.mean([np.log10(P[int(round(k * f0 / 7.8125))] / med + 1e-10) for k in range(1, 5)]) for f0 in (50, 60))
    return np.concatenate([A - ref, B - ref, gust, [d1, d2]])


def make_clip(label, rng):
    c = speechlike(rng.uniform(3, 7), rng); n = len(c)
    if label == 0: y = mix(c, noise("white", n, rng), rng.uniform(-5, 22))
    elif label == 1: y = mix(c, noise("pink", n, rng), rng.uniform(-5, 22))
    elif label == 2: y = mix(c, noise(rng.choice(["hum", "hum60"]), n, rng), rng.uniform(-5, 22))
    elif label == 3: y = mix(c, noise("aircon", n, rng), rng.uniform(-5, 22))
    elif label == 4: y = mix(c, noise("wind", n, rng), rng.uniform(-5, 22))
    elif label == 5: y = mix(c, noise("babble", n, rng), rng.uniform(-5, 22))
    elif label == 6:
        y = mix(c, noise(rng.choice(["white", "pink"]), n, rng), rng.uniform(18, 35))
        y = sosfilt(butter(4, [300, 3400], btype="band", fs=SR, output="sos"), y) + 1e-3 * np.std(y) * rng.standard_normal(n)
    else: y = mix(c, noise("white", n, rng), rng.uniform(40, 60))
    return y * rng.uniform(0.2, 1.0)
