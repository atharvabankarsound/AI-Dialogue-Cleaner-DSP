"""Hybrid cleaner: DSP front end + small neural mask predictor + DSP back end.

  DSP in : STFT, mel band energies, tracked noise floor (from baseline.py)
  Neural : MLP sees [log band energy, log band SNR estimate] over a short
           context window and predicts the ideal ratio mask per band
  DSP out: gain smoothing, gain floor, bands->bins interpolation, ISTFT

The prototype uses scikit-learn's MLP so it trains on CPU in seconds.
train_torch.py holds the GRU version meant for real datasets.
"""
import numpy as np
from sklearn.neural_network import MLPRegressor
from .stft import stft, istft
from .bands import band_energy, bands_to_bins, N_BANDS
from .baseline import noise_floor, smooth_gain

CONTEXT = 3  # past frames only -> causal, plugin-friendly (adds no lookahead latency)


def features(X):
    mag = np.abs(X)
    e = band_energy(mag ** 2)
    nf = band_energy(noise_floor(mag) ** 2)
    log_e = np.log10(e + 1e-8)
    log_snr = log_e - np.log10(nf + 1e-8)          # the DSP stage's opinion, fed to the net
    f = np.concatenate([log_e - log_e.mean(), log_snr], axis=1)
    padded = np.vstack([np.repeat(f[:1], CONTEXT, 0), f])
    return np.hstack([padded[CONTEXT - k:len(padded) - k] for k in range(CONTEXT + 1)])


def target_mask(clean, noisy_minus_clean):
    S = band_energy(np.abs(stft(clean)) ** 2)
    N = band_energy(np.abs(stft(noisy_minus_clean)) ** 2)
    return np.sqrt(S / (S + N + 1e-10))             # ideal ratio mask


class HybridCleaner:
    def __init__(self, hidden=(128, 128), seed=0):
        self.net = MLPRegressor(hidden_layer_sizes=hidden, max_iter=60, early_stopping=True,
                                random_state=seed, learning_rate_init=2e-3)

    def fit(self, pairs):
        """pairs: list of (noisy, clean) arrays."""
        Xs = [features(stft(n)) for n, c in pairs]
        Ys = [target_mask(c, n - c) for n, c in pairs]
        self.net.fit(np.vstack(Xs), np.vstack(Ys))
        return self

    def clean(self, x, floor_db=-30.0):
        X = stft(x)
        g = np.clip(self.net.predict(features(X)), 0, 1)
        g = bands_to_bins(g)
        g = np.maximum(smooth_gain(g, attack=0.3, release=0.7), 10 ** (floor_db / 20))
        return istft(X * g, len(x))
