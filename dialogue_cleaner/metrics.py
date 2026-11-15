import numpy as np


def si_sdr(est, ref):
    """Scale-invariant signal-to-distortion ratio in dB (higher is better)."""
    ref = ref - ref.mean(); est = est - est.mean()
    s = (est @ ref) / (ref @ ref + 1e-12) * ref
    return 10 * np.log10((s @ s + 1e-12) / ((est - s) @ (est - s) + 1e-12))


def log_spectral_distance(est, ref):
    """Mean log-spectral distance in dB (lower is better)."""
    from .stft import stft
    a = 20 * np.log10(np.abs(stft(est)) + 1e-5)
    b = 20 * np.log10(np.abs(stft(ref)) + 1e-5)
    return float(np.mean(np.sqrt(np.mean((a - b) ** 2, axis=1))))
