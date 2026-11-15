"""PyTorch version of the mask predictor, for training on real dialogue.

NOT YET RUN: PyTorch was not available where this prototype was built.
Treat it as a starting point and expect to debug it on first run.

Usage:
    pip install torch soundfile
    python train_torch.py --clean path/to/clean_wavs --noise path/to/noise_wavs

Same inputs/outputs as dialogue_cleaner/hybrid.py, but a GRU replaces the
fixed context window, so the model carries its own memory frame to frame
(which is what the real-time plugin will do).
"""
import argparse
import glob
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from scipy.signal import resample_poly

from dialogue_cleaner.stft import SR, stft
from dialogue_cleaner.bands import band_energy, N_BANDS
from dialogue_cleaner.baseline import noise_floor
from dialogue_cleaner.hybrid import target_mask
from dialogue_cleaner.synth import mix


class MaskGRU(nn.Module):
    def __init__(self, n_in=2 * N_BANDS, hidden=96):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(n_in, hidden), nn.Tanh())
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.out = nn.Sequential(nn.Linear(hidden, N_BANDS), nn.Sigmoid())

    def forward(self, x, h=None):               # x: (batch, frames, 2*bands)
        y, h = self.gru(self.inp(x), h)
        return self.out(y), h


def frame_features(noisy):
    mag = np.abs(stft(noisy))
    e = band_energy(mag ** 2)
    nf = band_energy(noise_floor(mag) ** 2)
    log_e = np.log10(e + 1e-8)
    return np.concatenate([log_e - log_e.mean(), log_e - np.log10(nf + 1e-8)], axis=1)


def load(path, seconds=4.0):
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != SR:
        x = resample_poly(x, SR, sr)
    n = int(seconds * SR)
    if len(x) < n:
        x = np.tile(x, n // len(x) + 1)
    start = np.random.randint(0, len(x) - n + 1)
    return x[start:start + n]


def batch(clean_files, noise_files, size=16):
    X, Y = [], []
    for _ in range(size):
        c = load(np.random.choice(clean_files))
        nz = load(np.random.choice(noise_files))
        nz = nz / (np.std(nz) + 1e-9)
        noisy = mix(c, nz, np.random.uniform(-5, 25))
        X.append(frame_features(noisy))
        Y.append(target_mask(c, noisy - c))
    return torch.tensor(np.stack(X), dtype=torch.float32), torch.tensor(np.stack(Y), dtype=torch.float32)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", required=True)
    ap.add_argument("--noise", required=True)
    ap.add_argument("--steps", type=int, default=5000)
    args = ap.parse_args()
    clean_files = glob.glob(f"{args.clean}/**/*.wav", recursive=True)
    noise_files = glob.glob(f"{args.noise}/**/*.wav", recursive=True)

    model = MaskGRU()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for step in range(args.steps):
        x, y = batch(clean_files, noise_files)
        pred, _ = model(x)
        loss = nn.functional.mse_loss(pred, y)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 50 == 0:
            print(f"step {step:5d}  loss {loss.item():.4f}")
    torch.save(model.state_dict(), "mask_gru.pt")
    # Next: torch.onnx.export(...) so the JUCE plugin can run it with ONNX Runtime or RTNeural.
