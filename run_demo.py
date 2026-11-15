"""Train the hybrid model on synthetic data and compare with the DSP baseline."""
import numpy as np
from scipy.io import wavfile
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dialogue_cleaner.stft import SR, stft
from dialogue_cleaner.synth import speechlike, noise, mix
from dialogue_cleaner.baseline import spectral_gate
from dialogue_cleaner.hybrid import HybridCleaner
from dialogue_cleaner.metrics import si_sdr, log_spectral_distance

TRAIN_NOISES = ["white", "pink", "hum", "aircon"]
TEST_NOISES = TRAIN_NOISES + ["babble"]            # babble is never seen in training


def make(rng, kind, snr, dur=4.0):
    c = speechlike(dur, rng)
    return mix(c, noise(kind, len(c), rng), snr), c


rng = np.random.default_rng(0)
train = [make(rng, rng.choice(TRAIN_NOISES), rng.uniform(-5, 20)) for _ in range(150)]
model = HybridCleaner().fit(train)

rng = np.random.default_rng(999)                   # held-out seeds
print(f"{'noise':8s} {'SNR':>4s} | SI-SDR dB: {'noisy':>6s} {'gate':>6s} {'hybrid':>6s} | LSD dB: {'gate':>5s} {'hybrid':>6s}")
rows = []
for kind in TEST_NOISES:
    for snr in (0, 10):
        r = []
        for _ in range(8):
            n, c = make(rng, kind, snr)
            g, h = spectral_gate(n), model.clean(n)
            r.append([si_sdr(n, c), si_sdr(g, c), si_sdr(h, c),
                      log_spectral_distance(g, c), log_spectral_distance(h, c)])
        m = np.mean(r, axis=0); rows.append(m)
        tag = kind + ("*" if kind not in TRAIN_NOISES else "")
        print(f"{tag:8s} {snr:4d} | {'':10s} {m[0]:6.1f} {m[1]:6.1f} {m[2]:6.1f} | {'':7s} {m[3]:5.1f} {m[4]:6.1f}")
m = np.mean(rows, axis=0)
print(f"{'MEAN':8s} {'':4s} | {'':10s} {m[0]:6.1f} {m[1]:6.1f} {m[2]:6.1f} | {'':7s} {m[3]:5.1f} {m[4]:6.1f}")
print("* = noise type not seen during training")

# one listening example + spectrograms
n, c = make(np.random.default_rng(7), "aircon", 3, dur=6.0)
outs = {"1_noisy": n, "2_spectral_gate": spectral_gate(n), "3_hybrid": model.clean(n), "4_clean_reference": c}
fig, ax = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
for a, (name, y) in zip(ax, outs.items()):
    wavfile.write(f"out/{name}.wav", SR, (np.clip(y, -1, 1) * 32767).astype(np.int16))
    a.imshow(20 * np.log10(np.abs(stft(y)).T + 1e-4), origin="lower", aspect="auto",
             extent=[0, len(y) / SR, 0, SR / 2000], vmin=-70, vmax=20, cmap="magma")
    a.set_title(name[2:].replace("_", " ")); a.set_ylabel("kHz")
ax[-1].set_xlabel("seconds"); plt.tight_layout(); plt.savefig("out/spectrograms.png", dpi=110)
