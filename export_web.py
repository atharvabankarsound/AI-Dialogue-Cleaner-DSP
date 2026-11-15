"""Train the hybrid model and export weights + example clips for the web demo."""
import json, base64, warnings
import numpy as np
from dialogue_cleaner.synth import speechlike, noise, mix
from dialogue_cleaner.hybrid import HybridCleaner
from dialogue_cleaner.baseline import spectral_gate
from dialogue_cleaner.bands import FB, FB_NORM
warnings.filterwarnings("ignore")

kinds = ["white", "pink", "hum", "aircon"]
rng = np.random.default_rng(0)
def make(rng, kind, snr, dur=4.0):
    c = speechlike(dur, rng); return mix(c, noise(kind, len(c), rng), snr), c
model = HybridCleaner().fit([make(rng, rng.choice(kinds), rng.uniform(-5, 20)) for _ in range(150)])

b64 = lambda a: base64.b64encode(np.ascontiguousarray(a, dtype="<f4").tobytes()).decode()
pcm = lambda a: base64.b64encode((np.clip(a, -1, 1) * 32767).astype("<i2").tobytes()).decode()
out = {"layers": [{"w": b64(w.T), "b": b64(b), "nin": w.shape[0], "nout": w.shape[1]}
                  for w, b in zip(model.net.coefs_, model.net.intercepts_)],
       "fb": b64(FB), "fbNorm": b64(FB_NORM), "examples": []}
test = {}
for name, kind, snr, seed in [("Air-con rumble", "aircon", 3, 7), ("Mains hum", "hum", 2, 11), ("Broadband hiss", "white", 5, 23)]:
    n, c = make(np.random.default_rng(seed), kind, snr, dur=5.0)
    out["examples"].append({"name": name, "noisy": pcm(n), "clean": pcm(c)})
    if not test:
        q = (np.clip(n, -1, 1) * 32767).astype("<i2") / 32767.0   # what the browser will decode
        test = {"gate": spectral_gate(q)[:4000].tolist(), "hybrid": model.clean(q)[:4000].tolist()}
json.dump(out, open("web_model.json", "w"))
json.dump(test, open("web_test_ref.json", "w"))
print("ok", [ (l["nin"], l["nout"]) for l in out["layers"]])
