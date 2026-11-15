"""Train the noise-type classifier and add it to web_model.json for the browser demo."""
import json, base64, warnings
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from dialogue_cleaner.noise_id import features, make_clip, LABELS
warnings.filterwarnings("ignore")

PER_CLASS = 220
rng = np.random.default_rng(1)
X, y = [], []
for label in range(len(LABELS)):
    for _ in range(PER_CLASS):
        X.append(features(make_clip(label, rng))); y.append(label)
X, y = np.array(X), np.array(y)
idx = np.random.default_rng(2).permutation(len(y)); cut = int(0.8 * len(y))
tr, te = idx[:cut], idx[cut:]
mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
clf = MLPClassifier(hidden_layer_sizes=(64,), max_iter=400, random_state=0).fit((X[tr] - mu) / sd, y[tr])
pred = clf.predict((X[te] - mu) / sd)
print(f"held-out accuracy on synthetic clips: {np.mean(pred == y[te]):.3f}  ({len(te)} clips)")
print(confusion_matrix(y[te], pred))

b64 = lambda a: base64.b64encode(np.ascontiguousarray(a, dtype="<f4").tobytes()).decode()
m = json.load(open("web_model.json"))
m["noiseId"] = {"labels": LABELS, "mu": b64(mu), "sd": b64(sd),
                "layers": [{"w": b64(w.T), "b": b64(b), "nin": w.shape[0], "nout": w.shape[1]}
                           for w, b in zip(clf.coefs_, clf.intercepts_)]}
json.dump(m, open("web_model.json", "w"))

# reference values for checking the JavaScript port
pcm = lambda b: np.frombuffer(base64.b64decode(b), dtype="<i2") / 32767.0
ref = []
for ex in m["examples"]:
    f = features(pcm(ex["noisy"])); p = clf.predict_proba(((f - mu) / sd)[None])[0]
    ref.append({"features": f.tolist(), "probs": p.tolist()})
    print(f'{ex["name"]:16s} -> {LABELS[int(p.argmax())]} ({p.max():.2f})')
json.dump(ref, open("/tmp/noise_ref.json", "w"))
