/* Hybrid Dialogue Cleaner: browser port of the Python prototype.
   Same maths as dialogue_cleaner/{stft,bands,baseline,hybrid}.py. */
const DC = (() => {
  const SR = 16000, N = 512, HOP = 128, BINS = N / 2 + 1, BANDS = 40, CONTEXT = 3;
  const win = new Float64Array(N);
  for (let i = 0; i < N; i++) win[i] = Math.sqrt(0.5 - 0.5 * Math.cos(2 * Math.PI * i / N));

  // in-place radix-2 complex FFT
  const rev = new Uint16Array(N), cosT = new Float64Array(N / 2), sinT = new Float64Array(N / 2);
  for (let i = 0, bits = Math.log2(N); i < N; i++) { let r = 0; for (let b = 0; b < bits; b++) r |= ((i >> b) & 1) << (bits - 1 - b); rev[i] = r; }
  for (let i = 0; i < N / 2; i++) { cosT[i] = Math.cos(2 * Math.PI * i / N); sinT[i] = Math.sin(2 * Math.PI * i / N); }
  function fft(re, im, inverse) {
    for (let i = 0; i < N; i++) { const j = rev[i]; if (j > i) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; } }
    for (let size = 2; size <= N; size <<= 1) {
      const half = size >> 1, step = N / size;
      for (let s = 0; s < N; s += size) for (let k = 0; k < half; k++) {
        const c = cosT[k * step], sn = inverse ? sinT[k * step] : -sinT[k * step];
        const a = s + k, b = a + half, tr = re[b] * c - im[b] * sn, ti = re[b] * sn + im[b] * c;
        re[b] = re[a] - tr; im[b] = im[a] - ti; re[a] += tr; im[a] += ti;
      }
    }
  }

  function fftAny(re, im) {                     // forward FFT, any power-of-two length
    const n = re.length;
    for (let i = 1, j = 0; i < n; i++) { let bit = n >> 1; for (; j & bit; bit >>= 1) j ^= bit; j ^= bit; if (i < j) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; } }
    for (let size = 2; size <= n; size <<= 1) {
      const half = size >> 1, ang = -2 * Math.PI / size;
      for (let s = 0; s < n; s += size) for (let k = 0; k < half; k++) {
        const c = Math.cos(ang * k), sn = Math.sin(ang * k), a = s + k, b = a + half;
        const tr = re[b] * c - im[b] * sn, ti = re[b] * sn + im[b] * c;
        re[b] = re[a] - tr; im[b] = im[a] - ti; re[a] += tr; im[a] += ti;
      }
    }
  }

  function stft(x) {
    const padded = new Float64Array(x.length + 2 * N); padded.set(x, N);
    const T = 1 + Math.floor((padded.length - N) / HOP);
    const re = new Float32Array(T * BINS), im = new Float32Array(T * BINS), mag = new Float32Array(T * BINS);
    const fr = new Float64Array(N), fi = new Float64Array(N);
    for (let t = 0; t < T; t++) {
      for (let i = 0; i < N; i++) { fr[i] = padded[t * HOP + i] * win[i]; fi[i] = 0; }
      fft(fr, fi, false);
      for (let k = 0; k < BINS; k++) { const o = t * BINS + k; re[o] = fr[k]; im[o] = fi[k]; mag[o] = Math.hypot(fr[k], fi[k]); }
    }
    return { re, im, mag, T };
  }

  function istft(S, gain, length) {
    const out = new Float64Array(N + HOP * (S.T - 1)), norm = new Float64Array(out.length);
    const fr = new Float64Array(N), fi = new Float64Array(N);
    for (let t = 0; t < S.T; t++) {
      for (let k = 0; k < BINS; k++) { const o = t * BINS + k, g = gain[o]; fr[k] = S.re[o] * g; fi[k] = S.im[o] * g; }
      for (let k = 1; k < N / 2; k++) { fr[N - k] = fr[k]; fi[N - k] = -fi[k]; }
      fft(fr, fi, true);
      for (let i = 0; i < N; i++) { out[t * HOP + i] += fr[i] / N * win[i]; norm[t * HOP + i] += win[i] * win[i]; }
    }
    const y = new Float32Array(length);
    for (let i = 0; i < length; i++) y[i] = out[N + i] / Math.max(norm[N + i], 1e-8);
    return y;
  }

  // running 10th percentile over 100 frames per bin (scipy percentile_filter, mode="nearest"), x1.5
  function noiseFloor(mag, T) {
    const nf = new Float32Array(T * BINS), buf = new Float32Array(100);
    for (let k = 0; k < BINS; k++) for (let t = 0; t < T; t++) {
      for (let j = 0; j < 100; j++) { const tt = Math.min(T - 1, Math.max(0, t - 50 + j)); buf[j] = mag[tt * BINS + k]; }
      nf[t * BINS + k] = Float32Array.from(buf).sort()[10] * 1.5;
    }
    return nf;
  }

  function smoothGain(g, T, attack, release, floor) {
    const prev = Float32Array.from(g.subarray(0, BINS)), row = new Float32Array(BINS), out = new Float32Array(T * BINS);
    for (let t = 0; t < T; t++) {
      for (let k = 0; k < BINS; k++) { const v = g[t * BINS + k], c = v > prev[k] ? attack : release; prev[k] = c * prev[k] + (1 - c) * v; row[k] = prev[k]; }
      for (let k = 0; k < BINS; k++) {
        const a = row[Math.max(0, k - 1)], b = row[Math.min(BINS - 1, k + 1)];
        out[t * BINS + k] = Math.max((a + row[k] + b) / 3, floor);
      }
    }
    return out;
  }

  function gateGain(S, nf, floorDb) {
    const g = new Float32Array(S.T * BINS);
    for (let i = 0; i < g.length; i++) {
      const snr = S.mag[i] * S.mag[i] / (2 * nf[i] * nf[i] + 1e-12);
      g[i] = Math.min(1, Math.max(0, 1 - 1 / Math.max(snr, 1e-8)));
    }
    return smoothGain(g, S.T, 0.6, 0.92, Math.pow(10, floorDb / 20));
  }

  function hybridGain(S, nf, model, floorDb) {
    const T = S.T, F = 2 * BANDS, f = new Float32Array(T * F);
    let mean = 0;
    for (let t = 0; t < T; t++) for (let b = 0; b < BANDS; b++) {
      let e = 0, n = 0; const fb = model.fb, o = b * BINS;
      for (let k = 0; k < BINS; k++) { const w = fb[o + k]; if (w) { const m = S.mag[t * BINS + k], q = nf[t * BINS + k]; e += w * m * m; n += w * q * q; } }
      const le = Math.log10(e + 1e-8); f[t * F + b] = le; f[t * F + BANDS + b] = le - Math.log10(n + 1e-8); mean += le;
    }
    mean /= T * BANDS;
    for (let t = 0; t < T; t++) for (let b = 0; b < BANDS; b++) f[t * F + b] -= mean;

    const g = new Float32Array(T * BINS);
    let a = new Float32Array(F * (CONTEXT + 1));
    for (let t = 0; t < T; t++) {
      for (let c = 0; c <= CONTEXT; c++) a.set(f.subarray(Math.max(0, t - c) * F, Math.max(0, t - c) * F + F), c * F);
      let h = a;
      model.layers.forEach((L, li) => {
        const y = new Float32Array(L.nout);
        for (let o = 0; o < L.nout; o++) { let s = L.b[o]; const off = o * L.nin; for (let i = 0; i < L.nin; i++) s += L.w[off + i] * h[i]; y[o] = li < model.layers.length - 1 ? Math.max(0, s) : Math.min(1, Math.max(0, s)); }
        h = y;
      });
      for (let b = 0; b < BANDS; b++) { const v = h[b]; if (!v) continue; const o = b * BINS; for (let k = 0; k < BINS; k++) g[t * BINS + k] += v * model.fbNorm[o + k]; }
    }
    return smoothGain(g, T, 0.3, 0.7, Math.pow(10, floorDb / 20));
  }

  // analyse() does the heavy work once; render() applies the gains at a chosen amount (0..1).
  function analyse(x, model, S, gateFloorDb = -18, hybridFloorDb = -30) {
    S = S || stft(x); const nf = noiseFloor(S.mag, S.T);
    return { S, length: x.length, gGate: gateGain(S, nf, gateFloorDb), gHybrid: hybridGain(S, nf, model, hybridFloorDb) };
  }
  function render(an, amount = 1) {
    const blend = g => { if (amount >= 1) return g; const o = new Float32Array(g.length); for (let i = 0; i < g.length; i++) o[i] = 1 - amount * (1 - g[i]); return o; };
    return { gate: istft(an.S, blend(an.gGate), an.length), hybrid: istft(an.S, blend(an.gHybrid), an.length) };
  }
  function process(x, model) { return render(analyse(x, model), 1); }

  // ---- noise-type identification (port of dialogue_cleaner/noise_id.py) ----
  function welchLow(x) {
    const n = 2048, hop = 1024, F = x.length < n ? 1 : 1 + Math.floor((x.length - n) / hop);
    const cols = Array.from({ length: 65 }, () => new Float32Array(F)), re = new Float64Array(n), im = new Float64Array(n);
    for (let f = 0; f < F; f++) {
      for (let i = 0; i < n; i++) { re[i] = (x[f * hop + i] || 0) * (0.5 - 0.5 * Math.cos(2 * Math.PI * i / n)); im[i] = 0; }
      fftAny(re, im);
      for (let k = 0; k < 65; k++) cols[k][f] = Math.hypot(re[k], im[k]);
    }
    return cols.map(c => c.sort()[Math.floor(0.2 * F)]);
  }
  function noiseFeatures(x, model, S) {
    S = S || stft(x); const T = S.T, mag = S.mag, nf2 = new Float64Array(BINS), mean2 = new Float64Array(BINS), col = new Float32Array(T);
    for (let k = 0; k < BINS; k++) {
      let m = 0; for (let t = 0; t < T; t++) { const v = mag[t * BINS + k]; col[t] = v; m += v * v; }
      mean2[k] = m / T; const q = Float32Array.from(col).sort()[Math.floor(0.1 * T)]; nf2[k] = q * q;
    }
    const f = new Float64Array(84); let ref = 0;
    for (let b = 0; b < BANDS; b++) {
      let a = 0, m = 0; for (let k = 0; k < BINS; k++) { const w = model.fb[b * BINS + k]; a += w * nf2[k]; m += w * mean2[k]; }
      f[b] = Math.log10(a + 1e-10); f[BANDS + b] = Math.log10(m + 1e-10); ref += f[BANDS + b];
    }
    ref /= BANDS; for (let i = 0; i < 2 * BANDS; i++) f[i] -= ref;
    const nb = Math.floor((T - 8) / 25), mins = new Float64Array(nb);
    for (let b = 0; b < nb; b++) {
      let mn = Infinity;
      for (let t = 4 + b * 25; t < 4 + (b + 1) * 25; t++) { let e = 0; for (let k = 1; k < 10; k++) e += mag[t * BINS + k] ** 2; mn = Math.min(mn, Math.log10(e + 1e-10)); }
      mins[b] = mn;
    }
    const mm = mins.reduce((a, v) => a + v, 0) / nb, srt = Float64Array.from(mins).sort();
    f[80] = Math.sqrt(mins.reduce((a, v) => a + (v - mm) ** 2, 0) / nb);
    f[81] = srt[Math.floor(0.9 * (nb - 1))] - srt[Math.floor(0.1 * (nb - 1))];
    const P = welchLow(x), band = Float32Array.from(P.slice(3, 65)).sort(), med = (band[30] + band[31]) / 2 + 1e-10;
    f[82] = Math.log10(band[61] / med + 1e-10);
    f[83] = Math.max(...[50, 60].map(f0 => [1, 2, 3, 4].reduce((a, k) => a + Math.log10(P[Math.round(k * f0 / 7.8125)] / med + 1e-10), 0) / 4));
    return f;
  }
  function identifyNoise(x, model, S) {
    const nid = model.noiseId, f = noiseFeatures(x, model, S);
    let h = Float32Array.from(f, (v, i) => (v - nid.mu[i]) / nid.sd[i]);
    nid.layers.forEach((L, li) => {
      const y = new Float32Array(L.nout);
      for (let o = 0; o < L.nout; o++) { let s = L.b[o]; for (let i = 0; i < L.nin; i++) s += L.w[o * L.nin + i] * h[i]; y[o] = li < nid.layers.length - 1 ? Math.max(0, s) : s; }
      h = y;
    });
    const mx = Math.max(...h), e = Array.from(h, v => Math.exp(v - mx)), sum = e.reduce((a, v) => a + v, 0);
    return { features: f, probs: e.map(v => v / sum), ranked: e.map((v, i) => ({ label: nid.labels[i], index: i, p: v / sum })).sort((a, b) => b.p - a.p) };
  }

  function siSdr(est, ref) {
    let me = 0, mr = 0; const n = ref.length;
    for (let i = 0; i < n; i++) { me += est[i]; mr += ref[i]; } me /= n; mr /= n;
    let er = 0, rr = 0; for (let i = 0; i < n; i++) { er += (est[i] - me) * (ref[i] - mr); rr += (ref[i] - mr) ** 2; }
    const a = er / (rr + 1e-12); let ss = 0, nn = 0;
    for (let i = 0; i < n; i++) { const s = a * (ref[i] - mr), d = est[i] - me - s; ss += s * s; nn += d * d; }
    return 10 * Math.log10((ss + 1e-12) / (nn + 1e-12));
  }

  const f32 = b64 => { const s = atob(b64), u = new Uint8Array(s.length); for (let i = 0; i < s.length; i++) u[i] = s.charCodeAt(i); return new Float32Array(u.buffer); };
  const pcm = b64 => { const s = atob(b64), u = new Uint8Array(s.length); for (let i = 0; i < s.length; i++) u[i] = s.charCodeAt(i); const p = new Int16Array(u.buffer), y = new Float32Array(p.length); for (let i = 0; i < p.length; i++) y[i] = p[i] / 32767; return y; };
  function loadModel(j) {
    const lay = Ls => Ls.map(L => ({ w: f32(L.w), b: f32(L.b), nin: L.nin, nout: L.nout }));
    return { fb: f32(j.fb), fbNorm: f32(j.fbNorm), layers: lay(j.layers),
             noiseId: j.noiseId && { labels: j.noiseId.labels, mu: f32(j.noiseId.mu), sd: f32(j.noiseId.sd), layers: lay(j.noiseId.layers) },
             examples: j.examples.map(e => ({ name: e.name, noisy: pcm(e.noisy), clean: pcm(e.clean) })) };
  }
  return { SR, N, HOP, BINS, stft, analyse, render, process, identifyNoise, siSdr, loadModel };
})();
if (typeof module !== "undefined") module.exports = DC;
