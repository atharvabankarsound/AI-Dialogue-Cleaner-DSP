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

  function process(x, model, gateFloorDb = -18, hybridFloorDb = -30) {
    const S = stft(x), nf = noiseFloor(S.mag, S.T);
    return { gate: istft(S, gateGain(S, nf, gateFloorDb), x.length), hybrid: istft(S, hybridGain(S, nf, model, hybridFloorDb), x.length) };
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
    return { fb: f32(j.fb), fbNorm: f32(j.fbNorm), layers: j.layers.map(L => ({ w: f32(L.w), b: f32(L.b), nin: L.nin, nout: L.nout })),
             examples: j.examples.map(e => ({ name: e.name, noisy: pcm(e.noisy), clean: pcm(e.clean) })) };
  }
  return { SR, N, HOP, BINS, stft, process, siSdr, loadModel };
})();
if (typeof module !== "undefined") module.exports = DC;
