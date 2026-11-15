// Checks the JavaScript port against reference values written by the Python code.
// usage: node web/test_port.js web_model.json [denoise_ref.json] [noise_ref.json]
const DC = require("./dsp.js"), fs = require("fs");
const m = DC.loadModel(JSON.parse(fs.readFileSync(process.argv[2])));
const ex = m.examples[0], r = DC.process(ex.noisy, m);
console.log("SI-SDR noisy/gate/hybrid:", [ex.noisy, r.gate, r.hybrid].map(y => DC.siSdr(y, ex.clean).toFixed(1)).join(" / "));
const half = DC.render(DC.analyse(ex.noisy, m), 0.5), zero = DC.render(DC.analyse(ex.noisy, m), 0);
console.log("amount 50% hybrid SI-SDR:", DC.siSdr(half.hybrid, ex.clean).toFixed(1), "| amount 0% max diff from input:", Math.max(...zero.hybrid.map((v, i) => Math.abs(v - ex.noisy[i]))).toExponential(1));
if (process.argv[3] && fs.existsSync(process.argv[3])) {
  const ref = JSON.parse(fs.readFileSync(process.argv[3]));
  for (const k of ["gate", "hybrid"]) { let e = 0; ref[k].forEach((v, i) => e = Math.max(e, Math.abs(v - r[k][i]))); console.log(k, "max abs diff vs Python:", e.toExponential(2)); }
}
if (process.argv[4]) {
  const ref = JSON.parse(fs.readFileSync(process.argv[4]));
  m.examples.forEach((e, i) => {
    const id = DC.identifyNoise(e.noisy, m);
    const df = Math.max(...ref[i].features.map((v, j) => Math.abs(v - id.features[j]))), dp = Math.max(...ref[i].probs.map((v, j) => Math.abs(v - id.probs[j])));
    console.log(e.name.padEnd(16), "->", id.ranked[0].label, id.ranked[0].p.toFixed(2), "| feature diff", df.toExponential(1), "prob diff", dp.toExponential(1));
  });
}
