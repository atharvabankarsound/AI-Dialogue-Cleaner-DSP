const DC = require("./dsp.js"), fs = require("fs");
const m = DC.loadModel(JSON.parse(fs.readFileSync(process.argv[2]))), ref = JSON.parse(fs.readFileSync(process.argv[3]));
const t0 = Date.now(), r = DC.process(m.examples[0].noisy, m), ms = Date.now() - t0;
for (const k of ["gate", "hybrid"]) { let e = 0; ref[k].forEach((v, i) => e = Math.max(e, Math.abs(v - r[k][i]))); console.log(k, "max abs diff vs Python:", e.toExponential(2)); }
const ex = m.examples[0];
console.log("SI-SDR noisy/gate/hybrid:", [ex.noisy, r.gate, r.hybrid].map(y => DC.siSdr(y, ex.clean).toFixed(1)).join(" / "), "| time", ms, "ms for", ex.noisy.length / 16000, "s");
