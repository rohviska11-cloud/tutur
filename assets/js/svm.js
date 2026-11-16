/* svm.js — Pipeline(SimpleImputer median -> StandardScaler -> SVC RBF / OneVsRest(SVC RBF)) hasil ekspor Colab.
 * decision_k(x) = sum_i coef_i * exp(-gamma * ||sv_i - z||^2) + b
 * Biner : kelas = classes[1] bila decision > 0, probabilitas = sigmoid Platt sklearn.
 * OvR   : kelas = argmax decision_k, probabilitas = Platt_k dinormalisasi (sama dengan sklearn).
 */
(function (root) {
  "use strict";

  function f16ToF32(u16) {
    const out = new Float32Array(u16.length);
    for (let i = 0; i < u16.length; i++) {
      const h = u16[i], s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
      out[i] = e === 0 ? s * Math.pow(2, -14) * (m / 1024) : e === 31 ? (m ? NaN : s * Infinity) : s * Math.pow(2, e - 15) * (1 + m / 1024);
    }
    return out;
  }

  async function load(baseUrl, jsonName) {
    const meta = await (await fetch(`${baseUrl}/${jsonName}`, { cache: "no-cache" })).json();
    const buf = await (await fetch(`${baseUrl}/${meta.bin}`, { cache: "no-cache" })).arrayBuffer();
    const rows = meta.dtype === "float16" ? f16ToF32(new Uint16Array(buf)) : new Float32Array(buf);
    return { meta, rows };
  }

  const platt = (p, d) => {
    let q = 1 / (1 + Math.exp(p.A * (p.s * d) + p.B));
    return p.flip ? 1 - q : q;
  };

  function predict(model, emb) {
    const m = model.meta, F = m.n_feat, R = model.rows;
    const z = new Float64Array(F);
    for (let j = 0; j < F; j++) {
      const v = Number.isFinite(emb[j]) ? emb[j] : m.imputer_median[j];
      z[j] = (v - m.scaler_mean[j]) / m.scaler_scale[j];
    }
    // jarak ke setiap baris support vector (dipakai bersama oleh semua estimator OvR)
    const dist = new Float64Array(m.n_rows);
    for (let r = 0; r < m.n_rows; r++) {
      let s = 0;
      const o = r * F;
      for (let j = 0; j < F; j++) { const d = R[o + j] - z[j]; s += d * d; }
      dist[r] = s;
    }
    const dec = m.estimators.map((e) => {
      let acc = e.b;
      for (let i = 0; i < e.idx.length; i++) acc += e.coef[i] * Math.exp(-e.gamma * dist[e.idx[i]]);
      return acc;
    });
    let proba, label;
    if (!m.ovr) {
      const p1 = platt(m.estimators[0].platt, dec[0]);
      proba = [1 - p1, p1];
      label = dec[0] > 0 ? m.classes[1] : m.classes[0];
    } else {
      const p = dec.map((d, k) => platt(m.estimators[k].platt, d));
      const s = p.reduce((a, b) => a + b, 0) || 1;
      proba = p.map((v) => v / s);
      let best = 0;
      for (let k = 1; k < dec.length; k++) if (dec[k] > dec[best]) best = k;
      label = m.classes[best];
    }
    return { proba, label, classes: m.classes, decision: dec };
  }

  const SVM = { load, predict };
  if (typeof module !== "undefined" && module.exports) module.exports = SVM;
  else root.SVM = SVM;
})(typeof self !== "undefined" ? self : this);
