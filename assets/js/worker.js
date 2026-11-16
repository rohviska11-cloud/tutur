/* worker.js — unduh model (sekali, lalu disimpan di cache browser), hitung embedding WavLM/HuBERT
 * dengan onnxruntime-web, klasifikasi dengan SVM, plus ciri akustik ringkas untuk ditampilkan. */
const ORT_VER = "1.20.1";
const ORT_CDN = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VER}/dist/`;
importScripts(ORT_CDN + "ort.wasm.min.js", "dsp.js", "praat.js", "features.js", "svm.js");
ort.env.wasm.wasmPaths = ORT_CDN;
ort.env.wasm.numThreads = 1;

let manifest = null, base = null;
const sessions = {}, svms = {};
const TASKS = ["speech_status", "severity"];

async function getManifest(b) {
  if (manifest) return manifest;
  base = b;
  manifest = await (await fetch(`${base}/manifest.json`, { cache: "no-cache" })).json();
  return manifest;
}

async function cachedBytes(url, cache, onChunk) {
  if (cache) {
    const hit = await cache.match(url);
    if (hit) { const ab = await hit.arrayBuffer(); onChunk(ab.byteLength); return new Uint8Array(ab); }
  }
  const res = await fetch(url);
  if (!res.ok) throw new Error(`gagal mengunduh ${url.split("/").pop()} (${res.status})`);
  const reader = res.body.getReader(), parts = [];
  let n = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value); n += value.length; onChunk(value.length);
  }
  const out = new Uint8Array(n);
  let o = 0;
  for (const p of parts) { out.set(p, o); o += p.length; }
  if (cache) { try { await cache.put(url, new Response(out.slice())); } catch (e) { /* kuota penuh: lewati */ } }
  return out;
}

async function ensureModels(id) {
  const m = await getManifest(base);
  const need = TASKS.filter((t) => !sessions[t]);
  if (!need.length) return;
  let cache = null;
  try { cache = await caches.open(`tutur-model-${m.versi}`); } catch (e) { /* tanpa Cache API */ }
  const total = need.reduce((a, t) => a + m.tasks[t].onnx_bytes, 0);
  let loaded = 0, lastPost = 0;
  const tick = (k) => {
    loaded += k;
    const now = Date.now();
    if (now - lastPost > 120 || loaded >= total) { lastPost = now; self.postMessage({ id, type: "progress", stage: "unduh", loaded, total }); }
  };
  for (const t of need) {
    const info = m.tasks[t];
    const chunks = [];
    for (const p of info.onnx_parts) chunks.push(await cachedBytes(`${base}/${p}`, cache, tick));
    const bytes = new Uint8Array(info.onnx_bytes);
    let o = 0;
    for (const c of chunks) { bytes.set(c, o); o += c.length; }
    self.postMessage({ id, type: "progress", stage: "siapkan", task: t });
    sessions[t] = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
  }
}

async function embed(task, wave) {
  const s = sessions[task];
  const out = await s.run({ [s.inputNames[0]]: new ort.Tensor("float32", wave, [1, wave.length]) });
  return out[s.outputNames[0]].data;
}

self.onmessage = async (ev) => {
  const { type, id } = ev.data;
  try {
    if (type === "init") {
      const m = await getManifest(ev.data.base);
      for (const t of TASKS) if (!svms[t]) svms[t] = await SVM.load(base, m.tasks[t].svm);
      let cached = false;
      try {
        const c = await caches.open(`tutur-model-${m.versi}`);
        cached = (await Promise.all(TASKS.flatMap((t) => m.tasks[t].onnx_parts.map((p) => c.match(`${base}/${p}`))))).every(Boolean);
      } catch (e) { /* abaikan */ }
      const mb = TASKS.reduce((a, t) => a + m.tasks[t].onnx_bytes, 0) / 1e6;
      self.postMessage({ id, type: "ready", manifest: m, cached, mb });
    } else if (type === "analyze") {
      await getManifest(ev.data.base);
      await ensureModels(id);
      self.postMessage({ id, type: "progress", stage: "hitung" });
      const t0 = performance.now();
      // preprocessing identik dengan notebook: mono, DC offset, puncak 0,95, lalu pad sampai >= 0,5 s
      let y = DSP.preprocess(ev.data.samples, 0.95);
      if (y.length < 8000) { const p = new Float32Array(8000); p.set(y); y = p; }
      const res = {};
      for (const t of TASKS) {
        const e = await embed(t, y);
        res[t] = SVM.predict(svms[t], e);
      }
      const { feats, duration } = Features.extract(ev.data.samples, ev.data.sr);
      self.postMessage({ id, type: "result", status: res.speech_status, severity: res.severity, feats, duration, ms: performance.now() - t0 });
    }
  } catch (err) {
    self.postMessage({ id, type: "error", message: String((err && err.message) || err) });
  }
};
