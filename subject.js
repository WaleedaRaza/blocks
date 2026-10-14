/* Subject finder: a small background-removal model run on the viewer's own device.
 * The photo never leaves the browser. First use downloads the model (~44 MB) once;
 * the browser caches it after that.
 *
 * Model: IS-Net based cut-out from @imgly/background-removal-data 1.4.5 (AGPL-3.0,
 * see models/LICENSE-subject-model.md). Runtime: onnxruntime-web 1.17.3 from jsDelivr.
 */
(function (root) {
  'use strict';
  const ORT = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.3/dist/';
  const MODEL = 'models/subject.onnx';
  const SIZE = 1024;
  let sessionP = null;

  function loadScript(src) {
    return new Promise((res, rej) => {
      const s = document.createElement('script'); s.src = src; s.onload = res; s.onerror = () => rej(new Error('Could not load ' + src));
      document.head.appendChild(s);
    });
  }
  function session() {
    if (!sessionP) sessionP = (async () => {
      if (!root.ort) await loadScript(ORT + 'ort.min.js');
      root.ort.env.wasm.wasmPaths = ORT;
      root.ort.env.wasm.numThreads = root.crossOriginIsolated ? 0 : 1;
      return root.ort.InferenceSession.create(MODEL, { executionProviders: ['wasm'] });
    })().catch(e => { sessionP = null; throw e; });
    return sessionP;
  }

  // Returns a Uint8Array (w*h, 0..255) where 255 = subject, for any source canvas/image.
  async function find(src, w, h) {
    const s = await session();
    const c = document.createElement('canvas'); c.width = c.height = SIZE;
    const g = c.getContext('2d', { willReadFrequently: true });
    g.drawImage(src, 0, 0, SIZE, SIZE);
    const d = g.getImageData(0, 0, SIZE, SIZE).data, n = SIZE * SIZE;
    const x = new Float32Array(3 * n);
    for (let i = 0; i < n; i++) { x[i] = (d[i * 4] - 128) / 256; x[n + i] = (d[i * 4 + 1] - 128) / 256; x[2 * n + i] = (d[i * 4 + 2] - 128) / 256; }
    const out = await s.run({ input: new root.ort.Tensor('float32', x, [1, 3, SIZE, SIZE]) });
    const y = out.output.data;
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < n; i++) { if (y[i] < lo) lo = y[i]; if (y[i] > hi) hi = y[i]; }
    const img = g.createImageData(SIZE, SIZE), k = 255 / (hi - lo || 1);
    for (let i = 0; i < n; i++) { const v = (y[i] - lo) * k; img.data[i * 4] = img.data[i * 4 + 1] = img.data[i * 4 + 2] = v; img.data[i * 4 + 3] = 255; }
    g.putImageData(img, 0, 0);
    const r = document.createElement('canvas'); r.width = w; r.height = h;
    const rg = r.getContext('2d', { willReadFrequently: true }); rg.drawImage(c, 0, 0, w, h);
    const rd = rg.getImageData(0, 0, w, h).data, m = new Uint8Array(w * h);
    for (let i = 0; i < w * h; i++) m[i] = rd[i * 4];
    return m;
  }

  root.BlocksSubject = { find, warm: () => session().then(() => true) };
})(window);
