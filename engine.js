/* Blocks engine v2 — photo -> flat tone blocks, the way a block-in is built.
 *
 * Pipeline (each step is named after the studio habit it stands in for):
 *   1. See value like the eye: sRGB -> CIE Lab, tones decided on L* (perceptual lightness).
 *   2. Squint: edge-preserving smoothing (guided filter) kills texture, keeps form edges.
 *   3. Small patches: graph-based over-segmentation (Felzenszwalb) that follows edges.
 *   4. Big shapes: one merge hierarchy over the patches using the piecewise-constant
 *      Mumford–Shah trade-off — merge the pair whose value difference is cheapest
 *      relative to the edge it removes. Cutting the hierarchy at different depths gives
 *      big shapes, medium shapes, fine shapes; cuts are nested by construction.
 *   5. Focus: a subject map (distinct-from-surroundings + centre bias). Detail is kept
 *      inside it; the background is merged harder.
 *   6. Tones per shape, not per pixel: cuts found on the area-weighted values of the
 *      shapes, split divisively (2 -> 3 -> 4 -> 5), so each step only splits a family
 *      from the step before. The engine picks which family to split.
 *   7. A shape must earn its place: islands below a size are absorbed by the neighbour
 *      they share the most edge with.
 *   8. Drawn as a shared map: every border between two shapes is simplified once and
 *      used by both sides, so there are no gaps or slivers.
 *
 * Works on plain arrays so it runs in the app and in the test harness alike.
 */
(function (root) {
  'use strict';

  /* ---------------- colour ---------------- */
  const LIN = new Float32Array(256);
  for (let i = 0; i < 256; i++) { const c = i / 255; LIN[i] = c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4); }
  const fLab = t => (t > 0.008856 ? Math.cbrt(t) : 7.787 * t + 16 / 116);

  function toLab(rgba, n) {
    const L = new Float32Array(n), A = new Float32Array(n), B = new Float32Array(n);
    for (let i = 0, j = 0; i < n; i++, j += 4) {
      const r = LIN[rgba[j]], g = LIN[rgba[j + 1]], b = LIN[rgba[j + 2]];
      const X = (0.4124 * r + 0.3576 * g + 0.1805 * b) / 0.95047;
      const Y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
      const Z = (0.0193 * r + 0.1192 * g + 0.9505 * b) / 1.08883;
      const fx = fLab(X), fy = fLab(Y), fz = fLab(Z);
      L[i] = 116 * fy - 16; A[i] = 500 * (fx - fy); B[i] = 200 * (fy - fz);
    }
    return { L, A, B };
  }

  /* ---------------- smoothing ---------------- */
  function boxMean(src, w, h, r) {
    const W = w + 1, I = new Float64Array(W * (h + 1)), out = new Float32Array(w * h);
    for (let y = 0; y < h; y++) { let s = 0; for (let x = 0; x < w; x++) { s += src[y * w + x]; I[(y + 1) * W + x + 1] = I[y * W + x + 1] + s; } }
    for (let y = 0; y < h; y++) {
      const y0 = Math.max(0, y - r), y1 = Math.min(h, y + r + 1);
      for (let x = 0; x < w; x++) {
        const x0 = Math.max(0, x - r), x1 = Math.min(w, x + r + 1);
        out[y * w + x] = (I[y1 * W + x1] - I[y0 * W + x1] - I[y1 * W + x0] + I[y0 * W + x0]) / ((y1 - y0) * (x1 - x0));
      }
    }
    return out;
  }
  // Self-guided filter (He et al.): flattens areas whose variance is below eps, keeps edges.
  function guided(I, w, h, r, eps) {
    const n = w * h, II = new Float32Array(n);
    for (let i = 0; i < n; i++) II[i] = I[i] * I[i];
    const m = boxMean(I, w, h, r), m2 = boxMean(II, w, h, r);
    const a = new Float32Array(n), b = new Float32Array(n);
    for (let i = 0; i < n; i++) { const v = m2[i] - m[i] * m[i]; a[i] = v / (v + eps); b[i] = m[i] - a[i] * m[i]; }
    const ma = boxMean(a, w, h, r), mb = boxMean(b, w, h, r), q = new Float32Array(n);
    for (let i = 0; i < n; i++) q[i] = ma[i] * I[i] + mb[i];
    return q;
  }

  /* ---------------- union-find ---------------- */
  function UF(n) { const p = new Int32Array(n); for (let i = 0; i < n; i++) p[i] = i; return p; }
  function find(p, x) { while (p[x] !== x) { p[x] = p[p[x]]; x = p[x]; } return x; }

  /* ---------------- over-segmentation (Felzenszwalb–Huttenlocher) ---------------- */
  function patches(L, A, B, w, h, k, minSize, side) {
    const n = w * h, m = (w - 1) * h + w * (h - 1);
    const ea = new Int32Array(m), eb = new Int32Array(m), ew = new Float32Array(m);
    let e = 0;
    const wt = (i, j) => { const dl = L[i] - L[j], da = A[i] - A[j], db = B[i] - B[j]; return Math.sqrt(dl * dl + 0.5 * (da * da + db * db)) + (side && side[i] !== side[j] ? 1e4 : 0); };
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const i = y * w + x;
      if (x < w - 1) { ea[e] = i; eb[e] = i + 1; ew[e] = wt(i, i + 1); e++; }
      if (y < h - 1) { ea[e] = i; eb[e] = i + w; ew[e] = wt(i, i + w); e++; }
    }
    // counting sort on quantised weight
    const Q = 4096, cnt = new Int32Array(Q + 1), qk = new Int32Array(m);
    for (let i = 0; i < m; i++) { qk[i] = Math.min(Q - 1, (ew[i] * 30) | 0); cnt[qk[i] + 1]++; }
    for (let i = 0; i < Q; i++) cnt[i + 1] += cnt[i];
    const ord = new Int32Array(m);
    for (let i = 0; i < m; i++) ord[cnt[qk[i]]++] = i;
    const p = UF(n), sz = new Int32Array(n).fill(1), th = new Float32Array(n).fill(k);
    for (let t = 0; t < m; t++) {
      const i = ord[t], a = find(p, ea[i]), b = find(p, eb[i]);
      if (a === b) continue;
      const wv = ew[i];
      if (wv <= th[a] && wv <= th[b]) {
        const [big, small] = sz[a] >= sz[b] ? [a, b] : [b, a];
        p[small] = big; sz[big] += sz[small]; th[big] = wv + k / sz[big];
      }
    }
    for (let t = 0; t < m; t++) {
      const i = ord[t], a = find(p, ea[i]), b = find(p, eb[i]);
      if (a !== b && (sz[a] < minSize || sz[b] < minSize) && !(side && side[ea[i]] !== side[eb[i]])) { const [big, small] = sz[a] >= sz[b] ? [a, b] : [b, a]; p[small] = big; sz[big] += sz[small]; }
    }
    const lab = new Int32Array(n), map = new Int32Array(n).fill(-1);
    let R = 0;
    for (let i = 0; i < n; i++) { const r = find(p, i); if (map[r] < 0) map[r] = R++; lab[i] = map[r]; }
    return { lab, R };
  }

  /* ---------------- merge hierarchy (piecewise-constant Mumford–Shah) ---------------- */
  function Heap() { this.k = []; this.v = []; }
  Heap.prototype.push = function (key, val) {
    const k = this.k, v = this.v; let i = k.length; k.push(key); v.push(val);
    while (i > 0) { const pa = (i - 1) >> 1; if (k[pa] <= key) break; k[i] = k[pa]; v[i] = v[pa]; i = pa; }
    k[i] = key; v[i] = val;
  };
  Heap.prototype.pop = function () {
    const k = this.k, v = this.v, top = v[0], lk = k.pop(), lv = v.pop(), n = k.length;
    if (n) {
      let i = 0;
      for (;;) { let c = 2 * i + 1; if (c >= n) break; if (c + 1 < n && k[c + 1] < k[c]) c++; if (k[c] >= lk) break; k[i] = k[c]; v[i] = v[c]; i = c; }
      k[i] = lk; v[i] = lv;
    }
    return top;
  };

  function hierarchy(lab, R, L, A, B, w, h, abWeight, pside) {
    const n = w * h;
    const N = new Float64Array(R), sL = new Float64Array(R), sA = new Float64Array(R), sB = new Float64Array(R), sX = new Float64Array(R), sY = new Float64Array(R);
    for (let i = 0; i < n; i++) { const r = lab[i]; N[r]++; sL[r] += L[i]; sA[r] += A[i]; sB[r] += B[i]; sX[r] += i % w; sY[r] += (i / w) | 0; }
    const adj = Array.from({ length: R }, () => new Map());
    const link = (a, b) => { if (a === b) return; adj[a].set(b, (adj[a].get(b) || 0) + 1); adj[b].set(a, (adj[b].get(a) || 0) + 1); };
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const i = y * w + x;
      if (x < w - 1) link(lab[i], lab[i + 1]);
      if (y < h - 1) link(lab[i], lab[i + w]);
    }
    const ver = new Int32Array(R), alive = new Uint8Array(R).fill(1);
    const cost = (a, b, len) => {
      const dl = sL[a] / N[a] - sL[b] / N[b], da = sA[a] / N[a] - sA[b] / N[b], db = sB[a] / N[a] - sB[b] / N[b];
      return (N[a] * N[b] / (N[a] + N[b])) * (dl * dl + abWeight * (da * da + db * db)) / len;
    };
    const heap = new Heap();
    const ok = (a, b) => !pside || pside[a] === pside[b];
    for (let a = 0; a < R; a++) for (const [b, len] of adj[a]) if (a < b && ok(a, b)) heap.push(cost(a, b, len), [a, b, 0, 0]);
    const merges = []; // [keep, gone, lambda, cx, cy]
    while (heap.k.length) {
      const lam = heap.k[0], [a, b, va, vb] = heap.pop();
      if (!alive[a] || !alive[b] || ver[a] !== va || ver[b] !== vb) continue;
      const [keep, gone] = adj[a].size >= adj[b].size ? [a, b] : [b, a];
      N[keep] += N[gone]; sL[keep] += sL[gone]; sA[keep] += sA[gone]; sB[keep] += sB[gone]; sX[keep] += sX[gone]; sY[keep] += sY[gone];
      alive[gone] = 0;
      const ak = adj[keep];
      ak.delete(gone);
      for (const [c, len] of adj[gone]) {
        if (c === keep) continue;
        ak.set(c, (ak.get(c) || 0) + len);
        const ac = adj[c]; ac.delete(gone); ac.set(keep, (ac.get(keep) || 0) + len);
      }
      adj[gone] = null;
      ver[keep]++;
      merges.push([keep, gone, lam, sX[keep] / N[keep], sY[keep] / N[keep], pside ? pside[keep] : 1]);
      for (const [c, len] of ak) if (ok(keep, c)) heap.push(cost(keep, c, len), [keep, c, ver[keep], ver[c]]);
    }
    return merges;
  }

  function cutBy(base, R, merges, allow) {
    const p = UF(R);
    for (let i = 0; i < merges.length; i++) if (allow(i, merges[i])) { const a = find(p, merges[i][0]), b = find(p, merges[i][1]); if (a !== b) p[b] = a; }
    const out = new Int32Array(base.length);
    for (let i = 0; i < base.length; i++) out[i] = find(p, base[i]);
    return out;
  }

  // Apply the first `upto` merges everywhere, and merges up to `outside` only where the
  // merged region's centre lies outside the focus.
  function cutLabels(base, R, merges, upto, outside, focus, w) {
    const p = UF(R);
    for (let i = 0; i < merges.length; i++) {
      const m = merges[i];
      if (i < upto || (i < outside && !focus[((m[4] + 0.5) | 0) * w + ((m[3] + 0.5) | 0)])) {
        const a = find(p, m[0]), b = find(p, m[1]); if (a !== b) p[b] = a;
      } else if (i >= Math.max(upto, outside)) break;
    }
    const out = new Int32Array(base.length);
    for (let i = 0; i < base.length; i++) out[i] = find(p, base[i]);
    return out;
  }

  /* ---------------- focus (where detail goes) ---------------- */
  function focusMap(L, A, B, w, h, share) {
    const n = w * h; let mL = 0, mA = 0, mB = 0;
    for (let i = 0; i < n; i++) { mL += L[i]; mA += A[i]; mB += B[i]; }
    mL /= n; mA /= n; mB /= n;
    let s = new Float32Array(n);
    for (let i = 0; i < n; i++) { const dl = L[i] - mL, da = A[i] - mA, db = B[i] - mB; s[i] = Math.sqrt(dl * dl + da * da + db * db); }
    // Local contrast too: form detail draws the eye.
    const loc = boxMean(L, w, h, 3), loc2 = boxMean(L.map(v => v * v), w, h, 3);
    for (let i = 0; i < n; i++) s[i] = s[i] + 1.5 * Math.sqrt(Math.max(0, loc2[i] - loc[i] * loc[i]));
    const r = Math.max(2, Math.round(Math.hypot(w, h) * 0.035));
    s = boxMean(boxMean(s, w, h, r), w, h, r);
    const cx = w / 2, cy = h / 2, sx = 0.32 * w, sy = 0.32 * h;
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const i = y * w + x, c = Math.exp(-((x - cx) ** 2 / (2 * sx * sx) + (y - cy) ** 2 / (2 * sy * sy)));
      s[i] *= 0.35 + 0.65 * c;
    }
    const sorted = Float32Array.from(s).sort(), t = sorted[Math.floor((1 - share) * (n - 1))];
    const f = new Uint8Array(n);
    for (let i = 0; i < n; i++) f[i] = s[i] >= t ? 1 : 0;
    return f;
  }

  /* ---------------- tone cuts: divisive, nested ---------------- */
  function nestedCuts(vals, wts, nMax) {
    const H = new Float64Array(101), M = new Float64Array(101);
    let tot = 0;
    for (let i = 0; i < vals.length; i++) { const b = Math.max(0, Math.min(100, Math.round(vals[i]))); H[b] += wts[i]; M[b] += wts[i] * vals[i]; tot += wts[i]; }
    const P = new Float64Array(102), S = new Float64Array(102);
    for (let i = 0; i <= 100; i++) { P[i + 1] = P[i] + H[i]; S[i + 1] = S[i] + M[i]; }
    // best split of bins [a,b): returns [gain, cut] where cut c means values in bins >= c go up
    const best = (a, b) => {
      let bg = -1, bc = -1; const W = P[b] - P[a], Sm = S[b] - S[a];
      for (let c = a + 1; c < b; c++) {
        const w1 = P[c] - P[a], w2 = W - w1;
        if (w1 < 0.03 * tot || w2 < 0.03 * tot) continue;
        const m1 = (S[c] - S[a]) / w1, m2 = (Sm - (S[c] - S[a])) / w2, g = w1 * w2 / W * (m1 - m2) ** 2;
        if (g > bg) { bg = g; bc = c; }
      }
      return [bg, bc];
    };
    let ivs = [[0, 101]]; const order = [], byN = { 1: [] };
    for (let k = 2; k <= nMax; k++) {
      let bi = -1, bg = -1, bc = -1;
      ivs.forEach(([a, b], i) => { const [g, c] = best(a, b); if (g > bg) { bg = g; bi = i; bc = c; } });
      if (bi < 0) { byN[k] = byN[k - 1].slice(); continue; } // nothing left worth splitting
      const [a, b] = ivs[bi]; ivs.splice(bi, 1, [a, bc], [bc, b]);
      order.push(bc);
      byN[k] = order.slice().sort((x, y) => x - y);
    }
    return byN;
  }

  /* ---------------- shape clean-up ---------------- */
  function components(cls, w, h) {
    const n = w * h, comp = new Int32Array(n).fill(-1), stack = new Int32Array(n), sizes = [];
    let C = 0;
    for (let s = 0; s < n; s++) {
      if (comp[s] >= 0) continue;
      let top = 0, size = 0; stack[top++] = s; comp[s] = C;
      while (top) {
        const i = stack[--top]; size++;
        const x = i % w, c = cls[i];
        if (x > 0 && comp[i - 1] < 0 && cls[i - 1] === c) { comp[i - 1] = C; stack[top++] = i - 1; }
        if (x < w - 1 && comp[i + 1] < 0 && cls[i + 1] === c) { comp[i + 1] = C; stack[top++] = i + 1; }
        if (i >= w && comp[i - w] < 0 && cls[i - w] === c) { comp[i - w] = C; stack[top++] = i - w; }
        if (i < n - w && comp[i + w] < 0 && cls[i + w] === c) { comp[i + w] = C; stack[top++] = i + w; }
      }
      sizes.push(size); C++;
    }
    return { comp, C, sizes };
  }
  // Absorb islands smaller than minArea(componentCentre) into their longest-border neighbour.
  function absorb(cls, w, h, minAt, passes) {
    for (let pass = 0; pass < passes; pass++) {
      const { comp, C, sizes } = components(cls, w, h);
      const cx = new Float64Array(C), cy = new Float64Array(C), ccls = new Int32Array(C);
      for (let i = 0; i < cls.length; i++) { cx[comp[i]] += i % w; cy[comp[i]] += (i / w) | 0; ccls[comp[i]] = cls[i]; }
      const small = new Uint8Array(C); let any = false;
      for (let c = 0; c < C; c++) { if (sizes[c] < minAt(cx[c] / sizes[c], cy[c] / sizes[c])) { small[c] = 1; any = true; } }
      if (!any) break;
      const border = Array.from({ length: C }, () => null);
      const note = (a, b) => { if (!small[a] || ccls[a] === ccls[b]) return; const m = border[a] || (border[a] = new Map()); m.set(ccls[b], (m.get(ccls[b]) || 0) + 1); };
      for (let i = 0; i < cls.length; i++) {
        const x = i % w;
        if (x < w - 1 && comp[i] !== comp[i + 1]) { note(comp[i], comp[i + 1]); note(comp[i + 1], comp[i]); }
        if (i + w < cls.length && comp[i] !== comp[i + w]) { note(comp[i], comp[i + w]); note(comp[i + w], comp[i]); }
      }
      const to = new Int32Array(C).fill(-1);
      // smallest first, so a speck joins its surroundings rather than another speck
      const order = []; for (let c = 0; c < C; c++) if (small[c] && border[c]) order.push(c);
      order.sort((a, b) => sizes[a] - sizes[b]);
      for (const c of order) { let bc = -1, bl = -1; for (const [k, l] of border[c]) if (l > bl) { bl = l; bc = k; } to[c] = bc; }
      for (let i = 0; i < cls.length; i++) if (to[comp[i]] >= 0) cls[i] = to[comp[i]];
    }
    return cls;
  }

  /* ---------------- vectorise as a shared map ---------------- */
  function rdpOpen(xs, ys, eps) {
    const n = xs.length, keep = new Uint8Array(n); keep[0] = keep[n - 1] = 1;
    const st = [[0, n - 1]];
    while (st.length) {
      const [a, b] = st.pop(); if (b - a < 2) continue;
      const ax = xs[a], ay = ys[a], dx = xs[b] - ax, dy = ys[b] - ay, len = Math.hypot(dx, dy);
      let md = -1, mi = -1;
      for (let i = a + 1; i < b; i++) {
        const d = len > 1e-9 ? Math.abs(dy * (xs[i] - ax) - dx * (ys[i] - ay)) / len : Math.hypot(xs[i] - ax, ys[i] - ay);
        if (d > md) { md = d; mi = i; }
      }
      if (md > eps) { keep[mi] = 1; st.push([a, mi], [mi, b]); }
    }
    const out = []; for (let i = 0; i < n; i++) if (keep[i]) out.push(xs[i], ys[i]);
    return out;
  }
  // Pixel borders are staircases; average each vertex with its neighbours along the
  // chain (ends fixed) so the simplifier finds the true corners, not the stair steps.
  function relax(a, k, closed) {
    const n = a.length; if (n < 3) return a;
    const out = a.slice();
    for (let i = 1; i < n - 1; i++) {
      let s = 0, c = 0;
      for (let j = i - k; j <= i + k; j++) {
        let t = j;
        if (closed) t = ((j % (n - 1)) + (n - 1)) % (n - 1); else if (t < 0 || t >= n) continue;
        s += a[t]; c++;
      }
      out[i] = s / c;
    }
    return out;
  }
  function simplifyChain(vs, W1, eps, closed) {
    const k = Math.max(1, Math.min(4, Math.round(eps)));
    const xs = relax(vs.map(v => v % W1), k, closed), ys = relax(vs.map(v => (v / W1) | 0), k, closed);
    if (!closed) return rdpOpen(xs, ys, eps);
    // closed: split at the farthest point from the start so both halves have fixed ends
    let far = 0, fd = -1;
    for (let i = 1; i < xs.length - 1; i++) { const d = Math.hypot(xs[i] - xs[0], ys[i] - ys[0]); if (d > fd) { fd = d; far = i; } }
    const a = rdpOpen(xs.slice(0, far + 1), ys.slice(0, far + 1), eps), b = rdpOpen(xs.slice(far), ys.slice(far), eps);
    return a.concat(b.slice(2));
  }

  function vectorise(lab, w, h, eps) {
    const W1 = w + 1, NV = W1 * (h + 1);
    const at = (x, y) => (x < 0 || y < 0 || x >= w || y >= h ? -1 : lab[y * w + x]);
    // owner (right-hand pixel) and left pixel of the edge leaving vertex (x,y) in dir d
    const OX = [0, -1, -1, 0], OY = [0, 0, -1, -1], LX = [0, 0, -1, -1], LY = [-1, 0, 0, -1];
    const DV = [1, W1, -1, -W1];
    const node = new Uint8Array(NV);
    for (let y = 0; y <= h; y++) for (let x = 0; x <= w; x++) {
      const p1 = at(x - 1, y - 1), p2 = at(x, y - 1), p3 = at(x - 1, y), p4 = at(x, y);
      const s = new Set([p1, p2, p3, p4]);
      if (s.size >= 3 || (s.size === 2 && p1 === p4 && p2 === p3)) node[y * W1 + x] = 1;
    }
    const used = new Uint8Array(NV);
    const edgeOwner = (v, d) => {
      const x = v % W1, y = (v / W1) | 0, o = at(x + OX[d], y + OY[d]);
      return o >= 0 && o !== at(x + LX[d], y + LY[d]) ? o : -2;
    };
    const loopsBy = new Map(); // label -> array of vertex loops
    for (let v0 = 0; v0 < NV; v0++) for (let d0 = 0; d0 < 4; d0++) {
      if (used[v0] & (1 << d0)) continue;
      const l = edgeOwner(v0, d0); if (l < 0) continue;
      const loop = []; let v = v0, d = d0;
      for (;;) {
        used[v] |= 1 << d; loop.push(v); v += DV[d];
        if (v === v0) break;
        let nd = -1;
        for (const c of [(d + 1) & 3, d, (d + 3) & 3]) if (!(used[v] & (1 << c)) && edgeOwner(v, c) === l) { nd = c; break; }
        if (nd < 0) break;
        d = nd;
      }
      if (!loopsBy.has(l)) loopsBy.set(l, []);
      loopsBy.get(l).push(loop);
    }
    const cache = new Map(), chains = [];
    const regions = [];
    for (const [l, loops] of loopsBy) {
      const polys = [];
      for (const loop of loops) {
        const n = loop.length, nodesAt = [];
        for (let i = 0; i < n; i++) if (node[loop[i]]) nodesAt.push(i);
        let pieces = [];
        if (!nodesAt.length) {
          let mi = 0; for (let i = 1; i < n; i++) if (loop[i] < loop[mi]) mi = i;
          const rot = loop.slice(mi).concat(loop.slice(0, mi)); rot.push(rot[0]);
          pieces.push(rot);
        } else {
          for (let k = 0; k < nodesAt.length; k++) {
            const a = nodesAt[k], b = k + 1 < nodesAt.length ? nodesAt[k + 1] : nodesAt[0] + n;
            const pc = []; for (let i = a; i <= b; i++) pc.push(loop[i % n]);
            pieces.push(pc);
          }
        }
        const pts = [];
        for (const pc of pieces) {
          const s = pc[0], e = pc[pc.length - 1], closed = s === e;
          let fwd;
          if (closed) fwd = pc[1] <= pc[pc.length - 2];
          else fwd = s < e;
          const canon = fwd ? pc : pc.slice().reverse();
          const key = canon[0] + ',' + canon[canon.length - 1] + ',' + canon[1];
          let ch = cache.get(key);
          if (!ch) { ch = { pts: simplifyChain(canon, W1, eps, closed), owners: [] }; cache.set(key, ch); chains.push(ch); }
          ch.owners.push(l);
          let q = ch.pts;
          if (!fwd) { const r = []; for (let i = q.length - 2; i >= 0; i -= 2) r.push(q[i], q[i + 1]); q = r; }
          for (let i = pts.length ? 2 : 0; i < q.length; i++) pts.push(q[i]);
        }
        if (pts.length >= 6) polys.push(pts);
      }
      regions.push({ label: l, polys });
    }
    return { regions, chains };
  }

  /* ---------------- public API ---------------- */
  const DEFAULTS = { work: 600, coarse: 22, medium: 120, fine: 450, focusShare: 0.35,
    // with a subject mask: how many shapes the subject is allowed at each level
    kCoarse: 6, kDetail: 1.2, kFine: 0.4 };

  // opts.subject: optional Float32Array/Uint8Array (w*h, 0..1 or 0..255) from a subject cut-out.
  // opts.focus: optional Uint8Array (w*h, 0/1) — where detail goes when there is no subject.
  function analyze(rgba, w0, h0, opts) {
    const o = Object.assign({}, DEFAULTS, opts || {});
    const w = w0, h = h0, n = w * h;   // caller supplies pixels already at working size
    const lab = toLab(rgba, n);
    const diag = Math.hypot(w, h);
    const r = Math.max(2, Math.round(diag * 0.006));
    let L = guided(lab.L, w, h, r, 40), A = guided(lab.A, w, h, r, 40), B = guided(lab.B, w, h, r, 40);
    L = guided(L, w, h, r, 25);
    let side = null;
    if (o.subject) {
      const s = o.subject, mx = s.reduce((m, v) => Math.max(m, v), 0) > 1 ? 255 : 1;
      side = new Uint8Array(n);
      let cnt = 0; for (let i = 0; i < n; i++) { side[i] = s[i] / mx >= 0.5 ? 1 : 0; cnt += side[i]; }
      if (cnt < n * 0.02 || cnt > n * 0.97) side = null; // no usable subject: fall back
    }
    const seg = patches(L, A, B, w, h, 25, Math.max(6, Math.round(n / 20000)), side);
    let pside = null;
    if (side) {
      const c = new Float64Array(seg.R), t = new Float64Array(seg.R);
      for (let i = 0; i < n; i++) { c[seg.lab[i]]++; t[seg.lab[i]] += side[i]; }
      pside = new Uint8Array(seg.R); for (let k = 0; k < seg.R; k++) pside[k] = t[k] / c[k] >= 0.5 ? 1 : 0;
    }
    const merges = hierarchy(seg.lab, seg.R, L, A, B, w, h, 0.25, pside);
    const focus = side || (o.focus && o.focus.some(v => v) ? o.focus : focusMap(L, A, B, w, h, o.focusShare));
    // ordinal of each subject-side merge, so "allow the subject N shapes" is a simple test
    const sOrd = new Int32Array(merges.length); let sTotal = 0, ns = 0;
    if (pside) { for (let k = 0; k < seg.R; k++) ns += pside[k]; for (let i = 0; i < merges.length; i++) sOrd[i] = merges[i][5] ? sTotal++ : -1; }
    // with a subject, an optional focus box narrows where the detail goes inside it
    const zone = side && o.focus && o.focus.some(v => v) ? o.focus : null;
    const an = { w, h, L, seg, merges, focus, side, zone, pside, sOrd, ns, opts: o };
    // tone cuts from the subject's medium-sized shapes (background ignored when we know it)
    const med = regionsAt(an, 'medium');
    const acc = new Map();
    for (let i = 0; i < n; i++) { let a = acc.get(med[i]); if (!a) acc.set(med[i], a = [0, 0, 0]); a[0]++; a[1] += L[i]; a[2] += zone ? focus[i] * (0.25 + zone[i]) : focus[i]; }
    const vals = [], wts = [];
    for (const [, a] of acc) { vals.push(a[1] / a[0]); wts.push(a[0] * ((side ? 0.0 : 0.03) + a[2] / a[0])); }
    an.cuts = nestedCuts(vals, wts, 5);
    if (side) {
      // Figure against ground: the background keeps its real family (black or white) unless
      // too little of the subject's outline would stand out against it; then it flips so the
      // silhouette reads. Decided once, kept across every step.
      const t1 = an.cuts[2][0];
      const mean = new Map(); for (const [r, a] of acc) mean.set(r, a[1] / a[0]);
      let light = 0, edge = 0;
      for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
        const i = y * w + x; if (!side[i]) continue;
        if ((x > 0 && !side[i - 1]) || (x < w - 1 && !side[i + 1]) || (y > 0 && !side[i - w]) || (y < h - 1 && !side[i + w])) {
          edge++; if (mean.get(med[i]) >= t1) light++;
        }
      }
      let bl = 0, bc = 0; for (let i = 0; i < n; i++) if (!side[i]) { bl += L[i]; bc++; }
      const trueLight = bc ? bl / bc >= t1 : true;
      const contrast = edge ? (trueLight ? edge - light : light) / edge : 1;
      an.bgLight = contrast < 0.3 ? !trueLight : trueLight; // keep the real value unless the silhouette would vanish
    }
    return an;
  }

  function regionsAt(an, level) {
    const { seg, merges, focus, w, opts: o } = an, R = seg.R;
    if (an.pside) {
      // Merge while it's cheap: a merge's price is (size x value difference^2) / shared edge,
      // so texture goes first and a high-contrast feature (a face, an eye socket) survives
      // regardless of how much texture surrounds it. Threshold scales with image size.
      const k = level === 'coarse' ? o.kCoarse : level === 'medium' || level === 'detail' ? o.kDetail : o.kFine;
      const lam = k * Math.sqrt(an.w * an.h);
      // background: merge everything (one flat field); subject: merges cheaper than lam
      return cutBy(seg.lab, R, merges, (i, m) => !m[5] || m[2] <= lam);
    }
    if (level === 'coarse') return cutLabels(seg.lab, R, merges, Math.max(0, R - o.coarse), 0, focus, w);
    if (level === 'medium') return cutLabels(seg.lab, R, merges, Math.max(0, R - o.medium), 0, focus, w);
    return cutLabels(seg.lab, R, merges, Math.max(0, R - o.fine), Math.max(0, R - o.medium), focus, w);
  }

  // level: 'coarse' (big shapes) | 'detail' (more shapes in the subject) | 'fine'
  // extra: { shift: L* units (+ = darker), simplify: 0.5..2 (bigger, fewer shapes),
  //          background: 'auto' | 'dark' | 'light' }
  function step(an, tones, level, extra) {
    const { w, h, L, focus } = an, n = w * h;
    const shift = (extra && extra.shift) || 0, simp = (extra && extra.simplify) || 1;
    const bgMode = (extra && extra.background) || 'auto';
    const cuts = an.cuts[tones].map(c => c + shift);
    // Decide light/dark on medium shapes first (squint), then let each step drop the
    // shapes too small to matter at that step. Merging first would average a face into
    // the dark mass around it.
    const regs = regionsAt(an, level === 'fine' ? 'fine' : 'detail');
    const sum = new Map();
    for (let i = 0; i < n; i++) { let a = sum.get(regs[i]); if (!a) sum.set(regs[i], a = [0, 0]); a[0]++; a[1] += L[i]; }
    const clsOf = new Map();
    for (const [r, a] of sum) { const m = a[1] / a[0]; let c = 0; while (c < cuts.length && m >= cuts[c]) c++; clsOf.set(r, c); }
    const cls = new Int32Array(n);
    for (let i = 0; i < n; i++) cls[i] = clsOf.get(regs[i]);
    if (an.side) {
      // background stays one flat value across every step (chosen in analyze)
      const bg = (bgMode === 'auto' ? an.bgLight : bgMode === 'light') ? tones - 1 : 0;
      for (let i = 0; i < n; i++) if (!an.side[i]) cls[i] = bg;
    }
    let area = n; if (an.side) { area = 0; for (let i = 0; i < n; i++) area += an.side[i]; }
    const frac = { coarse: 0.008, detail: 0.0015, fine: 0.0006 }[level] || 0.0015;
    const minA = area * frac * simp, bgMin = n * 0.008 * simp;
    const at = (m, x, y) => m[((y + 0.5) | 0) * w + ((x + 0.5) | 0)];
    const bigA = area * 0.008 * simp;
    absorb(cls, w, h, level === 'coarse' ? () => minA
      : an.zone ? (x, y) => (!at(focus, x, y) ? bgMin : at(an.zone, x, y) ? minA : bigA)
      : (x, y) => (at(focus, x, y) ? minA : bgMin), 3);
    const { comp, C } = components(cls, w, h);
    const compCls = new Int32Array(C);
    for (let i = 0; i < n; i++) compCls[comp[i]] = cls[i];
    const eps = (extra && extra.eps) || (level === 'coarse' ? 0.009 : level === 'detail' ? 0.005 : 0.004) * Math.hypot(w, h) * Math.sqrt(simp);
    const vec = vectorise(comp, w, h, eps);
    // family of each tone in the 2-tone split, for outline weight
    const t1 = an.cuts[2][0] + shift;
    const family = c => (c === 0 ? 0 : (cuts[c - 1] >= t1 ? 1 : 0));
    return { w, h, tones, compCls, vec, family };
  }

  function draw(ctx, res, o) {
    const s = o.scale || 1, pal = o.palette || greys(res.tones);
    ctx.save(); ctx.scale(s, s); ctx.lineJoin = 'round'; ctx.lineCap = 'round';
    ctx.fillStyle = pal[0]; ctx.fillRect(0, 0, res.w, res.h);
    for (const r of res.vec.regions) {
      const p = new Path2D();
      for (const q of r.polys) { p.moveTo(q[0], q[1]); for (let i = 2; i < q.length; i += 2) p.lineTo(q[i], q[i + 1]); p.closePath(); }
      const col = pal[res.compCls[r.label]];
      ctx.fillStyle = col; ctx.fill(p, 'evenodd');
      ctx.strokeStyle = col; ctx.lineWidth = 1 / s; ctx.stroke(p); // hide anti-alias seams
    }
    if (o.outline !== false) {
      ctx.strokeStyle = o.lineColor || '#e5484d';
      const lw = (o.lineWidth || 3) / s;
      for (const ch of res.vec.chains) {
        if (ch.owners.length < 2) continue; // image border
        const [a, b] = ch.owners, fa = res.family(res.compCls[a]), fb = res.family(res.compCls[b]);
        ctx.lineWidth = fa !== fb ? lw : lw * 0.55;
        const q = ch.pts; ctx.beginPath(); ctx.moveTo(q[0], q[1]);
        for (let i = 2; i < q.length; i += 2) ctx.lineTo(q[i], q[i + 1]);
        ctx.stroke();
      }
    }
    ctx.restore();
  }

  function greys(n) { return Array.from({ length: n }, (_, i) => { const v = Math.round(255 * i / (n - 1)); return `rgb(${v},${v},${v})`; }); }

  root.BlocksEngine = { analyze, step, draw, greys, DEFAULTS };
})(typeof window !== 'undefined' ? window : globalThis);
