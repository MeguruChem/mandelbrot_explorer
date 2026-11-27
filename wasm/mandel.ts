// Inner loops of the Mandelbrot renderer. No allocations: reference orbit and
// BLA tables live in linear memory at offsets chosen by the worker.
export let statIters: f64 = 0;
export let statSkip: f64 = 0;
// distance estimate of the last escaped point (in c units), 0 when not computed
export let lastDE: f64 = 0;
export function heapBase(): usize { return __heap_base; }
export function resetStats(): void { statIters = 0; statSkip = 0; }

// per-job parameters
let gScale: f64 = 0, gOx: f64 = 0, gOy: f64 = 0, gAbsX: f64 = 0, gAbsY: f64 = 0, gSp2: f64 = 0;
let gMaxIter: i32 = 0, gMode: i32 = 0, gLen: i32 = 0, gTop: i32 = 0;
let gZp: usize = 0, gLvp: usize = 0;
let gInterior: bool = false, gDE: bool = false;
// Julia mode: z0 = pixel, c fixed. The derivative tracked is then dz/dz0 (no +1 term).
let gJulia: bool = false, gJcr: f64 = 0, gJci: f64 = 0, gDcAdd: f64 = 1;

export function setup(scale: f64, ox: f64, oy: f64, absX: f64, absY: f64, maxIter: i32, mode: i32,
  zp: usize, len: i32, lvp: usize, top: i32, interior: i32, de: i32, julia: i32, jcr: f64, jci: f64): void {
  gInterior = interior != 0; gDE = de != 0;
  gJulia = julia != 0; gJcr = jcr; gJci = jci; gDcAdd = gJulia ? 0 : 1;
  gScale = scale; gOx = ox; gOy = oy; gAbsX = absX; gAbsY = absY; gMaxIter = maxIter; gMode = mode;
  gZp = zp; gLen = len; gLvp = lvp; gTop = top;
  // interior test tolerance: half a pixel's worth of z, capped
  gSp2 = scale * 0.5;   // (squaring scale would underflow beyond 1e-154)
}

// |z| log|z| / |dz/dc|: about the distance from c to the set (true distance is between ~1/4 and ~2x of it)
// @ts-ignore
@inline function de(mag: f64, dcr: f64, dci: f64): f64 {
  const d = Math.sqrt(dcr * dcr + dci * dci);
  return d > 0 ? 0.5 * Math.sqrt(mag) * Math.log(mag) / d : 0;
}
// @ts-ignore
@inline function smooth(n: f64, mag: f64): f64 {
  const logZn = Math.log(mag) * 0.5;
  const v = n - Math.log(logZn / Math.LN2) / Math.LN2;
  return v > 0 ? v : 0;
}

function ptD(cr: f64, ci: f64, maxIter: i32): f64 {
  const xq = cr - 0.25, q = xq * xq + ci * ci;
  if (q * (q + xq) <= 0.25 * ci * ci) return -1;
  const xb = cr + 1; if (xb * xb + ci * ci <= 0.0625) return -1;
  return ptDfrom(cr, ci, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, maxIter);
}
// scalar double iteration, resumable from a given state (used to finish a SIMD pair lane)
function ptDfrom(cr: f64, ci: f64, zr: f64, zi: f64, zr2: f64, zi2: f64, dcr: f64, dci: f64,
  pr: f64, pi: f64, per: i32, perLen: i32, n: i32, maxIter: i32): f64 {
  lastDE = 0;
  for (; n < maxIter; n++) {
    if (gDE) { const t = 2 * (zr * dcr - zi * dci) + gDcAdd; dci = 2 * (zr * dci + zi * dcr); dcr = t; }
    zi = 2 * zr * zi + ci;
    zr = zr2 - zi2 + cr;
    zr2 = zr * zr; zi2 = zi * zi;
    if (zr2 + zi2 > 1e6) break;
    if (zr == pr && zi == pi) { statIters += <f64>n; return -1; }
    if (++per > perLen) { per = 0; perLen <<= 1; pr = zr; pi = zi; }
  }
  statIters += <f64>n;
  if (n >= maxIter) return -1;
  if (gDE) lastDE = de(zr2 + zi2, dcr, dci);
  return smooth(<f64>(n + 1), zr2 + zi2);
}

// Perturbation (+ optional BLA) with optional interior detection, resumable from a state.
function ptPB(dr: f64, di: f64, useBla: bool): f64 {
  return ptPBfrom(dr, di, useBla, 0, 0, 1, 0, 0, 0, 0, 0, 0, 8, 0, 0);
}
function ptPBfrom(dr: f64, di: f64, useBla: bool, er: f64, ei: f64, dzr: f64, dzi: f64, dcr: f64, dci: f64,
  sr: f64, si: f64, per: i32, perLen: i32, m: i32, n: i32): f64 {
  const zp = gZp, len = gLen, maxIter = gMaxIter, lvp = gLvp, top = gTop;
  lastDE = 0;
  while (n < maxIter) {
    let applied = false;
    if (useBla && m > 0 && top >= 1) {
      const e2 = er * er + ei * ei;
      const mm = m - 1;
      let lmax: i32 = mm == 0 ? top : <i32>ctz<u32>(<u32>mm);
      if (lmax > top) lmax = top;
      for (let l = lmax; l >= 1; l--) {
        const step = 1 << l;
        if (m + step > len - 1 || n + step > maxIter) continue;
        const k = mm >> l;
        if (<u32>k >= load<u32>(lvp + (<usize>(top + 1 + l) << 2))) continue;
        const e = <usize>load<u32>(lvp + (<usize>l << 2)) + <usize>k * 40;
        const R = load<f64>(e, 32);
        if (e2 < R * R) {
          const Ar = load<f64>(e), Ai = load<f64>(e, 8), Br = load<f64>(e, 16), Bi = load<f64>(e, 24);
          const nr = Ar * er - Ai * ei + Br * dr - Bi * di;
          const ni = Ar * ei + Ai * er + Br * di + Bi * dr;
          const ndr = Ar * dzr - Ai * dzi;
          const ndi = Ar * dzi + Ai * dzr;
          if (gDE) { const t = Ar * dcr - Ai * dci + Br; dci = Ar * dci + Ai * dcr + Bi; dcr = t; }
          er = nr; ei = ni; dzr = ndr; dzi = ndi;
          m += step; n += step; statSkip += <f64>step;
          applied = true;
          break;
        }
      }
    }
    if (!applied) {
      const a = zp + (<usize>m << 4);
      const zr = load<f64>(a), zi = load<f64>(a, 8);
      const xr = zr + er, xi = zi + ei;            // full z_n
      const ndr = 2 * (xr * dzr - xi * dzi);
      const ndi = 2 * (xr * dzi + xi * dzr);
      if (gDE) { const t = 2 * (xr * dcr - xi * dci) + 1; dci = 2 * (xr * dci + xi * dcr); dcr = t; }
      const ner = 2 * (zr * er - zi * ei) + er * er - ei * ei + dr;
      const nei = 2 * (zr * ei + zi * er) + 2 * er * ei + di;
      er = ner; ei = nei; dzr = ndr; dzi = ndi; m++; n++;
    }
    const b = zp + (<usize>m << 4);
    const fr = load<f64>(b) + er, fi = load<f64>(b, 8) + ei;
    const mag = fr * fr + fi * fi;
    if (mag > 1e6) { statIters += <f64>n; if (gDE) lastDE = de(mag, dcr, dci); return smooth(<f64>n, mag); }
    // interior: back at the saved point (relative 1e-6) and the loop contracts (|dz/dz_saved| < 1/2)
    const ddx = fr - sr, ddy = fi - si;
    if (gInterior && ddx * ddx + ddy * ddy < 1e-12 * (sr * sr + si * si) && dzr * dzr + dzi * dzi < 0.25) { statIters += <f64>n; return -1; }
    if (++per > perLen) { per = 0; perLen <<= 1; sr = fr; si = fi; dzr = 1; dzi = 0; }
    if (mag < er * er + ei * ei) { er = fr; ei = fi; m = 0; }
    else if (m >= len - 1) { statIters += <f64>n; return -2; }
  }
  statIters += <f64>n;
  return -1;
}

// ===================== SIMD: two pixels in lockstep =====================
// Both lanes run the same instruction stream while nothing lane-specific happens. On the first
// event in either lane (escape, periodicity/interior hit, rebase, glitch) the pair goes back one
// step and each lane finishes with the scalar code from that saved state.
export let out0: f64 = 0, out1: f64 = 0, de0: f64 = 0, de1: f64 = 0;

// @ts-ignore
@inline function L0(v: v128): f64 { return f64x2.extract_lane(v, 0); }
// @ts-ignore
@inline function L1(v: v128): f64 { return f64x2.extract_lane(v, 1); }
// @ts-ignore
@inline function S(x: f64): v128 { return f64x2.splat(x); }

function pairD(cr0: f64, ci0: f64, cr1: f64, ci1: f64, z0r: f64, z0i: f64, z1r: f64, z1i: f64, dc0: f64): void {
  const maxIter = gMaxIter;
  const cr = f64x2(cr0, cr1), ci = f64x2(ci0, ci1);
  let zr = f64x2(z0r, z1r), zi = f64x2(z0i, z1i);
  let zr2 = f64x2.mul(zr, zr), zi2 = f64x2.mul(zi, zi), dcr = S(dc0), dci = S(0), pr = S(0), pi = S(0);
  let per: i32 = 0, perLen: i32 = 8;
  const two = S(2), one = S(gDcAdd), big = S(1e6);
  for (let n: i32 = 0; n < maxIter; n++) {
    const p_zr = zr, p_zi = zi, p_zr2 = zr2, p_zi2 = zi2, p_dcr = dcr, p_dci = dci, p_pr = pr, p_pi = pi;
    const p_per = per, p_perLen = perLen;
    if (gDE) {
      const t = f64x2.add(f64x2.mul(two, f64x2.sub(f64x2.mul(zr, dcr), f64x2.mul(zi, dci))), one);
      dci = f64x2.mul(two, f64x2.add(f64x2.mul(zr, dci), f64x2.mul(zi, dcr)));
      dcr = t;
    }
    zi = f64x2.add(f64x2.mul(f64x2.mul(two, zr), zi), ci);
    zr = f64x2.add(f64x2.sub(zr2, zi2), cr);
    zr2 = f64x2.mul(zr, zr); zi2 = f64x2.mul(zi, zi);
    const esc = f64x2.gt(f64x2.add(zr2, zi2), big);
    const cyc = v128.and(f64x2.eq(zr, pr), f64x2.eq(zi, pi));
    if (v128.any_true(v128.or(esc, cyc))) {
      out0 = ptDfrom(cr0, ci0, L0(p_zr), L0(p_zi), L0(p_zr2), L0(p_zi2), L0(p_dcr), L0(p_dci), L0(p_pr), L0(p_pi), p_per, p_perLen, n, maxIter);
      de0 = lastDE;
      out1 = ptDfrom(cr1, ci1, L1(p_zr), L1(p_zi), L1(p_zr2), L1(p_zi2), L1(p_dcr), L1(p_dci), L1(p_pr), L1(p_pi), p_per, p_perLen, n, maxIter);
      de1 = lastDE;
      return;
    }
    if (++per > perLen) { per = 0; perLen <<= 1; pr = zr; pi = zi; }
  }
  statIters += 2 * <f64>maxIter;
  out0 = -1; out1 = -1; de0 = 0; de1 = 0;
}

function pairP(dr0: f64, di0: f64, dr1: f64, di1: f64, useBla: bool): void {
  const zp = gZp, len = gLen, maxIter = gMaxIter, lvp = gLvp, top = gTop;
  const dr = f64x2(dr0, dr1), di = f64x2(di0, di1);
  let er = S(0), ei = S(0), dzr = S(1), dzi = S(0), dcr = S(0), dci = S(0), sr = S(0), si = S(0);
  let per: i32 = 0, perLen: i32 = 8, m: i32 = 0, n: i32 = 0;
  const two = S(2), one = S(1), big = S(1e6), tiny = S(1e-12), quarter = S(0.25);
  while (n < maxIter) {
    const p_er = er, p_ei = ei, p_dzr = dzr, p_dzi = dzi, p_dcr = dcr, p_dci = dci, p_sr = sr, p_si = si;
    const p_per = per, p_perLen = perLen, p_m = m, p_n = n;
    let applied = false;
    if (useBla && m > 0 && top >= 1) {
      const e2v = f64x2.add(f64x2.mul(er, er), f64x2.mul(ei, ei));
      const e2 = max(L0(e2v), L1(e2v));              // a skip must be valid for both lanes
      const mm = m - 1;
      let lmax: i32 = mm == 0 ? top : <i32>ctz<u32>(<u32>mm);
      if (lmax > top) lmax = top;
      for (let l = lmax; l >= 1; l--) {
        const step = 1 << l;
        if (m + step > len - 1 || n + step > maxIter) continue;
        const k = mm >> l;
        if (<u32>k >= load<u32>(lvp + (<usize>(top + 1 + l) << 2))) continue;
        const e = <usize>load<u32>(lvp + (<usize>l << 2)) + <usize>k * 40;
        const R = load<f64>(e, 32);
        if (e2 < R * R) {
          const Ar = S(load<f64>(e)), Ai = S(load<f64>(e, 8)), Br = S(load<f64>(e, 16)), Bi = S(load<f64>(e, 24));
          const nr = f64x2.sub(f64x2.add(f64x2.sub(f64x2.mul(Ar, er), f64x2.mul(Ai, ei)), f64x2.mul(Br, dr)), f64x2.mul(Bi, di));
          const ni = f64x2.add(f64x2.add(f64x2.add(f64x2.mul(Ar, ei), f64x2.mul(Ai, er)), f64x2.mul(Br, di)), f64x2.mul(Bi, dr));
          const ndr = f64x2.sub(f64x2.mul(Ar, dzr), f64x2.mul(Ai, dzi));
          const ndi = f64x2.add(f64x2.mul(Ar, dzi), f64x2.mul(Ai, dzr));
          if (gDE) {
            const t = f64x2.add(f64x2.sub(f64x2.mul(Ar, dcr), f64x2.mul(Ai, dci)), Br);
            dci = f64x2.add(f64x2.add(f64x2.mul(Ar, dci), f64x2.mul(Ai, dcr)), Bi);
            dcr = t;
          }
          er = nr; ei = ni; dzr = ndr; dzi = ndi;
          m += step; n += step; statSkip += 2 * <f64>step;
          applied = true;
          break;
        }
      }
    }
    if (!applied) {
      const a = zp + (<usize>m << 4);
      const zr = S(load<f64>(a)), zi = S(load<f64>(a, 8));
      const xr = f64x2.add(zr, er), xi = f64x2.add(zi, ei);
      const ndr = f64x2.mul(two, f64x2.sub(f64x2.mul(xr, dzr), f64x2.mul(xi, dzi)));
      const ndi = f64x2.mul(two, f64x2.add(f64x2.mul(xr, dzi), f64x2.mul(xi, dzr)));
      if (gDE) {
        const t = f64x2.add(f64x2.mul(two, f64x2.sub(f64x2.mul(xr, dcr), f64x2.mul(xi, dci))), one);
        dci = f64x2.mul(two, f64x2.add(f64x2.mul(xr, dci), f64x2.mul(xi, dcr)));
        dcr = t;
      }
      // same operation order as the scalar code: ((2(zr er - zi ei) + er^2) - ei^2) + dr
      const ner = f64x2.add(f64x2.sub(f64x2.add(f64x2.mul(two, f64x2.sub(f64x2.mul(zr, er), f64x2.mul(zi, ei))), f64x2.mul(er, er)), f64x2.mul(ei, ei)), dr);
      const nei = f64x2.add(f64x2.add(f64x2.mul(two, f64x2.add(f64x2.mul(zr, ei), f64x2.mul(zi, er))), f64x2.mul(f64x2.mul(two, er), ei)), di);
      er = ner; ei = nei; dzr = ndr; dzi = ndi; m++; n++;
    }
    const b = zp + (<usize>m << 4);
    const fr = f64x2.add(S(load<f64>(b)), er), fi = f64x2.add(S(load<f64>(b, 8)), ei);
    const mag = f64x2.add(f64x2.mul(fr, fr), f64x2.mul(fi, fi));
    let ev = v128.or(f64x2.gt(mag, big), f64x2.lt(mag, f64x2.add(f64x2.mul(er, er), f64x2.mul(ei, ei))));
    if (gInterior) {
      const ddx = f64x2.sub(fr, sr), ddy = f64x2.sub(fi, si);
      const near = f64x2.lt(f64x2.add(f64x2.mul(ddx, ddx), f64x2.mul(ddy, ddy)), f64x2.mul(tiny, f64x2.add(f64x2.mul(sr, sr), f64x2.mul(si, si))));
      const contr = f64x2.lt(f64x2.add(f64x2.mul(dzr, dzr), f64x2.mul(dzi, dzi)), quarter);
      ev = v128.or(ev, v128.and(near, contr));
    }
    if (v128.any_true(ev) || m >= len - 1) {
      out0 = ptPBfrom(dr0, di0, useBla, L0(p_er), L0(p_ei), L0(p_dzr), L0(p_dzi), L0(p_dcr), L0(p_dci), L0(p_sr), L0(p_si), p_per, p_perLen, p_m, p_n);
      de0 = lastDE;
      out1 = ptPBfrom(dr1, di1, useBla, L1(p_er), L1(p_ei), L1(p_dzr), L1(p_dzi), L1(p_dcr), L1(p_dci), L1(p_sr), L1(p_si), p_per, p_perLen, p_m, p_n);
      de1 = lastDE;
      return;
    }
    if (++per > perLen) { per = 0; perLen <<= 1; sr = fr; si = fi; dzr = S(1); dzi = S(0); }
  }
  statIters += 2 * <f64>n;
  out0 = -1; out1 = -1; de0 = 0; de1 = 0;
}

// two pixels at once; results in out0/out1 (and de0/de1)
export function pt2(px0: f64, py0: f64, px1: f64, py1: f64): void {
  if (gMode == 0 && gJulia) {
    pairD(gJcr, gJci, gJcr, gJci, gAbsX + px0 * gScale, gAbsY - py0 * gScale, gAbsX + px1 * gScale, gAbsY - py1 * gScale, 1);
    return;
  }
  if (gMode == 0) {
    const cr0 = gAbsX + px0 * gScale, ci0 = gAbsY - py0 * gScale, cr1 = gAbsX + px1 * gScale, ci1 = gAbsY - py1 * gScale;
    // points that the cardioid/bulb test settles at once do not need the pair loop
    const in0 = inMain(cr0, ci0), in1 = inMain(cr1, ci1);
    if (in0 || in1) {
      out0 = in0 ? -1 : ptD(cr0, ci0, gMaxIter); de0 = in0 ? 0 : lastDE;
      out1 = in1 ? -1 : ptD(cr1, ci1, gMaxIter); de1 = in1 ? 0 : lastDE;
      return;
    }
    pairD(cr0, ci0, cr1, ci1, 0, 0, 0, 0, 0);
    return;
  }
  pairP(px0 * gScale + gOx, -py0 * gScale + gOy, px1 * gScale + gOx, -py1 * gScale + gOy, gMode == 2);
}
function inMain(cr: f64, ci: f64): bool {
  const xq = cr - 0.25, q = xq * xq + ci * ci;
  if (q * (q + xq) <= 0.25 * ci * ci) return true;
  const xb = cr + 1;
  return xb * xb + ci * ci <= 0.0625;
}

// value of the pixel whose centre is (px, py) pixels from the view centre
export function pt(px: f64, py: f64): f64 {
  if (gMode == 0 && gJulia) {
    const x = gAbsX + px * gScale, y = gAbsY - py * gScale;
    return ptDfrom(gJcr, gJci, x, y, x * x, y * y, 1, 0, 0, 0, 0, 8, 0, gMaxIter);
  }
  if (gMode == 0) return ptD(gAbsX + px * gScale, gAbsY - py * gScale, gMaxIter);
  return ptPB(px * gScale + gOx, -py * gScale + gOy, gMode == 2);
}
