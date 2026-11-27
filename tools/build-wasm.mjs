// Builds the WebAssembly kernels from wasm/mandel.ts and embeds them in mandelbrot/index.html.
//
// Two builds are made from the same source:
//   - SIMD build (128-bit SIMD, two pixels per instruction)      -> const WASM_SIMD_B64
//   - scalar build (the SIMD section removed), for browsers
//     without wasm SIMD                                          -> const WASM_B64
// Usage: npm install && npm run build:wasm
import { execFileSync } from 'node:child_process';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const src = join(root, 'wasm', 'mandel.ts');
const out = join(root, 'wasm', 'build');
const page = join(root, 'mandelbrot', 'index.html');
mkdirSync(out, { recursive: true });

// scalar variant: drop everything from the SIMD marker up to inMain()
const full = readFileSync(src, 'utf8');
const a = full.indexOf('// ===================== SIMD');
const b = full.indexOf('function inMain');
if (a < 0 || b < 0) throw new Error('SIMD section markers not found in wasm/mandel.ts');
const scalarSrc = join(out, 'mandel_nosimd.ts');
writeFileSync(scalarSrc, full.slice(0, a) + full.slice(b));

const asc = join(root, 'node_modules', '.bin', 'asc');
const common = ['-O3', '--noAssert', '--runtime', 'stub'];
execFileSync(asc, [src, ...common, '--enable', 'simd', '-o', join(out, 'mandel_simd.wasm')], { stdio: 'inherit' });
execFileSync(asc, [scalarSrc, ...common, '-o', join(out, 'mandel.wasm')], { stdio: 'inherit' });

let html = readFileSync(page, 'utf8');
for (const [name, file] of [['WASM_B64', 'mandel.wasm'], ['WASM_SIMD_B64', 'mandel_simd.wasm']]) {
  const b64 = readFileSync(join(out, file)).toString('base64');
  const re = new RegExp(`const ${name} = "[^"]*";`);
  if (!re.test(html)) throw new Error(`${name} not found in mandelbrot/index.html`);
  html = html.replace(re, `const ${name} = "${b64}";`);
  console.log(`${name}: ${file} (${Buffer.from(b64, 'base64').length} bytes)`);
}
writeFileSync(page, html);
