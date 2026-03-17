/**
 * Shared build-time helpers for demo HTML generation.
 *
 * Demos inline the ax.global.js and ax-viz.global.js IIFE bundles so they
 * work as standalone file:// URLs without a server. Shared visualization
 * utilities are accessed via the `Ax.viz.*` namespace.
 */
import { readFileSync } from 'fs';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const root = join(__dirname, '..');

export const iifeBundle = readFileSync(join(root, 'dist/ax.global.js'), 'utf8');
export const vizBundle = readFileSync(join(root, 'dist/ax-viz.global.js'), 'utf8');
export const braninFixture = readFileSync(join(root, 'test/fixtures/branin_matern25.json'), 'utf8');
export const hartmannMixedFixture = readFileSync(join(root, 'test/fixtures/hartmann_mixed.json'), 'utf8');
export const penicillinFixture = readFileSync(join(root, 'test/fixtures/penicillin_modellist.json'), 'utf8');

/** Inline the main ax-js IIFE bundle. Exposes `window.Ax`. */
export function libraryScript() {
  return `<script>\n${iifeBundle}\n</script>`;
}

/** Inline the ax-viz IIFE bundle. Exposes `Ax.viz`. */
export function vizScript() {
  return `<script>\n${vizBundle}\n</script>`;
}

export function fixtureScript(varName, json) {
  return `<script>\nvar ${varName} = ${json};\n</script>`;
}

// Inline Ax logo SVG (white wireframe, links back to index)
export const axIconSvg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 1200" width="29" height="29" style="vertical-align:-4px;margin-right:8px;opacity:0.7"><path fill="#fff" d="M761.76,600h0l200-346.16H550.12l-400,692.32H961.76ZM573.41,274H926.82L750.12,579.85ZM550.12,926H185.06L555.94,284.07,738.47,600Zm23.29,0L750.12,620.15,926.82,926Z"/></svg>`;
export const axHomeLink = `<a href="index.html" style="text-decoration:none">${axIconSvg}</a>`;
