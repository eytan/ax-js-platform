import { defineConfig } from "tsup";

export default defineConfig([
  // ── Main library (ESM + CJS + types) ────────────────────────────────────
  {
    entry: ["src/index.ts"],
    format: ["esm", "cjs"],
    dts: true,
    sourcemap: true,
    clean: true,
  },
  // ── Main library IIFE — <script src="ax.global.js"> → window.Ax ────────
  {
    entry: { "ax": "src/index.ts" },
    format: ["iife"],
    globalName: "Ax",
    outDir: "dist",
    outExtension: () => ({ js: ".global.js" }),
    sourcemap: false,
    clean: false,
  },
  // ── Acquisition (ESM + CJS + types) ─────────────────────────────────────
  {
    entry: ["src/acquisition/index.ts"],
    format: ["esm", "cjs"],
    dts: true,
    outDir: "dist/acquisition",
    clean: false,
  },
  // ── Acquisition IIFE — extends window.Ax.acquisition ────────────────────
  {
    entry: { "ax-acquisition": "src/acquisition/index.ts" },
    format: ["iife"],
    globalName: "Ax.acquisition",
    outDir: "dist",
    outExtension: () => ({ js: ".global.js" }),
    sourcemap: false,
    clean: false,
  },
  // ── Viz (ESM + CJS + types) ─────────────────────────────────────────────
  {
    entry: ["src/viz/index.ts"],
    format: ["esm", "cjs"],
    dts: true,
    outDir: "dist/viz",
    clean: false,
  },
  // ── Viz IIFE — extends window.Ax.viz ────────────────────────────────────
  {
    entry: { "ax-viz": "src/viz/index.ts" },
    format: ["iife"],
    globalName: "Ax.viz",
    outDir: "dist",
    outExtension: () => ({ js: ".global.js" }),
    sourcemap: false,
    clean: false,
  },
]);
