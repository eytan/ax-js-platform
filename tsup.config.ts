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
  // ── Main library — <script src="ax.js"> ────────────────────────────────
  {
    entry: { "ax": "src/index.ts" },
    format: ["iife"],
    globalName: "Ax",
    outDir: "dist",
    outExtension: () => ({ js: ".js" }),
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
  // ── Acquisition (experimental) — <script src="ax-acquisition.js"> ───────
  {
    entry: { "ax-acquisition": "src/acquisition/index.ts" },
    format: ["iife"],
    globalName: "Ax.acquisition",
    outDir: "dist",
    outExtension: () => ({ js: ".js" }),
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
  // ── Viz — <script src="ax-viz.js"> ──────────────────────────────────────
  {
    entry: { "ax-viz": "src/viz/index.ts" },
    format: ["iife"],
    globalName: "Ax.viz",
    outDir: "dist",
    outExtension: () => ({ js: ".js" }),
    sourcemap: false,
    clean: false,
  },
]);
