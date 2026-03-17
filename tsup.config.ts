import { defineConfig } from "tsup";

export default defineConfig([
  {
    entry: ["src/index.ts"],
    format: ["esm", "cjs"],
    dts: true,
    sourcemap: true,
    clean: true,
  },
  {
    entry: ["src/index.ts"],
    format: ["iife"],
    globalName: "axjs",
    outDir: "dist",
    sourcemap: false,
    clean: false,
  },
  {
    entry: ["src/acquisition/index.ts"],
    format: ["esm", "cjs"],
    dts: true,
    outDir: "dist/acquisition",
    clean: false,
  },
]);
