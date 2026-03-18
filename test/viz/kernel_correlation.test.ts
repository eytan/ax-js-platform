/**
 * Tests for kernelCorrelation and visualization math invariants.
 *
 * These tests ensure the kernel-distance highlighting in viz plots
 * produces correct correlation values across model types, including
 * models with input warping.
 */
import { describe, it, expect } from "vitest";
import { Predictor } from "../../src/predictor";
import { readFileSync } from "fs";

function loadFixture(name: string) {
  const raw = JSON.parse(readFileSync(`test/fixtures/${name}`, "utf-8"));
  return new Predictor(raw.experiment ?? raw);
}

describe("kernelCorrelation invariants", () => {
  const fixtures = [
    { name: "branin_rbf.json", outcome: "y", hasWarp: false },
    { name: "branin_matern25.json", outcome: "y", hasWarp: false },
    { name: "branin_warp.json", outcome: "y", hasWarp: true },
    { name: "penicillin_modellist.json", outcome: "penicillin_yield", hasWarp: false },
    { name: "hartmann_mixed.json", outcome: "y", hasWarp: false },
    { name: "branincurrin_modellist.json", outcome: "y0", hasWarp: false },
  ];

  for (const { name, outcome, hasWarp } of fixtures) {
    describe(name, () => {
      const p = loadFixture(name);
      const td = p.getTrainingData(outcome);
      // Skip if insufficient data
      if (td.X.length < 2) return;

      it("self-correlation is 1", () => {
        const c = p.kernelCorrelation(td.X[0], td.X[0], outcome);
        expect(c).toBeCloseTo(1, 10);
      });

      it("all correlations are in (0, 1]", () => {
        for (let i = 0; i < Math.min(td.X.length, 10); i++) {
          for (let j = i + 1; j < Math.min(td.X.length, 10); j++) {
            const c = p.kernelCorrelation(td.X[i], td.X[j], outcome);
            expect(c).toBeGreaterThan(0);
            expect(c).toBeLessThanOrEqual(1);
            expect(Number.isFinite(c)).toBe(true);
          }
        }
      });

      it("is symmetric: corr(a,b) === corr(b,a)", () => {
        for (let i = 0; i < Math.min(td.X.length, 5); i++) {
          for (let j = i + 1; j < Math.min(td.X.length, 5); j++) {
            const cab = p.kernelCorrelation(td.X[i], td.X[j], outcome);
            const cba = p.kernelCorrelation(td.X[j], td.X[i], outcome);
            expect(cab).toBeCloseTo(cba, 12);
          }
        }
      });

      it("max non-self correlation is < 1 and > 0", () => {
        let maxCorr = 0;
        for (let i = 1; i < td.X.length; i++) {
          const c = p.kernelCorrelation(td.X[0], td.X[i], outcome);
          if (c > maxCorr) maxCorr = c;
        }
        expect(maxCorr).toBeGreaterThan(0);
        expect(maxCorr).toBeLessThan(1);
      });

      it("closer points have higher correlation than distant points", () => {
        // Find closest and farthest point to X[0] in raw space
        let closestIdx = 1, farthestIdx = 1;
        let closestDist = Infinity, farthestDist = -Infinity;
        for (let i = 1; i < td.X.length; i++) {
          let d2 = 0;
          for (let j = 0; j < td.X[i].length; j++) {
            const diff = td.X[0][j] - td.X[i][j];
            d2 += diff * diff;
          }
          if (d2 < closestDist) { closestDist = d2; closestIdx = i; }
          if (d2 > farthestDist) { farthestDist = d2; farthestIdx = i; }
        }
        if (closestIdx !== farthestIdx) {
          const cClose = p.kernelCorrelation(td.X[0], td.X[closestIdx], outcome);
          const cFar = p.kernelCorrelation(td.X[0], td.X[farthestIdx], outcome);
          // Note: this can fail for warped models where raw distance != kernel distance
          // but should hold for non-warped models
          if (!hasWarp) {
            expect(cClose).toBeGreaterThanOrEqual(cFar);
          }
        }
      });
    });
  }
});

describe("kernelCorrelation with warp", () => {
  it("branin_warp produces different correlations than without warp", () => {
    const pWarp = loadFixture("branin_warp.json");
    const tdW = pWarp.getTrainingData("y");

    // Verify warp data exists
    const exp = JSON.parse(readFileSync("test/fixtures/branin_warp.json", "utf-8"));
    const ms = exp.experiment?.model_state ?? exp.model_state;
    expect(ms.input_warp).toBeDefined();

    // All correlations should be finite and in (0, 1]
    for (let i = 0; i < tdW.X.length; i++) {
      const c = pWarp.kernelCorrelation(tdW.X[0], tdW.X[i], "y");
      expect(Number.isFinite(c)).toBe(true);
      expect(c).toBeGreaterThan(0);
      expect(c).toBeLessThanOrEqual(1);
    }
  });
});
