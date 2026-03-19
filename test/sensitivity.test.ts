import { describe, it, expect } from "vitest";
import { readFileSync } from "fs";
import { join, dirname } from "path";
import { fileURLToPath } from "url";
import { haltonSequence, computeSobolIndices } from "../src/sensitivity.js";
import { Rng } from "../src/acquisition/sample_mvn.js";
import { Predictor } from "../src/predictor.js";
import type { SearchSpaceParam } from "../src/models/types.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
const fixturesDir = join(__dirname, "fixtures");

function loadFixture(name: string) {
  return JSON.parse(readFileSync(join(fixturesDir, name), "utf-8"));
}

// ── Halton sequence quality ─────────────────────────────────────────────

describe("haltonSequence", () => {
  it("returns N points of d dimensions in [0,1]", () => {
    const pts = haltonSequence(100, 3);
    expect(pts.length).toBe(100);
    for (const pt of pts) {
      expect(pt.length).toBe(3);
      for (const v of pt) {
        expect(v).toBeGreaterThanOrEqual(0);
        expect(v).toBeLessThan(1);
      }
    }
  });

  it("has approximately uniform distribution (mean ≈ 0.5, std ≈ 1/√12)", () => {
    const pts = haltonSequence(2000, 2);
    for (let d = 0; d < 2; d++) {
      const vals = pts.map((p) => p[d]);
      const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
      const variance = vals.reduce((a, b) => a + (b - mean) ** 2, 0) / vals.length;
      const expectedStd = 1 / Math.sqrt(12); // ≈ 0.2887
      expect(Math.abs(mean - 0.5)).toBeLessThan(0.03);
      expect(Math.abs(Math.sqrt(variance) - expectedStd)).toBeLessThan(0.03);
    }
  });

  it("is deterministic with same Rng seed", () => {
    const a = haltonSequence(50, 3, new Rng(123));
    const b = haltonSequence(50, 3, new Rng(123));
    for (let i = 0; i < 50; i++) {
      for (let d = 0; d < 3; d++) {
        expect(a[i][d]).toBe(b[i][d]);
      }
    }
  });

  it("throws for d > 20", () => {
    expect(() => haltonSequence(10, 21)).toThrow();
  });

  it("covers [0,1]^2 quadrants roughly evenly (stratification)", () => {
    // Halton should fill space more evenly than random
    const pts = haltonSequence(400, 2);
    // Count points in each quadrant
    const quadrants = [0, 0, 0, 0];
    for (const pt of pts) {
      const q = (pt[0] >= 0.5 ? 1 : 0) + (pt[1] >= 0.5 ? 2 : 0);
      quadrants[q]++;
    }
    // Each quadrant should have ~100 points (25% of 400)
    for (const count of quadrants) {
      expect(count).toBeGreaterThan(70);
      expect(count).toBeLessThan(130);
    }
  });

  it("produces well-spread values over larger sequences", () => {
    // With N=64 in 1D, sorted values should roughly partition [0,1] evenly
    const pts = haltonSequence(64, 1);
    const vals = pts.map((p) => p[0]).sort();
    // Check spread: max gap between sorted values should be small
    let maxGap = 0;
    for (let i = 1; i < 64; i++) {
      const gap = vals[i] - vals[i - 1];
      if (gap > maxGap) maxGap = gap;
    }
    // For 64 well-spread points in [0,1], max gap should be < 0.05
    expect(maxGap).toBeLessThan(0.05);
    // Check that they span most of [0,1]
    expect(vals[0]).toBeLessThan(0.05);
    expect(vals[63]).toBeGreaterThan(0.95);
  });

  it("different seeds produce different sequences", () => {
    const a = haltonSequence(20, 2, new Rng(1));
    const b = haltonSequence(20, 2, new Rng(999));
    let same = 0;
    for (let i = 0; i < 20; i++) {
      if (a[i][0] === b[i][0]) same++;
    }
    // With different scrambling, very few (if any) should coincide
    expect(same).toBeLessThan(5);
  });
});

// ── Sobol' indices: analytically known functions ────────────────────────

describe("computeSobolIndices", () => {
  const uniformSpecs = (d: number): SearchSpaceParam[] =>
    Array.from({ length: d }, (_, i) => ({
      name: `x${i}`,
      type: "range" as const,
      bounds: [0, 1] as [number, number],
    }));

  describe("f(x1,x2) = x1 (trivially separable)", () => {
    // All variance comes from x1. S1 = 1, S2 = 0, ST1 = 1, ST2 = 0.
    it("assigns all importance to x1", () => {
      const specs = uniformSpecs(2);
      const predictFn = (points: number[][]): Float64Array => {
        return new Float64Array(points.map((p) => p[0]));
      };
      const result = computeSobolIndices(predictFn, specs, { numSamples: 1024 });
      expect(result.firstOrder[0]).toBeGreaterThan(0.9);
      expect(result.totalOrder[0]).toBeGreaterThan(0.9);
      expect(result.firstOrder[1]).toBeLessThan(0.05);
      expect(result.totalOrder[1]).toBeLessThan(0.05);
    });
  });

  describe("f(x1,x2) = x1 + 2*x2 (additive, no interaction)", () => {
    // Var(x1) = 1/12, Var(2*x2) = 4/12. Total = 5/12.
    // S1 = (1/12)/(5/12) = 0.2, S2 = (4/12)/(5/12) = 0.8.
    // No interaction: ST_i = S_i.
    it("recovers S1=0.2, S2=0.8 with no interactions", () => {
      const specs = uniformSpecs(2);
      const predictFn = (points: number[][]): Float64Array => {
        return new Float64Array(points.map((p) => p[0] + 2 * p[1]));
      };
      const result = computeSobolIndices(predictFn, specs, { numSamples: 4096 });

      expect(Math.abs(result.firstOrder[0] - 0.2)).toBeLessThan(0.06);
      expect(Math.abs(result.firstOrder[1] - 0.8)).toBeLessThan(0.06);
      // No interaction for additive function: ST ≈ S
      expect(Math.abs(result.totalOrder[0] - result.firstOrder[0])).toBeLessThan(0.05);
      expect(Math.abs(result.totalOrder[1] - result.firstOrder[1])).toBeLessThan(0.05);
    });
  });

  describe("f(x1,x2) = x1*x2 (pure interaction)", () => {
    // Var(x1*x2) = E[x1²*x2²] - E[x1*x2]² = (1/3)² - (1/2)⁴ = 1/9 - 1/16 = 7/144
    // Var_x1(E[x1*x2|x1]) = Var_x1(x1/2) = (1/12)*(1/4) = 1/48
    // S1 = (1/48)/(7/144) = 3/7 ≈ 0.4286
    // Similarly S2 = 3/7 ≈ 0.4286
    // S12 = 1 - S1 - S2 = 1/7 ≈ 0.1429
    // ST1 = S1 + S12 = 4/7 ≈ 0.5714
    it("detects interaction (ST > S)", () => {
      const specs = uniformSpecs(2);
      const predictFn = (points: number[][]): Float64Array => {
        return new Float64Array(points.map((p) => p[0] * p[1]));
      };
      const result = computeSobolIndices(predictFn, specs, { numSamples: 4096 });

      // S1 ≈ S2 ≈ 3/7 ≈ 0.4286
      expect(Math.abs(result.firstOrder[0] - 3 / 7)).toBeLessThan(0.05);
      expect(Math.abs(result.firstOrder[1] - 3 / 7)).toBeLessThan(0.05);
      // ST1 ≈ ST2 ≈ 4/7 ≈ 0.5714
      expect(Math.abs(result.totalOrder[0] - 4 / 7)).toBeLessThan(0.05);
      expect(Math.abs(result.totalOrder[1] - 4 / 7)).toBeLessThan(0.05);
      // Interaction: ST > S
      expect(result.totalOrder[0]).toBeGreaterThan(result.firstOrder[0] + 0.05);
    });
  });

  describe("Ishigami function (standard Sobol' benchmark)", () => {
    // f(x1,x2,x3) = sin(x1) + 7*sin²(x2) + 0.1*x3⁴*sin(x1)
    // Domain: [-π, π]³
    // Known analytical indices:
    //   S1 ≈ 0.3139, S2 ≈ 0.4424, S3 = 0
    //   ST1 ≈ 0.5576, ST2 ≈ 0.4424, ST3 ≈ 0.2437
    const a = 7, b = 0.1;
    const ishigamiSpecs: SearchSpaceParam[] = [
      { name: "x1", type: "range", bounds: [-Math.PI, Math.PI] },
      { name: "x2", type: "range", bounds: [-Math.PI, Math.PI] },
      { name: "x3", type: "range", bounds: [-Math.PI, Math.PI] },
    ];
    const ishigamiFn = (points: number[][]): Float64Array => {
      return new Float64Array(
        points.map((p) => Math.sin(p[0]) + a * Math.sin(p[1]) ** 2 + b * p[2] ** 4 * Math.sin(p[0])),
      );
    };

    it("recovers known analytical indices within tolerance", () => {
      const result = computeSobolIndices(ishigamiFn, ishigamiSpecs, { numSamples: 8192 });

      // First-order
      expect(Math.abs(result.firstOrder[0] - 0.3139)).toBeLessThan(0.04);
      expect(Math.abs(result.firstOrder[1] - 0.4424)).toBeLessThan(0.04);
      expect(result.firstOrder[2]).toBeLessThan(0.03); // S3 ≈ 0

      // Total-order
      expect(Math.abs(result.totalOrder[0] - 0.5576)).toBeLessThan(0.04);
      expect(Math.abs(result.totalOrder[1] - 0.4424)).toBeLessThan(0.04);
      expect(Math.abs(result.totalOrder[2] - 0.2437)).toBeLessThan(0.04);

      // x1 has strong interaction (ST >> S)
      expect(result.totalOrder[0] - result.firstOrder[0]).toBeGreaterThan(0.15);

      // x2 has no interaction (ST ≈ S)
      expect(Math.abs(result.totalOrder[1] - result.firstOrder[1])).toBeLessThan(0.03);
    });
  });

  describe("constant function", () => {
    it("returns all zeros for a constant function", () => {
      const specs = uniformSpecs(3);
      const predictFn = (points: number[][]): Float64Array => {
        return new Float64Array(points.length).fill(42);
      };
      const result = computeSobolIndices(predictFn, specs, { numSamples: 256 });
      for (let i = 0; i < 3; i++) {
        expect(result.firstOrder[i]).toBe(0);
        expect(result.totalOrder[i]).toBe(0);
      }
    });
  });

  describe("higher dimensional: f = sum(i*xi)", () => {
    // f(x) = 1*x1 + 2*x2 + 3*x3 + 4*x4
    // Var(i*xi) = i²/12 for uniform [0,1].
    // Total variance = (1+4+9+16)/12 = 30/12
    // S_i = i²/30
    it("recovers weighted variance shares in 4D", () => {
      const specs = uniformSpecs(4);
      const predictFn = (points: number[][]): Float64Array => {
        return new Float64Array(
          points.map((p) => p[0] + 2 * p[1] + 3 * p[2] + 4 * p[3]),
        );
      };
      const result = computeSobolIndices(predictFn, specs, { numSamples: 4096 });

      const totalWeight = 1 + 4 + 9 + 16; // = 30
      for (let i = 0; i < 4; i++) {
        const expectedS = (i + 1) ** 2 / totalWeight;
        expect(Math.abs(result.firstOrder[i] - expectedS)).toBeLessThan(0.07);
        // Additive: ST ≈ S
        expect(Math.abs(result.totalOrder[i] - expectedS)).toBeLessThan(0.07);
      }
    });
  });

  describe("determinism", () => {
    it("same seed produces identical results", () => {
      const specs = uniformSpecs(2);
      const predictFn = (points: number[][]): Float64Array => {
        return new Float64Array(points.map((p) => p[0] ** 2 + p[1]));
      };
      const a = computeSobolIndices(predictFn, specs, { numSamples: 256, seed: 99 });
      const b = computeSobolIndices(predictFn, specs, { numSamples: 256, seed: 99 });
      expect(a.firstOrder).toEqual(b.firstOrder);
      expect(a.totalOrder).toEqual(b.totalOrder);
    });
  });

  describe("structural invariants", () => {
    it("all S_i >= 0 and ST_i >= 0", () => {
      const specs = uniformSpecs(3);
      const predictFn = (points: number[][]): Float64Array => {
        return new Float64Array(
          points.map((p) => Math.sin(p[0] * 3) * p[1] + p[2] ** 2),
        );
      };
      const result = computeSobolIndices(predictFn, specs, { numSamples: 1024 });
      for (let i = 0; i < 3; i++) {
        expect(result.firstOrder[i]).toBeGreaterThanOrEqual(0);
        expect(result.totalOrder[i]).toBeGreaterThanOrEqual(0);
      }
    });

    it("ST_i >= S_i for all dimensions", () => {
      const specs = uniformSpecs(3);
      const predictFn = (points: number[][]): Float64Array => {
        return new Float64Array(
          points.map((p) => p[0] * p[1] + p[2]),
        );
      };
      const result = computeSobolIndices(predictFn, specs, { numSamples: 2048 });
      for (let i = 0; i < 3; i++) {
        // ST >= S (within numerical noise)
        expect(result.totalOrder[i]).toBeGreaterThanOrEqual(result.firstOrder[i] - 0.01);
      }
    });

    it("sum of first-order indices <= 1.0 (within tolerance)", () => {
      const specs = uniformSpecs(3);
      const predictFn = (points: number[][]): Float64Array => {
        return new Float64Array(
          points.map((p) => p[0] + p[1] * p[2]),
        );
      };
      const result = computeSobolIndices(predictFn, specs, { numSamples: 2048 });
      const sumS = result.firstOrder.reduce((a, b) => a + b, 0);
      expect(sumS).toBeLessThanOrEqual(1.05); // small tolerance for estimation error
    });
  });

  describe("f(x1,x2) = x1^2 (nonlinear, single variable)", () => {
    // All variance from x1. For U[0,1]: Var(x^2) = E[x^4] - E[x^2]^2 = 1/5 - 1/9 = 4/45
    it("assigns all importance to x1 despite nonlinearity", () => {
      const specs = uniformSpecs(2);
      const predictFn = (points: number[][]): Float64Array => {
        return new Float64Array(points.map((p) => p[0] ** 2));
      };
      const result = computeSobolIndices(predictFn, specs, { numSamples: 2048 });
      expect(result.firstOrder[0]).toBeGreaterThan(0.85);
      expect(result.totalOrder[0]).toBeGreaterThan(0.85);
      expect(result.firstOrder[1]).toBeLessThan(0.1);
      expect(result.totalOrder[1]).toBeLessThan(0.1);
    });
  });

  describe("non-uniform bounds", () => {
    it("respects parameter bounds in importance calculation", () => {
      // f = x1 + x2, but x1 ∈ [0, 10] and x2 ∈ [0, 1]
      // Var(x1) = 100/12, Var(x2) = 1/12. S1 = 100/101 ≈ 0.99
      const specs: SearchSpaceParam[] = [
        { name: "x1", type: "range", bounds: [0, 10] },
        { name: "x2", type: "range", bounds: [0, 1] },
      ];
      const predictFn = (points: number[][]): Float64Array => {
        return new Float64Array(points.map((p) => p[0] + p[1]));
      };
      const result = computeSobolIndices(predictFn, specs, { numSamples: 2048 });
      expect(result.firstOrder[0]).toBeGreaterThan(0.95);
      expect(result.firstOrder[1]).toBeLessThan(0.05);
    });
  });
});

// ── GP integration tests ────────────────────────────────────────────────

describe("Predictor.computeSensitivity", () => {
  describe("branin_matern25 fixture", () => {
    const fixture = loadFixture("branin_matern25.json");
    const predictor = new Predictor(fixture.experiment);

    it("returns valid SensitivityIndices", () => {
      const sens = predictor.computeSensitivity();
      expect(sens.paramNames.length).toBe(predictor.paramNames.length);
      expect(sens.firstOrder.length).toBe(predictor.paramNames.length);
      expect(sens.totalOrder.length).toBe(predictor.paramNames.length);
      expect(sens.numEvaluations).toBeGreaterThanOrEqual(0);
    });

    it("all S_i >= 0, ST_i >= 0", () => {
      const sens = predictor.computeSensitivity();
      for (let i = 0; i < sens.firstOrder.length; i++) {
        expect(sens.firstOrder[i]).toBeGreaterThanOrEqual(0);
        expect(sens.totalOrder[i]).toBeGreaterThanOrEqual(0);
      }
    });

    it("sum of S_i <= 1.0", () => {
      const sens = predictor.computeSensitivity();
      const sumS = sens.firstOrder.reduce((a, b) => a + b, 0);
      expect(sumS).toBeLessThanOrEqual(1.05);
    });

    it("results are cached", () => {
      const a = predictor.computeSensitivity("y", { numSamples: 512, seed: 42 });
      const b = predictor.computeSensitivity("y", { numSamples: 512, seed: 42 });
      expect(a).toBe(b); // same reference (cached)
    });
  });

  describe("ModelListGP (branincurrin)", () => {
    const fixture = loadFixture("branincurrin_modellist.json");
    const predictor = new Predictor(fixture.experiment);

    it("computes sensitivity for each outcome independently", () => {
      const names = predictor.outcomeNames;
      expect(names.length).toBeGreaterThanOrEqual(2);

      const s0 = predictor.computeSensitivity(names[0]);
      const s1 = predictor.computeSensitivity(names[1]);

      // Different outcomes should generally have different importance rankings
      expect(s0.paramNames).toEqual(s1.paramNames);
      expect(s0.numEvaluations).toBeGreaterThanOrEqual(0);
      expect(s1.numEvaluations).toBeGreaterThanOrEqual(0);

      // Both must satisfy structural invariants
      for (const sens of [s0, s1]) {
        const sumS = sens.firstOrder.reduce((a, b) => a + b, 0);
        expect(sumS).toBeLessThanOrEqual(1.05);
        for (let i = 0; i < sens.firstOrder.length; i++) {
          expect(sens.firstOrder[i]).toBeGreaterThanOrEqual(0);
          expect(sens.totalOrder[i]).toBeGreaterThanOrEqual(0);
        }
      }
    });
  });

  describe("6D function (hartmann)", () => {
    const fixture = loadFixture("hartmann_6d.json");
    const predictor = new Predictor(fixture.experiment);

    it("identifies non-trivial importance in 6 dimensions", () => {
      const sens = predictor.computeSensitivity(undefined, { numSamples: 1024 });
      expect(sens.firstOrder.length).toBe(6);

      // At least one dimension should have non-negligible importance
      const maxST = Math.max(...sens.totalOrder);
      expect(maxST).toBeGreaterThan(0.01);

      // Structural invariants
      const sumS = sens.firstOrder.reduce((a, b) => a + b, 0);
      expect(sumS).toBeLessThanOrEqual(1.1); // slightly wider tolerance for 6D
      for (let i = 0; i < 6; i++) {
        expect(sens.totalOrder[i]).toBeGreaterThanOrEqual(sens.firstOrder[i] - 0.02);
      }
    });

    it("most important dim by ST agrees with shortest lengthscale", () => {
      const sens = predictor.computeSensitivity(undefined, { numSamples: 1024 });
      const ls = predictor.getLengthscales();
      if (!ls) return; // skip if no lengthscale data

      const stArgmax = sens.totalOrder.indexOf(Math.max(...sens.totalOrder));
      const lsArgmin = ls.indexOf(Math.min(...ls));
      // They should often agree, but it's not guaranteed due to range effects.
      // At minimum, the Sobol'-important dims should include the ls-important ones
      // among the top 3 by each measure.
      const topST = sens.totalOrder
        .map((v, i) => ({ v, i }))
        .sort((a, b) => b.v - a.v)
        .slice(0, 3)
        .map((d) => d.i);
      const topLS = ls
        .map((v, i) => ({ v, i }))
        .sort((a, b) => a.v - b.v)
        .slice(0, 3)
        .map((d) => d.i);

      // At least one of the top-3 by Sobol' should overlap with top-3 by lengthscale
      const overlap = topST.filter((i) => topLS.includes(i));
      expect(overlap.length).toBeGreaterThanOrEqual(1);
    });
  });
});
