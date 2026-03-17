import { describe, it, expect } from "vitest";
import {
  normalPdf,
  normalCdf,
  logNormalPdf,
  logNormalCdf,
} from "../../src/acquisition/normal.js";

describe("normalPdf", () => {
  it("peaks at x=0 with value 1/√(2π)", () => {
    expect(normalPdf(0)).toBeCloseTo(1 / Math.sqrt(2 * Math.PI), 10);
  });

  it("is symmetric", () => {
    expect(normalPdf(1.5)).toBeCloseTo(normalPdf(-1.5), 12);
  });

  it("matches known values", () => {
    expect(normalPdf(1)).toBeCloseTo(0.24197072451914337, 10);
    expect(normalPdf(2)).toBeCloseTo(0.05399096651318806, 10);
  });
});

describe("normalCdf", () => {
  it("Φ(0) = 0.5", () => {
    expect(normalCdf(0)).toBeCloseTo(0.5, 12);
  });

  it("Φ(-∞) → 0, Φ(+∞) → 1", () => {
    expect(normalCdf(-10)).toBeLessThan(1e-15);
    expect(normalCdf(10)).toBeGreaterThan(1 - 1e-15);
  });

  it("matches known values", () => {
    expect(normalCdf(1)).toBeCloseTo(0.8413447460685429, 6);
    expect(normalCdf(-1)).toBeCloseTo(0.15865525393145702, 6);
    expect(normalCdf(1.96)).toBeCloseTo(0.975, 3);
  });

  it("Φ(x) + Φ(-x) ≈ 1", () => {
    for (const x of [0.5, 1, 2, 3]) {
      expect(normalCdf(x) + normalCdf(-x)).toBeCloseTo(1, 7);
    }
  });
});

describe("logNormalPdf", () => {
  it("equals log of normalPdf", () => {
    for (const x of [-3, -1, 0, 1, 3]) {
      expect(logNormalPdf(x)).toBeCloseTo(Math.log(normalPdf(x)), 10);
    }
  });
});

describe("logNormalCdf", () => {
  it("equals log of normalCdf in the normal range", () => {
    for (const x of [-3, -1, 0, 1, 3]) {
      expect(logNormalCdf(x)).toBeCloseTo(Math.log(normalCdf(x)), 5);
    }
  });

  it("is finite for large negative x (tail stability)", () => {
    const val = logNormalCdf(-10);
    expect(Number.isFinite(val)).toBe(true);
    expect(val).toBeLessThan(-40); // log(Φ(-10)) ≈ -53
  });

  it("is approximately 0 for large positive x", () => {
    expect(logNormalCdf(10)).toBeCloseTo(0, 5);
  });

  it("is monotonically increasing", () => {
    const xs = [-8, -5, -2, 0, 2, 5, 8];
    for (let i = 1; i < xs.length; i++) {
      expect(logNormalCdf(xs[i])).toBeGreaterThan(logNormalCdf(xs[i - 1]));
    }
  });
});
