// Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

import { describe, it, expect } from "vitest";

import {
  relativize,
  unrelativize,
  relativizePredictions,
} from "../../src/transforms/relativize.js";

describe("relativize", () => {
  it("status quo vs itself returns (0, 0)", () => {
    const r = relativize(5, 0.1, 5, 0.1);
    expect(r.mean).toBe(0);
    expect(r.sem).toBe(0);
  });

  it("computes percentage change (default: asPercent=true)", () => {
    // Test arm: mean=110, SEM=0
    // Control arm: mean=100, SEM=0
    // Expected: (110-100)/100 = 10%
    const r = relativize(110, 0, 100, 0, { biasCorrection: false });
    expect(r.mean).toBeCloseTo(10, 10); // 10%
    expect(r.sem).toBeCloseTo(0, 10);
  });

  it("handles negative control mean using absolute value", () => {
    // Control mean=-100, test mean=-80
    // (−80 − (−100)) / |−100| = 20/100 = 0.2 → 20%
    const r = relativize(-80, 0, -100, 0, { biasCorrection: false });
    expect(r.mean).toBeCloseTo(20, 10);
  });

  it("throws on zero control mean", () => {
    expect(() => relativize(10, 1, 0, 1)).toThrow(/control mean is zero/);
  });

  it("propagates variance via delta method", () => {
    // Test: mean=150, SEM=10; Control: mean=100, SEM=5
    // c = 150/100 = 1.5
    // Var = (10² - 2*1.5*0 + 1.5²*5²) / 100² = (100 + 56.25) / 10000 = 0.015625
    // SEM = sqrt(0.015625) = 0.125 → as percent: 12.5%
    const r = relativize(150, 10, 100, 5, { biasCorrection: false });
    expect(r.sem).toBeCloseTo(12.5, 8);
  });

  it("bias correction adjusts mean", () => {
    const noBias = relativize(150, 10, 100, 5, { biasCorrection: false });
    const withBias = relativize(150, 10, 100, 5, { biasCorrection: true });
    // Bias correction subtracts m_t * s_c² / |m_c|³ = 150 * 25 / 1e6 = 0.00375
    // As percent: 0.375%
    expect(withBias.mean).toBeLessThan(noBias.mean);
    expect(noBias.mean - withBias.mean).toBeCloseTo(0.375, 6);
  });

  it("controlAsConstant ignores control SEM", () => {
    const r = relativize(150, 10, 100, 5, {
      biasCorrection: false,
      controlAsConstant: true,
    });
    // SEM = semT / |meanC| = 10/100 = 0.1 → as percent: 10%
    expect(r.sem).toBeCloseTo(10, 10);
  });

  it("asPercent=false returns fractional values", () => {
    const r = relativize(110, 0, 100, 0, {
      biasCorrection: false,
      asPercent: false,
    });
    expect(r.mean).toBeCloseTo(0.1, 10); // 10% as fraction
  });

  it("covariance reduces variance when positive", () => {
    const noCov = relativize(150, 10, 100, 5, {
      biasCorrection: false,
      covariance: 0,
    });
    const withCov = relativize(150, 10, 100, 5, {
      biasCorrection: false,
      covariance: 20, // positive covariance
    });
    expect(withCov.sem).toBeLessThan(noCov.sem);
  });
});

describe("unrelativize", () => {
  it("inverts relativize (no bias correction)", () => {
    const meanC = 100,
      semC = 5;
    const meanT = 150,
      semT = 10;
    const rel = relativize(meanT, semT, meanC, semC, { biasCorrection: false });
    const abs = unrelativize(rel.mean, rel.sem, meanC, semC, {
      biasCorrection: false,
    });
    expect(abs.mean).toBeCloseTo(meanT, 8);
    expect(abs.sem).toBeCloseTo(semT, 8);
  });

  it("inverts relativize (with bias correction)", () => {
    const meanC = 100,
      semC = 5;
    const meanT = 150,
      semT = 10;
    const rel = relativize(meanT, semT, meanC, semC, { biasCorrection: true });
    const abs = unrelativize(rel.mean, rel.sem, meanC, semC, {
      biasCorrection: true,
    });
    expect(abs.mean).toBeCloseTo(meanT, 6);
    expect(abs.sem).toBeCloseTo(semT, 6);
  });

  it("inverts relativize (constant control)", () => {
    const meanC = 100,
      semC = 5;
    const meanT = 150,
      semT = 10;
    const rel = relativize(meanT, semT, meanC, semC, {
      controlAsConstant: true,
    });
    const abs = unrelativize(rel.mean, rel.sem, meanC, semC, {
      controlAsConstant: true,
    });
    expect(abs.mean).toBeCloseTo(meanT, 8);
    expect(abs.sem).toBeCloseTo(semT, 8);
  });
});

describe("relativizePredictions", () => {
  it("batch relativizes predictions vs status quo", () => {
    const means = new Float64Array([110, 120, 130]);
    const variances = new Float64Array([4, 9, 16]); // SEMs: 2, 3, 4
    const sqMean = 100,
      sqVariance = 1; // SEM: 1

    const result = relativizePredictions(means, variances, sqMean, sqVariance, {
      biasCorrection: false,
    });

    // Expected relative means: 10%, 20%, 30%
    expect(result.mean[0]).toBeCloseTo(10, 6);
    expect(result.mean[1]).toBeCloseTo(20, 6);
    expect(result.mean[2]).toBeCloseTo(30, 6);
    expect(result.mean.length).toBe(3);
    expect(result.variance.length).toBe(3);
  });

  it("returns Float64Arrays", () => {
    const result = relativizePredictions([100], [1], 100, 1);
    expect(result.mean).toBeInstanceOf(Float64Array);
    expect(result.variance).toBeInstanceOf(Float64Array);
  });
});
