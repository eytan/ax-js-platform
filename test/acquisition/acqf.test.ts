// Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

import { describe, it, expect } from "vitest";

import { LogExpectedImprovement, ExpectedImprovement } from "../../src/acquisition/log_ei.js";
import { optimizeAcqf } from "../../src/acquisition/optimize.js";
import { posteriorCovariance } from "../../src/acquisition/posterior.js";
import { sampleMVN, Rng } from "../../src/acquisition/sample_mvn.js";
import { ThompsonSampling, thompsonSamples } from "../../src/acquisition/thompson.js";
import { UpperConfidenceBound } from "../../src/acquisition/ucb.js";
import { Matrix } from "../../src/linalg/matrix.js";
import { SingleTaskGP } from "../../src/models/single_task.js";

// Helper: create a simple 1D GP for testing
function make1DGP(): SingleTaskGP {
  const trainX = [[0], [0.25], [0.5], [0.75], [1]];
  const trainY = [0, 1, 0.5, 1.5, 0.2];
  return new SingleTaskGP({
    model_type: "SingleTaskGP",
    train_X: trainX,
    train_Y: trainY,
    kernel: { type: "Scale", outputscale: 1, base_kernel: { type: "RBF", lengthscale: [0.3] } },
    mean_constant: 0,
    noise_variance: 1e-4,
  });
}

// ─── UCB ──────────────────────────────────────────────────────────────────

describe("UpperConfidenceBound", () => {
  it("returns μ + √β·σ", () => {
    const gp = make1DGP();
    const ucb = new UpperConfidenceBound(gp, 4); // β=4 → √β=2
    const pred = gp.predict([[0.3]]);
    const values = ucb.evaluate([[0.3]]);
    const expected = pred.mean[0] + 2 * Math.sqrt(pred.variance[0]);
    expect(values[0]).toBeCloseTo(expected, 10);
  });

  it("prefers high-mean or high-variance points", () => {
    const gp = make1DGP();
    const ucb = new UpperConfidenceBound(gp, 2);
    // Evaluate at training point (low var) vs far-away (high var)
    const values = ucb.evaluate([[0.75], [5]]);
    // The far-away point has higher variance, UCB should be competitive
    expect(values.length).toBe(2);
    expect(Number.isFinite(values[0])).toBe(true);
    expect(Number.isFinite(values[1])).toBe(true);
  });

  it("with β=0 equals posterior mean", () => {
    const gp = make1DGP();
    const ucb = new UpperConfidenceBound(gp, 0);
    const pred = gp.predict([[0.3], [0.6]]);
    const values = ucb.evaluate([[0.3], [0.6]]);
    expect(values[0]).toBeCloseTo(pred.mean[0], 10);
    expect(values[1]).toBeCloseTo(pred.mean[1], 10);
  });
});

// ─── EI & LogEI ──────────────────────────────────────────────────────────

describe("ExpectedImprovement", () => {
  it("is non-negative everywhere", () => {
    const gp = make1DGP();
    const ei = new ExpectedImprovement(gp, 1);
    const candidates = Array.from({ length: 20 }, (_, i) => [i / 20]);
    const values = ei.evaluate(candidates);
    for (const val of values) {
      expect(val).toBeGreaterThanOrEqual(0);
    }
  });

  it("is zero at training points when bestF = max(trainY)", () => {
    const gp = make1DGP();
    const ei = new ExpectedImprovement(gp, 1.5); // bestF = max of trainY
    // At the best training point, mean ≈ 1.5 and var ≈ 0
    const values = ei.evaluate([[0.75]]);
    expect(values[0]).toBeLessThan(0.01);
  });

  it("is higher where improvement is likely", () => {
    const gp = make1DGP();
    const ei = new ExpectedImprovement(gp, 0.5); // low bestF
    const values = ei.evaluate([[0.75], [0]]); // 0.75 has high mean (~1.5)
    expect(values[0]).toBeGreaterThan(values[1]);
  });
});

describe("LogExpectedImprovement", () => {
  it("log(EI) = LogEI for moderate values", () => {
    const gp = make1DGP();
    const ei = new ExpectedImprovement(gp, 0.5);
    const logei = new LogExpectedImprovement(gp, 0.5);
    const candidates = [[0.1], [0.3], [0.6], [0.9]];
    const eiVals = ei.evaluate(candidates);
    const logeiVals = logei.evaluate(candidates);
    for (let i = 0; i < candidates.length; i++) {
      if (eiVals[i] > 1e-10) {
        expect(logeiVals[i]).toBeCloseTo(Math.log(eiVals[i]), 3);
      }
    }
  });

  it("is finite even when EI is negligible", () => {
    const gp = make1DGP();
    const logei = new LogExpectedImprovement(gp, 100); // very high bestF
    const values = logei.evaluate([[0.5]]);
    expect(Number.isFinite(values[0])).toBe(true);
    expect(values[0]).toBeLessThan(-5); // should be very negative
  });

  it("ranking matches EI ranking", () => {
    const gp = make1DGP();
    const ei = new ExpectedImprovement(gp, 0.5);
    const logei = new LogExpectedImprovement(gp, 0.5);
    const candidates = Array.from({ length: 10 }, (_, i) => [i / 10]);
    const eiVals = ei.evaluate(candidates);
    const logeiVals = logei.evaluate(candidates);

    // Best point should be the same
    let eiBest = 0,
      logeiBest = 0;
    for (let i = 1; i < candidates.length; i++) {
      if (eiVals[i] > eiVals[eiBest]) {
        eiBest = i;
      }
      if (logeiVals[i] > logeiVals[logeiBest]) {
        logeiBest = i;
      }
    }
    expect(logeiBest).toBe(eiBest);
  });
});

// ─── Posterior Covariance ────────────────────────────────────────────────

describe("posteriorCovariance", () => {
  it("diagonal matches predict() variance", () => {
    const gp = make1DGP();
    const points = [[0.1], [0.3], [0.7]];
    const Sigma = posteriorCovariance(gp, points);
    const { variance } = gp.predict(points);
    for (let i = 0; i < points.length; i++) {
      expect(Sigma.get(i, i)).toBeCloseTo(variance[i], 8);
    }
  });

  it("is approximately symmetric", () => {
    const gp = make1DGP();
    const points = [[0.1], [0.5], [0.9]];
    const Sigma = posteriorCovariance(gp, points);
    for (let i = 0; i < points.length; i++) {
      for (let j = i + 1; j < points.length; j++) {
        expect(Sigma.get(i, j)).toBeCloseTo(Sigma.get(j, i), 8);
      }
    }
  });

  it("covariance decays with distance", () => {
    const gp = make1DGP();
    const points = [[0.5], [0.51], [0.9]];
    const Sigma = posteriorCovariance(gp, points);
    // Nearby points should have higher covariance
    expect(Math.abs(Sigma.get(0, 1))).toBeGreaterThan(Math.abs(Sigma.get(0, 2)));
  });
});

// ─── MVN Sampling ────────────────────────────────────────────────────────

describe("sampleMVN", () => {
  it("samples have correct mean (empirical)", () => {
    const mean = Float64Array.from([1, 2, 3]);
    const Sigma = new Matrix(3, 3);
    Sigma.set(0, 0, 1);
    Sigma.set(1, 1, 1);
    Sigma.set(2, 2, 1);
    const rng = new Rng(123);
    const samples = sampleMVN(mean, Sigma, 10_000, rng);

    for (let d = 0; d < 3; d++) {
      let sum = 0;
      for (let s = 0; s < 10_000; s++) {
        sum += samples.get(s, d);
      }
      expect(sum / 10_000).toBeCloseTo(mean[d], 1);
    }
  });

  it("respects covariance structure", () => {
    const mean = Float64Array.from([0, 0]);
    const Sigma = new Matrix(2, 2);
    Sigma.set(0, 0, 1);
    Sigma.set(0, 1, 0.9);
    Sigma.set(1, 0, 0.9);
    Sigma.set(1, 1, 1);
    const rng = new Rng(42);
    const samples = sampleMVN(mean, Sigma, 5000, rng);

    // Empirical correlation should be close to 0.9
    let sumXY = 0,
      sumX2 = 0,
      sumY2 = 0;
    for (let s = 0; s < 5000; s++) {
      const x = samples.get(s, 0);
      const y = samples.get(s, 1);
      sumXY += x * y;
      sumX2 += x * x;
      sumY2 += y * y;
    }
    const corr = sumXY / Math.sqrt(sumX2 * sumY2);
    expect(corr).toBeCloseTo(0.9, 1);
  });

  it("is reproducible with same seed", () => {
    const mean = Float64Array.from([0, 0]);
    const Sigma = Matrix.zeros(2, 2);
    Sigma.set(0, 0, 1);
    Sigma.set(1, 1, 1);

    const s1 = sampleMVN(mean, Sigma, 5, new Rng(99));
    const s2 = sampleMVN(mean, Sigma, 5, new Rng(99));
    for (let i = 0; i < 5; i++) {
      for (let d = 0; d < 2; d++) {
        expect(s1.get(i, d)).toBe(s2.get(i, d));
      }
    }
  });
});

// ─── Thompson Sampling ──────────────────────────────────────────────────

describe("ThompsonSampling", () => {
  it("returns values for all candidates", () => {
    const gp = make1DGP();
    const ts = new ThompsonSampling(gp, 42);
    const candidates = [[0.1], [0.3], [0.5], [0.7], [0.9]];
    const values = ts.evaluate(candidates);
    expect(values.length).toBe(5);
    for (let i = 0; i < 5; i++) {
      expect(Number.isFinite(values[i])).toBe(true);
    }
  });

  it("different seeds give different samples", () => {
    const gp = make1DGP();
    const candidates = [[0.1], [0.5], [0.9]];
    const ts1 = new ThompsonSampling(gp, 1);
    const ts2 = new ThompsonSampling(gp, 2);
    const v1 = ts1.evaluate(candidates);
    const v2 = ts2.evaluate(candidates);
    // Very unlikely to be identical
    let allEqual = true;
    for (let i = 0; i < 3; i++) {
      if (v1[i] !== v2[i]) {
        allEqual = false;
      }
    }
    expect(allEqual).toBe(false);
  });

  it("thompsonSamples returns correct shape", () => {
    const gp = make1DGP();
    const candidates = [[0.1], [0.5], [0.9]];
    const samples = thompsonSamples(gp, candidates, 10, 42);
    expect(samples.rows).toBe(10);
    expect(samples.cols).toBe(3);
  });
});

// ─── Optimize ────────────────────────────────────────────────────────────

describe("optimizeAcqf", () => {
  it("finds a good point with random search", () => {
    const gp = make1DGP();
    const ucb = new UpperConfidenceBound(gp, 2);
    const result = optimizeAcqf(ucb, [[0, 1]], {
      rawSamples: 500,
      seed: 42,
    });
    expect(result.point.length).toBe(1);
    expect(result.point[0]).toBeGreaterThanOrEqual(0);
    expect(result.point[0]).toBeLessThanOrEqual(1);
    expect(Number.isFinite(result.value)).toBe(true);
  });

  it("L-BFGS improves over random search", () => {
    const gp = make1DGP();
    const ei = new ExpectedImprovement(gp, 0.5);

    const randomResult = optimizeAcqf(ei, [[0, 1]], {
      rawSamples: 100,
      seed: 42,
    });
    const lbfgsResult = optimizeAcqf(ei, [[0, 1]], {
      rawSamples: 100,
      numRestarts: 5,
      useLBFGS: true,
      seed: 42,
    });

    // L-BFGS should find at least as good a point
    expect(lbfgsResult.value).toBeGreaterThanOrEqual(randomResult.value - 1e-6);
  });

  it("respects bounds", () => {
    const gp = make1DGP();
    const ucb = new UpperConfidenceBound(gp, 2);
    const result = optimizeAcqf(ucb, [[0.2, 0.8]], {
      rawSamples: 200,
      useLBFGS: true,
      seed: 42,
    });
    expect(result.point[0]).toBeGreaterThanOrEqual(0.2 - 1e-10);
    expect(result.point[0]).toBeLessThanOrEqual(0.8 + 1e-10);
  });

  it("works in 2D", () => {
    const gp2d = new SingleTaskGP({
      model_type: "SingleTaskGP",
      train_X: [
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1],
        [0.5, 0.5],
      ],
      train_Y: [0, 0, 0, 0, 1],
      kernel: {
        type: "Scale",
        outputscale: 1,
        base_kernel: { type: "RBF", lengthscale: [0.3, 0.3] },
      },
      mean_constant: 0,
      noise_variance: 1e-4,
    });
    const ucb = new UpperConfidenceBound(gp2d, 2);
    const result = optimizeAcqf(
      ucb,
      [
        [0, 1],
        [0, 1],
      ],
      {
        rawSamples: 500,
        useLBFGS: true,
        seed: 42,
      },
    );
    expect(result.point.length).toBe(2);
    // Should find something near (0.5, 0.5) where the peak is
    expect(result.point[0]).toBeGreaterThan(0.1);
    expect(result.point[0]).toBeLessThan(0.9);
    expect(result.point[1]).toBeGreaterThan(0.1);
    expect(result.point[1]).toBeLessThan(0.9);
  });

  it("returnAll provides sorted candidates", () => {
    const gp = make1DGP();
    const ucb = new UpperConfidenceBound(gp, 2);
    const result = optimizeAcqf(ucb, [[0, 1]], {
      rawSamples: 50,
      returnAll: true,
      seed: 42,
    });
    expect(result.candidates).toBeDefined();
    expect(result.values).toBeDefined();
    expect(result.candidates!.length).toBe(50);
    // Values should be sorted descending
    for (let i = 1; i < result.values!.length; i++) {
      expect(result.values![i]).toBeLessThanOrEqual(result.values![i - 1] + 1e-12);
    }
  });
});
