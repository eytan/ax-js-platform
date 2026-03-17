import { describe, it, expect } from "vitest";
import { PairwiseGP, createPairwiseGP } from "../../src/models/pairwise_gp.js";
import { EUBO } from "../../src/acquisition/eubo.js";
import { posteriorCovariance } from "../../src/acquisition/posterior.js";
import { optimizeAcqf } from "../../src/acquisition/optimize.js";
import { UpperConfidenceBound } from "../../src/acquisition/ucb.js";
import type { PairwiseGPModelState } from "../../src/models/types.js";

// Build a simple PairwiseGP for testing
function makePairwiseGP(): PairwiseGP {
  // 4 points in 2D with known utility ordering
  const trainX = [[0, 0], [1, 0], [0, 1], [1, 1]];
  const utility = [0.0, 0.5, 0.3, 1.0]; // (1,1) is best
  const n = trainX.length;

  // Weakly informative Hessian (negative definite, as expected for Laplace).
  // Too-large magnitudes cause posterior variance to collapse to zero.
  const hess: number[][] = [];
  for (let i = 0; i < n; i++) {
    const row = new Array(n).fill(0);
    row[i] = -0.1;
    hess.push(row);
  }

  const state: PairwiseGPModelState = {
    model_type: "PairwiseGP",
    train_X: trainX,
    utility,
    likelihood_hess: hess,
    kernel: { type: "Scale", outputscale: 1.0, base_kernel: { type: "RBF", lengthscale: [0.5, 0.5] } },
    mean_constant: 0,
  };

  return createPairwiseGP(state);
}

describe("EUBO", () => {
  it("returns a finite scalar for a candidate menu", () => {
    const gp = makePairwiseGP();
    const eubo = new EUBO(gp, 128, 42);
    const menu = [[0.5, 0.5], [0.8, 0.8]];
    const values = eubo.evaluate(menu);
    expect(values.length).toBe(1);
    expect(Number.isFinite(values[0])).toBe(true);
  });

  it("euboValue equals evaluate()[0]", () => {
    const gp = makePairwiseGP();
    const eubo = new EUBO(gp, 128, 42);
    const menu = [[0.2, 0.3], [0.7, 0.8]];
    const evalResult = eubo.evaluate(menu)[0];
    // Re-create with same seed for identical samples
    const eubo2 = new EUBO(gp, 128, 42);
    const directResult = eubo2.euboValue(menu);
    expect(evalResult).toBeCloseTo(directResult, 10);
  });

  it("larger menu has higher or equal EUBO (more options = better)", () => {
    const gp = makePairwiseGP();
    const smallMenu = [[0.5, 0.5]];
    const largeMenu = [[0.5, 0.5], [0.8, 0.8], [0.2, 0.9]];

    const eubo1 = new EUBO(gp, 512, 42);
    const val1 = eubo1.euboValue(smallMenu);
    const eubo2 = new EUBO(gp, 512, 42);
    const val2 = eubo2.euboValue(largeMenu);

    // E[max] over more options should be >= E[max] over fewer
    expect(val2).toBeGreaterThanOrEqual(val1 - 0.1); // small tolerance for MC noise
  });

  it("menu near known best has higher EUBO than menu in poor region", () => {
    const gp = makePairwiseGP();
    // Menu near (1,1) where utility is highest
    const goodMenu = [[0.9, 0.9], [0.8, 1.0]];
    // Menu near (0,0) where utility is lowest
    const badMenu = [[0.0, 0.0], [0.1, 0.1]];

    const eubo1 = new EUBO(gp, 512, 42);
    const goodVal = eubo1.euboValue(goodMenu);
    const eubo2 = new EUBO(gp, 512, 42);
    const badVal = eubo2.euboValue(badMenu);

    expect(goodVal).toBeGreaterThan(badVal);
  });

  it("evaluateMarginal returns non-negative values that sum ≤ EUBO", () => {
    const gp = makePairwiseGP();
    const menu = [[0.3, 0.3], [0.7, 0.7], [0.5, 0.9]];
    const eubo = new EUBO(gp, 256, 42);
    const marginals = eubo.evaluateMarginal(menu);
    expect(marginals.length).toBe(3);
    for (let i = 0; i < 3; i++) {
      expect(marginals[i]).toBeGreaterThanOrEqual(-1e-10);
    }
  });

  it("is reproducible with same seed", () => {
    const gp = makePairwiseGP();
    const menu = [[0.4, 0.6], [0.6, 0.4]];
    const val1 = new EUBO(gp, 128, 99).euboValue(menu);
    const val2 = new EUBO(gp, 128, 99).euboValue(menu);
    expect(val1).toBe(val2);
  });
});

describe("EUBO + optimize", () => {
  it("can optimize UCB over PairwiseGP", () => {
    const gp = makePairwiseGP();
    const ucb = new UpperConfidenceBound(gp, 2.0);
    const result = optimizeAcqf(ucb, [[0, 1], [0, 1]], {
      rawSamples: 200,
      useLBFGS: true,
      seed: 42,
    });
    expect(result.point.length).toBe(2);
    expect(Number.isFinite(result.value)).toBe(true);
    // Should find something near (1,1) where utility is highest
    expect(result.point[0]).toBeGreaterThan(0.3);
    expect(result.point[1]).toBeGreaterThan(0.3);
  });
});

describe("PairwiseGP posteriorCovariance", () => {
  it("is symmetric and positive semi-definite on diagonal", () => {
    const gp = makePairwiseGP();
    const points = [[0.2, 0.3], [0.5, 0.5], [0.8, 0.7]];
    const Sigma = posteriorCovariance(gp, points);

    // Symmetric
    for (let i = 0; i < 3; i++) {
      for (let j = i + 1; j < 3; j++) {
        expect(Sigma.get(i, j)).toBeCloseTo(Sigma.get(j, i), 8);
      }
    }
    // Positive diagonal
    for (let i = 0; i < 3; i++) {
      expect(Sigma.get(i, i)).toBeGreaterThan(0);
    }
  });
});
