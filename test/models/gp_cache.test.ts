// Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

import { describe, it, expect } from "vitest";

import { RBFKernel } from "../../src/kernels/rbf.js";
import { ScaleKernel } from "../../src/kernels/scale.js";
import { Matrix } from "../../src/linalg/matrix.js";
import { ConstantMean } from "../../src/means/constant.js";
import { ExactGP } from "../../src/models/gp.js";

describe("ExactGP K* caching (Tier 2.2)", () => {
  it("caches cross-covariance K* for repeated predictions", () => {
    const trainX = Matrix.from2D([[0], [0.5], [1]]);
    const trainY = Matrix.vector([0, 1, 0]);
    const kernel = new ScaleKernel(new RBFKernel([0.3]), 1);
    const mean = new ConstantMean(0);

    const gp = new ExactGP(trainX, trainY, kernel, mean, 1e-6);

    // Create test grid
    const testX = Matrix.from2D([[0.25], [0.75]]);

    // First prediction - computes K*
    const result1 = gp.predict(testX);

    // Second prediction with same test points - should use cached K*
    const result2 = gp.predict(testX);

    // Results should be identical
    expect(result1.mean.length).toBe(result2.mean.length);
    expect(result1.variance.length).toBe(result2.variance.length);

    for (let i = 0; i < result1.mean.length; i++) {
      expect(result1.mean[i]).toBe(result2.mean[i]);
      expect(result1.variance[i]).toBe(result2.variance[i]);
    }
  });

  it("invalidates cache when test points change", () => {
    const trainX = Matrix.from2D([[0], [0.5], [1]]);
    const trainY = Matrix.vector([0, 1, 0]);
    const kernel = new ScaleKernel(new RBFKernel([0.3]), 1);
    const mean = new ConstantMean(0);

    const gp = new ExactGP(trainX, trainY, kernel, mean, 1e-6);

    // First prediction
    const testX1 = Matrix.from2D([[0.25], [0.75]]);
    const result1 = gp.predict(testX1);

    // Second prediction with different test points
    const testX2 = Matrix.from2D([[0.3], [0.7]]);
    const result2 = gp.predict(testX2);

    // Results should be different
    expect(result1.mean[0]).not.toBe(result2.mean[0]);
  });

  it("cache works with predictCovarianceWith", () => {
    const trainX = Matrix.from2D([[0], [0.5], [1]]);
    const trainY = Matrix.vector([0, 1, 0]);
    const kernel = new ScaleKernel(new RBFKernel([0.3]), 1);
    const mean = new ConstantMean(0);

    const gp = new ExactGP(trainX, trainY, kernel, mean, 1e-6);

    const testX = Matrix.from2D([[0.25], [0.75]]);
    const refX = Matrix.from2D([[0.5]]);

    // First call to predict() caches K*
    gp.predict(testX);

    // predictCovarianceWith should reuse cached K* for testX
    const cov = gp.predictCovarianceWith(testX, refX);

    // Verify covariance is computed (basic sanity check)
    expect(cov.length).toBe(testX.rows);
    expect(cov[0]).toBeGreaterThan(-1);
    expect(cov[0]).toBeLessThan(1);
  });

  it("cache handles identical matrices with different references", () => {
    const trainX = Matrix.from2D([[0], [0.5], [1]]);
    const trainY = Matrix.vector([0, 1, 0]);
    const kernel = new ScaleKernel(new RBFKernel([0.3]), 1);
    const mean = new ConstantMean(0);

    const gp = new ExactGP(trainX, trainY, kernel, mean, 1e-6);

    // Create two identical matrices (different objects)
    const testX1 = Matrix.from2D([[0.25], [0.75]]);
    const testX2 = Matrix.from2D([[0.25], [0.75]]);

    const result1 = gp.predict(testX1);
    const result2 = gp.predict(testX2);

    // Results should be identical (cache key based on content)
    for (let i = 0; i < result1.mean.length; i++) {
      expect(result1.mean[i]).toBe(result2.mean[i]);
      expect(result1.variance[i]).toBe(result2.variance[i]);
    }
  });
});
