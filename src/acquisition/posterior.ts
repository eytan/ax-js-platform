// Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

import type { GPModel } from "./types.js";

import { Matrix } from "../linalg/matrix.js";

/**
 * Compute the full posterior covariance matrix for a set of test points.
 *
 * Returns Σ where Σ[i,j] = Cov(f(x_i), f(x_j)).
 *
 * This builds the matrix column-by-column using predictCovarianceWith,
 * which computes Cov(f(x_i), f(x_ref)) for all i against a single ref point.
 *
 * The result is symmetric, so we only compute the lower triangle and mirror.
 */
export function posteriorCovariance(model: GPModel, points: Array<Array<number>>): Matrix {
  const n = points.length;
  const Sigma = new Matrix(n, n);

  // Build full matrix via predictCovarianceWith (one column per ref point).
  // This is more consistent than mixing predict() variance (which may clamp)
  // with predictCovarianceWith off-diagonal values.
  for (let j = 0; j < n; j++) {
    const cov = model.predictCovarianceWith(points, points[j]);
    for (let i = 0; i < n; i++) {
      Sigma.set(i, j, cov[i]);
    }
  }

  return Sigma;
}

/**
 * Compute posterior mean vector as a Float64Array.
 * Convenience wrapper around model.predict().
 */
export function posteriorMean(model: GPModel, points: Array<Array<number>>): Float64Array {
  return model.predict(points).mean;
}
