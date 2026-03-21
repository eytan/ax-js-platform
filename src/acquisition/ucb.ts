// Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

import type { GPModel, AcquisitionFunction } from "./types.js";

/**
 * Upper Confidence Bound (UCB) acquisition function.
 *
 *   UCB(x) = μ(x) + β·σ(x)
 *
 * where β controls the exploration-exploitation tradeoff.
 * Higher β → more exploration (wider confidence bound).
 * Default β = 2.0 (approximately 95% confidence bound).
 */
export class UpperConfidenceBound implements AcquisitionFunction {
  constructor(
    private readonly model: GPModel,
    private readonly beta: number = 2,
  ) {}

  evaluate(candidates: Array<Array<number>>): Float64Array {
    const { mean, variance } = this.model.predict(candidates);
    const n = mean.length;
    const values = new Float64Array(n);
    const sqrtBeta = Math.sqrt(this.beta);
    for (let i = 0; i < n; i++) {
      values[i] = mean[i] + sqrtBeta * Math.sqrt(variance[i]);
    }
    return values;
  }
}
