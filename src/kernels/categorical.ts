// Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

import type { Kernel } from "./types.js";

import { Matrix } from "../linalg/matrix.js";

/**
 * Categorical (Hamming distance) kernel.
 *
 * For categorical features encoded as integers, computes:
 *   k(x1, x2) = exp(-mean_d((x1_d != x2_d) / lengthscale_d))
 *
 * Supports both scalar lengthscale and ARD (per-dimension) lengthscales.
 * Matches BoTorch CategoricalKernel behavior.
 */
export class CategoricalKernel implements Kernel {
  readonly lengthscales: Float64Array;

  constructor(lengthscale: number | Array<number>) {
    if (typeof lengthscale === "number") {
      this.lengthscales = Float64Array.from([lengthscale]);
    } else {
      this.lengthscales = Float64Array.from(lengthscale);
    }
  }

  compute(x1: Matrix, x2: Matrix): Matrix {
    const result = new Matrix(x1.rows, x2.rows);
    const d = x1.cols;
    const ard = this.lengthscales.length > 1;

    for (let i = 0; i < x1.rows; i++) {
      for (let j = 0; j < x2.rows; j++) {
        let dist = 0;
        for (let k = 0; k < d; k++) {
          if (Math.abs(x1.get(i, k) - x2.get(j, k)) > 1e-8) {
            dist += 1 / (ard ? this.lengthscales[k] : this.lengthscales[0]);
          }
        }
        result.set(i, j, Math.exp(-dist / d));
      }
    }
    return result;
  }

  computeDiag(x: Matrix): Float64Array {
    const diag = new Float64Array(x.rows);
    diag.fill(1);
    return diag;
  }
}
