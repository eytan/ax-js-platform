// Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

import type { Kernel } from "./types.js";

import { Matrix } from "../linalg/matrix.js";

import { cdistSquared } from "./distance.js";

/**
 * RBF (Squared Exponential) kernel with ARD lengthscales.
 *
 * Unlike Matérn, RBF does NOT mean-center inputs in GPyTorch.
 * k(x1, x2) = exp(-||x1/l - x2/l||² / 2)
 */
export class RBFKernel implements Kernel {
  readonly lengthscale: Float64Array;

  constructor(lengthscale: Array<number>) {
    this.lengthscale = Float64Array.from(lengthscale);
  }

  compute(x1: Matrix, x2: Matrix): Matrix {
    const d = x1.cols;

    // Scale by lengthscale (no mean-centering)
    const x1s = new Matrix(x1.rows, d);
    for (let i = 0; i < x1.rows; i++) {
      for (let j = 0; j < d; j++) {
        x1s.set(i, j, x1.get(i, j) / this.lengthscale[j]);
      }
    }
    const x2s = new Matrix(x2.rows, d);
    for (let i = 0; i < x2.rows; i++) {
      for (let j = 0; j < d; j++) {
        x2s.set(i, j, x2.get(i, j) / this.lengthscale[j]);
      }
    }

    // Squared distances via quadratic expansion
    const sq = cdistSquared(x1s, x2s);

    // k = exp(-r²/2)
    const result = new Matrix(x1.rows, x2.rows);
    for (let i = 0; i < sq.data.length; i++) {
      result.data[i] = Math.exp(-sq.data[i] / 2);
    }
    return result;
  }

  computeDiag(x: Matrix): Float64Array {
    // Stationary kernel: k(x, x) = 1 for all x
    const diag = new Float64Array(x.rows);
    diag.fill(1);
    return diag;
  }
}
