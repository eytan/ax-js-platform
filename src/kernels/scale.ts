// Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

import type { Kernel } from "./types.js";
import type { Matrix } from "../linalg/matrix.js";

/**
 * ScaleKernel wraps a base kernel and multiplies by outputscale.
 * Matches GPyTorch's ScaleKernel behavior.
 */
export class ScaleKernel implements Kernel {
  readonly base: Kernel;
  readonly outputscale: number;

  constructor(base: Kernel, outputscale: number) {
    this.base = base;
    this.outputscale = outputscale;
  }

  compute(x1: Matrix, x2: Matrix): Matrix {
    const K = this.base.compute(x1, x2);
    for (let i = 0; i < K.data.length; i++) {
      K.data[i] *= this.outputscale;
    }
    return K;
  }

  computeDiag(x: Matrix): Float64Array {
    if (this.base.computeDiag) {
      const diag = this.base.computeDiag(x);
      for (let i = 0; i < diag.length; i++) {
        diag[i] *= this.outputscale;
      }
      return diag;
    }
    const K = this.base.compute(x, x);
    const diag = new Float64Array(x.rows);
    for (let i = 0; i < x.rows; i++) {
      diag[i] = K.get(i, i) * this.outputscale;
    }
    return diag;
  }
}
