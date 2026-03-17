import { Matrix } from "../linalg/matrix.js";
import type { Kernel } from "./types.js";

/**
 * Extracts a subset of columns (active dimensions) from a matrix.
 */
export function extractActiveDims(x: Matrix, activeDims: number[]): Matrix {
  const result = new Matrix(x.rows, activeDims.length);
  for (let i = 0; i < x.rows; i++) {
    for (let j = 0; j < activeDims.length; j++) {
      result.set(i, j, x.get(i, activeDims[j]));
    }
  }
  return result;
}

/**
 * Wraps a kernel to operate only on specified input dimensions.
 * The base kernel receives inputs with only the active dimensions.
 */
export class ActiveDimsKernel implements Kernel {
  readonly base: Kernel;
  readonly activeDims: number[];

  constructor(base: Kernel, activeDims: number[]) {
    this.base = base;
    this.activeDims = activeDims;
  }

  compute(x1: Matrix, x2: Matrix): Matrix {
    return this.base.compute(
      extractActiveDims(x1, this.activeDims),
      extractActiveDims(x2, this.activeDims),
    );
  }

  computeDiag(x: Matrix): Float64Array {
    if (this.base.computeDiag) {
      return this.base.computeDiag(extractActiveDims(x, this.activeDims));
    }
    const K = this.compute(x, x);
    const diag = new Float64Array(x.rows);
    for (let i = 0; i < x.rows; i++) diag[i] = K.get(i, i);
    return diag;
  }
}

/**
 * Additive kernel: k(x1, x2) = k1(x1, x2) + k2(x1, x2).
 */
export class AdditiveKernel implements Kernel {
  readonly kernels: Kernel[];

  constructor(kernels: Kernel[]) {
    this.kernels = kernels;
  }

  compute(x1: Matrix, x2: Matrix): Matrix {
    const result = this.kernels[0].compute(x1, x2);
    for (let k = 1; k < this.kernels.length; k++) {
      const K = this.kernels[k].compute(x1, x2);
      for (let i = 0; i < result.data.length; i++) {
        result.data[i] += K.data[i];
      }
    }
    return result;
  }

  computeDiag(x: Matrix): Float64Array {
    const diag = kernelDiag(this.kernels[0], x);
    for (let k = 1; k < this.kernels.length; k++) {
      const d = kernelDiag(this.kernels[k], x);
      for (let i = 0; i < diag.length; i++) diag[i] += d[i];
    }
    return diag;
  }
}

/**
 * Product kernel: k(x1, x2) = k1(x1, x2) * k2(x1, x2).
 * Used for mixing continuous and categorical kernels, e.g.:
 *   ScaleKernel(RBFKernel(active_dims=[0,1]) * CategoricalKernel(active_dims=[2]))
 */
export class ProductKernel implements Kernel {
  readonly kernels: Kernel[];

  constructor(kernels: Kernel[]) {
    this.kernels = kernels;
  }

  compute(x1: Matrix, x2: Matrix): Matrix {
    const result = this.kernels[0].compute(x1, x2);
    for (let k = 1; k < this.kernels.length; k++) {
      const K = this.kernels[k].compute(x1, x2);
      for (let i = 0; i < result.data.length; i++) {
        result.data[i] *= K.data[i];
      }
    }
    return result;
  }

  computeDiag(x: Matrix): Float64Array {
    const diag = kernelDiag(this.kernels[0], x);
    for (let k = 1; k < this.kernels.length; k++) {
      const d = kernelDiag(this.kernels[k], x);
      for (let i = 0; i < diag.length; i++) diag[i] *= d[i];
    }
    return diag;
  }
}

/**
 * Helper: compute kernel diagonal, using computeDiag if available,
 * falling back to full matrix computation.
 */
export function kernelDiag(kernel: Kernel, x: Matrix): Float64Array {
  if (kernel.computeDiag) {
    return kernel.computeDiag(x);
  }
  const K = kernel.compute(x, x);
  const diag = new Float64Array(x.rows);
  for (let i = 0; i < x.rows; i++) diag[i] = K.get(i, i);
  return diag;
}
