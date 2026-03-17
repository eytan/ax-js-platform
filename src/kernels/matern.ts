import { Matrix } from "../linalg/matrix.js";
import { cdist } from "./distance.js";
import type { Kernel } from "./types.js";

/**
 * Matérn kernel with ARD lengthscales.
 *
 * Key GPyTorch behavior replicated:
 * 1. Mean-center: subtract x1.mean(axis=0) from both x1 and x2
 * 2. Scale by lengthscale: x_ = (x - mean) / lengthscale
 * 3. Compute Euclidean distance, clamped ≥ 1e-15
 * 4. Apply Matérn formula
 */
export class MaternKernel implements Kernel {
  readonly lengthscale: Float64Array;
  readonly nu: 0.5 | 1.5 | 2.5;

  constructor(lengthscale: number[], nu: 0.5 | 1.5 | 2.5 = 2.5) {
    this.lengthscale = Float64Array.from(lengthscale);
    this.nu = nu;
  }

  compute(x1: Matrix, x2: Matrix): Matrix {
    const d = x1.cols;

    // Step 1: Compute mean of x1 along axis 0 (GPyTorch mean-centering)
    const mean = new Float64Array(d);
    for (let j = 0; j < d; j++) {
      let s = 0;
      for (let i = 0; i < x1.rows; i++) s += x1.get(i, j);
      mean[j] = s / x1.rows;
    }

    // Step 2: Mean-center and divide by lengthscale
    const x1s = new Matrix(x1.rows, d);
    for (let i = 0; i < x1.rows; i++) {
      for (let j = 0; j < d; j++) {
        x1s.set(i, j, (x1.get(i, j) - mean[j]) / this.lengthscale[j]);
      }
    }
    const x2s = new Matrix(x2.rows, d);
    for (let i = 0; i < x2.rows; i++) {
      for (let j = 0; j < d; j++) {
        x2s.set(i, j, (x2.get(i, j) - mean[j]) / this.lengthscale[j]);
      }
    }

    // Step 3: Euclidean distances (clamped via cdist)
    const dist = cdist(x1s, x2s);

    // Step 4: Apply Matérn formula
    const result = new Matrix(x1.rows, x2.rows);
    if (this.nu === 2.5) {
      const sqrt5 = Math.sqrt(5);
      for (let i = 0; i < dist.data.length; i++) {
        const r = dist.data[i];
        const sr = sqrt5 * r;
        result.data[i] = (1 + sr + (5 * r * r) / 3) * Math.exp(-sr);
      }
    } else if (this.nu === 1.5) {
      const sqrt3 = Math.sqrt(3);
      for (let i = 0; i < dist.data.length; i++) {
        const r = dist.data[i];
        const sr = sqrt3 * r;
        result.data[i] = (1 + sr) * Math.exp(-sr);
      }
    } else {
      // nu = 0.5: exponential kernel k(r) = exp(-r)
      for (let i = 0; i < dist.data.length; i++) {
        result.data[i] = Math.exp(-dist.data[i]);
      }
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
