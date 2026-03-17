import { Matrix } from "../linalg/matrix.js";

/**
 * Kumaraswamy CDF input warping.
 *
 * Matches BoTorch's Warp transform:
 *   K(x; a, b) = 1 - (1 - x^concentration1)^concentration0
 *
 * Applied element-wise to selected dimensions (indices).
 * Input should already be in [0, 1] (i.e., after Normalize).
 */
export class InputWarp {
  readonly concentration0: Float64Array;
  readonly concentration1: Float64Array;
  readonly indices: number[] | null;

  constructor(
    concentration0: number[],
    concentration1: number[],
    indices?: number[],
  ) {
    this.concentration0 = Float64Array.from(concentration0);
    this.concentration1 = Float64Array.from(concentration1);
    this.indices = indices ?? null;
  }

  forward(X: Matrix): Matrix {
    const result = X.clone();
    const dims = this.indices ?? Array.from({ length: X.cols }, (_, i) => i);
    const eps = 1e-7;
    const range = 1 - 2 * eps;

    for (let i = 0; i < X.rows; i++) {
      for (let k = 0; k < dims.length; k++) {
        const j = dims[k];
        const x = result.get(i, j);
        const a = this.concentration1[k];
        const b = this.concentration0[k];
        // Kumaraswamy CDF: 1 - (1 - x^a)^b
        // Match BoTorch: normalize to [eps, 1-eps], then clamp
        const xn = Math.max(eps, Math.min(1 - eps, x * range + eps));
        result.set(i, j, 1 - Math.pow(1 - Math.pow(xn, a), b));
      }
    }
    return result;
  }
}
