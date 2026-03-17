import { Matrix } from "../linalg/matrix.js";

/**
 * Squared pairwise distances using the expansion ||a-b||² = ||a||² + ||b||² - 2a·b.
 * This is more numerically stable than computing (a-b)² directly.
 */
export function cdistSquared(x1: Matrix, x2: Matrix): Matrix {
  const n = x1.rows;
  const m = x2.rows;
  const d = x1.cols;
  const result = new Matrix(n, m);

  // Compute ||x1||² per row
  const norms1 = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let k = 0; k < d; k++) {
      const v = x1.get(i, k);
      s += v * v;
    }
    norms1[i] = s;
  }

  // Compute ||x2||² per row
  const norms2 = new Float64Array(m);
  for (let j = 0; j < m; j++) {
    let s = 0;
    for (let k = 0; k < d; k++) {
      const v = x2.get(j, k);
      s += v * v;
    }
    norms2[j] = s;
  }

  // result[i][j] = ||x1[i]||² + ||x2[j]||² - 2 * x1[i] · x2[j]
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < m; j++) {
      let dot = 0;
      for (let k = 0; k < d; k++) {
        dot += x1.get(i, k) * x2.get(j, k);
      }
      const val = norms1[i] + norms2[j] - 2 * dot;
      // Clamp to avoid tiny negatives from floating-point arithmetic
      result.set(i, j, Math.max(val, 0));
    }
  }
  return result;
}

/**
 * Euclidean pairwise distances. Clamps squared distances ≥ 1e-30 before sqrt,
 * then clamps result ≥ 1e-15 (matching GPyTorch behavior for Matérn).
 */
export function cdist(x1: Matrix, x2: Matrix): Matrix {
  const sq = cdistSquared(x1, x2);
  const result = new Matrix(sq.rows, sq.cols);
  for (let i = 0; i < sq.data.length; i++) {
    result.data[i] = Math.sqrt(Math.max(sq.data[i], 1e-30));
    result.data[i] = Math.max(result.data[i], 1e-15);
  }
  return result;
}
