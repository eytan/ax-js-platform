import { Matrix } from "./matrix.js";

/**
 * Cholesky decomposition with GPyTorch-style jitter escalation.
 * Matches psd_safe_cholesky behavior: tries no jitter first,
 * then escalates through [1e-6, 1e-5, 1e-4, 1e-3].
 */
export function cholesky(A: Matrix): Matrix {
  const result = choleskyRaw(A);
  if (result !== null) return result;

  const jitters = [1e-6, 1e-5, 1e-4, 1e-3];
  for (const jitter of jitters) {
    const Aj = A.clone();
    Aj.addDiag(jitter);
    const result = choleskyRaw(Aj);
    if (result !== null) return result;
  }
  throw new Error("Cholesky decomposition failed even with jitter 1e-3");
}

function choleskyRaw(A: Matrix): Matrix | null {
  const n = A.rows;
  const L = Matrix.zeros(n, n);

  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let s = A.get(i, j);
      for (let k = 0; k < j; k++) {
        s -= L.get(i, k) * L.get(j, k);
      }
      if (i === j) {
        if (s <= 0) return null;
        L.set(i, j, Math.sqrt(s));
      } else {
        L.set(i, j, s / L.get(j, j));
      }
    }
  }
  return L;
}
