import type { GPModel, AcquisitionFunction } from "./types.js";
import { logNormalCdf, logNormalPdf, normalCdf, normalPdf } from "./normal.js";

/**
 * Log Expected Improvement acquisition function.
 *
 * Computes log(EI(x)) in a numerically stable way, avoiding the catastrophic
 * cancellation that occurs with raw EI when the improvement probability is small.
 *
 * From BoTorch's LogExpectedImprovement:
 *   LogEI(x) = log(σ) + log_h(z)
 *   where z = (μ - f_best) / σ
 *   log_h(z) = log(z·Φ(z) + φ(z))
 *            = log_softplus(z) [numerically stable formulation]
 *
 * The log_softplus formulation uses:
 *   log_h(z) = log(Φ(z)) + log(1 + φ(z)/(z·Φ(z)))   for z ≥ 0
 *   log_h(z) = log(φ(z)) + log(1 + z·Φ(z)/φ(z))     for z < 0
 */
export class LogExpectedImprovement implements AcquisitionFunction {
  constructor(
    private model: GPModel,
    private bestF: number,
  ) {}

  evaluate(candidates: number[][]): Float64Array {
    const { mean, variance } = this.model.predict(candidates);
    const n = mean.length;
    const values = new Float64Array(n);

    for (let i = 0; i < n; i++) {
      const sigma = Math.sqrt(variance[i]);
      if (sigma < 1e-12) {
        // Near-zero variance: EI is deterministic improvement or zero
        values[i] = mean[i] > this.bestF ? Math.log(mean[i] - this.bestF) : -Infinity;
        continue;
      }
      const z = (mean[i] - this.bestF) / sigma;
      values[i] = Math.log(sigma) + logH(z);
    }

    return values;
  }
}

/**
 * Expected Improvement (non-log, for convenience).
 * Returns EI(x), not log(EI(x)).
 */
export class ExpectedImprovement implements AcquisitionFunction {
  constructor(
    private model: GPModel,
    private bestF: number,
  ) {}

  evaluate(candidates: number[][]): Float64Array {
    const { mean, variance } = this.model.predict(candidates);
    const n = mean.length;
    const values = new Float64Array(n);

    for (let i = 0; i < n; i++) {
      const sigma = Math.sqrt(variance[i]);
      if (sigma < 1e-12) {
        values[i] = Math.max(0, mean[i] - this.bestF);
        continue;
      }
      const z = (mean[i] - this.bestF) / sigma;
      values[i] = Math.max(0, sigma * (z * normalCdf(z) + normalPdf(z)));
    }

    return values;
  }
}

/**
 * Numerically stable computation of log(z·Φ(z) + φ(z)).
 *
 * h(z) = z·Φ(z) + φ(z)
 *
 * Three regimes:
 * - z > 0:  h = z·Φ(z)·(1 + φ(z)/(z·Φ(z)))
 *           log h = log(z) + log(Φ(z)) + log1p(φ/(z·Φ))
 * - z = 0:  h = φ(0) = 1/√(2π)
 * - z < 0:  h = φ(z)·(1 + z·Φ(z)/φ(z))
 *           log h = log(φ(z)) + log1p(z·Φ(z)/φ(z))
 */
function logH(z: number): number {
  if (z > 0) {
    const cdf = normalCdf(z);
    const pdf = normalPdf(z);
    // h = z·Φ(z)·(1 + φ/(z·Φ))
    return Math.log(z) + logNormalCdf(z) + Math.log1p(pdf / (z * cdf));
  } else if (z === 0) {
    return logNormalPdf(0);
  } else {
    const cdf = normalCdf(z);
    const pdf = normalPdf(z);
    if (pdf < 1e-300) return logNormalCdf(z); // extreme tail fallback
    return logNormalPdf(z) + Math.log1p((z * cdf) / pdf);
  }
}
