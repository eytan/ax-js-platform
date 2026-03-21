// Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

/**
 * Relativization: compute percentage change vs. a status quo (baseline) arm.
 *
 * Matches the formulas in Ax's `ax.utils.stats.math_utils.relativize()`.
 * The delta method propagates uncertainty through the nonlinear ratio estimator.
 *
 * Estimand: r = (mu_t / mu_c) - 1  (fractional change from control)
 *
 * This is a CALLER-LEVEL operation, not a model-level transform. The GP model
 * predicts absolute values; relativization is applied afterward for display.
 */

/** Result of relativization: percentage change and its standard error. */
export interface RelativizeResult {
  mean: number;
  sem: number;
}

export interface RelativizeOptions {
  /** Covariance between test and control means. Defaults to 0 (conservative). */
  covariance?: number;
  /** Apply second-order bias correction. Defaults to true. */
  biasCorrection?: boolean;
  /** Express result as percentage (multiply by 100). Defaults to true. */
  asPercent?: boolean;
  /**
   * Treat the control mean as a known constant (ignore control SEM).
   * Used by RelativizeWithConstantControl in Ax.
   * Defaults to false.
   */
  controlAsConstant?: boolean;
}

/**
 * Compute relative effect of test arm vs. control arm using delta method.
 *
 * @param meanT  - Test arm posterior mean (absolute)
 * @param semT   - Test arm posterior SEM (sqrt of variance)
 * @param meanC  - Control/status quo arm posterior mean (absolute)
 * @param semC   - Control/status quo arm posterior SEM
 * @param opts   - Options (covariance, bias correction, percent, constant control)
 * @returns       Relative effect mean and SEM
 *
 * Formulas (matching Ax `math_utils.relativize`):
 *
 *   r_hat = (m_t - m_c) / |m_c|                          [first-order]
 *   r_hat -= m_t * s_c² / |m_c|³                          [bias correction]
 *
 *   Var(r) = (s_t² - 2*c*cov + c²*s_c²) / m_c²
 *   where c = m_t / m_c
 *
 * When the test IS the control (same mean and SEM), returns exactly (0, 0).
 *
 * @example
 * ```ts
 * import { relativize } from "ax-js";
 * const result = relativize(1.1, 0.05, 1.0, 0.03);
 * // result.mean ≈ 10.0 (10% improvement), result.sem ≈ 5.6
 * ```
 */
export function relativize(
  meanT: number,
  semT: number,
  meanC: number,
  semC: number,
  opts: RelativizeOptions = {},
): RelativizeResult {
  const {
    covariance = 0,
    biasCorrection = true,
    asPercent = true,
    controlAsConstant = false,
  } = opts;

  // Status quo relative to itself is definitionally zero
  if (meanT === meanC && semT === semC) {
    return { mean: 0, sem: 0 };
  }

  const absMC = Math.abs(meanC);
  if (absMC === 0) {
    throw new Error(
      "Cannot relativize: control mean is zero. " +
        "Relative effects are undefined when the baseline is zero.",
    );
  }

  let rHat: number;
  let sem: number;

  if (controlAsConstant) {
    // Simple: treat control as known constant, only test uncertainty matters
    rHat = (meanT - meanC) / absMC;
    sem = semT / absMC;
  } else {
    // Full delta method
    rHat = (meanT - meanC) / absMC;

    if (biasCorrection) {
      // Second-order bias correction: subtract E[ratio] bias term
      rHat -= (meanT * semC * semC) / (absMC * absMC * absMC);
    }

    // Delta method variance for ratio estimator
    const c = meanT / meanC;
    const varT = semT * semT;
    const varC = semC * semC;
    const relVar = (varT - 2 * c * covariance + c * c * varC) / (meanC * meanC);
    sem = Math.sqrt(Math.max(0, relVar));
  }

  if (asPercent) {
    rHat *= 100;
    sem *= 100;
  }

  return { mean: rHat, sem };
}

/**
 * Inverse of relativize: convert relative effect back to absolute.
 *
 * Matches Ax `math_utils.unrelativize()`.
 *
 * @param relMean - Relative effect mean
 * @param relSem  - Relative effect SEM
 * @param meanC   - Control/status quo arm posterior mean
 * @param semC    - Control/status quo arm posterior SEM
 * @param opts    - Options (must match the options used to relativize)
 * @returns        Absolute mean and SEM of the test arm
 */
export function unrelativize(
  relMean: number,
  relSem: number,
  meanC: number,
  semC: number,
  opts: RelativizeOptions = {},
): RelativizeResult {
  const {
    covariance = 0,
    biasCorrection = true,
    asPercent = true,
    controlAsConstant = false,
  } = opts;

  // Undo percent scaling
  let rHat = relMean;
  let rSem = relSem;
  if (asPercent) {
    rHat /= 100;
    rSem /= 100;
  }

  const absMC = Math.abs(meanC);
  if (absMC === 0) {
    throw new Error("Cannot unrelativize: control mean is zero.");
  }

  let meanT: number;
  let semT: number;

  if (controlAsConstant) {
    meanT = rHat * absMC + meanC;
    semT = rSem * absMC;
  } else {
    // Undo relativization
    meanT = rHat * absMC + meanC;
    if (biasCorrection) {
      // Undo bias correction: m_t = m_t_raw / (1 - (s_c/|m_c|)²)
      const corrFactor = 1 - (semC * semC) / (absMC * absMC);
      if (Math.abs(corrFactor) < 1e-15) {
        throw new Error("Cannot unrelativize with bias correction: control SEM ≈ |control mean|.");
      }
      meanT /= corrFactor;
    }

    // Undo variance transformation
    const c = meanT / meanC;
    const relVar = rSem * rSem;
    const varC = semC * semC;
    const varT = relVar * meanC * meanC + 2 * c * covariance - c * c * varC;
    semT = Math.sqrt(Math.max(0, varT));
  }

  return { mean: meanT, sem: semT };
}

/**
 * Batch relativize: compute relative effects for arrays of predictions.
 *
 * Convenience wrapper for relativizing all predictions from a GP model
 * against a single status quo prediction.
 *
 * @param means      - Array of test arm posterior means
 * @param variances  - Array of test arm posterior variances (NOT SEM)
 * @param sqMean     - Status quo posterior mean
 * @param sqVariance - Status quo posterior variance (NOT SEM)
 * @param opts       - Relativization options
 * @param covariances - Optional: Cov(f(x_i), f(x_sq)) from predictCovarianceWith.
 *                      When provided, tightens confidence intervals vs the
 *                      conservative default of assuming independence.
 * @returns           Arrays of relative means and relative variances (NOT SEM)
 */
export function relativizePredictions(
  means: Float64Array | Array<number>,
  variances: Float64Array | Array<number>,
  sqMean: number,
  sqVariance: number,
  opts: RelativizeOptions = {},
  covariances?: Float64Array | Array<number>,
): { mean: Float64Array; variance: Float64Array } {
  const n = means.length;
  const relMeans = new Float64Array(n);
  const relVariances = new Float64Array(n);
  const sqSem = Math.sqrt(sqVariance);

  for (let i = 0; i < n; i++) {
    const sem = Math.sqrt(variances[i]);
    const cov = covariances ? covariances[i] : 0;
    const result = relativize(means[i], sem, sqMean, sqSem, {
      ...opts,
      covariance: cov,
    });
    relMeans[i] = result.mean;
    relVariances[i] = result.sem * result.sem;
  }

  return { mean: relMeans, variance: relVariances };
}
