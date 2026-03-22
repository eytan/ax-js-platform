// Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

import type { GPInternals, PredictionResult } from "./types.js";
import type { Kernel } from "../kernels/types.js";
import type { ConstantMean } from "../means/constant.js";
import type { InputNormalize } from "../transforms/normalize.js";
import type { OutcomeUntransform } from "../transforms/outcome.js";
import type { InputWarp } from "../transforms/warp.js";

import { kernelDiag } from "../kernels/composite.js";
import { cholesky } from "../linalg/cholesky.js";
import { Matrix } from "../linalg/matrix.js";
import { forwardSolveTransposed, solveCholesky } from "../linalg/solve.js";

/**
 * Core ExactGP with Cholesky-based posterior.
 *
 * Pre-computes at construction:
 *   L = cholesky(K + noise * I)
 *   alpha = L^T \ (L \ (y - m(X)))
 *
 * Prediction at X*:
 *   mu = m(X*) + k* · alpha
 *   v = L \ k*^T
 *   var = diag(k**) - sum(v²)   [diagonal only — O(m) not O(m²)]
 */
export class ExactGP {
  private readonly kernel: Kernel;
  private readonly mean: ConstantMean;
  private readonly inputTransform: InputNormalize | null;
  private readonly inputWarp: InputWarp | null;
  private readonly outcomeTransform: OutcomeUntransform | null;
  private readonly trainXNorm: Matrix;
  private readonly L: Matrix;
  private readonly alpha: Matrix;

  // V matrix cache for sharing between predict() and predictCovarianceWith()
  private cachedV: Matrix | null = null;
  private cachedTestX: Matrix | null = null;

  // K* (cross-covariance) cache for interactive use (Tier 2.2)
  private cachedKstar: Matrix | null = null;
  private cachedKstarKey: string | null = null;

  constructor(
    trainX: Matrix,
    trainY: Matrix,
    kernel: Kernel,
    mean: ConstantMean,
    noiseVariance: number | Float64Array,
    inputTransform?: InputNormalize,
    outcomeTransform?: OutcomeUntransform,
    inputWarp?: InputWarp,
  ) {
    this.kernel = kernel;
    this.mean = mean;
    this.inputTransform = inputTransform ?? null;
    this.inputWarp = inputWarp ?? null;
    this.outcomeTransform = outcomeTransform ?? null;

    // Step 1: Normalize and warp training inputs
    let xNorm = inputTransform ? inputTransform.forward(trainX) : trainX.clone();
    if (inputWarp) {
      xNorm = inputWarp.forward(xNorm);
    }
    this.trainXNorm = xNorm;

    // Step 2: Compute kernel matrix
    const K = kernel.compute(this.trainXNorm, this.trainXNorm);

    // Step 3: Add noise
    if (typeof noiseVariance === "number") {
      K.addDiag(noiseVariance);
    } else {
      K.addDiagVec(noiseVariance);
    }

    // Step 4: Cholesky factorization
    this.L = cholesky(K);

    // Step 5: Compute alpha = (K+noise*I)^{-1} (y - m(X))
    const meanVals = mean.forward(this.trainXNorm);
    const residuals = new Matrix(trainY.rows, 1);
    for (let i = 0; i < trainY.rows; i++) {
      residuals.data[i] = trainY.get(i, 0) - meanVals.data[i];
    }
    this.alpha = solveCholesky(this.L, residuals);
  }

  /** Expose GP internals for analytic Sobol' computation. */
  getInternals(): GPInternals {
    const n = this.alpha.rows;
    const d = this.trainXNorm.cols;
    const alphaArr = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      alphaArr[i] = this.alpha.get(i, 0);
    }
    const trainXArr: Array<Array<number>> = new Array(n);
    for (let i = 0; i < n; i++) {
      const row = new Array(d);
      for (let j = 0; j < d; j++) {
        row[j] = this.trainXNorm.get(i, j);
      }
      trainXArr[i] = row;
    }
    return {
      alpha: alphaArr,
      trainXNorm: trainXArr,
      meanConstant: this.mean.constant,
    };
  }

  /** Transform test inputs through normalize + warp pipeline. */
  private transformInputs(testX: Matrix): Matrix {
    let testXNorm = this.inputTransform ? this.inputTransform.forward(testX) : testX;
    if (this.inputWarp) {
      testXNorm = this.inputWarp.forward(testXNorm);
    }
    return testXNorm;
  }

  /** Check if the cached V matrix is valid for the given test points. */
  private isCachedVValid(testX: Matrix): boolean {
    if (!this.cachedV || !this.cachedTestX) {
      return false;
    }
    // Quick reference check first
    if (this.cachedTestX === testX) {
      return true;
    }
    // Fallback: dimension and content check
    if (this.cachedTestX.rows !== testX.rows || this.cachedTestX.cols !== testX.cols) {
      return false;
    }
    // Check if all values match
    for (let i = 0; i < testX.rows * testX.cols; i++) {
      if (this.cachedTestX.data[i] !== testX.data[i]) {
        return false;
      }
    }
    return true;
  }

  /** Generate cache key for K* based on transformed test points. */
  private makeKstarCacheKey(testXNorm: Matrix): string {
    const n = testXNorm.rows * testXNorm.cols;
    if (n === 0) {
      return "empty";
    }
    // FNV-1a-inspired hash over all data elements for collision resistance.
    // The previous first/last-only key missed changes to interior dimensions.
    let h = 2166136261;
    for (let i = 0; i < n; i++) {
      // Mix in the float bits via a 32-bit view of the upper half
      const v = testXNorm.data[i];
      const bits = (v * 2654435761) | 0; // multiplicative hash of the float
      h = ((h ^ bits) * 16777619) | 0;
    }
    return `${testXNorm.rows}:${testXNorm.cols}:${h >>> 0}`;
  }

  /** Compute V = L⁻¹ @ K*ᵀ for a set of transformed test points. */
  private computeV(Kstar: Matrix): Matrix {
    return forwardSolveTransposed(this.L, Kstar);
  }

  predict(testX: Matrix): PredictionResult {
    const testXNorm = this.transformInputs(testX);

    // Cross-covariance k(X*, X_train) — check cache first
    const cacheKey = this.makeKstarCacheKey(testXNorm);
    let Kstar: Matrix;
    if (this.cachedKstarKey === cacheKey && this.cachedKstar !== null) {
      Kstar = this.cachedKstar;
    } else {
      Kstar = this.kernel.compute(testXNorm, this.trainXNorm);
      this.cachedKstar = Kstar;
      this.cachedKstarKey = cacheKey;
    }

    // Posterior mean: m(X*) + k* · alpha
    const meanPrior = this.mean.forward(testXNorm);
    const mu = new Float64Array(testX.rows);
    for (let i = 0; i < testX.rows; i++) {
      let dot = 0;
      for (let j = 0; j < this.trainXNorm.rows; j++) {
        dot += Kstar.get(i, j) * this.alpha.get(j, 0);
      }
      mu[i] = meanPrior.data[i] + dot;
    }

    // Posterior variance: diag(k**) - ||v||²
    const kssDiag = kernelDiag(this.kernel, testXNorm);
    const V = this.computeV(Kstar);

    // Cache V for potential reuse in predictCovarianceWith
    this.cachedV = V;
    this.cachedTestX = testX;

    const variance = new Float64Array(testX.rows);
    for (let i = 0; i < testX.rows; i++) {
      let vSq = 0;
      for (let j = 0; j < this.trainXNorm.rows; j++) {
        const vji = V.get(j, i);
        vSq += vji * vji;
      }
      variance[i] = Math.max(0, kssDiag[i] - vSq);
    }

    // Un-transform if outcome transform exists
    if (this.outcomeTransform) {
      for (let i = 0; i < mu.length; i++) {
        const ut = this.outcomeTransform.untransform(mu[i], variance[i]);
        mu[i] = ut.mean;
        variance[i] = ut.variance;
      }
    }

    return { mean: mu, variance };
  }

  /**
   * Analytic Leave-One-Out Cross-Validation predictions (Rasmussen & Williams, Eq. 5.12).
   *
   * LOO mean:     μ_LOO(i) = y_i - α_i / [K⁻¹]_ii
   * LOO variance: σ²_LOO(i) = 1 / [K⁻¹]_ii
   *
   * where α = K⁻¹(y - m(X)) and K = K(X,X) + σ²I.
   * No refitting required — computed from the full GP's Cholesky factor.
   *
   * Returns predictions in model-internal space (before outcome untransform).
   * Caller (Predictor) is responsible for applying outcome + adapter untransforms.
   */
  loocvPredictions(trainY: Matrix): PredictionResult {
    const n = this.L.rows;

    // Compute diag(K⁻¹) = diag(L⁻ᵀ L⁻¹).
    // For each column i of L⁻¹ (= forward solve of L with e_i),
    // [K⁻¹]_ii = ||L⁻¹[:,i]||².
    // We solve one column at a time to avoid storing the full inverse.
    const diagKinv = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      // Forward-solve L * x = e_i
      const x = new Float64Array(n);
      for (let row = 0; row < n; row++) {
        let s = row === i ? 1 : 0;
        for (let j = 0; j < row; j++) {
          s -= this.L.get(row, j) * x[j];
        }
        x[row] = s / this.L.get(row, row);
      }
      let sumSq = 0;
      for (let j = 0; j < n; j++) {
        sumSq += x[j] * x[j];
      }
      diagKinv[i] = sumSq;
    }

    // LOO predictions
    const mean = new Float64Array(n);
    const variance = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const yi = trainY.get(i, 0);
      mean[i] = yi - this.alpha.get(i, 0) / diagKinv[i];
      variance[i] = Math.max(0, 1 / diagKinv[i]);
    }

    // Apply outcome untransform (same as predict())
    if (this.outcomeTransform) {
      for (let i = 0; i < n; i++) {
        const ut = this.outcomeTransform.untransform(mean[i], variance[i]);
        mean[i] = ut.mean;
        variance[i] = ut.variance;
      }
    }

    return { mean, variance };
  }

  /**
   *
   * Cov(f(x_i), f(x_ref)) = k(x_i, x_ref) - v_i · v_ref
   * where v = L⁻¹ @ k(X, x)
   *
   * Note: returns covariance in the MODEL's transformed space (before outcome
   * untransform). For Standardize this scales by std²; for nonlinear transforms
   * the covariance in original space requires additional delta-method terms.
   * For relativization, the Standardize scaling cancels out in the ratio.
   */
  predictCovarianceWith(testX: Matrix, refX: Matrix): Float64Array {
    const testXNorm = this.transformInputs(testX);
    const refXNorm = this.transformInputs(refX);

    // V matrices: L⁻¹ @ K*ᵀ
    // Reuse cached Vtest if available, otherwise compute fresh
    let Vtest: Matrix;
    if (this.isCachedVValid(testX)) {
      Vtest = this.cachedV!;
    } else {
      // Check K* cache for test points
      const testCacheKey = this.makeKstarCacheKey(testXNorm);
      let KstarTest: Matrix;
      if (this.cachedKstarKey === testCacheKey && this.cachedKstar !== null) {
        KstarTest = this.cachedKstar;
      } else {
        KstarTest = this.kernel.compute(testXNorm, this.trainXNorm);
        this.cachedKstar = KstarTest;
        this.cachedKstarKey = testCacheKey;
      }
      Vtest = this.computeV(KstarTest);
    }

    // Cross-covariance of ref with training data (no cache, typically 1 ref point)
    const KstarRef = this.kernel.compute(refXNorm, this.trainXNorm);
    const Vref = this.computeV(KstarRef);

    // Prior covariance k(x_test, x_ref)
    const Ktr = this.kernel.compute(testXNorm, refXNorm);

    const n = testX.rows;
    const covariance = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      // k(x_i, x_ref) - v_i · v_ref
      let vDot = 0;
      for (let j = 0; j < this.trainXNorm.rows; j++) {
        vDot += Vtest.get(j, i) * Vref.get(j, 0);
      }
      covariance[i] = Ktr.get(i, 0) - vDot;
    }

    // Scale covariance by outcome transform
    if (this.outcomeTransform) {
      for (let i = 0; i < n; i++) {
        covariance[i] = this.outcomeTransform.untransformCovariance(covariance[i]);
      }
    }

    return covariance;
  }
}
