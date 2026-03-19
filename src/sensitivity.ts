/**
 * Parameter importance via Sobol indices, gradient-based sensitivity (DGSM),
 * and lengthscale-based ranking.
 *
 * All functions accept a minimal predictor interface so they work with both
 * the full {@link Predictor} and the viz {@link RenderPredictor} shape.
 *
 * @module
 */

import { Rng } from "./acquisition/sample_mvn.js";

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/** Which importance method to use. */
export type ImportanceMethod = "lengthscale" | "sobol" | "gradient";

/** A single parameter's importance score. */
export interface ParameterImportance {
  dimIndex: number;
  paramName: string;
  /** Normalized importance — the most important dimension has value 1.0. */
  importance: number;
  /** Un-normalized value (lengthscale, first-order Sobol index, or mean squared gradient). */
  raw: number;
}

/** Options for Sobol / gradient computation. */
export interface SensitivityOptions {
  /** Number of base samples (default 512 for Sobol, 256 for gradient). */
  numSamples?: number;
  /** PRNG seed for reproducibility (default 42). */
  seed?: number;
}

// ---------------------------------------------------------------------------
// Minimal predictor shape consumed by sensitivity functions
// ---------------------------------------------------------------------------

/** Minimal predictor interface needed by sensitivity functions. */
export interface SensitivityPredictor {
  readonly paramNames: string[];
  readonly paramBounds: [number, number][];
  readonly paramSpecs?: ReadonlyArray<{ type: string; values?: (string | number | boolean)[] }>;
  predict(points: number[][]): Record<string, { mean: Float64Array; variance: Float64Array }>;
  rankDimensionsByImportance(
    outcomeName?: string,
  ): { dimIndex: number; paramName: string; lengthscale: number }[];
}

// ---------------------------------------------------------------------------
// Dispatcher
// ---------------------------------------------------------------------------

/**
 * Compute parameter importance using the specified method.
 *
 * @returns Array sorted by importance (descending), each value normalized so
 *          the most-important dimension has importance = 1.
 */
export function computeImportance(
  predictor: SensitivityPredictor,
  outcomeName: string,
  method: ImportanceMethod,
  options?: SensitivityOptions,
): ParameterImportance[] {
  switch (method) {
    case "lengthscale":
      return computeLengthscaleImportance(predictor, outcomeName);
    case "sobol":
      return computeSobolIndices(predictor, outcomeName, options);
    case "gradient":
      return computeGradientImportance(predictor, outcomeName, options);
  }
}

// ---------------------------------------------------------------------------
// 1. Lengthscale-based importance (wraps existing API)
// ---------------------------------------------------------------------------

export function computeLengthscaleImportance(
  predictor: SensitivityPredictor,
  outcomeName?: string,
): ParameterImportance[] {
  const ranked = predictor.rankDimensionsByImportance(outcomeName);
  if (!ranked || ranked.length === 0) return [];

  const inv = ranked.map((d) => 1 / d.lengthscale);
  const maxInv = Math.max(...inv);

  return ranked.map((d, i) => ({
    dimIndex: d.dimIndex,
    paramName: d.paramName,
    importance: maxInv > 0 ? inv[i] / maxInv : 1,
    raw: d.lengthscale,
  }));
}

// ---------------------------------------------------------------------------
// 2. Sobol indices (Saltelli 2010 estimator)
// ---------------------------------------------------------------------------

/**
 * First-order Sobol indices via the Saltelli (2010) estimator.
 *
 * For each parameter i the first-order index S_i estimates the fraction of
 * output variance attributable to that parameter alone.  The estimator is:
 *
 *   S_i ≈ (1/N) Σ_j [ f(B_j) · (f(AB_i_j) − f(A_j)) ] / Var(Y)
 *
 * where A, B are independent sample matrices and AB_i replaces column i of A
 * with column i of B.
 */
export function computeSobolIndices(
  predictor: SensitivityPredictor,
  outcomeName: string,
  options?: SensitivityOptions,
): ParameterImportance[] {
  const N = options?.numSamples ?? 512;
  const seed = options?.seed ?? 42;
  const d = predictor.paramNames.length;
  if (d === 0) return [];

  const rng = new Rng(seed);
  const bounds = predictor.paramBounds;
  const specs = predictor.paramSpecs;

  // Generate base matrices A and B — each N×d
  const A = generateSampleMatrix(N, d, bounds, specs, rng);
  const B = generateSampleMatrix(N, d, bounds, specs, rng);

  // Build AB_i matrices (N×d each, d total)
  const ABs: number[][][] = [];
  for (let i = 0; i < d; i++) {
    const ABi: number[][] = [];
    for (let j = 0; j < N; j++) {
      const row = A[j].slice();
      row[i] = B[j][i];
      ABi.push(row);
    }
    ABs.push(ABi);
  }

  // Evaluate model at all matrices in one batch per matrix
  const fA = evalMean(predictor, A, outcomeName);
  const fB = evalMean(predictor, B, outcomeName);
  const fABs: Float64Array[] = [];
  for (let i = 0; i < d; i++) {
    fABs.push(evalMean(predictor, ABs[i], outcomeName));
  }

  // Total variance of output (over combined A∪B sample)
  const allY = new Float64Array(2 * N);
  allY.set(fA, 0);
  allY.set(fB, N);
  const varY = variance(allY);

  if (varY < 1e-30) {
    // Constant model — all indices zero
    return predictor.paramNames.map((name, i) => ({
      dimIndex: i,
      paramName: name,
      importance: 0,
      raw: 0,
    }));
  }

  // Saltelli 2010 first-order estimator: S_i = (1/N) Σ fB * (fABi - fA) / Var
  const rawIndices = new Float64Array(d);
  for (let i = 0; i < d; i++) {
    let sum = 0;
    for (let j = 0; j < N; j++) {
      sum += fB[j] * (fABs[i][j] - fA[j]);
    }
    rawIndices[i] = Math.max(0, sum / (N * varY));
  }

  const maxIdx = Math.max(...rawIndices);
  return predictor.paramNames
    .map((name, i) => ({
      dimIndex: i,
      paramName: name,
      importance: maxIdx > 0 ? rawIndices[i] / maxIdx : 0,
      raw: rawIndices[i],
    }))
    .sort((a, b) => b.importance - a.importance);
}

// ---------------------------------------------------------------------------
// 3. Gradient-based importance (DGSM)
// ---------------------------------------------------------------------------

/**
 * Derivative-based Global Sensitivity Measure (DGSM).
 *
 * Importance_i = E[(∂μ/∂x_i)²] estimated by averaging squared central
 * finite differences over random sample points.
 */
export function computeGradientImportance(
  predictor: SensitivityPredictor,
  outcomeName: string,
  options?: SensitivityOptions,
): ParameterImportance[] {
  const N = options?.numSamples ?? 256;
  const seed = options?.seed ?? 42;
  const d = predictor.paramNames.length;
  if (d === 0) return [];

  const rng = new Rng(seed);
  const bounds = predictor.paramBounds;
  const specs = predictor.paramSpecs;

  // Step sizes per dimension
  const h = new Float64Array(d);
  const isChoice = new Uint8Array(d);
  for (let i = 0; i < d; i++) {
    if (specs?.[i]?.type === "choice") {
      isChoice[i] = 1;
    } else {
      h[i] = 1e-5 * (bounds[i][1] - bounds[i][0]);
      if (h[i] < 1e-15) h[i] = 1e-8; // fallback for zero-width bounds
    }
  }

  // Generate sample points
  const basePoints = generateSampleMatrix(N, d, bounds, specs, rng);

  // For each continuous dimension, compute mean squared gradient via finite differences.
  // Build perturbed points for all dims at once to minimize predict() calls.
  const rawImportance = new Float64Array(d);

  // Evaluate base points once
  const fBase = evalMean(predictor, basePoints, outcomeName);

  for (let i = 0; i < d; i++) {
    if (isChoice[i]) continue;

    // Build +h and -h perturbed copies
    const plusPts: number[][] = [];
    const minusPts: number[][] = [];
    for (let j = 0; j < N; j++) {
      const pp = basePoints[j].slice();
      const mp = basePoints[j].slice();
      pp[i] = Math.min(bounds[i][1], pp[i] + h[i]);
      mp[i] = Math.max(bounds[i][0], mp[i] - h[i]);
      plusPts.push(pp);
      minusPts.push(mp);
    }

    const fPlus = evalMean(predictor, plusPts, outcomeName);
    const fMinus = evalMean(predictor, minusPts, outcomeName);

    let sumSq = 0;
    for (let j = 0; j < N; j++) {
      const actualH = (plusPts[j][i] - minusPts[j][i]);
      if (actualH > 0) {
        const grad = (fPlus[j] - fMinus[j]) / actualH;
        sumSq += grad * grad;
      }
    }
    // Scale by domain width² to make it domain-independent (matches Sobol's variance fraction)
    const width = bounds[i][1] - bounds[i][0];
    rawImportance[i] = (sumSq / N) * (width * width);
  }

  const maxImp = Math.max(...rawImportance);
  return predictor.paramNames
    .map((name, i) => ({
      dimIndex: i,
      paramName: name,
      importance: maxImp > 0 ? rawImportance[i] / maxImp : 0,
      raw: rawImportance[i],
    }))
    .sort((a, b) => b.importance - a.importance);
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Generate N random points in the parameter space. */
function generateSampleMatrix(
  N: number,
  d: number,
  bounds: readonly [number, number][],
  specs: ReadonlyArray<{ type: string; values?: (string | number | boolean)[] }> | undefined,
  rng: Rng,
): number[][] {
  const pts: number[][] = [];
  for (let j = 0; j < N; j++) {
    const row: number[] = [];
    for (let i = 0; i < d; i++) {
      if (specs?.[i]?.type === "choice" && specs[i].values) {
        // Uniformly pick one of the discrete values (as numeric index)
        const vals = specs[i].values!;
        const idx = Math.floor(rng.uniform() * vals.length);
        row.push(typeof vals[idx] === "number" ? (vals[idx] as number) : idx);
      } else {
        const lo = bounds[i][0];
        const hi = bounds[i][1];
        row.push(lo + rng.uniform() * (hi - lo));
      }
    }
    pts.push(row);
  }
  return pts;
}

/** Evaluate GP posterior mean at a batch of points for a single outcome. */
function evalMean(
  predictor: SensitivityPredictor,
  points: number[][],
  outcomeName: string,
): Float64Array {
  const preds = predictor.predict(points);
  return preds[outcomeName].mean;
}

/** Sample variance of a Float64Array. */
function variance(arr: Float64Array): number {
  const n = arr.length;
  if (n < 2) return 0;
  let sum = 0;
  for (let i = 0; i < n; i++) sum += arr[i];
  const mean = sum / n;
  let ss = 0;
  for (let i = 0; i < n; i++) {
    const d = arr[i] - mean;
    ss += d * d;
  }
  return ss / (n - 1);
}
