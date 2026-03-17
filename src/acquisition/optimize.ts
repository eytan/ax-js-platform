import type { AcquisitionFunction, Bounds, OptimizeResult } from "./types.js";
import { Rng } from "./sample_mvn.js";

/**
 * Optimize an acquisition function via random search + optional L-BFGS refinement.
 *
 * Mirrors BoTorch's optimize_acqf pattern:
 * 1. Generate `rawSamples` random candidates in bounds
 * 2. Evaluate AF at all candidates
 * 3. Pick top `numRestarts` candidates as starting points
 * 4. (If useLBFGS) Run projected L-BFGS from each starting point
 * 5. Return the best result
 *
 * @param acqf        - Acquisition function to optimize
 * @param bounds      - Bounds for each dimension [[lo,hi], ...]
 * @param opts        - Optimization options
 * @returns Best candidate found
 */
export function optimizeAcqf(
  acqf: AcquisitionFunction,
  bounds: Bounds,
  opts: OptimizeAcqfOptions = {},
): OptimizeResult {
  const {
    rawSamples = 1024,
    numRestarts = 10,
    seed = 42,
    useLBFGS = false,
    lbfgsOptions = {},
    returnAll = false,
  } = opts;

  const d = bounds.length;
  const rng = new Rng(seed);

  // Stage 1: Generate random candidates and evaluate
  const candidates = generateCandidates(d, bounds, rawSamples, rng);
  const values = acqf.evaluate(candidates);

  // Stage 2: Find top-K starting points
  const indices = argsort(values);
  const topK = Math.min(numRestarts, candidates.length);

  let bestPoint = candidates[indices[0]];
  let bestValue = values[indices[0]];

  if (useLBFGS) {
    // Stage 3: Run L-BFGS from each starting point
    for (let k = 0; k < topK; k++) {
      const startPoint = candidates[indices[k]];
      const result = lbfgsOptimize(acqf, startPoint, bounds, lbfgsOptions);
      if (result.value > bestValue) {
        bestValue = result.value;
        bestPoint = result.point;
      }
    }
  }

  if (returnAll) {
    // Sort all candidates by value (descending)
    const sortedCandidates = indices.map((i) => candidates[i]);
    const sortedValues = new Float64Array(indices.length);
    for (let i = 0; i < indices.length; i++) {
      sortedValues[i] = values[indices[i]];
    }
    return {
      point: bestPoint,
      value: bestValue,
      candidates: sortedCandidates,
      values: sortedValues,
    };
  }

  return { point: bestPoint, value: bestValue };
}

export interface OptimizeAcqfOptions {
  /** Number of random candidates to evaluate (default 1024). */
  rawSamples?: number;
  /** Number of top candidates to use as L-BFGS starting points (default 10). */
  numRestarts?: number;
  /** RNG seed (default 42). */
  seed?: number;
  /** Whether to refine with L-BFGS (default false — random search only). */
  useLBFGS?: boolean;
  /** Options for L-BFGS refinement. */
  lbfgsOptions?: LBFGSOptions;
  /** Whether to return all evaluated candidates (default false). */
  returnAll?: boolean;
}

export interface LBFGSOptions {
  /** Maximum iterations (default 50). */
  maxIter?: number;
  /** Finite difference step size (default 1e-5). */
  fdEps?: number;
  /** L-BFGS history size (default 10). */
  historySize?: number;
  /** Convergence tolerance on function value change (default 1e-8). */
  ftol?: number;
  /** Initial step size (default 1.0). */
  lr?: number;
}

// ─── Random candidate generation ─────────────────────────────────────────

function generateCandidates(
  d: number,
  bounds: Bounds,
  n: number,
  rng: Rng,
): number[][] {
  const candidates: number[][] = [];
  for (let i = 0; i < n; i++) {
    const point = new Array(d);
    for (let j = 0; j < d; j++) {
      const [lo, hi] = bounds[j];
      point[j] = lo + rng.uniform() * (hi - lo);
    }
    candidates.push(point);
  }
  return candidates;
}

// ─── Argsort (descending) ────────────────────────────────────────────────

function argsort(values: Float64Array): number[] {
  const indices = Array.from({ length: values.length }, (_, i) => i);
  indices.sort((a, b) => values[b] - values[a]);
  return indices;
}

// ─── Projected L-BFGS ───────────────────────────────────────────────────

/**
 * L-BFGS with bound projection and finite-difference gradients.
 *
 * Maximizes the acquisition function (negates internally for minimization).
 * After each step, projects back into the feasible box.
 *
 * Uses the two-loop recursion for the L-BFGS direction and
 * Armijo backtracking line search for step size selection.
 */
function lbfgsOptimize(
  acqf: AcquisitionFunction,
  x0: number[],
  bounds: Bounds,
  opts: LBFGSOptions = {},
): { point: number[]; value: number } {
  const {
    maxIter = 50,
    fdEps = 1e-5,
    historySize = 10,
    ftol = 1e-8,
    lr = 1.0,
  } = opts;

  const d = x0.length;
  let x = projectToBounds(x0.slice(), bounds);
  let fx = evalSingle(acqf, x);

  // L-BFGS history buffers
  const S: number[][] = []; // s_k = x_{k+1} - x_k
  const Y: number[][] = []; // y_k = g_{k+1} - g_k
  const rhos: number[] = [];

  let gPrev = finiteDiffGradient(acqf, x, bounds, fdEps);

  for (let iter = 0; iter < maxIter; iter++) {
    // Compute L-BFGS direction via two-loop recursion
    const dir = lbfgsDirection(gPrev, S, Y, rhos, historySize);

    // Negate for ascent (we're maximizing)
    for (let j = 0; j < d; j++) dir[j] = -dir[j];

    // Backtracking line search (Armijo condition for ascent)
    let step = lr;
    const dirDotGrad = dotProduct(dir, gPrev);
    let xNew = projectToBounds(stepAdd(x, dir, step), bounds);
    let fNew = evalSingle(acqf, xNew);

    for (let ls = 0; ls < 20; ls++) {
      if (fNew >= fx + 1e-4 * step * dirDotGrad) break;
      step *= 0.5;
      xNew = projectToBounds(stepAdd(x, dir, step), bounds);
      fNew = evalSingle(acqf, xNew);
    }

    // Check convergence
    if (Math.abs(fNew - fx) < ftol * (1 + Math.abs(fx))) {
      if (fNew > fx) {
        x = xNew;
        fx = fNew;
      }
      break;
    }

    // Update history
    const gNew = finiteDiffGradient(acqf, xNew, bounds, fdEps);
    const s = new Array(d);
    const y = new Array(d);
    for (let j = 0; j < d; j++) {
      s[j] = xNew[j] - x[j];
      y[j] = gNew[j] - gPrev[j];
    }
    const sy = dotProduct(s, y);
    if (sy > 1e-10) {
      // Only update if curvature condition is satisfied
      if (S.length >= historySize) {
        S.shift();
        Y.shift();
        rhos.shift();
      }
      S.push(s);
      Y.push(y);
      rhos.push(1 / sy);
    }

    x = xNew;
    fx = fNew;
    gPrev = gNew;
  }

  return { point: x, value: fx };
}

/**
 * L-BFGS two-loop recursion to compute the search direction.
 * Returns the negative gradient preconditioned by the L-BFGS Hessian approximation.
 */
function lbfgsDirection(
  g: number[],
  S: number[][],
  Y: number[][],
  rhos: number[],
  _historySize: number,
): number[] {
  const d = g.length;
  const m = S.length;

  // If no history, return steepest descent direction
  if (m === 0) return g.slice();

  const q = g.slice();
  const alphas = new Array(m);

  // Backward pass
  for (let i = m - 1; i >= 0; i--) {
    alphas[i] = rhos[i] * dotProduct(S[i], q);
    for (let j = 0; j < d; j++) {
      q[j] -= alphas[i] * Y[i][j];
    }
  }

  // Scale by initial Hessian approximation: H0 = (s·y)/(y·y) · I
  const yLast = Y[m - 1];
  const sLast = S[m - 1];
  const yy = dotProduct(yLast, yLast);
  const gamma = dotProduct(sLast, yLast) / (yy + 1e-300);
  for (let j = 0; j < d; j++) {
    q[j] *= gamma;
  }

  // Forward pass
  for (let i = 0; i < m; i++) {
    const beta = rhos[i] * dotProduct(Y[i], q);
    for (let j = 0; j < d; j++) {
      q[j] += (alphas[i] - beta) * S[i][j];
    }
  }

  return q;
}

// ─── Finite-difference gradient ──────────────────────────────────────────

function finiteDiffGradient(
  acqf: AcquisitionFunction,
  x: number[],
  bounds: Bounds,
  eps: number,
): number[] {
  const d = x.length;
  const grad = new Array(d);

  for (let j = 0; j < d; j++) {
    const xp = x.slice();
    const xm = x.slice();
    // Respect bounds: use one-sided differences at boundaries
    const lo = bounds[j][0];
    const hi = bounds[j][1];
    const step = Math.min(eps, (hi - lo) * 0.01);

    if (x[j] + step > hi) {
      // Backward difference at upper bound
      xm[j] -= step;
      const fp = evalSingle(acqf, x);
      const fm = evalSingle(acqf, xm);
      grad[j] = (fp - fm) / step;
    } else if (x[j] - step < lo) {
      // Forward difference at lower bound
      xp[j] += step;
      const fp = evalSingle(acqf, xp);
      const fm = evalSingle(acqf, x);
      grad[j] = (fp - fm) / step;
    } else {
      // Central difference
      xp[j] += step;
      xm[j] -= step;
      const fp = evalSingle(acqf, xp);
      const fm = evalSingle(acqf, xm);
      grad[j] = (fp - fm) / (2 * step);
    }
  }

  return grad;
}

// ─── Helpers ─────────────────────────────────────────────────────────────

function evalSingle(acqf: AcquisitionFunction, x: number[]): number {
  return acqf.evaluate([x])[0];
}

function projectToBounds(x: number[], bounds: Bounds): number[] {
  for (let j = 0; j < x.length; j++) {
    x[j] = Math.max(bounds[j][0], Math.min(bounds[j][1], x[j]));
  }
  return x;
}

function stepAdd(x: number[], dir: number[], step: number): number[] {
  const result = new Array(x.length);
  for (let j = 0; j < x.length; j++) {
    result[j] = x[j] + step * dir[j];
  }
  return result;
}

function dotProduct(a: number[], b: number[]): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += a[i] * b[i];
  return sum;
}
