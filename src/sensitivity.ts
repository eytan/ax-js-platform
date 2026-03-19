import { Rng } from "./acquisition/sample_mvn.js";
import type { SearchSpaceParam } from "./models/types.js";

/** Sobol' sensitivity analysis results for a GP posterior mean. */
export interface SensitivityIndices {
  /** First-order Sobol' index per continuous dimension. */
  firstOrder: number[];
  /** Total-order Sobol' index per continuous dimension. */
  totalOrder: number[];
  /** Parameter names corresponding to each index. */
  paramNames: string[];
  /** Total number of function evaluations used. */
  numEvaluations: number;
}

/** Options for Saltelli-based Sobol' analysis. */
export interface SaltelliOptions {
  numSamples?: number; // Default: 512
  seed?: number; // Default: 42
}

// First 20 primes for Halton sequence bases
const PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71];

/**
 * Generate a scrambled Halton quasi-random sequence in [0,1]^d.
 *
 * Uses Owen-style random digit scrambling via the provided Rng.
 * More uniform than pseudo-random for d <= 20 dimensions.
 */
export function haltonSequence(n: number, d: number, rng?: Rng): number[][] {
  if (d > PRIMES.length) {
    throw new Error(`haltonSequence supports up to ${PRIMES.length} dimensions, got ${d}`);
  }
  const r = rng ?? new Rng(42);
  const result: number[][] = [];

  // Precompute per-dimension scrambling permutations
  const permutations: number[][][] = [];
  for (let dim = 0; dim < d; dim++) {
    const base = PRIMES[dim];
    // For each digit position (up to ~20 digits), create a random permutation of [0, base-1]
    const dimPerms: number[][] = [];
    for (let digitPos = 0; digitPos < 20; digitPos++) {
      const perm: number[] = [];
      for (let k = 0; k < base; k++) perm.push(k);
      // Fisher-Yates shuffle
      for (let k = base - 1; k > 0; k--) {
        const j = Math.floor(r.uniform() * (k + 1));
        const tmp = perm[k];
        perm[k] = perm[j];
        perm[j] = tmp;
      }
      dimPerms.push(perm);
    }
    permutations.push(dimPerms);
  }

  for (let i = 0; i < n; i++) {
    const point = new Array(d);
    for (let dim = 0; dim < d; dim++) {
      const base = PRIMES[dim];
      let val = 0;
      let denom = 1;
      let idx = i + 1; // 1-based to avoid the origin
      let digitPos = 0;
      while (idx > 0) {
        denom *= base;
        const digit = idx % base;
        // Apply scrambling permutation
        const scrambled = permutations[dim][digitPos % 20][digit];
        val += scrambled / denom;
        idx = Math.floor(idx / base);
        digitPos++;
      }
      point[dim] = val;
    }
    result.push(point);
  }
  return result;
}

/** Separate continuous and categorical dimensions from parameter specs. */
function splitDimensions(paramSpecs: SearchSpaceParam[]): {
  contDims: number[];
  catDims: number[];
} {
  const contDims: number[] = [];
  const catDims: number[] = [];
  for (let i = 0; i < paramSpecs.length; i++) {
    if (paramSpecs[i].type === "choice") {
      catDims.push(i);
    } else {
      contDims.push(i);
    }
  }
  return { contDims, catDims };
}

/**
 * Scale a [0,1] value to the parameter's actual range.
 * Rounds for integer parameters.
 */
function scaleValue(u: number, spec: SearchSpaceParam): number {
  const [lo, hi] = spec.bounds ?? [0, 1];
  const v = lo + u * (hi - lo);
  return spec.parameter_type === "int" ? Math.round(v) : v;
}

/**
 * Generate all categorical combinations, or a random subset if too many.
 * Returns arrays of categorical values indexed by catDims positions.
 */
function categoricalCombinations(
  paramSpecs: SearchSpaceParam[],
  catDims: number[],
  maxCombinations: number,
  rng: Rng,
): number[][] {
  if (catDims.length === 0) return [[]];

  const valueSets = catDims.map((d) => {
    const vals = paramSpecs[d].values;
    if (!vals) return [0];
    // Choice values are encoded as numeric indices in the model
    return vals.map((_, i) => i);
  });

  // Total combinations
  let totalCombos = 1;
  for (const vs of valueSets) {
    totalCombos *= vs.length;
    if (totalCombos > maxCombinations) break;
  }

  if (totalCombos <= maxCombinations) {
    // Enumerate all
    const combos: number[][] = [[]];
    for (const vs of valueSets) {
      const next: number[][] = [];
      for (const combo of combos) {
        for (const v of vs) {
          next.push([...combo, v]);
        }
      }
      combos.length = 0;
      combos.push(...next);
    }
    return combos;
  }

  // Random subset
  const combos: number[][] = [];
  for (let i = 0; i < maxCombinations; i++) {
    const combo = valueSets.map((vs) => vs[Math.floor(rng.uniform() * vs.length)]);
    combos.push(combo);
  }
  return combos;
}

/**
 * Build a full parameter point from continuous values and categorical values.
 */
function buildPoint(
  contValues: number[],
  catValues: number[],
  contDims: number[],
  catDims: number[],
  totalDims: number,
): number[] {
  const pt = new Array(totalDims);
  for (let i = 0; i < contDims.length; i++) {
    pt[contDims[i]] = contValues[i];
  }
  for (let i = 0; i < catDims.length; i++) {
    pt[catDims[i]] = catValues[i];
  }
  return pt;
}

const MAX_BATCH = 4096;

/**
 * Compute Sobol' sensitivity indices using Saltelli's estimator.
 *
 * Uses the GP posterior mean as the function. Integrates over the original
 * parameter space (uniform over [lower, upper] bounds). Categorical parameters
 * are marginalized by averaging over all (or a random subset of) choice values.
 *
 * @param predictFn - Function mapping N points to N posterior means
 * @param paramSpecs - Parameter specifications with bounds and types
 * @param options - Number of samples and random seed
 * @returns First-order and total-order indices per continuous dimension
 */
export function computeSobolIndices(
  predictFn: (points: number[][]) => Float64Array,
  paramSpecs: SearchSpaceParam[],
  options?: SaltelliOptions,
): SensitivityIndices {
  const N = options?.numSamples ?? 512;
  const seed = options?.seed ?? 42;
  const rng = new Rng(seed);
  const { contDims, catDims } = splitDimensions(paramSpecs);
  const dCont = contDims.length;

  if (dCont === 0) {
    return {
      firstOrder: [],
      totalOrder: [],
      paramNames: [],
      numEvaluations: 0,
    };
  }

  // Generate two independent random matrices A and B.
  // We use plain pseudo-random sampling rather than Halton, because
  // Saltelli's estimator requires truly independent A and B matrices,
  // and scrambled Halton with the same index range can introduce subtle
  // correlations that bias the cross-term estimates.
  const A: number[][] = [];
  const B: number[][] = [];
  for (let i = 0; i < N; i++) {
    const rowA: number[] = [];
    const rowB: number[] = [];
    for (let j = 0; j < dCont; j++) {
      rowA.push(scaleValue(rng.uniform(), paramSpecs[contDims[j]]));
      rowB.push(scaleValue(rng.uniform(), paramSpecs[contDims[j]]));
    }
    A.push(rowA);
    B.push(rowB);
  }

  // Construct AB_i matrices: A with column i replaced by B's column i
  // Total points: N*(dCont+2) for continuous dims
  const catCombos = categoricalCombinations(paramSpecs, catDims, 50, rng);
  const nCatCombos = catCombos.length;
  const totalDims = paramSpecs.length;

  // Build all evaluation points: for each categorical combo, we need A, B, and AB_0..AB_{d-1}
  // Strategy: evaluate all continuous sample matrices for each cat combo, then average
  const nMatrices = dCont + 2; // A, B, AB_0, ..., AB_{d-1}
  const pointsPerCombo = N * nMatrices;
  const totalPoints = pointsPerCombo * nCatCombos;

  // Build all points
  const allPoints: number[][] = new Array(totalPoints);
  let pIdx = 0;
  for (let cc = 0; cc < nCatCombos; cc++) {
    const catVals = catCombos[cc];
    // A matrix
    for (let i = 0; i < N; i++) {
      allPoints[pIdx++] = buildPoint(A[i], catVals, contDims, catDims, totalDims);
    }
    // B matrix
    for (let i = 0; i < N; i++) {
      allPoints[pIdx++] = buildPoint(B[i], catVals, contDims, catDims, totalDims);
    }
    // AB_j matrices
    for (let j = 0; j < dCont; j++) {
      for (let i = 0; i < N; i++) {
        const ab = A[i].slice();
        ab[j] = B[i][j];
        allPoints[pIdx++] = buildPoint(ab, catVals, contDims, catDims, totalDims);
      }
    }
  }

  // Evaluate in chunks
  const allY = new Float64Array(totalPoints);
  for (let start = 0; start < totalPoints; start += MAX_BATCH) {
    const end = Math.min(start + MAX_BATCH, totalPoints);
    const chunk = allPoints.slice(start, end);
    const chunkY = predictFn(chunk);
    allY.set(chunkY, start);
  }

  // Average over categorical combos
  const fA = new Float64Array(N);
  const fB = new Float64Array(N);
  const fAB: Float64Array[] = [];
  for (let j = 0; j < dCont; j++) fAB.push(new Float64Array(N));

  for (let cc = 0; cc < nCatCombos; cc++) {
    const base = cc * pointsPerCombo;
    for (let i = 0; i < N; i++) {
      fA[i] += allY[base + i] / nCatCombos;
      fB[i] += allY[base + N + i] / nCatCombos;
    }
    for (let j = 0; j < dCont; j++) {
      const abBase = base + (2 + j) * N;
      for (let i = 0; i < N; i++) {
        fAB[j][i] += allY[abBase + i] / nCatCombos;
      }
    }
  }

  // Compute variance of combined output
  let sumY = 0;
  let sumY2 = 0;
  for (let i = 0; i < N; i++) {
    sumY += fA[i];
    sumY2 += fA[i] * fA[i];
    sumY += fB[i];
    sumY2 += fB[i] * fB[i];
  }
  const totalN = 2 * N;
  const meanY = sumY / totalN;
  const varY = sumY2 / totalN - meanY * meanY;

  const firstOrder = new Array(dCont);
  const totalOrder = new Array(dCont);

  if (varY < 1e-30) {
    // Constant function — all indices are 0
    for (let j = 0; j < dCont; j++) {
      firstOrder[j] = 0;
      totalOrder[j] = 0;
    }
  } else {
    for (let j = 0; j < dCont; j++) {
      // Saltelli 2010 estimator
      // S_i = (1/N) * sum(fB[i] * (fAB_i[i] - fA[i])) / Var(Y)
      let numS = 0;
      let numST = 0;
      for (let i = 0; i < N; i++) {
        numS += fB[i] * (fAB[j][i] - fA[i]);
        const diff = fA[i] - fAB[j][i];
        numST += diff * diff;
      }
      firstOrder[j] = Math.max(0, numS / (N * varY));
      totalOrder[j] = Math.max(0, numST / (2 * N * varY));
    }
  }

  // Add categorical dimension indices if any
  const allFirstOrder: number[] = [];
  const allTotalOrder: number[] = [];
  const allParamNames: string[] = [];

  // Interleave continuous and categorical results in original parameter order
  let contIdx = 0;
  for (let i = 0; i < paramSpecs.length; i++) {
    if (paramSpecs[i].type === "choice") {
      // For categorical dims, compute conditional variance
      if (catDims.length > 0 && nCatCombos > 1) {
        const catSens = computeCategoricalSensitivity(
          i, catDims, catCombos, allY, N, pointsPerCombo, nCatCombos, varY,
        );
        allFirstOrder.push(catSens);
        allTotalOrder.push(catSens); // For categorical, S ≈ ST (no interaction decomposition)
      } else {
        allFirstOrder.push(0);
        allTotalOrder.push(0);
      }
      allParamNames.push(paramSpecs[i].name);
    } else {
      allFirstOrder.push(firstOrder[contIdx]);
      allTotalOrder.push(totalOrder[contIdx]);
      allParamNames.push(paramSpecs[i].name);
      contIdx++;
    }
  }

  return {
    firstOrder: allFirstOrder,
    totalOrder: allTotalOrder,
    paramNames: allParamNames,
    numEvaluations: totalPoints,
  };
}

/**
 * Compute sensitivity index for a categorical dimension via conditional variance.
 * S_cat = Var_cat(E[f|cat]) / Var(f)
 */
function computeCategoricalSensitivity(
  dimIdx: number,
  catDims: number[],
  catCombos: number[][],
  allY: Float64Array,
  N: number,
  pointsPerCombo: number,
  nCatCombos: number,
  varY: number,
): number {
  if (varY < 1e-30) return 0;

  const catPos = catDims.indexOf(dimIdx);
  if (catPos < 0) return 0;

  // Group categorical combos by the value of this specific dimension
  const groups = new Map<number, number[]>();
  for (let cc = 0; cc < nCatCombos; cc++) {
    const val = catCombos[cc][catPos];
    if (!groups.has(val)) groups.set(val, []);
    groups.get(val)!.push(cc);
  }

  if (groups.size <= 1) return 0;

  // Compute E[f|cat=v] for each value v (using A matrix points only)
  const condMeans: number[] = [];
  const condWeights: number[] = [];
  for (const [, combos] of groups) {
    let sum = 0;
    let count = 0;
    for (const cc of combos) {
      const base = cc * pointsPerCombo;
      for (let i = 0; i < N; i++) {
        sum += allY[base + i]; // A matrix values
      }
      count += N;
    }
    condMeans.push(sum / count);
    condWeights.push(combos.length / nCatCombos);
  }

  // Var_cat(E[f|cat]) = sum_v w_v * (E[f|cat=v] - E[f])^2
  let grandMean = 0;
  for (let k = 0; k < condMeans.length; k++) {
    grandMean += condWeights[k] * condMeans[k];
  }
  let condVar = 0;
  for (let k = 0; k < condMeans.length; k++) {
    const diff = condMeans[k] - grandMean;
    condVar += condWeights[k] * diff * diff;
  }

  return Math.max(0, condVar / varY);
}
