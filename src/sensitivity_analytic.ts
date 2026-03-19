/**
 * Analytic Sobol' sensitivity indices for GP posterior mean.
 *
 * For a GP with posterior mean μ(x) = m + k(x, X)ᵀα and a product kernel,
 * all integrals factor across dimensions into 1D closed forms — giving
 * exact indices in O(d×n²) operations with zero MC noise.
 *
 * Supports: RBF (closed-form), Matérn ν=0.5/1.5/2.5 (quadrature),
 * Categorical (finite sums), Product(RBF|Matérn, Categorical),
 * Additive kernels with disjoint active_dims, and warped dims (quadrature).
 * Also supports EnsembleGP (SAAS) with RBF or Matérn kernels via a generalized
 * cross integral that handles per-model lengthscales.
 * Falls back to null (caller uses MC) for PairwiseGP.
 */
import { normalCdf } from "./math.js";
import type {
  KernelState,
  SearchSpaceParam,
  SensitivityIndices,
} from "./models/types.js";

// ── RBF closed-form primitives ──────────────────────────────────────────

/** RBF marginal: ∫₀¹ exp(-0.5(x-c)²/ℓ²) dx = ℓ√(2π) [Φ((1-c)/ℓ) - Φ(-c/ℓ)] */
export function rbfMarginal(c: number, ell: number): number {
  return (
    ell *
    Math.sqrt(2 * Math.PI) *
    (normalCdf((1 - c) / ell) - normalCdf(-c / ell))
  );
}

/**
 * Generalized RBF cross integral with potentially different lengthscales:
 *   ∫₀¹ exp(-0.5(x-a)²/ℓ₁²) · exp(-0.5(x-b)²/ℓ₂²) dx
 *
 * The product of two Gaussians is Gaussian with:
 *   σ* = ℓ₁ℓ₂/√(ℓ₁²+ℓ₂²),  μ* = (aℓ₂² + bℓ₁²)/(ℓ₁²+ℓ₂²)
 *
 * When ℓ₁ = ℓ₂ = ℓ, reduces to σ* = ℓ/√2, μ* = (a+b)/2.
 */
export function rbfGeneralizedCross(
  a: number,
  b: number,
  ell1: number,
  ell2: number,
): number {
  const s2 = ell1 * ell1 + ell2 * ell2;
  const sigma = (ell1 * ell2) / Math.sqrt(s2);
  const mu = (a * ell2 * ell2 + b * ell1 * ell1) / s2;
  const diff = a - b;
  return (
    Math.exp(-(diff * diff) / (2 * s2)) *
    sigma *
    Math.sqrt(2 * Math.PI) *
    (normalCdf((1 - mu) / sigma) - normalCdf(-mu / sigma))
  );
}

// ── DimIntegrator interface ─────────────────────────────────────────────

/** 1D kernel integral primitives for one input dimension. */
export interface DimIntegrator {
  marginal(c: number): number;
  cross(a: number, b: number): number;
}

// ── RBF integrator ──────────────────────────────────────────────────────

export class RbfDimIntegrator implements DimIntegrator {
  constructor(private ell: number) {}

  marginal(c: number): number {
    return rbfMarginal(c, this.ell);
  }

  cross(a: number, b: number): number {
    return rbfGeneralizedCross(a, b, this.ell, this.ell);
  }
}

// ── Matérn integrator (quadrature) ───────────────────────────────────────

/**
 * 1D Matérn kernel function for a single dimension.
 * - ν = 0.5: k(x, c) = exp(-|x - c| / ℓ)
 * - ν = 1.5: k(x, c) = (1 + √3·|x-c|/ℓ) · exp(-√3·|x-c|/ℓ)
 * - ν = 2.5: k(x, c) = (1 + √5·|x-c|/ℓ + 5(x-c)²/(3ℓ²)) · exp(-√5·|x-c|/ℓ)
 */
function maternKernel1D(
  nu: 0.5 | 1.5 | 2.5,
  ell: number,
): (x: number, c: number) => number {
  if (nu === 0.5) {
    return (x, c) => Math.exp(-Math.abs(x - c) / ell);
  } else if (nu === 1.5) {
    const sqrt3 = Math.sqrt(3);
    return (x, c) => {
      const r = Math.abs(x - c) / ell;
      const sr = sqrt3 * r;
      return (1 + sr) * Math.exp(-sr);
    };
  } else {
    // nu === 2.5
    const sqrt5 = Math.sqrt(5);
    return (x, c) => {
      const d = x - c;
      const r = Math.abs(d) / ell;
      const sr = sqrt5 * r;
      return (1 + sr + (5 * d * d) / (3 * ell * ell)) * Math.exp(-sr);
    };
  }
}

export class MaternDimIntegrator implements DimIntegrator {
  private integrator: QuadratureDimIntegrator;

  constructor(nu: 0.5 | 1.5 | 2.5, ell: number, nNodes: number = 128) {
    this.integrator = new QuadratureDimIntegrator(
      maternKernel1D(nu, ell),
      nNodes,
    );
  }

  marginal(c: number): number {
    return this.integrator.marginal(c);
  }

  cross(a: number, b: number): number {
    return this.integrator.cross(a, b);
  }
}

// ── Categorical integrator (finite sums) ────────────────────────────────

export class CategoricalDimIntegrator implements DimIntegrator {
  private nCategories: number;
  private expNeg: number; // exp(-1/(d·ℓ))

  /**
   * @param nCategories Number of category values for this dimension
   * @param lengthscale ARD lengthscale for this categorical dimension
   * @param numCatDims Total number of categorical dimensions in the kernel
   */
  constructor(nCategories: number, lengthscale: number, numCatDims: number) {
    this.nCategories = nCategories;
    // CategoricalKernel: k(v, c) = exp(-indicator(v≠c) / (d·ℓ))
    this.expNeg = Math.exp(-1.0 / (numCatDims * lengthscale));
  }

  marginal(_c: number): number {
    // (1/K) [1 + (K-1) exp(-1/(d·ℓ))]
    const K = this.nCategories;
    return (1 + (K - 1) * this.expNeg) / K;
  }

  cross(a: number, b: number): number {
    const K = this.nCategories;
    const e2 = this.expNeg * this.expNeg;
    if (Math.abs(a - b) < 1e-8) {
      // a == b: (1/K) [1 + (K-1) exp(-2/(d·ℓ))]
      return (1 + (K - 1) * e2) / K;
    }
    // a != b: (1/K) [2 exp(-1/(d·ℓ)) + (K-2) exp(-2/(d·ℓ))]
    return (2 * this.expNeg + (K - 2) * e2) / K;
  }
}

// ── Gauss-Legendre quadrature ───────────────────────────────────────────

/**
 * Compute Gauss-Legendre nodes and weights on [0, 1].
 * Uses Newton's method on Legendre polynomials.
 */
export function gaussLegendre01(n: number): {
  nodes: Float64Array;
  weights: Float64Array;
} {
  const nodes = new Float64Array(n);
  const weights = new Float64Array(n);
  const m = Math.ceil(n / 2);

  for (let i = 0; i < m; i++) {
    // Initial guess (Chebyshev approximation)
    let x = Math.cos(Math.PI * (i + 0.75) / (n + 0.5));

    // Newton iteration to find root of P_n(x)
    for (let iter = 0; iter < 50; iter++) {
      let p0 = 1.0;
      let p1 = x;
      for (let j = 2; j <= n; j++) {
        const p2 = ((2 * j - 1) * x * p1 - (j - 1) * p0) / j;
        p0 = p1;
        p1 = p2;
      }
      // p1 = P_n(x), p0 = P_{n-1}(x)
      // P'_n(x) = n(P_{n-1}(x) - x·P_n(x)) / (1 - x²)
      const dp = (n * (p0 - x * p1)) / (1 - x * x);
      const dx = p1 / dp;
      x -= dx;
      if (Math.abs(dx) < 1e-15) break;
    }

    // Recompute P_{n-1}(x) at converged root for weight
    let p0 = 1.0;
    let p1 = x;
    for (let j = 2; j <= n; j++) {
      const p2 = ((2 * j - 1) * x * p1 - (j - 1) * p0) / j;
      p0 = p1;
      p1 = p2;
    }
    // At root: P_n(x) ≈ 0, so P'_n(x) = n·P_{n-1}(x) / (1-x²)
    const dp = (n * p0) / (1 - x * x);
    const w = 2.0 / ((1 - x * x) * dp * dp);

    // Map [-1,1] → [0,1]: node = (1±x)/2, weight *= 0.5
    const j = n - 1 - i;
    nodes[i] = (1 - x) / 2;
    nodes[j] = (1 + x) / 2;
    weights[i] = w / 2;
    weights[j] = w / 2;
  }

  return { nodes, weights };
}

// ── Quadrature integrator (for warped dims) ─────────────────────────────

export class QuadratureDimIntegrator implements DimIntegrator {
  private kernelFn: (x: number, c: number) => number;
  private nodes: Float64Array;
  private weights: Float64Array;

  constructor(
    kernelFn: (x: number, c: number) => number,
    nNodes: number = 32,
  ) {
    this.kernelFn = kernelFn;
    const gl = gaussLegendre01(nNodes);
    this.nodes = gl.nodes;
    this.weights = gl.weights;
  }

  marginal(c: number): number {
    let sum = 0;
    for (let i = 0; i < this.nodes.length; i++) {
      sum += this.weights[i] * this.kernelFn(this.nodes[i], c);
    }
    return sum;
  }

  cross(a: number, b: number): number {
    let sum = 0;
    for (let i = 0; i < this.nodes.length; i++) {
      const x = this.nodes[i];
      sum += this.weights[i] * this.kernelFn(x, a) * this.kernelFn(x, b);
    }
    return sum;
  }
}

// ── Kernel component (one term in an additive decomposition) ────────────

interface KernelComponent {
  activeDims: number[];
  integrators: DimIntegrator[]; // same order as activeDims
  outputscale: number;
}

// ── Warp parameters ────────────────────────────────────────────────────

interface WarpParams {
  concentration0: number[];
  concentration1: number[];
  indices?: number[];
}

// ── Build integrators from kernel state ─────────────────────────────────

/**
 * Walk a kernel state tree and extract DimIntegrators per dimension.
 * Returns null if the kernel is not decomposable.
 *
 * Supported structures:
 * - Scale(RBF) — closed-form integrals
 * - Scale(Matern) — quadrature-based integrals (ν = 0.5, 1.5, 2.5)
 * - Scale(Product(RBF|Matern, Categorical))
 * - Scale(Additive(Scale(RBF|Matern), ...)) — disjoint active_dims
 * - Any of above + input warp → quadrature for warped dims
 *
 * Unsupported (returns null → MC fallback):
 * - PairwiseGP (Laplace posterior, not standard GP form)
 */
export function extractKernelComponents(
  kernelState: KernelState,
  paramSpecs: SearchSpaceParam[],
  warpParams?: WarpParams,
): KernelComponent[] | null {
  // Handle Scale wrapper only — legacy outputscale (on RBF/Matern directly)
  // is handled by extractSingleComponent to avoid double-counting
  let outputscale = 1.0;
  let baseKernel = kernelState;
  if (kernelState.type === "Scale" && kernelState.base_kernel) {
    outputscale = kernelState.outputscale ?? 1.0;
    baseKernel = kernelState.base_kernel;
  }

  if (baseKernel.type === "Additive" && baseKernel.kernels) {
    // Additive: each sub-kernel is an independent component.
    // Requires disjoint active_dims across components.
    const components: KernelComponent[] = [];
    for (const sub of baseKernel.kernels) {
      const comp = extractSingleComponent(
        sub,
        paramSpecs,
        warpParams,
        outputscale,
      );
      if (!comp) return null;
      components.push(comp);
    }
    // Verify disjoint active_dims
    const seen = new Set<number>();
    for (const comp of components) {
      for (const d of comp.activeDims) {
        if (seen.has(d)) return null; // overlapping dims → fall back to MC
        seen.add(d);
      }
    }
    return components;
  }

  // Single component (Product, RBF, etc.)
  const comp = extractSingleComponent(
    baseKernel,
    paramSpecs,
    warpParams,
    outputscale,
  );
  if (!comp) return null;
  return [comp];
}

/**
 * Extract a single kernel component from a non-additive kernel state.
 * Handles RBF, Product(RBF, Categorical), Scale(RBF), etc.
 */
function extractSingleComponent(
  kernelState: KernelState,
  paramSpecs: SearchSpaceParam[],
  warpParams: WarpParams | undefined,
  parentOutputscale: number,
): KernelComponent | null {
  // Handle Scale wrapper on this component
  let outputscale = parentOutputscale;
  let baseKernel = kernelState;
  if (kernelState.type === "Scale" && kernelState.base_kernel) {
    outputscale *= kernelState.outputscale ?? 1.0;
    baseKernel = kernelState.base_kernel;
  } else if (
    kernelState.type !== "Scale" &&
    kernelState.outputscale !== undefined
  ) {
    outputscale *= kernelState.outputscale;
  }

  const d = paramSpecs.length;
  const warpIndicesSet = warpParams?.indices
    ? new Set(warpParams.indices)
    : null;
  const hasWarp = !!warpParams;

  if (baseKernel.type === "RBF" || baseKernel.type === "Matern") {
    const ls = baseKernel.lengthscale;
    if (!ls) return null;
    const activeDims =
      baseKernel.active_dims ?? Array.from({ length: d }, (_, i) => i);
    const integrators: DimIntegrator[] = [];
    const nu = baseKernel.nu as 0.5 | 1.5 | 2.5 | undefined;

    for (let k = 0; k < activeDims.length; k++) {
      const j = activeDims[k];
      const ell = ls[k];
      if (isWarpedDim(j, hasWarp, warpIndicesSet)) {
        if (baseKernel.type === "Matern" && nu !== undefined) {
          integrators.push(
            makeWarpedMaternIntegrator(j, nu, ell, warpParams!),
          );
        } else {
          integrators.push(
            makeWarpedRbfIntegrator(j, ell, warpParams!),
          );
        }
      } else {
        if (baseKernel.type === "Matern" && nu !== undefined) {
          integrators.push(new MaternDimIntegrator(nu, ell));
        } else {
          integrators.push(new RbfDimIntegrator(ell));
        }
      }
    }

    return { activeDims, integrators, outputscale };
  }

  if (baseKernel.type === "Product" && baseKernel.kernels) {
    const activeDims: number[] = [];
    const integrators: DimIntegrator[] = [];

    for (const sub of baseKernel.kernels) {
      // Unwrap Scale if present on sub-kernel
      let subKernel = sub;
      if (sub.type === "Scale" && sub.base_kernel) {
        outputscale *= sub.outputscale ?? 1.0;
        subKernel = sub.base_kernel;
      } else if (sub.type !== "Scale" && sub.outputscale !== undefined) {
        outputscale *= sub.outputscale;
      }

      if (subKernel.type === "RBF" || subKernel.type === "Matern") {
        const ls = subKernel.lengthscale;
        if (!ls) return null;
        const dims =
          subKernel.active_dims ??
          sub.active_dims ??
          Array.from({ length: ls.length }, (_, i) => i);
        const nu = subKernel.nu as 0.5 | 1.5 | 2.5 | undefined;
        for (let k = 0; k < dims.length; k++) {
          const j = dims[k];
          const ell = ls[k];
          activeDims.push(j);
          if (isWarpedDim(j, hasWarp, warpIndicesSet)) {
            if (subKernel.type === "Matern" && nu !== undefined) {
              integrators.push(
                makeWarpedMaternIntegrator(j, nu, ell, warpParams!),
              );
            } else {
              integrators.push(
                makeWarpedRbfIntegrator(j, ell, warpParams!),
              );
            }
          } else {
            if (subKernel.type === "Matern" && nu !== undefined) {
              integrators.push(new MaternDimIntegrator(nu, ell));
            } else {
              integrators.push(new RbfDimIntegrator(ell));
            }
          }
        }
      } else if (subKernel.type === "Categorical") {
        const dims =
          subKernel.active_dims ?? sub.active_dims ?? [];
        const numCatDims = dims.length;
        const ls = subKernel.lengthscale ?? [1];
        for (let k = 0; k < dims.length; k++) {
          const j = dims[k];
          const spec = paramSpecs[j];
          const nCats = spec.values?.length ?? 2;
          const ell = ls.length > 1 ? ls[k] : ls[0];
          activeDims.push(j);
          integrators.push(
            new CategoricalDimIntegrator(nCats, ell, numCatDims),
          );
        }
      } else {
        // Unsupported sub-kernel type
        return null;
      }
    }

    return { activeDims, integrators, outputscale };
  }

  // Categorical at top-level, unsupported kernel types, etc.
  return null;
}

function isWarpedDim(
  dim: number,
  hasWarp: boolean,
  warpIndicesSet: Set<number> | null,
): boolean {
  if (!hasWarp) return false;
  // If no indices specified, all dims are warped
  if (warpIndicesSet === null) return true;
  return warpIndicesSet.has(dim);
}

function makeWarpedMaternIntegrator(
  dim: number,
  nu: 0.5 | 1.5 | 2.5,
  ell: number,
  warpParams: WarpParams,
): QuadratureDimIntegrator {
  const wIdx = warpParams.indices
    ? warpParams.indices.indexOf(dim)
    : dim;
  const a = warpParams.concentration1[wIdx]; // Kumaraswamy a
  const b = warpParams.concentration0[wIdx]; // Kumaraswamy b
  const eps = 1e-7;
  const range = 1 - 2 * eps;
  const kFn = maternKernel1D(nu, ell);

  return new QuadratureDimIntegrator((x, c) => {
    const xn = Math.max(eps, Math.min(1 - eps, x * range + eps));
    const warped = 1 - Math.pow(1 - Math.pow(xn, a), b);
    return kFn(warped, c);
  });
}

function makeWarpedRbfIntegrator(
  dim: number,
  ell: number,
  warpParams: WarpParams,
): QuadratureDimIntegrator {
  const wIdx = warpParams.indices
    ? warpParams.indices.indexOf(dim)
    : dim;
  const a = warpParams.concentration1[wIdx]; // Kumaraswamy a
  const b = warpParams.concentration0[wIdx]; // Kumaraswamy b
  const eps = 1e-7;
  const range = 1 - 2 * eps;

  return new QuadratureDimIntegrator((x, c) => {
    // Warp x through Kumaraswamy CDF, then evaluate RBF
    const xn = Math.max(eps, Math.min(1 - eps, x * range + eps));
    const warped = 1 - Math.pow(1 - Math.pow(xn, a), b);
    const diff = warped - c;
    return Math.exp(-0.5 * diff * diff / (ell * ell));
  });
}

// ── Main computation ────────────────────────────────────────────────────

/**
 * Compute exact Sobol' sensitivity indices for a GP posterior mean
 * with a product kernel (or additive kernel with disjoint components).
 *
 * The GP posterior mean is:
 *   μ(x) = m + Σ_a σ_a² Σ_i α_i ∏_{j∈S_a} k_j(x_j, X_{ij})
 *
 * For a single component (product kernel):
 *   μ(x) = m + σ² Σ_i α_i ∏_j k_j(x_j, X_{ij})
 *
 * Two 1D primitives per dimension suffice:
 *   w_j(c) = ∫₀¹ k_j(x, c) dx        (marginal)
 *   W_j(a,b) = ∫₀¹ k_j(x,a)k_j(x,b) dx  (cross)
 */
export function computeAnalyticSobolIndices(
  alpha: Float64Array,
  trainX: number[][],
  components: KernelComponent[],
  meanConstant: number,
  paramNames: string[],
): SensitivityIndices {
  const n = alpha.length;
  const d = paramNames.length;

  // Map each dim to its component index (-1 if not in any component)
  const dimToComp = new Int32Array(d).fill(-1);
  const dimToLocalIdx = new Int32Array(d).fill(-1);
  for (let ci = 0; ci < components.length; ci++) {
    const comp = components[ci];
    for (let li = 0; li < comp.activeDims.length; li++) {
      dimToComp[comp.activeDims[li]] = ci;
      dimToLocalIdx[comp.activeDims[li]] = li;
    }
  }

  // ── Per-component precomputation ──
  const compData = components.map((comp) => {
    const nDims = comp.activeDims.length;
    const os = comp.outputscale;

    // w[localDim][i] = integrators[localDim].marginal(trainX[i][globalDim])
    const w: Float64Array[] = new Array(nDims);
    for (let li = 0; li < nDims; li++) {
      const gj = comp.activeDims[li];
      w[li] = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        w[li][i] = comp.integrators[li].marginal(trainX[i][gj]);
      }
    }

    // W[localDim][i*n+k] = integrators[localDim].cross(trainX[i][gj], trainX[k][gj])
    const W: Float64Array[] = new Array(nDims);
    for (let li = 0; li < nDims; li++) {
      const gj = comp.activeDims[li];
      W[li] = new Float64Array(n * n);
      for (let i = 0; i < n; i++) {
        for (let k = i; k < n; k++) {
          const val = comp.integrators[li].cross(
            trainX[i][gj],
            trainX[k][gj],
          );
          W[li][i * n + k] = val;
          W[li][k * n + i] = val;
        }
      }
    }

    const productW = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let prod = 1.0;
      for (let li = 0; li < nDims; li++) prod *= w[li][i];
      productW[i] = prod;
    }

    let sumAlphaM = 0;
    for (let i = 0; i < n; i++) sumAlphaM += alpha[i] * os * productW[i];

    let crossSum = 0;
    for (let i = 0; i < n; i++) {
      for (let k = 0; k < n; k++) {
        let prod = 1.0;
        for (let li = 0; li < nDims; li++) prod *= W[li][i * n + k];
        crossSum += alpha[i] * alpha[k] * prod;
      }
    }
    const varA = os * os * crossSum - sumAlphaM * sumAlphaM;

    return { w, W, productW, sumAlphaM, varA, os, nDims };
  });

  // ── E[μ] and Var(μ) ──
  let emu = meanConstant;
  for (let ci = 0; ci < components.length; ci++) {
    emu += compData[ci].sumAlphaM;
  }

  let varMu = 0;
  for (let ci = 0; ci < components.length; ci++) {
    varMu += compData[ci].varA;
  }

  if (varMu < 1e-30) {
    return {
      firstOrder: new Array(d).fill(0),
      totalOrder: new Array(d).fill(0),
      paramNames,
      numEvaluations: 0,
    };
  }

  // ── Per-dimension S_i and ST_i ──
  const firstOrder = new Array(d).fill(0);
  const totalOrder = new Array(d).fill(0);

  for (let gj = 0; gj < d; gj++) {
    const ci = dimToComp[gj];
    if (ci < 0) {
      // Dim not in any component → zero sensitivity
      continue;
    }

    const li = dimToLocalIdx[gj];
    const comp = components[ci];
    const cd = compData[ci];
    const os = cd.os;

    // ── First-order S_i ──
    // β_m = σ² α_m ∏_{j≠dim, j∈S_a} w_j[m]
    // V_i = Σ_{mn} β_m β_n W_dim[m,n] - (Σ_m α_m M_m)²
    // S_i = V_i / Var(μ)
    const beta = new Float64Array(n);
    for (let m = 0; m < n; m++) {
      const wDim = cd.w[li][m];
      if (Math.abs(wDim) < 1e-300) {
        beta[m] = 0;
      } else {
        beta[m] = os * alpha[m] * cd.productW[m] / wDim;
      }
    }

    let crossS = 0;
    for (let m = 0; m < n; m++) {
      for (let p = 0; p < n; p++) {
        crossS += beta[m] * beta[p] * cd.W[li][m * n + p];
      }
    }
    firstOrder[gj] = Math.max(
      0,
      (crossS - cd.sumAlphaM * cd.sumAlphaM) / varMu,
    );

    // ── Total-order ST_i ──
    // γ_m = σ² α_m w_dim[m]
    // V_{~i} = variance of E_{x_i}[μ(x)] over remaining dims
    //        = Var_reduced_a(dim) + Σ_{b≠a} Var_b
    // ST_i = 1 - V_{~i} / Var(μ)

    // Var_reduced_a(dim): variance when dim is marginalized out
    const gamma = new Float64Array(n);
    for (let m = 0; m < n; m++) {
      gamma[m] = os * alpha[m] * cd.w[li][m];
    }

    let crossT = 0;
    for (let m = 0; m < n; m++) {
      for (let p = 0; p < n; p++) {
        // ∏_{j≠dim, j∈S_a} W_j[m,p]
        let prod = 1.0;
        for (let lk = 0; lk < cd.nDims; lk++) {
          if (lk !== li) prod *= cd.W[lk][m * n + p];
        }
        crossT += gamma[m] * gamma[p] * prod;
      }
    }
    let sumGammaW = 0;
    for (let m = 0; m < n; m++) {
      let prod = 1.0;
      for (let lk = 0; lk < cd.nDims; lk++) {
        if (lk !== li) prod *= cd.w[lk][m];
      }
      sumGammaW += gamma[m] * prod;
    }
    const varReduced = crossT - sumGammaW * sumGammaW;

    // V_{~i} = Var_reduced + Σ_{b≠a} Var_b
    let vNotI = varReduced;
    for (let bj = 0; bj < components.length; bj++) {
      if (bj !== ci) vNotI += compData[bj].varA;
    }

    totalOrder[gj] = Math.max(0, 1 - vNotI / varMu);
  }

  return { firstOrder, totalOrder, paramNames, numEvaluations: 0 };
}

// ── Ensemble Sobol' computation ─────────────────────────────────────────

/** Per-model info needed for ensemble analytic Sobol. */
export interface EnsembleSubModelInfo {
  alpha: Float64Array;
  trainXNorm: number[][];
  meanConstant: number;
  lengthscales: number[];
  outputscale: number;
  warpParams?: WarpParams;
  kernelType?: "RBF" | "Matern";
  nu?: 0.5 | 1.5 | 2.5;
}

/**
 * Compute analytic Sobol' indices for an EnsembleGP (e.g., SAAS).
 *
 * The ensemble mean is μ(x) = (1/M) Σ_m μ_m(x), where each μ_m has
 * its own kernel hyperparameters. All cross-model integrals factor
 * per-dimension because the product of two RBF kernels (with different
 * lengthscales) is still Gaussian.
 *
 * For warped dims, uses Gauss-Legendre quadrature. Falls back to null
 * if the pooled basis count exceeds a size limit.
 */
interface BasisFn {
  center: number[];
  coeff: number;
  lengthscales: number[];
  modelIdx: number;
}

export function computeEnsembleAnalyticSobol(
  models: EnsembleSubModelInfo[],
  paramNames: string[],
): SensitivityIndices | null {
  const M = models.length;
  const d = paramNames.length;

  const basis: BasisFn[] = [];
  let meanEns = 0;

  for (let mi = 0; mi < M; mi++) {
    const m = models[mi];
    meanEns += m.meanConstant / M;
    for (let i = 0; i < m.alpha.length; i++) {
      basis.push({
        center: m.trainXNorm[i],
        coeff: (m.outputscale / M) * m.alpha[i],
        lengthscales: m.lengthscales,
        modelIdx: mi,
      });
    }
  }

  const N = basis.length;

  // Size guard: O(d × N²) can be slow for very large ensembles
  if (N > 2000) return null;

  // Determine which dims are warped and precompute warp functions
  const hasAnyWarp = models.some((m) => !!m.warpParams);
  let glNodes: Float64Array | null = null;
  let glWeights: Float64Array | null = null;
  if (hasAnyWarp) {
    const gl = gaussLegendre01(32);
    glNodes = gl.nodes;
    glWeights = gl.weights;
  }

  // Per-model warp function cache (dim → warp function)
  const warpFns: (((x: number) => number) | null)[][] = models.map((m) => {
    if (!m.warpParams) return new Array(d).fill(null);
    const wp = m.warpParams;
    const indices = wp.indices ?? Array.from({ length: d }, (_, i) => i);
    const indicesSet = new Set(indices);
    const eps = 1e-7;
    const range = 1 - 2 * eps;

    return Array.from({ length: d }, (_, j) => {
      if (!indicesSet.has(j)) return null;
      const wIdx = wp.indices ? wp.indices.indexOf(j) : j;
      const a = wp.concentration1[wIdx];
      const b = wp.concentration0[wIdx];
      return (x: number) => {
        const xn = Math.max(eps, Math.min(1 - eps, x * range + eps));
        return 1 - Math.pow(1 - Math.pow(xn, a), b);
      };
    });
  });

  // Check if any model uses Matérn (need quadrature for non-warped dims too)
  const hasAnyMatern = models.some((m) => m.kernelType === "Matern");
  if (hasAnyMatern && !glNodes) {
    const gl = gaussLegendre01(32);
    glNodes = gl.nodes;
    glWeights = gl.weights;
  }

  // Per-model 1D kernel function builders
  function modelKernel1D(mi: number, j: number): (x: number, c: number) => number {
    const m = models[mi];
    const ell = m.lengthscales[j];
    if (m.kernelType === "Matern" && m.nu !== undefined) {
      return maternKernel1D(m.nu, ell);
    }
    // RBF (default)
    return (x, c) => Math.exp(-0.5 * (x - c) * (x - c) / (ell * ell));
  }

  // Helper: 1D marginal for basis function i on dim j
  function marginal1D(i: number, j: number): number {
    const mi = basis[i].modelIdx;
    const wFn = warpFns[mi][j];
    const isMatern = models[mi].kernelType === "Matern";

    if (!wFn && !isMatern) {
      return rbfMarginal(basis[i].center[j], basis[i].lengthscales[j]);
    }
    // Quadrature for warped dim or Matérn
    const c = basis[i].center[j];
    const kFn = modelKernel1D(mi, j);
    let sum = 0;
    for (let q = 0; q < glNodes!.length; q++) {
      const x = glNodes![q];
      const w = wFn ? wFn(x) : x;
      sum += glWeights![q] * kFn(w, c);
    }
    return sum;
  }

  // Helper: 1D cross for basis functions i, k on dim j
  function cross1D(i: number, k: number, j: number): number {
    const mi_i = basis[i].modelIdx;
    const mi_k = basis[k].modelIdx;
    const wFn_i = warpFns[mi_i][j];
    const wFn_k = warpFns[mi_k][j];
    const isMatern_i = models[mi_i].kernelType === "Matern";
    const isMatern_k = models[mi_k].kernelType === "Matern";

    if (!wFn_i && !wFn_k && !isMatern_i && !isMatern_k) {
      // Both non-warped RBF: closed-form generalized cross
      return rbfGeneralizedCross(
        basis[i].center[j],
        basis[k].center[j],
        basis[i].lengthscales[j],
        basis[k].lengthscales[j],
      );
    }

    // Quadrature: at least one warped or Matérn
    const a = basis[i].center[j];
    const b = basis[k].center[j];
    const kFn_i = modelKernel1D(mi_i, j);
    const kFn_k = modelKernel1D(mi_k, j);
    let sum = 0;
    for (let q = 0; q < glNodes!.length; q++) {
      const x = glNodes![q];
      const wi = wFn_i ? wFn_i(x) : x;
      const wk = wFn_k ? wFn_k(x) : x;
      sum += glWeights![q] * kFn_i(wi, a) * kFn_k(wk, b);
    }
    return sum;
  }

  // ── Precompute per-basis marginals and cross integrals ──
  const w: number[][] = new Array(N);
  const productW = new Float64Array(N);
  for (let i = 0; i < N; i++) {
    w[i] = new Array(d);
    let prod = 1.0;
    for (let j = 0; j < d; j++) {
      w[i][j] = marginal1D(i, j);
      prod *= w[i][j];
    }
    productW[i] = prod;
  }

  // Cache cross integrals: crossCache[j][i*N+k] (symmetric)
  const crossCache: Float64Array[] = new Array(d);
  for (let j = 0; j < d; j++) {
    crossCache[j] = new Float64Array(N * N);
    for (let i = 0; i < N; i++) {
      for (let k = i; k < N; k++) {
        const val = cross1D(i, k, j);
        crossCache[j][i * N + k] = val;
        crossCache[j][k * N + i] = val;
      }
    }
  }

  // ── E[μ] ──
  let emu = meanEns;
  for (let i = 0; i < N; i++) {
    emu += basis[i].coeff * productW[i];
  }
  const emuMinusMean = emu - meanEns;

  // ── Var(μ) = Σ_{ik} c_i c_k ∏_j W_j(i,k) - (E[μ]-mean)² ──
  let varMu = -emuMinusMean * emuMinusMean;
  for (let i = 0; i < N; i++) {
    for (let k = 0; k < N; k++) {
      let prod = 1.0;
      for (let j = 0; j < d; j++) {
        prod *= crossCache[j][i * N + k];
      }
      varMu += basis[i].coeff * basis[k].coeff * prod;
    }
  }

  if (varMu < 1e-30) {
    return {
      firstOrder: new Array(d).fill(0),
      totalOrder: new Array(d).fill(0),
      paramNames,
      numEvaluations: 0,
    };
  }

  // ── Per-dimension S_i and ST_i ──
  const firstOrder = new Array(d).fill(0);
  const totalOrder = new Array(d).fill(0);

  for (let dim = 0; dim < d; dim++) {
    // ── First-order: V_dim = Σ_{ik} β_i β_k W_dim(i,k) - (E[μ]-mean)² ──
    // β_i = c_i ∏_{j≠dim} w[i][j]
    const beta = new Float64Array(N);
    for (let i = 0; i < N; i++) {
      const wDim = w[i][dim];
      if (Math.abs(wDim) < 1e-300) {
        beta[i] = 0;
      } else {
        beta[i] = basis[i].coeff * productW[i] / wDim;
      }
    }

    let crossS = 0;
    for (let i = 0; i < N; i++) {
      for (let k = 0; k < N; k++) {
        crossS += beta[i] * beta[k] * crossCache[dim][i * N + k];
      }
    }
    firstOrder[dim] = Math.max(
      0,
      (crossS - emuMinusMean * emuMinusMean) / varMu,
    );

    // ── Total-order: ST_dim = 1 - V_{~dim}/Var(μ) ──
    const gamma = new Float64Array(N);
    for (let i = 0; i < N; i++) {
      gamma[i] = basis[i].coeff * w[i][dim];
    }

    let crossT = 0;
    for (let i = 0; i < N; i++) {
      for (let k = 0; k < N; k++) {
        let prod = 1.0;
        for (let j = 0; j < d; j++) {
          if (j !== dim) prod *= crossCache[j][i * N + k];
        }
        crossT += gamma[i] * gamma[k] * prod;
      }
    }
    let sumGammaW = 0;
    for (let i = 0; i < N; i++) {
      let prod = 1.0;
      for (let j = 0; j < d; j++) {
        if (j !== dim) prod *= w[i][j];
      }
      sumGammaW += gamma[i] * prod;
    }
    const vNotDim = crossT - sumGammaW * sumGammaW;
    totalOrder[dim] = Math.max(0, 1 - vNotDim / varMu);
  }

  return { firstOrder, totalOrder, paramNames, numEvaluations: 0 };
}
