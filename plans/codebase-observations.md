# Codebase Observations

Critical analysis of ax-js and upstream BoTorch/GPyTorch patterns.
Subsumes `docs/internal/observations.md`.

---

## 1. BoTorch/GPyTorch Patterns That Cause Friction

### Undocumented Matern mean-centering

GPyTorch's `MaternKernel.forward()` subtracts `x1.mean(dim=0)` from both inputs
before dividing by lengthscale. RBF does not. This is a numerical stability trick
that is nowhere in the API docs. Discovering it required reading GPyTorch source.
It silently changes kernel values when x1 has non-zero mean, which is the common
case for unnormalized data.

**Impact on ax-js:** Had to replicate exactly in `src/kernels/matern.ts`. Any
change to this behavior in GPyTorch would silently break parity.

### Private attributes required for serialization

No public BoTorch API exists for extracting prediction-sufficient state.
`_extraction.py` accesses private attributes (`_task_feature`,
`_eval_covar_matrix()`, `raw_var`, `raw_constant`) that can change without
deprecation warnings. BoTorch 0.16 broke two of these (IndexKernel to
PositiveIndexKernel, ConstantMean to MultitaskMean).

See `plans/upstream-proposals.md` for concrete stabilization proposals.

### Bilog and Power missing analytic moments

`Bilog` and `Power` outcome transforms implement only `sample_transform`. No
`mean_transform` or `variance_transform`. This means they cannot function as
model-level transforms -- only as adapter-level transforms. ax-js test fixtures
for these transforms are consistency-only (no BoTorch reference values to compare
against, since BoTorch itself cannot produce analytic posteriors through these
transforms).

### Breaking changes across versions

BoTorch 0.16/0.17 introduced several breaking changes:
- Default kernel: `ScaleKernel(MaternKernel(nu=2.5))` to `RBFKernel`
- `FixedNoiseGP` merged into `SingleTaskGP`
- `IndexKernel` to `PositiveIndexKernel`
- `ConstantMean` to `MultitaskMean` for multi-task models

ax-js handles all cases, but each required code changes and new fixtures.

### Warp epsilon normalization

BoTorch's Kumaraswamy warp uses `x * (1 - 2*eps) + eps` with `eps = 1e-7`.
This is not simple clamping. Getting this wrong produces subtle parity failures
that only show up near boundary values.

---

## 2. ax-js Patterns That Could Be Improved

### LU factorization recomputation

`solveLU(CKI, ...)` in PairwiseGP recomputes the LU factorization on every
`predict()` and `predictCovarianceWith()` call. The CKI matrix is fixed at
construction time -- the factorization should be cached.

**Files:** `src/models/pairwise_gp.ts`, `src/linalg/lu.ts`

### Full Kss when only diagonal is needed

The GP prediction computes the full m x m test covariance matrix `Kss` but only
reads the diagonal for variance. For a visualization grid of 80x80 = 6400 points,
this means computing a 6400x6400 matrix (328 MB of Float64) when only 6400 values
are needed.

**Files:** `src/models/gp.ts`, `src/kernels/*.ts`

### V matrix duplication

When calling both `predict(testPoints)` and `predictCovarianceWith(testPoints, refPoint)`,
the V matrix (`V = L^-1 @ K*^T`) is computed twice. A combined method or cached V
would halve the cost for relativization workflows that need both.

**Files:** `src/models/gp.ts`

### Transpose allocation

`KstarT` in `gp.ts:predict()` allocates an n x m matrix just for `forwardSolve`.
A `forwardSolveTransposed` variant could eliminate this copy.

**Files:** `src/models/gp.ts`, `src/linalg/solve.ts`

### `any` type usage

Several `any` casts remain, concentrated in `src/predictor.ts` (where
heterogeneous model types are dispatched) and scattered across
`src/transforms/build_outcome.ts`, `src/models/multi_task.ts`, and
`src/viz/`. Most could be replaced with discriminated unions or generics.

### Memory allocation in hot paths

Every `kernel.compute()` call allocates new `Matrix` objects. For interactive
visualization (slider-driven predictions), an arena allocator or pre-allocated
scratch matrices would reduce GC pressure.

---

## 3. Intentional Divergences

These are design decisions for a lightweight browser library, not bugs.

| Divergence | Python (BoTorch/Ax) | ax-js | Justification |
|-----------|---------------------|-------|---------------|
| Relativization | `cov_means=0` (conservative) | Covariance-aware via `getCovariances()` | Tighter CIs; GP provides covariance for free |
| LOO cross-validation | Refit-based `cross_validate` (O(n^4)) | Analytic R&W 5.12 (O(n^2)) | Orders of magnitude faster, exact for GPs |
| Sobol' sensitivity | Not available | Analytic + MC (`computeSensitivity()`) | Novel contribution; closed-form for RBF kernels |
| Diagonal-only Kss | Full m x m posterior covariance | Only diagonal used (variance) | Prediction-only; no covariance-dependent acquisition |
| Observation model | Batch structure, LazyTensor | Flattened Float64Array | No batch dimension needed for prediction |
| Kernel evaluation | Lazy evaluation, CG solver | Eager Cholesky, direct solves | Simpler, predictable; n < 200 target |
| Precision | Float64 default, Float32 optional | Float64 only | Float32 produces ~1e-3 errors in Cholesky |

---

## 4. Performance Opportunities

Ranked by estimated impact multiplied by implementation effort.

| Rank | Optimization | Impact | Effort | Notes |
|------|-------------|--------|--------|-------|
| 1 | **LU caching in PairwiseGP** | High (removes O(n^3) per predict) | Low | Cache at construction, reuse in predict |
| 2 | **Diagonal-only Kss** | High for large grids (O(m) vs O(m^2)) | Medium | Requires `computeDiag()` on all kernels |
| 3 | **V matrix sharing** | Medium (halves work for relativization) | Low | Cache V between predict and predictCovarianceWith |
| 4 | **Kernel caching** | Medium (for interactive use) | Medium | Cache K* when test points unchanged |
| 5 | **forwardSolveTransposed** | Low (eliminates one n x m allocation) | Low | New linalg primitive |
| 6 | **WASM linear algebra** | High for large n (>100) | High | Compiled BLAS, Web Workers |
| 7 | **Arena allocator** | Medium (reduces GC in hot paths) | High | Requires refactoring Matrix API |

Items 1, 3, and 5 are independent and could be done in a single PR.
Item 2 depends on adding `computeDiag()` to the kernel interface first.
Items 6 and 7 are large-scope changes that need design RFCs.

---

## 5. Testing Gaps

### Intentionally excluded features

These BoTorch features are not implemented and not tested:

- **LMC (Linear Model of Coregionalization):** Generalizes ICM. Rarely used in Ax.
- **Fantasy models:** Conditioning on hypothetical observations. Used for knowledge
  gradient, not needed for prediction-only.
- **GPU/WebGPU acceleration:** For n > 1000. Current target is n < 200.
- **LCEAGP (Latent Context Embedding Additive GP):** Transfer learning across
  experiments. Not yet needed.
- **CG/Lanczos solvers:** Iterative methods for large n. Unnecessary for target
  problem sizes.

### Consistency-only fixtures

Two fixtures use consistency-only testing (self-parity, no BoTorch reference):
- `branin_bilog.json` -- Bilog has no analytic moments in BoTorch
- `branin_power.json` -- Power has no analytic moments in BoTorch

These verify that ax-js's own untransform logic is self-consistent but cannot
verify parity with BoTorch since BoTorch itself cannot produce reference values.

### Sensitivity analysis coverage

Analytic Sobol' indices are validated against MC estimates (convergence tests)
and known closed-form solutions for simple kernels. There is no BoTorch reference
since BoTorch does not implement Sobol' sensitivity for GPs.

Known limitation: warped-dim analytic Sobol' can differ from MC for GPs where
the posterior mean is nearly constant along some dimensions (numerical
cancellation in the quadratic form).

---

## References

- Numerical details: `docs/internal/numerics.md`
- Testing documentation: `docs/testing.md`
- Upstream proposals: `plans/upstream-proposals.md`
- Action plan: `plans/action-plan.md`
