# ax-js — Observations & Future Work

Notes on remaining work, architectural observations, and items to revisit in future sessions.

## Performance Optimizations (Deferred)

### 1. LU Factorization Caching in PairwiseGP
`solveLU(CKI, ...)` recomputes LU factorization on every `predict()` and `predictCovarianceWith()` call. Should cache the LU factors (P, L, U) at construction time and only apply the forward/back substitutions during prediction. This is especially impactful for `predictCovarianceWith` which calls `solveLU` with a different RHS than `predict`.

### 2. Transpose Allocation
The explicit transpose `KstarT` in `gp.ts:predict()` allocates an n×m matrix just to feed into `forwardSolve`. A `forwardSolveTransposed` that reads K* in column-major order would eliminate this copy.

### 3. Kernel Matrix Reuse
When predicting at multiple test point sets, the training kernel matrix K and Cholesky factor L are already cached, but cross-covariance K* is recomputed each time. For interactive use (e.g., slider changes in demos), caching the last K* and checking if test points changed would help.

### 3a. Shared V Matrix for predict + predictCovarianceWith
When calling both `predict(testPoints)` and `predictCovarianceWith(testPoints, refPoint)`, the V matrix for the test points (`V_test = L⁻¹ @ K*ᵀ`) is computed twice. A combined method or cached V could avoid this redundant forward solve. For relativization workflows where you always need both, this would halve the cost of the test-point computation.

### 4. WASM Linear Algebra
For n > 100, the Cholesky decomposition and matrix multiplications would benefit from compiled linear algebra. Options:
- **wasm-blas**: Port BLAS level-3 routines to WASM
- **Web Workers**: Move GP prediction off the main thread for UI responsiveness
- **SharedArrayBuffer**: Share kernel matrix data with worker threads

### 5. Memory Allocation
Every kernel `compute()` call allocates new `Matrix` objects. For hot paths (repeated predictions during visualization), an arena allocator or pre-allocated scratch matrices would reduce GC pressure.

## Architectural Notes

### Kernel State Backward Compatibility
The `KernelState` type supports two formats:
- **Legacy** (all existing fixtures): `{ type: "Matern", lengthscale: [...], outputscale: 0.5 }` — outputscale at top level implies ScaleKernel wrapping
- **New recursive**: `{ type: "Scale", base_kernel: { type: "Product", kernels: [...] } }` — explicit nesting

`buildKernel()` handles both. New fixtures should use the recursive format; legacy fixtures work unchanged.

### Multi-Task GP Prediction API
`MultiTaskGP.predict(testPoints, taskIndex)` takes test points WITHOUT the task column and a separate task index. This differs from BoTorch which embeds the task index in the last column of X. Our API is cleaner for client-side use (the UI knows which task it wants; no need to construct augmented inputs).

### Diagonal Kss Optimization
The `computeDiag` method on kernels is optional (not all kernels implement it). The `kernelDiag()` helper function falls back to computing the full matrix and extracting the diagonal. All built-in kernels implement `computeDiag` efficiently:
- Stationary kernels (Matern, RBF): return `1.0` for all points (self-distance = 0)
- ScaleKernel: `outputscale * base.computeDiag(x)`
- Composite: sum/product of component diagonals
This saves O(m^2) -> O(m) for the self-covariance computation.

### Adapter Transforms (Two-Layer Architecture)
The exported model state captures **BoTorch-level transforms only** by default. Ax adapter transforms (formerly "ModelBridge transforms") like LogY, BilogY, IntToFloat operate at the Ax layer and are NOT part of the GP model state. These are metadata-only — the caller handles the inverse transformation.

The exported `input_transform` (Normalize) already maps from raw parameter space to model space. Y-transforms (LogY, BilogY, PowerTransformY, StandardizeY) change the prediction space — the exported JSON includes `adapter_transforms` metadata so the caller can un-transform predictions back to original space.

### Integer Parameters
Integer parameters are treated as continuous (same as BoTorch). Rounding happens at the acquisition/generation layer, not in the surrogate model. axjs predicts on a continuous surface; the client handles discretization if needed.

### Relativization
axjs implements `relativize()`, `unrelativize()`, and `relativizePredictions()` matching Ax's `ax.utils.stats.math_utils.relativize()` exactly. Key features:
- Delta method variance propagation for the ratio estimator
- Second-order bias correction (default on)
- `controlAsConstant` mode (matches `RelativizeWithConstantControl`)
- Optional posterior covariance parameter for tighter confidence intervals

### Posterior Covariance
All model types implement `predictCovarianceWith()`:
- **ExactGP/SingleTaskGP**: `Cov(f(a), f(b)) = k(a,b) - v_a · v_b` where `v = L⁻¹ K*ᵀ`
- **PairwiseGP**: `Cov(f(a), f(b)) = k(a,b) - K(a,X) @ (CK+I)⁻¹ @ C @ K(X,b)` (Laplace approx)
- **MultiTaskGP**: Same as ExactGP but with task-augmented kernel
- **ModelListGP**: Per-output covariance (no cross-output covariance — outputs are independent)
- **EnsembleGP**: Law of total covariance: `E_k[Cov_k] + Cov_k(μ_a, μ_b)`

The covariance at the status quo is used to tighten relative effect confidence intervals. This is an improvement over Ax, which ignores posterior covariance (hardcodes `cov_means=0`).

## GPyTorch/BoTorch Observations

### CategoricalKernel Formula
The CategoricalKernel computes `exp(-mean_d(delta_d / ls_d))` where `delta_d` is 1 if categories differ in dimension d, 0 otherwise, and `ls_d` is the per-dimension lengthscale. Supports both scalar (shared) and ARD (per-dim) lengthscales.

### ProductKernel with Active Dims
The typical Ax pattern for mixed experiments is:
```
ScaleKernel(ProductKernel(
  MaternKernel(active_dims=[0,1,2]),    # continuous
  CategoricalKernel(active_dims=[3,4])   # categorical
))
```
Our `ActiveDimsKernel` wrapper handles this cleanly by extracting the right columns before passing to each sub-kernel.

### MultitaskKernel Task Covariance
BoTorch 0.16+ uses `PositiveIndexKernel` (replacing `IndexKernel`):
- `covar_factor`: W matrix (num_tasks x rank)
- `raw_var`: diagonal variance per task (softplus parameterization)
- B = (W @ W^T + diag(softplus(raw_var))) / B[target, target]

The normalization by `B[target, target]` ensures unit variance on the target task. We export the pre-computed `covar_matrix` directly to avoid reimplementing the normalization logic.

### MultiTaskGP Per-Task Mean Constants
BoTorch 0.16+ uses `MultitaskMean` which stores separate mean constants per task (not a single shared constant). The export captures all per-task means as an array. When a single value is provided (legacy format), axjs broadcasts it to all tasks.

### Input Transform Chain Order
BoTorch's `ChainedInputTransform` applies transforms in insertion order. For Ax, this is always: Normalize -> Warp. The warp (Kumaraswamy CDF) expects inputs in [0, 1], which is what Normalize produces. Our `ExactGP` applies transforms in the same order: normalize first, then warp.

### Warp Epsilon Normalization
BoTorch's Warp transform normalizes inputs to [eps, 1-eps] via `x * (1 - 2*eps) + eps` with `eps=1e-7` before applying the Kumaraswamy CDF. This is NOT the same as simple clamping `max(eps, min(1-eps, x))`. The affine normalization squeezes the entire [0,1] range into [eps, 1-eps], affecting all points — not just those at the boundaries.

## Features Not Yet Implemented

### 1. LMC (Linear Model of Coregionalization)
LMC generalizes ICM by using multiple latent functions:
```
K_LMC = sum_i (A_i ⊗ k_i)
```
where each k_i is a separate data kernel. ICM is the special case with one latent function. LMC is rarely used in Ax but could be needed for complex multi-task setups.

### 2. Fantasy Models
BoTorch supports "fantasizing" — conditioning on hypothetical observations. This is used for knowledge gradient and lookahead acquisition. Not needed for prediction-only use.

### 3. GPU/WebGPU
For very large n (>1000), WebGPU-based matrix operations could provide significant speedup. The matrix operations are regular and well-suited to GPU parallelism.

### 4. LCEAGP (Latent Context Embedding Additive GP)
Used for transfer learning across related experiments. Each context (e.g., different product surface, different population) gets a learned embedding vector, and the GP kernel operates on the joint (parameter, embedding) space.

## Smells in Ax/BoTorch

### Bilog/Power/Log Are Adapter-Only in Practice (Not Model-Level)

BoTorch defines `Bilog`, `Power`, and `Log` as model-level outcome transforms
(`botorch.models.transforms.outcome`), but **Ax never passes them as
`outcome_transform` to `SingleTaskGP`**. They are always applied at the adapter
level via `BilogY`, `PowerTransformY`, and `LogY` transforms in the Ax adapter
pipeline. No Ax source code constructs a model with `outcome_transform=Bilog()` etc.

The real pipeline for a BilogY-transformed metric:
```
Raw Y → [Adapter: BilogY.transform()] → bilog(Y)
     → [Model: outcome_transform=Standardize() ONLY] → GP trains on standardized bilog(Y)

Predict:
GP posterior → [Model: Standardize⁻¹] → bilog-space (μ, σ²)
            → [Adapter: BilogY.untransform()] → original-space (μ, σ²)
```

The model only ever sees `Standardize` as its `outcome_transform`.

**Why:** BoTorch's `Bilog` and `Power` provide **only** `sample_transform` (for MC
acquisition functions), with **no** `mean_transform` or `variance_transform`. Calling
`.mean` or `.variance` on a `TransformedPosterior` from these transforms raises
`NotImplementedError`, so `predict_from_model()` would crash.

**Impact on axjs:**
- axjs's `BilogUntransform` and `PowerUntransform` handle a code path that doesn't
  occur in production Ax. They exist for completeness and future-proofing.
- The `branin_bilog` and `branin_power` fixtures are consistency-only tests of this
  non-production path (no BoTorch reference predictions possible).
- In real usage, the adapter-level transforms are metadata-only in the export format
  (`adapter_transforms` field) — the **caller** applies them to axjs predictions.
- axjs's delta method approximations for Bilog/Power are strictly more than BoTorch
  offers (which provides nothing).

**Log is a special case:** BoTorch's `Log` DOES have analytic moments
(`norm_to_lognorm_mean/variance`), so it could work as a model-level transform. But
Ax still uses `LogY` at the adapter level in practice. The `branin_log` and
`branin_chained_log_std` parity fixtures confirm our formulas match BoTorch's.

**Covariance across the transform boundary:** `predictCovarianceWith()` returns
covariance in the GP's internal (transformed) space. For `Standardize`, we scale by
`std²`. For nonlinear transforms (Log/Bilog/Power), transforming covariance to original
space requires additional delta-method terms — but since Ax's relativization operates
on adapter-level predictions, and `predictCovarianceWith` is only used before the
adapter untransform, this is not a practical issue. It does mean the identity
`Cov(f(x),f(x)) = Var(f(x))` does NOT hold across the nonlinear transform boundary
(self-covariance is in GP space, variance from `predict()` is in original space).

### Log Transform Uses Exact Log-Normal, Not Delta Method
BoTorch's `Log` outcome transform uses exact log-normal distribution formulas for moments: `E[Y] = exp(μ + σ²/2)`, `Var[Y] = expm1(σ²)·exp(2μ + σ²)`. This is correct but means the mean prediction depends on the posterior variance — predictions at the same point can change if variance changes. The old axjs code incorrectly used `exp(μ)` (delta method zeroth-order approximation), which has been fixed.

### Relativization Ignores Posterior Covariance
Ax's `relativize()` always passes `cov_means=0`, treating test and control predictions as independent. For GP models where both arms share training data, predictions at the test and status quo points are typically positively correlated. Ignoring this covariance makes the relative effect confidence intervals wider than necessary (conservative).
