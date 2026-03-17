# axjs Numerical Details & GPyTorch/BoTorch Observations

This document records numerical implementation decisions, GPyTorch/BoTorch behaviors we replicate, and any inefficiencies or concerns discovered during implementation.

## Precision

- **Float64 throughout**: All computations use `Float64Array`. This matches BoTorch's default `torch.float64` precision. Mean parity is within 1e-6 and variance parity within 1e-5 across all test fixtures.
- **Float32 would not work**: Tested early on — accumulated rounding in Cholesky + triangular solves produces unacceptable errors (~1e-3) with float32.

## GPyTorch Kernel Numerics

### Matérn Mean-Centering (GPyTorch-specific)

GPyTorch's `MaternKernel` subtracts `x1.mean(dim=0)` from both `x1` and `x2` before dividing by lengthscale. This is intended to improve numerical stability by centering coordinates near zero before computing distances. **RBF does NOT do this.**

This behavior is undocumented and non-obvious. It affects computed distances when x1 has non-zero mean (which is the common case). We replicate it exactly.

**Source**: `gpytorch/kernels/matern_kernel.py`, in the `forward` method.

### Squared Distance Expansion

Both GPyTorch and axjs compute squared distances via:
```
||a - b||² = ||a||² + ||b||² - 2·a·b
```
This avoids forming the explicit difference vectors, which is more numerically stable and memory-efficient for high dimensions. The result can go slightly negative due to floating-point cancellation, so we clamp to ≥ 0.

### Distance Clamping

- **Euclidean distances**: Clamped ≥ 1e-15 for Matérn kernels. This prevents division by zero in the Matérn formula where `r` appears in the denominator (through the exponential term).
- **Squared distances**: Clamped ≥ 0 (not 1e-30 before sqrt; instead we clamp the sqrt result).

GPyTorch uses `torch.clamp(dist, min=1e-15)` in the Matérn kernel.

## Cholesky Decomposition

### Jitter Escalation

GPyTorch's `psd_safe_cholesky` adds escalating jitter when Cholesky fails:
1. Try without jitter
2. Try with 1e-6 · I
3. Try with 1e-5 · I
4. Try with 1e-4 · I
5. Try with 1e-3 · I

We replicate this exactly. In practice, the kernel matrix K + σ²I is almost always well-conditioned enough to factorize without jitter, since noise variance σ² acts as a regularizer.

### Potential Issue: Near-singular kernel matrices

For very small noise variance (<1e-8) and many training points in similar locations, the kernel matrix can become near-singular. GPyTorch handles this with jitter, but the added jitter effectively increases the noise variance, which changes the posterior. This is a known limitation of Cholesky-based GP inference.

**Observation**: For PairwiseGP, the K matrix has no noise diagonal (noise is in the Hessian, not added to K), making it more prone to near-singularity. The jitter escalation helps here.

## PairwiseGP (Laplace Approximation)

### Non-Symmetric System

The variance computation in PairwiseGP requires solving `(CK + I)x = b` where:
- C = negative Hessian of the probit log-likelihood (symmetric, positive semi-definite)
- K = kernel matrix (symmetric, positive definite)
- CK = C @ K (NOT symmetric, since C and K don't commute)

BoTorch uses `torch.linalg.solve` (LU decomposition) for this system. We initially attempted Cholesky (which assumes symmetry) and got wrong results. Fixed by implementing LU with partial pivoting.

**Observation**: BoTorch's `_scaled_psd_safe_cholesky` is used for the K matrix Cholesky (in `_update_covar`), but `torch.linalg.solve` is used for the variance system. This inconsistency is correct but non-obvious.

### Utility Values

PairwiseGP stores MAP utility estimates computed via Newton-Raphson (scipy's `fsolve`). These are the "training targets" analogous to Y in standard GP. They exist in the standardized kernel space (scaled by outputscale).

## BoTorch 0.16 Breaking Changes

### Default Kernel Changed

In BoTorch 0.16, `SingleTaskGP` defaults to:
- Kernel: `RBFKernel` (previously `ScaleKernel(MaternKernel(nu=2.5))`)
- No ScaleKernel wrapper (no outputscale parameter)

This means exported models may or may not have `outputscale` in the kernel state. Our code handles both cases.

### FixedNoiseGP Merged

`FixedNoiseGP` was removed as a separate class. Use `SingleTaskGP(train_X, train_Y, train_Yvar=...)` instead. Detection is via `isinstance(model.likelihood, FixedNoiseGaussianLikelihood)`.

The noise variance in the exported state is in the **standardized** space (after outcome transform), not the original observation space.

## Performance Characteristics

### Complexity

| Operation | Time | When |
|-----------|------|------|
| Construction (Cholesky + alpha) | O(n³) | Once per model load |
| Prediction (per test point) | O(n²) | Each predict call |
| Full Kss computation | O(m²n) | Wasteful — only diagonal needed |

Where n = training points, m = test points.

### Known Inefficiencies

1. **Full Kss matrix**: We compute the full m×m self-covariance matrix but only use the diagonal for variance. A diagonal-only kernel computation would be O(m) instead of O(m²). For visualization grids (e.g., 80×80 = 6400 points), this means computing a 6400×6400 matrix unnecessarily. However, the kernel computation is cheap compared to the solve step as long as m >> n.

2. **No batched kernel evaluation**: Each `kernel.compute` call allocates a new Matrix. For repeated predictions with different test points, the training-side kernel matrix K is recomputed in PairwiseGP (fixed by caching CKI) but the cross-covariance K* is always recomputed.

3. **LU solver is not cached**: For PairwiseGP, `solveLU(CKI, ...)` recomputes the LU factorization on every predict call. Should cache the LU factors.

## Comparison with GPyTorch

GPyTorch uses several optimizations we do not:
- **Lanczos-based prediction** (`fast_pred_var`): approximates posterior variance using iterative methods. Not needed for our use case (small n).
- **CG-based solves**: For large n, conjugate gradient is used instead of Cholesky. Not needed for n < 200.
- **Kernel caching and lazy evaluation**: GPyTorch's `LazyEvaluatedKernelTensor` delays computation. We eagerly compute everything.

For our target use case (n < 200 training points, browser visualization), these optimizations are unnecessary and our direct approach is simpler and more predictable.
