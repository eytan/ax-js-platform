# Parity Guide: Keeping ax-js in Sync with BoTorch

"I am looking at a BoTorch class. Where is the ax-js equivalent, and what must
stay in sync?"

Cross-references: [plans/file-correspondence.md](file-correspondence.md),
[docs/developer-guide.md](../docs/developer-guide.md),
[docs/data-model.md](../docs/data-model.md)

---

## 1. File Correspondence

See [plans/file-correspondence.md](file-correspondence.md) for the complete
mapping of Python classes to TypeScript files, including kernels, models,
transforms, linear algebra, acquisition functions, and test coverage.

---

## 2. Naming Convention Rules

JSON fields use snake_case (matching Python attribute names). TypeScript code
uses camelCase. Class names stay aligned across both codebases.

| Python name | JSON key | TypeScript variable | Notes |
|---|---|---|---|
| `model.train_inputs[0]` | `train_X` | `trainX` | 2D array of floats |
| `model.train_targets` | `train_Y` | `trainY` | 1D array of floats |
| `kernel.lengthscale` | `lengthscale` | `lengthscale` | Same in all three |
| `kernel.outputscale` | `outputscale` | `outputscale` | Same in all three |
| `model.likelihood.noise` | `noise_variance` | `noiseVariance` | Scalar or per-point array |
| `model.mean_module.constant` | `mean_constant` | `meanConstant` | Scalar |
| `transform.offset` | `offset` | `offset` | Same in all three |
| `transform.coefficient` | `coefficient` | `coefficient` | Same in all three |
| `model.input_transform` | `input_transform` | `inputTransform` | Normalize state dict |
| `model.outcome_transform` | `outcome_transform` | `outcomeTransform` | Standardize/Log/etc. state |

General rules:
- JSON is the contract boundary. Always snake_case. Matches Python attribute names.
- TypeScript interfaces for JSON states (in `src/models/types.ts`) use snake_case
  to match JSON directly (e.g., `train_X`, `noise_variance`).
- TypeScript runtime variables and class fields use camelCase.
- Class names match: `SingleTaskGP`, `MaternKernel`, `InputWarp`, etc.

---

## 3. Two-Layer Transform Architecture

```
                       ORIGINAL DATA
                            |
                   [Layer 1: Adapter transforms]
                   Ax applies BEFORE BoTorch:
                   LogY, BilogY, StandardizeY,
                   PowerTransformY
                            |
                       ADAPTER SPACE
                            |
                   [Layer 2: Model transforms]
                   BoTorch applies internally:
                   Standardize, Log, Bilog, Power
                   (stored in model.outcome_transform)
                            |
                       MODEL SPACE
                            |
                     train_Y lives here
                     (both layers applied)
```

**Critical rule**: `train_Y` in the exported model state is NOT in original data
space. Both layers have been applied. Any code returning Y values to users MUST
call `Predictor.untransformTrainY()`, which reverses both layers:

1. Layer 2 (innermost): undo model-level outcome transform (Standardize, etc.)
2. Layer 1 (outermost): undo adapter-level transforms (LogY, BilogY, etc.)

The `Predictor.predict()` method handles this automatically via
`applyAdapterUntransform()` (predictor.ts:598). The model's `ExactGP.predict()`
applies model-level untransforms internally (gp.ts:157-163).

Adapter transforms are exported in the `adapter_transforms` field of
`ExperimentState`. They are NOT part of the BoTorch model state -- they exist at
the Ax adapter layer and are invisible to BoTorch.

---

## 4. How to Add a Kernel / Model / Transform

### Adding a new kernel

1. Create `src/kernels/<name>.ts` implementing the `Kernel` interface from
   `src/kernels/types.ts`. Must implement `compute(x1, x2): Matrix` and
   `computeDiag(x): Float64Array`.
2. Add a case to `src/kernels/build.ts` in `buildKernel()` to handle the new
   kernel type string from JSON.
3. Add extraction logic in `python/_extraction.py` inside
   `extract_kernel_state()` to recognize the GPyTorch kernel class and serialize it.
4. Write unit tests in `test/kernels/<name>.test.ts` with known-good values
   from GPyTorch.
5. Add a fixture via `python/generate_fixtures.py` (add a `FixtureSpec` to
   `FIXTURE_SPECS`) and regenerate: `python python/generate_fixtures.py`.
6. Verify parity: `npx vitest run test/integration/botorch_parity.test.ts`.
7. Export from `src/kernels/types.ts` or `src/index.ts` if part of the public API.

### Adding a new model

1. Create `src/models/<name>.ts`. Typically wraps `ExactGP` from `src/models/gp.ts`.
2. Define the model state interface in `src/models/types.ts` and add it to the
   `AnyModelState` discriminated union.
3. Add a case to `src/io/deserialize.ts` in `loadModel()`.
4. Add extraction in `python/_extraction.py` (new `export_<name>()` function)
   and dispatch in `export_botorch_model()`.
5. Add a generator in `python/generators/` for fixture creation.
6. Add fixtures and parity tests.
7. Update `Predictor` in `src/predictor.ts` if the new model type needs special
   handling in `predict()`, `getCovariances()`, or `loocv()`.

### Adding a new outcome transform

1. Add the untransform class to `src/transforms/outcome.ts`, implementing the
   `OutcomeUntransform` interface (`untransform(mean, variance)` and
   `untransformCovariance(cov)`).
2. Add extraction in `python/_extraction.py` inside
   `_extract_single_outcome_transform()`.
3. Add a case to `src/transforms/build_outcome.ts` in `buildOutcomeTransform()`.
4. If it is an adapter-level transform (Ax, not BoTorch), also add a case in
   `buildAdapterUntransforms()` in `src/predictor.ts`.
5. Add unit tests in `test/transforms/outcome.test.ts`.
6. Add a fixture with the new transform active.

---

## 5. Tracing a Computation: SingleTaskGP Posterior Mean

This traces the posterior mean computation through both codebases for a
`SingleTaskGP` with `Normalize` input transform and `Standardize` outcome
transform.

### BoTorch side (Python)

```
User calls:  model.posterior(X)
  -> botorch.models.SingleTaskGP.posterior()
     -> self.transform_inputs(X)                    # Normalize: (X - offset) / coeff
     -> gpytorch.models.ExactGP.__call__(X_norm)
        -> self.mean_module(X_norm)                  # ConstantMean: returns c
        -> self.covar_module(X_norm, train_X_norm)   # ScaleKernel(RBFKernel): K*
        -> MultivariateNormal(mean, covar)
     -> GPyTorchPosterior(mvn)
  -> posterior.mean                                  # mu = m(X*) + K* @ alpha
  -> model.outcome_transform.untransform(mean, var)  # Standardize: y*std + mean
```

### ax-js side (TypeScript)

```
User calls:  predictor.predict([[x1, x2]])
  -> Predictor.predict()                              # predictor.ts:111
     -> SingleTaskGP.predict(points)                  # single_task.ts
        -> ExactGP.predict(testX)                     # gp.ts:125
           -> transformInputs(testX)                  # gp.ts:106  Normalize + Warp
           -> kernel.compute(testXNorm, trainXNorm)   # gp.ts:129  K*
           -> mean.forward(testXNorm)                 # gp.ts:132  constant mean
           -> mu[i] = meanPrior + K* @ alpha          # gp.ts:134-140
           -> kernelDiag + forwardSolve for variance  # gp.ts:143-154
           -> outcomeTransform.untransform(mu, var)   # gp.ts:157-163  Standardize
        <- { mean, variance }
     -> applyAdapterUntransform(name, result)         # predictor.ts:123  LogY, etc.
  <- { "metric_name": { mean, variance } }
```

### Key correspondence points

| Step | BoTorch | ax-js |
|---|---|---|
| Input normalization | `model.transform_inputs(X)` | `gp.ts:106-112` `transformInputs()` |
| Kernel matrix K* | `covar_module(X*, X_train)` | `gp.ts:129` `kernel.compute()` |
| Posterior mean | `mvn.mean` (implicitly K* @ alpha) | `gp.ts:134-140` explicit dot product |
| Posterior variance | `mvn.variance` (full covar diagonal) | `gp.ts:143-154` diagonal-only via forwardSolve |
| Model outcome untransform | `outcome_transform.untransform()` | `gp.ts:157-163` |
| Adapter untransform | Ax adapter reversal (not in BoTorch) | `predictor.ts:598-616` `applyAdapterUntransform()` |

### Pre-computation (constructor)

| Step | BoTorch | ax-js |
|---|---|---|
| Training kernel | `model.__call__` lazy evaluation | `gp.ts:61` `kernel.compute(trainXNorm, trainXNorm)` |
| Add noise | implicit in likelihood | `gp.ts:64-68` `K.addDiag(noiseVariance)` |
| Cholesky | `psd_safe_cholesky` (with jitter) | `gp.ts:71` -> `cholesky.ts:10` (jitter escalation) |
| Alpha vector | `L^T \ (L \ (y - m))` | `gp.ts:73-79` `solveCholesky(L, residuals)` |

---

## 6. Intentional Divergences

| Divergence | ax-js behavior | BoTorch behavior | Justification | Should BoTorch adopt? |
|---|---|---|---|---|
| Covariance-aware relativization | Uses full `Cov(f(x), f(x_sq))` in delta-method | Sets `cov_means=0` | More accurate relative predictions when SQ is uncertain | Under discussion |
| Analytic LOO-CV | Rasmussen & Williams Eq. 5.12, no refitting | `cross_validate()` with refitting | O(n^2) vs O(k * n^3); exact for the same model | Complementary, not replacement |
| Analytic Sobol' sensitivity | Closed-form integrals for RBF/Matern kernels | Not implemented | O(d * n^2), zero MC noise, deterministic | Yes, proposed upstream |
| Diagonal-only Kss | Only computes `diag(K**)`, never full m x m | `posterior()` returns full covariance lazy tensor | Sufficient for marginal predictions; O(m) not O(m^2) | N/A (different use case) |
| Flattened observation model | No batch dimensions in train_X/train_Y | Batch dims for multi-output | Simpler JSON serialization; batched models decomposed during export | N/A (serialization choice) |
| Eager dense computation | All matrices computed eagerly | LazyEvaluatedKernelTensor | Simpler, sufficient for n < 5000 | N/A (scale difference) |
| CMA-ES for acquisition | CMA-ES optimizer for acqf | L-BFGS-B with restarts | No gradient computation in JS; CMA-ES is derivative-free | No (L-BFGS is better with gradients) |
| EnsembleGP | First-class model type | No direct equivalent (SAAS uses Pyro) | Unified interface for SAAS/multi-restart MAP ensembles | Potentially |

---

## 7. What Must Stay in Sync

These are the computations where ax-js MUST produce numerically identical results
to BoTorch (within the 1e-6 tolerance). Any drift here causes parity test failures.

### Kernel math

- **Matern mean-centering**: GPyTorch subtracts `x1.mean(axis=0)` from BOTH x1
  and x2 before lengthscale division. RBF does NOT do this. See `matern.ts:30-52`
  vs `rbf.ts:25-37`. This is a GPyTorch implementation detail, not a mathematical
  property of the kernel, but it must be replicated exactly.

- **Distance computation**: Squared Euclidean uses the expansion
  `||a-b||^2 = ||a||^2 + ||b||^2 - 2a.b` (distance.ts:9-50). Results are
  clamped: squared distances at 0, Euclidean at 1e-15 (distance.ts:60-62).

- **Matern formulas**: Must match GPyTorch exactly:
  - nu=2.5: `(1 + sqrt(5)*r + 5*r^2/3) * exp(-sqrt(5)*r)`
  - nu=1.5: `(1 + sqrt(3)*r) * exp(-sqrt(3)*r)`
  - nu=0.5: `exp(-r)`

- **ScaleKernel**: `outputscale * base_kernel(x1, x2)`. Trivial but must not
  be accidentally applied twice or omitted.

### Cholesky jitter

The jitter escalation sequence `[1e-6, 1e-5, 1e-4, 1e-3]` must match
`psd_safe_cholesky` (cholesky.ts:16). Different jitter values produce different
L factors and therefore different predictions.

### Warp epsilon

BoTorch's Warp uses `x * (1 - 2*eps) + eps` with `eps = 1e-7` to squeeze inputs
into `(eps, 1-eps)` before applying the Kumaraswamy CDF (warp.ts:32-33). This is
an affine rescaling, NOT a simple clamp. Using `clamp(x, eps, 1-eps)` instead
produces different derivatives at the boundaries and fails parity.

### Mean-centering scope

Only Matern kernels mean-center. If a new kernel is added, verify whether
GPyTorch mean-centers for that kernel type. Getting this wrong typically causes
errors of 1e-2 to 1e-1, well above the tolerance.

### Normalize transform

`InputNormalize.forward(x) = (x - offset) / coefficient`. The offset and
coefficient come from BoTorch's `Normalize` transform bounds. This must be
applied BEFORE warp (warp.ts expects inputs in [0,1]).

### Outcome transform ordering

For `ChainedOutcomeTransform`, the untransform must be applied in REVERSE order
of the forward chain. If forward is `[Standardize, Log]`, untransform is
`[LogUntransform, StandardizeUntransform]`.

### PairwiseGP: LU not Cholesky

The matrix `C @ K + I` in PairwiseGP is NOT symmetric in general. Using Cholesky
will silently produce wrong results or crash. Must use LU decomposition
(pairwise_gp.ts uses lu.ts).

### MultitaskKernel covariance matrix

The task covariance `B = WW^T / diag(WW^T)` normalization must match
`PositiveIndexKernel._eval_covar_matrix()`. The normalization divides each row
and column by the square root of the diagonal.

---

## 8. What Need NOT Stay in Sync

These are implementation details where ax-js is free to diverge from BoTorch
without affecting numerical parity.

| Aspect | Why it does not matter |
|---|---|
| Internal variable names | `alpha` vs `alpha_cache`, `L` vs `LL` -- naming is cosmetic. |
| Memory layout | Row-major Float64Array vs column-major torch.Tensor. Results are the same. |
| Class hierarchy depth | BoTorch: `SingleTaskGP -> ExactGP -> GPyTorchModel -> Model`. ax-js: `SingleTaskGP` wraps `ExactGP`. Flatter is fine. |
| Lazy evaluation | GPyTorch defers kernel matrix computation via `LazyEvaluatedKernelTensor`. ax-js computes eagerly. Same result, different timing. |
| GPU dispatch / dtype selection | ax-js is CPU-only Float64. BoTorch may use Float32 on GPU. Fixtures are generated in Float64 for comparison. |
| Batch dimensions | BoTorch uses `[batch, n, d]` tensors. ax-js uses flat `[n, d]` matrices. Batched models are decomposed during export. |
| Module registration / `nn.Module` | GPyTorch kernels are `nn.Module` subclasses for autograd. ax-js kernels are plain classes. |
| Caching / memoization strategy | BoTorch caches kernel matrices, posteriors. ax-js pre-computes L and alpha once at construction. |
| Error message text | Free to differ. Just be clear and actionable. |
| File organization | BoTorch splits models across `botorch.models.*`. ax-js puts all kernels in `src/kernels/`. Organization is a local choice. |
| Test framework | BoTorch uses pytest. ax-js uses vitest. |
| Serialization format details | The JSON schema is an ax-js design choice. BoTorch has no canonical export format (this is the gap ax-js fills). |
