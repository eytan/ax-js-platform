# Action Plan

Prioritized cleanup and improvement tasks for ax-js.
Cross-references: `plans/codebase-observations.md`, `plans/upstream-proposals.md`.

---

## Tier 1: High Impact, Low Risk

These are independent changes that can each be done in a single PR with no
upstream coordination.

### 1.1 Cache LU factorization in PairwiseGP

Cache the LU factors (P, L, U) of the CKI matrix at construction time. Currently
`solveLU(CKI, ...)` recomputes the factorization on every `predict()` and
`predictCovarianceWith()` call, adding an unnecessary O(n^3) cost per prediction.

**Files:** `src/models/pairwise_gp.ts`, `src/linalg/lu.ts`
**Test:** Parity fixtures `branin_pairwise.json`, `pairwise_warp.json` must still pass.

### 1.2 Share V matrix between predict and predictCovarianceWith

When relativization workflows call both methods with the same test points, V is
computed twice. Add an optional cache or combined method.

**Files:** `src/models/gp.ts`
**Test:** Verify identical numerical results with and without caching.

### 1.3 Add computeDiag to all kernels

Add a `computeDiag(X)` method to the kernel interface that returns only the
diagonal of `K(X, X)`. This is the prerequisite for diagonal-only Kss (Tier 2).

**Files:** `src/kernels/types.ts`, `src/kernels/rbf.ts`, `src/kernels/matern.ts`,
`src/kernels/scale.ts`, `src/kernels/composite.ts`, `src/kernels/categorical.ts`,
`src/kernels/multitask.ts`
**Test:** For each kernel, verify `computeDiag(X) === diag(compute(X, X))`.

### 1.4 Reduce `any` usage

Replace remaining `any` casts with proper types. Most are in `src/predictor.ts`
where heterogeneous model types are dispatched — use discriminated unions.
Run `grep -r 'as any\|: any' src/` to find current occurrences.

---

## Tier 2: Medium Impact, Independent

### 2.1 Diagonal-only Kss path

Use `computeDiag` (from 1.3) in `gp.ts` prediction when only variance is needed
(not full covariance). For an 80x80 grid this reduces Kss from 6400x6400
(328 MB) to a 6400-element vector.

**Depends on:** Tier 1.3 (computeDiag on all kernels)
**Files:** `src/models/gp.ts`

### 2.2 Kernel caching for interactive use

Cache the cross-covariance K* and check if test points have changed before
recomputing. Useful for slider-driven visualization where the same grid is
reused across many parameter updates.

**Files:** `src/models/gp.ts`
**Risk:** Cache invalidation complexity. Consider a simple hash of test point
coordinates.

### 2.3 forwardSolveTransposed

Eliminate the explicit transpose allocation in `gp.ts:predict()` by adding a
`forwardSolveTransposed` that reads K* in column-major order.

**Files:** `src/linalg/solve.ts`, `src/models/gp.ts`

### 2.4 ESLint configuration cleanup

The `eslint.config.mjs` and `tsconfig.eslint.json` files are untracked. Integrate
them into the build or remove them. Incrementally enable stricter rules.

---

## Tier 3: Needs Upstream Coordination

These require engagement with the Ax or BoTorch teams. See
`plans/upstream-proposals.md` for full proposals.

### 3.1 Push for `_task_feature` and `_eval_covar_matrix()` stabilization

These are the two highest-risk private attributes. File issues or PRs to make
them public properties on `MultiTaskGP` and `PositiveIndexKernel`.

**Priority:** P0 in upstream-proposals.md
**Blocked on:** BoTorch team review

### 3.2 Propose `model.prediction_state()`

Draft a BoTorch RFC for a `prediction_state()` method that returns a
self-contained dict with everything needed for prediction replication.

**Priority:** P1 in upstream-proposals.md
**Blocked on:** BoTorch design review

### 3.3 Propose adapter transform metadata

Either `adapter.prediction_metadata()` or documentation of `adapter.transforms`
as a stable public API.

**Priority:** P2 in upstream-proposals.md
**Blocked on:** Ax team review

### 3.4 Propose Bilog/Power analytic moments

File an issue for adding delta-method moment approximations to `Bilog` and
`Power` outcome transforms, enabling them as model-level transforms.

**Blocked on:** BoTorch team interest and mathematical review

---

## Tier 4: Future / Large Scope

These require RFCs and are not actionable without further design work.

### 4.1 WASM linear algebra

For n > 100, compiled BLAS (via WASM) would significantly accelerate Cholesky
and matrix multiplications. Requires evaluating WASM-BLAS libraries, Web Worker
integration, and SharedArrayBuffer support.

### 4.2 Predictor-acquisition integration

Currently `Predictor` and `optimizeAcqf` are loosely coupled. A tighter
integration could share kernel evaluations and cache the posterior state across
acquisition function evaluations.

### 4.3 Hierarchical search space support

Ax supports hierarchical (conditional) search spaces where some parameters are
only active when others take specific values. ax-js currently treats all
parameters as unconditionally active.

### 4.4 WebGPU acceleration

For very large n (>1000), WebGPU-based matrix operations. Requires the WebGPU
API to stabilize across browsers and a compute shader implementation of Cholesky
and triangular solves.

---

## Sequencing

```
Tier 1 (all independent, can parallelize):
  1.1 LU caching
  1.2 V matrix sharing
  1.3 computeDiag on kernels
  1.4 Reduce any count

Tier 2 (after prerequisites):
  2.1 Diagonal Kss  <-- depends on 1.3
  2.2 Kernel caching (independent)
  2.3 forwardSolveTransposed (independent)
  2.4 ESLint cleanup (independent)

Tier 3 (blocked on upstream):
  3.1 Stabilize private attrs  <-- file issues first
  3.2 prediction_state() RFC   <-- after 3.1 accepted
  3.3 Adapter metadata         <-- independent of 3.1/3.2
  3.4 Bilog/Power moments      <-- independent

Tier 4 (needs RFCs):
  4.1 WASM linalg
  4.2 Predictor-acquisition integration
  4.3 Hierarchical search space
  4.4 WebGPU
```

Tier 1 items are the immediate focus. Tier 2.1 is the highest-impact item that
has a dependency. Tier 3 should be initiated in parallel via upstream issues.
Tier 4 items are tracked but not scheduled.
