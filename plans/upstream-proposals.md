# Upstream Proposals for Ax and BoTorch

Concrete proposals to reduce ax-js fragility and improve the Ax/BoTorch ecosystem.
Subsumes `docs/internal/serialization-contract.md`.

---

## Proposal A: `model.prediction_state()` for BoTorch

### Problem

There is no public BoTorch API for "give me everything needed to replicate
predictions." The ax-js export code (`_extraction.py`) crawls the model object
tree using a mix of public, semi-public, and private attributes. This broke
when BoTorch 0.16 changed `IndexKernel` to `PositiveIndexKernel` and
`ConstantMean` to `MultitaskMean`. It will break again.

### Proposed method

```python
class Model:
    def prediction_state(self) -> dict:
        """Return a self-contained dict sufficient for posterior mean + variance."""
```

Draft schema:

```python
{
    "format_version": 2,
    "model_type": "SingleTaskGP",          # discriminant
    "train_X": Tensor,                     # (n, d)
    "train_Y": Tensor,                     # (n, 1) or (n, m) for ModelList
    "kernel": {                            # recursive kernel tree
        "type": "Scale",
        "outputscale": float,
        "base_kernel": {
            "type": "RBF",
            "lengthscale": [float, ...],   # per-dimension (ARD)
            "active_dims": [int, ...],
        },
    },
    "mean_constant": float | list[float],
    "noise_variance": float | list[float],
    "input_transform": { ... } | None,
    "outcome_transform": { ... } | None,
    # MultiTaskGP additions:
    "task_feature": int,
    "num_tasks": int,
    "task_covar": { "covar_matrix": [[float, ...], ...] },
}
```

### Benefits

- BoTorch owns which internals matter for prediction.
- Enables a roundtrip regression test: `predict_from_dict(model.prediction_state(), X) ~ model.posterior(X)`.
- Eliminates all private attribute access in ax-js extraction.
- Versionable via `format_version` field.
- Could serve other consumers (model cards, debugging tools, model comparison).

### Versioning strategy

Include `format_version` in the dict. BoTorch bumps the version when the schema
changes. Consumers declare which versions they support. ax-js currently uses
format version 2.

---

## Proposal A (minimal): Stabilize 5 Private Attributes

If `prediction_state()` is too large a surface area, stabilizing five attributes
would cover the highest-risk breakage points.

| # | Attribute | Current status | Needed for | Breakage risk |
|---|-----------|---------------|------------|---------------|
| 1 | `MultiTaskGP._task_feature` | Private | Task column index in train_X | **High** -- no alternative |
| 2 | `PositiveIndexKernel._eval_covar_matrix()` | Private method | Fully-evaluated B matrix | **High** -- only way to get it |
| 3 | `MultitaskMean.constants` | Semi-public | Per-task mean constants | Medium |
| 4 | `ModelListGP._outcome_names` | Private | Outcome-to-model mapping | Medium |
| 5 | `IndexKernel.raw_var` | Private | Diagonal variance (legacy) | Low (0.16+ uses PositiveIndexKernel) |

**Minimal ask:** Make items 1 and 2 public properties. These have no alternative
access path and are the most likely to break silently in a refactor.

---

## Proposal B: `adapter.prediction_metadata()` for Ax

### Problem

Ax adapter-level transforms (`LogY`, `BilogY`, `PowerTransformY`, `StandardizeY`)
are applied before data reaches BoTorch and are invisible in the model state.
Without explicit metadata, client-side predictions are in the wrong space.

The transform pipeline:

```
                    Ax Adapter                              BoTorch Model
               +-------------------+                  +-----------------------+
               |                   |                  |                       |
  Raw Y_raw ---+  LogY / BilogY    +--- Y_adapted --- +  Standardize (model)  |
               |  PowerTransformY  |                  |  outcome_transform    |
               |  StandardizeY     |                  |                       |
               |                   |                  |                       |
  Raw X_raw ---+  IntToFloat       +--- X_adapted --- +  Normalize            |
               |  OneHot           |                  |  Warp (Kumaraswamy)   |
               |  Normalize        |                  |  input_transform      |
               |                   |                  |                       |
               +-------------------+                  +-----------------------+
                                                                |
               INVISIBLE to model export                        | VISIBLE
               (need adapter metadata)               +---------+-----------+
                                                      |  ax-js JSON export |
                                                      +--------------------+
```

### Proposed method

```python
class Adapter:
    def prediction_metadata(self) -> dict:
        """Return Y-transform chain applied before model fitting."""
```

Return value:

```python
{
    "y_transforms": [
        {"type": "LogY", "metrics": ["latency"]},
        {"type": "StandardizeY", "Ymean": {...}, "Ystd": {...}},
    ],
    "x_transforms": [
        {"type": "IntToFloat", "parameters": ["batch_size"]},
    ],
}
```

### Minimal alternative

Document `adapter.transforms` (an `OrderedDict[str, Transform]`) as a stable
public API. Currently ax-js introspects this dict, but it is not documented as
public and could change without notice.

---

## BoTorch Design Observations

These are patterns in BoTorch that cause friction for downstream consumers.
They are not bugs, but documenting them would help.

### 1. Bilog and Power lack analytic moments

BoTorch's `Bilog` and `Power` outcome transforms provide only `sample_transform`
(for MC acquisition). Calling `.mean` or `.variance` raises `NotImplementedError`.
This means they cannot be used as model-level outcome transforms in practice --
Ax always applies them at the adapter layer. Log is the exception: it has exact
log-normal moments via `norm_to_lognorm_mean/variance`.

**Suggestion:** Either add analytic moment approximations (delta method) or
document that these are adapter-only transforms.

### 2. Relativization ignores posterior covariance

Ax's `relativize()` always passes `cov_means=0`, treating test and status quo
predictions as independent. For GP models, predictions at nearby points are
positively correlated. Ignoring this covariance makes relative CIs wider than
necessary (conservative but wasteful).

ax-js implements covariance-aware relativization via `Predictor.getCovariances()`
and `relativizePredictions()`. See "Potential backports" below.

### 3. Matern mean-centering is undocumented

GPyTorch's `MaternKernel` subtracts `x1.mean(dim=0)` from both inputs before
computing distances. RBF does not. This is a numerical stability trick that
affects computed distances when x1 has non-zero mean. It is not documented in
the GPyTorch API or docstrings.

### 4. Warp epsilon normalization

BoTorch's Warp (Kumaraswamy) transform uses `x * (1 - 2*eps) + eps` with
`eps = 1e-7`, not simple clamping. This is a subtle detail that affects parity
if implemented incorrectly.

---

## Potential Backports (JS to Python)

ax-js implements several features that have no Python equivalent. These could
benefit the broader Ax/BoTorch ecosystem.

### 1. Analytic Sobol' sensitivity indices

Closed-form Sobol' indices for RBF product kernels, Gauss-Legendre quadrature
for warped dimensions, and ensemble support for SAAS models. Significantly faster
than MC-based sensitivity (no sampling needed). See `src/sensitivity_analytic.ts`.

### 2. Analytic leave-one-out cross-validation

Rasmussen & Williams (2006) Equation 5.12: LOO-CV from a single Cholesky
factorization, O(n^2) instead of O(n * n^3) for refit-based CV. Ax's
`cross_validate` refits the model for each fold.

### 3. Covariance-aware relativization

Uses `Cov(f(x), f(x_sq))` from the GP posterior to compute tighter CIs for
relative effects. Drop-in replacement for `cov_means=0`.

---

## Prioritization

| Priority | Proposal | Team | Effort | Unblocks |
|----------|----------|------|--------|----------|
| **P0** | Stabilize `_task_feature` and `_eval_covar_matrix()` | BoTorch | Small | Reduces highest-risk breakage |
| **P1** | `model.prediction_state()` | BoTorch | Medium | Eliminates all private attr access |
| **P2** | `adapter.prediction_metadata()` or document `adapter.transforms` | Ax | Small | Adapter transform round-trip |
| **P3** | Backport analytic Sobol' to BoTorch | BoTorch | Medium | Python-side sensitivity analysis |
| **P3** | Backport analytic LOO-CV to Ax | Ax | Small | Faster cross-validation |
| **P3** | Backport covariance-aware relativization | Ax | Small | Tighter relative CIs |

P0 is independent and low-risk. P1 is the real goal but requires design review.
P2 is independent of P0/P1. P3 items are nice-to-haves that can proceed in
parallel.

---

## References

- Current parity evidence: 46 fixtures, worst discrepancy ~1e-10. See `docs/testing.md`.
- Numerical details: `docs/internal/numerics.md`.
- JSON schema and transform formats: `docs/data-model.md`.
- Extraction code: `python/_extraction.py`.
