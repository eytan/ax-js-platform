# Testing Guide

## Quick Reference

```bash
npx vitest run                        # Run all tests
npx tsc --noEmit                      # Type-check
python python/generate_fixtures.py    # Regenerate BoTorch parity fixtures
npx tsup                              # Build bundle
```

## Testing Architecture

Three layers of tests ensure correctness at increasing scope:

| Layer | File | Ground Truth | Purpose |
|-------|------|-------------|---------|
| Unit | `test/` | Hand-computed | Mathematical properties |
| BoTorch-level | `botorch_parity.test.ts` | Raw BoTorch posterior | Low-level model math consistency |
| **Ax-level** | **`predictor_parity.test.ts`** | **Ax `adapter.predict()`** | **Authoritative end-to-end parity** |

### 1. Unit Tests (`test/`)

File-mirroring convention: `src/kernels/matern.ts` → `test/kernels/matern.test.ts`.

- Mathematical properties (symmetry, positive-definiteness, special cases)
- Formula correctness against hand-computed values
- Edge cases (empty inputs, single points, degenerate parameters)

### 2. BoTorch Parity Fixtures (`test/integration/botorch_parity.test.ts`)

Low-level tests that verify axjs model predictions match BoTorch exactly
(within tolerance). Tests raw model math — NOT authoritative for end-user
predictions (see Ax-level parity below).

```
BoTorch model → generate_fixtures.py → JSON fixture → botorch_parity.test.ts → axjs prediction → compare
```

Each fixture uses the `{experiment, test}` schema:

```typescript
interface FixtureData {
  experiment: ExperimentState;  // search_space, model_state, outcome_names, etc.
  test: {
    metadata: { botorch_version, seed, benchmark, ... };
    test_points: number[][];
    expected: { mean: number[] | null; variance: number[] | null };
    expected_relative?: { mean: number[]; variance: number[] };
  };
}
```

- **No per-fixture tolerances.** A single global tolerance of `1e-6` is used with numpy `allclose`
  semantics: `|actual - expected| <= atol + atol * |expected|`. This handles large-magnitude values
  correctly (e.g., variances of ~10¹¹ for PressureVessel).
- **Consistency-only fixtures** (Bilog/Power) have `expected.mean === null` — the test infers
  this means "no BoTorch reference; verify model loads and predictions are finite."
- **Status quo** point (if present) is the first entry in `test.test_points` with its
  expected values at index 0 of the expected arrays.

### 3. Predictor ↔ Ax Parity (`test/integration/predictor_parity.test.ts`)

**Authoritative end-to-end parity.** Ground truth comes from Ax's actual
`adapter.predict()` — the same pipeline real users interact with. This is the
definitive test that axjs predictions match Ax.

```
Ax Experiment + Data → Adapter(transforms) → adapter.predict() → reference
Fixture JSON → Predictor.predict() → compare against reference
```

Ax-level fixtures have `test.metadata.ax_level = true` and test:
- Adapter transforms: LogY, BilogY, StandardizeY, PowerTransformY
- Combined adapter + model transforms (e.g., LogY adapter + Standardize model)
- Multi-output with per-metric adapter transforms
- MultiTaskGP predictions for all tasks
- Relativization via `Predictor.predictRelative()`

### 4. Integration Tests (`test/integration/relativize_parity.test.ts`)

Property-based tests for covariance and relativization:
- Self-covariance = variance (fundamental identity)
- Covariance symmetry and Cauchy-Schwarz bound
- Relativize/unrelativize round-trip
- Covariance tightens relative confidence intervals

## Fixture System

### Manifest

`test/fixtures/manifest.json` lists all fixtures. The test harness validates:

- **MINIMUM_FIXTURES** (32 BoTorch-level): prevents silent fixture loss. Ax-level
  fixtures are excluded from this count and tested in `predictor_parity.test.ts`.
- All referenced files exist on disk

### Report Card

After running all parity fixtures, the harness prints a discrepancy summary:

```
Fixture                       Model           N  Mean(avg)    Mean(max)    Var(avg)     Var(max)
branin_matern25               SingleTaskGP   20  2.30e-15     7.11e-15     1.60e-10     1.60e-10
...
Worst-case: mean=7.11e-15, variance=1.60e-10
Tolerance: 1.00e-6 (global)
```

## How to Add a Fixture

### 1. Define a FixtureSpec

In `python/generate_fixtures.py`, add to `FIXTURE_SPECS`:

```python
FixtureSpec(
    name="branin_my_feature",     # Unique name, becomes filename
    benchmark="Branin",            # "Branin" | "Hartmann" | "BraninCurrin"
    model_class="SingleTaskGP",    # Model type
    n_train=15,                    # Training points
    n_test=20,                     # Test points for parity comparison
    seed=99,                       # Deterministic seed (must be unique)
    description="...",             # Human-readable description
)
```

### 2. FixtureSpec Fields

| Field | Default | Description |
|-------|---------|-------------|
| `kernel_type` | `"Matern"` | `"Matern"` or `"RBF"` |
| `nu` | `2.5` | Matérn smoothness: 0.5, 1.5, or 2.5 |
| `use_warp` | `False` | Kumaraswamy input warping |
| `use_composite` | `False` | Product(continuous × Categorical) kernel |
| `use_additive` | `False` | AdditiveKernel (per-dim) |
| `cat_dims` | `None` | Categorical dimension indices |
| `int_dims` | `None` | Integer parameter dimensions |
| `noise` | `None` | Fixed noise level (makes FixedNoiseGP) |
| `heteroscedastic_noise` | `False` | Per-point noise ∝ \|y\| |
| `outcome_type` | `"Standardize"` | `"Standardize"`, `"Log"`, `"Bilog"`, `"Power"`, `"Chained"` |
| `negate_benchmark` | `True` | Negate Y (False for Log, which needs Y > 0) |
| `num_tasks` | `0` | >0 for MultiTaskGP |
| `status_quo` | `None` | `"center"` to include relativization reference |
| `ax_level` | `False` | Generate through Ax adapter pipeline |
| `adapter_transforms` | `None` | List of adapter Y-transforms: `["LogY"]`, etc. |
| `all_tasks` | `False` | Predict for all tasks (MultiTaskGP) |
| `objectives` | `None` | `[{name, minimize}]` — optimization objective(s) |
| `outcome_constraints` | `None` | `[{name, bound, op}]` — feasibility constraints |
| `objective_thresholds` | `None` | `[{name, bound, op}]` — MOO reference point thresholds |

### 3. Regenerate and Bump

```bash
python python/generate_fixtures.py   # Generate new fixture JSON
# Update MINIMUM_FIXTURES in botorch_parity.test.ts
npx vitest run                        # Verify all pass
```

## How to Add a Unit Test

Conventions:
- Mirror source structure: `src/foo/bar.ts` → `test/foo/bar.test.ts`
- Use vitest: `import { describe, it, expect } from "vitest"`
- Floating-point comparison: `expect(x).toBeCloseTo(y, decimals)` or custom `expectAllClose`
- Test mathematical properties (symmetry, bounds) not just happy paths
- Test all `throw` statements with `expect(() => ...).toThrow()`

## When to Run Tests

| Changed | Run |
|---------|-----|
| Any TypeScript source | `npx tsc --noEmit && npx vitest run` |
| Kernels, transforms, models | Also regenerate fixtures if extraction changed |
| `_extraction.py` or `generate_fixtures.py` | `python python/generate_fixtures.py && npx vitest run` |
| Upgrading BoTorch/GPyTorch | Full regenerate + test |
| Docs only, build config | No tests needed |

## Python Environment

```bash
pip install -r python/requirements.txt
# Requires: botorch>=0.17, gpytorch>=1.10, ax-platform>=1.2
```

Fixture generation is **deterministic** (seeded). Same BoTorch version → same fixtures.

## Fixture Coverage Matrix

| Fixture | Model | Feature |
|---------|-------|---------|
| `branin_matern25` | SingleTaskGP | Matérn 5/2, ARD, Normalize+Standardize |
| `branin_rbf` | SingleTaskGP | RBF kernel |
| `branin_matern05` | SingleTaskGP | Matérn ν=0.5 (exponential) |
| `branin_matern15` | SingleTaskGP | Matérn ν=1.5 |
| `branin_fixed_noise` | FixedNoiseGP | Homoscedastic fixed noise |
| `branin_heteroscedastic` | FixedNoiseGP | Heteroscedastic noise (var ∝ \|y\|) |
| `branin_warp` | SingleTaskGP | Kumaraswamy input warping |
| `branin_1pt` | SingleTaskGP | Edge case: n_train=1 |
| `branin_additive` | SingleTaskGP | AdditiveKernel (per-dim Matérn) |
| `branin_log` | SingleTaskGP | Log outcome transform |
| `branin_chained_log_std` | SingleTaskGP | Chained(Log, Standardize) |
| `branin_bilog` | SingleTaskGP | Bilog outcome (consistency only) |
| `branin_power` | SingleTaskGP | Power(λ=0.5) outcome (consistency only) |
| `branin_pairwise` | PairwiseGP | Laplace + LU |
| `pairwise_warp` | PairwiseGP | + input warping |
| `hartmann_6d` | SingleTaskGP | 6D, strong ARD |
| `hartmann_integer` | SingleTaskGP | Integer parameters |
| `hartmann_mixed` | SingleTaskGP | Product(Matérn × Categorical) |
| `hartmann_mixed_warp` | SingleTaskGP | Mixed + warp |
| `branincurrin_modellist` | ModelListGP | 2 outcomes |
| `branincurrin_fixednoise_modellist` | ModelListGP | + FixedNoise sub-models |
| `branincurrin_warp_fixednoise_modellist` | ModelListGP | + Warp + heteroscedastic |
| `branin_multitask` | MultiTaskGP | 2 tasks (ICM) |
| `multitask_mixed` | MultiTaskGP | + categorical |
| `multitask_warp` | MultiTaskGP | + input warping |
| `saas_highdim_nuts` | EnsembleGP | Fully Bayesian SAAS (NUTS) |
| `saas_highdim_map` | EnsembleGP | Multi-restart MAP |
| `branin_relative` | SingleTaskGP | Relativization (noiseless) |
| `branin_relative_fixed_noise` | FixedNoiseGP | Relativization (noisy) |
| `vsip_modellist` | ModelListGP | 7D, 9 outcomes (VSIP) |
| `c2dtlz2_constrained_moo` | ModelListGP | 4D, 2 obj + 1 constraint, constrained MOO |
| `discbrake_constrained_moo` | ModelListGP | 4D, 2 obj + 4 constraints, engineering MOO |
| `pressure_vessel_constrained` | ModelListGP | 4D, 1 obj + 4 constraints, constrained SOO |
| `trajectory_planning_30d` | SingleTaskGP | 30D, high-dimensional single objective |
| **Ax-level (Predictor parity)** | | |
| `ax_branin_logy` | SingleTaskGP | LogY adapter transform |
| `ax_branin_bilogy` | SingleTaskGP | BilogY adapter transform |
| `ax_branin_standardizey` | SingleTaskGP | StandardizeY adapter transform |
| `ax_branin_bilogy_standardize` | SingleTaskGP | BilogY adapter + Standardize model |
| `ax_branincurrin_logy` | ModelListGP | LogY on both metrics |
| `ax_branincurrin_batched` | ModelListGP | Ax default MOO (batched → ModelListGP) |
| `ax_branincurrin_relative` | ModelListGP | Multi-output relativization |
| `ax_multitask_relative` | MultiTaskGP | Multi-task relativization |
| `ax_multitask_all_tasks` | MultiTaskGP | All tasks prediction |
| `ax_branin_powery` | SingleTaskGP | PowerTransformY adapter |
| `ax_branin_log_standardize` | SingleTaskGP | LogY adapter + Standardize model |
