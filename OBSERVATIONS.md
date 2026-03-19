# Observations: BoTorch/Ax Foot-Guns

Discovered issues and gotchas when working with ax-js model states exported from Ax/BoTorch.

## 1. Missing `input_transform` silently breaks analytic Sobol

If a model state doesn't include `input_transform`, the `getInternals().trainXNorm` field contains raw parameter-space values (e.g., [-5, 10] for Branin). The analytic Sobol integrator computes integrals over [0, 1], so it produces near-zero sensitivity indices with no error.

**Fix**: `predictor.ts` now guards against this — returns `null` from the analytic path when `input_transform` is absent, triggering the MC Saltelli fallback which evaluates through `predictor.predict()` end-to-end.

## 2. Two-layer transform confusion

Ax applies transforms at **two separate layers**:

- **Layer 1 (Adapter)**: `LogY`, `BilogY`, `PowerTransformY`, `StandardizeY` — applied by Ax's adapter BEFORE data reaches BoTorch. Not stored in model state.
- **Layer 2 (Model)**: `Standardize`, `Log`, `Bilog`, `Power` — stored in `model.outcome_transform` within BoTorch.

`model_state.train_Y` has BOTH layers applied. To recover original-space Y values, you must undo both in reverse order. Easy to forget one layer.

## 3. Lengthscales depend on coordinate system

Lengthscales are in whatever space `trainXNorm` lives in:
- **With `input_transform`**: lengthscales are in [0, 1] normalized space
- **Without**: lengthscales are in original parameter space

Comparing lengthscales across models requires knowing whether normalization was applied. A lengthscale of 0.5 means very different things in [0, 1] space vs [0, 100] space.

## 4. `getInternals().trainXNorm` is a lie without `input_transform`

The field is called `trainXNorm` but it's only actually normalized if `input_transform` exists. The GP constructor (`gp.ts`) does `trainX.clone()` when there's no transform — the name is misleading.

## 5. Fixture generation foot-gun

`generate_fixtures.py` should always include `input_transform` for fixtures that will be used with analytic sensitivity analysis. Without it:
- The MC path works fine (evaluates end-to-end through `predict()`)
- The analytic path silently produces zeros (now correctly guarded)

Affected fixtures: `branin_rbf`, `branincurrin_modellist`, `hartmann_6d`, `vsip_modellist` — all lack `input_transform` and fall back to MC.
