# Add a new parity fixture

The user wants to add a new test fixture for BoTorch/Ax parity testing.
Use the description below to determine the right fixture configuration.

## User's description

$ARGUMENTS

## Steps

1. **Read the current FixtureSpecs** in `python/generate_fixtures.py` to find the next spec number and ensure the name/seed are unique.

2. **Determine the fixture type** from the user's description:
   - **SingleTaskGP**: Single-output, standard GP. Use `model_class="SingleTaskGP"`.
   - **FixedNoiseGP**: Fixed observation noise. Use `model_class="FixedNoiseGP"` with `noise=<value>`.
   - **ModelListGP**: Multi-output. Use `model_class="ModelListGP"` with `n_outcomes=<N>` and `outcome_names=[...]`.
   - **PairwiseGP**: Pairwise comparisons. Use `model_class="PairwiseGP"`.
   - **MultiTaskGP**: Multi-task. Use `model_class="MultiTaskGP"` with `num_tasks=<N>`.
   - **EnsembleGP**: Fully Bayesian ensemble. Use `model_class="EnsembleGP"`.
   - **Ax-level**: Tests adapter transforms. Add `ax_level=True` and `adapter_transforms=[...]`.

3. **Choose a benchmark** from the registry in `python/benchmarks.py`:
   - `Branin` (2D), `Hartmann` (6D), `BraninCurrin` (2D, 2 outputs), `VSIP` (7D, 9 outputs)
   - `C2DTLZ2` (4D, 2 obj + 1 constraint), `DiscBrake` (4D, 2 obj + 4 constraints)
   - `PressureVessel` (4D, 1 obj + 4 constraints), `TrajectoryPlanning` (30D)
   - `synthetic` for PairwiseGP

4. **Add the FixtureSpec** to the `FIXTURE_SPECS` list in `python/generate_fixtures.py`. Key fields:
   - `name`: Unique snake_case name (becomes filename)
   - `benchmark`: Benchmark name from registry
   - `model_class`: Model type
   - `n_train`, `n_test`: Training/test sizes (typical: 15-30 train, 15-20 test)
   - `seed`: Unique integer seed
   - `description`: Human-readable description
   - Optional: `kernel_type`, `nu`, `use_warp`, `use_composite`, `noise`, `outcome_type`, etc.
   - For constrained optimization: `objectives`, `outcome_constraints`, `objective_thresholds`

5. **Regenerate fixtures**: `python python/generate_fixtures.py`

6. **Update MINIMUM_FIXTURES** in `test/integration/botorch_parity.test.ts` if this is a BoTorch-level (non-ax-level) fixture.

7. **Run tests**: `npx vitest run`

8. **Update docs**: Add the fixture to the coverage matrix in `TESTING.md`.

## Sanity checks (automatic)

The test harness automatically validates that:
- Expected mean values have meaningful signal (not all near-zero / trivially ~0)
- Expected variance values exist and are valid
- Fixture count meets the minimum threshold
