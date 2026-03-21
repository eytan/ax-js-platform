# Agent Playbook: ax-js Parity PRs

Complete workflow for agents producing parity PRs against ax-js.
This document prevents common mistakes and ensures quality.

Cross-references: `CLAUDE.md`, `plans/parity-guide.md`, `plans/style-guide.md`

---

## 1. Detecting Upstream Changes

When a new version of BoTorch or Ax is released, follow this sequence:

1. **Read the changelogs first** — before running anything, check upstream release notes.
   - **BoTorch changelog**: look for renamed classes/attributes (e.g., `IndexKernel` to
     `PositiveIndexKernel`), new defaults (e.g., RBF replacing Matern), new model types,
     restructured state dict keys.
   - **Ax changelog**: look for changes to adapter transforms (LogY, BilogY,
     PowerTransformY, StandardizeY), `ExperimentState` schema, deprecated APIs.
   - Search ax-js for references to changed components: `grep -r "OldClassName" python/ src/`.
   - This step scopes the effort and prevents debugging cryptic extraction errors blind.

2. **Regenerate fixtures** — run `python python/generate_fixtures.py`.
   - Extraction failures (Python errors) reveal breaking changes in BoTorch internals:
     renamed attributes, new class hierarchies, removed fields.
   - If extraction succeeds but produces different JSON, the model numerics changed.

3. **Run parity tests** — run `npx vitest run`.
   - Parity failures in `botorch_parity.test.ts` reveal math changes (new defaults,
     changed kernel parameterizations, updated transforms).
   - Parity failures in `predictor_parity.test.ts` reveal Ax-level changes (adapter
     transforms, outcome naming, search space handling).

---

## 2. PR Checklist

Before submitting any PR, verify ALL of the following:

- [ ] `npx tsc --noEmit` passes with zero errors.
- [ ] `npx vitest run` passes with zero failures.
- [ ] `npx eslint .` passes (or only pre-existing warnings remain).
- [ ] If `_extraction.py` changed: fixtures regenerated via `python python/generate_fixtures.py`.
- [ ] If a new model, kernel, or transform was added: a new fixture with a unique seed
      exists, and `MINIMUM_FIXTURES` in the test suite has been bumped.
- [ ] If types changed (any `*State` interface in `types.ts`): `docs/data-model.md` updated.
- [ ] No formatting changes mixed with logic changes in the same commit.
- [ ] No new `@ts-ignore` directives.
- [ ] No new untyped `any` without a justifying comment.
- [ ] No relaxed tolerances (global ceiling is `1e-6`; never exceed `1e-5`).
- [ ] Commit messages are descriptive (not "fix stuff" or "update").
- [ ] `bash scripts/verify-docs.sh` passes (checks doc-codebase drift).

---

## 3. Guardrails

### Size limits

- **Max 600 lines of logic diff per PR, not counting fixtures.** Split larger changes into sequential PRs
  (e.g., kernel implementation first, then model integration).
- Fixture JSON changes do not count toward the 400-line limit.
- Pure rename/refactor PRs are separate from logic PRs.

### Quality gates

- **No unnecessary abstractions.** Do not add wrapper classes, factory patterns, or
  indirection layers unless they solve a concrete, demonstrated problem.
- **One concern per PR.** A kernel fix and a viz enhancement are two PRs.
- **No speculative code.** Do not add support for model types or transforms that are
  not yet exported by `_extraction.py`.

### Hard rules (violations are blocking)

- NEVER relax a tolerance past `1e-5`. If parity fails at `1e-6`, investigate the
  root cause. The computation pipeline is Float64 throughout; discrepancies above
  `1e-5` indicate a real numerical bug.
- NEVER skip fixture regeneration after changing extraction logic.
- NEVER use `AxClient`. It is deprecated. Use only `ax.api.Client`.
- NEVER push to remote without explicit user confirmation.
- NEVER use `loadModel` in application code. Use `Predictor`.

---

## 4. Verifying Parity

After making changes, run the full test suite:

```bash
npx vitest run
```

The parity test suites (`botorch_parity.test.ts`, `predictor_parity.test.ts`) print a
report card after execution showing per-fixture discrepancy statistics. Review this output:

1. **All fixtures must pass.** Zero tolerance for skipped or failing fixtures.
2. **Investigate discrepancies > 1e-10.** Discrepancies in the 1e-10 to 1e-6 range are
   acceptable but worth understanding. They typically arise from:
   - Cholesky jitter differences (escalating sequence: 1e-6, 1e-5, 1e-4, 1e-3).
   - Matérn mean-centering (GPyTorch subtracts `x1.mean(axis=0)` from both inputs).
   - Warp epsilon normalization (`x*(1-2e)+e` with `e=1e-7`).
3. **Check variance predictions**, not just means. Variance parity failures often
   indicate Cholesky or solve bugs that are masked in the mean.
4. **For multi-output models**, check each outcome independently. ModelListGP can
   have parity on outcome 0 but fail on outcome 1.

---

## 5. When to Escalate

Stop and ask for human guidance when:

- **Ambiguous design decisions**: The BoTorch change could be implemented multiple
  valid ways, and the choice affects the public API or serialization format.
- **Diff exceeds 400 lines of logic**: The change is too large for a single PR and
  needs a decomposition plan.
- **Tolerance questions**: A fixture fails at 1e-6 and you cannot identify the
  numerical root cause after investigating.
- **New model types**: Adding support for a model type not yet in the codebase
  (e.g., a new GP variant) requires design review.
- **Predictor API changes**: Any change to the public `Predictor` interface affects
  all downstream consumers and needs review.
- **ExperimentState schema changes**: Any change to the wire format affects both
  Python export and TypeScript deserialization.
- **Transform pipeline changes**: The two-layer transform architecture (adapter +
  model) is subtle. Changes here have high bug potential.
- **Breaking changes in _extraction.py**: If the shared extraction module needs
  restructuring, both `axjs_export.py` and `generate_fixtures.py` are affected.

---

## 6. Common Patterns

### Adding a new kernel

1. Create `src/kernels/<name>.ts` with the kernel class implementing the kernel interface.
2. Add the kernel to `src/kernels/build.ts` (the kernel factory/dispatcher).
3. Add deserialization support in `src/io/deserialize.ts` if the kernel has new state keys.
4. Add extraction support in `python/_extraction.py`.
5. Create a fixture spec in `python/generate_fixtures.py` with a unique seed.
6. Run `python python/generate_fixtures.py` to generate the fixture JSON.
7. Add unit tests in `test/kernels/<name>.test.ts`.
8. Run `npx vitest run` to confirm parity.
9. Bump `MINIMUM_FIXTURES` in the test suite.

### Fixing a parity regression

1. Identify the failing fixture(s) and the magnitude of discrepancy.
2. Check if the regression is in mean, variance, or both.
3. Compare the fixture JSON against BoTorch output (use `python/generate_fixtures.py`
   with `--fixture <name>` to regenerate a single fixture).
4. Narrow down: is the issue in kernel evaluation, Cholesky factorization, solve,
   or transform application?
5. Write a minimal unit test reproducing the discrepancy.
6. Fix the root cause. Do not adjust tolerances.
7. Confirm all other fixtures still pass.

### Updating for a new BoTorch version

1. Update BoTorch in the Python environment.
2. Read the BoTorch changelog — identify renamed classes, new defaults, removed APIs.
3. Search ax-js for references to changed components before running anything.
4. Run `python python/generate_fixtures.py` and note all failures.
5. Group failures by category (renamed attributes, new defaults, new classes).
6. Fix extraction (`_extraction.py`) first — this is the bridge between BoTorch and ax-js.
7. Fix TypeScript deserialization and model code as needed.
8. Regenerate all fixtures once extraction is stable.
9. Run full parity suite and verify report card.

---

## 7. Negative Rules

Agents must NEVER do the following:

1. **NEVER relax a test tolerance.** The 1e-6 ceiling is absolute. Discrepancies
   above 1e-5 are bugs, not precision limits.

2. **NEVER skip or disable a fixture.** If a fixture fails, fix the code, not the test.

3. **NEVER use `AxClient`.** It is deprecated. Use `ax.api.Client` exclusively.

4. **NEVER use `loadModel` in application-level code.** Use `Predictor`, which handles
   input transforms, output transforms, adapter untransforms, and outcome naming.

5. **NEVER read `train_Y` directly for original-space values.** `train_Y` has been
   transformed by both adapter and model layers. Use `Predictor.untransformTrainY()`.

6. **NEVER mix formatting changes with logic changes.** Formatting PRs and logic PRs
   are separate.

7. **NEVER add `@ts-ignore` or `as any` casts** without a comment explaining why
   the type system cannot express the correct type.

8. **NEVER push to remote** without explicit user confirmation.

9. **NEVER add speculative features.** Only implement what is needed to match current
   BoTorch/Ax behavior. Do not add "future-proofing" abstractions.

10. **NEVER modify the ExperimentState schema** without updating both Python export
    (`axjs_export.py`) and TypeScript deserialization (`deserialize.ts`), plus
    `docs/data-model.md`.
