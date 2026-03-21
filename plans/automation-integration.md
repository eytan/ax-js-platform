# Automation Integration Plan

How to keep ax-js documentation, tests, and parity checks running as the
project is adopted by the Ax/BoTorch engineering team.

---

## What exists today

### Manual checks

| Check | Command | When to run |
|-------|---------|-------------|
| Doc-codebase drift | `bash scripts/verify-docs.sh` | Before doc PRs, periodically |
| Type safety | `npx tsc --noEmit` | Before any PR |
| Full test suite | `npx vitest run` | Before any PR |
| Fixture regeneration | `python python/generate_fixtures.py` | After extraction or model changes |

### What `verify-docs.sh` checks

Structural integrity only — things that break silently:
- File paths referenced in docs still exist on disk
- Cross-references between `plans/*.md` docs resolve
- No active links to deleted docs
- `CLAUDE.md` references the `plans/` directory

It does NOT enforce counts, duplicate test-suite checks, or verify semantic
accuracy. Those require periodic human or agent review.

---

## Proposed CI integration

When ax-js moves into a CI environment (GitHub Actions, Meta internal CI, etc.),
these checks map naturally to pipeline stages:

### Pre-commit / pre-push hooks

```yaml
# Lightweight, < 5 seconds
- npx tsc --noEmit
- bash scripts/verify-docs.sh
```

These catch type errors and doc drift before code reaches review. Fast enough
to run on every commit without friction.

### PR gate checks

```yaml
# Full validation, < 2 minutes
- npx tsc --noEmit
- npx vitest run
- bash scripts/verify-docs.sh
```

Block merge if any check fails. The parity test suite prints a report card
showing per-fixture discrepancy statistics — reviewers can check this for
regressions even when all tests pass.

### Periodic / nightly checks

```yaml
# Upstream drift detection, ~5 minutes
- pip install --upgrade botorch ax-platform
- python python/generate_fixtures.py
- npx vitest run
- # Compare fixture JSON against checked-in versions
- # Alert if discrepancies found (BoTorch changed behavior)
```

This catches upstream breaking changes before they surprise developers.
The agent playbook (`plans/agent-playbook.md`) documents the manual workflow
for handling these — CI just automates the trigger.

### Periodic agent review (weekly/monthly)

Run fresh zero-context agents against the documentation with tasks like:
- "Add a PeriodicKernel" (tests checklist completeness)
- "BoTorch renamed an attribute" (tests incident workflow)
- "Review this code for violations" (tests rule discoverability)

This catches semantic drift that structural checks cannot. See the test
protocol used during initial doc validation for examples.

---

## Integration points for Meta infrastructure

| Meta system | What it would run | Notes |
|-------------|-------------------|-------|
| Sandcastle / CI | PR gate checks | Standard TS project setup |
| Cron job | Nightly upstream drift check | pip upgrade + fixture regen |
| Code review bot | Remind to run `verify-docs.sh` if `plans/` or `docs/` changed | Low-effort integration |
| Pre-commit hook (lint) | `tsc --noEmit` | Already standard for TS projects |

The script and test suite are portable — no Meta-specific dependencies.
The only Meta-specific step would be wiring them into the CI system.

---

## What NOT to automate

- **Semantic accuracy of docs** — whether the intentional divergences table is
  still current, whether upstream proposals are still relevant. This requires
  judgment, not scripts.
- **Hard counts** (fixture count, `any` count) — these change as the codebase
  improves. The docs describe patterns, not inventory. The test suite enforces
  minimums where it matters (`MINIMUM_FIXTURES`).
- **Style guide compliance of existing code** — the style guide explicitly says
  "incremental adoption." Linting new code is ESLint's job, not a doc check.
