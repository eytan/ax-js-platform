/**
 * BoTorch-level parity tests — low-level model math consistency.
 *
 * Ground truth: raw BoTorch posterior (model-space, no adapter transforms).
 * These verify that axjs's kernel, Cholesky, and transform implementations
 * reproduce BoTorch's exact numerical output.
 *
 * NOT authoritative for end-user predictions. For Ax-level parity (what users
 * actually see after adapter transforms), see predictor_parity.test.ts.
 */
import { describe, it, expect } from "vitest";
import { readFileSync, existsSync } from "fs";
import { join, dirname } from "path";
import { fileURLToPath } from "url";
import { loadModel } from "../../src/io/deserialize.js";
import { SingleTaskGP } from "../../src/models/single_task.js";
import { ModelListGP } from "../../src/models/model_list.js";
import { PairwiseGP } from "../../src/models/pairwise_gp.js";
import { MultiTaskGP } from "../../src/models/multi_task.js";
import { EnsembleGP } from "../../src/models/ensemble_gp.js";
import type { FixtureData, Manifest, PredictionResult } from "../../src/models/types.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
const fixturesDir = join(__dirname, "..", "fixtures");

const MINIMUM_FIXTURES = 32;

// Single global tolerance for all fixtures.
// All computations use Float64 — any diff > 1e-5 indicates a real bug.
const TOLERANCE = 1e-6;

function loadFixture(filename: string): FixtureData {
  const raw = readFileSync(join(fixturesDir, filename), "utf-8");
  return JSON.parse(raw);
}

function isConsistencyOnly(fixture: FixtureData): boolean {
  return fixture.test.expected.mean === null;
}

function isAxLevel(fixture: FixtureData): boolean {
  return fixture.test.metadata?.ax_level === true;
}

function validateFixture(fixture: FixtureData, name: string): void {
  if (!fixture.experiment.model_state) {
    throw new Error(`Fixture "${name}": missing experiment.model_state`);
  }
  if (!fixture.test.test_points || !Array.isArray(fixture.test.test_points)) {
    throw new Error(`Fixture "${name}": missing or invalid test.test_points`);
  }
  // Consistency-only fixtures have null expected values
  if (isConsistencyOnly(fixture)) return;
  if (!fixture.test.expected?.mean || !fixture.test.expected?.variance) {
    throw new Error(`Fixture "${name}": missing test.expected.mean or test.expected.variance`);
  }

  // Sanity check: expected values must have meaningful signal.
  // A fixture where all means ≈ 0 is vacuous — any constant-zero predictor would pass.
  const TRIVIAL_THRESHOLD = 1e-5;
  const means = fixture.test.expected.mean;
  const flatMeans: number[] = Array.isArray(means)
    ? (Array.isArray(means[0]) ? (means as number[][]).flat() : (means as number[]))
    : Object.values(means as Record<string, number[]>).flat();
  const maxAbsMean = Math.max(...flatMeans.map(Math.abs));
  if (maxAbsMean < TRIVIAL_THRESHOLD) {
    throw new Error(
      `Fixture "${name}": all expected means are near-zero (max |mean| = ${maxAbsMean}). ` +
      `This makes the parity test vacuous — any constant predictor would pass.`,
    );
  }
}

interface DiffStats {
  mean: number;
  max: number;
  n: number;
}

function computeDiffStats(
  actual: Float64Array | number[],
  expected: number[],
): DiffStats {
  let sum = 0;
  let max = 0;
  for (let i = 0; i < expected.length; i++) {
    // Relative diff: |a-b| / (atol + rtol*|expected|), matches numpy allclose semantics
    const absDiff = Math.abs(actual[i] - expected[i]);
    const scale = TOLERANCE + TOLERANCE * Math.abs(expected[i]);
    const relDiff = absDiff / scale;
    sum += relDiff;
    if (relDiff > max) max = relDiff;
  }
  return { mean: sum / expected.length, max, n: expected.length };
}

function expectAllClose(
  actual: Float64Array | number[],
  expected: number[],
  atol: number,
  label: string,
) {
  expect(actual.length).toBe(expected.length);
  for (let i = 0; i < expected.length; i++) {
    const absDiff = Math.abs(actual[i] - expected[i]);
    // numpy allclose: |a-b| <= atol + rtol*|b|, with rtol=atol for uniform relative precision
    const tol = atol + atol * Math.abs(expected[i]);
    if (absDiff > tol) {
      throw new Error(
        `${label}[${i}]: expected ${expected[i]}, got ${actual[i]}, diff=${absDiff} > tol=${tol}`,
      );
    }
  }
}

// Collect report card data across all fixtures
const reportCard: {
  name: string;
  model: string;
  nTest: number;
  meanDiff: DiffStats;
  varDiff: DiffStats;
}[] = [];

// Load manifest — failure is a hard error, not a silent skip
let manifest: Manifest;
try {
  manifest = JSON.parse(
    readFileSync(join(fixturesDir, "manifest.json"), "utf-8"),
  );
} catch (e) {
  throw new Error(
    `Failed to load manifest.json: ${e}. Run: python python/generate_fixtures.py`,
  );
}

if (manifest.fixtures.length === 0) {
  throw new Error(
    "No fixtures found — run: python python/generate_fixtures.py",
  );
}

// Count BoTorch-level parity fixtures (exclude consistency-only and ax-level)
const parityFixtures = manifest.fixtures.filter((entry) => {
  const fp = join(fixturesDir, entry.file);
  if (!existsSync(fp)) return true; // will fail below
  const f = JSON.parse(readFileSync(fp, "utf-8"));
  return f.test.expected.mean !== null && !f.test.metadata?.ax_level;
});

if (parityFixtures.length < MINIMUM_FIXTURES) {
  throw new Error(
    `Expected >= ${MINIMUM_FIXTURES} parity fixtures, got ${parityFixtures.length}. ` +
      `Regenerate with: python python/generate_fixtures.py`,
  );
}

// Verify all referenced fixture files exist
for (const entry of manifest.fixtures) {
  const fp = join(fixturesDir, entry.file);
  if (!existsSync(fp)) {
    throw new Error(`Fixture file missing: ${entry.file}`);
  }
}

function formatSci(x: number): string {
  return x.toExponential(2);
}

/**
 * Dispatch prediction and collect report card entries.
 * Handles all model types: MultiTaskGP (needs task_index), ModelListGP (multi-output),
 * and single-output models (SingleTaskGP, PairwiseGP, EnsembleGP).
 */
function predictAndReport(
  model: ReturnType<typeof loadModel>,
  fixture: FixtureData,
  entryName: string,
): { results: { mean: Float64Array | number[]; variance: Float64Array | number[]; expectedMean: number[]; expectedVar: number[]; label: string }[] } {
  if (model instanceof ModelListGP) {
    const results = model.predict(fixture.test.test_points);
    const expectedMeans = fixture.test.expected.mean as number[][];
    const expectedVars = fixture.test.expected.variance as number[][];
    const out: { mean: Float64Array | number[]; variance: Float64Array | number[]; expectedMean: number[]; expectedVar: number[]; label: string }[] = [];
    for (let k = 0; k < results.length; k++) {
      reportCard.push({
        name: `${entryName}[${k}]`,
        model: "ModelListGP",
        nTest: expectedMeans[k].length,
        meanDiff: computeDiffStats(results[k].mean, expectedMeans[k]),
        varDiff: computeDiffStats(results[k].variance, expectedVars[k]),
      });
      out.push({
        mean: results[k].mean,
        variance: results[k].variance,
        expectedMean: expectedMeans[k],
        expectedVar: expectedVars[k],
        label: `outcome[${k}]`,
      });
    }
    return { results: out };
  }

  // Single-output models
  let result: PredictionResult;
  let modelName: string;
  if (model instanceof MultiTaskGP) {
    const taskIndex = fixture.test.metadata.task_index ?? 0;
    result = model.predict(fixture.test.test_points, taskIndex);
    modelName = "MultiTaskGP";
  } else if (model instanceof EnsembleGP) {
    result = model.predict(fixture.test.test_points);
    modelName = "EnsembleGP";
  } else {
    result = (model as SingleTaskGP | PairwiseGP).predict(fixture.test.test_points);
    modelName = fixture.experiment.model_state.model_type;
  }

  const expectedMean = fixture.test.expected.mean as number[];
  const expectedVar = fixture.test.expected.variance as number[];
  reportCard.push({
    name: entryName,
    model: modelName,
    nTest: expectedMean.length,
    meanDiff: computeDiffStats(result.mean, expectedMean),
    varDiff: computeDiffStats(result.variance, expectedVar),
  });

  return {
    results: [{
      mean: result.mean,
      variance: result.variance,
      expectedMean,
      expectedVar,
      label: "",
    }],
  };
}

describe("BoTorch parity", () => {
  for (const entry of manifest.fixtures) {
    const fixture = loadFixture(entry.file);
    validateFixture(fixture, entry.name);

    // Skip consistency-only fixtures (Bilog/Power) — tested separately
    if (isConsistencyOnly(fixture)) continue;

    // Skip ax-level fixtures — tested in predictor_parity.test.ts
    if (isAxLevel(fixture)) continue;

    describe(`fixture: ${entry.name}`, () => {
      const model = loadModel(fixture.experiment.model_state);

      const { results } = predictAndReport(model, fixture, entry.name);

      for (const r of results) {
        const prefix = r.label ? `${r.label} ` : "";
        it(`${prefix}mean matches BoTorch`, () => {
          expectAllClose(r.mean, r.expectedMean, TOLERANCE, `${prefix}mean`);
        });
        it(`${prefix}variance matches BoTorch`, () => {
          expectAllClose(r.variance, r.expectedVar, TOLERANCE, `${prefix}variance`);
        });
      }
    });
  }

  // Consistency-only fixtures (Bilog/Power): verify model loads and produces
  // finite predictions with positive variance (no BoTorch reference available)
  const consistencyFixtures = manifest.fixtures.filter((entry) => {
    const f = loadFixture(entry.file);
    return isConsistencyOnly(f);
  });

  for (const entry of consistencyFixtures) {
    describe(`consistency: ${entry.name}`, () => {
      const fixture = loadFixture(entry.file);

      it("model loads without error", () => {
        const model = loadModel(fixture.experiment.model_state);
        expect(model).toBeTruthy();
      });

      it("predictions are finite with positive variance", () => {
        const model = loadModel(fixture.experiment.model_state) as SingleTaskGP;
        const result = model.predict(fixture.test.test_points);
        for (let i = 0; i < fixture.test.test_points.length; i++) {
          expect(Number.isFinite(result.mean[i])).toBe(true);
          expect(Number.isFinite(result.variance[i])).toBe(true);
          expect(result.variance[i]).toBeGreaterThan(0);
        }
      });

      it("predictions are deterministic (repeated predict gives same result)", () => {
        const model = loadModel(fixture.experiment.model_state) as SingleTaskGP;
        const r1 = model.predict(fixture.test.test_points);
        const r2 = model.predict(fixture.test.test_points);
        for (let i = 0; i < fixture.test.test_points.length; i++) {
          expect(r1.mean[i]).toBe(r2.mean[i]);
          expect(r1.variance[i]).toBe(r2.variance[i]);
        }
      });
    });
  }

  // Print report card after all fixtures
  describe("report card", () => {
    it("prints discrepancy summary", () => {
      const header =
        "Fixture".padEnd(30) +
        "Model".padEnd(16) +
        "N".padStart(4) +
        "  Mean(avg)".padEnd(14) +
        "Mean(worst)".padEnd(14) +
        "Var(avg)".padEnd(14) +
        "Var(worst)".padEnd(14);
      const sep = "-".repeat(header.length);

      console.log("\n" + sep);
      console.log("axjs vs BoTorch Parity Report Card (Float64)");
      console.log(sep);
      console.log(header);
      console.log(sep);

      for (const r of reportCard) {
        console.log(
          r.name.padEnd(30) +
            r.model.padEnd(16) +
            String(r.nTest).padStart(4) +
            ("  " + formatSci(r.meanDiff.mean)).padEnd(14) +
            formatSci(r.meanDiff.max).padEnd(14) +
            formatSci(r.varDiff.mean).padEnd(14) +
            formatSci(r.varDiff.max).padEnd(14),
        );
      }

      console.log(sep);

      const worstMean = Math.max(...reportCard.map((r) => r.meanDiff.max));
      const worstVar = Math.max(...reportCard.map((r) => r.varDiff.max));
      console.log(
        `Worst-case: mean=${formatSci(worstMean)}, variance=${formatSci(worstVar)}`,
      );
      console.log(
        `Tolerance: ${formatSci(TOLERANCE)} (global)`,
      );
      console.log(
        "Diffs normalized: |a-b| / (atol + rtol*|b|) with atol=rtol=1e-6; values \u2264 1.0 pass",
      );
      console.log(sep + "\n");

      // DiffStats are now normalized: value <= 1.0 means within tolerance
      expect(worstMean).toBeLessThanOrEqual(1.0);
      expect(worstVar).toBeLessThanOrEqual(1.0);
    });
  });
});
