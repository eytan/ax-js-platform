/**
 * Integration tests for relativization with GP models.
 *
 * Tests the full pipeline: load model → predict absolute → predict at status quo
 * → compute covariance → relativize → verify properties.
 *
 * Covers all model types: SingleTaskGP, FixedNoiseGP, MultiTaskGP, ModelListGP,
 * PairwiseGP, EnsembleGP — plus heteroscedastic and warped variants.
 */
import { describe, it, expect } from "vitest";
import { readFileSync } from "fs";
import { join, dirname } from "path";
import { fileURLToPath } from "url";
import { loadModel } from "../../src/io/deserialize.js";
import { SingleTaskGP } from "../../src/models/single_task.js";
import { MultiTaskGP } from "../../src/models/multi_task.js";
import { ModelListGP } from "../../src/models/model_list.js";
import { PairwiseGP } from "../../src/models/pairwise_gp.js";
import { EnsembleGP } from "../../src/models/ensemble_gp.js";
import {
  relativize,
  unrelativize,
  relativizePredictions,
} from "../../src/transforms/relativize.js";
import type { FixtureData, Manifest } from "../../src/models/types.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
const fixturesDir = join(__dirname, "..", "fixtures");

function loadFixture(filename: string): FixtureData {
  const raw = readFileSync(join(fixturesDir, filename), "utf-8");
  return JSON.parse(raw);
}

const manifest: Manifest = JSON.parse(
  readFileSync(join(fixturesDir, "manifest.json"), "utf-8"),
);

describe("posterior covariance numerics", () => {
  const entry = manifest.fixtures.find((f) => f.name === "branin_matern25")!;
  const fixture = loadFixture(entry.file);
  const model = loadModel(fixture.experiment.model_state) as SingleTaskGP;
  const testPoints = fixture.test.test_points;
  const testResult = model.predict(testPoints);

  it("Cov(f(x), f(x)) = Var(f(x)) for each test point", () => {
    // Self-covariance should equal variance — fundamental identity
    for (let i = 0; i < testPoints.length; i++) {
      const cov = model.predictCovarianceWith([testPoints[i]], testPoints[i]);
      expect(cov[0]).toBeCloseTo(testResult.variance[i], 8);
    }
  });

  it("covariance is symmetric: Cov(f(a), f(b)) = Cov(f(b), f(a))", () => {
    const a = testPoints[0];
    const b = testPoints[1];
    const covAB = model.predictCovarianceWith([a], b);
    const covBA = model.predictCovarianceWith([b], a);
    expect(covAB[0]).toBeCloseTo(covBA[0], 10);
  });

  it("|Cov| <= sqrt(Var_a * Var_b) (Cauchy-Schwarz)", () => {
    for (let i = 0; i < testPoints.length; i++) {
      for (let j = i + 1; j < Math.min(i + 3, testPoints.length); j++) {
        const cov = model.predictCovarianceWith([testPoints[i]], testPoints[j]);
        const bound = Math.sqrt(testResult.variance[i] * testResult.variance[j]);
        expect(Math.abs(cov[0])).toBeLessThanOrEqual(bound + 1e-10);
      }
    }
  });

  it("batch covariance matches per-point covariance", () => {
    const refPoint = testPoints[0];
    const covBatch = model.predictCovarianceWith(testPoints, refPoint);
    for (let i = 0; i < testPoints.length; i++) {
      const covSingle = model.predictCovarianceWith([testPoints[i]], refPoint);
      expect(covBatch[i]).toBeCloseTo(covSingle[0], 10);
    }
  });
});

describe("relativize with covariance (tighter CIs)", () => {
  const entry = manifest.fixtures.find((f) => f.name === "branin_matern25")!;
  const fixture = loadFixture(entry.file);
  const model = loadModel(fixture.experiment.model_state) as SingleTaskGP;
  const allPoints = fixture.test.test_points;

  const sqPoint = allPoints[0];
  const testPoints = allPoints.slice(1);
  const sqResult = model.predict([sqPoint]);
  const testResult = model.predict(testPoints);
  const covariances = model.predictCovarianceWith(testPoints, sqPoint);

  it("covariance reduces relative variance (points share training data)", () => {
    const noCov = relativizePredictions(
      testResult.mean,
      testResult.variance,
      sqResult.mean[0],
      sqResult.variance[0],
      { biasCorrection: false },
    );
    const withCov = relativizePredictions(
      testResult.mean,
      testResult.variance,
      sqResult.mean[0],
      sqResult.variance[0],
      { biasCorrection: false },
      covariances,
    );

    // With positive covariance (typical for GP), relative variance should be smaller
    let anySmaller = false;
    for (let i = 0; i < testPoints.length; i++) {
      if (covariances[i] > 0) {
        expect(withCov.variance[i]).toBeLessThanOrEqual(noCov.variance[i] + 1e-15);
        if (withCov.variance[i] < noCov.variance[i] - 1e-15) anySmaller = true;
      }
    }
    expect(anySmaller).toBe(true);
  });

  it("relative means are the same with or without covariance", () => {
    const noCov = relativizePredictions(
      testResult.mean,
      testResult.variance,
      sqResult.mean[0],
      sqResult.variance[0],
      { biasCorrection: false },
    );
    const withCov = relativizePredictions(
      testResult.mean,
      testResult.variance,
      sqResult.mean[0],
      sqResult.variance[0],
      { biasCorrection: false },
      covariances,
    );

    for (let i = 0; i < testPoints.length; i++) {
      expect(withCov.mean[i]).toBeCloseTo(noCov.mean[i], 10);
    }
  });
});

describe("relativize integration (internal consistency)", () => {
  // Noiseless: SingleTaskGP with learned noise
  describe("noiseless (SingleTaskGP: branin_matern25)", () => {
    const entry = manifest.fixtures.find((f) => f.name === "branin_matern25")!;
    const fixture = loadFixture(entry.file);
    const model = loadModel(fixture.experiment.model_state) as SingleTaskGP;
    const allPoints = fixture.test.test_points;

    const sqPoint = allPoints[0];
    const testPoints = allPoints.slice(1);
    const sqResult = model.predict([sqPoint]);
    const testResult = model.predict(testPoints);
    const sqMean = sqResult.mean[0];
    const sqVar = sqResult.variance[0];
    const sqSem = Math.sqrt(sqVar);

    it("status quo prediction matches expected", () => {
      const expectedMean = (fixture.test.expected.mean as number[])[0];
      expect(sqMean).toBeCloseTo(expectedMean, 5);
    });

    it("relative effect of status quo vs itself is (0, 0)", () => {
      const r = relativize(sqMean, sqSem, sqMean, sqSem);
      expect(r.mean).toBe(0);
      expect(r.sem).toBe(0);
    });

    it("batch relativize produces correct length", () => {
      const rel = relativizePredictions(
        testResult.mean,
        testResult.variance,
        sqMean,
        sqVar,
      );
      expect(rel.mean.length).toBe(testPoints.length);
      expect(rel.variance.length).toBe(testPoints.length);
    });

    it("relativize/unrelativize round-trips", () => {
      for (let i = 0; i < testPoints.length; i++) {
        const absMean = testResult.mean[i];
        const absSem = Math.sqrt(testResult.variance[i]);
        const rel = relativize(absMean, absSem, sqMean, sqSem, {
          biasCorrection: false,
        });
        const abs = unrelativize(rel.mean, rel.sem, sqMean, sqSem, {
          biasCorrection: false,
        });
        expect(abs.mean).toBeCloseTo(absMean, 8);
        expect(abs.sem).toBeCloseTo(absSem, 6);
      }
    });

    it("relative means have correct sign", () => {
      for (let i = 0; i < testPoints.length; i++) {
        const absMean = testResult.mean[i];
        const rel = relativize(absMean, 0, sqMean, 0, {
          biasCorrection: false,
        });
        if (sqMean > 0) {
          if (absMean > sqMean) expect(rel.mean).toBeGreaterThan(0);
          if (absMean < sqMean) expect(rel.mean).toBeLessThan(0);
        }
      }
    });

    it("controlAsConstant gives simpler variance", () => {
      for (let i = 0; i < Math.min(5, testPoints.length); i++) {
        const absMean = testResult.mean[i];
        const absSem = Math.sqrt(testResult.variance[i]);
        const rel = relativize(absMean, absSem, sqMean, sqSem, {
          controlAsConstant: true,
          biasCorrection: false,
        });
        const expectedSem = (absSem / Math.abs(sqMean)) * 100;
        expect(rel.sem).toBeCloseTo(expectedSem, 8);
      }
    });
  });

  // Noisy: FixedNoiseGP with known per-point noise
  describe("noisy (FixedNoiseGP: branin_fixed_noise)", () => {
    const entry = manifest.fixtures.find(
      (f) => f.name === "branin_fixed_noise",
    )!;
    const fixture = loadFixture(entry.file);
    const model = loadModel(fixture.experiment.model_state) as SingleTaskGP;
    const allPoints = fixture.test.test_points;

    const sqPoint = allPoints[Math.floor(allPoints.length / 2)];
    const testPoints = allPoints;
    const sqResult = model.predict([sqPoint]);
    const testResult = model.predict(testPoints);
    const sqMean = sqResult.mean[0];
    const sqVar = sqResult.variance[0];
    const sqSem = Math.sqrt(sqVar);

    it("status quo has nonzero variance (noisy model)", () => {
      expect(sqVar).toBeGreaterThan(0);
    });

    it("full delta method variance >= constant-control variance", () => {
      for (let i = 0; i < testPoints.length; i++) {
        const absMean = testResult.mean[i];
        const absSem = Math.sqrt(testResult.variance[i]);
        if (absMean === sqMean && absSem === sqSem) continue;

        const relFull = relativize(absMean, absSem, sqMean, sqSem, {
          biasCorrection: false,
          controlAsConstant: false,
        });
        const relConst = relativize(absMean, absSem, sqMean, sqSem, {
          biasCorrection: false,
          controlAsConstant: true,
        });
        expect(relFull.sem).toBeGreaterThanOrEqual(relConst.sem - 1e-10);
      }
    });

    it("relativize/unrelativize round-trips with bias correction", () => {
      for (let i = 0; i < testPoints.length; i++) {
        const absMean = testResult.mean[i];
        const absSem = Math.sqrt(testResult.variance[i]);
        if (absMean === sqMean && absSem === sqSem) continue;

        const rel = relativize(absMean, absSem, sqMean, sqSem, {
          biasCorrection: true,
        });
        const abs = unrelativize(rel.mean, rel.sem, sqMean, sqSem, {
          biasCorrection: true,
        });
        expect(abs.mean).toBeCloseTo(absMean, 5);
        expect(abs.sem).toBeCloseTo(absSem, 4);
      }
    });

    it("batch relativize matches per-point relativize", () => {
      const batch = relativizePredictions(
        testResult.mean,
        testResult.variance,
        sqMean,
        sqVar,
        { biasCorrection: false },
      );

      for (let i = 0; i < testPoints.length; i++) {
        const absSem = Math.sqrt(testResult.variance[i]);
        const perPoint = relativize(
          testResult.mean[i],
          absSem,
          sqMean,
          sqSem,
          { biasCorrection: false },
        );
        expect(batch.mean[i]).toBeCloseTo(perPoint.mean, 10);
        expect(batch.variance[i]).toBeCloseTo(
          perPoint.sem * perPoint.sem,
          10,
        );
      }
    });

    it("covariance with status quo is symmetric", () => {
      const covs = model.predictCovarianceWith(testPoints, sqPoint);
      for (let i = 0; i < testPoints.length; i++) {
        const covReverse = model.predictCovarianceWith([sqPoint], testPoints[i]);
        expect(covs[i]).toBeCloseTo(covReverse[0], 10);
      }
    });
  });

  // Heteroscedastic noise
  describe("heteroscedastic (FixedNoiseGP: branin_heteroscedastic)", () => {
    const entry = manifest.fixtures.find(
      (f) => f.name === "branin_heteroscedastic",
    )!;
    const fixture = loadFixture(entry.file);
    const model = loadModel(fixture.experiment.model_state) as SingleTaskGP;

    const sqPoint = fixture.test.test_points[0];
    const testPoints = fixture.test.test_points;
    const sqResult = model.predict([sqPoint]);
    const testResult = model.predict(testPoints);

    it("relativize works with heteroscedastic model", () => {
      const rel = relativizePredictions(
        testResult.mean,
        testResult.variance,
        sqResult.mean[0],
        sqResult.variance[0],
      );
      expect(rel.mean.length).toBe(testPoints.length);
      expect(rel.mean[0]).toBeCloseTo(0, 8);
    });

    it("covariance satisfies Cauchy-Schwarz", () => {
      const covs = model.predictCovarianceWith(testPoints, sqPoint);
      const sqVar = sqResult.variance[0];
      for (let i = 0; i < testPoints.length; i++) {
        const bound = Math.sqrt(testResult.variance[i] * sqVar);
        expect(Math.abs(covs[i])).toBeLessThanOrEqual(bound + 1e-10);
      }
    });
  });

  // Warped inputs: test that covariance works through the transform pipeline
  describe("warped (SingleTaskGP: branin_warp)", () => {
    const entry = manifest.fixtures.find((f) => f.name === "branin_warp")!;
    const fixture = loadFixture(entry.file);
    const model = loadModel(fixture.experiment.model_state) as SingleTaskGP;

    const sqPoint = fixture.test.test_points[0];
    const testPoints = fixture.test.test_points;
    const sqResult = model.predict([sqPoint]);
    const testResult = model.predict(testPoints);

    it("self-covariance equals variance after warp", () => {
      for (let i = 0; i < testPoints.length; i++) {
        const cov = model.predictCovarianceWith([testPoints[i]], testPoints[i]);
        expect(cov[0]).toBeCloseTo(testResult.variance[i], 8);
      }
    });

    it("relativize with covariance produces tighter CIs", () => {
      const covs = model.predictCovarianceWith(testPoints, sqPoint);
      const noCov = relativizePredictions(
        testResult.mean,
        testResult.variance,
        sqResult.mean[0],
        sqResult.variance[0],
        { biasCorrection: false },
      );
      const withCov = relativizePredictions(
        testResult.mean,
        testResult.variance,
        sqResult.mean[0],
        sqResult.variance[0],
        { biasCorrection: false },
        covs,
      );

      let anyTighter = false;
      for (let i = 0; i < testPoints.length; i++) {
        if (covs[i] > 0 && withCov.variance[i] < noCov.variance[i] - 1e-15) {
          anyTighter = true;
        }
      }
      expect(anyTighter).toBe(true);
    });
  });
});

/**
 * Helper: verify core covariance properties for any model that supports
 * predictCovarianceWith. Avoids repeating the same assertions per model type.
 */
function verifyCovarianceProperties(
  label: string,
  predict: (pts: number[][]) => { mean: Float64Array | number[]; variance: Float64Array | number[] },
  predictCov: (pts: number[][], ref: number[]) => Float64Array,
  testPoints: number[][],
) {
  describe(`${label}: covariance properties`, () => {
    const result = predict(testPoints);

    it("self-covariance equals variance", () => {
      for (let i = 0; i < testPoints.length; i++) {
        const cov = predictCov([testPoints[i]], testPoints[i]);
        expect(cov[0]).toBeCloseTo(result.variance[i] as number, 6);
      }
    });

    it("covariance is symmetric", () => {
      const a = testPoints[0];
      const b = testPoints[1];
      const covAB = predictCov([a], b);
      const covBA = predictCov([b], a);
      expect(covAB[0]).toBeCloseTo(covBA[0], 8);
    });

    it("|Cov| <= sqrt(Var_a * Var_b) (Cauchy-Schwarz)", () => {
      for (let i = 0; i < testPoints.length; i++) {
        for (let j = i + 1; j < Math.min(i + 3, testPoints.length); j++) {
          const cov = predictCov([testPoints[i]], testPoints[j]);
          const bound = Math.sqrt(
            (result.variance[i] as number) * (result.variance[j] as number),
          );
          expect(Math.abs(cov[0])).toBeLessThanOrEqual(bound + 1e-8);
        }
      }
    });

    it("batch matches per-point", () => {
      const refPoint = testPoints[0];
      const covBatch = predictCov(testPoints, refPoint);
      for (let i = 0; i < testPoints.length; i++) {
        const covSingle = predictCov([testPoints[i]], refPoint);
        expect(covBatch[i]).toBeCloseTo(covSingle[0], 8);
      }
    });

    it("covariance tightens relative variance", () => {
      const sqPoint = testPoints[0];
      const otherPoints = testPoints.slice(1);
      const sqResult = predict([sqPoint]);
      const testResult = predict(otherPoints);
      const covs = predictCov(otherPoints, sqPoint);

      const noCov = relativizePredictions(
        testResult.mean,
        testResult.variance,
        sqResult.mean[0],
        sqResult.variance[0] as number,
        { biasCorrection: false },
      );
      const withCov = relativizePredictions(
        testResult.mean,
        testResult.variance,
        sqResult.mean[0],
        sqResult.variance[0] as number,
        { biasCorrection: false },
        covs,
      );

      let anyTighter = false;
      for (let i = 0; i < otherPoints.length; i++) {
        if (covs[i] > 0 && withCov.variance[i] < noCov.variance[i] - 1e-15) {
          anyTighter = true;
        }
      }
      expect(anyTighter).toBe(true);
    });
  });
}

describe("covariance: MultiTaskGP", () => {
  const entry = manifest.fixtures.find((f) => f.name === "branin_multitask")!;
  const fixture = loadFixture(entry.file);
  const model = loadModel(fixture.experiment.model_state) as MultiTaskGP;
  const taskIndex = fixture.test.metadata.task_index ?? 0;

  verifyCovarianceProperties(
    "MultiTaskGP",
    (pts) => model.predict(pts, taskIndex),
    (pts, ref) => model.predictCovarianceWith(pts, taskIndex, ref),
    fixture.test.test_points,
  );
});

describe("covariance: ModelListGP", () => {
  const entry = manifest.fixtures.find(
    (f) => f.name === "branincurrin_modellist",
  )!;
  const fixture = loadFixture(entry.file);
  const model = loadModel(fixture.experiment.model_state) as ModelListGP;

  // Test each output independently
  for (let k = 0; k < model.outcomeNames.length; k++) {
    verifyCovarianceProperties(
      `ModelListGP[${k}]`,
      (pts) => model.predictOutcome(k, pts),
      (pts, ref) => model.predictCovarianceWith(pts, ref)[k],
      fixture.test.test_points,
    );
  }
});

describe("covariance: PairwiseGP", () => {
  const entry = manifest.fixtures.find((f) => f.name === "branin_pairwise")!;
  const fixture = loadFixture(entry.file);
  const model = loadModel(fixture.experiment.model_state) as PairwiseGP;

  verifyCovarianceProperties(
    "PairwiseGP",
    (pts) => model.predict(pts),
    (pts, ref) => model.predictCovarianceWith(pts, ref),
    fixture.test.test_points,
  );
});

describe("covariance: EnsembleGP (SAAS)", () => {
  const entry = manifest.fixtures.find(
    (f) => f.name === "saas_highdim_nuts",
  )!;
  const fixture = loadFixture(entry.file);
  const model = loadModel(fixture.experiment.model_state) as EnsembleGP;

  verifyCovarianceProperties(
    "EnsembleGP",
    (pts) => model.predict(pts),
    (pts, ref) => model.predictCovarianceWith(pts, ref),
    fixture.test.test_points,
  );
});

/**
 * Parity tests against Ax's relativize() (when fixtures include expected_relative).
 * These are only generated when fixture specs have status_quo="center".
 * The status quo point is the first entry in test_points, its predictions are at index 0.
 */
const relFixtures = manifest.fixtures.filter((entry) => {
  const f = loadFixture(entry.file);
  // Skip ax-level fixtures — tested in predictor_parity.test.ts with full Predictor pipeline
  if (f.test.metadata?.ax_level) return false;
  return f.experiment.status_quo && f.test.expected_relative;
});

describe.skipIf(relFixtures.length === 0)(
  "relativize parity with Ax",
  () => {
    for (const entry of relFixtures) {
      describe(`fixture: ${entry.name}`, () => {
        const fixture = loadFixture(entry.file);
        const model = loadModel(fixture.experiment.model_state);
        if (!(model instanceof SingleTaskGP)) return;

        // Status quo is the first test point; remaining are treatment points
        const allPoints = fixture.test.test_points;
        const sqPoint = allPoints[0];
        const testPoints = allPoints.slice(1);
        const allExpectedMean = fixture.test.expected.mean as number[];
        const allExpectedVar = fixture.test.expected.variance as number[];

        const testResult = model.predict(testPoints);
        const sqResult = model.predict([sqPoint]);

        it("status quo prediction matches expected", () => {
          expect(sqResult.mean[0]).toBeCloseTo(allExpectedMean[0], 5);
          expect(sqResult.variance[0]).toBeCloseTo(allExpectedVar[0], 5);
        });

        it("relative means match Ax", () => {
          const rel = relativizePredictions(
            testResult.mean,
            testResult.variance,
            sqResult.mean[0],
            sqResult.variance[0],
          );
          const expected = fixture.test.expected_relative!.mean;
          for (let i = 0; i < expected.length; i++) {
            expect(rel.mean[i]).toBeCloseTo(expected[i], 5);
          }
        });

        it("relative variances match Ax", () => {
          const rel = relativizePredictions(
            testResult.mean,
            testResult.variance,
            sqResult.mean[0],
            sqResult.variance[0],
          );
          const expected = fixture.test.expected_relative!.variance;
          for (let i = 0; i < expected.length; i++) {
            expect(rel.variance[i]).toBeCloseTo(expected[i], 5);
          }
        });
      });
    }
  },
);
