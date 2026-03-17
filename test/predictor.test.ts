import { describe, it, expect } from "vitest";
import { readFileSync } from "fs";
import { join, dirname } from "path";
import { fileURLToPath } from "url";
import { Predictor } from "../src/predictor.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
const fixturesDir = join(__dirname, "fixtures");

function loadFixture(name: string) {
  return JSON.parse(readFileSync(join(fixturesDir, name), "utf-8"));
}

const TOLERANCE = 1e-6;

describe("Predictor", () => {
  describe("single-output model (branin)", () => {
    const fixture = loadFixture("branin_matern25.json");
    const predictor = new Predictor(fixture.experiment);

    it("exposes param names from search_space", () => {
      expect(predictor.paramNames.length).toBe(2);
      expect(predictor.paramNames).toEqual(
        fixture.experiment.search_space.parameters.map((p: any) => p.name),
      );
    });

    it("exposes param bounds from search_space", () => {
      expect(predictor.paramBounds.length).toBe(2);
      for (let i = 0; i < 2; i++) {
        const expected = fixture.experiment.search_space.parameters[i].bounds || [0, 1];
        expect(predictor.paramBounds[i]).toEqual(expected);
      }
    });

    it("defaults outcome name to 'y' for single-output", () => {
      expect(predictor.outcomeNames).toEqual(["y"]);
    });

    it("predict returns results keyed by outcome name", () => {
      const preds = predictor.predict(fixture.test.test_points);
      expect(preds).toHaveProperty("y");
      expect(preds["y"].mean.length).toBe(fixture.test.test_points.length);
      expect(preds["y"].variance.length).toBe(fixture.test.test_points.length);
    });

    it("predictions match expected values", () => {
      const preds = predictor.predict(fixture.test.test_points);
      for (let i = 0; i < fixture.test.expected.mean.length; i++) {
        expect(Math.abs(preds["y"].mean[i] - fixture.test.expected.mean[i])).toBeLessThan(TOLERANCE);
      }
    });
  });

  describe("multi-output model (ModelListGP)", () => {
    const fixture = loadFixture("vsip_modellist.json");
    const predictor = new Predictor(fixture.experiment);

    it("exposes outcome names from ModelListGP", () => {
      expect(predictor.outcomeNames.length).toBe(9);
      expect(predictor.outcomeNames[0]).toBe("weight");
    });

    it("predict returns one key per outcome", () => {
      const preds = predictor.predict(fixture.test.test_points);
      for (const name of predictor.outcomeNames) {
        expect(preds).toHaveProperty(name);
        expect(preds[name].mean.length).toBe(fixture.test.test_points.length);
      }
    });

    it("predictions match expected values for all outcomes", () => {
      const preds = predictor.predict(fixture.test.test_points);
      const expectedMeans = fixture.test.expected.mean as number[][];
      for (let k = 0; k < predictor.outcomeNames.length; k++) {
        const name = predictor.outcomeNames[k];
        for (let i = 0; i < expectedMeans[k].length; i++) {
          expect(Math.abs(preds[name].mean[i] - expectedMeans[k][i])).toBeLessThan(TOLERANCE);
        }
      }
    });
  });

  describe("custom outcome_names override", () => {
    const fixture = loadFixture("branin_matern25.json");
    const predictor = new Predictor({
      ...fixture.experiment,
      outcome_names: ["branin_value"],
    });

    it("uses provided outcome_names", () => {
      expect(predictor.outcomeNames).toEqual(["branin_value"]);
    });

    it("predict keys results by custom name", () => {
      const preds = predictor.predict(fixture.test.test_points);
      expect(preds).toHaveProperty("branin_value");
      expect(preds).not.toHaveProperty("y");
    });
  });

  describe("adapter transforms", () => {
    // Use branin_matern25 as base — its predictions are in the raw GP space
    const fixture = loadFixture("branin_matern25.json");

    it("no adapter transforms: passthrough (same as raw model)", () => {
      const predictor = new Predictor(fixture.experiment);
      const preds = predictor.predict(fixture.test.test_points);
      for (let i = 0; i < fixture.test.expected.mean.length; i++) {
        expect(Math.abs(preds["y"].mean[i] - fixture.test.expected.mean[i])).toBeLessThan(TOLERANCE);
      }
    });

    it("LogY adapter transform applies exp(mu + var/2) mean", () => {
      const predictor = new Predictor({
        ...fixture.experiment,
        adapter_transforms: [{ type: "LogY" as const }],
      });
      const rawPredictor = new Predictor(fixture.experiment);
      const rawPreds = rawPredictor.predict(fixture.test.test_points);
      const preds = predictor.predict(fixture.test.test_points);

      for (let i = 0; i < fixture.test.test_points.length; i++) {
        const mu = rawPreds["y"].mean[i];
        const v = rawPreds["y"].variance[i];
        const expectedMean = Math.exp(mu + v / 2);
        expect(preds["y"].mean[i]).toBeCloseTo(expectedMean, 8);
      }
    });

    it("StandardizeY adapter transform reverses standardization", () => {
      const Ymean = 10.0;
      const Ystd = 2.5;
      const predictor = new Predictor({
        ...fixture.experiment,
        adapter_transforms: [{
          type: "StandardizeY" as const,
          Ymean: { y: Ymean },
          Ystd: { y: Ystd },
        }],
      });
      const rawPredictor = new Predictor(fixture.experiment);
      const rawPreds = rawPredictor.predict(fixture.test.test_points);
      const preds = predictor.predict(fixture.test.test_points);

      for (let i = 0; i < fixture.test.test_points.length; i++) {
        const expectedMean = Ymean + Ystd * rawPreds["y"].mean[i];
        const expectedVar = Ystd * Ystd * rawPreds["y"].variance[i];
        expect(preds["y"].mean[i]).toBeCloseTo(expectedMean, 8);
        expect(preds["y"].variance[i]).toBeCloseTo(expectedVar, 8);
      }
    });

    it("LogY with metrics only applies to matching outcomes", () => {
      const vsipFixture = loadFixture("vsip_modellist.json");
      // Apply LogY only to "weight" metric
      const predictor = new Predictor({
        ...vsipFixture.experiment,
        adapter_transforms: [{ type: "LogY" as const, metrics: ["weight"] }],
      });
      const rawPredictor = new Predictor(vsipFixture.experiment);
      const rawPreds = rawPredictor.predict(vsipFixture.test.test_points);
      const preds = predictor.predict(vsipFixture.test.test_points);

      // "weight" should be log-transformed
      const mu0 = rawPreds["weight"].mean[0];
      const v0 = rawPreds["weight"].variance[0];
      expect(preds["weight"].mean[0]).toBeCloseTo(Math.exp(mu0 + v0 / 2), 8);

      // "acceleration" should be unchanged
      expect(preds["acceleration"].mean[0]).toBeCloseTo(rawPreds["acceleration"].mean[0], 10);
    });
  });
});
