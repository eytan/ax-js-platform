import { describe, it, expect } from "vitest";
import { MultiTaskGP } from "../../src/models/multi_task.js";
import type { MultiTaskGPModelState } from "../../src/models/types.js";

function makeState(): MultiTaskGPModelState {
  return {
    model_type: "MultiTaskGP",
    train_X: [
      [0.1, 0.2, 0],
      [0.3, 0.4, 0],
      [0.5, 0.6, 0],
      [0.7, 0.8, 0],
      [0.2, 0.3, 1],
      [0.4, 0.5, 1],
      [0.6, 0.7, 1],
      [0.8, 0.9, 1],
    ],
    train_Y: [1.0, 1.5, 2.0, 2.5, 0.5, 1.0, 1.5, 2.0],
    task_feature: -1,
    num_tasks: 2,
    data_kernel: {
      type: "Matern",
      lengthscale: [0.5, 0.5],
      nu: 2.5,
      outputscale: 1.0,
    },
    task_covar: {
      covar_factor: [[0.7], [0.4]],
      log_var: [-1.0, -0.5],
    },
    mean_constant: 1.0,
    noise_variance: 0.01,
  };
}

describe("MultiTaskGP", () => {
  it("predicts correct shapes", () => {
    const model = new MultiTaskGP(makeState());
    const testPts = [
      [0.25, 0.35],
      [0.55, 0.65],
      [0.85, 0.95],
    ];
    const result = model.predict(testPts, 0);
    expect(result.mean.length).toBe(3);
    expect(result.variance.length).toBe(3);
  });

  it("produces positive variance", () => {
    const model = new MultiTaskGP(makeState());
    const testPts = [
      [0.1, 0.2],
      [0.5, 0.5],
      [0.9, 0.9],
    ];
    const result = model.predict(testPts, 0);
    for (let i = 0; i < result.variance.length; i++) {
      expect(result.variance[i]).toBeGreaterThanOrEqual(0);
    }
  });

  it("predictions differ between tasks", () => {
    const model = new MultiTaskGP(makeState());
    const testPts = [[0.5, 0.5]];
    const r0 = model.predict(testPts, 0);
    const r1 = model.predict(testPts, 1);
    // Different tasks should generally give different means
    expect(r0.mean[0]).not.toBeCloseTo(r1.mean[0], 1);
  });

  it("throws on empty test points", () => {
    const model = new MultiTaskGP(makeState());
    expect(() => model.predict([], 0)).toThrow("testPoints must not be empty");
  });

  it("throws on dimension mismatch", () => {
    const model = new MultiTaskGP(makeState());
    // Model has 2 data dims, passing 3
    expect(() => model.predict([[1, 2, 3]], 0)).toThrow("Dimension mismatch");
  });

  it("throws on empty train_X", () => {
    const state = makeState();
    state.train_X = [];
    expect(() => new MultiTaskGP(state)).toThrow("train_X must not be empty");
  });
});
