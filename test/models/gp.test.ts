import { describe, it, expect } from "vitest";
import { Matrix } from "../../src/linalg/matrix.js";
import { ExactGP } from "../../src/models/gp.js";
import { RBFKernel } from "../../src/kernels/rbf.js";
import { ScaleKernel } from "../../src/kernels/scale.js";
import { ConstantMean } from "../../src/means/constant.js";

describe("ExactGP", () => {
  it("predicts at training points with near-zero variance", () => {
    const trainX = Matrix.from2D([[0], [0.5], [1]]);
    const trainY = Matrix.vector([0, 1, 0]);
    const kernel = new ScaleKernel(new RBFKernel([0.3]), 1.0);
    const mean = new ConstantMean(0);

    const gp = new ExactGP(trainX, trainY, kernel, mean, 1e-6);
    const result = gp.predict(trainX);

    // Means should match training targets
    expect(result.mean[0]).toBeCloseTo(0, 3);
    expect(result.mean[1]).toBeCloseTo(1, 3);
    expect(result.mean[2]).toBeCloseTo(0, 3);

    // Variances should be near zero at training points
    for (let i = 0; i < 3; i++) {
      expect(result.variance[i]).toBeLessThan(1e-3);
    }
  });

  it("has higher variance far from training data", () => {
    const trainX = Matrix.from2D([[0], [1]]);
    const trainY = Matrix.vector([0, 1]);
    const kernel = new ScaleKernel(new RBFKernel([0.3]), 1.0);
    const mean = new ConstantMean(0);

    const gp = new ExactGP(trainX, trainY, kernel, mean, 1e-6);
    const nearResult = gp.predict(Matrix.from2D([[0.5]]));
    const farResult = gp.predict(Matrix.from2D([[5.0]]));

    expect(farResult.variance[0]).toBeGreaterThan(nearResult.variance[0]);
  });

  it("mean reverts to prior far from data", () => {
    const trainX = Matrix.from2D([[0]]);
    const trainY = Matrix.vector([5]);
    const kernel = new ScaleKernel(new RBFKernel([0.1]), 1.0);
    const mean = new ConstantMean(0);

    const gp = new ExactGP(trainX, trainY, kernel, mean, 1e-6);
    const farResult = gp.predict(Matrix.from2D([[100]]));

    // Far from training data, mean should revert toward prior (0)
    expect(Math.abs(farResult.mean[0])).toBeLessThan(0.1);
  });
});
