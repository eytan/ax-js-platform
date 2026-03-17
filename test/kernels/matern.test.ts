import { describe, it, expect } from "vitest";
import { Matrix } from "../../src/linalg/matrix.js";
import { MaternKernel } from "../../src/kernels/matern.js";

describe("MaternKernel", () => {
  it("self-covariance is 1 for nu=2.5", () => {
    const k = new MaternKernel([1.0, 1.0], 2.5);
    const x = Matrix.from2D([[0.5, 0.5]]);
    const K = k.compute(x, x);
    expect(K.get(0, 0)).toBeCloseTo(1, 10);
  });

  it("self-covariance is 1 for nu=1.5", () => {
    const k = new MaternKernel([1.0], 1.5);
    const x = Matrix.from2D([[0.5]]);
    const K = k.compute(x, x);
    expect(K.get(0, 0)).toBeCloseTo(1, 10);
  });

  it("nu=0.5 is exponential kernel: exp(-r)", () => {
    const ls = 0.5;
    const k = new MaternKernel([ls], 0.5);
    const x1 = Matrix.from2D([[0]]);
    const x2 = Matrix.from2D([[0.3]]);
    const expected = Math.exp(-0.3 / ls);
    expect(k.compute(x1, x2).get(0, 0)).toBeCloseTo(expected, 10);
  });

  it("covariance decreases with distance", () => {
    const k = new MaternKernel([1.0], 2.5);
    const x1 = Matrix.from2D([[0]]);
    const x2Near = Matrix.from2D([[0.1]]);
    const x2Far = Matrix.from2D([[1.0]]);
    const kNear = k.compute(x1, x2Near).get(0, 0);
    const kFar = k.compute(x1, x2Far).get(0, 0);
    expect(kNear).toBeGreaterThan(kFar);
  });

  it("shorter lengthscale means faster decay", () => {
    const kShort = new MaternKernel([0.1], 2.5);
    const kLong = new MaternKernel([1.0], 2.5);
    const x1 = Matrix.from2D([[0]]);
    const x2 = Matrix.from2D([[0.5]]);
    expect(kLong.compute(x1, x2).get(0, 0)).toBeGreaterThan(
      kShort.compute(x1, x2).get(0, 0),
    );
  });

  it("is symmetric", () => {
    const k = new MaternKernel([0.5, 0.3], 2.5);
    const x1 = Matrix.from2D([[0.1, 0.2]]);
    const x2 = Matrix.from2D([[0.7, 0.9]]);
    expect(k.compute(x1, x2).get(0, 0)).toBeCloseTo(
      k.compute(x2, x1).get(0, 0),
      10,
    );
  });

  it("produces correct shape", () => {
    const k = new MaternKernel([1.0, 1.0], 2.5);
    const x1 = Matrix.from2D([
      [0, 0],
      [1, 1],
      [0.5, 0.5],
    ]);
    const x2 = Matrix.from2D([
      [0.2, 0.3],
      [0.8, 0.9],
    ]);
    const K = k.compute(x1, x2);
    expect(K.rows).toBe(3);
    expect(K.cols).toBe(2);
  });
});
