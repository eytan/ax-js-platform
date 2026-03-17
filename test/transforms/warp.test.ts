import { describe, it, expect } from "vitest";
import { InputWarp } from "../../src/transforms/warp.js";
import { Matrix } from "../../src/linalg/matrix.js";

describe("InputWarp (Kumaraswamy)", () => {
  it("identity when a=1, b=1", () => {
    const warp = new InputWarp([1], [1]);
    const X = Matrix.from2D([[0.2], [0.5], [0.8]]);
    const result = warp.forward(X);
    for (let i = 0; i < X.rows; i++) {
      expect(result.get(i, 0)).toBeCloseTo(X.get(i, 0), 5);
    }
  });

  it("maps [0,1] → [0,1]", () => {
    const warp = new InputWarp([2.0], [3.0]);
    const X = Matrix.from2D([[0.0], [0.25], [0.5], [0.75], [1.0]]);
    const result = warp.forward(X);
    for (let i = 0; i < X.rows; i++) {
      expect(result.get(i, 0)).toBeGreaterThanOrEqual(0);
      expect(result.get(i, 0)).toBeLessThanOrEqual(1);
    }
  });

  it("matches Kumaraswamy CDF formula with eps normalization", () => {
    const a = 2.0; // concentration1
    const b = 3.0; // concentration0
    const warp = new InputWarp([b], [a]);
    const x = 0.6;
    const X = Matrix.from2D([[x]]);
    const result = warp.forward(X);
    // BoTorch normalizes: xn = x * (1 - 2*eps) + eps, then CDF
    const eps = 1e-7;
    const xn = x * (1 - 2 * eps) + eps;
    const expected = 1 - Math.pow(1 - Math.pow(xn, a), b);
    expect(result.get(0, 0)).toBeCloseTo(expected, 10);
  });

  it("respects indices (partial warping)", () => {
    const warp = new InputWarp([2.0], [3.0], [1]); // warp only dim 1
    const X = Matrix.from2D([[0.3, 0.6], [0.7, 0.4]]);
    const result = warp.forward(X);
    // Dim 0 should be unchanged
    expect(result.get(0, 0)).toBe(0.3);
    expect(result.get(1, 0)).toBe(0.7);
    // Dim 1 should be warped
    expect(result.get(0, 1)).not.toBe(0.6);
    expect(result.get(1, 1)).not.toBe(0.4);
  });

  it("is monotonically increasing", () => {
    const warp = new InputWarp([1.5], [2.5]);
    const xs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
    const X = Matrix.from2D(xs.map((x) => [x]));
    const result = warp.forward(X);
    for (let i = 1; i < xs.length; i++) {
      expect(result.get(i, 0)).toBeGreaterThan(result.get(i - 1, 0));
    }
  });
});
