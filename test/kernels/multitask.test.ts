import { describe, it, expect } from "vitest";
import { Matrix } from "../../src/linalg/matrix.js";
import { RBFKernel } from "../../src/kernels/rbf.js";
import { IndexKernel, MultitaskKernel } from "../../src/kernels/multitask.js";

describe("IndexKernel", () => {
  it("computes task covariance B = W@W^T + diag(exp(log_var))", () => {
    // W = [[1, 0], [0, 1]], log_var = [0, 0]
    // B = I + I = 2*I
    const idx = new IndexKernel([[1, 0], [0, 1]], [0, 0]);
    const x1 = Matrix.from2D([[0], [1]]);
    const K = idx.compute(x1, x1);
    expect(K.get(0, 0)).toBeCloseTo(2.0, 10); // B[0,0] = 1 + 1
    expect(K.get(1, 1)).toBeCloseTo(2.0, 10); // B[1,1] = 1 + 1
    expect(K.get(0, 1)).toBeCloseTo(0.0, 10); // B[0,1] = 0
  });

  it("handles rank-1 covariance", () => {
    // W = [[1], [2]], log_var = [0, 0]
    // B = [[1,2],[2,4]] + [[1,0],[0,1]] = [[2,2],[2,5]]
    const idx = new IndexKernel([[1], [2]], [0, 0]);
    const x1 = Matrix.from2D([[0]]);
    const x2 = Matrix.from2D([[1]]);
    const K = idx.compute(x1, x2);
    expect(K.get(0, 0)).toBeCloseTo(2.0, 10); // B[0,1] = 1*2 = 2
  });
});

describe("MultitaskKernel", () => {
  it("computes K_data * B element-wise", () => {
    const rbf = new RBFKernel([1.0]);
    // W = [[1]], log_var = [0, 0] → B = [[2, 1], [1, 2]]
    const mt = new MultitaskKernel(rbf, [[1], [1]], [0, 0], -1);

    // x1 = (feature=0.5, task=0), x2 = (feature=0.5, task=1)
    const x1 = Matrix.from2D([[0.5, 0]]);
    const x2 = Matrix.from2D([[0.5, 1]]);
    const K = mt.compute(x1, x2);

    // Data kernel: k(0.5, 0.5) = 1 (same point)
    // Task covar: B[0, 1] = 1
    expect(K.get(0, 0)).toBeCloseTo(1.0, 5);
  });

  it("computeDiag gives correct diagonal", () => {
    const rbf = new RBFKernel([1.0]);
    const mt = new MultitaskKernel(rbf, [[1], [0.5]], [0, 0], -1);

    const x = Matrix.from2D([
      [0.1, 0],
      [0.5, 1],
      [0.9, 0],
    ]);

    const diag = mt.computeDiag(x);
    const K = mt.compute(x, x);

    for (let i = 0; i < x.rows; i++) {
      expect(diag[i]).toBeCloseTo(K.get(i, i), 10);
    }
  });
});
