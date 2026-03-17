import { describe, it, expect } from "vitest";
import { Matrix } from "../../src/linalg/matrix.js";
import { RBFKernel } from "../../src/kernels/rbf.js";
import { MaternKernel } from "../../src/kernels/matern.js";
import { ScaleKernel } from "../../src/kernels/scale.js";
import { CategoricalKernel } from "../../src/kernels/categorical.js";
import {
  ActiveDimsKernel,
  AdditiveKernel,
  ProductKernel,
  kernelDiag,
} from "../../src/kernels/composite.js";

describe("ActiveDimsKernel", () => {
  it("selects correct dimensions", () => {
    const rbf = new RBFKernel([1.0]);
    const wrapped = new ActiveDimsKernel(rbf, [1]); // only use dim 1

    const x1 = Matrix.from2D([[0.1, 0.5], [0.9, 0.2]]);
    const x2 = Matrix.from2D([[0.3, 0.5]]);

    const K = wrapped.compute(x1, x2);
    // x1[0, dim1]=0.5 vs x2[0, dim1]=0.5: distance=0, so k=1
    expect(K.get(0, 0)).toBeCloseTo(1.0, 5);
    // x1[1, dim1]=0.2 vs x2[0, dim1]=0.5: non-zero distance
    expect(K.get(1, 0)).toBeLessThan(1.0);
  });
});

describe("AdditiveKernel", () => {
  it("sums kernel matrices", () => {
    const k1 = new RBFKernel([1.0]);
    const k2 = new RBFKernel([2.0]);
    const add = new AdditiveKernel([k1, k2]);

    const x = Matrix.from2D([[0.0], [1.0]]);
    const K = add.compute(x, x);
    const K1 = k1.compute(x, x);
    const K2 = k2.compute(x, x);

    for (let i = 0; i < K.data.length; i++) {
      expect(K.data[i]).toBeCloseTo(K1.data[i] + K2.data[i], 10);
    }
  });

  it("diagonal matches full matrix diagonal", () => {
    const k1 = new RBFKernel([1.0]);
    const k2 = new RBFKernel([2.0]);
    const add = new AdditiveKernel([k1, k2]);

    const x = Matrix.from2D([[0.1], [0.5], [0.9]]);
    const diag = add.computeDiag(x);
    const K = add.compute(x, x);

    for (let i = 0; i < x.rows; i++) {
      expect(diag[i]).toBeCloseTo(K.get(i, i), 10);
    }
  });
});

describe("ProductKernel", () => {
  it("multiplies kernel matrices element-wise", () => {
    const k1 = new RBFKernel([1.0]);
    const k2 = new RBFKernel([2.0]);
    const prod = new ProductKernel([k1, k2]);

    const x = Matrix.from2D([[0.0], [1.0]]);
    const K = prod.compute(x, x);
    const K1 = k1.compute(x, x);
    const K2 = k2.compute(x, x);

    for (let i = 0; i < K.data.length; i++) {
      expect(K.data[i]).toBeCloseTo(K1.data[i] * K2.data[i], 10);
    }
  });

  it("works with mixed continuous + categorical", () => {
    const rbf = new ActiveDimsKernel(new RBFKernel([1.0]), [0]);
    const cat = new ActiveDimsKernel(new CategoricalKernel(1.0), [1]);
    const prod = new ProductKernel([rbf, cat]);

    // x = [[0.5, 0], [0.5, 1], [0.5, 0]]
    const x = Matrix.from2D([
      [0.5, 0],
      [0.5, 1],
      [0.5, 0],
    ]);
    const K = prod.compute(x, x);

    // Points 0 and 2 are identical → k=1
    expect(K.get(0, 2)).toBeCloseTo(1.0, 5);
    // Points 0 and 1 have same continuous but different category
    // → rbf part = 1, cat part = exp(-1) ≈ 0.368
    expect(K.get(0, 1)).toBeCloseTo(Math.exp(-1), 5);
  });
});

describe("CategoricalKernel", () => {
  it("gives 1 for same category", () => {
    const cat = new CategoricalKernel(1.0);
    const x = Matrix.from2D([[0], [0]]);
    const K = cat.compute(x, x);
    expect(K.get(0, 0)).toBe(1);
    expect(K.get(0, 1)).toBe(1);
  });

  it("gives exp(-1/ls / d) for different categories in 1D", () => {
    const ls = 2.0;
    const cat = new CategoricalKernel(ls);
    const x1 = Matrix.from2D([[0]]);
    const x2 = Matrix.from2D([[1]]);
    const K = cat.compute(x1, x2);
    // 1D: mean(delta/ls) = (1/2) / 1 = 0.5
    expect(K.get(0, 0)).toBeCloseTo(Math.exp(-1 / ls), 10);
  });

  it("handles multi-dimensional categoricals", () => {
    const cat = new CategoricalKernel(1.0);
    const x1 = Matrix.from2D([[0, 1]]);
    const x2 = Matrix.from2D([[0, 2]]); // 1 of 2 dims differ
    const K = cat.compute(x1, x2);
    // 2D, ls=1.0: mean(delta/ls) = (0 + 1/1) / 2 = 0.5
    expect(K.get(0, 0)).toBeCloseTo(Math.exp(-0.5), 10);
  });
});

describe("kernelDiag", () => {
  it("matches diagonal of full matrix for ScaleKernel", () => {
    const kernel = new ScaleKernel(new MaternKernel([1.0, 2.0]), 3.5);
    const x = Matrix.from2D([
      [0.1, 0.2],
      [0.5, 0.8],
      [0.9, 0.1],
    ]);
    const diag = kernelDiag(kernel, x);
    const K = kernel.compute(x, x);
    for (let i = 0; i < x.rows; i++) {
      expect(diag[i]).toBeCloseTo(K.get(i, i), 10);
    }
  });
});
