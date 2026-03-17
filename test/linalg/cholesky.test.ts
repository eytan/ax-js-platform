import { describe, it, expect } from "vitest";
import { Matrix } from "../../src/linalg/matrix.js";
import { cholesky } from "../../src/linalg/cholesky.js";
import { matmul, transpose } from "../../src/linalg/ops.js";

describe("cholesky", () => {
  it("decomposes 2x2 positive definite matrix", () => {
    const A = Matrix.from2D([
      [4, 2],
      [2, 5],
    ]);
    const L = cholesky(A);

    // L * L^T should equal A
    const LLT = matmul(L, transpose(L));
    for (let i = 0; i < 2; i++) {
      for (let j = 0; j < 2; j++) {
        expect(LLT.get(i, j)).toBeCloseTo(A.get(i, j), 10);
      }
    }
  });

  it("decomposes 3x3 matrix", () => {
    const A = Matrix.from2D([
      [25, 15, -5],
      [15, 18, 0],
      [-5, 0, 11],
    ]);
    const L = cholesky(A);
    const LLT = matmul(L, transpose(L));
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        expect(LLT.get(i, j)).toBeCloseTo(A.get(i, j), 10);
      }
    }
  });

  it("L is lower triangular", () => {
    const A = Matrix.from2D([
      [4, 2],
      [2, 5],
    ]);
    const L = cholesky(A);
    expect(L.get(0, 1)).toBe(0);
  });

  it("applies jitter for near-singular matrix", () => {
    // Singular matrix: should succeed with jitter
    const A = Matrix.from2D([
      [1, 1],
      [1, 1],
    ]);
    const L = cholesky(A);
    expect(L.get(0, 0)).toBeGreaterThan(0);
  });
});
