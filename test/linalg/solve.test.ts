import { describe, it, expect } from "vitest";
import { Matrix } from "../../src/linalg/matrix.js";
import { cholesky } from "../../src/linalg/cholesky.js";
import { forwardSolve, backSolve, solveCholesky } from "../../src/linalg/solve.js";

describe("triangular solvers", () => {
  const A = Matrix.from2D([
    [4, 2],
    [2, 5],
  ]);
  const L = cholesky(A);
  const b = Matrix.vector([6, 9]);

  it("forwardSolve solves Lx = b", () => {
    const x = forwardSolve(L, b);
    // Verify L * x ≈ b
    for (let i = 0; i < 2; i++) {
      let sum = 0;
      for (let j = 0; j <= i; j++) {
        sum += L.get(i, j) * x.get(j, 0);
      }
      expect(sum).toBeCloseTo(b.get(i, 0), 10);
    }
  });

  it("backSolve solves L^T x = b", () => {
    const x = backSolve(L, b);
    // Verify L^T * x ≈ b
    for (let i = 0; i < 2; i++) {
      let sum = 0;
      for (let j = i; j < 2; j++) {
        sum += L.get(j, i) * x.get(j, 0);
      }
      expect(sum).toBeCloseTo(b.get(i, 0), 10);
    }
  });

  it("solveCholesky solves Ax = b", () => {
    const x = solveCholesky(L, b);
    // Verify A * x ≈ b
    for (let i = 0; i < 2; i++) {
      let sum = 0;
      for (let j = 0; j < 2; j++) {
        sum += A.get(i, j) * x.get(j, 0);
      }
      expect(sum).toBeCloseTo(b.get(i, 0), 10);
    }
  });

  it("solves matrix RHS (multiple columns)", () => {
    const B = Matrix.from2D([
      [6, 1],
      [9, 2],
    ]);
    const X = solveCholesky(L, B);
    expect(X.cols).toBe(2);

    // Verify each column
    for (let col = 0; col < 2; col++) {
      for (let i = 0; i < 2; i++) {
        let sum = 0;
        for (let j = 0; j < 2; j++) {
          sum += A.get(i, j) * X.get(j, col);
        }
        expect(sum).toBeCloseTo(B.get(i, col), 10);
      }
    }
  });
});
