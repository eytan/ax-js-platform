// Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

import { describe, it, expect } from "vitest";

import { cholesky } from "../../src/linalg/cholesky.js";
import { Matrix } from "../../src/linalg/matrix.js";
import {
  forwardSolve,
  forwardSolveTransposed,
  backSolve,
  solveCholesky,
} from "../../src/linalg/solve.js";

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

  it("forwardSolveTransposed solves L X = B^T without allocating transpose", () => {
    // B is m×n, we want to solve L X = B^T (which is n×m)
    const B = Matrix.from2D([
      [1, 2],
      [3, 4],
      [5, 6],
    ]); // 3×2

    // Expected: solve L X = B^T manually
    const BT = new Matrix(2, 3);
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 2; j++) {
        BT.set(j, i, B.get(i, j));
      }
    }
    const expected = forwardSolve(L, BT);

    // Test optimized version
    const result = forwardSolveTransposed(L, B);

    // Verify dimensions
    expect(result.rows).toBe(L.rows);
    expect(result.cols).toBe(B.rows);

    // Verify values match
    for (let i = 0; i < result.rows; i++) {
      for (let j = 0; j < result.cols; j++) {
        expect(result.get(i, j)).toBeCloseTo(expected.get(i, j), 10);
      }
    }
  });

  it("forwardSolveTransposed matches forwardSolve on single row", () => {
    // B is 1×2, B^T is 2×1
    const B = Matrix.from2D([[1, 2]]); // 1×2
    const BT = Matrix.from2D([[1], [2]]); // 2×1 (transpose)

    const expected = forwardSolve(L, BT);
    const result = forwardSolveTransposed(L, B);

    expect(result.rows).toBe(expected.rows);
    expect(result.cols).toBe(expected.cols);

    for (let i = 0; i < result.rows; i++) {
      for (let j = 0; j < result.cols; j++) {
        expect(result.get(i, j)).toBeCloseTo(expected.get(i, j), 10);
      }
    }
  });
});
