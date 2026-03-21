// Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

import { describe, it, expect } from "vitest";

import { lu, solveLU } from "../../src/linalg/lu.js";
import { Matrix } from "../../src/linalg/matrix.js";
import { matmul } from "../../src/linalg/ops.js";

describe("LU decomposition", () => {
  it("decomposes a 3×3 matrix such that PA = LU", () => {
    const A = Matrix.from2D([
      [2, 3, 1],
      [6, 13, 5],
      [2, 19, 10],
    ]);
    const { L, U, perm } = lu(A);

    // Reconstruct PA from L * U
    const LU = matmul(L, U);

    // Reconstruct PA by permuting rows of A
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        expect(LU.get(i, j)).toBeCloseTo(A.get(perm[i], j), 12);
      }
    }
  });

  it("L is unit lower triangular", () => {
    const A = Matrix.from2D([
      [4, 3],
      [6, 3],
    ]);
    const { L } = lu(A);
    // Diagonal should be 1
    expect(L.get(0, 0)).toBe(1);
    expect(L.get(1, 1)).toBe(1);
    // Upper triangle should be 0
    expect(L.get(0, 1)).toBe(0);
  });

  it("U is upper triangular", () => {
    const A = Matrix.from2D([
      [4, 3],
      [6, 3],
    ]);
    const { U } = lu(A);
    expect(U.get(1, 0)).toBe(0);
  });
});

describe("solveLU", () => {
  it("solves Ax = b for a non-symmetric matrix", () => {
    const A = Matrix.from2D([
      [1, 2, 3],
      [4, 5, 6],
      [7, 8, 10],
    ]);
    const b = Matrix.from2D([[1], [2], [3]]);
    const x = solveLU(A, b);
    const Ax = matmul(A, x);
    for (let i = 0; i < 3; i++) {
      expect(Ax.get(i, 0)).toBeCloseTo(b.get(i, 0), 10);
    }
  });

  it("solves multiple right-hand sides", () => {
    const A = Matrix.from2D([
      [2, 1],
      [5, 3],
    ]);
    const B = Matrix.from2D([
      [1, 4],
      [2, 7],
    ]);
    const X = solveLU(A, B);
    const AX = matmul(A, X);
    for (let i = 0; i < 2; i++) {
      for (let j = 0; j < 2; j++) {
        expect(AX.get(i, j)).toBeCloseTo(B.get(i, j), 10);
      }
    }
  });

  it("handles identity matrix", () => {
    const I = Matrix.eye(3);
    const b = Matrix.from2D([[3], [7], [11]]);
    const x = solveLU(I, b);
    for (let i = 0; i < 3; i++) {
      expect(x.get(i, 0)).toBeCloseTo(b.get(i, 0), 14);
    }
  });
});
