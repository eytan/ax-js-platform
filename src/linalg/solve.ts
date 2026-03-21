// Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

import { Matrix } from "./matrix.js";

/** Solve Lx = b where L is lower triangular. Returns x as column vector. */
export function forwardSolve(L: Matrix, b: Matrix): Matrix {
  const n = L.rows;
  const x = Matrix.zeros(n, b.cols);

  for (let col = 0; col < b.cols; col++) {
    for (let i = 0; i < n; i++) {
      let s = b.get(i, col);
      for (let j = 0; j < i; j++) {
        s -= L.get(i, j) * x.get(j, col);
      }
      x.set(i, col, s / L.get(i, i));
    }
  }
  return x;
}

/** Solve L^T x = b where L is lower triangular. Returns x as column vector. */
export function backSolve(L: Matrix, b: Matrix): Matrix {
  const n = L.rows;
  const x = Matrix.zeros(n, b.cols);

  for (let col = 0; col < b.cols; col++) {
    for (let i = n - 1; i >= 0; i--) {
      let s = b.get(i, col);
      for (let j = i + 1; j < n; j++) {
        s -= L.get(j, i) * x.get(j, col);
      }
      x.set(i, col, s / L.get(i, i));
    }
  }
  return x;
}

/** Solve (LL^T)x = b using forward then back substitution. */
export function solveCholesky(L: Matrix, b: Matrix): Matrix {
  return backSolve(L, forwardSolve(L, b));
}
