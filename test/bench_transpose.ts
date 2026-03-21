// Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

/**
 * Benchmark: forwardSolveTransposed vs transpose + forwardSolve
 *
 * Measures the allocation cost of explicitly transposing K* before solving.
 */

import { cholesky } from "../src/linalg/cholesky.js";
import { Matrix } from "../src/linalg/matrix.js";
import { forwardSolve, forwardSolveTransposed } from "../src/linalg/solve.js";

function benchmark(): void {
  const n = 100; // Training points
  const m = 1000; // Test points (typical interactive use)

  // Create a random SPD matrix for L
  const A = Matrix.zeros(n, n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      const val = Math.random();
      A.set(i, j, val);
      A.set(j, i, val);
    }
    A.set(i, i, A.get(i, i) + n); // Make diagonally dominant
  }
  const L = cholesky(A);

  // Create random K* matrix (m × n)
  const Kstar = Matrix.zeros(m, n);
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      Kstar.set(i, j, Math.random());
    }
  }

  // Verify correctness first
  const KstarT = new Matrix(n, m);
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      KstarT.set(j, i, Kstar.get(i, j));
    }
  }
  const result1 = forwardSolve(L, KstarT);
  const result2 = forwardSolveTransposed(L, Kstar);

  let maxDiff = 0;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < m; j++) {
      const diff = Math.abs(result1.get(i, j) - result2.get(i, j));
      maxDiff = Math.max(maxDiff, diff);
    }
  }
  console.log(`\nCorrectness check: max diff = ${maxDiff.toExponential(2)}`);
  if (maxDiff > 1e-10) {
    throw new Error("Results do not match!");
  }

  const iters = 100;

  // Method 1: Explicit transpose + forwardSolve
  const start1 = performance.now();
  for (let iter = 0; iter < iters; iter++) {
    const KstarTrans = new Matrix(n, m);
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) {
        KstarTrans.set(j, i, Kstar.get(i, j));
      }
    }
    forwardSolve(L, KstarTrans);
  }
  const time1 = performance.now() - start1;

  // Method 2: forwardSolveTransposed (no allocation)
  const start2 = performance.now();
  for (let iter = 0; iter < iters; iter++) {
    forwardSolveTransposed(L, Kstar);
  }
  const time2 = performance.now() - start2;

  console.log(`\n=== Transpose Allocation Benchmark (n=${n}, m=${m}, iters=${iters}) ===`);
  console.log(`Method 1 (transpose + solve): ${time1.toFixed(2)}ms`);
  console.log(`Method 2 (forwardSolveTransposed): ${time2.toFixed(2)}ms`);
  console.log(`Speedup: ${(time1 / time2).toFixed(2)}x`);
  console.log(`Allocation saved per call: ${((n * m * 8) / 1024).toFixed(1)} KB`);
}

benchmark();
