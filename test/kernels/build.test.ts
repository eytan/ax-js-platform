// Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

import type { KernelState } from "../../src/models/types.js";

import { describe, it, expect } from "vitest";

import { buildKernel } from "../../src/kernels/build.js";
import { Matrix } from "../../src/linalg/matrix.js";

describe("buildKernel", () => {
  const x = Matrix.from2D([
    [0.1, 0.5],
    [0.9, 0.2],
  ]);

  it("builds Matern kernel (legacy format)", () => {
    const state: KernelState = {
      type: "Matern",
      lengthscale: [1, 2],
      nu: 2.5,
      outputscale: 0.5,
    };
    const kernel = buildKernel(state);
    const K = kernel.compute(x, x);
    expect(K.get(0, 0)).toBeCloseTo(0.5, 5); // diagonal = outputscale
    expect(K.get(0, 1)).toBeLessThan(0.5);
  });

  it("builds RBF kernel without outputscale", () => {
    const state: KernelState = {
      type: "RBF",
      lengthscale: [1, 1],
    };
    const kernel = buildKernel(state);
    const K = kernel.compute(x, x);
    expect(K.get(0, 0)).toBeCloseTo(1, 5);
  });

  it("builds recursive Scale(Product(RBF, Categorical))", () => {
    const state: KernelState = {
      type: "Scale",
      outputscale: 2,
      base_kernel: {
        type: "Product",
        kernels: [
          { type: "RBF", lengthscale: [1], active_dims: [0] },
          { type: "Categorical", lengthscale: [1], active_dims: [1] },
        ],
      },
    };
    const kernel = buildKernel(state);

    // Same point → k = outputscale * 1 * 1 = 2
    const x1 = Matrix.from2D([[0.5, 0]]);
    const K = kernel.compute(x1, x1);
    expect(K.get(0, 0)).toBeCloseTo(2, 5);

    // Different category → k = outputscale * rbf_part * exp(-1)
    const x2 = Matrix.from2D([[0.5, 1]]);
    const Kcross = kernel.compute(x1, x2);
    expect(Kcross.get(0, 0)).toBeCloseTo(2 * Math.exp(-1), 5);
  });

  it("builds Additive kernel", () => {
    const state: KernelState = {
      type: "Additive",
      kernels: [
        { type: "RBF", lengthscale: [1, 1] },
        { type: "RBF", lengthscale: [2, 2] },
      ],
    };
    const kernel = buildKernel(state);
    const K = kernel.compute(x, x);
    // Diagonal should be 1 + 1 = 2 (sum of two stationary kernels)
    expect(K.get(0, 0)).toBeCloseTo(2, 5);
  });
});
