// Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

import { describe, it, expect } from "vitest";

import { RBFKernel } from "../../src/kernels/rbf.js";
import { Matrix } from "../../src/linalg/matrix.js";

describe("RBFKernel", () => {
  it("self-covariance is 1", () => {
    const k = new RBFKernel([1, 1]);
    const x = Matrix.from2D([[0.5, 0.5]]);
    const K = k.compute(x, x);
    expect(K.get(0, 0)).toBeCloseTo(1, 10);
  });

  it("covariance decreases with distance", () => {
    const k = new RBFKernel([1]);
    const x1 = Matrix.from2D([[0]]);
    const x2Near = Matrix.from2D([[0.1]]);
    const x2Far = Matrix.from2D([[2]]);
    const kNear = k.compute(x1, x2Near).get(0, 0);
    const kFar = k.compute(x1, x2Far).get(0, 0);
    expect(kNear).toBeGreaterThan(kFar);
  });

  it("matches known value: exp(-d^2/(2*l^2))", () => {
    const ls = 0.5;
    const k = new RBFKernel([ls]);
    const x1 = Matrix.from2D([[0]]);
    const x2 = Matrix.from2D([[1]]);
    const expected = Math.exp(-1 / (2 * ls * ls));
    expect(k.compute(x1, x2).get(0, 0)).toBeCloseTo(expected, 10);
  });

  it("ARD: different lengthscales per dimension", () => {
    const k = new RBFKernel([0.1, 10]);
    const x1 = Matrix.from2D([[0, 0]]);
    // Moving along dim 0 (short ls) should decay fast
    const x2a = Matrix.from2D([[0.5, 0]]);
    // Moving along dim 1 (long ls) should decay slow
    const x2b = Matrix.from2D([[0, 0.5]]);
    expect(k.compute(x1, x2b).get(0, 0)).toBeGreaterThan(k.compute(x1, x2a).get(0, 0));
  });
});
