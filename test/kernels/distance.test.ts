// Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

import { describe, it, expect } from "vitest";

import { cdist, cdistSquared } from "../../src/kernels/distance.js";
import { Matrix } from "../../src/linalg/matrix.js";

describe("cdistSquared", () => {
  it("computes squared distances correctly", () => {
    const x1 = Matrix.from2D([
      [0, 0],
      [1, 0],
    ]);
    const x2 = Matrix.from2D([
      [0, 1],
      [1, 1],
    ]);
    const d = cdistSquared(x1, x2);
    expect(d.get(0, 0)).toBeCloseTo(1, 10); // (0,0)->(0,1) = 1
    expect(d.get(0, 1)).toBeCloseTo(2, 10); // (0,0)->(1,1) = 2
    expect(d.get(1, 0)).toBeCloseTo(2, 10); // (1,0)->(0,1) = 2
    expect(d.get(1, 1)).toBeCloseTo(1, 10); // (1,0)->(1,1) = 1
  });

  it("self-distance diagonal is zero", () => {
    const x = Matrix.from2D([
      [1, 2],
      [3, 4],
    ]);
    const d = cdistSquared(x, x);
    expect(d.get(0, 0)).toBeCloseTo(0, 10);
    expect(d.get(1, 1)).toBeCloseTo(0, 10);
  });
});

describe("cdist", () => {
  it("computes Euclidean distances", () => {
    const x1 = Matrix.from2D([[0, 0]]);
    const x2 = Matrix.from2D([[3, 4]]);
    const d = cdist(x1, x2);
    expect(d.get(0, 0)).toBeCloseTo(5, 10);
  });

  it("clamps distances >= 1e-15", () => {
    const x = Matrix.from2D([[0, 0]]);
    const d = cdist(x, x);
    expect(d.get(0, 0)).toBeGreaterThanOrEqual(1e-15);
  });
});
