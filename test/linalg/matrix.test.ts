// Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

import { describe, it, expect } from "vitest";

import { Matrix } from "../../src/linalg/matrix.js";

describe("Matrix", () => {
  it("from2D creates correct matrix", () => {
    const m = Matrix.from2D([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    expect(m.rows).toBe(2);
    expect(m.cols).toBe(3);
    expect(m.get(0, 0)).toBe(1);
    expect(m.get(1, 2)).toBe(6);
  });

  it("eye creates identity", () => {
    const m = Matrix.eye(3);
    expect(m.get(0, 0)).toBe(1);
    expect(m.get(0, 1)).toBe(0);
    expect(m.get(1, 1)).toBe(1);
    expect(m.get(2, 2)).toBe(1);
  });

  it("addDiag adds scalar to diagonal", () => {
    const m = Matrix.zeros(3, 3);
    m.addDiag(5);
    expect(m.get(0, 0)).toBe(5);
    expect(m.get(1, 1)).toBe(5);
    expect(m.get(0, 1)).toBe(0);
  });

  it("addDiagVec adds vector to diagonal", () => {
    const m = Matrix.zeros(3, 3);
    m.addDiagVec([1, 2, 3]);
    expect(m.get(0, 0)).toBe(1);
    expect(m.get(1, 1)).toBe(2);
    expect(m.get(2, 2)).toBe(3);
  });

  it("row returns correct subarray", () => {
    const m = Matrix.from2D([
      [1, 2],
      [3, 4],
    ]);
    const r = m.row(1);
    expect(r[0]).toBe(3);
    expect(r[1]).toBe(4);
  });

  it("col returns correct values", () => {
    const m = Matrix.from2D([
      [1, 2],
      [3, 4],
    ]);
    const c = m.col(1);
    expect(c[0]).toBe(2);
    expect(c[1]).toBe(4);
  });

  it("clone creates independent copy", () => {
    const m = Matrix.from2D([[1, 2]]);
    const c = m.clone();
    c.set(0, 0, 99);
    expect(m.get(0, 0)).toBe(1);
    expect(c.get(0, 0)).toBe(99);
  });

  it("vector creates n×1 matrix", () => {
    const v = Matrix.vector([1, 2, 3]);
    expect(v.rows).toBe(3);
    expect(v.cols).toBe(1);
    expect(v.get(1, 0)).toBe(2);
  });

  it("toArray round-trips", () => {
    const arr = [
      [1, 2],
      [3, 4],
    ];
    const m = Matrix.from2D(arr);
    expect(m.toArray()).toEqual(arr);
  });
});
