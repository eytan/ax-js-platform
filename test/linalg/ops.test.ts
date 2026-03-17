import { describe, it, expect } from "vitest";
import { Matrix } from "../../src/linalg/matrix.js";
import { matmul, transpose, add, scale, dot } from "../../src/linalg/ops.js";

describe("transpose", () => {
  it("transposes a 2×3 matrix", () => {
    const A = Matrix.from2D([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    const AT = transpose(A);
    expect(AT.rows).toBe(3);
    expect(AT.cols).toBe(2);
    expect(AT.get(0, 0)).toBe(1);
    expect(AT.get(0, 1)).toBe(4);
    expect(AT.get(2, 0)).toBe(3);
    expect(AT.get(2, 1)).toBe(6);
  });

  it("double transpose is identity", () => {
    const A = Matrix.from2D([
      [1, 2],
      [3, 4],
    ]);
    const ATT = transpose(transpose(A));
    for (let i = 0; i < 2; i++) {
      for (let j = 0; j < 2; j++) {
        expect(ATT.get(i, j)).toBe(A.get(i, j));
      }
    }
  });
});

describe("matmul", () => {
  it("multiplies 2×3 by 3×2", () => {
    const A = Matrix.from2D([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    const B = Matrix.from2D([
      [7, 8],
      [9, 10],
      [11, 12],
    ]);
    const C = matmul(A, B);
    expect(C.rows).toBe(2);
    expect(C.cols).toBe(2);
    expect(C.get(0, 0)).toBe(58);
    expect(C.get(0, 1)).toBe(64);
    expect(C.get(1, 0)).toBe(139);
    expect(C.get(1, 1)).toBe(154);
  });

  it("multiplies by identity", () => {
    const A = Matrix.from2D([
      [3, 7],
      [2, 5],
    ]);
    const I = Matrix.eye(2);
    const AI = matmul(A, I);
    for (let i = 0; i < 2; i++) {
      for (let j = 0; j < 2; j++) {
        expect(AI.get(i, j)).toBe(A.get(i, j));
      }
    }
  });
});

describe("add", () => {
  it("adds two matrices", () => {
    const A = Matrix.from2D([[1, 2], [3, 4]]);
    const B = Matrix.from2D([[5, 6], [7, 8]]);
    const C = add(A, B);
    expect(C.get(0, 0)).toBe(6);
    expect(C.get(1, 1)).toBe(12);
  });
});

describe("scale", () => {
  it("scales a matrix", () => {
    const A = Matrix.from2D([[2, 4], [6, 8]]);
    const B = scale(A, 0.5);
    expect(B.get(0, 0)).toBe(1);
    expect(B.get(1, 1)).toBe(4);
  });
});

describe("dot", () => {
  it("computes dot product", () => {
    const a = Float64Array.from([1, 2, 3]);
    const b = Float64Array.from([4, 5, 6]);
    expect(dot(a, b)).toBe(32);
  });
});
