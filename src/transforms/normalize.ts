import { Matrix } from "../linalg/matrix.js";

/**
 * Input normalization: X_norm = (X - offset) / coefficient
 * Matches BoTorch's Normalize input transform.
 */
export class InputNormalize {
  readonly offset: Float64Array;
  readonly coefficient: Float64Array;

  constructor(offset: number[], coefficient: number[]) {
    this.offset = Float64Array.from(offset);
    this.coefficient = Float64Array.from(coefficient);
  }

  forward(X: Matrix): Matrix {
    const result = new Matrix(X.rows, X.cols);
    for (let i = 0; i < X.rows; i++) {
      for (let j = 0; j < X.cols; j++) {
        result.set(i, j, (X.get(i, j) - this.offset[j]) / this.coefficient[j]);
      }
    }
    return result;
  }
}
