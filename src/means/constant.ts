import { Matrix } from "../linalg/matrix.js";

export class ConstantMean {
  readonly constant: number;

  constructor(constant: number) {
    this.constant = constant;
  }

  forward(x: Matrix): Matrix {
    const result = new Matrix(x.rows, 1);
    for (let i = 0; i < x.rows; i++) {
      result.data[i] = this.constant;
    }
    return result;
  }
}
