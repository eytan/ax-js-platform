import { Matrix } from "../linalg/matrix.js";

export interface Kernel {
  compute(x1: Matrix, x2: Matrix): Matrix;
  computeDiag?(x: Matrix): Float64Array;
}
