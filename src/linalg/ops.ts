import { Matrix } from "./matrix.js";

export function transpose(a: Matrix): Matrix {
  const result = new Matrix(a.cols, a.rows);
  for (let i = 0; i < a.rows; i++) {
    for (let j = 0; j < a.cols; j++) {
      result.data[j * a.rows + i] = a.data[i * a.cols + j];
    }
  }
  return result;
}

export function matmul(a: Matrix, b: Matrix): Matrix {
  const result = new Matrix(a.rows, b.cols);
  for (let i = 0; i < a.rows; i++) {
    for (let k = 0; k < a.cols; k++) {
      const aik = a.data[i * a.cols + k];
      if (aik === 0) continue;
      for (let j = 0; j < b.cols; j++) {
        result.data[i * b.cols + j] += aik * b.data[k * b.cols + j];
      }
    }
  }
  return result;
}

export function add(a: Matrix, b: Matrix): Matrix {
  const result = new Matrix(a.rows, a.cols);
  for (let i = 0; i < a.data.length; i++) {
    result.data[i] = a.data[i] + b.data[i];
  }
  return result;
}

export function scale(a: Matrix, s: number): Matrix {
  const result = new Matrix(a.rows, a.cols);
  for (let i = 0; i < a.data.length; i++) {
    result.data[i] = a.data[i] * s;
  }
  return result;
}

export function dot(a: Float64Array, b: Float64Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}
