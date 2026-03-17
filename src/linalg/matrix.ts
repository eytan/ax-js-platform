/**
 * Float64Array-backed matrix class with row-major storage.
 * Matches BoTorch's default float64 precision.
 */
export class Matrix {
  readonly data: Float64Array;
  readonly rows: number;
  readonly cols: number;

  constructor(rows: number, cols: number, data?: Float64Array) {
    this.rows = rows;
    this.cols = cols;
    this.data = data ?? new Float64Array(rows * cols);
  }

  static from2D(arr: number[][]): Matrix {
    const rows = arr.length;
    const cols = arr[0]?.length ?? 0;
    const m = new Matrix(rows, cols);
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        m.data[i * cols + j] = arr[i][j];
      }
    }
    return m;
  }

  static fromFlat(rows: number, cols: number, data: number[]): Matrix {
    return new Matrix(rows, cols, Float64Array.from(data));
  }

  static zeros(rows: number, cols: number): Matrix {
    return new Matrix(rows, cols);
  }

  static eye(n: number): Matrix {
    const m = new Matrix(n, n);
    for (let i = 0; i < n; i++) m.data[i * n + i] = 1;
    return m;
  }

  static vector(arr: number[]): Matrix {
    return new Matrix(arr.length, 1, Float64Array.from(arr));
  }

  get(i: number, j: number): number {
    return this.data[i * this.cols + j];
  }

  set(i: number, j: number, v: number): void {
    this.data[i * this.cols + j] = v;
  }

  row(i: number): Float64Array {
    return this.data.subarray(i * this.cols, (i + 1) * this.cols);
  }

  col(j: number): Float64Array {
    const c = new Float64Array(this.rows);
    for (let i = 0; i < this.rows; i++) c[i] = this.data[i * this.cols + j];
    return c;
  }

  addDiag(value: number): void {
    const n = Math.min(this.rows, this.cols);
    for (let i = 0; i < n; i++) {
      this.data[i * this.cols + i] += value;
    }
  }

  addDiagVec(values: number[] | Float64Array): void {
    const n = Math.min(this.rows, this.cols, values.length);
    for (let i = 0; i < n; i++) {
      this.data[i * this.cols + i] += values[i];
    }
  }

  clone(): Matrix {
    return new Matrix(this.rows, this.cols, new Float64Array(this.data));
  }

  toArray(): number[][] {
    const result: number[][] = [];
    for (let i = 0; i < this.rows; i++) {
      result.push(Array.from(this.row(i)));
    }
    return result;
  }
}
