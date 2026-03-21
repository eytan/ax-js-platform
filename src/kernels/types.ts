// Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

import type { Matrix } from "../linalg/matrix.js";

export interface Kernel {
  compute(x1: Matrix, x2: Matrix): Matrix;
  computeDiag?(x: Matrix): Float64Array;
}
