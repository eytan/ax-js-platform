// Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

import type { Kernel } from "./types.js";

import { Matrix } from "../linalg/matrix.js";
import { matmul, transpose } from "../linalg/ops.js";

import { kernelDiag } from "./composite.js";

/**
 * Index kernel (task covariance) for multi-task GPs.
 *
 * Computes B[i, j] where B = W @ W^T + diag(v).
 * The input matrices contain task indices (integers) as values.
 */
export class IndexKernel implements Kernel {
  readonly taskCovar: Matrix;

  /**
   * Build task covariance matrix B.
   *
   * Supports three modes:
   * 1. Pre-computed covariance matrix (new BoTorch 0.16+ with PositiveIndexKernel)
   * 2. covar_factor + log_var: B = W @ W^T + diag(exp(log_var))
   * 3. covar_factor + var: B = W @ W^T + diag(var)
   */
  constructor(
    covarFactor: Array<Array<number>>,
    diagVar: Array<number>,
    varIsLog: boolean = true,
    precomputedMatrix?: Array<Array<number>>,
  ) {
    if (precomputedMatrix) {
      this.taskCovar = Matrix.from2D(precomputedMatrix);
    } else {
      const numTasks = diagVar.length;
      const W = Matrix.from2D(covarFactor);
      const WtW = matmul(W, transpose(W));
      this.taskCovar = WtW;
      for (let i = 0; i < numTasks; i++) {
        this.taskCovar.data[i * numTasks + i] += varIsLog ? Math.exp(diagVar[i]) : diagVar[i];
      }
    }
  }

  compute(x1: Matrix, x2: Matrix): Matrix {
    const result = new Matrix(x1.rows, x2.rows);
    for (let i = 0; i < x1.rows; i++) {
      const t1 = Math.round(x1.get(i, 0));
      for (let j = 0; j < x2.rows; j++) {
        const t2 = Math.round(x2.get(j, 0));
        result.set(i, j, this.taskCovar.get(t1, t2));
      }
    }
    return result;
  }

  computeDiag(x: Matrix): Float64Array {
    const diag = new Float64Array(x.rows);
    for (let i = 0; i < x.rows; i++) {
      const t = Math.round(x.get(i, 0));
      diag[i] = this.taskCovar.get(t, t);
    }
    return diag;
  }
}

/**
 * Multi-task kernel (ICM / Intrinsic Coregionalization Model).
 *
 * K_ICM((x1, t1), (x2, t2)) = K_data(x1, x2) * B[t1, t2]
 *
 * The input matrices have the task index as the last column.
 * The data kernel operates on all columns except the last.
 */
export class MultitaskKernel implements Kernel {
  readonly dataKernel: Kernel;
  readonly indexKernel: IndexKernel;
  readonly taskFeature: number;

  constructor(
    dataKernel: Kernel,
    covarFactor: Array<Array<number>>,
    diagVar: Array<number>,
    taskFeature: number = -1,
    varIsLog: boolean = true,
    precomputedMatrix?: Array<Array<number>>,
  ) {
    this.dataKernel = dataKernel;
    this.indexKernel = new IndexKernel(covarFactor, diagVar, varIsLog, precomputedMatrix);
    this.taskFeature = taskFeature;
  }

  private splitDataTask(x: Matrix): { data: Matrix; tasks: Matrix } {
    const tf = this.taskFeature < 0 ? x.cols + this.taskFeature : this.taskFeature;
    const dataCols: Array<number> = [];
    for (let j = 0; j < x.cols; j++) {
      if (j !== tf) {
        dataCols.push(j);
      }
    }

    const data = new Matrix(x.rows, dataCols.length);
    const tasks = new Matrix(x.rows, 1);
    for (let i = 0; i < x.rows; i++) {
      for (let j = 0; j < dataCols.length; j++) {
        data.set(i, j, x.get(i, dataCols[j]));
      }
      tasks.set(i, 0, x.get(i, tf));
    }
    return { data, tasks };
  }

  compute(x1: Matrix, x2: Matrix): Matrix {
    const { data: d1, tasks: t1 } = this.splitDataTask(x1);
    const { data: d2, tasks: t2 } = this.splitDataTask(x2);

    const Kdata = this.dataKernel.compute(d1, d2);
    const Ktask = this.indexKernel.compute(t1, t2);

    // Element-wise product
    const result = new Matrix(x1.rows, x2.rows);
    for (let i = 0; i < result.data.length; i++) {
      result.data[i] = Kdata.data[i] * Ktask.data[i];
    }
    return result;
  }

  computeDiag(x: Matrix): Float64Array {
    const { data, tasks } = this.splitDataTask(x);
    const dataDiag = kernelDiag(this.dataKernel, data);
    const taskDiag = this.indexKernel.computeDiag(tasks);

    const diag = new Float64Array(x.rows);
    for (let i = 0; i < x.rows; i++) {
      diag[i] = dataDiag[i] * taskDiag[i];
    }
    return diag;
  }
}
