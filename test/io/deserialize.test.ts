// Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

import type {
  GPModelState,
  ModelListState,
  PairwiseGPModelState,
  MultiTaskGPModelState,
} from "../../src/models/types.js";

import { describe, it, expect } from "vitest";

import { loadModel } from "../../src/io/deserialize.js";
import { ModelListGP } from "../../src/models/model_list.js";
import { MultiTaskGP } from "../../src/models/multi_task.js";
import { PairwiseGP } from "../../src/models/pairwise_gp.js";
import { SingleTaskGP } from "../../src/models/single_task.js";

const MINIMAL_KERNEL = { type: "Matern" as const, lengthscale: [1], nu: 2.5, outputscale: 1 };
const TRAIN_X = [
  [0.1, 0.2],
  [0.3, 0.4],
  [0.5, 0.6],
];
const TRAIN_Y = [1, 2, 3];

describe("loadModel dispatch", () => {
  it("returns SingleTaskGP for SingleTaskGP model_type", () => {
    const state: GPModelState = {
      model_type: "SingleTaskGP",
      train_X: TRAIN_X,
      train_Y: TRAIN_Y,
      kernel: MINIMAL_KERNEL,
      mean_constant: 0,
      noise_variance: 0.1,
    };
    const model = loadModel(state);
    expect(model).toBeInstanceOf(SingleTaskGP);
  });

  it("returns SingleTaskGP for FixedNoiseGP model_type", () => {
    const state: GPModelState = {
      model_type: "FixedNoiseGP",
      train_X: TRAIN_X,
      train_Y: TRAIN_Y,
      kernel: MINIMAL_KERNEL,
      mean_constant: 0,
      noise_variance: [0.1, 0.1, 0.1],
    };
    const model = loadModel(state);
    expect(model).toBeInstanceOf(SingleTaskGP);
  });

  it("returns ModelListGP for ModelListGP model_type", () => {
    const state: ModelListState = {
      model_type: "ModelListGP",
      outcome_names: ["y0", "y1"],
      models: [
        {
          model_type: "SingleTaskGP",
          train_X: TRAIN_X,
          train_Y: TRAIN_Y,
          kernel: MINIMAL_KERNEL,
          mean_constant: 0,
          noise_variance: 0.1,
        },
        {
          model_type: "SingleTaskGP",
          train_X: TRAIN_X,
          train_Y: [4, 5, 6],
          kernel: MINIMAL_KERNEL,
          mean_constant: 0,
          noise_variance: 0.1,
        },
      ],
    };
    const model = loadModel(state);
    expect(model).toBeInstanceOf(ModelListGP);
  });

  it("returns PairwiseGP for PairwiseGP model_type", () => {
    const state: PairwiseGPModelState = {
      model_type: "PairwiseGP",
      train_X: TRAIN_X,
      utility: [0.5, 1, 1.5],
      likelihood_hess: [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
      ],
      kernel: MINIMAL_KERNEL,
      mean_constant: 0,
    };
    const model = loadModel(state);
    expect(model).toBeInstanceOf(PairwiseGP);
  });

  it("rejects NaN in train_X", () => {
    const state: GPModelState = {
      model_type: "SingleTaskGP",
      train_X: [[0.1, Number.NaN]],
      train_Y: [1],
      kernel: MINIMAL_KERNEL,
      mean_constant: 0,
      noise_variance: 0.1,
    };
    expect(() => loadModel(state)).toThrow(/NaN/);
  });

  it("returns MultiTaskGP for MultiTaskGP model_type", () => {
    const state: MultiTaskGPModelState = {
      model_type: "MultiTaskGP",
      train_X: [
        [0.1, 0.2, 0],
        [0.3, 0.4, 0],
        [0.5, 0.6, 1],
        [0.7, 0.8, 1],
      ],
      train_Y: [1, 2, 3, 4],
      task_feature: -1,
      num_tasks: 2,
      data_kernel: MINIMAL_KERNEL,
      task_covar: {
        covar_factor: [[0.5], [0.3]],
        log_var: [-1, -1],
      },
      mean_constant: 0,
      noise_variance: 0.1,
    };
    const model = loadModel(state);
    expect(model).toBeInstanceOf(MultiTaskGP);
  });
});
