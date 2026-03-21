// Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

import type { PairwiseGP } from "../models/pairwise_gp.js";
import type { AnyModelState } from "../models/types.js";

import { EnsembleGP } from "../models/ensemble_gp.js";
import { ModelListGP } from "../models/model_list.js";
import { MultiTaskGP } from "../models/multi_task.js";
import { createPairwiseGP } from "../models/pairwise_gp.js";
import { SingleTaskGP } from "../models/single_task.js";

export type AnyModel = SingleTaskGP | ModelListGP | PairwiseGP | MultiTaskGP | EnsembleGP;

/**
 * Validate a model state for NaN/Inf in numeric fields.
 * Throws with a clear message referencing the regeneration workflow.
 */
function validateNumericFields(state: AnyModelState): void {
  const check = (arr: Array<number> | Array<Array<number>> | undefined, label: string): void => {
    if (!arr) {
      return;
    }
    const flat = Array.isArray(arr[0])
      ? (arr as Array<Array<number>>).flat()
      : (arr as Array<number>);
    for (let i = 0; i < flat.length; i++) {
      if (!isFinite(flat[i])) {
        throw new TypeError(
          `Invalid ${label}[${i}] = ${flat[i]}. ` +
            `Model state contains NaN/Inf. Regenerate with: python python/generate_fixtures.py`,
        );
      }
    }
  };

  if ("train_X" in state) {
    check(state.train_X, "train_X");
  }
  if ("train_Y" in state && Array.isArray(state.train_Y)) {
    check(state.train_Y, "train_Y");
  }
}

export function loadModel(state: AnyModelState): AnyModel {
  validateNumericFields(state);

  switch (state.model_type) {
    case "ModelListGP": {
      return new ModelListGP(state);
    }
    case "PairwiseGP": {
      return createPairwiseGP(state);
    }
    case "MultiTaskGP": {
      return new MultiTaskGP(state);
    }
    case "EnsembleGP": {
      return new EnsembleGP(state);
    }
    case "SingleTaskGP":
    case "FixedNoiseGP":
    default: {
      return new SingleTaskGP(state);
    }
  }
}

export { type AnyModelState } from "../models/types.js";
