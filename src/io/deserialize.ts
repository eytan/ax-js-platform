import { SingleTaskGP } from "../models/single_task.js";
import { ModelListGP } from "../models/model_list.js";
import { PairwiseGP, createPairwiseGP } from "../models/pairwise_gp.js";
import { MultiTaskGP } from "../models/multi_task.js";
import { EnsembleGP } from "../models/ensemble_gp.js";
import type {
  GPModelState,
  ModelListState,
  PairwiseGPModelState,
  MultiTaskGPModelState,
  EnsembleGPModelState,
  AnyModelState,
} from "../models/types.js";

export type { AnyModelState };

export type AnyModel =
  | SingleTaskGP
  | ModelListGP
  | PairwiseGP
  | MultiTaskGP
  | EnsembleGP;

/**
 * Validate a model state for NaN/Inf in numeric fields.
 * Throws with a clear message referencing the regeneration workflow.
 */
function validateNumericFields(state: AnyModelState): void {
  const check = (arr: number[] | number[][] | undefined, label: string) => {
    if (!arr) return;
    const flat = Array.isArray(arr[0])
      ? (arr as number[][]).flat()
      : (arr as number[]);
    for (let i = 0; i < flat.length; i++) {
      if (!isFinite(flat[i])) {
        throw new Error(
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
    check(state.train_Y as number[], "train_Y");
  }
}

export function loadModel(state: AnyModelState): AnyModel {
  validateNumericFields(state);

  switch (state.model_type) {
    case "ModelListGP":
      return new ModelListGP(state as ModelListState);
    case "PairwiseGP":
      return createPairwiseGP(state as PairwiseGPModelState);
    case "MultiTaskGP":
      return new MultiTaskGP(state as MultiTaskGPModelState);
    case "EnsembleGP":
      return new EnsembleGP(state as EnsembleGPModelState);
    default:
      return new SingleTaskGP(state as GPModelState);
  }
}
