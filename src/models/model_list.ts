// Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

import type { GPInternals, ModelListState, PredictionResult } from "./types.js";

import { SingleTaskGP } from "./single_task.js";

export class ModelListGP {
  readonly outcomeNames: Array<string>;
  private readonly models: Array<SingleTaskGP>;

  constructor(state: ModelListState) {
    this.outcomeNames = state.outcome_names;
    this.models = state.models.map((s) => new SingleTaskGP(s));
  }

  /** Expose GP internals for a specific sub-model. */
  getInternals(index: number): GPInternals {
    return this.models[index].getInternals();
  }

  predict(testPoints: Array<Array<number>>): Array<PredictionResult> {
    return this.models.map((m) => m.predict(testPoints));
  }

  predictOutcome(index: number, testPoints: Array<Array<number>>): PredictionResult {
    return this.models[index].predict(testPoints);
  }

  /** Analytic LOO-CV predictions per outcome. */
  loocvPredictions(): Array<PredictionResult> {
    return this.models.map((m) => m.loocvPredictions());
  }

  /**
   * Posterior covariance with a reference point, per output.
   * Each sub-model is independent, so no cross-output covariance.
   */
  predictCovarianceWith(
    testPoints: Array<Array<number>>,
    refPoint: Array<number>,
  ): Array<Float64Array> {
    return this.models.map((m) => m.predictCovarianceWith(testPoints, refPoint));
  }
}
