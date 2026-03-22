// Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

import type { ExperimentState, FixtureData } from "../models/types.js";

/** Normalized experiment data shape for visualization. */
export interface NormalizedExperiment {
  search_space: ExperimentState["search_space"];
  model_state: ExperimentState["model_state"];
  metadata: {
    name: string;
    description: string;
    [key: string]: unknown;
  };
  test_points: Array<Array<number>>;
  outcome_names?: Array<string>;
  optimization_config?: ExperimentState["optimization_config"];
  status_quo?: ExperimentState["status_quo"];
  adapter_transforms?: ExperimentState["adapter_transforms"];
  observations?: ExperimentState["observations"];
  candidates?: ExperimentState["candidates"];
}

/**
 * Normalize experiment data into a flat shape for visualization.
 *
 * Handles both the `{experiment, test}` fixture format and plain
 * ExperimentState objects, extracting search_space, model_state,
 * metadata, and optional fields into a consistent shape.
 */
export function normalizeExperimentData(
  data: FixtureData | ExperimentState,
): NormalizedExperiment | ExperimentState {
  // Type guard: if it has an 'experiment' field, it's a FixtureData
  if ("experiment" in data && data.experiment) {
    const fixtureData = data;
    const result: NormalizedExperiment = {
      search_space: fixtureData.experiment.search_space,
      model_state: fixtureData.experiment.model_state,
      metadata: {
        name: fixtureData.experiment.name || "",
        description: fixtureData.experiment.description || "",
        ...fixtureData.test?.metadata,
      },
      test_points: fixtureData.test?.test_points || [],
    };
    if (fixtureData.experiment.outcome_names) {
      result.outcome_names = fixtureData.experiment.outcome_names;
    }
    if (fixtureData.experiment.optimization_config) {
      result.optimization_config = fixtureData.experiment.optimization_config;
    }
    if (fixtureData.experiment.status_quo) {
      result.status_quo = fixtureData.experiment.status_quo;
    }
    if (fixtureData.experiment.adapter_transforms) {
      result.adapter_transforms = fixtureData.experiment.adapter_transforms;
    }
    if (fixtureData.experiment.observations) {
      result.observations = fixtureData.experiment.observations;
    }
    if (fixtureData.experiment.candidates) {
      result.candidates = fixtureData.experiment.candidates;
    }
    return result;
  }
  // Already an ExperimentState
  return data as ExperimentState;
}

/**
 * @deprecated Use `normalizeExperimentData` instead.
 */
export const normalizeFixture = normalizeExperimentData;
