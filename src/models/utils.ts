// Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

import type { MetricConfig, MetricIntent, OptimizationConfig } from "./types.js";

/**
 * Derive `MetricConfig[]` from an `OptimizationConfig` and outcome names.
 *
 * Cross-references objectives, outcome_constraints, and objective_thresholds
 * to produce a single flat array. Outcomes not mentioned in any config field
 * are classified as "tracking" metrics.
 *
 * If no `optimizationConfig` is provided, all outcomes are treated as tracking.
 */
export function buildMetricConfigs(
  optimizationConfig: OptimizationConfig | undefined,
  outcomeNames: Array<string>,
): Array<MetricConfig> {
  if (!optimizationConfig) {
    return outcomeNames.map((name) => ({ name, intent: "tracking" as MetricIntent }));
  }

  const objectiveMap = new Map<string, { minimize: boolean }>();
  for (const obj of optimizationConfig.objectives) {
    objectiveMap.set(obj.name, { minimize: obj.minimize });
  }

  const constraintMap = new Map<string, { bound: number; op: "LEQ" | "GEQ"; relative?: boolean }>();
  if (optimizationConfig.outcome_constraints) {
    for (const c of optimizationConfig.outcome_constraints) {
      constraintMap.set(c.name, { bound: c.bound, op: c.op, relative: c.relative });
    }
  }

  const thresholdMap = new Map<
    string,
    { bound: number; op: "LEQ" | "GEQ"; relative?: boolean }
  >();
  if (optimizationConfig.objective_thresholds) {
    for (const t of optimizationConfig.objective_thresholds) {
      thresholdMap.set(t.name, { bound: t.bound, op: t.op, relative: t.relative });
    }
  }

  return outcomeNames.map((name) => {
    const obj = objectiveMap.get(name);
    if (obj) {
      const threshold = thresholdMap.get(name);
      const config: MetricConfig = {
        name,
        intent: "objective",
        lower_is_better: obj.minimize,
      };
      if (threshold) {
        config.bound = threshold.bound;
        config.op = threshold.op;
        config.relative = threshold.relative;
      }
      return config;
    }

    const con = constraintMap.get(name);
    if (con) {
      return {
        name,
        intent: "constraint" as MetricIntent,
        bound: con.bound,
        op: con.op,
        relative: con.relative,
      };
    }

    return { name, intent: "tracking" as MetricIntent };
  });
}
