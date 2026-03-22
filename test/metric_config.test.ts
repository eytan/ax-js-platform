// Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

import { describe, it, expect } from "vitest";

import { buildMetricConfigs } from "../src/models/utils.js";
import type { OptimizationConfig } from "../src/models/types.js";

describe("buildMetricConfigs", () => {
  it("returns all tracking when no optimization config", () => {
    const configs = buildMetricConfigs(undefined, ["a", "b", "c"]);
    expect(configs).toEqual([
      { name: "a", intent: "tracking" },
      { name: "b", intent: "tracking" },
      { name: "c", intent: "tracking" },
    ]);
  });

  it("classifies objectives, constraints, and tracking", () => {
    const optConfig: OptimizationConfig = {
      objectives: [
        { name: "weight", minimize: true },
        { name: "acceleration", minimize: true },
      ],
      outcome_constraints: [{ name: "door_velocity", bound: 32.0, op: "LEQ" }],
    };
    const configs = buildMetricConfigs(optConfig, [
      "weight",
      "acceleration",
      "door_velocity",
      "some_tracking_metric",
    ]);

    expect(configs[0]).toEqual({ name: "weight", intent: "objective", lower_is_better: true });
    expect(configs[1]).toEqual({
      name: "acceleration",
      intent: "objective",
      lower_is_better: true,
    });
    expect(configs[2]).toEqual({
      name: "door_velocity",
      intent: "constraint",
      bound: 32.0,
      op: "LEQ",
      relative: undefined,
    });
    expect(configs[3]).toEqual({ name: "some_tracking_metric", intent: "tracking" });
  });

  it("includes objective thresholds on objective configs", () => {
    const optConfig: OptimizationConfig = {
      objectives: [{ name: "weight", minimize: true }],
      objective_thresholds: [{ name: "weight", bound: 35.0, op: "LEQ" }],
    };
    const configs = buildMetricConfigs(optConfig, ["weight"]);
    expect(configs[0]).toEqual({
      name: "weight",
      intent: "objective",
      lower_is_better: true,
      bound: 35.0,
      op: "LEQ",
      relative: undefined,
    });
  });

  it("preserves outcome name ordering", () => {
    const optConfig: OptimizationConfig = {
      objectives: [{ name: "b", minimize: false }],
      outcome_constraints: [{ name: "a", bound: 1.0, op: "GEQ" }],
    };
    const configs = buildMetricConfigs(optConfig, ["c", "a", "b"]);
    expect(configs.map((c) => c.name)).toEqual(["c", "a", "b"]);
    expect(configs[0].intent).toBe("tracking");
    expect(configs[1].intent).toBe("constraint");
    expect(configs[2].intent).toBe("objective");
  });

  it("handles relative constraints", () => {
    const optConfig: OptimizationConfig = {
      objectives: [{ name: "x", minimize: true }],
      outcome_constraints: [{ name: "y", bound: 5.0, op: "LEQ", relative: true }],
    };
    const configs = buildMetricConfigs(optConfig, ["x", "y"]);
    expect(configs[1].relative).toBe(true);
  });
});
