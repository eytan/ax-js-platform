import type { ParamSpec, DimensionRanker } from "./types";

/** Returns true if the parameter is a choice parameter. */
export function isChoice(p: ParamSpec): boolean {
  return p.type === "choice";
}

/** Returns true if the parameter is an integer range parameter. */
export function isInteger(p: ParamSpec): boolean {
  return p.type === "range" && p.parameter_type === "int";
}

/** Returns a sensible default value for a parameter (midpoint or first choice). */
export function defaultParamValue(
  p: ParamSpec,
): number | string | boolean {
  if (isChoice(p)) return p.values![0];
  if (isInteger(p)) return Math.round((p.bounds![0] + p.bounds![1]) / 2);
  return (p.bounds![0] + p.bounds![1]) / 2;
}

/** Format a parameter value for display. */
export function formatParamValue(
  val: number | string | boolean,
  p: ParamSpec,
): string {
  if (isChoice(p)) return String(val);
  if (isInteger(p)) return String(Math.round(val as number));
  return (val as number).toFixed(3);
}

/**
 * Normalize a fixture or ExperimentState into a flat shape for visualization.
 *
 * Handles both the `{experiment, test}` fixture format and plain
 * ExperimentState objects, extracting search_space, model_state,
 * metadata, and optional fields into a consistent shape.
 */
export function normalizeFixture(data: any): any {
  if (data.experiment) {
    const result: any = {
      search_space: data.experiment.search_space,
      model_state: data.experiment.model_state,
      metadata: {
        name: data.experiment.name || "",
        description: data.experiment.description || "",
        ...(data.test?.metadata || {}),
      },
      test_points: data.test?.test_points || [],
    };
    if (data.experiment.outcome_names)
      result.outcome_names = data.experiment.outcome_names;
    if (data.experiment.optimization_config)
      result.optimization_config = data.experiment.optimization_config;
    if (data.experiment.status_quo)
      result.status_quo = data.experiment.status_quo;
    if (data.experiment.adapter_transforms)
      result.adapter_transforms = data.experiment.adapter_transforms;
    if (data.experiment.observations)
      result.observations = data.experiment.observations;
    if (data.experiment.candidates)
      result.candidates = data.experiment.candidates;
    return result;
  }
  return data;
}

/**
 * Compute dimension display order, sorted by importance (shortest lengthscale first).
 * Falls back to natural order if no importance data is available.
 */
export function computeDimOrder(
  predictor: DimensionRanker,
  nDim: number,
  selectedOutcome?: string,
): number[] {
  const ranked = predictor.rankDimensionsByImportance(selectedOutcome);
  if (!ranked || ranked.length === 0) {
    return Array.from({ length: nDim }, (_, i) => i);
  }
  const order = ranked.map((d) => d.dimIndex);
  if (order.length < nDim) {
    const inRanked = new Set(order);
    for (let i = 0; i < nDim; i++) {
      if (!inRanked.has(i)) order.push(i);
    }
  }
  return order;
}

/**
 * Compute kernel-distance relevance between a training point and a reference.
 *
 * Returns `exp(-0.5 * d²)` where d² is the scaled squared distance across
 * non-plotted dimensions (dimensions in `plottedDims` are skipped).
 * Accounts for input warp (Kumaraswamy CDF) when present.
 */
export function pointRelevance(
  pt: number[],
  fixedValues: number[],
  plottedDims: number[],
  ls: number[] | null,
  inputTf: { offset?: number[]; coefficient?: number[] } | null,
  params?: ParamSpec[],
  inputWarp?: { concentration0: number[]; concentration1: number[]; indices?: number[] } | null,
): number {
  const warpIndicesSet = inputWarp?.indices ? new Set(inputWarp.indices) : null;
  const eps = 1e-7;
  const warpRange = 1 - 2 * eps;
  let d2 = 0;
  for (let j = 0; j < fixedValues.length; j++) {
    if (plottedDims.indexOf(j) >= 0) continue;
    if (params && params[j] && isChoice(params[j])) {
      if (pt[j] !== fixedValues[j]) d2 += 4;
      continue;
    }
    const offset = inputTf?.offset?.[j] ?? 0;
    const coeff = inputTf?.coefficient?.[j] ?? 1;
    let v1 = (pt[j] - offset) / coeff;
    let v2 = (fixedValues[j] - offset) / coeff;
    if (inputWarp && (warpIndicesSet === null || warpIndicesSet.has(j))) {
      const wIdx = inputWarp.indices ? inputWarp.indices.indexOf(j) : j;
      if (wIdx >= 0 && wIdx < inputWarp.concentration0.length) {
        const a = inputWarp.concentration1[wIdx];
        const b = inputWarp.concentration0[wIdx];
        const xn1 = Math.max(eps, Math.min(1 - eps, v1 * warpRange + eps));
        const xn2 = Math.max(eps, Math.min(1 - eps, v2 * warpRange + eps));
        v1 = 1 - Math.pow(1 - Math.pow(xn1, a), b);
        v2 = 1 - Math.pow(1 - Math.pow(xn2, a), b);
      }
    }
    const lsj = ls && j < ls.length ? ls[j] : 1;
    const scaled = (v1 - v2) / lsj;
    d2 += scaled * scaled;
  }
  return Math.exp(-0.5 * d2);
}
