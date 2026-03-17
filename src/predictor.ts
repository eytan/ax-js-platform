import { loadModel } from "./io/deserialize.js";
import type { AnyModel } from "./io/deserialize.js";
import { ModelListGP } from "./models/model_list.js";
import { MultiTaskGP } from "./models/multi_task.js";
import type {
  ExperimentState,
  SearchSpaceParam,
  PredictionResult,
  AdapterTransform,
  Observation,
} from "./models/types.js";
import {
  LogUntransform,
  BilogUntransform,
  StandardizeUntransform,
  PowerUntransform,
  ChainedOutcomeUntransform,
} from "./transforms/outcome.js";
import type { OutcomeUntransform } from "./transforms/outcome.js";
import {
  relativizePredictions,
  type RelativizeOptions,
} from "./transforms/relativize.js";

/** Predictions keyed by outcome/metric name. */
export type PredictionsByOutcome = Record<string, PredictionResult>;

/**
 * High-level prediction interface mirroring Ax's Adapter.predict().
 *
 * Wraps a loaded model + search space metadata so callers never need to:
 * - Check model type (SingleTaskGP vs ModelListGP vs ...)
 * - Handle input normalization (the model's input_transform does it)
 * - Know about outcome indexing (results are keyed by name)
 * - Apply adapter-level untransforms (LogY, BilogY, etc.)
 */
export class Predictor {
  readonly outcomeNames: string[];
  readonly paramNames: string[];
  readonly paramBounds: [number, number][];
  readonly paramSpecs: SearchSpaceParam[];
  readonly statusQuoPoint: number[] | null;
  private model: AnyModel;
  private _state: ExperimentState;
  private adapterUntransforms: Map<string, OutcomeUntransform> | null;

  constructor(exported: ExperimentState) {
    this._state = exported;
    this.model = loadModel(exported.model_state);
    this.paramSpecs = exported.search_space.parameters;
    this.paramNames = exported.search_space.parameters.map((p) => p.name);
    this.paramBounds = exported.search_space.parameters.map(
      (p) => (p.bounds || [0, 1]) as [number, number],
    );
    this.statusQuoPoint = exported.status_quo?.point ?? null;
    if (exported.outcome_names) {
      this.outcomeNames = exported.outcome_names;
    } else if (this.model instanceof ModelListGP) {
      this.outcomeNames = this.model.outcomeNames;
    } else {
      this.outcomeNames = ["y"];
    }
    this.adapterUntransforms = buildAdapterUntransforms(
      exported.adapter_transforms,
      this.outcomeNames,
    );
  }

  /** Observed trial data, if included in the ExperimentState. */
  get observations(): Observation[] | undefined {
    return this._state.observations;
  }

  /**
   * Predict at one or more points (raw parameter space).
   * Points are positional arrays matching search_space parameter order.
   * The model's input_transform handles normalization if present.
   *
   * Returns predictions keyed by outcome name.
   */
  predict(points: number[][]): PredictionsByOutcome {
    if (this.model instanceof ModelListGP) {
      const results = this.model.predict(points);
      const out: PredictionsByOutcome = {};
      for (let k = 0; k < results.length; k++) {
        out[this.outcomeNames[k]] = this.applyAdapterUntransform(
          this.outcomeNames[k],
          results[k],
        );
      }
      return out;
    }
    if (this.model instanceof MultiTaskGP) {
      const out: PredictionsByOutcome = {};
      for (let t = 0; t < this.model.numTasks; t++) {
        const result = this.model.predict(points, t);
        const name = this.outcomeNames[t] || `task_${t}`;
        out[name] = this.applyAdapterUntransform(name, result);
      }
      return out;
    }
    const result = (
      this.model as Exclude<AnyModel, ModelListGP | MultiTaskGP>
    ).predict(points);
    return {
      [this.outcomeNames[0]]: this.applyAdapterUntransform(
        this.outcomeNames[0],
        result,
      ),
    };
  }

  /**
   * Predict relative effects (% change vs status quo) at given points.
   * Requires status_quo to be set in ExperimentState.
   * Returns predictions keyed by outcome name with relativized mean/variance.
   *
   * By default, uses model covariance between test and status quo points for
   * tighter confidence intervals. Pass `useCovariance: false` to assume
   * independence (matches Ax's default `cov_means=0` behavior).
   */
  predictRelative(
    points: number[][],
    opts?: RelativizeOptions & { useCovariance?: boolean },
  ): PredictionsByOutcome {
    if (!this.statusQuoPoint) {
      throw new Error("Cannot relativize: no status_quo defined");
    }

    const absPreds = this.predict(points);
    const sqPreds = this.predict([this.statusQuoPoint]);
    const useCovariance = opts?.useCovariance !== false;

    const out: PredictionsByOutcome = {};
    for (const name of Object.keys(absPreds)) {
      const sqMean = sqPreds[name].mean[0];
      const sqVar = sqPreds[name].variance[0];
      const covariances = useCovariance
        ? this.getCovariances(name, points)
        : undefined;

      out[name] = relativizePredictions(
        absPreds[name].mean,
        absPreds[name].variance,
        sqMean,
        sqVar,
        opts,
        covariances,
      );
    }
    return out;
  }

  /**
   * Get covariances between predictions at points and the status quo.
   * Used internally by predictRelative for tighter confidence intervals.
   */
  private getCovariances(
    outcomeName: string,
    points: number[][],
  ): Float64Array | undefined {
    if (!this.statusQuoPoint) return undefined;
    const sqPoint = this.statusQuoPoint;

    if (this.model instanceof ModelListGP) {
      const idx = this.outcomeNames.indexOf(outcomeName);
      if (idx < 0) return undefined;
      const covs = this.model.predictCovarianceWith(points, sqPoint);
      return covs[idx];
    }
    if (this.model instanceof MultiTaskGP) {
      const taskIdx = this.outcomeNames.indexOf(outcomeName);
      if (taskIdx < 0) return undefined;
      return this.model.predictCovarianceWith(points, taskIdx, sqPoint);
    }
    // SingleTaskGP, PairwiseGP, EnsembleGP
    const m = this.model as { predictCovarianceWith?: (a: number[][], b: number[]) => Float64Array };
    if (typeof m.predictCovarianceWith === "function") {
      return m.predictCovarianceWith(points, sqPoint);
    }
    return undefined;
  }

  private applyAdapterUntransform(
    outcomeName: string,
    result: PredictionResult,
  ): PredictionResult {
    if (!this.adapterUntransforms) return result;
    const untransform = this.adapterUntransforms.get(outcomeName);
    if (!untransform) return result;

    const n = result.mean.length;
    const mean = new Float64Array(n);
    const variance = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const ut = untransform.untransform(result.mean[i], result.variance[i]);
      mean[i] = ut.mean;
      variance[i] = ut.variance;
    }
    return { mean, variance };
  }
}

/**
 * Build per-outcome adapter untransforms from the adapter_transforms list.
 * Returns null if no adapter transforms are present.
 *
 * Ax may apply multiple adapter transforms per metric (e.g., BilogY then
 * StandardizeY). These are collected per metric and composed into a
 * ChainedOutcomeUntransform that applies them in the correct reverse order.
 */
function buildAdapterUntransforms(
  transforms: AdapterTransform[] | undefined,
  outcomeNames: string[],
): Map<string, OutcomeUntransform> | null {
  if (!transforms || transforms.length === 0) return null;

  // Collect transforms per metric in forward order
  const perMetric = new Map<string, OutcomeUntransform[]>();

  for (const tf of transforms) {
    // Determine which metrics this transform applies to
    const metrics =
      "metrics" in tf && tf.metrics ? tf.metrics : outcomeNames;

    switch (tf.type) {
      case "LogY":
        for (const metric of metrics) {
          if (!perMetric.has(metric)) perMetric.set(metric, []);
          perMetric.get(metric)!.push(new LogUntransform());
        }
        break;
      case "BilogY":
        for (const metric of metrics) {
          if (!perMetric.has(metric)) perMetric.set(metric, []);
          perMetric.get(metric)!.push(new BilogUntransform());
        }
        break;
      case "StandardizeY":
        if (tf.Ymean && tf.Ystd) {
          for (const metric of metrics) {
            if (metric in tf.Ymean && metric in tf.Ystd) {
              if (!perMetric.has(metric)) perMetric.set(metric, []);
              perMetric.get(metric)!.push(
                new StandardizeUntransform(tf.Ymean[metric], tf.Ystd[metric]),
              );
            }
          }
        }
        break;
      case "PowerTransformY":
        if (tf.power_params) {
          for (const metric of metrics) {
            if (metric in tf.power_params) {
              const entry = tf.power_params[metric];
              // Support both old format ({metric: [lambdas]}) and new
              // ({metric: {lambdas, scaler_mean, scaler_scale}})
              const lambdas = Array.isArray(entry) ? entry : entry.lambdas;
              const scalerMean = Array.isArray(entry)
                ? undefined
                : entry.scaler_mean?.[0];
              const scalerScale = Array.isArray(entry)
                ? undefined
                : entry.scaler_scale?.[0];
              if (!perMetric.has(metric)) perMetric.set(metric, []);
              perMetric
                .get(metric)!
                .push(new PowerUntransform(lambdas[0], scalerMean, scalerScale));
            }
          }
        }
        break;
    }
  }

  if (perMetric.size === 0) return null;

  // Build final map: single transform or chained
  const map = new Map<string, OutcomeUntransform>();
  for (const [metric, tfs] of perMetric) {
    if (tfs.length === 1) {
      map.set(metric, tfs[0]);
    } else {
      // ChainedOutcomeUntransform reverses internally for correct undo order
      map.set(metric, new ChainedOutcomeUntransform(tfs));
    }
  }

  return map.size > 0 ? map : null;
}
