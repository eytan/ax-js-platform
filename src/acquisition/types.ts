import type { PredictionResult } from "../models/types.js";

/**
 * Minimal model interface for acquisition functions.
 *
 * All axjs model types (SingleTaskGP, PairwiseGP, MultiTaskGP, EnsembleGP)
 * satisfy this interface. ModelListGP does NOT (it returns PredictionResult[]
 * for multi-output); use individual sub-models instead.
 */
export interface GPModel {
  predict(testPoints: number[][]): PredictionResult;
  predictCovarianceWith(
    testPoints: number[][],
    refPoint: number[],
  ): Float64Array;
}

/**
 * An acquisition function maps candidate points to scalar values.
 * Higher values = more desirable points to evaluate.
 */
export interface AcquisitionFunction {
  evaluate(candidates: number[][]): Float64Array;
}

/** Result of acquisition function optimization. */
export interface OptimizeResult {
  /** Best candidate found. */
  point: number[];
  /** Acquisition function value at the best point. */
  value: number;
  /** All evaluated candidates (sorted best-first). */
  candidates?: number[][];
  /** All acquisition values (sorted best-first). */
  values?: Float64Array;
}

/** Bounds for each dimension: [lower, upper]. */
export type Bounds = [number, number][];
