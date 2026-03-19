/**
 * **ax-js** — Client-side Gaussian Process predictions mirroring Ax/BoTorch.
 *
 * Prediction-only: model hyperparameters are fitted in Python and exported
 * via `axjs_export()`. This package deserializes the exported state and
 * computes GP posteriors entirely in the browser or Node.js.
 *
 * Start with {@link Predictor} for the high-level API, or use {@link loadModel}
 * for direct model access. Acquisition functions are available via `"ax-js-platform/acquisition"`.
 *
 * @example
 * ```ts
 * import { Predictor } from "ax-js-platform";
 *
 * const predictor = new Predictor(experimentState);
 * const preds = predictor.predict([[0.5, 1.0, 3.0]]);
 * ```
 *
 * @packageDocumentation
 */

// ── Public API ────────────────────────────────────────────────────────────

// Predictor (Ax-aligned high-level API)
export { Predictor } from "./predictor.js";
export type { PredictionsByOutcome } from "./predictor.js";

// Low-level model loading (bypasses transforms — prefer Predictor)
export { loadModel } from "./io/deserialize.js";
export type { AnyModel } from "./io/deserialize.js";

// Input transform helpers (for constructing synthetic model states)
export {
  identityInputTransform,
  boundsInputTransform,
} from "./transforms/normalize.js";
export type { InputTransformState } from "./transforms/normalize.js";

// Relativization (% change vs status quo)
export {
  relativize,
  unrelativize,
  relativizePredictions,
} from "./transforms/relativize.js";
export type { RelativizeResult, RelativizeOptions } from "./transforms/relativize.js";

// Sensitivity analysis
export { computeSobolIndices, haltonSequence } from "./sensitivity.js";
export type { SaltelliOptions } from "./sensitivity.js";
export {
  computeAnalyticSobolIndices,
  computeEnsembleAnalyticSobol,
} from "./sensitivity_analytic.js";

// Math utilities
export { erf, normalCdf } from "./math.js";

// Types — schema, config, and data interfaces
export type {
  ExperimentState,
  SearchSpaceParam,
  PredictionResult,
  AnyModelState,
  ObjectiveConfig,
  OutcomeConstraintConfig,
  ObjectiveThresholdConfig,
  OptimizationConfig,
  ParameterConstraint,
  Observation,
  AdapterTransform,
  Candidate,
  TrainingData,
  LOOCVResult,
  DimensionImportance,
  SensitivityIndices,
  GPInternals,
} from "./models/types.js";
