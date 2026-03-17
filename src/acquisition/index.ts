// Normal distribution utilities
export { normalPdf, normalCdf, logNormalPdf, logNormalCdf } from "./normal.js";

// Acquisition functions
export { UpperConfidenceBound } from "./ucb.js";
export { LogExpectedImprovement, ExpectedImprovement } from "./log_ei.js";
export { ThompsonSampling, thompsonSamples } from "./thompson.js";
export { EUBO } from "./eubo.js";

// Posterior utilities
export { posteriorCovariance, posteriorMean } from "./posterior.js";

// Sampling
export { sampleMVN, Rng } from "./sample_mvn.js";

// Optimization
export { optimizeAcqf } from "./optimize.js";
export type { OptimizeAcqfOptions, LBFGSOptions } from "./optimize.js";

// Types
export type {
  GPModel,
  AcquisitionFunction,
  OptimizeResult,
  Bounds,
} from "./types.js";
