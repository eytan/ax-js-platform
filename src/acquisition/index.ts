/**
 * **ax-js/acquisition** — Client-side acquisition functions and optimization.
 *
 * Provides standard BO acquisition functions (UCB, EI, Thompson) plus
 * EUBO for preference-based optimization. Includes a random-search +
 * projected L-BFGS optimizer for maximizing acquisition values over bounds.
 *
 * @packageDocumentation
 */

// Normal distribution utilities (used internally by EI/LogEI)
export { normalPdf, normalCdf, logNormalPdf, logNormalCdf } from "./normal.js";

// Acquisition functions — each implements the AcquisitionFunction interface
export { UpperConfidenceBound } from "./ucb.js";
export { LogExpectedImprovement, ExpectedImprovement } from "./log_ei.js";
export { ThompsonSampling, thompsonSamples } from "./thompson.js";
export { EUBO } from "./eubo.js";

// Posterior utilities — compute GP posterior mean/covariance at test points
export { posteriorCovariance, posteriorMean } from "./posterior.js";

// Multivariate normal sampling (Cholesky-based)
export { sampleMVN, Rng } from "./sample_mvn.js";

// Acquisition optimization — random search + projected L-BFGS
export { optimizeAcqf } from "./optimize.js";
export type { OptimizeAcqfOptions, LBFGSOptions } from "./optimize.js";

// Types — GPModel is the minimal GP interface for acqf evaluation;
// AcquisitionFunction defines `evaluate(x) → number`.
export type {
  GPModel,
  AcquisitionFunction,
  OptimizeResult,
  Bounds,
} from "./types.js";
