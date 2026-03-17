import type { GPModel, AcquisitionFunction } from "./types.js";
import { posteriorCovariance, posteriorMean } from "./posterior.js";
import { sampleMVN, Rng } from "./sample_mvn.js";

/**
 * Thompson Sampling via posterior sampling at candidate points.
 *
 * Algorithm:
 * 1. Compute posterior mean μ and covariance Σ at all candidate points
 * 2. Draw a sample f ~ N(μ, Σ)
 * 3. Return the sampled values as acquisition function values
 *
 * The candidate with the highest sampled value is the TS recommendation.
 *
 * This is the "finite candidate set" version of TS — the sample is only
 * drawn at the provided candidate points. For continuous optimization,
 * pass a large set of random candidates and pick the best.
 */
export class ThompsonSampling implements AcquisitionFunction {
  private rng: Rng;

  constructor(
    private model: GPModel,
    seed: number = 42,
  ) {
    this.rng = new Rng(seed);
  }

  evaluate(candidates: number[][]): Float64Array {
    const mean = posteriorMean(this.model, candidates);
    const Sigma = posteriorCovariance(this.model, candidates);

    // Draw one sample from the joint posterior
    const sample = sampleMVN(mean, Sigma, 1, this.rng);

    // Extract the single sample row
    const n = candidates.length;
    const values = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      values[i] = sample.get(0, i);
    }
    return values;
  }
}

/**
 * Draw multiple Thompson samples and return all of them.
 * Useful for batch TS (selecting q diverse points).
 *
 * @returns Matrix of shape (nSamples × nCandidates)
 */
export function thompsonSamples(
  model: GPModel,
  candidates: number[][],
  nSamples: number,
  seed: number = 42,
) {
  const mean = posteriorMean(model, candidates);
  const Sigma = posteriorCovariance(model, candidates);
  const rng = new Rng(seed);
  return sampleMVN(mean, Sigma, nSamples, rng);
}
