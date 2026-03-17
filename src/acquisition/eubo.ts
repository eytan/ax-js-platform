import type { GPModel, AcquisitionFunction } from "./types.js";
import { posteriorCovariance, posteriorMean } from "./posterior.js";
import { sampleMVN, Rng } from "./sample_mvn.js";

/**
 * Expected Utility of Best Option (EUBO) for preference exploration.
 *
 * Used with PairwiseGP (BOPE: Bayesian Optimization via Preference Exploration).
 * The "utility" is the latent function the PairwiseGP models.
 *
 * Given a set of q candidate arms (the "menu" presented to the user):
 *   EUBO(x_1, ..., x_q) = E[max_i u(x_i)]
 *
 * where u ~ posterior (PairwiseGP posterior over latent utility).
 *
 * Computed via Monte Carlo:
 * 1. Get posterior mean μ and covariance Σ at the q candidate arms
 * 2. Draw K samples: u^(k) ~ N(μ, Σ)
 * 3. EUBO ≈ (1/K) Σ_k max_i u^(k)(x_i)
 *
 * For optimization, we evaluate EUBO for many candidate menus and pick
 * the one with the highest expected utility of the best option.
 *
 * When q = 2 (pairwise comparison), this reduces to finding the pair
 * where the user's choice is most informative.
 */
export class EUBO implements AcquisitionFunction {
  private rng: Rng;

  /**
   * @param model     - PairwiseGP (or any GPModel modeling latent utility)
   * @param nSamples  - Number of MC samples (default 256)
   * @param seed      - RNG seed for reproducibility
   */
  constructor(
    private model: GPModel,
    private nSamples: number = 256,
    seed: number = 42,
  ) {
    this.rng = new Rng(seed);
  }

  /**
   * Evaluate EUBO for a single menu of candidates.
   *
   * NOTE: Unlike other AFs, EUBO evaluates the JOINT value of the candidate set.
   * The returned Float64Array has length 1 — it's the EUBO value for the
   * entire menu, not per-candidate values.
   *
   * For per-candidate marginal contributions, see evaluateMarginal().
   */
  evaluate(candidates: number[][]): Float64Array {
    const value = this.euboValue(candidates);
    return Float64Array.from([value]);
  }

  /**
   * Compute EUBO value: E[max_i u(x_i)] for the given candidate set.
   */
  euboValue(candidates: number[][]): number {
    const q = candidates.length;
    const mean = posteriorMean(this.model, candidates);
    const Sigma = posteriorCovariance(this.model, candidates);

    // Draw posterior samples: (nSamples × q)
    const samples = sampleMVN(mean, Sigma, this.nSamples, this.rng);

    // MC estimate: E[max_i u(x_i)]
    let sumMax = 0;
    for (let s = 0; s < this.nSamples; s++) {
      let maxU = -Infinity;
      for (let i = 0; i < q; i++) {
        const u = samples.get(s, i);
        if (u > maxU) maxU = u;
      }
      sumMax += maxU;
    }

    return sumMax / this.nSamples;
  }

  /**
   * Evaluate marginal contribution of each candidate to the menu.
   *
   * For each candidate x_i, computes:
   *   Δ_i = EUBO({x_i} ∪ rest) - EUBO(rest)
   *
   * This is O(q · nSamples) using the same set of posterior samples.
   * Useful for greedy menu construction.
   */
  evaluateMarginal(candidates: number[][]): Float64Array {
    const q = candidates.length;
    const mean = posteriorMean(this.model, candidates);
    const Sigma = posteriorCovariance(this.model, candidates);
    const samples = sampleMVN(mean, Sigma, this.nSamples, this.rng);

    // For each sample, compute max over all candidates and max excluding each
    const marginals = new Float64Array(q);
    for (let s = 0; s < this.nSamples; s++) {
      // Find best and second-best indices
      let best = -Infinity;
      let bestIdx = 0;
      let secondBest = -Infinity;
      for (let i = 0; i < q; i++) {
        const u = samples.get(s, i);
        if (u > best) {
          secondBest = best;
          best = u;
          bestIdx = i;
        } else if (u > secondBest) {
          secondBest = u;
        }
      }
      // Marginal contribution: only non-zero for the best candidate in this sample
      // Δ_i = max(all) - max(all \ {i})
      for (let i = 0; i < q; i++) {
        if (i === bestIdx) {
          marginals[i] += (best - secondBest) / this.nSamples;
        }
      }
    }

    return marginals;
  }
}
