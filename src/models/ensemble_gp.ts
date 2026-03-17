import { SingleTaskGP } from "./single_task.js";
import type {
  EnsembleGPModelState,
  GPModelState,
  PredictionResult,
} from "./types.js";

/**
 * Ensemble GP that averages predictions from K posterior-sample GPs.
 *
 * Used for fully Bayesian models like SAAS where NUTS produces K sets of
 * hyperparameters. Each member is a standard SingleTaskGP with different
 * kernel hyperparameters.
 *
 * Prediction:
 *   mean = (1/K) * sum(mu_k)
 *   variance = E[Var] + Var[E]
 *            = (1/K) * sum(var_k) + (1/K) * sum(mu_k^2) - mean^2
 */
export class EnsembleGP {
  private models: SingleTaskGP[];

  constructor(state: EnsembleGPModelState) {
    if (!state.models || state.models.length === 0) {
      throw new Error("EnsembleGP requires at least one model");
    }
    this.models = state.models.map((ms) => new SingleTaskGP(ms));
  }

  predict(testPoints: number[][]): PredictionResult {
    if (testPoints.length === 0) {
      throw new Error("testPoints must not be empty");
    }
    const results = this.models.map((m) => m.predict(testPoints));
    const n = testPoints.length;
    const K = results.length;

    const mean = new Float64Array(n);
    const variance = new Float64Array(n);

    for (let i = 0; i < n; i++) {
      let sumMu = 0;
      let sumVar = 0;
      let sumMu2 = 0;
      for (let k = 0; k < K; k++) {
        const mu_k = results[k].mean[i];
        sumMu += mu_k;
        sumVar += results[k].variance[i];
        sumMu2 += mu_k * mu_k;
      }
      mean[i] = sumMu / K;
      // Law of total variance: E[Var] + Var[E]
      variance[i] = sumVar / K + (sumMu2 / K - mean[i] * mean[i]);
    }

    return { mean, variance };
  }

  /**
   * Posterior covariance with a reference point, averaged across ensemble.
   *
   * Law of total covariance:
   *   Cov(f(a), f(b)) = E_k[Cov_k(f(a), f(b))] + Cov_k(mu_k(a), mu_k(b))
   *
   * First term: average within-model covariance.
   * Second term: sample covariance of per-model means.
   */
  predictCovarianceWith(
    testPoints: number[][],
    refPoint: number[],
  ): Float64Array {
    const K = this.models.length;
    const n = testPoints.length;

    // Collect per-model covariances and means
    const covs = this.models.map((m) =>
      m.predictCovarianceWith(testPoints, refPoint),
    );
    const testResults = this.models.map((m) => m.predict(testPoints));
    const refResults = this.models.map((m) => m.predict([refPoint]));

    const result = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let sumCov = 0; // E[Cov_k]
      let sumMuA = 0; // for Cov(mu_a, mu_b)
      let sumMuB = 0;
      let sumMuAB = 0;
      for (let k = 0; k < K; k++) {
        sumCov += covs[k][i];
        const muA = testResults[k].mean[i];
        const muB = refResults[k].mean[0];
        sumMuA += muA;
        sumMuB += muB;
        sumMuAB += muA * muB;
      }
      // E[Cov_k] + Cov(mu_a, mu_b)
      result[i] =
        sumCov / K + (sumMuAB / K - (sumMuA / K) * (sumMuB / K));
    }

    return result;
  }
}
