// Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

import { cholesky } from "../linalg/cholesky.js";
import { Matrix } from "../linalg/matrix.js";

/**
 * Simple xoshiro128** PRNG for reproducible random number generation.
 * Seeded, fast, and good enough for MC acquisition functions.
 */
export class Rng {
  private s: Uint32Array;

  constructor(seed: number = 42) {
    // Initialize state via splitmix32
    this.s = new Uint32Array(4);
    for (let i = 0; i < 4; i++) {
      seed += 0x9e_37_79_b9;
      let t = seed;
      t = Math.imul(t ^ (t >>> 16), 0x21_f0_aa_ad);
      t = Math.imul(t ^ (t >>> 15), 0x73_5a_2d_97);
      this.s[i] = (t ^ (t >>> 15)) >>> 0;
    }
  }

  /** Uniform [0, 1) */
  uniform(): number {
    const result = this.s[0] + this.s[3];
    const t = this.s[1] << 9;
    this.s[2] ^= this.s[0];
    this.s[3] ^= this.s[1];
    this.s[1] ^= this.s[2];
    this.s[0] ^= this.s[3];
    this.s[2] ^= t;
    this.s[3] = (this.s[3] << 11) | (this.s[3] >>> 21);
    return (result >>> 0) / 4_294_967_296;
  }

  /** Standard normal via Box-Muller transform. */
  randn(): number {
    const u1 = this.uniform();
    const u2 = this.uniform();
    return Math.sqrt(-2 * Math.log(u1 + 1e-300)) * Math.cos(2 * Math.PI * u2);
  }

  /** Fill a Float64Array with standard normal samples. */
  randnArray(n: number): Float64Array {
    const arr = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      arr[i] = this.randn();
    }
    return arr;
  }
}

/**
 * Draw samples from a multivariate normal distribution.
 *
 *   x = μ + L·z,  where z ~ N(0, I), L = cholesky(Σ)
 *
 * @param mean      - Mean vector (length d)
 * @param covariance - Covariance matrix (d × d), or its Cholesky factor if precomputed
 * @param nSamples  - Number of samples to draw
 * @param rng       - Random number generator
 * @param isCholesky - If true, `covariance` is already the lower Cholesky factor
 * @returns Matrix of shape (nSamples × d), each row is one sample
 */
export function sampleMVN(
  mean: Float64Array,
  covariance: Matrix,
  nSamples: number,
  rng: Rng,
  isCholesky: boolean = false,
): Matrix {
  const d = mean.length;
  const L = isCholesky ? covariance : cholesky(covariance);
  const samples = new Matrix(nSamples, d);

  for (let s = 0; s < nSamples; s++) {
    // z ~ N(0, I)
    const z = rng.randnArray(d);
    // x = μ + L·z
    for (let i = 0; i < d; i++) {
      let val = mean[i];
      for (let j = 0; j <= i; j++) {
        val += L.get(i, j) * z[j];
      }
      samples.set(s, i, val);
    }
  }

  return samples;
}
