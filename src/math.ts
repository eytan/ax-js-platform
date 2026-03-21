// Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

/**
 * Error function approximation (Abramowitz & Stegun 7.1.26).
 * Maximum absolute error: 1.5 × 10⁻⁷.
 */
export function erf(x: number): number {
  const a1 = 0.254_829_592;
  const a2 = -0.284_496_736;
  const a3 = 1.421_413_741;
  const a4 = -1.453_152_027;
  const a5 = 1.061_405_429;
  const p = 0.327_591_1;

  const sign = x < 0 ? -1 : 1;
  const ax = Math.abs(x);
  const t = 1 / (1 + p * ax);
  const y = 1 - ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * Math.exp(-ax * ax);
  return sign * y;
}

/** Standard normal CDF: Φ(x) = 0.5 × (1 + erf(x/√2)). */
export function normalCdf(x: number): number {
  return 0.5 * (1 + erf(x / Math.SQRT2));
}
