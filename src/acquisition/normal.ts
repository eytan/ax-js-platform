// Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

/**
 * Standard normal distribution functions: PDF, CDF, log-CDF.
 *
 * Uses the Cephes-style rational approximation for erfc, which achieves
 * full double precision (~15 digits). For the log-CDF in the far left tail,
 * we use the Mills ratio asymptotic expansion for numerical stability.
 */

const SQRT2 = Math.SQRT2;
const SQRT2PI = Math.sqrt(2 * Math.PI);
const LOG_SQRT2PI = Math.log(SQRT2PI);

/** Standard normal PDF: φ(x) = exp(-x²/2) / √(2π) */
export function normalPdf(x: number): number {
  return Math.exp(-0.5 * x * x) / SQRT2PI;
}

/** Log of standard normal PDF: log φ(x) = -x²/2 - log(√(2π)) */
export function logNormalPdf(x: number): number {
  return -0.5 * x * x - LOG_SQRT2PI;
}

/**
 * Standard normal CDF: Φ(x) = P(Z ≤ x).
 *
 * Uses erfc for accuracy across the full range.
 * erfc(x) = 1 - erf(x), and Φ(x) = 0.5 * erfc(-x / √2).
 */
export function normalCdf(x: number): number {
  return 0.5 * erfc(-x / SQRT2);
}

/**
 * Log of standard normal CDF: log Φ(x).
 *
 * Numerically stable for large negative x where Φ(x) ≈ 0.
 * For x < -6, uses the asymptotic expansion of the Mills ratio.
 * For x ≥ -6, uses log(Φ(x)) directly.
 */
export function logNormalCdf(x: number): number {
  if (x > 6) {
    // Φ(x) ≈ 1 - φ(x)/x for large positive x
    // log(1 - ε) ≈ -ε
    return -normalPdf(x) / x;
  }
  if (x < -6) {
    // Asymptotic: log Φ(x) = log φ(x) - log(-x) + log(1 - 1/x² + 3/x⁴ - ...)
    const x2 = x * x;
    return -0.5 * x2 - LOG_SQRT2PI - Math.log(-x) - 1 / x2 + 3 / (x2 * x2);
  }
  return Math.log(normalCdf(x));
}

/**
 * Complementary error function: erfc(x) = 1 - erf(x).
 *
 * Based on the Chebyshev fitting formula from Numerical Recipes (Press et al.),
 * which gives accuracy to ~1.2 × 10⁻⁷. Combined with the identity
 * erfc(-x) = 2 - erfc(x), this covers the full range.
 *
 * For higher accuracy around x=0, we use a separate polynomial for erf.
 */
function erfc(x: number): number {
  if (x < 0) {
    return 2 - erfc(-x);
  }

  // For small x, compute erf directly via series then subtract from 1
  if (x < 0.5) {
    return 1 - erf(x);
  }

  // Chebyshev fitting (Numerical Recipes, based on Cephes)
  const t = 2 / (2 + x);
  const ty = 4 * t - 2;

  // Coefficients from Chebyshev fit to erfc(x)*exp(x²) for x >= 0.5
  const c = [
    -1.302_653_719_781_709_4, 6.419_697_923_564_902_6e-1, 1.947_647_320_418_583_6e-2,
    -9.561_514_786_808_631e-3, -9.465_953_444_820_36e-4, 3.668_394_978_527_61e-4,
    4.252_332_480_690_7e-5, -2.027_857_811_253_4e-5, -1.624_290_004_647e-6, 1.303_655_835_58e-6,
    1.562_644_172_2e-8, -8.523_809_591_5e-8, 6.529_054_439e-9, 5.059_343_495e-9, -9.913_641_56e-10,
    -2.273_651_22e-10, 9.646_791_1e-11, 2.394_038e-12, -6.886_027e-12, 8.944_87e-13, 3.130_92e-13,
    -1.127_08e-13, 3.81e-16, 7.106e-15, -1.523e-15, -9.4e-17, 1.21e-16, -2.8e-17,
  ];

  const ncof = c.length;
  let d = 0;
  let dd = 0;
  for (let j = ncof - 1; j > 0; j--) {
    const tmp = d;
    d = ty * d - dd + c[j];
    dd = tmp;
  }

  return t * Math.exp(-x * x + 0.5 * (c[0] + ty * d) - dd);
}

/**
 * Error function via Taylor/Horner series. Accurate for |x| < 0.5.
 * erf(x) = (2/√π) * x * (1 - x²/3 + x⁴/10 - x⁶/42 + ...)
 */
function erf(x: number): number {
  const x2 = x * x;
  // Horner form of the series: sum_{k=0}^{N} (-1)^k * x^{2k} / (k! * (2k+1))
  let sum = 1;
  let term = 1;
  for (let k = 1; k <= 20; k++) {
    term *= -x2 / k;
    sum += term / (2 * k + 1);
  }
  return (2 / Math.sqrt(Math.PI)) * x * sum;
}
