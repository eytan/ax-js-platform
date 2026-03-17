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
  if (x < 0) return 2 - erfc(-x);

  // For small x, compute erf directly via series then subtract from 1
  if (x < 0.5) {
    return 1 - erf(x);
  }

  // Chebyshev fitting (Numerical Recipes, based on Cephes)
  const t = 2 / (2 + x);
  const ty = 4 * t - 2;

  // Coefficients from Chebyshev fit to erfc(x)*exp(x²) for x >= 0.5
  const c = [
    -1.3026537197817094, 6.4196979235649026e-1, 1.9476473204185836e-2,
    -9.561514786808631e-3, -9.46595344482036e-4, 3.66839497852761e-4,
    4.2523324806907e-5, -2.0278578112534e-5, -1.624290004647e-6,
    1.303655835580e-6, 1.5626441722e-8, -8.5238095915e-8,
    6.529054439e-9, 5.059343495e-9, -9.91364156e-10,
    -2.27365122e-10, 9.6467911e-11, 2.394038e-12,
    -6.886027e-12, 8.94487e-13, 3.13092e-13,
    -1.12708e-13, 3.81e-16, 7.106e-15,
    -1.523e-15, -9.4e-17, 1.21e-16, -2.8e-17,
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
  let sum = 1.0;
  let term = 1.0;
  for (let k = 1; k <= 20; k++) {
    term *= -x2 / k;
    sum += term / (2 * k + 1);
  }
  return (2 / Math.sqrt(Math.PI)) * x * sum;
}
