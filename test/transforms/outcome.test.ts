import { describe, it, expect } from "vitest";
import {
  LogUntransform,
  BilogUntransform,
  PowerUntransform,
  ChainedOutcomeUntransform,
  StandardizeUntransform,
} from "../../src/transforms/outcome.js";

describe("StandardizeUntransform", () => {
  const tf = new StandardizeUntransform(2, 3);

  it("untransforms mean: offset + scale * mu", () => {
    expect(tf.untransform(0, 0).mean).toBe(2);
    expect(tf.untransform(1, 0).mean).toBe(5);
  });

  it("untransforms variance: scale² * var", () => {
    expect(tf.untransform(0, 1).variance).toBe(9);
    expect(tf.untransform(0, 4).variance).toBe(36);
  });
});

describe("LogUntransform (exact log-normal)", () => {
  const tf = new LogUntransform();

  it("mean uses exact log-normal: exp(mu + var/2)", () => {
    // With zero variance, reduces to exp(mu)
    expect(tf.untransform(0, 0).mean).toBe(1);
    expect(tf.untransform(1, 0).mean).toBeCloseTo(Math.E, 10);
    // With nonzero variance, includes the var/2 correction
    const r = tf.untransform(1, 0.5);
    expect(r.mean).toBeCloseTo(Math.exp(1 + 0.25), 10);
  });

  it("variance uses exact log-normal: expm1(var) * exp(2*mu + var)", () => {
    // Var[exp(X)] = (exp(sigma²) - 1) * exp(2*mu + sigma²)
    const mu = 1, v = 0.5;
    const expected = Math.expm1(v) * Math.exp(2 * mu + v);
    expect(tf.untransform(mu, v).variance).toBeCloseTo(expected, 10);
  });

  it("zero variance gives zero variance", () => {
    expect(tf.untransform(2, 0).variance).toBe(0);
  });
});

describe("BilogUntransform (delta method)", () => {
  const tf = new BilogUntransform();

  // bilog(y) = sign(y) * log(|y| + 1), inverse: sign(z) * (exp(|z|) - 1)
  it("inverse of bilog(0) = 0", () => {
    expect(tf.untransform(0, 0).mean).toBe(0);
  });

  it("inverse of bilog for positive value", () => {
    const y = 5;
    const z = Math.log(y + 1); // bilog forward
    expect(tf.untransform(z, 0).mean).toBeCloseTo(y, 10);
  });

  it("inverse of bilog for negative value", () => {
    const y = -3;
    const z = -Math.log(Math.abs(y) + 1); // bilog forward
    expect(tf.untransform(z, 0).mean).toBeCloseTo(y, 10);
  });

  it("variance uses delta method: exp(2|mu|) * var", () => {
    const mu = 2, v = 0.5;
    // (f⁻¹)'(mu) = exp(|mu|) = exp(2), so Var = exp(4) * 0.5
    const expected = Math.exp(4) * 0.5;
    expect(tf.untransform(mu, v).variance).toBeCloseTo(expected, 10);
  });

  it("variance at mu=0 is unchanged (derivative = 1)", () => {
    // exp(|0|) = 1, so Var = 1² * var = var
    expect(tf.untransform(0, 1.5).variance).toBeCloseTo(1.5, 10);
  });
});

describe("PowerUntransform (Yeo-Johnson, CI-width matching)", () => {
  it("λ=0: inverse is exp(z) - 1", () => {
    const tf = new PowerUntransform(0);
    expect(tf.untransform(0, 0).mean).toBeCloseTo(0, 10);
    expect(tf.untransform(Math.log(4), 0).mean).toBeCloseTo(3, 10);
  });

  it("λ=0 variance: positive and larger than delta method for large variance", () => {
    const tf = new PowerUntransform(0);
    const z = 1, v = 0.5;
    const result = tf.untransform(z, v);
    // CI-width matching gives larger variance than delta method for exp inverse
    const deltaVar = Math.exp(2 * z) * v; // delta method: (deriv)² * v
    expect(result.variance).toBeGreaterThan(0);
    expect(result.variance).toBeGreaterThan(deltaVar);
  });

  it("λ=0 zero variance: variance is 0", () => {
    const tf = new PowerUntransform(0);
    expect(tf.untransform(1, 0).variance).toBeCloseTo(0, 10);
  });

  it("λ=0 tiny variance: CI-width ≈ delta method", () => {
    const tf = new PowerUntransform(0);
    const z = 1, v = 1e-8;
    const deltaVar = Math.exp(2 * z) * v;
    expect(tf.untransform(z, v).variance).toBeCloseTo(deltaVar, 4);
  });

  it("λ=2: inverse for negative z uses -log(-y+1) formula", () => {
    const tf = new PowerUntransform(2);
    const y = -2;
    const z = -Math.log(-y + 1);
    expect(tf.untransform(z, 0).mean).toBeCloseTo(y, 10);
  });

  it("λ=0.5: round-trip for positive value", () => {
    const lam = 0.5;
    const tf = new PowerUntransform(lam);
    const y = 2;
    const z = (Math.pow(y + 1, lam) - 1) / lam;
    expect(tf.untransform(z, 0).mean).toBeCloseTo(y, 10);
  });

  it("λ=0.5 variance: positive for nonzero input", () => {
    const lam = 0.5;
    const tf = new PowerUntransform(lam);
    const z = 1, v = 0.3;
    const result = tf.untransform(z, v);
    expect(result.variance).toBeGreaterThan(0);
  });

  it("λ=0.5 tiny variance: CI-width ≈ delta method", () => {
    const lam = 0.5;
    const tf = new PowerUntransform(lam);
    const z = 1, v = 1e-8;
    const deriv = Math.pow(z * lam + 1, 1 / lam - 1);
    const deltaVar = deriv * deriv * v;
    expect(tf.untransform(z, v).variance).toBeCloseTo(deltaVar, 4);
  });

  it("with scaler: un-standardizes before inverse YJ", () => {
    const lam = 0.5;
    const scalerMean = 2.0;
    const scalerScale = 0.5;
    const tf = new PowerUntransform(lam, scalerMean, scalerScale);
    // For mean only (variance=0): z → z*0.5+2.0 → inverse YJ
    const z = 1.0;
    const unstd = z * scalerScale + scalerMean;
    const expected = Math.pow(unstd * lam + 1, 1 / lam) - 1;
    expect(tf.untransform(z, 0).mean).toBeCloseTo(expected, 10);
  });
});

describe("ChainedOutcomeUntransform", () => {
  it("applies transforms in reverse order with joint (mu, var)", () => {
    // Forward: Log → Standardize(mean=2, std=3)
    // Untransform: Standardize⁻¹ first, then Log⁻¹
    const chain = new ChainedOutcomeUntransform([
      new LogUntransform(),
      new StandardizeUntransform(2, 3),
    ]);
    // Input z=0 in standardized-log space
    // Standardize⁻¹ mean: 2 + 3*0 = 2, variance: 9*0.1 = 0.9
    // Log⁻¹ mean: exp(2 + 0.9/2) = exp(2.45)
    // Log⁻¹ variance: expm1(0.9) * exp(2*2 + 0.9)
    const r = chain.untransform(0, 0.1);
    const afterStdMu = 2, afterStdVar = 9 * 0.1;
    const expectedMean = Math.exp(afterStdMu + afterStdVar / 2);
    const expectedVar = Math.expm1(afterStdVar) * Math.exp(2 * afterStdMu + afterStdVar);
    expect(r.mean).toBeCloseTo(expectedMean, 10);
    expect(r.variance).toBeCloseTo(expectedVar, 10);
  });

  it("single-transform chain behaves like the transform itself", () => {
    const log = new LogUntransform();
    const chain = new ChainedOutcomeUntransform([log]);
    const direct = log.untransform(1, 0.5);
    const chained = chain.untransform(1, 0.5);
    expect(chained.mean).toBeCloseTo(direct.mean, 10);
    expect(chained.variance).toBeCloseTo(direct.variance, 10);
  });
});
