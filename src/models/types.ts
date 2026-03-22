// Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

/** Recursive kernel tree structure exported from BoTorch/GPyTorch. */
export interface KernelState {
  type: "Matern" | "RBF" | "Scale" | "Categorical" | "Additive" | "Product";
  /** Subset of input dimensions this kernel operates on. */
  active_dims?: Array<number>;

  // Matern/RBF
  /** Per-dimension ARD lengthscales (in normalized [0,1] space when input_transform is present). */
  lengthscale?: Array<number>;
  /** Matern smoothness parameter: 0.5, 1.5, or 2.5. */
  nu?: number;

  // Scale: legacy format has outputscale at top level with type="Matern"|"RBF"
  // New format: type="Scale" with base_kernel
  /** Output amplitude scaling factor for ScaleKernel. */
  outputscale?: number;
  /** Inner kernel wrapped by a ScaleKernel. */
  base_kernel?: KernelState;

  // Composite (Additive/Product)
  /** Child kernels for Additive or Product composition. */
  kernels?: Array<KernelState>;
}

/** Model-level outcome transform applied within BoTorch (Layer 2 of the two-layer pipeline). */
export type OutcomeTransformState =
  | { type?: "Standardize"; mean: number; std: number }
  | { type: "Log" }
  | { type: "Bilog" }
  | { type: "Power"; power: number }
  | { type: "Chained"; transforms: Array<OutcomeTransformState> };

/** State of a single-output GP model (SingleTaskGP or FixedNoiseGP). */
export interface GPModelState {
  model_type: "SingleTaskGP" | "FixedNoiseGP";
  /** Training inputs, shape [n_train, d]. */
  train_X: Array<Array<number>>;
  /** Training targets in doubly-transformed space (adapter + model transforms applied). */
  train_Y: Array<number>;
  kernel: KernelState;
  /** Constant mean function value. */
  mean_constant: number;
  /** Observation noise: scalar (homoskedastic) or per-point array (heteroskedastic/FixedNoiseGP). */
  noise_variance: number | Array<number>;
  /** Normalize transform: `x_norm = (x - offset) / coefficient`. Maps raw inputs to [0,1]. */
  input_transform?: { offset: Array<number>; coefficient: Array<number> };
  outcome_transform?: OutcomeTransformState;
  /** Kumaraswamy CDF input warping for non-stationary modeling. */
  input_warp?: {
    concentration0: Array<number>;
    concentration1: Array<number>;
    indices?: Array<number>;
  };
}

/** Independent GP per outcome, used for multi-output predictions. */
export interface ModelListState {
  model_type: "ModelListGP";
  outcome_names: Array<string>;
  models: Array<GPModelState>;
}

/** GP for preference/pairwise comparison data (Thurstone-Mosteller model). */
export interface PairwiseGPModelState {
  model_type: "PairwiseGP";
  train_X: Array<Array<number>>;
  /** MAP utility estimates at training points. */
  utility: Array<number>;
  /** Hessian of the pairwise likelihood at the MAP, used for Laplace approximation. */
  likelihood_hess: Array<Array<number>>;
  kernel: KernelState;
  mean_constant: number;
  input_transform?: { offset: Array<number>; coefficient: Array<number> };
  input_warp?: {
    concentration0: Array<number>;
    concentration1: Array<number>;
    indices?: Array<number>;
  };
}

/** Multi-task GP with inter-task covariance (ICM/LMC kernel). Used for transfer learning. */
export interface MultiTaskGPModelState {
  model_type: "MultiTaskGP";
  /** Training inputs including task column, shape [n_train, d+1]. */
  train_X: Array<Array<number>>;
  train_Y: Array<number>;
  /** Column index in train_X that encodes the task identifier. */
  task_feature: number;
  num_tasks: number;
  /** Kernel over data dimensions only (task column excluded). */
  data_kernel: KernelState;
  /** Inter-task covariance (IndexKernel/PositiveIndexKernel parameters). */
  task_covar: {
    covar_factor: Array<Array<number>>;
    log_var?: Array<number>;
    var?: Array<number>;
    /** Pre-computed task covariance matrix (used by PositiveIndexKernel). */
    covar_matrix?: Array<Array<number>>;
  };
  /** Scalar (shared) or per-task array of mean constants. */
  mean_constant: number | Array<number>;
  noise_variance: number | Array<number>;
  input_transform?: { offset: Array<number>; coefficient: Array<number> };
  outcome_transform?: OutcomeTransformState;
  input_warp?: {
    concentration0: Array<number>;
    concentration1: Array<number>;
    indices?: Array<number>;
  };
}

/** Ensemble of GPs whose predictions are combined via law of total variance. */
export interface EnsembleGPModelState {
  model_type: "EnsembleGP";
  models: Array<GPModelState>;
}

/** Union of all supported model state types, discriminated by `model_type`. */
export type AnyModelState =
  | GPModelState
  | ModelListState
  | PairwiseGPModelState
  | MultiTaskGPModelState
  | EnsembleGPModelState;

/** Search space parameter definition, matching Ax's parameter config format. */
export interface SearchSpaceParam {
  name: string;
  type: "range" | "choice";
  /** Lower and upper bounds for range parameters. */
  bounds?: [number, number];
  /** Allowed values for choice parameters. */
  values?: Array<string | number | boolean>;
  parameter_type?: "int" | "float";
  /** If true, parameter is sampled on a log scale. */
  log_scale?: boolean;
  is_fidelity?: boolean;
  target_value?: number;
  /** Whether choice parameter values have a natural ordering. */
  is_ordered?: boolean;
}

// ── Parameter constraints ─────────────────────────────────────────────────

/** Linear constraint on parameters: sum(weight_i * param_i) <= or >= bound. */
export interface ParameterConstraint {
  type: "sum" | "order" | "linear";
  /** Maps parameter names to their weights in the linear combination. */
  constraint_dict: Record<string, number>;
  bound: number;
  op: "LEQ" | "GEQ";
}

// ── Optimization config (objectives + constraints) ────────────────────────

/** Objective metric with optimization direction. */
export interface ObjectiveConfig {
  name: string;
  minimize: boolean;
}

/** Constraint on an outcome metric (e.g., latency <= 100ms). */
export interface OutcomeConstraintConfig {
  name: string;
  bound: number;
  op: "LEQ" | "GEQ";
  /** If true, bound is relative to status quo (percentage change). */
  relative?: boolean;
}

/** Worst acceptable value per objective in MOO. Defines the reference point for hypervolume. */
export interface ObjectiveThresholdConfig {
  name: string;
  bound: number;
  op: "LEQ" | "GEQ";
  relative?: boolean;
}

/** Full optimization configuration: objectives, constraints, and MOO thresholds. */
export interface OptimizationConfig {
  objectives: Array<ObjectiveConfig>;
  outcome_constraints?: Array<OutcomeConstraintConfig>;
  /** Objective thresholds for MOO — defines worst acceptable value per objective. */
  objective_thresholds?: Array<ObjectiveThresholdConfig>;
}

// ── Adapter transforms (applied by Ax BEFORE BoTorch) ─────────────────────

/**
 * Adapter-level Y transforms applied by Ax before data reaches BoTorch (Layer 1).
 * These are invisible to the model and must be explicitly undone after prediction.
 */
export type AdapterTransform =
  | { type: "LogY"; metrics?: Array<string> }
  | { type: "BilogY"; metrics?: Array<string> }
  | { type: "StandardizeY"; Ymean?: Record<string, number>; Ystd?: Record<string, number> }
  | {
      type: "PowerTransformY";
      power_params?: Record<
        string,
        | Array<number>
        | {
            lambdas: Array<number>;
            scaler_mean?: Array<number>;
            scaler_scale?: Array<number>;
          }
      >;
    };

// ── Observations (trial data) ─────────────────────────────────────────────

/** A single observed trial arm with its parameter values and metric outcomes. */
export interface Observation {
  arm_name: string;
  /** Parameter name to value mapping. */
  parameters: Record<string, number>;
  /** Metric name to observed mean (and optional SEM). */
  metrics: Record<string, { mean: number; sem?: number }>;
  trial_index?: number;
  trial_status?: string;
  /** How this arm was generated (e.g., "Sobol", "BoTorch"). */
  generation_method?: string;
  /** Explicit batch grouping index. When absent, inferred from generation_method. */
  batch_index?: number;
}

// ── ExperimentState: shared schema for both exports and fixtures ──────────

/**
 * Complete experiment state exported from Ax via `axjs_export()`.
 * This is the primary input to {@link Predictor}.
 */
export interface ExperimentState {
  search_space: {
    parameters: Array<SearchSpaceParam>;
    parameter_constraints?: Array<ParameterConstraint>;
  };
  /** Serialized model with all hyperparameters, training data, and transforms. */
  model_state: AnyModelState;
  name?: string;
  description?: string;
  /** Ordered outcome/metric names. For ModelListGP, matches the order of sub-models. */
  outcome_names?: Array<string>;
  /** Baseline arm for relativization, as a positional array matching parameter order. */
  status_quo?: { point: Array<number> };
  /** Adapter-level transforms to undo after model prediction. */
  adapter_transforms?: Array<AdapterTransform>;
  optimization_config?: OptimizationConfig;
  /** Completed trial observations (for visualization, not used by the model). */
  observations?: Array<Observation>;
  /** Unevaluated candidate arms proposed by the optimizer. */
  candidates?: Array<Candidate>;
}

// ── Fixture schema: experiment + test expectations ────────────────────────

/** Test fixture: experiment state paired with expected predictions for parity testing. */
export interface FixtureData {
  experiment: ExperimentState;
  test: {
    metadata: {
      botorch_version: string;
      gpytorch_version: string;
      torch_version: string;
      generated_at: string;
      seed: number;
      benchmark?: string;
      task_index?: number | null;
      ax_level?: boolean;
      all_tasks?: boolean;
    };
    test_points: Array<Array<number>>;
    expected: {
      mean: Array<number> | Array<Array<number>> | Record<string, Array<number>> | null;
      variance: Array<number> | Array<Array<number>> | Record<string, Array<number>> | null;
    };
    expected_relative?:
      | { mean: Array<number>; variance: Array<number> }
      | Record<string, { mean: Array<number>; variance: Array<number> }>;
  };
}

/** Manifest of test fixtures in the fixture directory. */
export interface Manifest {
  fixtures: Array<{ name: string; file: string; description: string }>;
}

/** Posterior mean and variance arrays from a GP prediction. */
export interface PredictionResult {
  mean: Float64Array;
  variance: Float64Array;
}

// ── Predictor convenience types ───────────────────────────────────────────

/** Training data for one outcome, in raw parameter space with un-standardized Y. */
export interface TrainingData {
  X: Array<Array<number>>;
  Y: Array<number>;
  paramNames: Array<string>;
}

/** Leave-one-out cross-validation result for one outcome. */
export interface LOOCVResult {
  observed: Array<number>; // Y values in original space
  mean: Array<number>; // LOO predicted means in original space
  variance: Array<number>; // LOO predicted variances in original space
}

/** GP internals needed for analytic Sobol' computation. */
export interface GPInternals {
  /** Posterior weights α = (K+σ²I)⁻¹(y - m). */
  alpha: Float64Array;
  /** Normalized + warped training inputs (data dims only for MultiTaskGP). */
  trainXNorm: Array<Array<number>>;
  /** Constant mean function value. */
  meanConstant: number;
}

/** Dimension importance derived from kernel lengthscales. */
export interface DimensionImportance {
  dimIndex: number;
  paramName: string;
  lengthscale: number;
}

/** Sobol' sensitivity indices for variance decomposition of GP predictions. */
export interface SensitivityIndices {
  /** First-order Sobol' index per dimension. */
  firstOrder: Array<number>;
  /** Total-order Sobol' index per dimension. */
  totalOrder: Array<number>;
  /** Parameter names corresponding to each index. */
  paramNames: Array<string>;
  /** Total number of function evaluations used. */
  numEvaluations: number;
}

// ── Trial status & metric classification ─────────────────────────────────

/** Trial lifecycle status, matching Ax's TrialStatus enum. */
export type TrialStatus =
  | "CANDIDATE"
  | "STAGED"
  | "RUNNING"
  | "COMPLETED"
  | "FAILED"
  | "ABANDONED"
  | "EARLY_STOPPED";

/** Metric role derived from OptimizationConfig. */
export type MetricIntent = "objective" | "constraint" | "tracking";

/**
 * Unified metric configuration: consolidates objective direction,
 * constraint bounds, and thresholds into a single per-metric record.
 * Can be derived from OptimizationConfig via `buildMetricConfigs()` or
 * supplied directly (e.g., from a GraphQL layer).
 */
export interface MetricConfig {
  name: string;
  intent: MetricIntent;
  lower_is_better?: boolean;
  bound?: number;
  op?: "LEQ" | "GEQ";
  relative?: boolean;
}

// ── Candidate arms (round-trip workflow) ──────────────────────────────────

/** An unevaluated candidate arm proposed by the optimizer. */
export interface Candidate {
  arm_name?: string;
  /** Parameter name to proposed value mapping. */
  parameters: Record<string, number>;
  trial_index?: number;
  generation_method?: string;
}
