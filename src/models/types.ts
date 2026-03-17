/** Recursive kernel tree structure exported from BoTorch/GPyTorch. */
export interface KernelState {
  type: "Matern" | "RBF" | "Scale" | "Categorical" | "Additive" | "Product";
  /** Subset of input dimensions this kernel operates on. */
  active_dims?: number[];

  // Matern/RBF
  /** Per-dimension ARD lengthscales (in normalized [0,1] space when input_transform is present). */
  lengthscale?: number[];
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
  kernels?: KernelState[];
}

/** Model-level outcome transform applied within BoTorch (Layer 2 of the two-layer pipeline). */
export type OutcomeTransformState =
  | { type?: "Standardize"; mean: number; std: number }
  | { type: "Log" }
  | { type: "Bilog" }
  | { type: "Power"; power: number }
  | { type: "Chained"; transforms: OutcomeTransformState[] };

/** State of a single-output GP model (SingleTaskGP or FixedNoiseGP). */
export interface GPModelState {
  model_type: "SingleTaskGP" | "FixedNoiseGP";
  /** Training inputs, shape [n_train, d]. */
  train_X: number[][];
  /** Training targets in doubly-transformed space (adapter + model transforms applied). */
  train_Y: number[];
  kernel: KernelState;
  /** Constant mean function value. */
  mean_constant: number;
  /** Observation noise: scalar (homoskedastic) or per-point array (heteroskedastic/FixedNoiseGP). */
  noise_variance: number | number[];
  /** Normalize transform: `x_norm = (x - offset) / coefficient`. Maps raw inputs to [0,1]. */
  input_transform?: { offset: number[]; coefficient: number[] };
  outcome_transform?: OutcomeTransformState;
  /** Kumaraswamy CDF input warping for non-stationary modeling. */
  input_warp?: {
    concentration0: number[];
    concentration1: number[];
    indices?: number[];
  };
}

/** Independent GP per outcome, used for multi-output predictions. */
export interface ModelListState {
  model_type: "ModelListGP";
  outcome_names: string[];
  models: GPModelState[];
}

/** GP for preference/pairwise comparison data (Thurstone-Mosteller model). */
export interface PairwiseGPModelState {
  model_type: "PairwiseGP";
  train_X: number[][];
  /** MAP utility estimates at training points. */
  utility: number[];
  /** Hessian of the pairwise likelihood at the MAP, used for Laplace approximation. */
  likelihood_hess: number[][];
  kernel: KernelState;
  mean_constant: number;
  input_transform?: { offset: number[]; coefficient: number[] };
  input_warp?: {
    concentration0: number[];
    concentration1: number[];
    indices?: number[];
  };
}

/** Multi-task GP with inter-task covariance (ICM/LMC kernel). Used for transfer learning. */
export interface MultiTaskGPModelState {
  model_type: "MultiTaskGP";
  /** Training inputs including task column, shape [n_train, d+1]. */
  train_X: number[][];
  train_Y: number[];
  /** Column index in train_X that encodes the task identifier. */
  task_feature: number;
  num_tasks: number;
  /** Kernel over data dimensions only (task column excluded). */
  data_kernel: KernelState;
  /** Inter-task covariance (IndexKernel/PositiveIndexKernel parameters). */
  task_covar: {
    covar_factor: number[][];
    log_var?: number[];
    var?: number[];
    /** Pre-computed task covariance matrix (used by PositiveIndexKernel). */
    covar_matrix?: number[][];
  };
  /** Scalar (shared) or per-task array of mean constants. */
  mean_constant: number | number[];
  noise_variance: number | number[];
  input_transform?: { offset: number[]; coefficient: number[] };
  outcome_transform?: OutcomeTransformState;
  input_warp?: {
    concentration0: number[];
    concentration1: number[];
    indices?: number[];
  };
}

/** Ensemble of GPs whose predictions are combined via law of total variance. */
export interface EnsembleGPModelState {
  model_type: "EnsembleGP";
  models: GPModelState[];
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
  values?: (string | number | boolean)[];
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
  objectives: ObjectiveConfig[];
  outcome_constraints?: OutcomeConstraintConfig[];
  /** Objective thresholds for MOO — defines worst acceptable value per objective. */
  objective_thresholds?: ObjectiveThresholdConfig[];
}

// ── Adapter transforms (applied by Ax BEFORE BoTorch) ─────────────────────

/**
 * Adapter-level Y transforms applied by Ax before data reaches BoTorch (Layer 1).
 * These are invisible to the model and must be explicitly undone after prediction.
 */
export type AdapterTransform =
  | { type: "LogY"; metrics?: string[] }
  | { type: "BilogY"; metrics?: string[] }
  | { type: "StandardizeY"; Ymean?: Record<string, number>; Ystd?: Record<string, number> }
  | {
      type: "PowerTransformY";
      power_params?: Record<
        string,
        | number[]
        | {
            lambdas: number[];
            scaler_mean?: number[];
            scaler_scale?: number[];
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
}

// ── ExperimentState: shared schema for both exports and fixtures ──────────

/**
 * Complete experiment state exported from Ax via `axjs_export()`.
 * This is the primary input to {@link Predictor}.
 */
export interface ExperimentState {
  search_space: {
    parameters: SearchSpaceParam[];
    parameter_constraints?: ParameterConstraint[];
  };
  /** Serialized model with all hyperparameters, training data, and transforms. */
  model_state: AnyModelState;
  name?: string;
  description?: string;
  /** Ordered outcome/metric names. For ModelListGP, matches the order of sub-models. */
  outcome_names?: string[];
  /** Baseline arm for relativization, as a positional array matching parameter order. */
  status_quo?: { point: number[] };
  /** Adapter-level transforms to undo after model prediction. */
  adapter_transforms?: AdapterTransform[];
  optimization_config?: OptimizationConfig;
  /** Completed trial observations (for visualization, not used by the model). */
  observations?: Observation[];
  /** Unevaluated candidate arms proposed by the optimizer. */
  candidates?: Candidate[];
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
    test_points: number[][];
    expected: {
      mean: number[] | number[][] | Record<string, number[]> | null;
      variance: number[] | number[][] | Record<string, number[]> | null;
    };
    expected_relative?:
      | { mean: number[]; variance: number[] }
      | Record<string, { mean: number[]; variance: number[] }>;
  };
}

/** Manifest of test fixtures in the fixture directory. */
export interface Manifest {
  fixtures: { name: string; file: string; description: string }[];
}

/** Posterior mean and variance arrays from a GP prediction. */
export interface PredictionResult {
  mean: Float64Array;
  variance: Float64Array;
}

// ── Predictor convenience types ───────────────────────────────────────────

/** Training data for one outcome, in raw parameter space with un-standardized Y. */
export interface TrainingData {
  X: number[][];
  Y: number[];
  paramNames: string[];
}

/** Leave-one-out cross-validation result for one outcome. */
export interface LOOCVResult {
  observed: number[];   // Y values in original space
  mean: number[];       // LOO predicted means in original space
  variance: number[];   // LOO predicted variances in original space
}

/** Dimension importance derived from kernel lengthscales. */
export interface DimensionImportance {
  dimIndex: number;
  paramName: string;
  lengthscale: number;
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
