// Linear algebra
export { Matrix } from "./linalg/matrix.js";
export { cholesky } from "./linalg/cholesky.js";
export { forwardSolve, backSolve, solveCholesky } from "./linalg/solve.js";
export { solveLU } from "./linalg/lu.js";
export { transpose, matmul, add, scale, dot } from "./linalg/ops.js";

// Kernels
export type { Kernel } from "./kernels/types.js";
export { cdist, cdistSquared } from "./kernels/distance.js";
export { MaternKernel } from "./kernels/matern.js";
export { RBFKernel } from "./kernels/rbf.js";
export { ScaleKernel } from "./kernels/scale.js";
export { CategoricalKernel } from "./kernels/categorical.js";
export {
  ActiveDimsKernel,
  AdditiveKernel,
  ProductKernel,
  kernelDiag,
} from "./kernels/composite.js";
export { IndexKernel, MultitaskKernel } from "./kernels/multitask.js";
export { buildKernel } from "./kernels/build.js";

// Mean functions
export { ConstantMean } from "./means/constant.js";

// Transforms
export { InputNormalize } from "./transforms/normalize.js";
export type { OutcomeUntransform } from "./transforms/outcome.js";
export {
  StandardizeUntransform,
  StandardizeUntransform as OutputUntransform,
  LogUntransform,
  BilogUntransform,
  PowerUntransform,
  ChainedOutcomeUntransform,
} from "./transforms/outcome.js";
export { InputWarp } from "./transforms/warp.js";
export { buildOutcomeUntransform } from "./transforms/build_outcome.js";
export {
  relativize,
  unrelativize,
  relativizePredictions,
} from "./transforms/relativize.js";
export type { RelativizeResult, RelativizeOptions } from "./transforms/relativize.js";

// Models
export { ExactGP } from "./models/gp.js";
export { SingleTaskGP } from "./models/single_task.js";
export { ModelListGP } from "./models/model_list.js";
export { PairwiseGP, createPairwiseGP } from "./models/pairwise_gp.js";
export { MultiTaskGP } from "./models/multi_task.js";
export { EnsembleGP } from "./models/ensemble_gp.js";

// IO
export { loadModel } from "./io/deserialize.js";
export type { AnyModel } from "./io/deserialize.js";

// Predictor (Ax-aligned high-level API)
export { Predictor } from "./predictor.js";
export type { PredictionsByOutcome } from "./predictor.js";

// Acquisition functions & optimization
export {
  normalPdf,
  normalCdf,
  logNormalPdf,
  logNormalCdf,
  UpperConfidenceBound,
  LogExpectedImprovement,
  ExpectedImprovement,
  ThompsonSampling,
  thompsonSamples,
  EUBO,
  posteriorCovariance,
  posteriorMean,
  sampleMVN,
  Rng,
  optimizeAcqf,
} from "./acquisition/index.js";
export type {
  GPModel,
  AcquisitionFunction,
  OptimizeResult,
  Bounds,
  OptimizeAcqfOptions,
  LBFGSOptions,
} from "./acquisition/index.js";

// Types
export type {
  GPModelState,
  ModelListState,
  PairwiseGPModelState,
  MultiTaskGPModelState,
  EnsembleGPModelState,
  AnyModelState,
  KernelState,
  OutcomeTransformState,
  SearchSpaceParam,
  AdapterTransform,
  Observation,
  ObjectiveConfig,
  OutcomeConstraintConfig,
  OptimizationConfig,
  ExperimentState,
  FixtureData,
  Manifest,
  PredictionResult,
} from "./models/types.js";
