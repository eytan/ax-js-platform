/** RGB triplet in 0-255 range. */
export type RGB = [number, number, number];

/** Minimal parameter shape accepted by search-space helpers. */
export interface ParamSpec {
  type: "range" | "choice";
  bounds?: [number, number];
  values?: (string | number | boolean)[];
  parameter_type?: "int" | "float";
}

/** Structural type for the predictor methods used by render functions. */
export interface RenderPredictor {
  readonly outcomeNames: string[];
  readonly paramNames: string[];
  readonly paramBounds: [number, number][];
  /** Full parameter specs (type, values, parameter_type). Optional for backward compat. */
  readonly paramSpecs?: ParamSpec[];
  predict(points: number[][]): Record<string, { mean: Float64Array; variance: Float64Array }>;
  getTrainingData(outcomeName?: string): { X: number[][]; Y: number[]; paramNames: string[] };
  loocv(outcomeName?: string): { observed: number[]; mean: number[]; variance: number[] };
  rankDimensionsByImportance(outcomeName?: string): { dimIndex: number; paramName: string; lengthscale: number }[];
  kernelCorrelation(point: number[], refPoint: number[], outcomeName?: string): number;
  computeSensitivity?(outcomeName?: string, options?: { numSamples?: number; seed?: number }):
    { firstOrder: number[]; totalOrder: number[]; paramNames: string[]; numEvaluations: number };
}

/** Info tracked for each visible training dot in an SVG plot. */
export interface DotInfo {
  cx: number;
  cy: number;
  idx: number;
  pt: number[];
  el: SVGCircleElement;
  /** Optional whisker element (CV plot). */
  whisker?: SVGLineElement;
  /** Default fill when not highlighted. */
  defaultFill: string;
  /** Default stroke when not highlighted. */
  defaultStroke: string;
  /** Default radius. */
  defaultR: number;
}

/** Options for renderFeatureImportance. */
export interface FeatureImportanceOptions {
  outcome?: string;
  interactive?: boolean;
  backgroundColor?: string;
  /** Importance mode. Default: "lengthscale". */
  mode?: "lengthscale" | "sobol";
  /** Number of Saltelli samples for Sobol' mode. Default: 512. */
  sobolSamples?: number;
}

/** Options for renderCrossValidation. */
export interface CrossValidationOptions {
  outcome?: string;
  width?: number;
  height?: number;
  interactive?: boolean;
  backgroundColor?: string;
}

/** Options for renderOptimizationTrace. */
export interface OptimizationTraceOptions {
  outcome?: string;
  minimize?: boolean;
  width?: number;
  height?: number;
  interactive?: boolean;
  backgroundColor?: string;
}

/** Options for renderSlicePlot. */
export interface SlicePlotOptions {
  outcome?: string;
  fixedValues?: number[];
  numPoints?: number;
  width?: number;
  height?: number;
  interactive?: boolean;
  backgroundColor?: string;
}

/** Options for renderResponseSurface. */
export interface ResponseSurfaceOptions {
  outcome?: string;
  dimX?: number;
  dimY?: number;
  fixedValues?: number[];
  gridSize?: number;
  width?: number;
  height?: number;
  interactive?: boolean;
  backgroundColor?: string;
}

/** Minimal predictor shape for dimension ranking. */
export interface DimensionRanker {
  rankDimensionsByImportance(
    outcome?: string,
  ): { dimIndex: number }[] | null;
  computeSensitivity?(outcomeName?: string):
    { firstOrder: number[]; totalOrder: number[]; paramNames: string[] };
}

/** Minimal predictor shape accepted by embedding helpers. */
export interface EmbeddingPredictor {
  readonly outcomeNames: string[];
  readonly paramNames: string[];
  readonly paramBounds: [number, number][];
}
