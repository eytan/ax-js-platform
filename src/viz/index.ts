/**
 * Visualization utilities for ax-js.
 *
 * Colormaps, data-point rendering, fixture normalization, and search-space
 * helpers used by the demo suite and available for custom visualizations.
 *
 * @module ax-js/viz
 */

// ── Types ─────────────────────────────────────────────────────────────────
export type { RGB, ParamSpec, RenderPredictor, DotInfo, ImportanceMethod } from "./types";
export type {
  FeatureImportanceOptions,
  CrossValidationOptions,
  OptimizationTraceOptions,
  SlicePlotOptions,
  ResponseSurfaceOptions,
} from "./types";

// ── Sensitivity analysis ─────────────────────────────────────────────────
export {
  computeImportance,
  computeLengthscaleImportance,
  computeSobolIndices,
  computeGradientImportance,
} from "../sensitivity";
export type { ParameterImportance, SensitivityOptions, SensitivityPredictor } from "../sensitivity";

// ── Styles ────────────────────────────────────────────────────────────────
export { injectStyles, injectScopedStyles } from "./styles";

// ── Colormaps ─────────────────────────────────────────────────────────────
export { viridis, plasma, drawColorbar, renderHeatmap } from "./colormaps";

// ── Data point rendering ──────────────────────────────────────────────────
export { drawDataDot } from "./drawDataDot";

// ── Search-space & fixture helpers ────────────────────────────────────────
export {
  isChoice,
  isInteger,
  defaultParamValue,
  formatParamValue,
  normalizeFixture,
  computeDimOrder,
  pointRelevance,
} from "./params";

// ── Widget helpers ────────────────────────────────────────────────────────
export {
  createOutcomeSelector,
  createParamSliders,
  setupFileUpload,
  showTooltip,
  hideTooltip,
  removeTooltip,
} from "./widgets";

// ── Dot interactivity helpers ─────────────────────────────────────────────
export {
  computeKernelRels,
  applyDotHighlight,
  clearDotHighlight,
  findNearestDot,
  buildPointTooltipHtml,
  attachDotInteractivity,
} from "./dots";

// ── Embeddable render functions ───────────────────────────────────────────
export { renderFeatureImportance } from "./plots/importance";
export { renderCrossValidation } from "./plots/cv";
export { renderOptimizationTrace } from "./plots/trace";
export { renderSlicePlot } from "./plots/slice";
export { renderResponseSurface } from "./plots/surface";
