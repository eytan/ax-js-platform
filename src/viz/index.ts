// Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

/**
 * Visualization utilities for ax-js.
 *
 * Colormaps, data-point rendering, fixture normalization, and search-space
 * helpers used by the demo suite and available for custom visualizations.
 *
 * @module ax-js/viz
 */

// ── Types ─────────────────────────────────────────────────────────────────
export type { RGB, ParamSpec, RenderPredictor, DotInfo } from "./types";
export type {
  FeatureImportanceOptions,
  CrossValidationOptions,
  OptimizationTraceOptions,
  SlicePlotOptions,
  ResponseSurfaceOptions,
} from "./types";
export type { NormalizedExperiment } from "./fixture";

// ── Styles ────────────────────────────────────────────────────────────────
export { injectStyles, injectScopedStyles } from "./styles";

// ── Colormaps ─────────────────────────────────────────────────────────────
export { viridis, plasma, piYG, drawColorbar, renderHeatmap } from "./colormaps";

// ── Data point rendering ──────────────────────────────────────────────────
export { drawDataDot } from "./dots";

// ── Search-space & fixture helpers ────────────────────────────────────────
export {
  isChoice,
  isInteger,
  defaultParamValue,
  formatParamValue,
  normalizeFixture,
  normalizeExperimentData,
  computeDimOrder,
  computeParamSigns,
  pointRelevance,
} from "./params";

// ── Tooltip helpers ───────────────────────────────────────────────────────
export {
  showTooltip,
  hideTooltip,
  createTooltipDiv,
  positionTooltip,
  removeTooltip,
} from "./tooltip";

// ── Widget helpers ────────────────────────────────────────────────────────
export {
  createOutcomeSelector,
  createParamSliders,
  setupFileUpload,
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

// ── Range estimation ─────────────────────────────────────────────────
export { estimateRange } from "./estimateRange";
export type { EstimatedRange } from "./estimateRange";

// ── Embeddable render functions ───────────────────────────────────────────
export { renderFeatureImportance } from "./plots/importance";
export { renderCrossValidation } from "./plots/cv";
export { renderOptimizationTrace } from "./plots/trace";
export { renderSlicePlot } from "./plots/slice";
export type { SlicePlotController } from "./plots/slice";
export { renderResponseSurface } from "./plots/surface";
export type { ResponseSurfaceController } from "./plots/surface";

// ── Cockpit (multi-panel experiment explorer) ─────────────────────────────
export * as cockpit from "./cockpit/index";
