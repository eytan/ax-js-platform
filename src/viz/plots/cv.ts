import type { RenderPredictor, CrossValidationOptions, DotInfo } from "../types";
import { svgEl } from "./_svg";
import { createOutcomeSelector, createTooltipDiv, positionTooltip, removeTooltip, makeSelectEl } from "../widgets";
import { computeKernelRels, applyDotHighlight, clearDotHighlight, findNearestDot, buildPointTooltipHtml } from "../dots";
import { injectScopedStyles } from "../styles";

const CTRL_CSS = "display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-bottom:8px;pointer-events:auto";

/** Panel interface for cross-panel highlight coordination. */
interface HighlightPanel {
  highlight(idx: number, rels: { raw: number[]; max: number }): void;
  clear(): void;
}

/**
 * Render a leave-one-out cross-validation scatter plot into a container.
 *
 * In interactive mode: outcome selector, "All outcomes" small multiples,
 * side-by-side CV + optimization trace for single outcome, and cross-panel
 * highlight coordination via click-to-pin.
 */
export function renderCrossValidation(
  container: HTMLElement,
  predictor: RenderPredictor,
  options?: CrossValidationOptions,
): void {
  const interactive = options?.interactive !== false;

  if (!interactive) {
    renderCrossValidationStatic(container, predictor, options?.outcome ?? predictor.outcomeNames[0], options);
    return;
  }

  if (!container.id) container.id = "axjs_" + Math.random().toString(36).slice(2, 10);
  removeTooltip(container.id);
  container.innerHTML = "";
  injectScopedStyles(container);
  const hasMulti = predictor.outcomeNames.length > 1;
  let selectedOutcome = options?.outcome ?? (hasMulti ? "__all__" : predictor.outcomeNames[0]);
  const tooltip = createTooltipDiv(container.id);

  const controlsDiv = document.createElement("div");
  controlsDiv.style.cssText = CTRL_CSS;
  const plotsDiv = document.createElement("div");
  plotsDiv.style.cssText = "display:flex;gap:12px;flex-wrap:wrap";
  container.appendChild(controlsDiv);
  container.appendChild(plotsDiv);

  if (hasMulti) {
    const { wrapper, select } = makeSelectEl("Outcome:");
    // Add "All" option
    const allOpt = document.createElement("option");
    allOpt.value = "__all__";
    allOpt.textContent = "All outcomes";
    select.appendChild(allOpt);
    predictor.outcomeNames.forEach((name) => {
      const opt = document.createElement("option");
      opt.value = name;
      opt.textContent = name;
      select.appendChild(opt);
    });
    select.value = selectedOutcome;
    select.onchange = () => { selectedOutcome = select.value; redraw(); };
    controlsDiv.appendChild(wrapper);
  }

  let pinnedIdx = -1;
  let panels: HighlightPanel[] = [];

  function broadcastHighlight(idx: number, rels: { raw: number[]; max: number }) {
    panels.forEach((p) => p.highlight(idx, rels));
  }
  function broadcastClear() {
    panels.forEach((p) => p.clear());
    pinnedIdx = -1;
  }

  function redraw() {
    plotsDiv.innerHTML = "";
    pinnedIdx = -1;
    panels = [];

    if (selectedOutcome === "__all__") {
      // Small multiples: one CV per outcome
      const tileW = 280, tileH = 280;
      for (const name of predictor.outcomeNames) {
        const tile = document.createElement("div");
        tile.style.cssText = `background:#141418;border:0.5px solid #222;border-radius:8px;overflow:hidden;width:${tileW}px;height:${tileH + 28}px`;
        const title = document.createElement("div");
        title.style.cssText = "font-size:13px;font-weight:500;color:#ccc;text-align:center;padding:6px 0 0";
        title.textContent = name;
        tile.appendChild(title);
        renderCVPanel(tile, predictor, name, tileW, tileH, true, tooltip, panels,
          () => pinnedIdx, (v) => { pinnedIdx = v; }, broadcastHighlight, broadcastClear);
        plotsDiv.appendChild(tile);
      }
    } else {
      // Single outcome: full-size CV
      const W = options?.width ?? 440;
      const H = options?.height ?? 440;
      const cvTile = document.createElement("div");
      cvTile.style.cssText = `background:#141418;border:0.5px solid #222;border-radius:8px;overflow:hidden;width:${W}px;height:${H}px`;
      renderCVPanel(cvTile, predictor, selectedOutcome, W, H, false, tooltip, panels,
        () => pinnedIdx, (v) => { pinnedIdx = v; }, broadcastHighlight, broadcastClear);
      plotsDiv.appendChild(cvTile);
    }
  }
  redraw();
}

/** Render a single CV panel with cross-panel coordination. */
function renderCVPanel(
  target: HTMLElement,
  predictor: RenderPredictor,
  outcome: string,
  W: number,
  H: number,
  isSmall: boolean,
  tooltip: HTMLDivElement,
  panels: HighlightPanel[],
  getPinnedIdx: () => number,
  setPinnedIdx: (v: number) => void,
  broadcastHighlight: (idx: number, rels: { raw: number[]; max: number }) => void,
  broadcastClear: () => void,
): void {
  const loo = predictor.loocv(outcome);
  if (loo.observed.length === 0) { target.textContent = "No data"; return; }

  const { observed, mean: predicted, variance } = loo;
  const predStd = variance.map((v) => Math.sqrt(v));
  const n = observed.length;

  // R-squared
  const meanObs = observed.reduce((a, b) => a + b, 0) / n;
  const ssTot = observed.reduce((s, v) => s + (v - meanObs) ** 2, 0);
  const ssRes = observed.reduce((s, v, i) => s + (v - predicted[i]) ** 2, 0);
  const r2 = 1 - ssRes / ssTot;

  // Axis range
  let lo = Math.min(...observed, ...predicted);
  let hi = Math.max(...observed, ...predicted);
  for (let i = 0; i < n; i++) {
    lo = Math.min(lo, predicted[i] - 2 * predStd[i]);
    hi = Math.max(hi, predicted[i] + 2 * predStd[i]);
  }
  const pad = 0.08 * (hi - lo); lo -= pad; hi += pad;

  const margin = isSmall
    ? { top: 24, right: 12, bottom: 30, left: 42 }
    : { top: 30, right: 20, bottom: 40, left: 55 };
  const pw = W - margin.left - margin.right;
  const ph = H - margin.top - margin.bottom;
  const sx = (v: number) => margin.left + ((v - lo) / (hi - lo)) * pw;
  const sy = (v: number) => margin.top + ph - ((v - lo) / (hi - lo)) * ph;

  const svg = svgEl("svg", { width: W, height: H });

  // Diagonal
  svg.appendChild(svgEl("line", {
    x1: sx(lo), y1: sy(lo), x2: sx(hi), y2: sy(hi),
    stroke: "rgba(255,255,255,0.15)", "stroke-width": 1, "stroke-dasharray": "4,4",
  }));

  // Grid + ticks
  const nTicks = isSmall ? 3 : 5;
  const tickFontSize = isSmall ? "8" : "10";
  for (let t = 0; t <= nTicks; t++) {
    const v = lo + ((hi - lo) * t) / nTicks;
    svg.appendChild(svgEl("line", {
      x1: margin.left, x2: margin.left + pw, y1: sy(v), y2: sy(v),
      stroke: "rgba(255,255,255,0.04)",
    }));
    svg.appendChild(Object.assign(svgEl("text", {
      x: sx(v), y: margin.top + ph + (isSmall ? 12 : 16), fill: "#555", "font-size": tickFontSize, "text-anchor": "middle",
    }), { textContent: v.toFixed(isSmall ? 0 : 2) }));
    svg.appendChild(Object.assign(svgEl("text", {
      x: margin.left - 4, y: sy(v) + 3, fill: "#555", "font-size": tickFontSize, "text-anchor": "end",
    }), { textContent: v.toFixed(isSmall ? 0 : 2) }));
  }

  // CI whiskers + dots
  const td = predictor.getTrainingData(outcome);
  const dotR = isSmall ? 3 : 4;
  const cvDots: DotInfo[] = [];
  const defaultFill = "rgba(124,154,255,0.85)";
  const defaultStroke = "rgba(255,255,255,0.5)";
  for (let i = 0; i < n; i++) {
    const cx = sx(observed[i]), cy = sy(predicted[i]);
    const whisker = svgEl("line", {
      x1: cx, x2: cx,
      y1: sy(predicted[i] + 2 * predStd[i]),
      y2: sy(predicted[i] - 2 * predStd[i]),
      stroke: "rgba(124,154,255,0.3)", "stroke-width": isSmall ? 1 : 1.5,
    });
    svg.appendChild(whisker);
    const dot = svgEl("circle", {
      cx, cy, r: dotR, fill: defaultFill,
      stroke: defaultStroke, "stroke-width": 1,
    });
    svg.appendChild(dot);
    cvDots.push({
      cx, cy, idx: i, pt: td.X[i] ?? [], el: dot, whisker,
      defaultFill, defaultStroke, defaultR: dotR,
    });
  }

  // Axis labels
  if (!isSmall) {
    svg.appendChild(Object.assign(svgEl("text", {
      x: margin.left + pw / 2, y: H - 6, fill: "#888", "font-size": 13, "text-anchor": "middle",
    }), { textContent: "Observed" }));
    svg.appendChild(Object.assign(svgEl("text", {
      x: 14, y: margin.top + ph / 2, fill: "#888", "font-size": 13, "text-anchor": "middle",
      transform: `rotate(-90,14,${margin.top + ph / 2})`,
    }), { textContent: "LOO Predicted" }));
  }

  // R-squared
  svg.appendChild(Object.assign(svgEl("text", {
    x: margin.left + 6, y: margin.top + (isSmall ? 14 : 18), fill: "#7c9aff", "font-size": isSmall ? 11 : 14, "font-weight": "600",
  }), { textContent: `R\u00B2 = ${r2.toFixed(4)}` }));

  target.appendChild(svg);

  // Register this panel for cross-panel coordination
  function highlightDots(activeIdx: number, rels: { raw: number[]; max: number }) {
    applyDotHighlight(cvDots, activeIdx, rels);
  }
  function clearDots() {
    clearDotHighlight(cvDots);
  }
  panels.push({ highlight: highlightDots, clear: clearDots });

  // Interactivity
  const HOVER_R = isSmall ? 8 : 12;
  svg.addEventListener("mousemove", (e: MouseEvent) => {
    const rect = svg.getBoundingClientRect();
    const px = e.clientX - rect.left, py = e.clientY - rect.top;
    const best = findNearestDot(cvDots, px, py, HOVER_R);
    if (best >= 0) {
      const d = cvDots[best];
      let html = `<div style="font-weight:600;color:#aaa;margin-bottom:4px">${outcome} — trial ${d.idx}</div>`;
      html += `observed = <span style="color:#7c9aff">${observed[d.idx].toFixed(4)}</span><br>`;
      html += `LOO predicted = <span style="color:#7c9aff">${predicted[d.idx].toFixed(4)}</span><br>`;
      html += `\u00B1 2\u03C3 = [${(predicted[d.idx] - 2 * predStd[d.idx]).toFixed(4)}, ${(predicted[d.idx] + 2 * predStd[d.idx]).toFixed(4)}]<br>`;
      html += buildPointTooltipHtml(predictor, d.idx, outcome);
      tooltip.innerHTML = html;
      tooltip.style.display = "block";
      positionTooltip(tooltip, e.clientX, e.clientY);
      if (getPinnedIdx() < 0) {
        broadcastHighlight(best, computeKernelRels(predictor, cvDots, best, outcome));
      }
    } else {
      tooltip.style.display = "none";
      if (getPinnedIdx() < 0) broadcastClear();
    }
  });

  svg.addEventListener("click", (e: MouseEvent) => {
    const rect = svg.getBoundingClientRect();
    const px = e.clientX - rect.left, py = e.clientY - rect.top;
    const best = findNearestDot(cvDots, px, py, HOVER_R);
    if (best >= 0) {
      setPinnedIdx(best);
      broadcastHighlight(best, computeKernelRels(predictor, cvDots, best, outcome));
    } else {
      broadcastClear();
    }
  });

  svg.addEventListener("mouseleave", () => {
    tooltip.style.display = "none";
    if (getPinnedIdx() < 0) broadcastClear();
  });
}

/** Static (non-interactive) CV rendering used for backward compat. */
function renderCrossValidationStatic(
  target: HTMLElement,
  predictor: RenderPredictor,
  outcome: string,
  options?: CrossValidationOptions,
): void {
  const W = options?.width ?? 440;
  const H = options?.height ?? 440;
  const panels: HighlightPanel[] = [];
  let pinnedIdx = -1;
  if (!target.id) target.id = "axjs_" + Math.random().toString(36).slice(2, 10);
  injectScopedStyles(target);
  const tooltip = createTooltipDiv(target.id);
  renderCVPanel(target, predictor, outcome, W, H, false, tooltip, panels,
    () => pinnedIdx, (v) => { pinnedIdx = v; },
    (idx, rels) => { panels.forEach((p) => p.highlight(idx, rels)); },
    () => { panels.forEach((p) => p.clear()); pinnedIdx = -1; });
}
