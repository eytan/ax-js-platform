import type { RenderPredictor, FeatureImportanceOptions, ImportanceMethod } from "../types";
import type { ParameterImportance } from "../../sensitivity.js";
import { computeImportance } from "../../sensitivity.js";
import { svgEl } from "./_svg";
import { createOutcomeSelector, createTooltipDiv, positionTooltip, removeTooltip, makeSelectEl } from "../widgets";
import { injectScopedStyles } from "../styles";

const CTRL_CSS = "display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-bottom:8px;pointer-events:auto";

const METHOD_LABELS: Record<ImportanceMethod, string> = {
  lengthscale: "Lengthscale",
  sobol: "Sobol Indices",
  gradient: "Gradient (DGSM)",
};

/**
 * Render a horizontal bar chart of feature importance into a container.
 *
 * Supports three importance methods selectable via a dropdown:
 * - **lengthscale** (default): 1 / ARD lengthscale
 * - **sobol**: First-order Sobol indices (variance-based global sensitivity)
 * - **gradient**: DGSM (derivative-based global sensitivity measure)
 */
export function renderFeatureImportance(
  container: HTMLElement,
  predictor: RenderPredictor,
  options?: FeatureImportanceOptions,
): void {
  const interactive = options?.interactive !== false;

  if (!interactive) {
    const method = options?.method ?? "lengthscale";
    const outcome = options?.outcome ?? predictor.outcomeNames[0];
    const items = computeImportance(predictor, outcome, method, {
      numSamples: options?.numSamples,
      seed: options?.seed,
    });
    renderBars(container, items, method);
    return;
  }

  if (!container.id) container.id = "axjs_" + Math.random().toString(36).slice(2, 10);
  removeTooltip(container.id);
  container.innerHTML = "";
  injectScopedStyles(container);
  let selectedOutcome = options?.outcome ?? predictor.outcomeNames[0];
  let selectedMethod: ImportanceMethod = options?.method ?? "lengthscale";
  const tooltip = createTooltipDiv(container.id);

  const controlsDiv = document.createElement("div");
  controlsDiv.style.cssText = CTRL_CSS;
  const plotsDiv = document.createElement("div");
  container.appendChild(controlsDiv);
  container.appendChild(plotsDiv);

  // Outcome selector
  if (predictor.outcomeNames.length > 1) {
    const { wrapper, select } = makeSelectEl("Outcome:");
    createOutcomeSelector(predictor, select, (name) => {
      selectedOutcome = name;
      redraw();
    });
    controlsDiv.appendChild(wrapper);
  }

  // Method selector
  {
    const { wrapper, select } = makeSelectEl("Method:");
    const methods: ImportanceMethod[] = ["lengthscale", "sobol", "gradient"];
    for (const m of methods) {
      const opt = document.createElement("option");
      opt.value = m;
      opt.textContent = METHOD_LABELS[m];
      if (m === selectedMethod) opt.selected = true;
      select.appendChild(opt);
    }
    select.onchange = () => {
      selectedMethod = select.value as ImportanceMethod;
      redraw();
    };
    controlsDiv.appendChild(wrapper);
  }

  function redraw() {
    plotsDiv.innerHTML = "";
    const items = computeImportance(predictor, selectedOutcome, selectedMethod, {
      numSamples: options?.numSamples,
      seed: options?.seed,
    });
    renderBars(plotsDiv, items, selectedMethod, tooltip, container);
  }
  redraw();
}

// ---------------------------------------------------------------------------
// Static bar chart renderer (works with any ParameterImportance[])
// ---------------------------------------------------------------------------

function renderBars(
  target: HTMLElement,
  items: ParameterImportance[],
  method: ImportanceMethod,
  tooltip?: HTMLDivElement,
  tooltipContainer?: HTMLElement,
): void {
  if (!items || items.length === 0) {
    target.textContent = "No importance data";
    return;
  }

  const barColors = ["#4872f9", "#5478fa", "#6088fa", "#7098fb", "#85a8fb", "#9ab8fc", "#b0c8fc", "#c7d4fd"];

  const W = Math.min((tooltipContainer ?? target).clientWidth || 500, 500);
  const labelW = 130;
  const valueW = 70;
  const barH = 24;
  const rowGap = 6;
  const H = items.length * (barH + rowGap) + 8;
  const trackW = W - labelW - valueW;

  const svg = svgEl("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}` });

  items.forEach((item, i) => {
    const y = i * (barH + rowGap) + 4;
    const pct = item.importance;
    const barW = Math.max(2, pct * trackW);

    // Label
    svg.appendChild(
      Object.assign(svgEl("text", {
        x: labelW - 8, y: y + barH / 2 + 4,
        fill: "#333", "font-size": 13, "text-anchor": "end",
      }), { textContent: item.paramName }),
    );

    // Track
    svg.appendChild(svgEl("rect", {
      x: labelW, y, width: trackW, height: barH,
      rx: 4, fill: "#f0f0f0",
    }));

    // Fill bar
    svg.appendChild(svgEl("rect", {
      x: labelW, y, width: barW, height: barH,
      rx: 4, fill: barColors[item.dimIndex % barColors.length],
    }));

    // Tooltip on bar hover
    if (tooltip) {
      const hoverRect = svgEl("rect", {
        x: labelW, y, width: trackW, height: barH,
        fill: "transparent", cursor: "pointer",
      });
      hoverRect.addEventListener("mouseenter", (e: MouseEvent) => {
        tooltip.innerHTML = tooltipHtml(item, method, pct);
        tooltip.style.display = "block";
        positionTooltip(tooltip, e.clientX, e.clientY);
      });
      hoverRect.addEventListener("mousemove", (e: MouseEvent) => {
        positionTooltip(tooltip, e.clientX, e.clientY);
      });
      hoverRect.addEventListener("mouseleave", () => {
        tooltip.style.display = "none";
      });
      svg.appendChild(hoverRect);
    }

    // Value annotation
    svg.appendChild(
      Object.assign(svgEl("text", {
        x: labelW + trackW + 4, y: y + barH / 2 + 4,
        fill: "#666", "font-size": 11, "text-anchor": "start",
        "pointer-events": "none",
      }), { textContent: annotationText(item, method) }),
    );
  });

  target.appendChild(svg);
}

// ---------------------------------------------------------------------------
// Annotation helpers
// ---------------------------------------------------------------------------

function annotationText(item: ParameterImportance, method: ImportanceMethod): string {
  switch (method) {
    case "lengthscale":
      return `ls=${item.raw.toFixed(3)}`;
    case "sobol":
      return `S=${item.raw.toFixed(3)}`;
    case "gradient":
      return `G=${item.raw.toFixed(3)}`;
  }
}

function tooltipHtml(item: ParameterImportance, method: ImportanceMethod, pct: number): string {
  const name = `<b>${item.paramName}</b>`;
  const impLine = `Importance: ${(pct * 100).toFixed(1)}%`;
  switch (method) {
    case "lengthscale":
      return `${name}<br>Lengthscale: ${item.raw.toFixed(4)}<br>${impLine}`;
    case "sobol":
      return `${name}<br>Sobol index: ${item.raw.toFixed(4)}<br>${impLine}`;
    case "gradient":
      return `${name}<br>Gradient magnitude: ${item.raw.toFixed(4)}<br>${impLine}`;
  }
}
