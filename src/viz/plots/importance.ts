import type { RenderPredictor, FeatureImportanceOptions } from "../types";
import { svgEl } from "./_svg";
import { createOutcomeSelector, createTooltipDiv, positionTooltip, removeTooltip, makeSelectEl } from "../widgets";
import { injectScopedStyles } from "../styles";

const CTRL_CSS = "display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-bottom:8px;pointer-events:auto";

/**
 * Render a horizontal bar chart of feature importance into a container.
 *
 * Each bar shows `1 / lengthscale` (normalized to the most important
 * dimension). Longer bars = more sensitive. Sorted by importance.
 */
export function renderFeatureImportance(
  container: HTMLElement,
  predictor: RenderPredictor,
  options?: FeatureImportanceOptions,
): void {
  const interactive = options?.interactive !== false;

  if (!interactive) {
    renderFeatureImportanceStatic(container, predictor, options?.outcome ?? predictor.outcomeNames[0]);
    return;
  }

  if (!container.id) container.id = "axjs_" + Math.random().toString(36).slice(2, 10);
  removeTooltip(container.id);
  container.innerHTML = "";
  injectScopedStyles(container);
  let selectedOutcome = options?.outcome ?? predictor.outcomeNames[0];
  const tooltip = createTooltipDiv(container.id);

  const controlsDiv = document.createElement("div");
  controlsDiv.style.cssText = CTRL_CSS;
  const plotsDiv = document.createElement("div");
  container.appendChild(controlsDiv);
  container.appendChild(plotsDiv);

  if (predictor.outcomeNames.length > 1) {
    const { wrapper, select } = makeSelectEl("Outcome:");
    createOutcomeSelector(predictor, select, (name) => {
      selectedOutcome = name;
      redraw();
    });
    controlsDiv.appendChild(wrapper);
  }

  function redraw() {
    plotsDiv.innerHTML = "";
    renderFeatureImportanceStatic(plotsDiv, predictor, selectedOutcome, tooltip, container);
  }
  redraw();
}

function renderFeatureImportanceStatic(
  target: HTMLElement,
  predictor: RenderPredictor,
  outcome: string,
  tooltip?: HTMLDivElement,
  tooltipContainer?: HTMLElement,
): void {
  const ranked = predictor.rankDimensionsByImportance(outcome);
  if (!ranked || ranked.length === 0) {
    target.textContent = "No lengthscale data";
    return;
  }

  const barColors = ["#7c6ff7", "#6fa0f7", "#6fcff7", "#6ff7c8", "#a0f76f", "#f7e06f", "#f7a06f", "#f76f6f"];
  const importances = ranked.map((d) => 1 / d.lengthscale);
  const maxImp = Math.max(...importances);

  const W = Math.min((tooltipContainer ?? target).clientWidth || 500, 500);
  const labelW = 130;
  const valueW = 70;
  const barH = 24;
  const rowGap = 6;
  const H = ranked.length * (barH + rowGap) + 8;
  const trackW = W - labelW - valueW;

  const svg = svgEl("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}` });

  ranked.forEach((dim, i) => {
    const y = i * (barH + rowGap) + 4;
    const pct = importances[i] / maxImp;
    const barW = Math.max(2, pct * trackW);

    // Label
    svg.appendChild(
      Object.assign(svgEl("text", {
        x: labelW - 8, y: y + barH / 2 + 4,
        fill: "#ccc", "font-size": 13, "text-anchor": "end",
      }), { textContent: dim.paramName }),
    );

    // Track
    svg.appendChild(svgEl("rect", {
      x: labelW, y, width: trackW, height: barH,
      rx: 4, fill: "#1a1a1d",
    }));

    // Fill bar
    svg.appendChild(svgEl("rect", {
      x: labelW, y, width: barW, height: barH,
      rx: 4, fill: barColors[dim.dimIndex % barColors.length],
    }));

    // Tooltip on bar hover
    if (tooltip) {
      const hoverRect = svgEl("rect", {
        x: labelW, y, width: trackW, height: barH,
        fill: "transparent", cursor: "pointer",
      });
      hoverRect.addEventListener("mouseenter", (e: MouseEvent) => {
        tooltip.innerHTML = `<b>${dim.paramName}</b><br>Lengthscale: ${dim.lengthscale.toFixed(4)}<br>Importance: ${(pct * 100).toFixed(1)}%`;
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
        fill: "#999", "font-size": 11, "text-anchor": "start",
        "pointer-events": "none",
      }), { textContent: `ls=${dim.lengthscale.toFixed(3)}` }),
    );
  });

  target.appendChild(svg);
}
