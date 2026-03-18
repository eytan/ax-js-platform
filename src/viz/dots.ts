import type { DotInfo, RenderPredictor } from "./types";
import { positionTooltip } from "./widgets";

export type { DotInfo };

/** Compute kernel correlations from a reference point to all dots. */
export function computeKernelRels(
  predictor: RenderPredictor,
  dots: DotInfo[],
  activeIdx: number,
  outcome: string,
): { raw: number[]; max: number } {
  const rawRels: number[] = [];
  let maxRel = 0;
  for (let i = 0; i < dots.length; i++) {
    if (i === activeIdx) { rawRels.push(1); continue; }
    const r = predictor.kernelCorrelation(dots[i].pt, dots[activeIdx].pt, outcome);
    rawRels.push(r);
    if (r > maxRel) maxRel = r;
  }
  return { raw: rawRels, max: maxRel };
}

/** Apply kernel-distance-based highlight styling to SVG dot elements. */
export function applyDotHighlight(
  dots: DotInfo[],
  activeIdx: number,
  rels: { raw: number[]; max: number },
): void {
  for (let i = 0; i < dots.length; i++) {
    const d = dots[i];
    if (i === activeIdx) {
      d.el.setAttribute("fill", "rgba(255,80,80,0.95)");
      d.el.setAttribute("stroke", "rgba(255,255,255,1)");
      d.el.setAttribute("stroke-width", "2");
      d.el.setAttribute("r", String(d.defaultR + 2));
      if (d.whisker) d.whisker.setAttribute("stroke", "rgba(255,80,80,0.5)");
    } else {
      const relNorm = rels.max > 0 ? rels.raw[i] / rels.max : 0;
      const fa = Math.max(0.08, Math.min(0.90, Math.sqrt(relNorm)));
      d.el.setAttribute("fill", `rgba(255,80,80,${fa.toFixed(3)})`);
      d.el.setAttribute("stroke", `rgba(255,255,255,${Math.max(0.15, fa * 0.6).toFixed(3)})`);
      d.el.setAttribute("stroke-width", "1");
      d.el.setAttribute("r", String(d.defaultR));
      if (d.whisker) d.whisker.setAttribute("stroke", `rgba(255,80,80,${(fa * 0.35).toFixed(3)})`);
    }
  }
}

/** Reset all dots to their default style. */
export function clearDotHighlight(dots: DotInfo[]): void {
  for (const d of dots) {
    d.el.setAttribute("fill", d.defaultFill);
    d.el.setAttribute("stroke", d.defaultStroke);
    d.el.setAttribute("stroke-width", "1");
    d.el.setAttribute("r", String(d.defaultR));
    if (d.whisker) d.whisker.setAttribute("stroke", "rgba(124,154,255,0.3)");
  }
}

/** Find nearest dot within a pixel radius. Returns index or -1. */
export function findNearestDot(dots: DotInfo[], px: number, py: number, maxDist = 12): number {
  let best = -1;
  let bestD = maxDist * maxDist;
  for (let i = 0; i < dots.length; i++) {
    const dx = px - dots[i].cx;
    const dy = py - dots[i].cy;
    const d2 = dx * dx + dy * dy;
    if (d2 < bestD) { bestD = d2; best = i; }
  }
  return best;
}

/** Build rich tooltip HTML showing all outcomes and parameters for a training point. */
export function buildPointTooltipHtml(
  predictor: RenderPredictor,
  idx: number,
  selectedOutcome: string,
): string {
  let html = `<b>Trial ${idx}</b><br>`;
  for (const name of predictor.outcomeNames) {
    const td = predictor.getTrainingData(name);
    if (idx >= td.Y.length) continue;
    const isSel = name === selectedOutcome;
    const color = isSel ? "#7c9aff" : "#888";
    html += `${isSel ? "<b>" : ""}${name} = <span style="color:${color}">${td.Y[idx].toFixed(4)}</span>${isSel ? "</b>" : ""}<br>`;
  }
  html += '<hr style="border-color:#333;margin:4px 0">';
  const td0 = predictor.getTrainingData(predictor.outcomeNames[0]);
  if (idx < td0.X.length) {
    for (let j = 0; j < predictor.paramNames.length; j++) {
      html += `<span style="color:#888">${predictor.paramNames[j]}</span> = ${td0.X[idx][j].toFixed(4)}<br>`;
    }
  }
  return html;
}

/**
 * Attach click-to-pin and hover-highlight handlers to an SVG for training dots.
 * This is the shared interactivity layer used by slice, CV, and trace plots.
 */
export function attachDotInteractivity(
  svg: SVGSVGElement,
  dots: DotInfo[],
  predictor: RenderPredictor,
  outcome: string,
  tooltip: HTMLDivElement,
  tooltipContainer: HTMLElement,
  options?: {
    fallbackMouseMove?: (e: MouseEvent) => void;
    onPinChange?: (pinnedIdx: number) => void;
    getPinnedIdx?: () => number;
    setPinnedIdx?: (idx: number) => void;
  },
): { getPinnedIdx: () => number } {
  let localPinnedIdx = -1;
  let hoverHighlight = false;

  const getPinnedIdx = options?.getPinnedIdx ?? (() => localPinnedIdx);
  const setPinnedIdx = options?.setPinnedIdx ?? ((v: number) => { localPinnedIdx = v; });

  svg.addEventListener("mousemove", (e: MouseEvent) => {
    const rect = svg.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const py = e.clientY - rect.top;
    const hitIdx = findNearestDot(dots, px, py);

    if (hitIdx >= 0) {
      svg.style.cursor = "pointer";
      if (getPinnedIdx() < 0) {
        applyDotHighlight(dots, hitIdx, computeKernelRels(predictor, dots, hitIdx, outcome));
        hoverHighlight = true;
      }
      const html = buildPointTooltipHtml(predictor, dots[hitIdx].idx, outcome);
      tooltip.innerHTML = html;
      tooltip.style.display = "block";
      positionTooltip(tooltip, e.clientX, e.clientY);
    } else {
      svg.style.cursor = "crosshair";
      if (getPinnedIdx() < 0 && hoverHighlight) {
        clearDotHighlight(dots);
        hoverHighlight = false;
      }
      if (options?.fallbackMouseMove) {
        options.fallbackMouseMove(e);
      } else {
        tooltip.style.display = "none";
      }
    }
  });

  svg.addEventListener("click", (e: MouseEvent) => {
    const rect = svg.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const py = e.clientY - rect.top;
    const hitIdx = findNearestDot(dots, px, py);

    if (hitIdx >= 0) {
      const hitTrainIdx = dots[hitIdx].idx;
      if (getPinnedIdx() === hitTrainIdx) {
        setPinnedIdx(-1);
        clearDotHighlight(dots);
      } else {
        setPinnedIdx(hitTrainIdx);
        applyDotHighlight(dots, hitIdx, computeKernelRels(predictor, dots, hitIdx, outcome));
      }
    } else {
      if (getPinnedIdx() >= 0) {
        setPinnedIdx(-1);
        clearDotHighlight(dots);
      }
    }
    hoverHighlight = false;
    options?.onPinChange?.(getPinnedIdx());
  });

  svg.addEventListener("mouseleave", () => {
    svg.style.cursor = "crosshair";
    tooltip.style.display = "none";
    if (getPinnedIdx() < 0 && hoverHighlight) {
      clearDotHighlight(dots);
      hoverHighlight = false;
    }
  });

  return { getPinnedIdx };
}
