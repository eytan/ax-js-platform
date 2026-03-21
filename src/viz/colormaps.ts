// Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

import type { RGB } from "./types";

const VIRIDIS_STOPS: Array<RGB> = [
  [68, 1, 84],
  [72, 32, 111],
  [63, 64, 153],
  [50, 101, 176],
  [38, 130, 142],
  [63, 151, 120],
  [92, 170, 98],
  [140, 188, 80],
  [195, 203, 72],
  [253, 231, 37],
];

const PLASMA_STOPS: Array<RGB> = [
  [13, 8, 135],
  [75, 3, 161],
  [125, 3, 168],
  [168, 34, 150],
  [203, 70, 121],
  [229, 107, 93],
  [245, 144, 66],
  [252, 180, 36],
  [241, 229, 29],
];

function interpolateStops(t: number, stops: Array<RGB>): RGB {
  t = Math.max(0, Math.min(1, t));
  const idx = t * (stops.length - 1);
  const lo = Math.floor(idx);
  const hi = Math.min(lo + 1, stops.length - 1);
  const f = idx - lo;
  return [
    Math.round(stops[lo][0] + f * (stops[hi][0] - stops[lo][0])),
    Math.round(stops[lo][1] + f * (stops[hi][1] - stops[lo][1])),
    Math.round(stops[lo][2] + f * (stops[hi][2] - stops[lo][2])),
  ];
}

/** Viridis colormap. Maps t in [0, 1] to an RGB triplet. */
export function viridis(t: number): RGB {
  return interpolateStops(t, VIRIDIS_STOPS);
}

/** Plasma colormap. Maps t in [0, 1] to an RGB triplet. */
export function plasma(t: number): RGB {
  return interpolateStops(t, PLASMA_STOPS);
}

const PIYG_STOPS: Array<RGB> = [
  [140, 0, 75],
  [212, 0, 112],
  [238, 65, 158],
  [250, 148, 203],
  [254, 214, 233],
  [247, 247, 247],
  [216, 248, 190],
  [168, 232, 88],
  [98, 198, 30],
  [45, 162, 8],
  [15, 115, 0],
];

/** PiYG divergent colormap. Pink → grey → green, grey at t=0.5 (zero change). */
export function piYG(t: number): RGB {
  return interpolateStops(t, PIYG_STOPS);
}

/**
 * Render a horizontal colorbar into a canvas element.
 * @param canvasId - DOM id of the `<canvas>` element.
 * @param colorFn - Colormap function mapping [0,1] to RGB.
 */
export function drawColorbar(canvasId: string, colorFn: (t: number) => RGB): void {
  const cvs = document.getElementById(canvasId) as HTMLCanvasElement | null;
  if (!cvs) {
    return;
  }
  cvs.width = cvs.offsetWidth || 200;
  cvs.height = cvs.offsetHeight || 24;
  const ctx = cvs.getContext("2d");
  if (!ctx) {
    return;
  }
  const w = cvs.width;
  const h = cvs.height;
  for (let i = 0; i < w; i++) {
    const rgb = colorFn(i / w);
    ctx.fillStyle = `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;
    ctx.fillRect(i, 0, 1, h);
  }
}

/**
 * Render a 2D heatmap onto a canvas context from a flat array of values.
 */
export function renderHeatmap(
  ctx: CanvasRenderingContext2D,
  values: Array<number>,
  gridW: number,
  gridH: number,
  canvasW: number,
  canvasH: number,
  colorFn: (t: number) => RGB,
  minVal: number,
  maxVal: number,
): void {
  const img = ctx.createImageData(canvasW, canvasH);
  const range = maxVal - minVal || 1;
  const cellW = canvasW / gridW;
  const cellH = canvasH / gridH;
  for (let k = 0; k < values.length; k++) {
    const gi = k % gridW;
    const gj = Math.floor(k / gridW);
    const t = Math.max(0, Math.min(1, (values[k] - minVal) / range));
    const rgb = colorFn(t);
    const x0 = Math.round(gi * cellW);
    const y0 = Math.round(gj * cellH);
    const x1 = Math.round((gi + 1) * cellW);
    const y1 = Math.round((gj + 1) * cellH);
    for (let py = y0; py < y1; py++) {
      for (let px = x0; px < x1; px++) {
        const idx = (py * canvasW + px) * 4;
        img.data[idx] = rgb[0];
        img.data[idx + 1] = rgb[1];
        img.data[idx + 2] = rgb[2];
        img.data[idx + 3] = 255;
      }
    }
  }
  ctx.putImageData(img, 0, 0);
}
