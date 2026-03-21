// Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

import type { RGB } from "./types";

/**
 * Draw a training-data point with the standard outer-ring + inner-fill style.
 *
 * @param ctx - Canvas 2D rendering context.
 * @param x - Pixel x coordinate.
 * @param y - Pixel y coordinate.
 * @param alpha - Opacity in [0, 1] (distance-based fade).
 * @param isActive - Whether the point is click-pinned (larger, full opacity).
 * @param isHovered - Whether the mouse is hovering (larger).
 * @param fillRGB - Inner fill color as [r, g, b]. Defaults to red [255, 60, 60].
 */
export function drawDataDot(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  alpha: number,
  isActive: boolean,
  isHovered: boolean,
  fillRGB: RGB = [217, 95, 78],
): void {
  if (alpha < 0.04) {
    return;
  }
  const outerR = isActive || isHovered ? 7.5 : 5;
  const innerR = isActive || isHovered ? 4 : 2.5;
  ctx.beginPath();
  ctx.arc(x, y, outerR, 0, 2 * Math.PI);
  ctx.strokeStyle = isActive
    ? "rgba(68,68,68,1)"
    : `rgba(68,68,68,${Math.max(0.15, alpha * 0.6).toFixed(3)})`;
  ctx.lineWidth = isActive ? 2.5 : isHovered ? 2 : 1.5;
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(x, y, innerR, 0, 2 * Math.PI);
  ctx.fillStyle =
    isActive || isHovered
      ? `rgba(${fillRGB[0]},${fillRGB[1]},${fillRGB[2]},1)`
      : `rgba(${fillRGB[0]},${fillRGB[1]},${fillRGB[2]},${alpha.toFixed(3)})`;
  ctx.fill();
}
