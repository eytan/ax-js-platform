// Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

/** Show a tooltip element at the given screen coordinates. */
export function showTooltip(el: HTMLElement, html: string, screenX: number, screenY: number): void {
  el.innerHTML = html;
  el.style.display = "block";
  el.style.left = screenX + 16 + "px";
  el.style.top = screenY - 10 + "px";
}

/** Hide a tooltip element. */
export function hideTooltip(el: HTMLElement): void {
  el.style.display = "none";
}

/**
 * Create a body-level tooltip div with `position: fixed`.
 *
 * Appends to `document.body` to avoid overflow clipping from any ancestor.
 * Tagged with `data-axjs-for="containerId"` for cleanup via `removeTooltip`.
 */
export function createTooltipDiv(containerId?: string): HTMLDivElement {
  // Remove any stale tooltip for this container
  if (containerId) {
    removeTooltip(containerId);
  }
  const tooltip = document.createElement("div");
  tooltip.className = "axjs-tooltip";
  if (containerId) {
    tooltip.dataset.axjsFor = containerId;
  }
  // Inline all styles — tooltip lives in body, outside any scoped CSS container
  tooltip.style.cssText =
    "position:fixed;display:none;background:rgba(255,255,255,0.97);" +
    "border:1px solid #d0d0d0;border-radius:6px;padding:8px 12px;" +
    "font-size:12px;color:#333;pointer-events:none;" +
    "z-index:10000;white-space:nowrap;";
  document.body.append(tooltip);
  return tooltip;
}

/**
 * Position a fixed-position tooltip near the cursor.
 * Uses viewport coordinates directly — no container-relative math needed.
 */
export function positionTooltip(tooltip: HTMLDivElement, clientX: number, clientY: number): void {
  tooltip.style.left = clientX + 16 + "px";
  tooltip.style.top = clientY - 10 + "px";
}

/**
 * Remove a body-level tooltip associated with a container id.
 * Call before clearing a container to prevent orphaned tooltips.
 */
export function removeTooltip(containerId: string): void {
  const stale = document.body.querySelector(`div[data-axjs-for="${containerId}"]`);
  if (stale) {
    stale.remove();
  }
}
