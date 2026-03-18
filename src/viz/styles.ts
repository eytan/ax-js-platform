/**
 * CSS styles for ax-js viz components.
 *
 * Injected as a scoped <style> tag inside each container so pseudo-element
 * rules (e.g. ::-webkit-slider-thumb) work correctly and selectors win
 * specificity battles against host CSS (Jupyter, nbconvert).
 *
 * The `:scope` placeholder is replaced with `#containerId` at injection time,
 * producing selectors like `#axjs_abc .axjs-slider` with specificity (1,1,0).
 */

export const AXJS_STYLES = `
:scope .axjs-slider {
  -webkit-appearance: auto;
  appearance: auto;
  cursor: pointer;
  touch-action: none;
  pointer-events: auto;
  user-select: auto;
  -webkit-user-select: auto;
  flex: 1;
  min-width: 100px;
  accent-color: #7c6ff7;
}
:scope .axjs-slider::-webkit-slider-thumb {
  -webkit-appearance: auto;
  cursor: grab;
  pointer-events: auto;
}
:scope .axjs-slider:active::-webkit-slider-thumb {
  cursor: grabbing;
}
:scope .axjs-slider::-webkit-slider-runnable-track {
  pointer-events: auto;
}
:scope .axjs-select {
  pointer-events: auto;
  cursor: pointer;
  background: #1a1a1d;
  color: #e0e0e0;
  border: 1px solid #333;
  border-radius: 4px;
  padding: 3px 8px;
  font-size: 12px;
  -webkit-user-drag: none;
}
:scope .axjs-tooltip {
  position: fixed;
  display: none;
  background: rgba(20,20,24,0.95);
  border: 1px solid #444;
  border-radius: 6px;
  padding: 8px 12px;
  font-size: 12px;
  color: #e0e0e0;
  pointer-events: none;
  z-index: 10000;
  white-space: nowrap;
}
:scope .axjs-ctrl-row {
  display: flex;
  gap: 10px;
  align-items: center;
  margin-bottom: 6px;
  max-width: 600px;
  pointer-events: auto;
  user-select: auto;
  -webkit-user-select: auto;
}
:scope .axjs-slider-label {
  font-size: 13px;
  color: #888;
  min-width: 140px;
  text-align: right;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
:scope .axjs-slider-value {
  font-size: 13px;
  font-weight: 500;
  color: #ccc;
  min-width: 70px;
  text-align: right;
}
:scope .axjs-controls {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  align-items: center;
  margin-bottom: 8px;
  pointer-events: auto;
}
`;

/**
 * Inject a `<style>` element inside `container`, scoped to the container's id.
 *
 * Replaces `:scope` in AXJS_STYLES with `#containerId`, producing selectors
 * like `#axjs_abc .axjs-slider` that beat any class-only host CSS.
 */
export function injectScopedStyles(container: HTMLElement): void {
  if (typeof document === "undefined") return;
  if (!container.id) container.id = "axjs_" + Math.random().toString(36).slice(2, 10);
  if (container.querySelector("style[data-axjs]")) return;
  const s = document.createElement("style");
  s.setAttribute("data-axjs", "");
  s.textContent = AXJS_STYLES.replace(/:scope/g, `#${container.id}`);
  container.prepend(s);
}

/** @deprecated Use `injectScopedStyles(container)` instead. */
export function injectStyles(): void {
  if (typeof document === "undefined") return;
  injectScopedStyles(document.body);
}
