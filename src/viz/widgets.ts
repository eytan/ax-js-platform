import type { ParamSpec, EmbeddingPredictor } from "./types";
import { isChoice, isInteger, formatParamValue } from "./params";
import { injectScopedStyles } from "./styles";

/**
 * Populate a `<select>` element with the predictor's outcome names.
 *
 * Clears existing options, adds one `<option>` per outcome, and selects
 * the first. Attaches a `change` listener that calls `onChange(selectedName)`.
 * Safe to call repeatedly — replaces the previous listener each time by
 * cloning the element's event handlers.
 */
export function createOutcomeSelector(
  predictor: EmbeddingPredictor,
  selectEl: HTMLSelectElement,
  onChange: (name: string) => void,
): void {
  selectEl.innerHTML = "";
  predictor.outcomeNames.forEach((name) => {
    const opt = document.createElement("option");
    opt.value = name;
    opt.textContent = name;
    selectEl.appendChild(opt);
  });
  selectEl.onchange = () => onChange(selectEl.value);
}

/**
 * Build parameter sliders for non-plotted dimensions inside `container`.
 *
 * For each dimension not in `excludeDims`, creates a range slider (or
 * dropdown for choice parameters). Calls `onChange(dimIndex, newValue)`
 * whenever a slider value changes.
 */
export function createParamSliders(
  predictor: EmbeddingPredictor,
  params: ParamSpec[],
  container: HTMLElement,
  currentValues: (number | string | boolean)[],
  onChange: (dimIndex: number, value: number | string | boolean) => void,
  options?: { excludeDims?: Set<number>; dimOrder?: number[] },
): void {
  container.innerHTML = "";
  // Inject scoped styles into nearest ancestor with an id, or the container itself
  const styleTarget = container.closest("[id]") as HTMLElement ?? container;
  injectScopedStyles(styleTarget);
  const excludeDims = options?.excludeDims ?? new Set<number>();
  const order =
    options?.dimOrder ??
    Array.from({ length: predictor.paramNames.length }, (_, i) => i);

  order.forEach((i) => {
    if (excludeDims.has(i)) return;
    const name = predictor.paramNames[i];
    const p = params[i];
    const row = document.createElement("div");
    row.className = "axjs-ctrl-row";
    // Inline styles to override any host user-select/pointer-events cascade
    row.style.cssText =
      "display:flex;gap:10px;align-items:center;margin-bottom:6px;" +
      "max-width:600px;pointer-events:auto;user-select:auto;-webkit-user-select:auto;";
    const lbl = document.createElement("span");
    lbl.className = "axjs-slider-label";
    lbl.textContent = name;
    lbl.title = name;

    const val = document.createElement("span");
    val.className = "axjs-slider-value";
    val.textContent = formatParamValue(currentValues[i] as number, p);

    if (isChoice(p)) {
      const sel = document.createElement("select");
      sel.className = "axjs-select";
      p.values!.forEach((v) => {
        const o = document.createElement("option");
        o.value = String(v);
        o.textContent = String(v);
        if (v == currentValues[i]) o.selected = true;
        sel.appendChild(o);
      });
      sel.addEventListener("change", () => {
        const nv = +sel.value;
        currentValues[i] = nv;
        val.textContent = formatParamValue(nv, p);
        onChange(i, nv);
      });
      row.appendChild(lbl);
      row.appendChild(sel);
      row.appendChild(val);
    } else {
      const lo = predictor.paramBounds[i][0];
      const hi = predictor.paramBounds[i][1];
      const sl = document.createElement("input");
      sl.type = "range";
      sl.className = "axjs-slider";
      // Inline critical styles — immune to any host CSS specificity battle
      sl.style.cssText =
        "-webkit-appearance:auto;appearance:auto;cursor:pointer;" +
        "touch-action:none;pointer-events:auto;user-select:auto;" +
        "-webkit-user-select:auto;flex:1;min-width:100px;accent-color:#4872f9;";
      sl.min = String(lo);
      sl.max = String(hi);
      sl.step = isInteger(p) ? "1" : String((hi - lo) / 200);
      sl.value = String(currentValues[i]);
      // Stop propagation so notebook/nbconvert drag handlers don't intercept
      sl.addEventListener("pointerdown", (e: Event) => { e.stopPropagation(); });
      sl.addEventListener("mousedown", (e: Event) => { e.stopPropagation(); });
      sl.addEventListener("touchstart", (e: Event) => { e.stopPropagation(); });
      sl.addEventListener("input", () => {
        const nv = isInteger(p) ? Math.round(+sl.value) : +sl.value;
        currentValues[i] = nv;
        val.textContent = formatParamValue(nv, p);
        onChange(i, nv);
      });
      row.appendChild(lbl);
      row.appendChild(sl);
      row.appendChild(val);
    }
    container.appendChild(row);
  });
}

/**
 * Wire up a `<input type="file">` element to parse JSON and invoke a callback.
 */
export function setupFileUpload(
  inputId: string,
  callback: (data: unknown) => void,
): void {
  const input = document.getElementById(inputId) as HTMLInputElement | null;
  if (!input) return;
  input.addEventListener("change", () => {
    const file = input.files?.[0];
    if (!file) return;
    file.text().then((text) => callback(JSON.parse(text)));
  });
}

/** Show a tooltip element at the given screen coordinates. */
export function showTooltip(
  el: HTMLElement,
  html: string,
  screenX: number,
  screenY: number,
): void {
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
  if (containerId) removeTooltip(containerId);
  const tooltip = document.createElement("div");
  tooltip.className = "axjs-tooltip";
  if (containerId) tooltip.setAttribute("data-axjs-for", containerId);
  // Inline all styles — tooltip lives in body, outside any scoped CSS container
  tooltip.style.cssText =
    "position:fixed;display:none;background:rgba(255,255,255,0.97);" +
    "border:1px solid #d0d0d0;border-radius:6px;padding:8px 12px;" +
    "font-size:12px;color:#333;pointer-events:none;" +
    "z-index:10000;white-space:nowrap;";
  document.body.appendChild(tooltip);
  return tooltip;
}

/**
 * Position a fixed-position tooltip near the cursor.
 * Uses viewport coordinates directly — no container-relative math needed.
 */
export function positionTooltip(tooltip: HTMLDivElement, clientX: number, clientY: number): void {
  tooltip.style.left = (clientX + 16) + "px";
  tooltip.style.top = (clientY - 10) + "px";
}

/**
 * Remove a body-level tooltip associated with a container id.
 * Call before clearing a container to prevent orphaned tooltips.
 */
export function removeTooltip(containerId: string): void {
  const stale = document.body.querySelector(`div[data-axjs-for="${containerId}"]`);
  if (stale) stale.remove();
}

export function makeSelectEl(label: string): { wrapper: HTMLDivElement; select: HTMLSelectElement } {
  const wrapper = document.createElement("div");
  wrapper.style.cssText = "display:flex;align-items:center;gap:4px";
  const lbl = document.createElement("span");
  lbl.style.cssText = "color:#666;font-size:12px";
  lbl.textContent = label;
  const select = document.createElement("select");
  select.className = "axjs-select";
  wrapper.appendChild(lbl);
  wrapper.appendChild(select);
  return { wrapper, select };
}
