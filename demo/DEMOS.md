# axjs Demo Suite

Five self-contained HTML demos built by `node demo/build_demos.js`.
Each inlines the axjs IIFE bundle and default fixture data — no server required.

## Architecture

```
build_demos.js          # Node.js build script
├── Shared code blocks
│   ├── sharedUtilsCode      — normalizeFixture, closestToCenter, formatParamValue, etc.
│   └── sharedColormapCode   — viridis(), plasma(), drawColorbar()
├── Template literals (one per demo)
│   └── Each produces a self-contained HTML file
└── writeFileSync × 5
```

**Data flow**: `test/fixtures/*.json` → `fixtureScript()` embeds as `__DEFAULT_FIXTURE__` → `loadFixtureData()` at runtime → `Predictor.predict()`.

## Demos

### 1. slice_plot — 1D Posterior Slices

- **Default fixture**: `hartmann_mixed.json`
- **Features**: File upload, outcome selector, dimension sliders, training point overlay with kernel-distance relevance, stable y-axis range via `precomputeYRange()`
- **Rendering**: Observable Plot (SVG)

### 2. response_surface — 2D Heatmaps

- **Default fixture**: `penicillin_modellist.json`
- **Features**: File upload, axis selectors, outcome selector, contour toggle, nearby-points filter, colorbar (viridis/plasma), tooltip with coordinates
- **Rendering**: Canvas (pixel-level), shared colormaps

### 3. radar — Multi-Objective Radar

- **Default fixture**: VSIP (built at build-time by `buildDefaultRadarFixture()`)
- **Features**: File upload, generic fixture-driven rendering, 95% CI band toggle
- **Data-driven**: Reads `optimization_config` from fixture to populate objectives/constraints. Falls back to treating all outcomes as objectives if no `optimization_config`.
- **Left panel**: Stat cards (objectives + constraints), P(feasibility) bar
- **Supports**: LEQ and GEQ constraints, minimize/maximize objectives
- **Rendering**: Canvas radar chart

### 4. scatteroid — Relativized Scatter

- **Default fixture**: None (generates VSIP synthetic data on-the-fly)
- **Features**: Relativized predictions vs status quo, candidate highlighting, deltoid bars, Pareto front overlay
- **Note**: VSIP code stays inline (only demo that generates data at runtime)
- **Rendering**: Canvas

### 5. point_proximity — Opacity Diagnostic Tool

- **Default fixture**: None (generates synthetic Ackley data on-the-fly via Predictor + input_transform)
- **Features**: Configurable dimensionality (3–20), distance mode (kernel vs euclidean), normalization, formula variants (exp(-d²), exp(-0.5d²) [RBF], exp(-d), exp(-2d²)), opacity mapping (linear/sqrt/relative)
- **Purpose**: Diagnostic tool for evaluating and comparing training point opacity formulas across varying dimensionality
- **Key detail**: Uses isotropic LS=0.5 in normalized [0,1] space with Normalize input_transform, matching real Ax exports
- **Rendering**: Canvas (1D slice + 2D heatmap + 2 histograms)

## Shared Utilities Reference

### `sharedUtilsCode` (inlined via `sharedUtilsScript()`)

| Function | Purpose |
|----------|---------|
| `normalizeFixture(data)` | Unwraps fixture format → flat `{search_space, model_state, metadata, ...}` |
| `isChoice(p)`, `isInteger(p)` | Parameter type checks |
| `defaultParamValue(p)` | Center of bounds / first choice value |
| `formatParamValue(val, p)` | Display formatting |
| `findLS(k)`, `extractLengthscales(ms)` | Recursive lengthscale extraction from kernel tree |
| `rankDims(ms, ...)` | Rank dimensions by lengthscale (shorter = more important) |
| `closestToCenter(ms, bounds, params)` | Find training point nearest to design space center |
| `getTrainData(ms, outcomeNames, sel)` | Extract train_X/train_Y for a specific outcome |
| `pointRelevance(pt, ...)` | Kernel-distance-based training point relevance |

### `sharedColormapCode` (inlined via `sharedColormapScript()`)

| Function | Purpose |
|----------|---------|
| `viridis(t)` | Viridis colormap (10-stop piecewise linear, t ∈ [0,1]) |
| `plasma(t)` | Plasma colormap (9-stop piecewise linear) |
| `drawColorbar(id, cfn)` | Render horizontal colorbar to a canvas element |

## Key Visualization Concepts

- **Normalization bounds**: Computed by sampling GP predictions across the search space (not from ground truth). This makes demos fully generic — no `evaluate()` function needed.
- **Fixture-driven**: Demos load `ExperimentState` JSON (with optional `optimization_config`). The same demo works with any Ax experiment export.
- **Build-time fixtures**: Some default fixtures are built at build-time in `build_demos.js` (e.g., VSIP for radar) to avoid runtime model construction.

## Adding a New Demo

1. Add a template literal to `build_demos.js`: `const myDemo = \`<!DOCTYPE html>...\`;`
2. Include `${libraryScript()}` and optionally `${sharedUtilsScript()}`, `${sharedColormapScript()}`
3. For fixture-driven demos: `${fixtureScript('__DEFAULT_FIXTURE__', myFixture)}` + `loadFixtureData(__DEFAULT_FIXTURE__)`
4. Add `writeFileSync(join(__dirname, 'my_demo.html'), myDemo);`
5. Add a card to `index.html`
6. Update this file
