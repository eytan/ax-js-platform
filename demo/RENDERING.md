# Demo Rendering: Training Point Relevance

## Core Concept

Every visualization context defines a **virtual point** — a location in
parameter space that the user is currently inspecting. Some dimensions of
this point may be **unspecified** (free), meaning the visualization shows
all values along those axes simultaneously.

Training point opacity is determined by **kernel distance from the virtual
point**, computed only over the specified dimensions:

```
d² = Σⱼ ((xⱼ − vⱼ) · coeffⱼ / lsⱼ)²    for j ∈ specified dims
α  = exp(−d²)
```

This single concept unifies all proximity-based rendering:

| Context | Virtual point | Unspecified dims | Effect |
|---------|--------------|------------------|--------|
| 1D slice (static) | slider values | plotted axis | Opacity based on slider proximity |
| 2D surface (static) | slider values | both axes | Opacity based on slider proximity |
| 1D slice (hover at x) | sliders + x_hover | none | Full distance, nearest-point highlight |
| Point-to-point | a training point | none | Neighbor highlighting in scatterplots |

The `pointRelevance(pt, virtualPoint, unspecifiedDims, ls, inputTf)` function
handles all cases. The only difference is which dimensions are specified.

## Distance Computation

```
d² = Σⱼ ((xⱼ − vⱼ) · coeffⱼ / lsⱼ)²    for j ∉ unspecified dims
```

Where:
- `xⱼ` is the training point's coordinate in dimension j (raw parameter space)
- `vⱼ` is the virtual point's coordinate (slider value, hover position, etc.)
- `coeffⱼ` from `input_transform.coefficient` (raw → normalized space)
- `lsⱼ` is the kernel lengthscale in dimension j (normalized space)

If no `input_transform` is present, `coeffⱼ = 1`.

## Why `exp(−d²)` Instead of `exp(−0.5·d²)`

The RBF kernel value `exp(−0.5·d²)` is too generous for visualization. In
high-D (e.g., 5 slider dims), a point 0.5 lengthscales away in each dim has
total `d ≈ 1.12`, giving `exp(−0.5·1.25) ≈ 0.54` — more than half opacity.
That point appears bright but is genuinely not on the current slice, and its
observed y-value can be far from the GP curve. This misleads the user into
thinking the model fits poorly.

The steeper `exp(−d²)` gives `0.29` for the same point — noticeably faded,
which honestly communicates that the observation was made under somewhat
different conditions.

## Opacity Rendering

- **Fill**: `rgba(255, 80, 80, α)` clamped to `[0.10, 0.85]`
- **Stroke/outline**: always visible at fixed low opacity
- **Cutoff**: points with `α < 0.03` hidden entirely (≈ 1.9 lengthscales away)

The outline always remains visible — the user sees where observations exist,
while fill intensity communicates relevance to the current context.

## Why This Works Across Lengthscale Regimes

**Short lengthscales** (GP varies rapidly): Most points are far in kernel
space → most fade. Correct — those points have near-zero influence on
predictions at the current slice.

**Long lengthscales** (GP is smooth): Most points are close → most stay
opaque. Correct — all training data contributes everywhere.

**Dimensionality-adaptive priors** (Hvarfner et al.): The lengthscale prior
accounts for dimensionality, so per-dimension scaling in `d²` produces
reasonable behavior without manual `√K` adjustments.

## Toggle

Both demos have a "nearby points only" checkbox (default: on). Unchecked
renders all points at full opacity.

## Neighbor Highlighting (Hover / Click-to-Pin)

Hovering over a training point highlights its neighbors: other points' fill
opacity reflects their kernel distance from the active point, using
`pointRelevance` with `unspecifiedDims = []` (all dimensions contribute).

**Relative scaling** is essential here because of the curse of dimensionality.
In 6D with 30 training points, even the nearest neighbor has a large total
kernel distance — using raw `exp(-d²)` would make everything invisible.
Instead, we normalize:

```
raw_αᵢ  = exp(−d²ᵢ)
relative_αᵢ = (raw_αᵢ / max(raw_α))^0.5
```

The nearest neighbor always appears bright; others fade relative to it.
The `√` (gamma=0.5) provides gentle decay so intermediate neighbors are
still visible.

**Interaction modes:**
- **Hover**: temporary neighbor highlighting, reverts on mouseleave
- **Click**: pins a point, persistent highlighting survives mouseleave
- **Click same point or empty space**: unpins, reverts to slice-based opacity
- When pinned, hover still shows tooltips but doesn't change highlighting

Note: slice-based opacity (non-neighbor mode) uses absolute `exp(-d²)` without
relative scaling — there, "all points are far" is honest information. Neighbor
mode is about showing *relative* structure, so normalization is appropriate.

## Future Extensions

### Dynamic Hover Opacity
As the user hovers across a slice plot, point opacity could update in
real-time using the fully-specified virtual point (sliders + hover position).
Points near the cursor would brighten; distant ones would fade. This makes
the kernel's local influence visible as you move through the space.
