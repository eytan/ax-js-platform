# Experimental Features

These features are functional but not yet stable. APIs may change without notice.

## Client-Side Acquisition Functions (`ax-js/acquisition`)

**Status: Experimental**

ax-js includes client-side implementations of several acquisition functions for
Bayesian optimization. These are used by the live BO and BOPE demos but are
not yet integrated with the Predictor API or Ax's generation strategy.

### Available Acquisition Functions

| Function | Description |
|----------|-------------|
| `UpperConfidenceBound` | UCB(x) = μ(x) + β·σ(x) |
| `ExpectedImprovement` | Analytic EI |
| `LogExpectedImprovement` | Numerically stable log-space EI |
| `ThompsonSampling` | Random Fourier Features for posterior sampling |
| `EUBO` | Expected Utility of Best Option (for preference learning) |

### Optimizer

`optimizeAcqf` provides a simple optimizer combining random search with
projected L-BFGS (two-loop recursion, Armijo line search). This is sufficient
for low-dimensional problems but not comparable to BoTorch's MC-based
optimization on GPU.

### Utilities

- `normalPdf`, `normalCdf`, `logNormalPdf`, `logNormalCdf` — normal distribution
- `posteriorCovariance`, `posteriorMean` — raw GP posterior access
- `sampleMVN`, `Rng` — multivariate normal sampling with seeded RNG

### What's Missing

- **No integration with Predictor**: Acquisition functions operate on raw model
  objects (`GPModel` interface), not the `Predictor` API. You must use
  `loadModel()` directly.
- **No generation strategy awareness**: ax-js doesn't know which acquisition
  function Ax would use, or whether the experiment is still in the Sobol phase.
- **No multi-output acquisition**: qEHVI/qNEHVI are not implemented. These
  require MC sampling and are better handled server-side.
- **No candidate round-trip**: Generated candidates can't yet be sent back to Ax.

### Usage

```typescript
import { Predictor } from "ax-js";
import { LogExpectedImprovement, optimizeAcqf } from "ax-js/acquisition";

// Acquisition functions need raw model access (experimental)
import { loadModel } from "ax-js";

const model = loadModel(experimentState.model_state);
const acqf = new LogExpectedImprovement(model, bestObservedValue);
const result = optimizeAcqf(acqf, bounds);
// result.x = optimal point, result.value = acquisition value
```

### Browser Usage

```html
<script src="dist/ax.js"></script>
<script src="dist/ax-acquisition.js"></script>
<script>
  // Ax.acquisition.LogExpectedImprovement, Ax.acquisition.optimizeAcqf, etc.
</script>
```

Load `ax.js` first, then `ax-acquisition.js` (it extends the `Ax` namespace).
