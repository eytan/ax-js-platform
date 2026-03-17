"""Shared helpers for fixture generation.

Extracted from generate_fixtures.py to eliminate duplication:
- build_warp_transform: ChainedInputTransform(Normalize + Warp) — was copy-pasted 4x
- build_composite_kernel: ScaleKernel(ProductKernel(cont, cat)) — was copy-pasted 2x
- get_benchmark_data, discretize_dims, round_to_int_dims: data prep utilities
- wrap_fixture: shared serialization wrapper all generators call
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from generate_fixtures import FixtureSpec


# ── Transform builders ──────────────────────────────────────────────────────

def build_warp_transform(d: int, warp_indices: list[int] | None = None):
    """Build ChainedInputTransform(Normalize + Warp).

    Args:
        d: Total input dimension (including task column for MultiTaskGP).
        warp_indices: Which dims to warp. Defaults to all dims (range(d)).
    """
    from botorch.models.transforms.input import Warp, Normalize, ChainedInputTransform

    if warp_indices is None:
        warp_indices = list(range(d))
    return ChainedInputTransform(
        normalize=Normalize(d=d),
        warp=Warp(d=d, indices=warp_indices),
    )


def build_composite_kernel(
    kernel_type: str,
    nu: float,
    cont_dims: list[int],
    cat_dims: list[int],
    cat_cardinality: int | None = None,
):
    """Build ScaleKernel(ProductKernel(continuous_kernel, CategoricalKernel)).

    Args:
        kernel_type: "Matern" or "RBF".
        nu: Matern smoothness (ignored for RBF).
        cont_dims: Indices of continuous dimensions.
        cat_dims: Indices of categorical dimensions.
        cat_cardinality: Not used in kernel construction, kept for API symmetry.
    """
    from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel, ProductKernel
    try:
        from gpytorch.kernels import CategoricalKernel
    except ImportError:
        from botorch.models.kernels import CategoricalKernel

    if kernel_type == "RBF":
        cont_kernel = RBFKernel(ard_num_dims=len(cont_dims), active_dims=cont_dims)
    else:
        cont_kernel = MaternKernel(
            nu=nu, ard_num_dims=len(cont_dims), active_dims=cont_dims
        )
    cat_kernel = CategoricalKernel(
        ard_num_dims=len(cat_dims), active_dims=cat_dims
    )
    return ScaleKernel(ProductKernel(cont_kernel, cat_kernel))


# ── Data utilities ──────────────────────────────────────────────────────────

def get_benchmark_data(
    spec: FixtureSpec,
    benchmarks: dict,
) -> tuple[Tensor, Tensor, Tensor]:
    """Generate train_X, train_Y, test_X from a benchmark function.

    For synthetic (PairwiseGP comparisons), returns train_X and test_X only
    with train_Y as dummy zeros (PairwiseGP doesn't use Y directly).

    Uses spec.negate_benchmark to control Y sign. When False, positive Y
    is needed (e.g. for Log outcome transform which requires Y > 0).
    """
    from benchmarks import make_benchmarks, sample_bounded

    if spec.benchmark == "synthetic":
        torch.manual_seed(spec.seed)
        d = 2
        train_X = torch.rand(spec.n_train, d, dtype=torch.float64)
        test_X = torch.rand(spec.n_test, d, dtype=torch.float64)
        train_Y = torch.zeros(spec.n_train, 1, dtype=torch.float64)
        return train_X, train_Y, test_X

    if not spec.negate_benchmark:
        local_benchmarks = make_benchmarks(negate=False)
        fn_factory = local_benchmarks.get(spec.benchmark)
    else:
        fn_factory = benchmarks.get(spec.benchmark)
    if fn_factory is None:
        raise ValueError(
            f"Unknown benchmark '{spec.benchmark}'. "
            f"Available: {list(benchmarks.keys())}"
        )

    fn = fn_factory()
    bounds = fn.bounds
    train_X = sample_bounded(bounds, spec.n_train, spec.seed)
    test_X = sample_bounded(bounds, spec.n_test, spec.seed + 10000)

    if spec.n_outcomes > 1:
        train_Y = fn(train_X)
        if train_Y.dim() == 1:
            train_Y = train_Y.unsqueeze(-1)
    else:
        train_Y = fn(train_X).unsqueeze(-1)

    return train_X, train_Y, test_X


def discretize_dims(X: Tensor, dims: list[int], n_bins: int) -> Tensor:
    """Discretize specified dimensions into n_bins bins (0, 1, ..., n_bins-1).

    Maps the continuous range in those dimensions to integer bin indices.
    """
    X = X.clone()
    for d in dims:
        col = X[:, d]
        lo, hi = col.min(), col.max()
        if hi > lo:
            normalized = (col - lo) / (hi - lo)
        else:
            normalized = torch.zeros_like(col)
        binned = torch.clamp((normalized * n_bins).long(), 0, n_bins - 1)
        X[:, d] = binned.double()
    return X


def round_to_int_dims(
    X: Tensor, dims: list[int], lo: int, hi: int
) -> Tensor:
    """Map continuous [0,1]-bounded dims to integer range [lo, hi], then round."""
    X = X.clone()
    for d in dims:
        col = X[:, d]
        col_min, col_max = col.min(), col.max()
        if col_max > col_min:
            normalized = (col - col_min) / (col_max - col_min)
        else:
            normalized = torch.zeros_like(col)
        X[:, d] = torch.clamp(torch.round(normalized * (hi - lo) + lo), lo, hi)
    return X


# ── Fixture wrapper ──────────────────────────────────────────────────────────

def wrap_fixture(
    spec: FixtureSpec,
    model_state: dict,
    test_X: Tensor,
    pred_mean,
    pred_var,
    extra_metadata: dict | None = None,
    sq_point: Tensor | None = None,
    sq_mean: float | None = None,
    sq_var: float | None = None,
    param_bounds: Tensor | None = None,
    observations: list[dict] | None = None,
) -> dict:
    """Build the fixture JSON structure in {experiment, test} format."""
    import botorch
    import gpytorch as gpy

    if isinstance(pred_mean, Tensor):
        mean_val = pred_mean.tolist()
    else:
        mean_val = pred_mean

    if isinstance(pred_var, Tensor):
        var_val = pred_var.tolist()
    else:
        var_val = pred_var

    # Build search_space from test_X dimensions
    d = test_X.shape[-1]
    pnames = spec.param_names or [f"x{i}" for i in range(d)]
    params = []
    cat_dims_set = set(spec.cat_dims or [])
    int_dims_set = set(spec.int_dims or [])
    for i in range(d):
        if i in cat_dims_set:
            params.append({
                "name": pnames[i],
                "type": "choice",
                "values": list(range(spec.cat_cardinality)),
            })
        else:
            lo_val = float(param_bounds[0, i]) if param_bounds is not None else 0.0
            hi_val = float(param_bounds[1, i]) if param_bounds is not None else 1.0
            p: dict[str, Any] = {
                "name": pnames[i],
                "type": "range",
                "bounds": [lo_val, hi_val],
            }
            if i in int_dims_set:
                int_lo, int_hi = spec.int_bounds
                p["bounds"] = [float(int_lo), float(int_hi)]
                p["parameter_type"] = "int"
            params.append(p)

    # Determine outcome_names
    outcome_names: list[str]
    if spec.outcome_names:
        outcome_names = spec.outcome_names
    elif model_state.get("model_type") == "ModelListGP":
        outcome_names = model_state.get("outcome_names", [f"y{i}" for i in range(len(model_state.get("models", [])))])
    else:
        outcome_names = ["y"]

    # Build experiment (shared schema with axjs_export.py)
    experiment: dict[str, Any] = {
        "search_space": {"parameters": params},
        "model_state": model_state,
        "name": spec.name,
        "description": spec.description,
        "outcome_names": outcome_names,
    }

    # Observations
    if observations:
        experiment["observations"] = observations

    # Status quo point
    if sq_point is not None:
        experiment["status_quo"] = {"point": sq_point.squeeze().tolist()}

    # Optimization config (objectives, constraints, thresholds)
    if spec.objectives:
        opt_config: dict[str, Any] = {"objectives": spec.objectives}
        if spec.outcome_constraints:
            opt_config["outcome_constraints"] = spec.outcome_constraints
        if spec.objective_thresholds:
            opt_config["objective_thresholds"] = spec.objective_thresholds
        experiment["optimization_config"] = opt_config

    # Build test section
    test_metadata: dict[str, Any] = {
        "botorch_version": botorch.__version__,
        "gpytorch_version": gpy.__version__,
        "torch_version": torch.__version__,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": spec.seed,
        "benchmark": spec.benchmark,
    }
    if extra_metadata:
        test_metadata.update(extra_metadata)

    # Test points: include status quo point as first entry if present
    test_points_list = test_X.tolist()
    if sq_point is not None:
        test_points_list = [sq_point.squeeze().tolist()] + test_points_list

    test_section: dict[str, Any] = {
        "metadata": test_metadata,
        "test_points": test_points_list,
        "expected": {"mean": mean_val, "variance": var_val},
    }

    # For status quo fixtures: prepend status quo expected values and add relative
    if sq_point is not None and sq_mean is not None and sq_var is not None:
        if mean_val is not None and var_val is not None:
            test_section["expected"]["mean"] = [float(sq_mean)] + mean_val
            test_section["expected"]["variance"] = [float(sq_var)] + var_val

            from ax.utils.stats.math_utils import relativize as ax_relativize

            pred_mean_t = torch.tensor(mean_val) if not isinstance(pred_mean, Tensor) else pred_mean
            pred_var_t = torch.tensor(var_val) if not isinstance(pred_var, Tensor) else pred_var
            sq_sem = float(sq_var) ** 0.5
            n = len(mean_val)

            rel_means = []
            rel_vars = []
            for i in range(n):
                m_t = float(pred_mean_t[i])
                s_t = float(pred_var_t[i]) ** 0.5
                r_mean, r_sem = ax_relativize(
                    means_t=m_t,
                    sems_t=s_t,
                    mean_c=float(sq_mean),
                    sem_c=sq_sem,
                    as_percent=True,
                    bias_correction=True,
                    cov_means=0.0,
                    control_as_constant=False,
                )
                rel_means.append(float(r_mean))
                rel_vars.append(float(r_sem) ** 2)

            test_section["expected_relative"] = {
                "mean": rel_means,
                "variance": rel_vars,
            }

    fixture = {
        "experiment": experiment,
        "test": test_section,
    }

    return fixture
