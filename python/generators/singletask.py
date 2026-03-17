"""SingleTaskGP / FixedNoiseGP fixture generator."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np
import torch

from fixture_helpers import (
    build_warp_transform,
    build_composite_kernel,
    get_benchmark_data,
    discretize_dims,
    round_to_int_dims,
    wrap_fixture,
)
from _extraction import export_single_gp

if TYPE_CHECKING:
    from generate_fixtures import FixtureSpec


def generate_singletask_fixture(spec: FixtureSpec, benchmarks: dict) -> dict:
    """Generate SingleTaskGP or FixedNoiseGP fixture."""
    import gpytorch as gpy
    from botorch.models import SingleTaskGP
    from botorch.fit import fit_gpytorch_mll

    # Get data from benchmark
    train_X, train_Y, test_X = get_benchmark_data(spec, benchmarks)

    torch.manual_seed(spec.seed)
    np.random.seed(spec.seed)

    # Round integer dims
    if spec.int_dims:
        lo, hi = spec.int_bounds
        train_X = round_to_int_dims(train_X, spec.int_dims, lo, hi)
        test_X = round_to_int_dims(test_X, spec.int_dims, lo, hi)

    # Discretize categorical dims if specified
    if spec.use_composite and spec.cat_dims:
        train_X = discretize_dims(train_X, spec.cat_dims, spec.cat_cardinality)
        test_X = discretize_dims(test_X, spec.cat_dims, spec.cat_cardinality)

    d = train_X.shape[-1]
    model_kwargs: dict[str, Any] = {}

    if spec.model_class == "FixedNoiseGP":
        if spec.heteroscedastic_noise:
            # Noise variance proportional to |y|: each point has its own variance
            base_noise = spec.noise or 0.1
            model_kwargs["train_Yvar"] = (base_noise * train_Y.abs()).clamp(min=1e-6)
        else:
            model_kwargs["train_Yvar"] = torch.full_like(train_Y, spec.noise or 0.1)

    # Determine continuous dims (for warp indices if needed)
    cat_dims_set = set(spec.cat_dims or [])
    cont_dims = [i for i in range(d) if i not in cat_dims_set]

    # Kernel configuration
    if spec.use_additive:
        from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel, AdditiveKernel as GPyAdditiveKernel
        per_dim_kernels = []
        for i in range(d):
            if spec.kernel_type == "RBF":
                k = RBFKernel(ard_num_dims=1, active_dims=[i])
            else:
                k = MaternKernel(nu=spec.nu, ard_num_dims=1, active_dims=[i])
            per_dim_kernels.append(k)
        model_kwargs["covar_module"] = ScaleKernel(
            GPyAdditiveKernel(*per_dim_kernels)
        )
    elif spec.use_composite and spec.cat_dims:
        model_kwargs["covar_module"] = build_composite_kernel(
            spec.kernel_type, spec.nu, cont_dims, spec.cat_dims
        )
    elif spec.kernel_type == "RBF":
        from gpytorch.kernels import RBFKernel, ScaleKernel
        model_kwargs["covar_module"] = ScaleKernel(RBFKernel(ard_num_dims=d))
    elif spec.nu == 1.5:
        from gpytorch.kernels import MaternKernel, ScaleKernel
        model_kwargs["covar_module"] = ScaleKernel(MaternKernel(nu=1.5, ard_num_dims=d))
    elif spec.nu == 0.5:
        from gpytorch.kernels import MaternKernel, ScaleKernel
        model_kwargs["covar_module"] = ScaleKernel(MaternKernel(nu=0.5, ard_num_dims=d))

    # Input warping
    if spec.use_warp:
        warp_indices = cont_dims if spec.use_composite and spec.cat_dims else list(range(d))
        model_kwargs["input_transform"] = build_warp_transform(d, warp_indices)

    # Outcome transform
    if spec.outcome_type == "Log":
        from botorch.models.transforms.outcome import Log
        model_kwargs["outcome_transform"] = Log()
    elif spec.outcome_type == "Bilog":
        from botorch.models.transforms.outcome import Bilog
        model_kwargs["outcome_transform"] = Bilog()
    elif spec.outcome_type == "Power":
        from botorch.models.transforms.outcome import Power
        model_kwargs["outcome_transform"] = Power(power=spec.power_lambda)
    elif spec.outcome_type == "Chained":
        from botorch.models.transforms.outcome import Log, Standardize, ChainedOutcomeTransform
        model_kwargs["outcome_transform"] = ChainedOutcomeTransform(
            log=Log(), standardize=Standardize(m=1),
        )

    model = SingleTaskGP(train_X, train_Y, **model_kwargs)

    mll = gpy.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    model.eval()

    # For Log/Chained outcome transforms, BoTorch provides analytic posterior
    # moments (norm_to_lognorm). For Bilog/Power, BoTorch does NOT provide
    # analytic mean/variance — we use None to signal consistency-only testing.
    is_consistency_only = spec.outcome_type in ("Bilog", "Power")

    with torch.no_grad(), gpy.settings.fast_pred_var():
        posterior = model.posterior(test_X)
        if is_consistency_only:
            # BoTorch raises NotImplementedError for .mean/.variance on
            # Bilog/Power posteriors. Set expected to null for consistency tests.
            pred_mean = None
            pred_var = None
        else:
            pred_mean = posterior.mean.squeeze(-1)
            pred_var = posterior.variance.squeeze(-1)

    # Status quo prediction at center of bounds
    sq_point = sq_mean_val = sq_var_val = None
    if spec.status_quo == "center":
        fn = benchmarks[spec.benchmark]()
        bounds = fn.bounds
        sq_point = ((bounds[0] + bounds[1]) / 2).unsqueeze(0)
        with torch.no_grad(), gpy.settings.fast_pred_var():
            sq_posterior = model.posterior(sq_point)
            sq_mean_val = sq_posterior.mean.squeeze().item()
            sq_var_val = sq_posterior.variance.squeeze().item()

    model_state = export_single_gp(model)

    return wrap_fixture(
        spec, model_state, test_X, pred_mean, pred_var,
        sq_point=sq_point, sq_mean=sq_mean_val, sq_var=sq_var_val,
    )
