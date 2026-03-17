"""ModelListGP fixture generator."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import torch

from benchmarks import sample_bounded
from fixture_helpers import build_warp_transform, wrap_fixture
from _extraction import export_single_gp

if TYPE_CHECKING:
    from generate_fixtures import FixtureSpec


def generate_model_list_fixture(
    spec: FixtureSpec, benchmarks: dict
) -> dict:
    """Generate ModelListGP fixture from multi-objective benchmark."""
    import gpytorch as gpy
    from botorch.models import SingleTaskGP
    from botorch.fit import fit_gpytorch_mll

    fn = benchmarks[spec.benchmark]()
    bounds = fn.bounds
    d = bounds.shape[-1]

    train_X = sample_bounded(bounds, spec.n_train, spec.seed)
    test_X = sample_bounded(bounds, spec.n_test, spec.seed + 10000)

    # Multi-objective: fn returns (n, n_outcomes)
    all_Y = fn(train_X)
    if all_Y.dim() == 1:
        all_Y = all_Y.unsqueeze(-1)

    n_outcomes = all_Y.shape[-1]

    model_states = []
    expected_means = []
    expected_vars = []

    use_fixed_noise = spec.noise is not None

    for k in range(n_outcomes):
        train_Y_k = all_Y[:, k].unsqueeze(-1)

        sub_kwargs: dict[str, Any] = {}

        # FixedNoiseGP: per-point observed variance
        if use_fixed_noise:
            if spec.heteroscedastic_noise:
                base_noise = spec.noise or 0.1
                sub_kwargs["train_Yvar"] = (base_noise * train_Y_k.abs()).clamp(min=1e-6)
            else:
                sub_kwargs["train_Yvar"] = torch.full_like(train_Y_k, spec.noise)

        # Input warping
        if spec.use_warp:
            sub_kwargs["input_transform"] = build_warp_transform(d)

        model_k = SingleTaskGP(train_X, train_Y_k, **sub_kwargs)
        mll = gpy.mlls.ExactMarginalLogLikelihood(model_k.likelihood, model_k)
        fit_gpytorch_mll(mll)
        model_k.eval()

        with torch.no_grad(), gpy.settings.fast_pred_var():
            posterior = model_k.posterior(test_X)
            pred_mean = posterior.mean.squeeze(-1)
            pred_var = posterior.variance.squeeze(-1)

        expected_means.append(pred_mean.tolist())
        expected_vars.append(pred_var.tolist())

        model_states.append(export_single_gp(model_k))

    outcome_names = spec.outcome_names or [f"y{i}" for i in range(n_outcomes)]
    model_state = {
        "model_type": "ModelListGP",
        "outcome_names": outcome_names,
        "models": model_states,
    }

    return wrap_fixture(
        spec, model_state, test_X, expected_means, expected_vars,
        param_bounds=bounds,
    )
