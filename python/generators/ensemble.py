"""EnsembleGP fixture generator."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np
import torch

from benchmarks import sample_bounded
from fixture_helpers import wrap_fixture
from _extraction import export_single_gp

if TYPE_CHECKING:
    from generate_fixtures import FixtureSpec


def _build_map_ensemble(train_X, train_Y, test_X, d_total, spec, gpy, BotorchSTGP):
    """Fit K=4 SingleTaskGPs with different random seeds."""
    K = 4
    model_states = []
    all_means = []
    all_vars = []

    for k in range(K):
        torch.manual_seed(spec.seed + k * 100)
        from gpytorch.kernels import ScaleKernel, MaternKernel
        covar = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=d_total))
        stgp = BotorchSTGP(train_X, train_Y, covar_module=covar)
        mll = gpy.mlls.ExactMarginalLogLikelihood(stgp.likelihood, stgp)
        from botorch.fit import fit_gpytorch_mll
        fit_gpytorch_mll(mll)
        stgp.eval()

        with torch.no_grad(), gpy.settings.fast_pred_var():
            posterior = stgp.posterior(test_X)
            pm = posterior.mean.squeeze(-1)
            pv = posterior.variance.squeeze(-1)

        all_means.append(pm)
        all_vars.append(pv)

        model_states.append(export_single_gp(stgp))

    means_stack = torch.stack(all_means)
    vars_stack = torch.stack(all_vars)
    ref_mean = means_stack.mean(dim=0)
    ref_var = vars_stack.mean(dim=0) + means_stack.var(dim=0, correction=0)
    return model_states, ref_mean, ref_var


def _build_nuts_ensemble(train_X, train_Y, test_X, d_total, spec, gpy, BotorchSTGP):
    """Fit SAAS fully Bayesian GP with NUTS, extract K posterior samples."""
    from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
    from botorch.fit import fit_fully_bayesian_model_nuts

    saas_model = SaasFullyBayesianSingleTaskGP(train_X, train_Y)
    fit_fully_bayesian_model_nuts(
        saas_model, warmup_steps=256, num_samples=128, thinning=32
    )
    saas_model.eval()

    K = min(4, saas_model.mean_module.constant.shape[0])
    model_states = []
    all_means = []
    all_vars = []
    for k in range(K):
        outputscale_k = saas_model.covar_module.outputscale[k].item()
        lengthscale_k = saas_model.covar_module.base_kernel.lengthscale[k].squeeze().tolist()
        noise_k = saas_model.likelihood.noise[k].item()
        mean_k = saas_model.mean_module.constant[k].item()

        from gpytorch.kernels import ScaleKernel, MaternKernel
        covar = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=d_total))
        covar.outputscale = torch.tensor(outputscale_k)
        covar.base_kernel.lengthscale = torch.tensor([lengthscale_k])

        stgp = BotorchSTGP(train_X, train_Y, covar_module=covar)
        stgp.likelihood.noise = torch.tensor([noise_k])
        stgp.mean_module.constant.data.fill_(mean_k)
        stgp.eval()

        with torch.no_grad():
            posterior = stgp.posterior(test_X)
            pm = posterior.mean.squeeze(-1)
            pv = posterior.variance.squeeze(-1)
        all_means.append(pm)
        all_vars.append(pv)

        model_states.append(export_single_gp(stgp))

    means_stack = torch.stack(all_means)
    vars_stack = torch.stack(all_vars)
    ref_mean = means_stack.mean(dim=0)
    ref_var = vars_stack.mean(dim=0) + means_stack.var(dim=0, correction=0)
    return model_states, ref_mean, ref_var


def generate_ensemble_fixture(spec: FixtureSpec, benchmarks: dict) -> dict:
    """Generate EnsembleGP fixture using SAAS fully Bayesian GP.

    Creates 20D data (6 Hartmann dims + 14 noise dims), fits SAAS GP with NUTS,
    extracts K=4 posterior samples, constructs K SingleTaskGP-compatible states.
    """
    import gpytorch as gpy
    from botorch.models import SingleTaskGP as BotorchSTGP

    fn = benchmarks[spec.benchmark]()
    bounds = fn.bounds
    d_signal = bounds.shape[-1]  # 6 for Hartmann
    d_noise = 14
    d_total = d_signal + d_noise

    torch.manual_seed(spec.seed)
    np.random.seed(spec.seed)

    # Generate training data: Hartmann dims + noise dims
    train_X_signal = sample_bounded(bounds, spec.n_train, spec.seed)
    train_X_noise = torch.rand(spec.n_train, d_noise, dtype=torch.float64)
    train_X = torch.cat([train_X_signal, train_X_noise], dim=-1)
    train_Y = fn(train_X_signal).unsqueeze(-1)

    # Test data
    test_X_signal = sample_bounded(bounds, spec.n_test, spec.seed + 10000)
    test_X_noise = torch.rand(spec.n_test, d_noise, dtype=torch.float64)
    test_X = torch.cat([test_X_signal, test_X_noise], dim=-1)

    method = spec.ensemble_method
    if method == "nuts":
        model_states, ref_mean, ref_var = _build_nuts_ensemble(
            train_X, train_Y, test_X, d_total, spec, gpy, BotorchSTGP
        )
    elif method == "map":
        model_states, ref_mean, ref_var = _build_map_ensemble(
            train_X, train_Y, test_X, d_total, spec, gpy, BotorchSTGP
        )
    else:
        # auto: try NUTS, fall back to MAP
        try:
            model_states, ref_mean, ref_var = _build_nuts_ensemble(
                train_X, train_Y, test_X, d_total, spec, gpy, BotorchSTGP
            )
        except (ImportError, Exception) as e:
            print(f"\n  SAAS NUTS failed ({e}), using multi-restart MAP ensemble")
            model_states, ref_mean, ref_var = _build_map_ensemble(
                train_X, train_Y, test_X, d_total, spec, gpy, BotorchSTGP
            )

    model_state = {
        "model_type": "EnsembleGP",
        "models": model_states,
    }

    return wrap_fixture(spec, model_state, test_X, ref_mean, ref_var)
