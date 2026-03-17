"""PairwiseGP fixture generator."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np
import torch

from benchmarks import sample_bounded
from fixture_helpers import build_warp_transform, wrap_fixture
from _extraction import export_pairwise_gp

if TYPE_CHECKING:
    from generate_fixtures import FixtureSpec


def generate_pairwise_fixture(spec: FixtureSpec, benchmarks: dict) -> dict:
    """Generate PairwiseGP fixture using benchmark for utility ordering."""
    from botorch.models.pairwise_gp import PairwiseGP, PairwiseLaplaceMarginalLogLikelihood
    from botorch.fit import fit_gpytorch_mll

    fn = benchmarks[spec.benchmark]()
    bounds = fn.bounds
    train_X = sample_bounded(bounds, spec.n_train, spec.seed)
    test_X = sample_bounded(bounds, spec.n_test, spec.seed + 10000)

    # Generate pairwise comparisons based on benchmark values
    torch.manual_seed(spec.seed)
    np.random.seed(spec.seed)
    utilities = fn(train_X)
    n_comparisons = spec.n_train * 2
    comparisons = []
    for _ in range(n_comparisons):
        i, j = np.random.choice(spec.n_train, 2, replace=False)
        if utilities[i] > utilities[j]:
            comparisons.append([int(i), int(j)])
        else:
            comparisons.append([int(j), int(i)])
    comp_tensor = torch.tensor(comparisons, dtype=torch.long)

    pairwise_kwargs: dict[str, Any] = {}
    if spec.use_warp:
        d = train_X.shape[-1]
        pairwise_kwargs["input_transform"] = build_warp_transform(d)

    model = PairwiseGP(train_X, comp_tensor, **pairwise_kwargs)
    mll = PairwiseLaplaceMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    model.eval()

    with torch.no_grad():
        posterior = model.posterior(test_X)
        pred_mean = posterior.mean.squeeze()
        pred_var = posterior.variance.squeeze()

    model_state = export_pairwise_gp(model)

    return wrap_fixture(spec, model_state, test_X, pred_mean, pred_var)
