"""MultiTaskGP fixture generator."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np
import torch

from benchmarks import sample_bounded
from fixture_helpers import (
    build_warp_transform,
    build_composite_kernel,
    discretize_dims,
    wrap_fixture,
)
from _extraction import (
    extract_multitask_kernel,
    extract_mean_constant,
    export_multi_task_gp,
)

if TYPE_CHECKING:
    from generate_fixtures import FixtureSpec


def generate_multitask_fixture(spec: FixtureSpec, benchmarks: dict) -> dict:
    """Generate MultiTaskGP (ICM) fixture.

    Creates multi-task data by evaluating the same benchmark with different
    noise seeds per task, simulating correlated but distinct tasks.
    """
    import gpytorch as gpy
    from botorch.models import MultiTaskGP
    from botorch.fit import fit_gpytorch_mll

    fn = benchmarks[spec.benchmark]()
    bounds = fn.bounds
    d = bounds.shape[-1]

    torch.manual_seed(spec.seed)
    np.random.seed(spec.seed)

    n_per_task = spec.n_train // spec.num_tasks
    train_Xs = []
    train_Ys = []
    for t in range(spec.num_tasks):
        X_t = sample_bounded(bounds, n_per_task, spec.seed + t)
        Y_t = fn(X_t)
        # Add task-specific offset to create task correlation structure
        Y_t = Y_t + t * 0.5
        task_col = torch.full((n_per_task, 1), float(t), dtype=torch.float64)
        train_Xs.append(torch.cat([X_t, task_col], dim=-1))
        train_Ys.append(Y_t)

    train_X = torch.cat(train_Xs, dim=0)
    train_Y = torch.cat(train_Ys, dim=0).unsqueeze(-1)

    # Discretize categorical dims in training data (before task column at -1)
    if spec.use_composite and spec.cat_dims:
        # cat_dims are indices in the data columns (before task column)
        data_cols = train_X[:, :d]
        data_cols = discretize_dims(data_cols, spec.cat_dims, spec.cat_cardinality)
        train_X = torch.cat([data_cols, train_X[:, d:]], dim=-1)

    mt_kwargs: dict[str, Any] = {}
    d_full = d + 1  # full dimension including task column

    # Custom data kernel for composite
    if spec.use_composite and spec.cat_dims:
        cont_dims = [i for i in range(d) if i not in spec.cat_dims]
        data_kernel = build_composite_kernel(
            spec.kernel_type, spec.nu, cont_dims, spec.cat_dims
        )
        mt_kwargs["covar_module"] = data_kernel

    # Input warping
    if spec.use_warp:
        # d_full includes task column; warp only data dims
        warp_indices = list(range(d))
        if spec.use_composite and spec.cat_dims:
            warp_indices = [i for i in range(d) if i not in spec.cat_dims]
        mt_kwargs["input_transform"] = build_warp_transform(d_full, warp_indices)

    model = MultiTaskGP(train_X, train_Y, task_feature=-1, **mt_kwargs)
    mll = gpy.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    model.eval()

    # Test predictions for task 0
    test_X_raw = sample_bounded(bounds, spec.n_test, spec.seed + 10000)
    if spec.use_composite and spec.cat_dims:
        test_X_raw = discretize_dims(test_X_raw, spec.cat_dims, spec.cat_cardinality)
    task_col = torch.zeros(spec.n_test, 1, dtype=torch.float64)
    test_X_with_task = torch.cat([test_X_raw, task_col], dim=-1)

    covar = model.covar_module
    data_kernel_state, task_covar_state = extract_multitask_kernel(covar)

    # Compute reference predictions using explicit Cholesky (matches axjs algorithm).
    # BoTorch's model.posterior() uses linear_operator which can diverge numerically
    # from explicit Cholesky for composite multitask kernels.
    with torch.no_grad():
        # Find data and task kernel components
        data_kernel_mod = None
        task_kernel_mod = None
        for k in covar.kernels:
            if hasattr(k, "covar_factor"):
                task_kernel_mod = k
            else:
                data_kernel_mod = k

        # Compute kernel matrices using the actual kernel modules
        tf_col = train_X.shape[-1] - 1  # task column
        train_data = train_X[:, :tf_col]
        test_data = test_X_raw
        train_tasks = train_X[:, tf_col].long()

        # Apply input transforms to data columns
        itf_mod = model.input_transform if hasattr(model, "input_transform") and model.input_transform is not None else None
        if itf_mod is not None:
            # Transform full X (including task col), then extract data
            train_full_tf = itf_mod(train_X)
            test_full_tf = itf_mod(test_X_with_task)
            train_data_tf = train_full_tf[:, :tf_col]
            test_data_tf = test_full_tf[:, :tf_col]
        else:
            train_data_tf = train_data
            test_data_tf = test_data

        # K_data matrices
        K_data_train = data_kernel_mod(train_data_tf, train_data_tf).evaluate()
        K_data_cross = data_kernel_mod(test_data_tf, train_data_tf).evaluate()
        K_data_diag = data_kernel_mod(test_data_tf, test_data_tf).evaluate().diag()

        # B matrix
        B = task_kernel_mod._eval_covar_matrix().detach()

        # Full kernel: K_data[i,j] * B[t_i, t_j]
        n_train = train_X.shape[0]
        n_test_pts = test_X_raw.shape[0]
        K_full = torch.zeros(n_train, n_train)
        for i in range(n_train):
            for j in range(n_train):
                K_full[i, j] = K_data_train[i, j] * B[train_tasks[i], train_tasks[j]]

        # Add noise to diagonal
        noise_t = model.likelihood.noise.detach().cpu().squeeze()
        if noise_t.dim() == 0:
            K_full += noise_t.item() * torch.eye(n_train)
        else:
            noise_per_point = noise_t[train_tasks]
            K_full += torch.diag(noise_per_point)

        # Cholesky and solve
        L = torch.linalg.cholesky(K_full)
        mean_const_val = extract_mean_constant(model)
        if isinstance(mean_const_val, list):
            mean_consts = mean_const_val
        else:
            mean_consts = [mean_const_val] * spec.num_tasks

        train_y = model.train_targets
        residuals = torch.zeros(n_train)
        for i in range(n_train):
            residuals[i] = train_y[i] - mean_consts[train_tasks[i]]
        alpha = torch.cholesky_solve(residuals.unsqueeze(-1), L)

        # Cross-covariance for task 0
        test_task_idx = 0
        K_star = torch.zeros(n_test_pts, n_train)
        for i in range(n_test_pts):
            for j in range(n_train):
                K_star[i, j] = K_data_cross[i, j] * B[test_task_idx, train_tasks[j]]

        # Posterior mean
        pred_mean = torch.zeros(n_test_pts)
        for i in range(n_test_pts):
            pred_mean[i] = mean_consts[test_task_idx] + (K_star[i] @ alpha).item()

        # Posterior variance
        V = torch.linalg.solve_triangular(L, K_star.T, upper=False)
        kss = K_data_diag * B[test_task_idx, test_task_idx]
        pred_var = kss - (V * V).sum(dim=0)
        pred_var = pred_var.clamp(min=0)

        # Untransform if needed
        otf_mod = model.outcome_transform if hasattr(model, "outcome_transform") else None
        if otf_mod is not None:
            otf_std = otf_mod.stdvs.detach().cpu().item()
            otf_mean_val = otf_mod.means.detach().cpu().item()
            pred_mean = pred_mean * otf_std + otf_mean_val
            pred_var = pred_var * otf_std ** 2

    model_state = export_multi_task_gp(model)

    return wrap_fixture(
        spec,
        model_state,
        test_X_raw,
        pred_mean,
        pred_var,
        extra_metadata={"task_index": 0},
    )
