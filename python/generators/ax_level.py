"""Ax-level fixture generator (through Ax adapter pipeline)."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np
import torch

from benchmarks import make_benchmarks, sample_bounded
from fixture_helpers import wrap_fixture
from _extraction import export_multi_task_gp
from generators._ax_helpers import (
    _build_ax_transforms,
    _compose_unitx_into_normalize,
)
from generators.multitask import generate_multitask_fixture

if TYPE_CHECKING:
    from generate_fixtures import FixtureSpec


def generate_ax_level_fixture(spec: FixtureSpec, benchmarks: dict) -> dict:
    """Generate fixture through Ax adapter pipeline for true end-to-end parity.

    Reference predictions come from Ax's actual adapter.predict(), not manual
    untransform math. This ensures the Predictor output matches what Ax users see.

    Handles single-output and multi-output (incl. batched MOO). MultiTaskGP
    is delegated to _generate_ax_level_multitask_fixture.
    """
    # MultiTaskGP needs different Ax setup (task features), handle separately
    if spec.num_tasks > 0:
        return _generate_ax_level_multitask_fixture(spec, benchmarks)

    import pandas as pd
    from ax.core import (
        Experiment,
        SearchSpace,
        RangeParameter,
        ParameterType,
        Metric,
        Objective,
        OptimizationConfig,
        ObservationFeatures,
    )
    from ax.core.arm import Arm
    from ax.core.data import Data
    try:
        from ax.adapter.registry import Generators
    except ImportError:
        from ax.modelbridge.registry import Models as Generators
    from axjs_export import _extract_adapter_transforms

    # ── Get benchmark data ──
    if not spec.negate_benchmark:
        fn = make_benchmarks(negate=False)[spec.benchmark]()
    else:
        fn = benchmarks[spec.benchmark]()
    bounds = fn.bounds
    d = bounds.shape[-1]

    torch.manual_seed(spec.seed)
    np.random.seed(spec.seed)

    train_X = sample_bounded(bounds, spec.n_train, spec.seed)
    test_X = sample_bounded(bounds, spec.n_test, spec.seed + 10000)

    raw_Y = fn(train_X)
    if raw_Y.dim() == 1:
        raw_Y = raw_Y.unsqueeze(-1)
    n_outcomes = spec.n_outcomes
    if n_outcomes == 1 and raw_Y.shape[-1] > 1:
        raw_Y = raw_Y[:, :1]

    # ── Build Ax search space ──
    param_names = spec.param_names or [f"x{i}" for i in range(d)]
    ax_params = [
        RangeParameter(
            name=param_names[i],
            parameter_type=ParameterType.FLOAT,
            lower=float(bounds[0, i]),
            upper=float(bounds[1, i]),
        )
        for i in range(d)
    ]
    search_space = SearchSpace(parameters=ax_params)

    # ── Determine outcome names ──
    outcome_names = spec.outcome_names or (
        [f"y{i}" for i in range(n_outcomes)] if n_outcomes > 1 else ["y"]
    )

    # ── Build optimization config ──
    objectives_spec = spec.objectives or [{"name": outcome_names[0], "minimize": True}]
    if len(objectives_spec) > 1:
        from ax.core import MultiObjective, MultiObjectiveOptimizationConfig
        objectives = [
            Objective(metric=Metric(name=obj["name"]), minimize=obj["minimize"])
            for obj in objectives_spec
        ]
        opt_config = MultiObjectiveOptimizationConfig(
            objective=MultiObjective(objectives=objectives)
        )
    else:
        obj = objectives_spec[0]
        opt_config = OptimizationConfig(
            objective=Objective(metric=Metric(name=obj["name"]), minimize=obj["minimize"])
        )

    # ── Create experiment ──
    experiment = Experiment(
        search_space=search_space,
        optimization_config=opt_config,
        name=spec.name,
    )

    # ── Attach training data as a batch trial ──
    trial = experiment.new_batch_trial()
    arms = []
    for i in range(spec.n_train):
        arm_params = {param_names[j]: float(train_X[i, j]) for j in range(d)}
        arms.append(Arm(parameters=arm_params, name=f"{i}_0"))
    trial.add_arms_and_weights(arms=arms)
    trial.mark_running(no_runner_required=True)
    trial.mark_completed()

    data_rows = []
    for i, arm in enumerate(arms):
        for k in range(n_outcomes):
            y_val = float(raw_Y[i, k]) if n_outcomes > 1 else float(raw_Y[i, 0])
            metric_name = outcome_names[k]
            data_rows.append({
                "arm_name": arm.name,
                "metric_name": metric_name,
                "metric_signature": metric_name,
                "mean": y_val,
                "sem": 0.0,
                "trial_index": trial.index,
            })
    data = Data(df=pd.DataFrame(data_rows))

    # ── Build transform list and create adapter ──
    transforms, transform_configs = _build_ax_transforms(
        spec.adapter_transforms, outcome_names
    )
    adapter = Generators.BOTORCH_MODULAR(
        experiment=experiment,
        data=data,
        transforms=transforms,
        transform_configs=transform_configs,
    )

    # ── Get reference predictions from adapter.predict() ──
    sq_point = None
    if spec.status_quo == "center":
        sq_point = (bounds[0] + bounds[1]) / 2.0

    all_features = []
    if sq_point is not None:
        all_features.append(ObservationFeatures(
            parameters={param_names[j]: float(sq_point[j]) for j in range(d)}
        ))
    for i in range(spec.n_test):
        all_features.append(ObservationFeatures(
            parameters={param_names[j]: float(test_X[i, j]) for j in range(d)}
        ))

    means_dict, cov_dict = adapter.predict(all_features)

    # ── Export model state ──
    # Ax 1.2+: adapter.generator.surrogate.model
    # Ax <1.2: adapter.model.surrogate.model
    _gen = getattr(adapter, "generator", None) or getattr(adapter, "model")
    botorch_model = _gen.surrogate.model
    from _extraction import export_botorch_model
    model_state = export_botorch_model(botorch_model)

    # Compose UnitX into Normalize so axjs can predict on raw parameter values
    _compose_unitx_into_normalize(model_state, bounds)

    # ── Extract adapter transforms ──
    adapter_transforms_meta = _extract_adapter_transforms(adapter)

    # ── Build fixture ──
    # Split status quo from test predictions
    offset = 1 if sq_point is not None else 0
    if n_outcomes > 1:
        expected_means = []
        expected_vars = []
        for k, name in enumerate(outcome_names):
            all_m = means_dict[name]
            all_v = [cov_dict[name][name][i] for i in range(len(all_m))]
            if sq_point is not None:
                expected_means.append([all_m[0]] + all_m[offset:])
                expected_vars.append([all_v[0]] + all_v[offset:])
            else:
                expected_means.append(all_m)
                expected_vars.append(all_v)
        mean_val = expected_means
        var_val = expected_vars
    else:
        name = outcome_names[0]
        all_m = means_dict[name]
        all_v = [cov_dict[name][name][i] for i in range(len(all_m))]
        mean_val = all_m
        var_val = all_v

    # Build search_space params for fixture
    params = []
    for i in range(d):
        params.append({
            "name": param_names[i],
            "type": "range",
            "bounds": [float(bounds[0, i]), float(bounds[1, i])],
        })

    # Build test points (status quo first if present)
    test_points = test_X.tolist()
    if sq_point is not None:
        test_points = [sq_point.tolist()] + test_points

    import botorch
    import gpytorch as gpy
    from datetime import datetime, timezone

    experiment_dict: dict[str, Any] = {
        "search_space": {"parameters": params},
        "model_state": model_state,
        "name": spec.name,
        "description": spec.description,
        "outcome_names": outcome_names,
    }
    if sq_point is not None:
        experiment_dict["status_quo"] = {"point": sq_point.tolist()}
    if adapter_transforms_meta:
        experiment_dict["adapter_transforms"] = adapter_transforms_meta

    test_metadata: dict[str, Any] = {
        "botorch_version": botorch.__version__,
        "gpytorch_version": gpy.__version__,
        "torch_version": torch.__version__,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": spec.seed,
        "benchmark": spec.benchmark,
        "ax_level": True,
    }

    fixture: dict[str, Any] = {
        "experiment": experiment_dict,
        "test": {
            "metadata": test_metadata,
            "test_points": test_points,
            "expected": {"mean": mean_val, "variance": var_val},
        },
    }

    # ── Compute expected_relative for status quo fixtures ──
    if sq_point is not None:
        from ax.utils.stats.math_utils import relativize as ax_relativize

        if n_outcomes > 1:
            # Per-outcome relativization
            expected_relative: dict[str, Any] = {}
            for k, name in enumerate(outcome_names):
                sq_m = means_dict[name][0]
                sq_v = cov_dict[name][name][0]
                sq_sem = sq_v ** 0.5
                rel_means = []
                rel_vars = []
                for i in range(offset, len(means_dict[name])):
                    m_t = means_dict[name][i]
                    s_t = cov_dict[name][name][i] ** 0.5
                    r_mean, r_sem = ax_relativize(
                        means_t=m_t, sems_t=s_t,
                        mean_c=sq_m, sem_c=sq_sem,
                        as_percent=True, bias_correction=True,
                        cov_means=0.0, control_as_constant=False,
                    )
                    rel_means.append(float(r_mean))
                    rel_vars.append(float(r_sem) ** 2)
                expected_relative[name] = {"mean": rel_means, "variance": rel_vars}
            fixture["test"]["expected_relative"] = expected_relative
        else:
            name = outcome_names[0]
            sq_m = means_dict[name][0]
            sq_v = cov_dict[name][name][0]
            sq_sem = sq_v ** 0.5
            rel_means = []
            rel_vars = []
            for i in range(offset, len(means_dict[name])):
                m_t = means_dict[name][i]
                s_t = cov_dict[name][name][i] ** 0.5
                r_mean, r_sem = ax_relativize(
                    means_t=m_t, sems_t=s_t,
                    mean_c=sq_m, sem_c=sq_sem,
                    as_percent=True, bias_correction=True,
                    cov_means=0.0, control_as_constant=False,
                )
                rel_means.append(float(r_mean))
                rel_vars.append(float(r_sem) ** 2)
            fixture["test"]["expected_relative"] = {
                "mean": rel_means, "variance": rel_vars,
            }

    return fixture


def _generate_ax_level_multitask_fixture(
    spec: FixtureSpec, benchmarks: dict,
) -> dict:
    """Generate Ax-level MultiTaskGP fixture using BoTorch posterior directly.

    MultiTaskGP doesn't go through Ax's adapter.predict() because:
    - Ax-level multitask fixtures test multi-task iteration and relativization,
      not adapter Y-transforms (none are specified on these fixtures)
    - Setting up Ax's multi-task machinery (task parameters, etc.) adds
      complexity without testing additional axjs code paths

    For all_tasks=True, uses BoTorch posterior for each task directly.
    """
    import gpytorch as gpy
    from botorch.models import MultiTaskGP as BotorchMTGP
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
        Y_t = fn(X_t) + t * 0.5
        task_col = torch.full((n_per_task, 1), float(t), dtype=torch.float64)
        train_Xs.append(torch.cat([X_t, task_col], dim=-1))
        train_Ys.append(Y_t)

    train_X = torch.cat(train_Xs, dim=0)
    train_Y = torch.cat(train_Ys, dim=0).unsqueeze(-1)

    model = BotorchMTGP(train_X, train_Y, task_feature=-1)
    mll = gpy.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    model.eval()

    test_X_raw = sample_bounded(bounds, spec.n_test, spec.seed + 10000)

    if spec.all_tasks:
        # Get predictions for ALL tasks using BoTorch posterior
        outcome_names = [f"y_task{t}" for t in range(spec.num_tasks)]
        all_task_means: dict[str, list] = {}
        all_task_vars: dict[str, list] = {}

        with torch.no_grad(), gpy.settings.fast_pred_var():
            for t in range(spec.num_tasks):
                task_col = torch.full((spec.n_test, 1), float(t), dtype=torch.float64)
                test_X_with_task = torch.cat([test_X_raw, task_col], dim=-1)
                posterior = model.posterior(test_X_with_task)
                pred_mean = posterior.mean.squeeze(-1)
                pred_var = posterior.variance.squeeze(-1)
                all_task_means[outcome_names[t]] = pred_mean.tolist()
                all_task_vars[outcome_names[t]] = pred_var.tolist()

        model_state = export_multi_task_gp(model)

        fixture = wrap_fixture(
            spec, model_state, test_X_raw,
            all_task_means, all_task_vars,
            extra_metadata={"ax_level": True, "all_tasks": True, "task_index": None},
        )
        fixture["experiment"]["outcome_names"] = outcome_names
        return fixture
    else:
        # Single task (task 0) — use the existing multitask generator, mark ax_level
        fixture = generate_multitask_fixture(spec, benchmarks)
        fixture["test"]["metadata"]["ax_level"] = True

        # Status quo handling for relativization
        if spec.status_quo == "center":
            sq_point = (bounds[0] + bounds[1]) / 2.0
            fixture["experiment"]["status_quo"] = {"point": sq_point.tolist()}

            # Get status quo prediction for task 0
            with torch.no_grad(), gpy.settings.fast_pred_var():
                task_col = torch.zeros(1, 1, dtype=torch.float64)
                sq_with_task = torch.cat([sq_point.unsqueeze(0), task_col], dim=-1)
                sq_posterior = model.posterior(sq_with_task)
                sq_m = float(sq_posterior.mean.squeeze())
                sq_v = float(sq_posterior.variance.squeeze())

            # Prepend status quo to test points and expected values
            fixture["test"]["test_points"] = [sq_point.tolist()] + fixture["test"]["test_points"]
            old_mean = fixture["test"]["expected"]["mean"]
            old_var = fixture["test"]["expected"]["variance"]
            fixture["test"]["expected"]["mean"] = [sq_m] + (old_mean if isinstance(old_mean, list) else old_mean.tolist())
            fixture["test"]["expected"]["variance"] = [sq_v] + (old_var if isinstance(old_var, list) else old_var.tolist())

            # Compute expected_relative
            from ax.utils.stats.math_utils import relativize as ax_relativize
            sq_sem = sq_v ** 0.5
            rel_means = []
            rel_vars = []
            test_means = fixture["test"]["expected"]["mean"][1:]
            test_vars = fixture["test"]["expected"]["variance"][1:]
            for i in range(len(test_means)):
                r_mean, r_sem = ax_relativize(
                    means_t=test_means[i], sems_t=test_vars[i] ** 0.5,
                    mean_c=sq_m, sem_c=sq_sem,
                    as_percent=True, bias_correction=True,
                    cov_means=0.0, control_as_constant=False,
                )
                rel_means.append(float(r_mean))
                rel_vars.append(float(r_sem) ** 2)
            fixture["test"]["expected_relative"] = {
                "mean": rel_means, "variance": rel_vars,
            }

        return fixture
