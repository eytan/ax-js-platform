"""
Generate test fixtures through the full Ax Client pipeline.

Unlike generate_fixtures.py (which constructs BoTorch models directly for parity testing),
this script goes through Ax's Client API to ensure the transform pipeline, normalization,
and model construction match what real Ax users get.

Usage:
    python generate_ax_fixtures.py [--output-dir test/fixtures]

Uses ax.api.Client (NOT AxClient, which is deprecated).

Fixtures generated here test the full Ax → BoTorch → axjs pipeline:
- Ax's adapter-level transforms (BilogY, StandardizeY, etc.)
- Ax's default model construction (batched SingleTaskGP for MOO)
- Ax's input normalization (Normalize transform)
- axjs_export.py's export_client() path
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from axjs_export import export_client, _extract_optimization_config
from _extraction import export_botorch_model


torch.set_default_dtype(torch.float64)


# ── Fixture specs ────────────────────────────────────────────────────────────

@dataclass
class AxFixtureSpec:
    """Specification for an Ax-pipeline fixture."""
    name: str
    description: str
    seed: int
    n_sobol: int  # number of Sobol (quasi-random) initialization trials
    n_bo: int  # number of Bayesian optimization trials (to trigger model fitting)
    n_test: int  # number of test points for parity
    # Problem setup
    parameters: list[dict[str, Any]] = field(default_factory=list)
    objective: str = ""  # e.g., "y" or "y1, y2, y3" for MOO
    outcome_constraints: list[str] | None = None
    # Evaluation function
    eval_fn_name: str = ""  # key into EVAL_FUNCTIONS registry


# ── Evaluation functions ─────────────────────────────────────────────────────

def _eval_branin(params: dict[str, float]) -> dict[str, float]:
    """Branin benchmark evaluation."""
    from botorch.test_functions import Branin
    fn = Branin(negate=True)
    x = torch.tensor([[params["x0"], params["x1"]]], dtype=torch.float64)
    y = fn(x).item()
    return {"branin": y}


def _eval_penicillin(params: dict[str, float]) -> dict[str, float]:
    """Penicillin benchmark evaluation (7D, 3 objectives).

    Penicillin param order: culture_volume, bioreactor_vol, culture_temp,
    aeration_rate, agitator_power, substrate_feed_conc, substrate_feed_rate
    """
    from botorch.test_functions import Penicillin
    fn = Penicillin()
    x = torch.tensor([[
        params["culture_volume"],
        params["bioreactor_vol"],
        params["culture_temp"],
        params["aeration_rate"],
        params["agitator_power"],
        params["substrate_feed_conc"],
        params["substrate_feed_rate"],
    ]], dtype=torch.float64)
    y = fn(x).squeeze()
    return {
        "penicillin_yield": y[0].item(),
        "fermentation_time": y[1].item(),
        "CO2_byproduct": y[2].item(),
    }


def _eval_branin_currin(params: dict[str, float]) -> dict[str, float]:
    """BraninCurrin benchmark evaluation (2D, 2 objectives)."""
    from botorch.test_functions.multi_objective import BraninCurrin
    fn = BraninCurrin()
    x = torch.tensor([[params["x0"], params["x1"]]], dtype=torch.float64)
    y = fn(x).squeeze()
    return {"branin": y[0].item(), "currin": y[1].item()}


EVAL_FUNCTIONS = {
    "branin": _eval_branin,
    "penicillin": _eval_penicillin,
    "branin_currin": _eval_branin_currin,
}


# ── Fixture specs registry ───────────────────────────────────────────────────

# Penicillin bounds from botorch.test_functions.Penicillin:
# dims: culture_volume, bioreactor_vol, culture_temp, aeration_rate,
#        agitator_power, substrate_feed_conc, substrate_feed_rate
# bounds: [[60, 0.05, 293, 0.05, 0.01, 500, 5], [120, 18, 303, 18, 0.5, 700, 6.5]]
PENICILLIN_PARAMS = [
    {"name": "culture_volume", "type": "range", "bounds": (60.0, 120.0), "parameter_type": "float"},
    {"name": "bioreactor_vol", "type": "range", "bounds": (0.05, 18.0), "parameter_type": "float"},
    {"name": "culture_temp", "type": "range", "bounds": (293.0, 303.0), "parameter_type": "float"},
    {"name": "aeration_rate", "type": "range", "bounds": (0.05, 18.0), "parameter_type": "float"},
    {"name": "agitator_power", "type": "range", "bounds": (0.01, 0.5), "parameter_type": "float"},
    {"name": "substrate_feed_conc", "type": "range", "bounds": (500.0, 700.0), "parameter_type": "float"},
    {"name": "substrate_feed_rate", "type": "range", "bounds": (5.0, 6.5), "parameter_type": "float"},
]

BRANIN_PARAMS = [
    {"name": "x0", "type": "range", "bounds": (-5.0, 10.0), "parameter_type": "float"},
    {"name": "x1", "type": "range", "bounds": (0.0, 15.0), "parameter_type": "float"},
]

BRANINCURRIN_PARAMS = [
    {"name": "x0", "type": "range", "bounds": (0.0, 1.0), "parameter_type": "float"},
    {"name": "x1", "type": "range", "bounds": (0.0, 1.0), "parameter_type": "float"},
]

AX_FIXTURE_SPECS = [
    AxFixtureSpec(
        name="penicillin_modellist",
        description="Penicillin 7D, 3 objectives via Ax Client pipeline (BilogY + StandardizeY)",
        seed=42,
        n_sobol=25,
        n_bo=15,
        n_test=15,
        parameters=PENICILLIN_PARAMS,
        objective="penicillin_yield, fermentation_time, CO2_byproduct",
        eval_fn_name="penicillin",
    ),
    AxFixtureSpec(
        name="branin_ax",
        description="Branin 2D, single-objective via Ax Client pipeline",
        seed=100,
        n_sobol=10,
        n_bo=10,
        n_test=20,
        parameters=BRANIN_PARAMS,
        objective="branin",
        eval_fn_name="branin",
    ),
    AxFixtureSpec(
        name="branincurrin_ax",
        description="BraninCurrin 2D, 2-objective MOO via Ax Client pipeline",
        seed=200,
        n_sobol=15,
        n_bo=10,
        n_test=15,
        parameters=BRANINCURRIN_PARAMS,
        objective="branin, currin",
        eval_fn_name="branin_currin",
    ),
]


# ── Core generation logic ────────────────────────────────────────────────────

def _make_parameter_configs(params: list[dict]) -> list:
    """Convert param dicts to Ax RangeParameterConfig/ChoiceParameterConfig."""
    from ax.api.configs import RangeParameterConfig, ChoiceParameterConfig

    configs = []
    for p in params:
        if p["type"] == "range":
            configs.append(RangeParameterConfig(
                name=p["name"],
                bounds=p["bounds"],
                parameter_type=p["parameter_type"],
            ))
        elif p["type"] == "choice":
            configs.append(ChoiceParameterConfig(
                name=p["name"],
                values=p["values"],
                parameter_type=p.get("parameter_type", "str"),
                is_ordered=p.get("is_ordered", False),
            ))
    return configs


def _generate_test_points(params: list[dict], n_test: int, seed: int) -> Tensor:
    """Generate uniform random test points within parameter bounds."""
    torch.manual_seed(seed + 99999)
    d = len(params)
    bounds = []
    for p in params:
        if p["type"] == "range":
            bounds.append(p["bounds"])
        else:
            bounds.append((0, len(p["values"]) - 1))
    bounds_t = torch.tensor(bounds, dtype=torch.float64)
    u = torch.rand(n_test, d, dtype=torch.float64)
    return bounds_t[:, 0] + u * (bounds_t[:, 1] - bounds_t[:, 0])


def _generate_ax_fixture(spec: AxFixtureSpec) -> dict:
    """Generate a single fixture through the full Ax Client pipeline."""
    from ax.api import Client

    eval_fn = EVAL_FUNCTIONS[spec.eval_fn_name]
    param_configs = _make_parameter_configs(spec.parameters)

    # Set up Ax Client
    client = Client(random_seed=spec.seed)
    client.configure_experiment(parameters=param_configs, name=spec.name)
    client.configure_optimization(objective=spec.objective)

    # Run Sobol + BO trials
    n_total = spec.n_sobol + spec.n_bo
    for batch_start in range(0, n_total, 5):
        batch_size = min(5, n_total - batch_start)
        trials = client.get_next_trials(max_trials=batch_size)
        for trial_index, trial_params in trials.items():
            results = eval_fn(trial_params)
            client.complete_trial(trial_index=trial_index, raw_data=results)

    # Export through axjs_export
    experiment_state = export_client(client)

    # Generate test points
    test_X = _generate_test_points(spec.parameters, spec.n_test, spec.seed)

    # Get model-level expected predictions (in model's internal space)
    gs = client._generation_strategy
    adapter = getattr(gs, "adapter", None) or getattr(gs, "model", None)
    botorch_model = getattr(adapter, "botorch_model", None)
    if botorch_model is None:
        botorch_model = adapter.model.surrogate.model

    # For batched multi-output, we need to handle prediction carefully
    with torch.no_grad():
        if botorch_model.train_inputs[0].dim() == 3:
            # Batched model: predict each output separately
            n_outputs = botorch_model.train_inputs[0].shape[0]
            test_X_batched = test_X.unsqueeze(0).expand(n_outputs, -1, -1)
            posterior = botorch_model.posterior(test_X_batched)
            # posterior.mean shape: [n_outputs, n_test, 1]
            pred_means = posterior.mean.squeeze(-1).tolist()  # [n_outputs][n_test]
            pred_vars = posterior.variance.squeeze(-1).tolist()
        else:
            posterior = botorch_model.posterior(test_X)
            pred_mean = posterior.mean.squeeze(-1)
            pred_var = posterior.variance.squeeze(-1)
            if pred_mean.dim() == 1:
                pred_means = pred_mean.tolist()
                pred_vars = pred_var.tolist()
            else:
                pred_means = pred_mean.T.tolist()
                pred_vars = pred_var.T.tolist()

    # Build fixture
    import botorch
    import gpytorch as gpy

    test_metadata = {
        "botorch_version": botorch.__version__,
        "gpytorch_version": gpy.__version__,
        "torch_version": torch.__version__,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": spec.seed,
        "ax_pipeline": True,
    }

    fixture = {
        "experiment": experiment_state,
        "test": {
            "metadata": test_metadata,
            "test_points": test_X.tolist(),
            "expected": {
                "mean": pred_means,
                "variance": pred_vars,
            },
        },
    }

    return fixture


# ── Main ─────────────────────────────────────────────────────────────────────

def generate_all(
    output_dir: str = "test/fixtures",
    specs: list[str] | None = None,
) -> None:
    """Generate Ax pipeline fixtures and update manifest."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load existing manifest
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        existing_entries = {e["name"]: e for e in manifest["fixtures"]}
    else:
        existing_entries = {}

    # Filter specs if names provided
    fixture_specs = AX_FIXTURE_SPECS
    if specs:
        fixture_specs = [s for s in fixture_specs if s.name in specs]

    failed = []
    for spec in fixture_specs:
        filename = f"{spec.name}.json"
        filepath = out / filename

        print(f"Generating {spec.name} (Ax pipeline)...", end=" ", flush=True)
        try:
            fixture = _generate_ax_fixture(spec)
            with open(filepath, "w") as f:
                json.dump(fixture, f, indent=2)
            print(f"OK ({filepath})")

            # Update manifest entry
            existing_entries[spec.name] = {
                "name": spec.name,
                "file": filename,
                "description": spec.description,
            }
        except Exception as e:
            import traceback
            print(f"FAILED: {e}")
            traceback.print_exc()
            failed.append(spec.name)

    # Write updated manifest (sorted)
    all_entries = sorted(existing_entries.values(), key=lambda e: e["name"])
    with open(manifest_path, "w") as f:
        json.dump({"fixtures": all_entries}, f, indent=2)
    print(f"\nManifest updated: {manifest_path}")

    if failed:
        print(f"ERROR: {len(failed)} fixtures failed: {failed}", file=sys.stderr)
        sys.exit(1)

    print(f"\nAll {len(fixture_specs)} Ax fixtures generated successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate axjs Ax pipeline fixtures")
    parser.add_argument("--output-dir", default="test/fixtures", help="Output directory")
    parser.add_argument("--specs", nargs="*", help="Specific specs to generate (default: all)")
    args = parser.parse_args()
    generate_all(args.output_dir, args.specs)
