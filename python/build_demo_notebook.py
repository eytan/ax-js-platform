#!/usr/bin/env python3
"""
Build pre-populated Jupyter demo notebook.

Creates an Ax experiment, runs BO, exports the model, and generates
a notebook with pre-populated visualization outputs. The notebook
works without execution — just open and see the plots.

Usage:
    python python/build_demo_notebook.py
"""

from __future__ import annotations
import json, sys
from pathlib import Path
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "python"))
from axjs_jupyter import _load_bundles, _render


def _output(html):
    return nbformat.v4.new_output(
        output_type="display_data", data={"text/html": html}
    )


def _generate_experiment_state():
    """Run a real Ax experiment and export the state."""
    from ax.api import Client
    from ax.api.configs import RangeParameterConfig
    from botorch.test_functions import Branin
    import torch

    client = Client()
    client.configure_experiment(
        name="branin_demo",
        parameters=[
            RangeParameterConfig(name="x1", parameter_type="float", bounds=(-5.0, 10.0)),
            RangeParameterConfig(name="x2", parameter_type="float", bounds=(0.0, 15.0)),
        ],
    )
    client.configure_optimization(objective="branin")

    branin = Branin(negate=True)
    for i in range(25):
        trials = client.get_next_trials(max_trials=1)
        for trial_index, params in trials.items():
            y = branin(torch.tensor([[params["x1"], params["x2"]]])).item()
            client.complete_trial(trial_index=trial_index, raw_data={"branin": y})

    from axjs_export import export_client
    return export_client(client)


def _viz_cell(code, state, viz_code, **kw):
    cell = new_code_cell(code)
    html = _render(state, viz_code, **kw)
    cell.outputs = [_output(html)]
    return cell


def build_notebook():
    print("Running Ax experiment (25 trials)...")
    state = _generate_experiment_state()
    print(f"Exported: {len(state['model_state']['train_X'])} training points")

    nb = new_notebook()
    nb.metadata.kernelspec = {
        "display_name": "Python 3", "language": "python", "name": "python3"
    }

    nb.cells.append(new_markdown_cell(
        "# ax-js Jupyter Demo\n\n"
        "Interactive GP diagnostics from a Branin BO experiment (25 trials).\n"
        "Pre-populated outputs — no execution needed. Or re-run to see live results."
    ))

    # Experiment setup cell
    nb.cells.append(new_code_cell(
        "from ax.api import Client\n"
        "from ax.api.configs import RangeParameterConfig\n"
        "from botorch.test_functions import Branin\n"
        "import torch\n"
        "\n"
        "client = Client()\n"
        "client.configure_experiment(\n"
        "    name='branin_demo',\n"
        "    parameters=[\n"
        "        RangeParameterConfig(name='x1', parameter_type='float', bounds=(-5.0, 10.0)),\n"
        "        RangeParameterConfig(name='x2', parameter_type='float', bounds=(0.0, 15.0)),\n"
        "    ],\n"
        ")\n"
        "client.configure_optimization(objective='branin')\n"
        "\n"
        "branin = Branin(negate=True)\n"
        "for i in range(25):\n"
        "    trials = client.get_next_trials(max_trials=1)\n"
        "    for trial_index, params in trials.items():\n"
        "        y = branin(torch.tensor([[params['x1'], params['x2']]])).item()\n"
        "        client.complete_trial(trial_index=trial_index, raw_data={'branin': y})\n"
        "\n"
        "print(f'Completed 25 trials')"
    ))

    # Import cell
    nb.cells.append(new_code_cell(
        "import sys; sys.path.insert(0, 'python')\n"
        "from axjs_jupyter import (\n"
        "    slice_plot, response_surface,\n"
        "    feature_importance, cross_validation, optimization_trace,\n"
        ")"
    ))

    nb.cells.append(new_markdown_cell("## 1D Slice Plots"))
    nb.cells.append(_viz_cell(
        "slice_plot(client)", state,
        'Ax.viz.renderSlicePlot(c,p,{interactive:true});',
        height="500px"))

    nb.cells.append(new_markdown_cell("## 2D Response Surface"))
    nb.cells.append(_viz_cell(
        "response_surface(client)", state,
        'Ax.viz.renderResponseSurface(c,p,{interactive:true,width:460,height:460});',
        width="500px", height="520px"))

    nb.cells.append(new_markdown_cell("## Feature Importance"))
    nb.cells.append(_viz_cell(
        "feature_importance(client)", state,
        'Ax.viz.renderFeatureImportance(c,p,{interactive:true});',
        width="500px", height="280px"))

    nb.cells.append(new_markdown_cell("## Leave-One-Out Cross-Validation"))
    nb.cells.append(_viz_cell(
        "cross_validation(client)", state,
        'Ax.viz.renderCrossValidation(c,p,{interactive:true,width:460,height:460});',
        width="500px", height="500px"))

    nb.cells.append(new_markdown_cell("## Optimization Trace"))
    nb.cells.append(_viz_cell(
        "optimization_trace(client)", state,
        'Ax.viz.renderOptimizationTrace(c,p,{interactive:true,width:660,height:380});',
        width="700px", height="420px"))

    nb.cells.append(new_markdown_cell(
        "---\nFive one-liners from Ax Client to interactive plots. "
        "All rendering is client-side JavaScript."))

    return nb


def main():
    nb = build_notebook()
    nb_path = ROOT / "demo" / "jupyter-demo.ipynb"
    nbformat.write(nb, str(nb_path))
    print(f"Notebook: {nb_path} ({nb_path.stat().st_size // 1024}KB)")

    try:
        from nbconvert import HTMLExporter
        exporter = HTMLExporter()
        exporter.template_name = "classic"
        body, _ = exporter.from_notebook_node(nb)
        html_path = ROOT / "demo" / "jupyter-demo.html"
        html_path.write_text(body)
        print(f"HTML: {html_path} ({html_path.stat().st_size // 1024}KB)")
    except ImportError:
        print("nbconvert not available")


if __name__ == "__main__":
    main()
