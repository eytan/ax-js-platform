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
    from botorch.test_functions.multi_objective import DTLZ2
    import torch

    dim = 5
    client = Client()
    client.configure_experiment(
        name="dtlz2_demo",
        parameters=[
            RangeParameterConfig(name=f"x{i+1}", parameter_type="float", bounds=(0.0, 1.0))
            for i in range(dim)
        ],
    )
    client.configure_optimization(objective="f0, f1")

    fn = DTLZ2(dim=dim, num_objectives=2, negate=True)
    for i in range(30):
        trials = client.get_next_trials(max_trials=1)
        for trial_index, params in trials.items():
            x = torch.tensor([[params[f"x{j+1}"] for j in range(dim)]])
            y = fn(x).squeeze()
            client.complete_trial(trial_index=trial_index, raw_data={
                "f0": y[0].item(),
                "f1": y[1].item(),
            })

    from axjs_export import export_client
    return export_client(client)


def _viz_cell(code, state, viz_code, **kw):
    cell = new_code_cell(code)
    html = _render(state, viz_code, **kw)
    cell.outputs = [_output(html)]
    return cell


def build_notebook():
    print("Running Ax experiment (30 DTLZ2 5D MOO trials)...")
    state = _generate_experiment_state()
    ms = state['model_state']
    n_train = len(ms.get('train_X', ms.get('models', [{}])[0].get('train_X', [])))
    print(f"Exported: {n_train} training points, {ms['model_type']}")

    nb = new_notebook()
    nb.metadata.kernelspec = {
        "display_name": "Python 3", "language": "python", "name": "python3"
    }

    nb.cells.append(new_markdown_cell(
        "# ax-js Jupyter Demo\n\n"
        "Interactive GP diagnostics from a DTLZ2 multi-objective experiment (5D, 2 objectives, 30 trials).\n"
        "Pre-populated outputs — no execution needed. Or re-run to see live results."
    ))

    # Experiment setup cell
    nb.cells.append(new_code_cell(
        "from ax.api import Client\n"
        "from ax.api.configs import RangeParameterConfig\n"
        "from botorch.test_functions.multi_objective import DTLZ2\n"
        "import torch\n"
        "\n"
        "dim = 5\n"
        "client = Client()\n"
        "client.configure_experiment(\n"
        "    name='dtlz2_demo',\n"
        "    parameters=[\n"
        "        RangeParameterConfig(name=f'x{i+1}', parameter_type='float', bounds=(0.0, 1.0))\n"
        "        for i in range(dim)\n"
        "    ],\n"
        ")\n"
        "client.configure_optimization(objective='f0, f1')\n"
        "\n"
        "fn = DTLZ2(dim=dim, num_objectives=2, negate=True)\n"
        "for i in range(30):\n"
        "    trials = client.get_next_trials(max_trials=1)\n"
        "    for trial_index, params in trials.items():\n"
        "        x = torch.tensor([[params[f'x{j+1}'] for j in range(dim)]])\n"
        "        y = fn(x).squeeze()\n"
        "        client.complete_trial(trial_index=trial_index, raw_data={'f0': y[0].item(), 'f1': y[1].item()})\n"
        "\n"
        "print(f'Completed 30 trials')"
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
        height="auto"))

    nb.cells.append(new_markdown_cell("## 2D Response Surface"))
    nb.cells.append(_viz_cell(
        "response_surface(client)", state,
        'Ax.viz.renderResponseSurface(c,p,{interactive:true,width:800,height:380});',
        width="860px", height="500px"))

    nb.cells.append(new_markdown_cell("## Feature Importance"))
    nb.cells.append(_viz_cell(
        "feature_importance(client)", state,
        'Ax.viz.renderFeatureImportance(c,p,{interactive:true});',
        width="500px", height="auto"))

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
        "5D DTLZ2 MOO exercises outcome selectors, sliders, importance rankings, and axis selectors."))

    return nb


def main():
    nb = build_notebook()
    nb_path = ROOT / "demo" / "jupyter-demo.ipynb"
    nbformat.write(nb, str(nb_path))
    print(f"Notebook: {nb_path} ({nb_path.stat().st_size // 1024}KB)")

    try:
        from nbconvert import HTMLExporter
        exporter = HTMLExporter()
        exporter.theme = "light"
        body, _ = exporter.from_notebook_node(nb)
        html_path = ROOT / "demo" / "jupyter-demo.html"
        html_path.write_text(body)
        print(f"HTML: {html_path} ({html_path.stat().st_size // 1024}KB)")
    except ImportError:
        print("nbconvert not available")


if __name__ == "__main__":
    main()
