#!/usr/bin/env python3
"""
Build an end-to-end demo notebook: Ax experiment → ax-js visualization.

Creates a Jupyter notebook that:
1. Sets up a Branin benchmark with ax.api.Client
2. Runs 15 Sobol + 10 BO trials
3. Exports the fitted model via axjs_export
4. Renders interactive ax-js visualizations in the notebook

Usage:
    python python/build_e2e_notebook.py

Output:
    demo/ax-js-e2e.ipynb  — executable notebook
    demo/ax-js-e2e.html   — standalone HTML export (after execution)

NOTE: Unlike build_demo_notebook.py which pre-populates outputs,
this notebook must be EXECUTED in Jupyter to see results. The cells
contain real Python code that fits a GP and exports it.
"""

from __future__ import annotations

import json
from pathlib import Path

import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

ROOT = Path(__file__).parent.parent
DIST = ROOT / "dist"


def build_notebook() -> nbformat.NotebookNode:
    nb = new_notebook()
    nb.metadata.kernelspec = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }

    # Title
    nb.cells.append(new_markdown_cell(
        "# ax-js End-to-End Demo\n\n"
        "Complete workflow: set up an Ax experiment, run Bayesian optimization, "
        "export the fitted GP model, and visualize predictions interactively — "
        "all client-side.\n\n"
        "**Requirements**: `pip install ax-platform botorch`"
    ))

    # Step 1: Set up experiment
    nb.cells.append(new_markdown_cell("## 1. Set Up Experiment"))
    nb.cells.append(new_code_cell(
        "from ax.api import Client\n"
        "from ax.api.configs import RangeParameterConfig\n"
        "\n"
        "client = Client()\n"
        "\n"
        "# Branin function: 2D continuous optimization\n"
        "client.configure_experiment(\n"
        "    name='branin_demo',\n"
        "    parameters=[\n"
        "        RangeParameterConfig(name='x1', parameter_type='float', bounds=(-5.0, 10.0)),\n"
        "        RangeParameterConfig(name='x2', parameter_type='float', bounds=(0.0, 15.0)),\n"
        "    ],\n"
        "    description='Branin benchmark for ax-js demo',\n"
        ")\n"
        "client.configure_optimization(objective='branin', outcome_constraints=[])\n"
        "print(f'Experiment configured: {client._experiment.name}')"
    ))

    # Step 2: Run trials
    nb.cells.append(new_markdown_cell("## 2. Run Bayesian Optimization"))
    nb.cells.append(new_code_cell(
        "from botorch.test_functions import Branin\n"
        "import torch\n"
        "\n"
        "branin_fn = Branin(negate=True)  # negate for maximization\n"
        "\n"
        "# Run 25 trials (Sobol init + BO)\n"
        "for i in range(25):\n"
        "    params, trial_index = client.get_next_trial()\n"
        "    x = torch.tensor([[params['x1'], params['x2']]])\n"
        "    y = branin_fn(x).item()\n"
        "    client.complete_trial(\n"
        "        trial_index=trial_index,\n"
        "        raw_data={'branin': y},\n"
        "    )\n"
        "    if (i + 1) % 5 == 0:\n"
        "        print(f'  Trial {i+1}: x1={params[\"x1\"]:.3f}, x2={params[\"x2\"]:.3f}, y={y:.3f}')\n"
        "\n"
        "print(f'\\nCompleted {i+1} trials')"
    ))

    # Step 3: Export model
    nb.cells.append(new_markdown_cell(
        "## 3. Export Model to ax-js\n\n"
        "The export captures everything needed for client-side prediction: "
        "kernel hyperparameters, training data, input/output transforms."
    ))
    nb.cells.append(new_code_cell(
        "import sys, json\n"
        "sys.path.insert(0, 'python')  # for axjs_export\n"
        "from axjs_export import export_client\n"
        "\n"
        "state = export_client(client)\n"
        "print(f'Exported: {len(state[\"model_state\"][\"train_X\"])} training points')\n"
        "print(f'Kernel: {state[\"model_state\"][\"kernel\"][\"type\"]}')\n"
        "print(f'Outcome: {state[\"outcome_names\"]}')\n"
        "\n"
        "# Save to JSON (optional)\n"
        "with open('demo/branin_export.json', 'w') as f:\n"
        "    json.dump(state, f)\n"
        "print(f'Saved to demo/branin_export.json ({len(json.dumps(state))//1024}KB)')"
    ))

    # Step 4: Load ax-js bundles
    nb.cells.append(new_markdown_cell(
        "## 4. Visualize with ax-js\n\n"
        "Load the ax-js JavaScript bundles and render interactive plots — "
        "all predictions computed client-side in the browser."
    ))

    ax_js = (DIST / "ax.js").read_text()
    viz_js = (DIST / "ax-viz.js").read_text()

    nb.cells.append(new_code_cell(
        "from IPython.display import display, HTML\n"
        "from pathlib import Path\n"
        "\n"
        "# Load ax-js bundles\n"
        "ax_js = Path('dist/ax.js').read_text()\n"
        "viz_js = Path('dist/ax-viz.js').read_text()\n"
        "state_json = json.dumps(state)\n"
        "\n"
        "# Initialize ax-js in the notebook\n"
        "display(HTML(f'<script>{ax_js}\\n{viz_js}\\nwindow.__AXJS_STATE__={state_json}</script>'\n"
        "             '<div style=\"color:#888;font-size:12px\">ax-js loaded.</div>'))"
    ))

    # Step 5: Slice plots
    nb.cells.append(new_markdown_cell("### 1D Slice Plots"))
    nb.cells.append(new_code_cell(
        "display(HTML('''\n"
        "<div id=\"sp\" style=\"width:100%;min-height:400px;position:relative;"
        "background:#0f0f11;border-radius:8px;overflow:visible;padding:8px\"></div>\n"
        "<script>(function(){\n"
        "  var c=document.getElementById('sp');\n"
        "  var p=new Ax.Predictor(window.__AXJS_STATE__);\n"
        "  Ax.viz.renderSlicePlot(c,p,{interactive:true});\n"
        "})()</script>\n"
        "'''))"
    ))

    # Step 6: Response surface
    nb.cells.append(new_markdown_cell("### 2D Response Surface"))
    nb.cells.append(new_code_cell(
        "display(HTML('''\n"
        "<div id=\"rs\" style=\"width:500px;min-height:500px;position:relative;"
        "background:#0f0f11;border-radius:8px;overflow:visible;padding:8px\"></div>\n"
        "<script>(function(){\n"
        "  var c=document.getElementById('rs');\n"
        "  var p=new Ax.Predictor(window.__AXJS_STATE__);\n"
        "  Ax.viz.renderResponseSurface(c,p,{interactive:true,width:460,height:460});\n"
        "})()</script>\n"
        "'''))"
    ))

    # Step 7: Cross-validation
    nb.cells.append(new_markdown_cell("### Leave-One-Out Cross-Validation"))
    nb.cells.append(new_code_cell(
        "display(HTML('''\n"
        "<div id=\"cv\" style=\"width:450px;min-height:450px;position:relative;"
        "background:#0f0f11;border-radius:8px;overflow:visible;padding:8px\"></div>\n"
        "<script>(function(){\n"
        "  var c=document.getElementById('cv');\n"
        "  var p=new Ax.Predictor(window.__AXJS_STATE__);\n"
        "  Ax.viz.renderCrossValidation(c,p,{interactive:true,width:420,height:420});\n"
        "})()</script>\n"
        "'''))"
    ))

    # Step 8: Feature importance + optimization trace
    nb.cells.append(new_markdown_cell("### Feature Importance & Optimization Trace"))
    nb.cells.append(new_code_cell(
        "display(HTML('''\n"
        "<div id=\"fi\" style=\"width:100%;min-height:250px;position:relative;"
        "background:#0f0f11;border-radius:8px;overflow:visible;padding:8px;margin-bottom:12px\"></div>\n"
        "<div id=\"ot\" style=\"width:700px;min-height:400px;position:relative;"
        "background:#0f0f11;border-radius:8px;overflow:visible;padding:8px\"></div>\n"
        "<script>(function(){\n"
        "  var p=new Ax.Predictor(window.__AXJS_STATE__);\n"
        "  Ax.viz.renderFeatureImportance(document.getElementById('fi'),p,{interactive:true});\n"
        "  Ax.viz.renderOptimizationTrace(document.getElementById('ot'),p,{interactive:true,width:660,height:360});\n"
        "})()</script>\n"
        "'''))"
    ))

    # About
    nb.cells.append(new_markdown_cell(
        "---\n"
        "## About\n\n"
        "This notebook demonstrates the full ax-js workflow:\n"
        "1. **Set up** an Ax experiment with `ax.api.Client`\n"
        "2. **Run** Bayesian optimization (Sobol init → GP-based BO)\n"
        "3. **Export** the fitted model to JSON via `axjs_export.export_client()`\n"
        "4. **Visualize** predictions interactively — all rendering happens "
        "in JavaScript, no Python backend needed\n\n"
        "Export this notebook to HTML with `jupyter nbconvert --to html` for a "
        "standalone page (after executing all cells)."
    ))

    return nb


def main():
    nb = build_notebook()
    nb_path = ROOT / "demo" / "ax-js-e2e.ipynb"
    nbformat.write(nb, str(nb_path))
    print(f"Notebook: {nb_path}")
    print("Run: jupyter notebook demo/ax-js-e2e.ipynb")
    print("Then execute all cells to see the visualizations.")


if __name__ == "__main__":
    main()
