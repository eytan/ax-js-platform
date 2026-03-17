#!/usr/bin/env python3
"""
Build an end-to-end demo notebook: Ax experiment → ax-js visualization.

Usage:
    python python/build_e2e_notebook.py
    # or: npm run build:e2e

Output:
    demo/ax-js-e2e.ipynb  — executable notebook (run in Jupyter)
"""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

ROOT = Path(__file__).parent.parent


def build_notebook() -> nbformat.NotebookNode:
    nb = new_notebook()
    nb.metadata.kernelspec = {
        "display_name": "Python 3", "language": "python", "name": "python3"
    }

    nb.cells.append(new_markdown_cell(
        "# ax-js End-to-End Demo\n\n"
        "Set up an Ax experiment, run Bayesian optimization, and visualize "
        "the fitted GP — all client-side.\n\n"
        "**Requirements**: `pip install ax-platform botorch`"
    ))

    nb.cells.append(new_markdown_cell("## 1. Create Experiment & Run BO"))
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
        "    params, trial_index = client.get_next_trial()\n"
        "    y = branin(torch.tensor([[params['x1'], params['x2']]])).item()\n"
        "    client.complete_trial(trial_index=trial_index, raw_data={'branin': y})\n"
        "\n"
        "print(f'Completed 25 trials')"
    ))

    nb.cells.append(new_markdown_cell("## 2. Load ax-js"))
    nb.cells.append(new_code_cell(
        "import sys; sys.path.insert(0, 'python')\n"
        "from axjs_jupyter import (\n"
        "    setup_axjs,\n"
        "    slice_plot, response_surface,\n"
        "    feature_importance, cross_validation, optimization_trace,\n"
        ")\n"
        "\n"
        "setup_axjs(client)"
    ))

    nb.cells.append(new_markdown_cell("## 3. Visualize"))

    nb.cells.append(new_code_cell("slice_plot()"))
    nb.cells.append(new_code_cell("response_surface()"))
    nb.cells.append(new_code_cell("feature_importance()"))
    nb.cells.append(new_code_cell("cross_validation()"))
    nb.cells.append(new_code_cell("optimization_trace()"))

    nb.cells.append(new_markdown_cell(
        "---\n"
        "That's it — five one-liners to go from a fitted Ax experiment to "
        "interactive GP diagnostics. All rendering happens in JavaScript."
    ))

    return nb


def main():
    nb = build_notebook()
    nb_path = ROOT / "demo" / "ax-js-e2e.ipynb"
    nbformat.write(nb, str(nb_path))
    print(f"Notebook: {nb_path}")
    print("Run: jupyter notebook demo/ax-js-e2e.ipynb")


if __name__ == "__main__":
    main()
