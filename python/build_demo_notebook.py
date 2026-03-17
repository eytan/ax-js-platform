#!/usr/bin/env python3
"""
Build the ax-js Jupyter demo notebook with pre-populated outputs.

Usage:
    python python/build_demo_notebook.py
    # or: npm run build:notebook

Output:
    demo/jupyter-demo.ipynb  — notebook with pre-populated outputs
    demo/jupyter-demo.html   — standalone HTML export
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

ROOT = Path(__file__).parent.parent
DIST = ROOT / "dist"
FIXTURES = ROOT / "test" / "fixtures"

# Make axjs_jupyter importable
sys.path.insert(0, str(ROOT / "python"))
from axjs_jupyter import _load_bundles, _render, _SETUP_DONE
import axjs_jupyter  # noqa: E402


def _load_fixture(name="penicillin_modellist.json"):
    data = json.loads((FIXTURES / name).read_text())
    return data["experiment"] if "experiment" in data else data


def _output(html):
    return nbformat.v4.new_output(
        output_type="display_data", data={"text/html": html}
    )


def _viz_cell(code, viz_call, state, **render_kw):
    cell = new_code_cell(code)
    html = _render(viz_call, use_global_state=True, state=state, **render_kw)
    cell.outputs = [_output(html)]
    return cell


def build_notebook():
    ax_js, viz_js = _load_bundles()
    state = _load_fixture("penicillin_modellist.json")
    state_json = json.dumps(state)
    outcomes = state.get("outcome_names", ["y"])

    nb = new_notebook()
    nb.metadata.kernelspec = {
        "display_name": "Python 3", "language": "python", "name": "python3"
    }

    nb.cells.append(new_markdown_cell(
        "# ax-js Jupyter Demo\n\n"
        "Interactive GP diagnostics rendered client-side.\n\n"
        f"**Fixture**: Penicillin benchmark ({len(outcomes)} outcomes: "
        f"{', '.join(outcomes)})\n\n"
        "Export with `jupyter nbconvert --to html` for a standalone page."
    ))

    # Setup
    setup_html = (
        f"<script>\n{ax_js}\n{viz_js}\n"
        f"window.__AXJS_STATE__={state_json};\n"
        f"</script>"
        '<div style="color:#888;font-size:12px">'
        f"ax-js loaded. {len(outcomes)} outcomes available.</div>"
    )
    setup_cell = new_code_cell(
        "import sys; sys.path.insert(0, 'python')\n"
        "from axjs_jupyter import setup_axjs\n"
        "import json\n\n"
        "state = json.load(open('test/fixtures/penicillin_modellist.json'))['experiment']\n"
        "setup_axjs(state)"
    )
    setup_cell.outputs = [_output(setup_html)]
    axjs_jupyter._SETUP_DONE = True  # so subsequent _render calls skip bundle inlining
    nb.cells.append(setup_cell)

    import_cell = new_code_cell(
        "from axjs_jupyter import (\n"
        "    slice_plot, response_surface,\n"
        "    feature_importance, cross_validation, optimization_trace,\n"
        ")"
    )
    import_cell.outputs = []
    nb.cells.append(import_cell)

    nb.cells.append(new_markdown_cell("## 1D Slice Plots"))
    nb.cells.append(_viz_cell(
        "slice_plot()", 'Ax.viz.renderSlicePlot(c,p,{interactive:true});',
        state, height="600px"))

    nb.cells.append(new_markdown_cell("## 2D Response Surface"))
    nb.cells.append(_viz_cell(
        "response_surface()",
        'Ax.viz.renderResponseSurface(c,p,{interactive:true,width:460,height:460});',
        state, width="500px", height="520px"))

    nb.cells.append(new_markdown_cell("## Feature Importance"))
    nb.cells.append(_viz_cell(
        "feature_importance()",
        'Ax.viz.renderFeatureImportance(c,p,{interactive:true});',
        state, width="500px", height="280px"))

    nb.cells.append(new_markdown_cell("## Leave-One-Out Cross-Validation"))
    nb.cells.append(_viz_cell(
        "cross_validation()",
        'Ax.viz.renderCrossValidation(c,p,{interactive:true,width:460,height:460});',
        state, width="500px", height="500px"))

    nb.cells.append(new_markdown_cell("## Optimization Trace"))
    nb.cells.append(_viz_cell(
        "optimization_trace()",
        'Ax.viz.renderOptimizationTrace(c,p,{interactive:true,width:660,height:380});',
        state, width="700px", height="420px"))

    nb.cells.append(new_markdown_cell(
        "---\n## About\n\n"
        "All visualizations rendered by "
        "[ax-js](https://github.com/eytan/ax-js-platform). "
        "GP predictions computed entirely in JavaScript."
    ))

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

        for tag in ["Ax.Predictor", "renderSlicePlot", "renderResponseSurface"]:
            assert tag in body, f"Missing: {tag}"
        print("Verification: all expected content present")
    except ImportError:
        print("nbconvert not available — run: jupyter nbconvert --to html", nb_path)


if __name__ == "__main__":
    main()
