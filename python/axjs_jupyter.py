"""
Jupyter notebook integration for ax-js visualizations.

Renders interactive GP visualizations in Jupyter cells. The JS bundles
are inlined so notebooks work offline and export to standalone HTML.

Usage:
    from axjs_jupyter import setup_axjs, slice_plot, response_surface

    # After running an Ax experiment:
    setup_axjs(client)

    slice_plot()
    response_surface()
    feature_importance()
    cross_validation()
    optimization_trace()

Or pass the client per-call:
    slice_plot(client, outcome="accuracy")
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Optional

_DIST_DIR = Path(__file__).parent / "../dist"
_AX_JS: Optional[str] = None
_AX_VIZ_JS: Optional[str] = None
_STATE: Optional[dict] = None
_SETUP_DONE = False


def _load_bundles() -> tuple[str, str]:
    global _AX_JS, _AX_VIZ_JS
    if _AX_JS is None:
        _AX_JS = (_DIST_DIR / "ax.js").read_text()
    if _AX_VIZ_JS is None:
        _AX_VIZ_JS = (_DIST_DIR / "ax-viz.js").read_text()
    return _AX_JS, _AX_VIZ_JS


def _export(client_or_state: Any) -> dict:
    """Accept Ax Client, ExperimentState dict, or JSON file path."""
    if isinstance(client_or_state, dict):
        return client_or_state
    if isinstance(client_or_state, (str, Path)):
        data = json.loads(Path(client_or_state).read_text())
        return data.get("experiment", data)
    from axjs_export import export_client
    return export_client(client_or_state)


def _cid() -> str:
    return f"axjs_{uuid.uuid4().hex[:8]}"


def setup_axjs(client_or_state: Any) -> None:
    """Load ax-js bundles and export model state. Call once per notebook.

    Args:
        client_or_state: An ``ax.api.Client``, ``ExperimentState`` dict,
            or path to a JSON file. The model is exported and stored for
            subsequent plot calls.
    """
    from IPython.display import display, HTML

    global _STATE, _SETUP_DONE
    _STATE = _export(client_or_state)
    ax_js, viz_js = _load_bundles()

    display(HTML(
        f"<script>\n{ax_js}\n{viz_js}\n"
        f"window.__AXJS_STATE__={json.dumps(_STATE)};\n</script>"
        '<div style="color:#888;font-size:12px">ax-js loaded.</div>'
    ))
    _SETUP_DONE = True


def _get_state(client_or_state: Any = None) -> dict:
    if client_or_state is not None:
        return _export(client_or_state)
    if _STATE is not None:
        return _STATE
    raise ValueError("No state. Pass client or call setup_axjs(client) first.")


def _render(viz_code: str, width: str = "100%", height: str = "400px",
            use_global_state: bool = True, state: Optional[dict] = None) -> str:
    cid = _cid()
    ax_js, viz_js = _load_bundles()
    bundle = "" if _SETUP_DONE else f"if(!window.Ax){{{ax_js}\n{viz_js}}}"
    state_line = ("var state=window.__AXJS_STATE__;" if use_global_state
                  else f"var state={json.dumps(state)};")

    return (
        f'<div id="{cid}" style="width:{width};min-height:{height};'
        f'position:relative;background:#0f0f11;border-radius:8px;'
        f'overflow:visible;padding:12px"></div>'
        f'<script>(function(){{'
        f'{bundle}{state_line}'
        f'var c=document.getElementById("{cid}");'
        f'var p=new Ax.Predictor(state);'
        f'{viz_code}'
        f'}})()</script>'
    )


def _show(html: str) -> None:
    from IPython.display import display, HTML
    display(HTML(html))


def _opts(outcome: Optional[str], extra: str = "") -> str:
    parts = ["interactive:true"]
    if outcome:
        parts.append(f'outcome:"{outcome}"')
    if extra:
        parts.append(extra)
    return ",".join(parts)


# ── Plot functions ─────────────────────────────────────────────────────────


def slice_plot(client_or_state: Any = None, *, outcome: Optional[str] = None) -> None:
    """1D posterior slices for each parameter, sorted by importance."""
    state = _get_state(client_or_state)
    use_global = client_or_state is None and _STATE is not None
    _show(_render(f"Ax.viz.renderSlicePlot(c,p,{{{_opts(outcome)}}});",
                  height="600px", use_global_state=use_global, state=state))


def response_surface(client_or_state: Any = None, *, outcome: Optional[str] = None) -> None:
    """2D posterior mean heatmap, auto-selects most important dimensions."""
    state = _get_state(client_or_state)
    use_global = client_or_state is None and _STATE is not None
    _show(_render(f"Ax.viz.renderResponseSurface(c,p,{{{_opts(outcome, 'width:460,height:460')}}});",
                  width="500px", height="520px", use_global_state=use_global, state=state))


def feature_importance(client_or_state: Any = None, *, outcome: Optional[str] = None) -> None:
    """Dimension importance from kernel lengthscales."""
    state = _get_state(client_or_state)
    use_global = client_or_state is None and _STATE is not None
    _show(_render(f"Ax.viz.renderFeatureImportance(c,p,{{{_opts(outcome)}}});",
                  width="500px", height="280px", use_global_state=use_global, state=state))


def cross_validation(client_or_state: Any = None, *, outcome: Optional[str] = None) -> None:
    """LOO cross-validation: observed vs predicted with CI."""
    state = _get_state(client_or_state)
    use_global = client_or_state is None and _STATE is not None
    _show(_render(f"Ax.viz.renderCrossValidation(c,p,{{{_opts(outcome, 'width:460,height:460')}}});",
                  width="500px", height="500px", use_global_state=use_global, state=state))


def optimization_trace(client_or_state: Any = None, *, outcome: Optional[str] = None) -> None:
    """Trial progression with best-so-far tracking."""
    state = _get_state(client_or_state)
    use_global = client_or_state is None and _STATE is not None
    _show(_render(f"Ax.viz.renderOptimizationTrace(c,p,{{{_opts(outcome, 'width:660,height:380')}}});",
                  width="700px", height="420px", use_global_state=use_global, state=state))


def all_diagnostics(client_or_state: Any = None, *, outcome: Optional[str] = None) -> None:
    """All diagnostic plots: slice, surface, importance, CV, trace."""
    slice_plot(client_or_state, outcome=outcome)
    response_surface(client_or_state, outcome=outcome)
    feature_importance(client_or_state, outcome=outcome)
    cross_validation(client_or_state, outcome=outcome)
    optimization_trace(client_or_state, outcome=outcome)
