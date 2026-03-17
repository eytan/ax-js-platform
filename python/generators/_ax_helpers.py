"""Ax adapter helpers for fixture generation.

Functions for building Ax transform lists and composing UnitX normalization
into exported model states, so axjs can predict on raw parameter values.
"""

from __future__ import annotations

from typing import Any

from torch import Tensor


def _build_ax_transforms(
    adapter_transform_names: list[str] | None,
    outcome_names: list[str] | None = None,
) -> tuple[list, dict]:
    """Build ordered Ax transform list and configs for adapter creation.

    Always includes UnitX (input normalization) and StandardizeY (output
    normalization). Inserts requested adapter Y-transforms (LogY, BilogY,
    PowerTransformY) before StandardizeY so they compose correctly.

    Returns (transforms, transform_configs) where transform_configs maps
    transform names to their config dicts (e.g., LogY needs metrics list).
    """
    try:
        from ax.adapter.transforms.unit_x import UnitX
        from ax.adapter.transforms.standardize_y import StandardizeY
    except ImportError:
        from ax.modelbridge.transforms.unit_x import UnitX
        from ax.modelbridge.transforms.standardize_y import StandardizeY

    y_tf_map: dict[str, type] = {}
    for name, modpath in [
        ("LogY", "log_y"),
        ("BilogY", "bilog_y"),
        ("PowerTransformY", "power_transform_y"),
    ]:
        try:
            mod = __import__(f"ax.adapter.transforms.{modpath}", fromlist=[name])
            y_tf_map[name] = getattr(mod, name)
        except ImportError:
            try:
                mod = __import__(f"ax.modelbridge.transforms.{modpath}", fromlist=[name])
                y_tf_map[name] = getattr(mod, name)
            except ImportError:
                pass

    # Build: UnitX → [requested Y-transforms] → StandardizeY
    transforms: list[type] = [UnitX]
    transform_configs: dict[str, dict] = {}
    has_standardize_y = False
    for name in (adapter_transform_names or []):
        if name == "StandardizeY":
            has_standardize_y = True
            transforms.append(StandardizeY)
        elif name in y_tf_map:
            transforms.append(y_tf_map[name])
            # Y-transforms (LogY, BilogY, PowerTransformY) need to know
            # which metrics to transform. Without config, they're no-ops.
            if outcome_names:
                transform_configs[name] = {"metrics": list(outcome_names)}
    if not has_standardize_y:
        transforms.append(StandardizeY)

    return transforms, transform_configs


def _compose_unitx_into_normalize(model_state: dict, bounds: Tensor) -> None:
    """Ensure the exported model's input_transform maps from raw parameter space.

    When fitted through Ax, the model receives UnitX-transformed inputs ([0,1]).
    Two cases:
    1. Model has NO input_transform (Ax 1.2+): Create one from search space bounds
       so axjs can normalize raw parameters to [0,1].
    2. Model HAS input_transform (Normalize): Compose UnitX into it so the
       combined transform maps raw params → normalized space.
    """
    if model_state.get("model_type") == "ModelListGP":
        for sub_model in model_state.get("models", []):
            _compose_unitx_single(sub_model, bounds)
    else:
        _compose_unitx_single(model_state, bounds)


def _compose_unitx_single(state: dict, bounds: Tensor) -> None:
    """Compose UnitX into a single model's input_transform and un-UnitX train_X.

    After export, train_X is in UnitX space ([0,1]) — either because:
    - Model had no input_transform (Ax 1.2+), so train_X stays as Ax passed it
    - Model had Normalize, and _untransform_train_X undid it back to UnitX space

    We map train_X to raw parameter space and ensure the input_transform
    maps from raw space → the space the model expects.
    """
    lo = bounds[0].tolist()
    hi = bounds[1].tolist()
    d = len(lo)

    # Un-UnitX train_X: x_raw = x_unitx * (hi - lo) + lo
    if "train_X" in state:
        for pt in state["train_X"]:
            for i in range(min(d, len(pt))):
                pt[i] = pt[i] * (hi[i] - lo[i]) + lo[i]

    itf = state.get("input_transform")
    if itf is None:
        # No model-level Normalize — create one equivalent to UnitX
        state["input_transform"] = {
            "offset": lo[:],
            "coefficient": [hi[i] - lo[i] for i in range(d)],
        }
    else:
        # Compose: combined = Normalize ∘ UnitX
        offset = itf["offset"]
        coeff = itf["coefficient"]
        n = min(d, len(offset))
        for i in range(n):
            span = hi[i] - lo[i]
            if span > 0:
                offset[i] = lo[i] + offset[i] * span
                coeff[i] = coeff[i] * span
