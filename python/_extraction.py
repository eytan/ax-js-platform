"""
Shared extraction logic for exporting BoTorch GP models to axjs JSON format.

Used by both axjs_export.py (user-facing export) and generate_fixtures.py (test fixtures).
Requires BoTorch >= 0.17 (ProductKernel/PositiveIndexKernel API).

## API Surface Relied Upon

This module accesses BoTorch/GPyTorch model attributes to extract prediction state.
All accesses are documented here for maintainability when BoTorch internals change.

### Public/semi-public (stable):
  model.train_inputs[0]          Training X tensor
  model.train_targets             Training Y tensor
  model.covar_module              Root covariance module
  model.mean_module               Mean function module
  model.likelihood.noise          Noise variance
  model.input_transform           ChainedInputTransform / Normalize / Warp
  model.outcome_transform         Standardize / ChainedOutcomeTransform
  kernel.lengthscale              ARD lengthscale tensor
  kernel.outputscale              ScaleKernel outputscale
  kernel.active_dims              Dimension selection
  kernel.base_kernel              ScaleKernel base
  kernel.kernels                  Composite kernel children
  kernel.covar_factor             PositiveIndexKernel W matrix
  kernel.data_covar_module        MultitaskKernel data kernel

### Private/internal (fragile — may break across versions):
  model._task_feature             MultiTaskGP task column index
  model._outcome_names            ModelListGP outcome labels
  kernel._eval_covar_matrix()     PositiveIndexKernel: computes B = normalized WW^T

### Ideal future state:
  BoTorch provides model.prediction_state() → dict with everything needed.
  See docs/SERIALIZATION_CONTRACT.md for the proposal.
"""

from __future__ import annotations

import warnings
from typing import Any


# ── Kernel extraction ────────────────────────────────────────────────────────


def extract_kernel_state(covar: Any) -> dict:
    """Recursively extract kernel state from a GPyTorch/BoTorch covariance module.

    Handles:
    - Base kernels: MaternKernel, RBFKernel, CategoricalKernel
    - ScaleKernel wrapping base or composite kernels
    - Composite kernels: AdditiveKernel, ProductKernel
    """
    from gpytorch.kernels import (
        ScaleKernel,
        MaternKernel,
        RBFKernel,
        ProductKernel,
        AdditiveKernel,
    )

    # Try importing CategoricalKernel from gpytorch first, then botorch
    CategoricalKernel = _get_categorical_kernel_class()

    state: dict[str, Any] = {}

    # Composite kernels (not wrapped in Scale)
    if isinstance(covar, AdditiveKernel):
        state["type"] = "Additive"
        state["kernels"] = [extract_kernel_state(k) for k in covar.kernels]
        if covar.active_dims is not None:
            state["active_dims"] = covar.active_dims.tolist()
        return state

    if isinstance(covar, ProductKernel):
        state["type"] = "Product"
        state["kernels"] = [extract_kernel_state(k) for k in covar.kernels]
        if covar.active_dims is not None:
            state["active_dims"] = covar.active_dims.tolist()
        return state

    # ScaleKernel: may wrap base or composite kernel
    if isinstance(covar, ScaleKernel):
        base = covar.base_kernel
        # If base is composite (Product/Additive), use recursive Scale format
        if isinstance(base, (ProductKernel, AdditiveKernel)):
            inner = extract_kernel_state(base)
            state["type"] = "Scale"
            state["outputscale"] = covar.outputscale.item()
            state["base_kernel"] = inner
            return state
        # Otherwise, legacy flat format: outputscale at top level
        state["outputscale"] = covar.outputscale.item()
    else:
        base = covar

    # Base kernel type
    if isinstance(base, MaternKernel):
        state["type"] = "Matern"
        nu = base.nu
        if abs(nu - 2.5) < 0.01:
            state["nu"] = 2.5
        elif abs(nu - 1.5) < 0.01:
            state["nu"] = 1.5
        elif abs(nu - 0.5) < 0.01:
            state["nu"] = 0.5
        else:
            state["nu"] = float(nu)
    elif isinstance(base, RBFKernel):
        state["type"] = "RBF"
    elif isinstance(base, CategoricalKernel):
        state["type"] = "Categorical"
    else:
        raise ValueError(f"Unsupported kernel type: {type(base)}")

    # Lengthscale
    if hasattr(base, "lengthscale") and base.lengthscale is not None:
        ls = base.lengthscale.detach().cpu().squeeze()
        state["lengthscale"] = ls.tolist() if ls.dim() > 0 else [ls.item()]

    # Active dims
    if hasattr(base, "active_dims") and base.active_dims is not None:
        state["active_dims"] = base.active_dims.tolist()

    return state


# ── Transform extraction ────────────────────────────────────────────────────


def extract_transforms(model: Any) -> tuple[dict | None, dict | None, dict | None]:
    """Extract (input_transform, input_warp, outcome_transform) from a model.

    Returns a tuple of (normalize_state, warp_state, outcome_state), each
    either a dict or None.
    """
    input_tf = None
    warp_tf = None
    outcome_tf = None

    if hasattr(model, "input_transform") and model.input_transform is not None:
        input_tf, warp_tf = _extract_input_transform(model.input_transform)

    if hasattr(model, "outcome_transform") and model.outcome_transform is not None:
        outcome_tf = _extract_outcome_transform(model.outcome_transform)

    return input_tf, warp_tf, outcome_tf


def _extract_input_transform(transform: Any) -> tuple[dict | None, dict | None]:
    """Extract input transform, handling ChainedInputTransform, Normalize, Warp."""
    normalize_state = None
    warp_state = None

    try:
        from botorch.models.transforms.input import ChainedInputTransform, Warp, Normalize
    except ImportError:
        # Fallback: try extracting offset/coefficient directly
        return _extract_normalize_fallback(transform), None

    if isinstance(transform, ChainedInputTransform):
        for _key, sub_tf in transform.items():
            if isinstance(sub_tf, Normalize):
                normalize_state = _normalize_to_dict(sub_tf)
            elif isinstance(sub_tf, Warp):
                warp_state = _warp_to_dict(sub_tf)
        return normalize_state, warp_state

    if isinstance(transform, Normalize):
        return _normalize_to_dict(transform), None

    if isinstance(transform, Warp):
        return None, _warp_to_dict(transform)

    # Unknown type: try Normalize-like extraction
    return _extract_normalize_fallback(transform), None


def _normalize_to_dict(tf: Any) -> dict:
    """Convert a Normalize transform to a dict."""
    offset = tf.offset.detach().cpu()
    coeff = tf.coefficient.detach().cpu()
    return {
        "offset": offset.squeeze().tolist() if offset.dim() > 1 else offset.tolist(),
        "coefficient": coeff.squeeze().tolist() if coeff.dim() > 1 else coeff.tolist(),
    }


def _warp_to_dict(tf: Any) -> dict:
    """Convert a Warp transform to a dict."""
    c0 = tf.concentration0.detach().cpu().squeeze()
    c1 = tf.concentration1.detach().cpu().squeeze()
    ws: dict[str, Any] = {
        "concentration0": c0.tolist() if c0.dim() > 0 else [c0.item()],
        "concentration1": c1.tolist() if c1.dim() > 0 else [c1.item()],
    }
    if hasattr(tf, "indices") and tf.indices is not None:
        ws["indices"] = tf.indices.tolist()
    return ws


def _extract_normalize_fallback(transform: Any) -> dict | None:
    """Fallback: try extracting offset/coefficient from unknown transform type."""
    try:
        offset = transform.offset.detach().cpu().squeeze().tolist()
        coefficient = transform.coefficient.detach().cpu().squeeze().tolist()
        if not isinstance(offset, list):
            offset = [offset]
        if not isinstance(coefficient, list):
            coefficient = [coefficient]
        return {"offset": offset, "coefficient": coefficient}
    except AttributeError:
        warnings.warn(f"Could not extract input transform: {type(transform)}")
        return None


def _extract_outcome_transform(transform: Any) -> dict | None:
    """Extract outcome transform parameters.

    Handles:
    - Standardize: {type: "Standardize", mean, std}
    - Log: {type: "Log"}
    - Bilog: {type: "Bilog"}
    - Power (Yeo-Johnson): {type: "Power", power}
    - ChainedOutcomeTransform: {type: "Chained", transforms: [...]}
    - Legacy Standardize (no type field): {mean, std}
    """
    # Try to import BoTorch outcome transform types
    try:
        from botorch.models.transforms.outcome import (
            Standardize,
            ChainedOutcomeTransform,
        )
        has_botorch_transforms = True
    except ImportError:
        has_botorch_transforms = False

    # ChainedOutcomeTransform
    if has_botorch_transforms and isinstance(transform, ChainedOutcomeTransform):
        sub_transforms = []
        for _key, sub_tf in transform.items():
            sub_state = _extract_single_outcome_transform(sub_tf)
            if sub_state is not None:
                sub_transforms.append(sub_state)
        if not sub_transforms:
            return None
        # If only one transform in the chain, unwrap
        if len(sub_transforms) == 1:
            return sub_transforms[0]
        return {"type": "Chained", "transforms": sub_transforms}

    return _extract_single_outcome_transform(transform)


def _extract_single_outcome_transform(transform: Any) -> dict | None:
    """Extract a single (non-chained) outcome transform."""
    type_name = type(transform).__name__

    # Standardize: has means and stdvs tensors
    if type_name == "Standardize" or (hasattr(transform, "means") and hasattr(transform, "stdvs")):
        try:
            means = transform.means.detach().cpu().squeeze()
            stds = transform.stdvs.detach().cpu().squeeze()
            return {
                "type": "Standardize",
                "mean": means.item() if means.dim() == 0 else means.tolist(),
                "std": stds.item() if stds.dim() == 0 else stds.tolist(),
            }
        except AttributeError:
            pass

    # Log
    if type_name == "Log":
        return {"type": "Log"}

    # Bilog
    if type_name == "Bilog":
        return {"type": "Bilog"}

    # Power (Yeo-Johnson)
    if type_name in ("Power", "PowerTransform"):
        power = 1.0
        if hasattr(transform, "power"):
            power = transform.power.detach().cpu().item() if hasattr(transform.power, "item") else float(transform.power)
        return {"type": "Power", "power": power}

    warnings.warn(f"Could not extract outcome transform: {type(transform)}")
    return None


# ── Multi-task kernel extraction ─────────────────────────────────────────────


def extract_multitask_kernel(covar: Any) -> tuple[dict, dict]:
    """Extract data kernel state and task covariance from MultiTaskGP's covar_module.

    Expects ProductKernel(data_kernel, PositiveIndexKernel) (BoTorch 0.17+).
    """
    from gpytorch.kernels import ProductKernel

    if not isinstance(covar, ProductKernel):
        raise ValueError(f"Unsupported MultiTaskGP kernel: {type(covar)}")

    kernels = list(covar.kernels)
    data_kernel = None
    task_kernel = None
    for k in kernels:
        if hasattr(k, "covar_factor"):
            task_kernel = k
        else:
            data_kernel = k
    if data_kernel is None or task_kernel is None:
        raise ValueError(
            f"Cannot identify data/task kernels in ProductKernel: "
            f"{[type(k).__name__ for k in kernels]}"
        )

    data_kernel_state = extract_kernel_state(data_kernel)
    covar_factor = task_kernel.covar_factor.detach().cpu().tolist()
    B = task_kernel._eval_covar_matrix().detach().cpu().tolist()
    task_covar_state = {
        "covar_factor": covar_factor,
        "covar_matrix": B,
    }
    return data_kernel_state, task_covar_state


# ── Mean constant extraction ────────────────────────────────────────────────


def extract_mean_constant(model: Any) -> float | list[float]:
    """Extract mean constant(s).

    Returns a scalar for ConstantMean, or a list for MultitaskMean.
    """
    mm = model.mean_module
    if hasattr(mm, "constant"):
        return mm.constant.detach().cpu().item()
    # MultitaskMean wrapping ConstantMean(s)
    if hasattr(mm, "base_means"):
        means = []
        for bm in mm.base_means:
            means.append(bm.constant.detach().cpu().item())
        if len(means) == 1:
            return means[0]
        return means
    raise ValueError(f"Cannot extract mean constant from {type(mm)}")


# ── Model export functions ───────────────────────────────────────────────────


def _untransform_train_X(model: Any, train_X: "Tensor") -> "Tensor":
    """Recover raw (pre-transform) training inputs.

    After fit_gpytorch_mll(), model.train_inputs[0] is in post-transform space.
    To export data that axjs can correctly re-transform, we undo the input_transform.
    """
    if hasattr(model, "input_transform"):
        return model.input_transform.untransform(train_X)
    return train_X


def export_single_gp(model: Any) -> dict:
    """Export a single BoTorch GP model (SingleTaskGP/FixedNoiseGP) to state dict."""
    from gpytorch.likelihoods import FixedNoiseGaussianLikelihood

    train_X = model.train_inputs[0].detach().cpu().double()
    train_X = _untransform_train_X(model, train_X)
    train_Y = model.train_targets.detach().cpu().double()

    is_fixed_noise = isinstance(model.likelihood, FixedNoiseGaussianLikelihood)
    model_type = "FixedNoiseGP" if is_fixed_noise else "SingleTaskGP"

    kernel_state = extract_kernel_state(model.covar_module)
    mean_constant = model.mean_module.constant.detach().cpu().item()

    if is_fixed_noise:
        noise = model.likelihood.noise.detach().cpu().squeeze().tolist()
        if not isinstance(noise, list):
            noise = [noise] * train_X.shape[0]
    else:
        noise = model.likelihood.noise.detach().cpu().item()

    itf, wtf, otf = extract_transforms(model)

    state: dict[str, Any] = {
        "model_type": model_type,
        "train_X": train_X.tolist(),
        "train_Y": train_Y.tolist(),
        "kernel": kernel_state,
        "mean_constant": mean_constant,
        "noise_variance": noise,
    }
    if itf is not None:
        state["input_transform"] = itf
    if wtf is not None:
        state["input_warp"] = wtf
    if otf is not None:
        state["outcome_transform"] = otf

    return state


def export_pairwise_gp(model: Any) -> dict:
    """Export a PairwiseGP model to state dict."""
    train_X = model.datapoints.detach().cpu().double()
    # PairwiseGP stores datapoints in raw (pre-transform) space,
    # unlike SingleTaskGP/MultiTaskGP which store post-transform data.
    # Do NOT call _untransform_train_X here.
    utility = model.utility.detach().cpu().double()
    likelihood_hess = model.likelihood_hess.detach().cpu().double()

    kernel_state = extract_kernel_state(model.covar_module)
    mean_constant = model.mean_module.constant.detach().cpu().item()

    state: dict[str, Any] = {
        "model_type": "PairwiseGP",
        "train_X": train_X.tolist(),
        "utility": utility.squeeze().tolist(),
        "likelihood_hess": likelihood_hess.squeeze().tolist(),
        "kernel": kernel_state,
        "mean_constant": mean_constant,
    }

    itf, wtf, _ = extract_transforms(model)
    if itf is not None:
        state["input_transform"] = itf
    if wtf is not None:
        state["input_warp"] = wtf

    return state


def export_multi_task_gp(model: Any) -> dict:
    """Export a MultiTaskGP model to state dict.

    Handles both old BoTorch (MultitaskKernel) and new 0.16+ (PositiveIndexKernel).
    """
    train_X = model.train_inputs[0].detach().cpu().double()
    train_X = _untransform_train_X(model, train_X)

    covar = model.covar_module
    data_kernel_state, task_covar_state = extract_multitask_kernel(covar)

    mean_constant = extract_mean_constant(model)

    # Determine num_tasks from the task covariance
    if "covar_matrix" in task_covar_state:
        num_tasks = len(task_covar_state["covar_matrix"])
    else:
        num_tasks = len(task_covar_state["covar_factor"])

    task_feature = model._task_feature if hasattr(model, "_task_feature") else -1

    # Noise
    noise_tensor = model.likelihood.noise.detach().cpu().squeeze()
    if noise_tensor.dim() == 0:
        noise: Any = noise_tensor.item()
    else:
        # Per-task noise: expand to per-point based on task indices
        tf_idx = train_X.shape[-1] + task_feature if task_feature < 0 else task_feature
        task_indices = train_X[:, tf_idx].long()
        noise = noise_tensor[task_indices].tolist()

    itf, wtf, otf = extract_transforms(model)

    state: dict[str, Any] = {
        "model_type": "MultiTaskGP",
        "train_X": train_X.tolist(),
        "train_Y": model.train_targets.tolist(),
        "task_feature": task_feature,
        "num_tasks": num_tasks,
        "data_kernel": data_kernel_state,
        "task_covar": task_covar_state,
        "mean_constant": mean_constant,
        "noise_variance": noise,
    }
    if itf is not None:
        state["input_transform"] = itf
    if wtf is not None:
        state["input_warp"] = wtf
    if otf is not None:
        state["outcome_transform"] = otf

    return state


def export_model_list(model: Any) -> dict:
    """Export a ModelListGP."""
    models = []
    for sub_model in model.models:
        models.append(export_single_gp(sub_model))

    outcome_names = []
    if hasattr(model, "_outcome_names"):
        outcome_names = list(model._outcome_names)
    else:
        outcome_names = [f"y{i}" for i in range(len(models))]

    return {
        "model_type": "ModelListGP",
        "outcome_names": outcome_names,
        "models": models,
    }


def export_ensemble_gp(models: list[Any]) -> dict:
    """Export an EnsembleGP (list of fitted SingleTaskGP models) to state dict.

    Args:
        models: List of fitted SingleTaskGP models (e.g., from SAAS posterior samples
                or multi-restart MAP).
    """
    model_states = []
    for m in models:
        model_states.append(export_single_gp(m))

    return {
        "model_type": "EnsembleGP",
        "models": model_states,
    }


def _export_batched_multi_output(model: Any) -> dict:
    """Export a batched multi-output SingleTaskGP as ModelListGP.

    Ax's default for MOO uses a single SingleTaskGP with batch dimensions
    (shape [n_outputs, n_train, d]) rather than a ModelListGP. This function
    decomposes the batched model into per-outcome sub-models for axjs.
    """
    import torch
    from gpytorch.likelihoods import FixedNoiseGaussianLikelihood

    train_X_batched = model.train_inputs[0].detach().cpu().double()
    train_Y_batched = model.train_targets.detach().cpu().double()
    n_outputs = train_X_batched.shape[0]

    # Untransform train_X if input_transform exists
    if hasattr(model, "input_transform") and model.input_transform is not None:
        train_X_batched = model.input_transform.untransform(train_X_batched)

    is_fixed_noise = isinstance(model.likelihood, FixedNoiseGaussianLikelihood)

    # Extract shared input transform (offset/coefficient are [1, d], shared)
    itf_state = None
    warp_state = None
    otf_state = None
    if hasattr(model, "input_transform") and model.input_transform is not None:
        itf_state, warp_state, otf_state = extract_transforms(model)
    if otf_state is not None:
        warnings.warn(
            "Batched multi-output SingleTaskGP has a model-level outcome_transform. "
            "Per-output decomposition of batched outcome transforms is not yet supported; "
            "the outcome_transform will be dropped. If this model came from Ax, the "
            "adapter-level transform (StandardizeY) should handle this correctly."
        )

    # Extract per-output kernel parameters manually since extract_kernel_state
    # doesn't handle batch dimensions. The kernel module has:
    #   lengthscale: [n_outputs, 1, d], outputscale: [n_outputs] (if Scale)
    covar = model.covar_module
    from gpytorch.kernels import ScaleKernel

    model_states = []
    for k in range(n_outputs):
        train_X_k = train_X_batched[k]  # [n_train, d]
        train_Y_k = train_Y_batched[k]  # [n_train]

        # Build kernel state for this output by slicing batch dim
        kernel_state = _extract_kernel_for_batch(covar, k)

        # Mean constant for this output
        mean_const = model.mean_module.constant.detach().cpu()[k].item()

        # Noise for this output
        if is_fixed_noise:
            noise = model.likelihood.noise.detach().cpu()[k].squeeze().tolist()
            if not isinstance(noise, list):
                noise = [noise] * train_X_k.shape[0]
            model_type = "FixedNoiseGP"
        else:
            noise = model.likelihood.noise.detach().cpu()[k].squeeze().item()
            model_type = "SingleTaskGP"

        state: dict[str, Any] = {
            "model_type": model_type,
            "train_X": train_X_k.tolist(),
            "train_Y": train_Y_k.tolist(),
            "kernel": kernel_state,
            "mean_constant": mean_const,
            "noise_variance": noise,
        }
        if itf_state is not None:
            state["input_transform"] = itf_state
        if warp_state is not None:
            state["input_warp"] = warp_state

        model_states.append(state)

    return {
        "model_type": "ModelListGP",
        "outcome_names": [f"y{i}" for i in range(n_outputs)],
        "models": model_states,
    }


def _extract_kernel_for_batch(covar: Any, batch_idx: int) -> dict:
    """Extract kernel state for a single batch index from a batched kernel.

    Handles bare RBF/Matern (batch lengthscale) and ScaleKernel wrapping.
    """
    from gpytorch.kernels import ScaleKernel, MaternKernel, RBFKernel

    state: dict[str, Any] = {}

    if isinstance(covar, ScaleKernel):
        os_val = covar.outputscale.detach().cpu()
        state["outputscale"] = os_val[batch_idx].item() if os_val.dim() > 0 else os_val.item()
        base = covar.base_kernel
    else:
        base = covar

    if isinstance(base, MaternKernel):
        state["type"] = "Matern"
        nu = base.nu
        for canonical in (0.5, 1.5, 2.5):
            if abs(nu - canonical) < 0.01:
                state["nu"] = canonical
                break
        else:
            state["nu"] = float(nu)
    elif isinstance(base, RBFKernel):
        state["type"] = "RBF"
    else:
        raise ValueError(f"Unsupported batched kernel type: {type(base)}")

    if hasattr(base, "lengthscale") and base.lengthscale is not None:
        ls = base.lengthscale.detach().cpu()
        # ls shape: [n_outputs, 1, d] → select batch, squeeze
        ls_k = ls[batch_idx].squeeze()
        state["lengthscale"] = ls_k.tolist() if ls_k.dim() > 0 else [ls_k.item()]

    if hasattr(base, "active_dims") and base.active_dims is not None:
        state["active_dims"] = base.active_dims.tolist()

    return state


def export_botorch_model(model: Any) -> dict:
    """Export any supported BoTorch model to JSON-serializable state dict.

    Dispatches to the appropriate export function based on model type.
    """
    from botorch.models import ModelListGP
    from botorch.models.pairwise_gp import PairwiseGP

    try:
        from botorch.models import MultiTaskGP
        if isinstance(model, MultiTaskGP):
            return export_multi_task_gp(model)
    except ImportError:
        pass

    if isinstance(model, ModelListGP):
        return export_model_list(model)
    if isinstance(model, PairwiseGP):
        return export_pairwise_gp(model)

    # Detect batched multi-output SingleTaskGP (Ax default for MOO).
    # These have train_inputs[0].dim() == 3 with shape [n_outputs, n_train, d].
    # Decompose into ModelListGP format since axjs handles that natively.
    if model.train_inputs[0].dim() == 3:
        return _export_batched_multi_output(model)

    return export_single_gp(model)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _get_categorical_kernel_class() -> type:
    """Get CategoricalKernel class from botorch (required, BoTorch >= 0.17)."""
    from botorch.models.kernels import CategoricalKernel
    return CategoricalKernel
