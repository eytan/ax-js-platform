// Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.

import type { Kernel } from "./types.js";
import type { KernelState } from "../models/types.js";

import { CategoricalKernel } from "./categorical.js";
import { ActiveDimsKernel, AdditiveKernel, ProductKernel } from "./composite.js";
import { MaternKernel } from "./matern.js";
import { RBFKernel } from "./rbf.js";
import { ScaleKernel } from "./scale.js";

function _validateLengthscale(ls: Array<number>, kernelType: string): void {
  for (let i = 0; i < ls.length; i++) {
    if (ls[i] <= 0 || !isFinite(ls[i])) {
      throw new Error(
        `${kernelType} kernel: lengthscale[${i}] = ${ls[i]} must be positive and finite`,
      );
    }
  }
}

/**
 * Build a Kernel from a serialized KernelState.
 *
 * Handles two formats:
 * - Legacy: { type: "Matern", lengthscale: [...], outputscale: 0.5 }
 *   → ScaleKernel(MaternKernel(...))
 * - New recursive: { type: "Scale", outputscale: 0.5, base_kernel: { type: "Product", kernels: [...] } }
 */
export function buildKernel(state: KernelState): Kernel {
  let kernel: Kernel;

  switch (state.type) {
    case "Matern": {
      _validateLengthscale(state.lengthscale!, "Matern");
      kernel = new MaternKernel(state.lengthscale!, (state.nu as 0.5 | 1.5 | 2.5) ?? 2.5);
      break;
    }

    case "RBF": {
      _validateLengthscale(state.lengthscale!, "RBF");
      kernel = new RBFKernel(state.lengthscale!);
      break;
    }

    case "Categorical": {
      kernel = new CategoricalKernel(
        state.lengthscale && state.lengthscale.length > 1
          ? state.lengthscale
          : (state.lengthscale?.[0] ?? 1),
      );
      break;
    }

    case "Scale": {
      kernel = new ScaleKernel(buildKernel(state.base_kernel!), state.outputscale!);
      break;
    }

    case "Additive": {
      kernel = new AdditiveKernel(state.kernels!.map(buildKernel));
      break;
    }

    case "Product": {
      kernel = new ProductKernel(state.kernels!.map(buildKernel));
      break;
    }

    default: {
      throw new Error(`Unknown kernel type: ${String(state.type)}`);
    }
  }

  // Legacy format: outputscale at top level for non-Scale types
  if (state.type !== "Scale" && state.outputscale !== undefined) {
    kernel = new ScaleKernel(kernel, state.outputscale);
  }

  // Wrap with active_dims if specified
  if (state.active_dims) {
    kernel = new ActiveDimsKernel(kernel, state.active_dims);
  }

  return kernel;
}
