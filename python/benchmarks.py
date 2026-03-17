"""Benchmark test functions for fixture generation.

Provides VehicleSideImpact (as a proper BoTorch MultiObjectiveTestProblem),
a lazy benchmark registry, and bounded sampling utility.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


# ── Vehicle Side-Impact Problem ────────────────────────────────────────────
# 7 design variables, 3 objectives, 6 constraints.
# Polynomial RSM adapted from Liao et al. (2008).

try:
    from botorch.test_functions.multi_objective import MultiObjectiveTestProblem

    class VehicleSideImpact(MultiObjectiveTestProblem):
        """Vehicle Side-Impact Problem: 7D, 9 outputs (3 obj + 6 constraints).

        Design variables are panel thicknesses with engineering bounds.
        Objectives (minimize): structural weight, peak acceleration, door intrusion.
        Constraints (value <= threshold): velocities, forces, criteria.

        Inheriting from MultiObjectiveTestProblem gives us negate, noise_std,
        proper bounds tensor, and standard evaluate_true/__call__ for free.
        """

        dim = 7
        num_objectives = 9
        _bounds = [
            (0.5, 1.5), (0.45, 1.35), (0.5, 1.5), (0.5, 1.5),
            (0.875, 2.625), (0.4, 1.2), (0.4, 1.2),
        ]
        continuous_inds = list(range(7))

        PARAM_NAMES = [
            "bpillar_inner", "bpillar_outer", "floor_side_inner",
            "cross_member", "door_beam", "door_belt_line",
            "roof_rail",
        ]
        OUTCOME_NAMES = [
            "weight", "acceleration", "intrusion",
            "door_velocity", "bpillar_top_vel", "bpillar_mid_vel",
            "pubic_force", "viscous_criterion", "abdomen_load",
        ]

        _ref_point = [0.0] * 9  # placeholder — not used for fixture generation

        def _evaluate_true(self, X: Tensor) -> Tensor:
            """Evaluate all 9 responses. X: (batch, 7) -> (batch, 9)."""
            x1, x2, x3, x4, x5, x6, x7 = X.unbind(-1)

            # Objectives
            f1 = 1.98 + 4.90*x1 + 6.67*x2 + 6.98*x3 + 4.01*x4 + 1.78*x5 + 0.001*x6 + 2.73*x7
            f2 = 7.50 - 0.80*x1 - 0.60*x2 - 0.40*x3 - 0.50*x4 - 0.30*x5 + 0.30*x1*x2 + 0.20*x2*x3
            f3 = 15.0 - 2.00*x1 - 1.50*x2 - 1.00*x3 - 1.50*x5 + 0.50*x1*x2 + 0.30*x5*x6

            # Constraints
            c1 = 38.0 - 4.00*x1 - 3.00*x2 - 2.00*x5 + 0.50*x1*x2 - 0.30*x3*x5 + x6
            c2 = 42.0 - 6.00*x1 - 5.00*x2 - 3.00*x5 + 0.50*x1*x2 - 0.20*x1*x1
            c3 = 40.0 - 5.00*x1 - 4.00*x2 - 2.00*x3 - 2.00*x5 + 0.80*x1*x2
            c4 = 6.00 - 0.80*x1 - 0.90*x2 - 0.60*x5 - 0.40*x6 + 0.30*x1*x5
            c5 = 0.50 - 0.08*x1 - 0.07*x2 - 0.05*x5 + 0.02*x1*x2
            c6 = 1.40 - 0.18*x1 - 0.15*x2 - 0.12*x3 - 0.08*x5 + 0.05*x2*x3

            return torch.stack([f1, f2, f3, c1, c2, c3, c4, c5, c6], dim=-1)

except ImportError:
    # Fallback if botorch not installed — keep standalone class
    class VehicleSideImpact:  # type: ignore[no-redef]
        """Standalone fallback (no botorch). Same polynomial RSM."""

        PARAM_NAMES = [
            "bpillar_inner", "bpillar_outer", "floor_side_inner",
            "cross_member", "door_beam", "door_belt_line",
            "roof_rail",
        ]
        OUTCOME_NAMES = [
            "weight", "acceleration", "intrusion",
            "door_velocity", "bpillar_top_vel", "bpillar_mid_vel",
            "pubic_force", "viscous_criterion", "abdomen_load",
        ]

        def __init__(self):
            self.bounds = torch.tensor(
                [
                    [0.5, 0.45, 0.5, 0.5, 0.875, 0.4, 0.4],
                    [1.5, 1.35, 1.5, 1.5, 2.625, 1.2, 1.2],
                ],
                dtype=torch.float64,
            )

        def __call__(self, X: Tensor) -> Tensor:
            x1, x2, x3, x4, x5, x6, x7 = X.unbind(-1)
            f1 = 1.98 + 4.90*x1 + 6.67*x2 + 6.98*x3 + 4.01*x4 + 1.78*x5 + 0.001*x6 + 2.73*x7
            f2 = 7.50 - 0.80*x1 - 0.60*x2 - 0.40*x3 - 0.50*x4 - 0.30*x5 + 0.30*x1*x2 + 0.20*x2*x3
            f3 = 15.0 - 2.00*x1 - 1.50*x2 - 1.00*x3 - 1.50*x5 + 0.50*x1*x2 + 0.30*x5*x6
            c1 = 38.0 - 4.00*x1 - 3.00*x2 - 2.00*x5 + 0.50*x1*x2 - 0.30*x3*x5 + x6
            c2 = 42.0 - 6.00*x1 - 5.00*x2 - 3.00*x5 + 0.50*x1*x2 - 0.20*x1*x1
            c3 = 40.0 - 5.00*x1 - 4.00*x2 - 2.00*x3 - 2.00*x5 + 0.80*x1*x2
            c4 = 6.00 - 0.80*x1 - 0.90*x2 - 0.60*x5 - 0.40*x6 + 0.30*x1*x5
            c5 = 0.50 - 0.08*x1 - 0.07*x2 - 0.05*x5 + 0.02*x1*x2
            c6 = 1.40 - 0.18*x1 - 0.15*x2 - 0.12*x3 - 0.08*x5 + 0.05*x2*x3
            return torch.stack([f1, f2, f3, c1, c2, c3, c4, c5, c6], dim=-1)


# ── Benchmark registry ──────────────────────────────────────────────────────

def make_benchmarks(negate: bool = True) -> dict[str, Any]:
    """Lazy benchmark registry. Each value is a callable returning a test function."""
    from botorch.test_functions import Branin, Hartmann

    benchmarks: dict[str, Any] = {
        "Branin": lambda: Branin(negate=negate),
        "Hartmann": lambda: Hartmann(negate=negate),
        "VSIP": lambda: VehicleSideImpact(),
    }

    try:
        from botorch.test_functions.multi_objective import BraninCurrin
        benchmarks["BraninCurrin"] = lambda: BraninCurrin()
    except ImportError:
        pass

    try:
        from botorch.test_functions import Penicillin
        benchmarks["Penicillin"] = lambda: Penicillin()
    except ImportError:
        pass

    try:
        from botorch.test_functions.multi_objective import C2DTLZ2
        benchmarks["C2DTLZ2"] = lambda: _ConstrainedWrapper(
            C2DTLZ2(dim=4, num_objectives=2),
            objective_names=["f0", "f1"],
            constraint_names=["c0"],
        )
    except ImportError:
        pass

    try:
        from botorch.test_functions.multi_objective import DiscBrake
        benchmarks["DiscBrake"] = lambda: _ConstrainedWrapper(
            DiscBrake(),
            objective_names=["mass", "stopping_dist"],
            constraint_names=["c0", "c1", "c2", "c3"],
        )
    except ImportError:
        pass

    try:
        from botorch.test_functions.synthetic import PressureVessel
        benchmarks["PressureVessel"] = lambda: _ConstrainedWrapper(
            PressureVessel(),
            objective_names=["cost"],
            constraint_names=["c0", "c1", "c2", "c3"],
        )
    except ImportError:
        pass

    try:
        from botorch.test_functions.synthetic import TrajectoryPlanning
        benchmarks["TrajectoryPlanning"] = lambda: TrajectoryPlanning(dim=30)
    except ImportError:
        pass

    return benchmarks


# ── Constrained benchmark wrappers ─────────────────────────────────────────

class _ConstrainedWrapper:
    """Wraps a constrained test function to output [objectives, constraints].

    Works for both MOO and SOO problems. BoTorch's constrained problems
    return only objectives from __call__ and constraints from evaluate_slack().
    This wrapper concatenates them so the GP can model everything as a single
    multi-output.
    """

    def __init__(
        self,
        problem: Any,
        objective_names: list[str],
        constraint_names: list[str],
    ):
        self._problem = problem
        self.bounds = problem.bounds
        self.dim = problem.dim
        self.objective_names = objective_names
        self.constraint_names = constraint_names
        self.outcome_names = objective_names + constraint_names
        self.num_objectives = len(objective_names)
        self.num_constraints = len(constraint_names)
        if hasattr(problem, "ref_point"):
            self.ref_point = problem.ref_point

    def __call__(self, X: Tensor) -> Tensor:
        objectives = self._problem(X)
        if objectives.dim() == 1:
            objectives = objectives.unsqueeze(-1)
        constraints = self._problem.evaluate_slack(X)
        return torch.cat([objectives, constraints], dim=-1)


def sample_bounded(bounds: Tensor, n: int, seed: int) -> Tensor:
    """Sample n points uniformly within bounds."""
    torch.manual_seed(seed)
    d = bounds.shape[-1]
    u = torch.rand(n, d, dtype=torch.float64)
    return bounds[0] + u * (bounds[1] - bounds[0])
