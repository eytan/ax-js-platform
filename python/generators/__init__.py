"""Generator functions for axjs test fixtures, organized by model type."""

from generators.singletask import generate_singletask_fixture
from generators.model_list import generate_model_list_fixture
from generators.pairwise import generate_pairwise_fixture
from generators.multitask import generate_multitask_fixture
from generators.ensemble import generate_ensemble_fixture
from generators.ax_level import generate_ax_level_fixture

__all__ = [
    "generate_singletask_fixture",
    "generate_model_list_fixture",
    "generate_pairwise_fixture",
    "generate_multitask_fixture",
    "generate_ensemble_fixture",
    "generate_ax_level_fixture",
]
