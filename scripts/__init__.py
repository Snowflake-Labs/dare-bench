"""
DARE-Bench Scripts

This module provides utilities for:
- Generating reference solutions from metadata
- Evaluating model predictions against ground truth
"""

from .evaluation import (
    evaluate_prediction,
    evaluate_batch,
    get_metric_for_single_column
)

from .reference_solution import (
    generate_reference_solution_code,
    generate_reference_solution_for_task,
    batch_generate_reference_solutions
)

__all__ = [
    "evaluate_prediction",
    "evaluate_batch", 
    "get_metric_for_single_column",
    "generate_reference_solution_code",
    "generate_reference_solution_for_task",
    "batch_generate_reference_solutions"
]

