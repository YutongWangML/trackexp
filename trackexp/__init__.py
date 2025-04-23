"""
trackexp - Lightweight experiment tracking for Python.

This module provides simple functions for tracking experiments,
primarily designed for machine learning but suitable for any program.
"""

from .core import init, log, metadata, get_current_experiment
from .utils import get_experiment_path, list_experiments, get_data, get_metadata

__version__ = "0.1.0"
__all__ = [
    "init",
    "log",
    "metadata",
    "get_current_experiment",
    "get_experiment_path",
    "list_experiments",
    "get_data",
    "get_metadata"
]
