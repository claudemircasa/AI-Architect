"""
> [!AML-DOC-FILE]
@file       viz/__init__.py
@description Capturing and encoding the real activations a model produces.
@module     nnarch.viz
@exports    Activation, ActivationResult, RenderHint, capture_activations, COLORMAPS
@created    2026-10-01
@context    [amm: B.7] serves the data-flow visualisation [task 11].
"""

from __future__ import annotations

from .encode import COLORMAPS
from .tensors import Activation, ActivationResult, RenderHint, capture_activations

__all__ = [
    "COLORMAPS",
    "Activation",
    "ActivationResult",
    "RenderHint",
    "capture_activations",
]
