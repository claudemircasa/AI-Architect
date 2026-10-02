"""
> [!AML-DOC-FILE]
@file       __init__.py
@description Package root for the AI Architect backend engine.
@module     nnarch
@exports    __version__, IR_VERSION
@created    2026-09-30
@context    Kept import-light on purpose: importing `nnarch` must NOT pull in
            TensorFlow, so the API server can answer /health in milliseconds
            [task 01].
"""

from __future__ import annotations

__version__ = "0.1.0"

IR_VERSION = 1
"""Current on-disk graph format version. Bump only alongside a migration [task 12]."""
