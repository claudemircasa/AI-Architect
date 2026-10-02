"""
> [!AML-DOC-FILE]
@file       export/__init__.py
@description Code generation: turns a graph IR into a standalone trainable project.
@module     nnarch.export
@exports    DatasetChoice, ExportOptions, ExportBundle, export_project, build_zip,
            BUILTIN_DATASETS
@created    2026-09-30
@context    [amm: B.4, E.4] semantic parity with the compiler is required.
"""

from __future__ import annotations

from nnarch.data import BUILTIN_DATASETS, DatasetChoice

from .codegen import ExportBundle, ExportOptions, build_zip, export_project

__all__ = [
    "BUILTIN_DATASETS",
    "DatasetChoice",
    "ExportBundle",
    "ExportOptions",
    "build_zip",
    "export_project",
]
