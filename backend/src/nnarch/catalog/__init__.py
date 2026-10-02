"""
> [!AML-DOC-FILE]
@file       catalog/__init__.py
@description Layer catalog package: the declarative description of every layer the
             tool can place, edit, compile and export.
@module     nnarch.catalog
@exports    registry, bootstrap, LayerSpec, ParamSpec, ParamType, Category, Modality,
            PortSpec, Diagnostic
@created    2026-09-30
@context    [amm: B.1] single source of truth for palette, forms, validation, codegen.
"""

from __future__ import annotations

from .registry import Registry, bootstrap, registry
from .spec import (
    CallStyle,
    Category,
    Diagnostic,
    LayerSpec,
    Modality,
    ParamSpec,
    ParamType,
    PortSpec,
)

__all__ = [
    "CallStyle",
    "Category",
    "Diagnostic",
    "LayerSpec",
    "Modality",
    "ParamSpec",
    "ParamType",
    "PortSpec",
    "Registry",
    "bootstrap",
    "registry",
]
