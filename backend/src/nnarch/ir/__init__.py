"""
> [!AML-DOC-FILE]
@file       ir/__init__.py
@description Graph intermediate representation: the project format, its validator
             and the compiler that turns it into a Keras model.
@module     nnarch.ir
@exports    GraphIR, Node, Edge, Position, ShapeInfo, validate_structure,
            compile_graph, CompileResult
@created    2026-09-30
@context    [amm: D] module topology; [amm: E.3] the compiler owns topological order.
"""

from __future__ import annotations

from .compiler import CompileResult, compile_graph
from .importers import import_model
from .schema import Edge, GraphIR, Node, Position, ShapeInfo
from .validate import validate_structure

__all__ = [
    "CompileResult",
    "import_model",
    "Edge",
    "GraphIR",
    "Node",
    "Position",
    "ShapeInfo",
    "compile_graph",
    "validate_structure",
]
