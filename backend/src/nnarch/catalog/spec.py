"""
> [!AML-DOC-FILE]
@file       catalog/spec.py
@description Declarative schema describing a neural-network layer and its editable
             parameters. This is the contract that drives the layer palette, the
             auto-generated property forms, graph validation and code generation.
@module     nnarch.catalog.spec
@exports    ParamType, ParamSpec, PortSpec, LayerSpec, Category, Modality
@created    2026-09-30
@context    Single source of truth [amm: B.1, E.2]. A new layer type must be
            expressible here alone; the frontend never gains per-layer code.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class ParamType(str, Enum):
    """
    > [!AML-DOC-UNIT]
    Widget/validation class of a layer parameter. The frontend maps each member to
    exactly one input control, so adding a member is the only way to introduce a
    new kind of form field.
    """

    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    STR = "str"
    ENUM = "enum"
    INT_TUPLE = "int_tuple"
    FLOAT_TUPLE = "float_tuple"
    SHAPE = "shape"
    ACTIVATION = "activation"
    INITIALIZER = "initializer"
    REGULARIZER = "regularizer"
    CONSTRAINT = "constraint"
    DTYPE = "dtype"
    LAYER_REF = "layer_ref"
    TEXT = "text"
    ANY = "any"


class CallStyle(str, Enum):
    """
    > [!AML-DOC-UNIT]
    How a layer receives its input tensors. Keras is not uniform here, and both the
    compiler and the code generator must agree on the convention, so it is recorded
    as catalog data rather than as branching logic in either of them [amm: E.4].

    - `SOURCE`: takes no tensor and produces one, e.g. `keras.Input(shape=...)`.
    - `SINGLE`: one positional tensor, e.g. `Dense()(x)`. The common case.
    - `LIST`: one positional list of tensors, e.g. `Concatenate()([a, b])`. Used by
      every merging layer and by `Attention` / `AdditiveAttention`.
    - `QUERY_VALUE_KEY`: keyword tensors, e.g. `MultiHeadAttention()(query=q,
      value=v, key=k)`. Used by `MultiHeadAttention` and `GroupQueryAttention`.
    - `RECURRENT`: one positional tensor plus an optional starting state passed by
      keyword, e.g. `GRU()(x, initial_state=[h])`. This is how a recurrent model is
      run step by step, carrying its own state forward, which is what a free-running
      generator does — and a model built that way could not be imported at all while
      the catalog insisted a recurrent layer had exactly one input [E-032].
    """

    SOURCE = "source"
    SINGLE = "single"
    LIST = "list"
    QUERY_VALUE_KEY = "query_value_key"
    RECURRENT = "recurrent"


class Category(str, Enum):
    """
    > [!AML-DOC-UNIT]
    Palette grouping. `RESEARCH_*` members carry frontier architectures sourced
    from cited papers and implemented in `nnarch.layers`.
    """

    CORE = "core"
    CONVOLUTION = "convolution"
    POOLING = "pooling"
    RECURRENT = "recurrent"
    NORMALIZATION = "normalization"
    REGULARIZATION = "regularization"
    ATTENTION = "attention"
    RESHAPING = "reshaping"
    MERGING = "merging"
    ACTIVATION = "activation"
    PREPROCESSING = "preprocessing"
    RESEARCH_SSM = "research/ssm"
    RESEARCH_KAN = "research/kan"
    RESEARCH_ATTENTION = "research/attention"
    RESEARCH_MOE = "research/moe"
    RESEARCH_FFN = "research/ffn"
    RESEARCH_RECURRENT = "research/recurrent"
    RESEARCH_CONV = "research/conv"
    RESEARCH_MIXER = "research/mixer"
    RESEARCH_GRAPH = "research/graph"
    RESEARCH_NEURO = "research/neuro"
    RESEARCH_MISC = "research/misc"


class Modality(str, Enum):
    """
    > [!AML-DOC-UNIT]
    Data kinds a layer is meaningful for. Drives palette filtering and picks the
    renderer used when showing real activations [task 11].
    """

    IMAGE = "image"
    TEXT = "text"
    AUDIO = "audio"
    SEQUENCE = "sequence"
    TABULAR = "tabular"
    GRAPH = "graph"
    VOLUMETRIC = "volumetric"
    ANY = "any"


class ParamSpec(BaseModel):
    """
    > [!AML-DOC-UNIT]
    One editable constructor argument of a layer.
    @param name     keyword name passed to the Keras constructor
    @param type     widget/validation class
    @param default  value used when the user does not override it
    @param choices  allowed values; required when `type is ENUM`
    @param minimum  inclusive lower bound for numeric types
    @param maximum  inclusive upper bound for numeric types
    @param arity    element count for tuple types (None means rank-driven)
    @param help     one-line explanation shown beside the control
    @param required when True the graph is invalid until the user supplies a value
    @param advanced when True the control is collapsed behind "Advanced"
    @param group    optional sub-heading inside the property panel
    """

    name: str
    type: ParamType
    default: Any = None
    choices: list[Any] | None = None
    minimum: float | None = None
    maximum: float | None = None
    arity: int | None = None
    help: str = ""
    required: bool = False
    advanced: bool = False
    group: str | None = None


class PortSpec(BaseModel):
    """
    > [!AML-DOC-UNIT]
    A named tensor slot on a layer. Multi-tensor layers (attention, merging)
    declare ordered ports so the compiler can wire arguments positionally
    [amm: E.3].
    @param name  identifier referenced by `Edge.source_port` / `Edge.target_port`
    @param label human-readable name drawn on the canvas node
    @param rank  expected tensor rank, or None when rank-agnostic
    """

    name: str
    label: str
    rank: int | None = None
    optional: bool = False


class LayerSpec(BaseModel):
    """
    > [!AML-DOC-UNIT]
    Complete description of a placeable layer.
    @param id          stable identifier stored in saved projects; never renamed
    @param label       display name in palette and on nodes
    @param category    palette grouping
    @param keras_path  dotted import path resolved by the compiler and emitted by codegen
    @param params      editable constructor arguments
    @param inputs      ordered input ports
    @param outputs     ordered output ports
    @param min_inputs  minimum connected inputs for a valid graph
    @param max_inputs  maximum connected inputs; None means unbounded (merging layers)
    @param rank_in     required input tensor rank, or None when rank-agnostic
    @param rank_out    produced tensor rank, or None when shape-preserving/derived
    @param modalities  data kinds this layer applies to
    @param paper       citation for research layers, e.g. "Gu & Dao 2023, arXiv:2312.00752"
    @param doc_url     upstream documentation link
    @param description short explanation shown in the palette tooltip
    @param tags        free-text search keywords
    @param call_style  how the compiler and code generator pass tensors to it
    @sideEffects none; specs are immutable data
    """

    id: str
    label: str
    category: Category
    keras_path: str
    params: list[ParamSpec] = Field(default_factory=list)
    inputs: list[PortSpec] = Field(default_factory=list)
    outputs: list[PortSpec] = Field(default_factory=list)
    min_inputs: int = 1
    max_inputs: int | None = 1
    rank_in: int | None = None
    rank_out: int | None = None
    modalities: list[Modality] = Field(default_factory=lambda: [Modality.ANY])
    paper: str | None = None
    doc_url: str | None = None
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    call_style: CallStyle = CallStyle.SINGLE
    is_research: bool = False

    def param(self, name: str) -> ParamSpec | None:
        """
        > [!AML-DOC-UNIT]
        Look up one parameter spec by keyword name.
        @param name constructor keyword to find
        @returns the matching ParamSpec, or None when this layer has no such parameter
        """
        return next((p for p in self.params if p.name == name), None)

    def defaults(self) -> dict[str, Any]:
        """
        > [!AML-DOC-UNIT]
        Build the parameter dict a freshly dropped node starts with.
        @returns mapping of parameter name to default value, omitting None defaults
                 so Keras' own defaults remain authoritative
        """
        return {p.name: p.default for p in self.params if p.default is not None}


Severity = Literal["error", "warning", "info"]


class Diagnostic(BaseModel):
    """
    > [!AML-DOC-UNIT]
    A validation result attached to the graph or to one node. User mistakes are
    reported as data, never as exceptions [task 03].
    @param severity  error blocks compilation; warning and info do not
    @param code      stable machine-readable identifier for tests and docs
    @param message   human-readable explanation
    @param node_id   node the problem belongs to, when node-scoped
    @param edge_id   edge the problem belongs to, when edge-scoped
    """

    severity: Severity
    code: str
    message: str
    node_id: str | None = None
    edge_id: str | None = None
