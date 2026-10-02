"""
> [!AML-DOC-FILE]
@file       ir/schema.py
@description The graph intermediate representation: the in-memory model of a user's
             architecture and, verbatim, the on-disk project format.
@module     nnarch.ir.schema
@exports    Position, Node, Edge, GraphIR, ShapeInfo
@created    2026-09-30
@context    RISK:HIGH [amm: B.2]. Every field here is persisted inside `.nnarch`
            files. PRESERVE back-compatibility: add optional fields only, and bump
            `nnarch.IR_VERSION` alongside an append-only migration [task 12].
"""

from __future__ import annotations

from collections import deque
from typing import Any

from pydantic import BaseModel, Field

from nnarch import IR_VERSION

OUTPUT_PORT = "output"
"""Default port name produced by a single-output layer."""

INPUT_PORT = "input"
"""Default port name consumed by a single-input layer."""


class Position(BaseModel):
    """
    > [!AML-DOC-UNIT]
    Canvas coordinates of a node, in graph space rather than screen space so the
    layout survives zooming and window resizing.
    @param x horizontal offset
    @param y vertical offset
    """

    x: float = 0.0
    y: float = 0.0


class Node(BaseModel):
    """
    > [!AML-DOC-UNIT]
    One placed layer.
    @param id       graph-unique identifier, stable across saves
    @param type     `LayerSpec.id` from the catalog, e.g. "keras.Conv2D"
    @param name     user-editable label, also the basis of the generated Python
                    identifier during export [task 06]
    @param params   constructor arguments; keys are `ParamSpec.name`
    @param position canvas coordinates
    @param notes    free-text annotation the user attaches to the node
    @param disabled when True the node is skipped and its input is passed through,
                    which lets the user ablate a layer without deleting it
    """

    id: str
    type: str
    name: str = ""
    params: dict[str, Any] = Field(default_factory=dict)
    position: Position = Field(default_factory=Position)
    notes: str | None = None
    disabled: bool = False


class Edge(BaseModel):
    """
    > [!AML-DOC-UNIT]
    A tensor flowing from one node's output port into another node's input port.
    @param id          graph-unique identifier
    @param source      id of the producing node
    @param source_port output port on the producing node
    @param target      id of the consuming node
    @param target_port input port on the consuming node; for variadic merging
                       layers every edge shares the same port name and `order`
                       decides the argument position
    @param order       position among the edges entering the same target port
    """

    id: str
    source: str
    source_port: str = OUTPUT_PORT
    target: str
    target_port: str = INPUT_PORT
    order: int = 0


class ShapeInfo(BaseModel):
    """
    > [!AML-DOC-UNIT]
    The shape and dtype Keras reported for a node's output during symbolic
    compilation. `None` in `shape` marks a dynamic axis, typically the batch.
    @param shape  tensor shape including the batch axis
    @param dtype  element type name
    @param params number of trainable and non-trainable weights in this layer
    """

    shape: list[int | None]
    dtype: str
    params: int = 0


class GraphIR(BaseModel):
    """
    > [!AML-DOC-UNIT]
    A complete architecture: the nodes, how they connect, and which nodes act as
    the model's inputs and outputs.
    @param ir_version format version, used to select migrations on load [task 12]
    @param name       project name
    @param nodes      every placed layer
    @param edges      every connection
    @param inputs     ids of the nodes that feed the model; empty means "infer
                      from the Input layers present"
    @param outputs    ids of the nodes whose tensors the model returns; empty means
                      "infer from the nodes with no outgoing edges"
    @param meta       viewport, tool version and other non-semantic state
    @sideEffects none; the IR is inert data and holds no Keras objects
    """

    ir_version: int = IR_VERSION
    name: str = "Untitled"
    nodes: list[Node] = Field(default_factory=list)
    edges: list[Edge] = Field(default_factory=list)
    inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)

    def node_map(self) -> dict[str, Node]:
        """
        > [!AML-DOC-UNIT]
        Index the nodes by id.
        @returns mapping of node id to Node; later duplicates overwrite earlier ones,
                 and `validate` reports the duplication separately
        """
        return {node.id: node for node in self.nodes}

    def incoming(self, node_id: str) -> list[Edge]:
        """
        > [!AML-DOC-UNIT]
        Edges entering a node, in the order the compiler must pass them.
        @param node_id the consuming node
        @returns edges sorted by target port then by explicit order
        """
        return sorted(
            (edge for edge in self.edges if edge.target == node_id),
            key=lambda edge: (edge.target_port, edge.order),
        )

    def outgoing(self, node_id: str) -> list[Edge]:
        """
        > [!AML-DOC-UNIT]
        Edges leaving a node.
        @param node_id the producing node
        @returns edges sorted by source port then by explicit order
        """
        return sorted(
            (edge for edge in self.edges if edge.source == node_id),
            key=lambda edge: (edge.source_port, edge.order),
        )

    def child_node_ids(self) -> set[str]:
        """
        > [!AML-DOC-UNIT]
        Nodes referenced as the wrapped child of another node, such as the layer
        inside a `Bidirectional`. These participate in compilation but are not part
        of the dataflow, so they are excluded from orphan and topology checks.
        @returns set of referenced node ids
        """
        referenced: set[str] = set()
        for node in self.nodes:
            child = node.params.get("layer")
            if isinstance(child, str) and child:
                referenced.add(child)
        return referenced

    def dataflow_nodes(self) -> list[Node]:
        """
        > [!AML-DOC-UNIT]
        Nodes that take part in the dataflow graph, excluding wrapped children.
        @returns Node list in declaration order
        """
        children = self.child_node_ids()
        return [node for node in self.nodes if node.id not in children]

    def topological_order(self) -> tuple[list[str], list[str]]:
        """
        > [!AML-DOC-UNIT]
        Order the dataflow nodes so every node follows its producers (Kahn's
        algorithm), and identify any nodes trapped in a cycle.
        @returns (ordered node ids, node ids that could not be ordered). A non-empty
                 second element means the graph contains at least one cycle, and the
                 first element is the acyclic prefix that could still be ordered.
        @sideEffects none
        @context The compiler depends on this ordering [amm: E.3]; the validator
                 uses the same call to report cycles as diagnostics rather than
                 letting the compiler recurse forever.
        """
        participating = {node.id for node in self.dataflow_nodes()}
        indegree = {node_id: 0 for node_id in participating}
        successors: dict[str, list[str]] = {node_id: [] for node_id in participating}

        for edge in self.edges:
            if edge.source not in participating or edge.target not in participating:
                continue
            successors[edge.source].append(edge.target)
            indegree[edge.target] += 1

        ready = deque(sorted(node_id for node_id, degree in indegree.items() if degree == 0))
        ordered: list[str] = []
        while ready:
            node_id = ready.popleft()
            ordered.append(node_id)
            for successor in successors[node_id]:
                indegree[successor] -= 1
                if indegree[successor] == 0:
                    ready.append(successor)

        return ordered, sorted(participating - set(ordered))

    def resolved_inputs(self) -> list[str]:
        """
        > [!AML-DOC-UNIT]
        The nodes that feed the model.
        @returns the explicit `inputs` list when set, otherwise every Input layer
                 in declaration order
        """
        if self.inputs:
            return list(self.inputs)
        return [node.id for node in self.nodes if node.type == "keras.Input"]

    def resolved_outputs(self) -> list[str]:
        """
        > [!AML-DOC-UNIT]
        The nodes whose tensors the model returns.
        @returns the explicit `outputs` list when set, otherwise every dataflow node
                 with no outgoing edge
        """
        if self.outputs:
            return list(self.outputs)
        consumed = {edge.source for edge in self.edges}
        return [node.id for node in self.dataflow_nodes() if node.id not in consumed]
