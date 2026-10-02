"""
> [!AML-DOC-FILE]
@file       ir/validate.py
@description Structural validation of a graph IR: reports every problem the user can
             cause as data, so the editor can annotate nodes instead of showing a
             stack trace.
@module     nnarch.ir.validate
@exports    Codes, validate_structure
@created    2026-09-30
@context    Runs before compilation and never imports TensorFlow, so the editor gets
            instant feedback while typing [task 09]. RULE: a malformed graph is a
            diagnostic, never an exception.
"""

from __future__ import annotations

from collections import Counter

from nnarch import IR_VERSION
from nnarch.catalog import Diagnostic, Registry
from nnarch.catalog.spec import ParamType

from .schema import GraphIR


class Codes:
    """
    > [!AML-DOC-UNIT]
    Stable diagnostic identifiers. Tests and documentation reference these strings,
    so they are part of the public contract and are never renamed.
    """

    DUPLICATE_NODE_ID = "duplicate_node_id"
    DUPLICATE_EDGE_ID = "duplicate_edge_id"
    UNKNOWN_LAYER_TYPE = "unknown_layer_type"
    DANGLING_EDGE = "dangling_edge"
    SELF_LOOP = "self_loop"
    CYCLE = "cycle"
    UNKNOWN_PORT = "unknown_port"
    TOO_FEW_INPUTS = "too_few_inputs"
    TOO_MANY_INPUTS = "too_many_inputs"
    MISSING_REQUIRED_PARAM = "missing_required_param"
    UNKNOWN_PARAM = "unknown_param"
    NO_INPUT_LAYER = "no_input_layer"
    NO_OUTPUT = "no_output"
    ORPHAN_NODE = "orphan_node"
    EMPTY_GRAPH = "empty_graph"
    UNKNOWN_NODE_REFERENCE = "unknown_node_reference"
    MISSING_WRAPPED_LAYER = "missing_wrapped_layer"
    WRAPPED_LAYER_WIRED = "wrapped_layer_wired"
    IR_VERSION_TOO_NEW = "ir_version_too_new"
    DUPLICATE_PORT_CONNECTION = "duplicate_port_connection"


def _check_versions(graph: GraphIR) -> list[Diagnostic]:
    """
    > [!AML-DOC-UNIT]
    Refuse a graph written by a newer version of the tool.
    @param graph the IR to inspect
    @returns a single error diagnostic when the format is from the future, else []
    @context Loading an unknown format partially would silently drop the user's
             work, so this is an error rather than a warning [amm: B.2].
    """
    if graph.ir_version > IR_VERSION:
        return [Diagnostic(
            severity="error",
            code=Codes.IR_VERSION_TOO_NEW,
            message=(
                f"This project uses graph format v{graph.ir_version}, but this build "
                f"understands up to v{IR_VERSION}. Update AI Architect to open it."
            ),
        )]
    return []


def _check_identity(graph: GraphIR) -> list[Diagnostic]:
    """
    > [!AML-DOC-UNIT]
    Verify node and edge ids are unique and every edge lands on a real node.
    @param graph the IR to inspect
    @returns diagnostics for duplicate ids, dangling edges and self loops
    """
    diagnostics: list[Diagnostic] = []
    known = {node.id for node in graph.nodes}

    for node_id, count in Counter(node.id for node in graph.nodes).items():
        if count > 1:
            diagnostics.append(Diagnostic(
                severity="error", code=Codes.DUPLICATE_NODE_ID, node_id=node_id,
                message=f"Node id {node_id!r} is used by {count} nodes.",
            ))
    for edge_id, count in Counter(edge.id for edge in graph.edges).items():
        if count > 1:
            diagnostics.append(Diagnostic(
                severity="error", code=Codes.DUPLICATE_EDGE_ID, edge_id=edge_id,
                message=f"Edge id {edge_id!r} is used by {count} connections.",
            ))

    for edge in graph.edges:
        for role, endpoint in (("source", edge.source), ("target", edge.target)):
            if endpoint not in known:
                diagnostics.append(Diagnostic(
                    severity="error", code=Codes.DANGLING_EDGE, edge_id=edge.id,
                    message=f"Connection {edge.id!r} has no {role} node {endpoint!r}.",
                ))
        if edge.source == edge.target:
            diagnostics.append(Diagnostic(
                severity="error", code=Codes.SELF_LOOP, edge_id=edge.id, node_id=edge.source,
                message="A layer cannot feed itself.",
            ))
    return diagnostics


def _check_types_and_params(graph: GraphIR, registry: Registry) -> list[Diagnostic]:
    """
    > [!AML-DOC-UNIT]
    Check every node against its catalog spec: the type exists, required parameters
    are supplied, and no unknown parameters are present.
    @param graph    the IR to inspect
    @param registry the layer catalog to validate against
    @returns diagnostics for unknown types and parameter problems
    """
    diagnostics: list[Diagnostic] = []
    known_nodes = {node.id for node in graph.nodes}

    for node in graph.nodes:
        spec = registry.get(node.type)
        if spec is None:
            diagnostics.append(Diagnostic(
                severity="error", code=Codes.UNKNOWN_LAYER_TYPE, node_id=node.id,
                message=(
                    f"Unknown layer type {node.type!r}. It may come from a newer "
                    f"build or a plugin that is not installed."
                ),
            ))
            continue

        declared = {param.name for param in spec.params}
        for name in node.params:
            if name not in declared:
                diagnostics.append(Diagnostic(
                    severity="warning", code=Codes.UNKNOWN_PARAM, node_id=node.id,
                    message=f"{spec.label} has no setting named {name!r}; it will be ignored.",
                ))

        for param in spec.params:
            if not param.required:
                continue
            value = node.params.get(param.name, param.default)
            if value is None or value == "":
                diagnostics.append(Diagnostic(
                    severity="error", code=Codes.MISSING_REQUIRED_PARAM, node_id=node.id,
                    message=f"{spec.label} needs a value for {param.name!r}.",
                ))
            if param.type is ParamType.LAYER_REF and isinstance(value, str) and value:
                if value not in known_nodes:
                    diagnostics.append(Diagnostic(
                        severity="error", code=Codes.UNKNOWN_NODE_REFERENCE, node_id=node.id,
                        message=f"{spec.label} wraps node {value!r}, which does not exist.",
                    ))
                elif any(edge.target == value or edge.source == value for edge in graph.edges):
                    diagnostics.append(Diagnostic(
                        severity="error", code=Codes.WRAPPED_LAYER_WIRED, node_id=value,
                        message=(
                            f"This layer is wrapped by {spec.label}, so it must not be "
                            f"connected on the canvas. Disconnect it."
                        ),
                    ))
            elif param.type is ParamType.LAYER_REF:
                diagnostics.append(Diagnostic(
                    severity="error", code=Codes.MISSING_WRAPPED_LAYER, node_id=node.id,
                    message=f"{spec.label} needs a layer to wrap. Pick one from the canvas.",
                ))
    return diagnostics


def _check_arity(graph: GraphIR, registry: Registry) -> list[Diagnostic]:
    """
    > [!AML-DOC-UNIT]
    Check each node receives a permissible number of inputs on valid port names.
    @param graph    the IR to inspect
    @param registry the layer catalog to validate against
    @returns diagnostics for under-connected, over-connected and mis-ported nodes
    """
    diagnostics: list[Diagnostic] = []
    children = graph.child_node_ids()

    for node in graph.dataflow_nodes():
        spec = registry.get(node.type)
        if spec is None:
            continue
        edges = [edge for edge in graph.incoming(node.id) if edge.source not in children]
        count = len(edges)

        if count < spec.min_inputs:
            diagnostics.append(Diagnostic(
                severity="error", code=Codes.TOO_FEW_INPUTS, node_id=node.id,
                message=(
                    f"{spec.label} needs at least {spec.min_inputs} "
                    f"input{'s' if spec.min_inputs != 1 else ''} but has {count}."
                ),
            ))
        if spec.max_inputs is not None and count > spec.max_inputs:
            diagnostics.append(Diagnostic(
                severity="error", code=Codes.TOO_MANY_INPUTS, node_id=node.id,
                message=(
                    f"{spec.label} accepts at most {spec.max_inputs} "
                    f"input{'s' if spec.max_inputs != 1 else ''} but has {count}."
                ),
            ))

        port_names = {port.name for port in spec.inputs}
        if port_names:
            for edge in edges:
                if edge.target_port not in port_names:
                    diagnostics.append(Diagnostic(
                        severity="error", code=Codes.UNKNOWN_PORT, edge_id=edge.id, node_id=node.id,
                        message=(
                            f"{spec.label} has no input called {edge.target_port!r}. "
                            f"Available: {', '.join(sorted(port_names))}."
                        ),
                    ))
            if spec.max_inputs == 1:
                continue
            for port, port_count in Counter(edge.target_port for edge in edges).items():
                spec_port = next((p for p in spec.inputs if p.name == port), None)
                if spec_port is not None and port_count > 1 and len(port_names) > 1:
                    diagnostics.append(Diagnostic(
                        severity="error", code=Codes.DUPLICATE_PORT_CONNECTION,
                        node_id=node.id,
                        message=(
                            f"{spec.label} input {port!r} has {port_count} connections "
                            f"but accepts one."
                        ),
                    ))
    return diagnostics


def _check_topology(graph: GraphIR, registry: Registry) -> list[Diagnostic]:
    """
    > [!AML-DOC-UNIT]
    Check the graph is a usable dataflow: acyclic, with at least one input and one
    output, and no node stranded away from the network.
    @param graph    the IR to inspect
    @param registry the layer catalog to validate against
    @returns diagnostics for cycles, missing endpoints and orphan nodes
    """
    diagnostics: list[Diagnostic] = []
    dataflow = graph.dataflow_nodes()

    if not dataflow:
        return [Diagnostic(
            severity="info", code=Codes.EMPTY_GRAPH,
            message="The canvas is empty. Drag an Input layer from the palette to start.",
        )]

    _, cyclic = graph.topological_order()
    for node_id in cyclic:
        diagnostics.append(Diagnostic(
            severity="error", code=Codes.CYCLE, node_id=node_id,
            message=(
                "This layer sits on a loop. Data has to flow one way, so remove a "
                "connection to break the cycle."
            ),
        ))

    if not graph.resolved_inputs():
        diagnostics.append(Diagnostic(
            severity="error", code=Codes.NO_INPUT_LAYER,
            message="The model has no Input layer, so there is nowhere for data to enter.",
        ))
    if not graph.resolved_outputs():
        diagnostics.append(Diagnostic(
            severity="error", code=Codes.NO_OUTPUT,
            message="The model has no output. Every layer feeds another one.",
        ))

    if len(dataflow) > 1 and not cyclic:
        connected = {edge.source for edge in graph.edges} | {edge.target for edge in graph.edges}
        for node in dataflow:
            if node.id in connected:
                continue
            spec = registry.get(node.type)
            label = spec.label if spec else node.type
            diagnostics.append(Diagnostic(
                severity="warning", code=Codes.ORPHAN_NODE, node_id=node.id,
                message=f"{label} is not connected to anything and will be ignored.",
            ))
    return diagnostics


def validate_structure(graph: GraphIR, registry: Registry) -> list[Diagnostic]:
    """
    > [!AML-DOC-UNIT]
    Run every structural check over a graph.
    @param graph    the IR to validate
    @param registry the layer catalog to validate against
    @returns diagnostics ordered errors-first, then warnings, then info. An empty
             list means the graph is structurally sound and worth compiling.
    @raises nothing for any user-authored graph: all problems are returned as data
    @sideEffects none; does not import TensorFlow
    """
    diagnostics = _check_versions(graph)
    if diagnostics:
        return diagnostics

    diagnostics.extend(_check_identity(graph))
    diagnostics.extend(_check_types_and_params(graph, registry))
    if not any(d.code == Codes.DANGLING_EDGE for d in diagnostics):
        diagnostics.extend(_check_arity(graph, registry))
        diagnostics.extend(_check_topology(graph, registry))

    rank = {"error": 0, "warning": 1, "info": 2}
    return sorted(diagnostics, key=lambda d: (rank[d.severity], d.code, d.node_id or ""))
