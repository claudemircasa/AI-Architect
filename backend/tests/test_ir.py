"""
> [!AML-DOC-FILE]
@file       tests/test_ir.py
@description Regression tests for the graph IR, its validator and the compiler.
@module     tests.test_ir
@exports    (pytest test functions)
@created    2026-09-30
@context    Covers the task 03 verification gates. The sweep test is the important
            one: it drives every catalog layer through the compiler, so a catalog
            row or calling convention that stops working fails here rather than in
            the editor.
"""

from __future__ import annotations

import keras
import pytest

from nnarch.catalog import CallStyle, Registry, bootstrap
from nnarch.ir import GraphIR, compile_graph
from nnarch.ir.schema import Edge, Node
from nnarch.ir.validate import Codes as StructuralCodes


@pytest.fixture(scope="session")
def registry() -> Registry:
    """
    > [!AML-DOC-UNIT]
    The bootstrapped layer catalog, built once for the whole test session.
    @returns the shared Registry
    """
    return bootstrap()


def node(node_id: str, node_type: str, **params: object) -> Node:
    """
    > [!AML-DOC-UNIT]
    Build a graph node concisely.
    @param node_id   graph-unique id, also used as the node's name
    @param node_type catalog layer id
    @param params    constructor parameters
    @returns the Node
    """
    return Node(id=node_id, type=node_type, name=node_id, params=dict(params))


def edge(source: str, target: str, **kwargs: object) -> Edge:
    """
    > [!AML-DOC-UNIT]
    Build a graph edge with a generated id.
    @param source id of the producing node
    @param target id of the consuming node
    @param kwargs `target_port` and `order` when the target has several inputs
    @returns the Edge
    """
    port = kwargs.get("target_port", "input")
    order = kwargs.get("order", 0)
    return Edge(id=f"{source}->{target}:{port}:{order}", source=source, target=target, **kwargs)


def test_mnist_cnn_param_count_matches_hand_built_model(registry: Registry) -> None:
    """
    > [!AML-DOC-UNIT]
    A compiled graph must have exactly the weights of the equivalent hand-written
    Keras model. This is the strongest single check that the compiler wires layers
    the way a human would.
    """
    graph = GraphIR(name="MNIST CNN", nodes=[
        node("inp", "keras.Input", shape=[28, 28, 1]),
        node("c1", "keras.Conv2D", filters=32, kernel_size=[3, 3], activation="relu"),
        node("p1", "keras.MaxPooling2D", pool_size=[2, 2]),
        node("c2", "keras.Conv2D", filters=64, kernel_size=[3, 3], activation="relu"),
        node("p2", "keras.MaxPooling2D", pool_size=[2, 2]),
        node("fl", "keras.Flatten"),
        node("do", "keras.Dropout", rate=0.5),
        node("out", "keras.Dense", units=10, activation="softmax"),
    ], edges=[edge("inp", "c1"), edge("c1", "p1"), edge("p1", "c2"), edge("c2", "p2"),
              edge("p2", "fl"), edge("fl", "do"), edge("do", "out")])

    reference = keras.Sequential([
        keras.Input((28, 28, 1)),
        keras.layers.Conv2D(32, (3, 3), activation="relu"),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation="relu"),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(10, activation="softmax"),
    ])

    result = compile_graph(graph, registry)
    assert result.ok, [d.message for d in result.diagnostics]
    assert result.params_total == reference.count_params()
    assert result.shapes["c1"].shape == [None, 26, 26, 32]
    assert result.shapes["fl"].shape == [None, 1600]


def test_cycle_is_reported_not_raised(registry: Registry) -> None:
    """
    > [!AML-DOC-UNIT]
    A loop in the graph must come back as diagnostics. Compiling it must not raise
    and must not recurse, because the editor validates on every keystroke.
    """
    graph = GraphIR(name="Cyclic", nodes=[
        node("inp", "keras.Input", shape=[8]),
        node("a", "keras.Dense", units=8),
        node("b", "keras.Dense", units=8),
    ], edges=[edge("inp", "a"), edge("a", "b"), edge("b", "a")])

    result = compile_graph(graph, registry)
    assert result.model is None
    assert {d.node_id for d in result.diagnostics if d.code == StructuralCodes.CYCLE} == {"a", "b"}


def test_shape_mismatch_names_the_offending_node(registry: Registry) -> None:
    """
    > [!AML-DOC-UNIT]
    A real Keras shape error must be surfaced as a diagnostic attached to the node
    that rejected the tensor, so the editor can ring that node.
    """
    graph = GraphIR(name="Bad", nodes=[
        node("inp", "keras.Input", shape=[28, 28, 1]),
        node("fl", "keras.Flatten"),
        node("cv", "keras.Conv2D", filters=8, kernel_size=[3, 3]),
    ], edges=[edge("inp", "fl"), edge("fl", "cv")])

    result = compile_graph(graph, registry)
    assert result.model is None
    failures = [d for d in result.diagnostics if d.severity == "error"]
    assert len(failures) == 1
    assert failures[0].node_id == "cv"
    assert "incompatible" in failures[0].message


def test_residual_add_respects_edge_order(registry: Registry) -> None:
    """
    > [!AML-DOC-UNIT]
    A merging layer must receive its inputs as an ordered list, which is what makes
    skip connections expressible.
    """
    graph = GraphIR(name="Residual", nodes=[
        node("inp", "keras.Input", shape=[32, 32, 16]),
        node("c1", "keras.Conv2D", filters=16, kernel_size=[3, 3], padding="same"),
        node("add", "keras.Add"),
        node("gap", "keras.GlobalAveragePooling2D"),
    ], edges=[edge("inp", "c1"),
              edge("c1", "add", target_port="inputs", order=0),
              edge("inp", "add", target_port="inputs", order=1),
              edge("add", "gap")])

    result = compile_graph(graph, registry)
    assert result.ok, [d.message for d in result.diagnostics]
    assert result.shapes["add"].shape == [None, 32, 32, 16]


def test_attention_receives_named_query_value_key(registry: Registry) -> None:
    """
    > [!AML-DOC-UNIT]
    MultiHeadAttention takes keyword tensors rather than a list, and the port names
    on the edges are what select them.
    """
    graph = GraphIR(name="Attention", nodes=[
        node("inp", "keras.Input", shape=[64, 128]),
        node("mha", "keras.MultiHeadAttention", num_heads=8, key_dim=16),
        node("pool", "keras.GlobalAveragePooling1D"),
    ], edges=[edge("inp", "mha", target_port="query"),
              edge("inp", "mha", target_port="value"),
              edge("inp", "mha", target_port="key"),
              edge("mha", "pool")])

    result = compile_graph(graph, registry)
    assert result.ok, [d.message for d in result.diagnostics]
    assert result.shapes["mha"].shape == [None, 64, 128]


def test_bidirectional_wraps_a_detached_child(registry: Registry) -> None:
    """
    > [!AML-DOC-UNIT]
    A wrapper layer builds the node it references, and that child stays out of the
    dataflow. Concatenating two 64-unit passes must yield 128 features.
    """
    graph = GraphIR(name="BiLSTM", nodes=[
        node("inp", "keras.Input", shape=[50, 32]),
        node("child", "keras.LSTM", units=64),
        node("bi", "keras.Bidirectional", layer="child"),
    ], edges=[edge("inp", "bi")])

    result = compile_graph(graph, registry)
    assert result.ok, [d.message for d in result.diagnostics]
    assert result.shapes["bi"].shape == [None, 128]


def test_wrapped_child_must_not_be_wired(registry: Registry) -> None:
    """
    > [!AML-DOC-UNIT]
    Connecting a wrapped child on the canvas is ambiguous, so it is refused with a
    diagnostic pointing at the child.
    """
    graph = GraphIR(name="BadWrap", nodes=[
        node("inp", "keras.Input", shape=[50, 32]),
        node("child", "keras.LSTM", units=64),
        node("bi", "keras.Bidirectional", layer="child"),
    ], edges=[edge("inp", "bi"), edge("inp", "child")])

    result = compile_graph(graph, registry)
    assert result.model is None
    assert any(d.code == StructuralCodes.WRAPPED_LAYER_WIRED and d.node_id == "child"
               for d in result.diagnostics)


def test_disabled_node_passes_its_input_through(registry: Registry) -> None:
    """
    > [!AML-DOC-UNIT]
    Disabling a node ablates it without deleting it, so the graph still compiles and
    the layer contributes no weights.
    """
    graph = GraphIR(name="Ablate", nodes=[
        node("inp", "keras.Input", shape=[8]),
        Node(id="do", type="keras.Dropout", name="do", params={"rate": 0.5}, disabled=True),
        node("out", "keras.Dense", units=3),
    ], edges=[edge("inp", "do"), edge("do", "out")])

    result = compile_graph(graph, registry)
    assert result.ok, [d.message for d in result.diagnostics]
    assert result.params_total == 27


def test_lambda_expression_cannot_reach_the_host(registry: Registry) -> None:
    """
    > [!AML-DOC-UNIT]
    A Lambda expression is evaluated without builtins, so it behaves like a formula
    field. An expression that tries to import must fail rather than run.
    """
    graph = GraphIR(name="Evil", nodes=[
        node("inp", "keras.Input", shape=[8]),
        node("lam", "keras.Lambda", function="__import__('os').listdir('/')"),
    ], edges=[edge("inp", "lam")])

    result = compile_graph(graph, registry)
    assert result.model is None
    assert any(d.severity == "error" and d.node_id == "lam" for d in result.diagnostics)


def test_missing_required_parameter_is_an_error(registry: Registry) -> None:
    """
    > [!AML-DOC-UNIT]
    A required parameter left empty is reported before compilation is attempted.
    """
    graph = GraphIR(name="NoUnits", nodes=[
        node("inp", "keras.Input", shape=[8]),
        Node(id="d", type="keras.Dense", name="d", params={"units": None}),
    ], edges=[edge("inp", "d")])

    result = compile_graph(graph, registry)
    assert result.model is None
    assert any(d.code == StructuralCodes.MISSING_REQUIRED_PARAM and d.node_id == "d"
               for d in result.diagnostics)


def test_graph_from_a_newer_format_is_refused(registry: Registry) -> None:
    """
    > [!AML-DOC-UNIT]
    A project saved by a newer build must be refused outright. Loading it partially
    would silently discard the user's work [amm: B.2].
    """
    graph = GraphIR(ir_version=99, name="Future", nodes=[node("inp", "keras.Input", shape=[8])])

    result = compile_graph(graph, registry)
    assert result.model is None
    assert [d.code for d in result.diagnostics] == [StructuralCodes.IR_VERSION_TOO_NEW]


# Input shape, excluding batch, that satisfies each declared input rank.
_SHAPE_FOR_RANK: dict[int, list[int]] = {
    2: [16], 3: [12, 8], 4: [16, 16, 3], 5: [8, 8, 8, 3], 6: [4, 6, 6, 6, 2],
}

# Layers whose starter parameters constrain the input beyond its rank.
_SHAPE_OVERRIDES: dict[str, list[int]] = {
    "keras.MelSpectrogram": [4096],
    "keras.STFTSpectrogram": [4096, 1],
    "keras.GroupNormalization": [16, 16, 32],
    "keras.Reshape": [784],
    # The codebook width is a declared constraint, not a default to be guessed.
    "research.VectorQuantizer": [12, 64],
    # Graph layers take node features and a square adjacency from the same probe
    # input, so the feature count has to equal the node count.
    "research.GCNConv": [12, 12],
    "research.GATConv": [12, 12],
    "research.GraphSAGEConv": [12, 12],
    "research.GINConv": [12, 12],
}

_STRING_INPUT = frozenset({
    "keras.TextVectorization", "keras.StringLookup", "keras.Hashing", "keras.HashedCrossing",
})


def _probe_graph(spec, registry: Registry) -> GraphIR:
    """
    > [!AML-DOC-UNIT]
    Build the smallest graph that exercises one layer: an Input wired into it.
    @param spec     the LayerSpec under test
    @param registry the layer catalog, used to pick a child for wrapper layers
    @returns a GraphIR containing an Input, the layer, and any child it wraps
    """
    shape = _SHAPE_OVERRIDES.get(spec.id, _SHAPE_FOR_RANK.get(spec.rank_in or 3, [12, 8]))
    dtype = "float32"
    if spec.id == "keras.Embedding":
        dtype = "int32"
    elif spec.id in _STRING_INPUT:
        dtype, shape = "string", [1]

    nodes = [node("inp", "keras.Input", shape=shape, dtype=dtype)]
    edges: list[Edge] = []

    if spec.param("layer") is not None:
        child_type = "keras.LSTM" if spec.id == "keras.Bidirectional" else "keras.Dense"
        nodes.append(node("child", child_type, units=8))
        nodes.append(node("lay", spec.id, layer="child"))
        edges.append(edge("inp", "lay"))
    elif spec.call_style is CallStyle.QUERY_VALUE_KEY:
        nodes.append(node("lay", spec.id))
        edges += [edge("inp", "lay", target_port="query"),
                  edge("inp", "lay", target_port="value")]
    elif spec.id in ("keras.Attention", "keras.AdditiveAttention"):
        nodes.append(node("lay", spec.id))
        edges += [edge("inp", "lay", target_port="query"),
                  edge("inp", "lay", target_port="value")]
    elif spec.min_inputs >= 2 and len(spec.inputs) >= 2:
        # Distinct named ports, as the graph layers declare: one edge each.
        nodes.append(node("lay", spec.id))
        edges += [
            edge("inp", "lay", target_port=port.name)
            for port in spec.inputs[: spec.min_inputs]
        ]
    elif spec.min_inputs >= 2:
        port = spec.inputs[0].name if spec.inputs else "inputs"
        nodes.append(node("lay", spec.id))
        edges += [edge("inp", "lay", target_port=port, order=0),
                  edge("inp", "lay", target_port=port, order=1)]
    else:
        nodes.append(node("lay", spec.id))
        edges.append(edge("inp", "lay"))

    return GraphIR(name="probe", nodes=nodes, edges=edges)


def test_every_catalog_layer_compiles(registry: Registry) -> None:
    """
    > [!AML-DOC-UNIT]
    Drive every layer in the catalog through the compiler in a minimal graph. This
    is the gate that keeps the catalog honest: a layer the palette offers but the
    compiler cannot wire would fail here.
    """
    failures: list[tuple[str, str]] = []
    for spec in registry.all():
        if spec.id == "keras.Input":
            continue
        result = compile_graph(_probe_graph(spec, registry), registry)
        if not result.ok:
            reason = next((d.message for d in result.diagnostics if d.severity == "error"), "?")
            failures.append((spec.id, reason.replace("\n", " ")[:120]))

    assert not failures, "layers that failed to compile:\n" + "\n".join(
        f"  {layer_id}: {reason}" for layer_id, reason in failures
    )
