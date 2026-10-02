"""
> [!AML-DOC-FILE]
@file       tests/test_multi_io.py
@description Regression tests for models that do not have one input and one output:
             recurrent layers given a starting state, layers that return their state
             as a second output, and models with several inputs.
@module     tests.test_multi_io
@exports    (pytest test functions)
@created    2026-10-01
@context    All of this came out of one real model that could not be opened at all
             [E-032..E-036]. A recurrent generator feeds its own state back, which is
             the whole point of it, and the tool could neither represent that wiring
             nor run a model built on it. The parity test matters most: a state
             connection that the exported code drops would produce a model that
             trains and generates differently from the one on the canvas [amm: E.4].
"""

from __future__ import annotations

import numpy as np
import pytest

from nnarch.catalog import Registry, bootstrap
from nnarch.data import make_adapter
from nnarch.data.spec import DatasetChoice
from nnarch.export import ExportOptions, export_project
from nnarch.ir import GraphIR, compile_graph
import json

from nnarch.ir.project import import_keras_json
from nnarch.ir.schema import Edge, Node
from nnarch.viz import capture_activations
from nnarch.viz.tensors import primary_input_index


@pytest.fixture(scope="module")
def registry() -> Registry:
    """
    > [!AML-DOC-UNIT]
    The catalog, built once for the module.
    @returns the bootstrapped registry
    """
    return bootstrap()


def _node(node_id: str, type_id: str, **params: object) -> Node:
    """
    > [!AML-DOC-UNIT]
    A graph node with the given parameters.
    @param node_id identifier and display name
    @param type_id catalog layer id
    @param params  layer parameters
    @returns the node
    """
    return Node(id=node_id, type=type_id, name=node_id, params=dict(params))


def _stateful_gru() -> GraphIR:
    """
    > [!AML-DOC-UNIT]
    A GRU given an explicit starting state, and returning its state as well.
    @returns the graph
    @context This is the shape of a free-running generator: it is handed the state it
             left off with, and hands back the state to resume from.
    """
    return GraphIR(
        name="stateful",
        nodes=[
            _node("seq", "keras.Input", shape=[8, 4]),
            _node("state_in", "keras.Input", shape=[16]),
            _node("gru", "keras.GRU", units=16, return_state=True),
            _node("head", "keras.Dense", units=3),
            _node("state_out", "keras.Identity"),
        ],
        edges=[
            Edge(id="e1", source="seq", target="gru", target_port="input"),
            Edge(id="e2", source="state_in", target="gru", target_port="initial_state"),
            Edge(id="e3", source="gru", target="head"),
            # The second thing the GRU returns is its state.
            Edge(id="e4", source="gru", source_port="output_1", target="state_out"),
        ],
    )


def test_a_recurrent_layer_accepts_a_starting_state(registry: Registry) -> None:
    """
    > [!AML-DOC-UNIT]
    A GRU wired to an `initial_state` input compiles.
    @param registry the catalog
    @raises AssertionError when the graph will not compile
    @context The catalog used to declare one input for every recurrent layer, so this
             wiring was rejected as "GRU has no input called 'initial_state'" and a
             model built this way could not be opened [E-032].
    """
    result = compile_graph(_stateful_gru(), registry)
    errors = [d for d in result.diagnostics if d.severity == "error"]
    assert result.model is not None, [d.message for d in errors]
    assert len(result.model.inputs) == 2


def test_the_second_output_of_a_layer_is_reachable(registry: Registry) -> None:
    """
    > [!AML-DOC-UNIT]
    An edge from `output_1` carries the state, not the output.
    @param registry the catalog
    @raises AssertionError when the wrong tensor is wired through
    @context Both outputs of this GRU have 16 channels, so a test that only checked
             the shape would pass while the wrong tensor flowed. The state is checked
             by identity instead: it must be the *second* thing the layer returned
             [E-034].
    """
    result = compile_graph(_stateful_gru(), registry)
    assert result.model is not None

    gru = next(layer for layer in result.model.layers if layer.name == "gru")
    sequence, state = gru.output
    state_out = next(layer for layer in result.model.layers if layer.name == "state_out")

    carried = state_out.input
    assert carried is state, "state_out received the GRU's output rather than its state"
    assert carried is not sequence


def test_a_starting_state_survives_export(registry: Registry) -> None:
    """
    > [!AML-DOC-UNIT]
    The generated code passes the state too.
    @param registry the catalog
    @raises AssertionError when the emitted call drops the keyword
    @context A dropped `initial_state` leaves a model that compiles and runs and is
             not the one that was drawn — the failure the export parity rule exists
             to prevent [amm: E.4].
    """
    bundle = export_project(_stateful_gru(), registry, ExportOptions(
        dataset=DatasetChoice(kind="synthetic"),
    ))
    source = bundle.files["model.py"]
    assert "initial_state=[" in source, source


def test_an_unknown_layer_does_not_sink_the_whole_model(registry: Registry) -> None:
    """
    > [!AML-DOC-UNIT]
    A layer the catalog does not have is imported as a pass-through.
    @param registry the catalog
    @raises AssertionError when the import refuses or hides the substitution
    @context One custom layer used to make an entire model impossible to compile, so
             nothing in it could be seen or run. It becomes a pass-through, and the
             diagnostic names it, because a silent substitution would be worse than
             the refusal it replaced [E-033].
    """
    config = {
        "class_name": "Functional",
        "config": {
            "name": "custom",
            "layers": [
                {"class_name": "InputLayer", "name": "in",
                 "config": {"name": "in", "batch_shape": [None, 4]},
                 "inbound_nodes": []},
                {"class_name": "SomebodysOwnLayer", "name": "mystery",
                 "config": {"name": "mystery", "rate": 0.25},
                 "inbound_nodes": [{"args": [{"class_name": "__keras_tensor__",
                                              "config": {"keras_history": ["in", 0, 0]}}],
                                    "kwargs": {}}]},
                {"class_name": "Dense", "name": "out",
                 "config": {"name": "out", "units": 2},
                 "inbound_nodes": [{"args": [{"class_name": "__keras_tensor__",
                                              "config": {"keras_history": ["mystery", 0, 0]}}],
                                    "kwargs": {}}]},
            ],
        },
    }
    graph, diagnostics = import_keras_json(json.dumps(config), registry)
    assert graph is not None

    said = " ".join(d.message for d in diagnostics)
    assert "SomebodysOwnLayer" in said, said
    assert "pass-through" in said, said

    mystery = next(node for node in graph.nodes if node.id == "mystery")
    assert mystery.params["original_type"] == "SomebodysOwnLayer"
    # What it was configured with is kept, so putting it back loses nothing.
    assert mystery.params["original_params"]["rate"] == 0.25

    result = compile_graph(graph, registry)
    assert result.model is not None, [
        d.message for d in result.diagnostics if d.severity == "error"
    ]


def test_a_lambda_carrying_compiled_code_is_refused_and_explained(
    registry: Registry,
) -> None:
    """
    > [!AML-DOC-UNIT]
    A Lambda whose body is a marshalled code object is not executed.
    @param registry the catalog
    @raises AssertionError when the body is kept or the refusal is not explained
    @context Executing it would mean running code out of a model file, which is a
             thing a file format should never be able to talk anyone into. The
             refusal has to say what to type instead, or it is just a dead end
             [E-035].
    """
    config = {
        "class_name": "Functional",
        "config": {
            "name": "lam",
            "layers": [
                {"class_name": "InputLayer", "name": "in",
                 "config": {"name": "in", "batch_shape": [None, 6, 4]},
                 "inbound_nodes": []},
                {"class_name": "Lambda", "name": "last",
                 "config": {"name": "last", "output_shape": [4],
                            "function": {"class_name": "__lambda__",
                                         "config": {"code": "4wEAAAA=", "closure": None}}},
                 "inbound_nodes": [{"args": [{"class_name": "__keras_tensor__",
                                              "config": {"keras_history": ["in", 0, 0]}}],
                                    "kwargs": {}}]},
            ],
        },
    }
    graph, diagnostics = import_keras_json(json.dumps(config), registry)
    assert graph is not None

    node = next(n for n in graph.nodes if n.id == "last")
    assert node.params["function"] == "", "the compiled body was kept"

    said = " ".join(d.message for d in diagnostics)
    assert "compiled code" in said, said
    assert "[4]" in said, "the shape the expression has to produce was not offered"


def test_a_model_with_several_inputs_can_be_previewed(registry: Registry) -> None:
    """
    > [!AML-DOC-UNIT]
    The sample feeds the input it fits, and the rest start at zero.
    @param registry the catalog
    @raises AssertionError when the preview fails or does not say what it did
    @context The preview fed `model.inputs[0]` a single array, so a model with more
             than one input could not be run at all — which is what "I cannot add an
             input to simulate the network" turned out to mean [E-036].
    """
    graph = GraphIR(
        name="two-in",
        nodes=[
            _node("picture", "keras.Input", shape=[8, 8, 1]),
            _node("knob", "keras.Input", shape=[3]),
            _node("flat", "keras.Flatten"),
            _node("join", "keras.Concatenate"),
            _node("out", "keras.Dense", units=2),
        ],
        edges=[
            Edge(id="e1", source="picture", target="flat"),
            Edge(id="e2", source="flat", target="join", target_port="inputs", order=0),
            Edge(id="e3", source="knob", target="join", target_port="inputs", order=1),
            Edge(id="e4", source="join", target="out"),
        ],
    )

    compiled = compile_graph(graph, registry)
    assert compiled.model is not None, [
        d.message for d in compiled.diagnostics if d.severity == "error"
    ]

    # The picture is the sample; the knob is the thing that steers it.
    primary = primary_input_index(compiled.model)
    assert compiled.model.inputs[primary].shape[1:] == (8, 8, 1)

    adapter = make_adapter(DatasetChoice(kind="synthetic"), [8, 8, 1], 2)
    result = capture_activations(graph, registry, adapter, colormap="viridis")

    assert result.activations, [d.message for d in result.diagnostics]
    said = " ".join(d.message for d in result.diagnostics)
    assert "started at zero" in said, said
    assert "knob" in said, "the preview did not say which inputs it made up"


def test_the_sample_reaches_the_input_it_fits(registry: Registry) -> None:
    """
    > [!AML-DOC-UNIT]
    An exact shape match wins over mere size.
    @param registry the catalog
    @raises AssertionError when the widest input is chosen over the matching one
    @context Without the sample, the widest input is the best guess. With it, a shape
             that matches exactly is not a guess at all, and preferring size would
             feed the data to the wrong place [E-036].
    """
    graph = GraphIR(
        name="pick",
        nodes=[
            _node("wide", "keras.Input", shape=[64]),
            _node("narrow", "keras.Input", shape=[4]),
            _node("join", "keras.Concatenate"),
            _node("out", "keras.Dense", units=2),
        ],
        edges=[
            Edge(id="e1", source="wide", target="join", target_port="inputs", order=0),
            Edge(id="e2", source="narrow", target="join", target_port="inputs", order=1),
            Edge(id="e3", source="join", target="out"),
        ],
    )
    model = compile_graph(graph, registry).model
    assert model is not None

    assert primary_input_index(model) == 0, "the widest input should be the fallback"
    sample = np.zeros((1, 4), dtype="float32")
    assert primary_input_index(model, sample) == 1, "an exact match must win"


def test_a_lambda_expression_naming_something_out_of_scope_is_refused(
    registry: Registry,
) -> None:
    """
    > [!AML-DOC-UNIT]
    An expression that mentions a name it cannot have is rejected when it is written.
    @param registry the catalog
    @raises AssertionError when the expression is accepted or the refusal is unclear
    @context Only the syntax used to be checked, so pasting the line of code that
             *built* the layer — rather than the expression inside it — was accepted
             and then raised `NameError` from inside Keras, mid forward pass, naming
             no layer. It is wrong the moment it is typed, and that is when to say so
             [E-040].
    """
    graph = GraphIR(
        name="bad-lambda",
        nodes=[
            _node("in", "keras.Input", shape=[10, 128]),
            _node(
                "lam",
                "keras.Lambda",
                function='layers.Lambda(lambda t: t[:, -1, :], name="f")(fast_x)',
                output_shape=[128],
            ),
        ],
        edges=[Edge(id="e", source="in", target="lam")],
    )
    result = compile_graph(graph, registry)
    assert result.model is None

    complaint = next(d for d in result.diagnostics if d.severity == "error")
    assert complaint.node_id == "lam", "the refusal has to say which layer"
    assert "'layers'" in complaint.message and "'fast_x'" in complaint.message
    assert "`x`" in complaint.message, "it must say what an expression may use"


def test_a_lambda_expression_may_bind_its_own_names(registry: Registry) -> None:
    """
    > [!AML-DOC-UNIT]
    A comprehension's variable is not an unknown name.
    @param registry the catalog
    @raises AssertionError when a valid expression is refused
    @context Refusing a valid expression would be a worse fault than the one the name
             check prevents, so the check subtracts whatever the expression binds for
             itself.
    """
    graph = GraphIR(
        name="good-lambda",
        nodes=[
            _node("in", "keras.Input", shape=[10, 128]),
            _node("lam", "keras.Lambda", function="ops.stack([x[:, i] for i in (0, 1)], axis=1)"),
        ],
        edges=[Edge(id="e", source="in", target="lam")],
    )
    result = compile_graph(graph, registry)
    assert result.model is not None, [
        d.message for d in result.diagnostics if d.severity == "error"
    ]


def test_the_two_expressions_this_model_needs(registry: Registry) -> None:
    """
    > [!AML-DOC-UNIT]
    The forms taken from the model's own source compile to the shapes it declares.
    @param registry the catalog
    @raises AssertionError when either expression is refused or reshapes wrongly
    @context `lambda t: t[:, -1, :]` and `lambda x: x` are what the generator's source
             actually contains. The only change is the variable's name, which is
             always `x` here, and this is the test that says so.
    """
    for expression, declared, expected in [
        ("x[:, -1, :]", [128], (None, 128)),
        ("x", [10, 128], (None, 10, 128)),
    ]:
        graph = GraphIR(
            name="shapes",
            nodes=[
                _node("in", "keras.Input", shape=[10, 128]),
                _node("lam", "keras.Lambda", function=expression, output_shape=declared),
            ],
            edges=[Edge(id="e", source="in", target="lam")],
        )
        result = compile_graph(graph, registry)
        assert result.model is not None, [
            d.message for d in result.diagnostics if d.severity == "error"
        ]
        layer = next(l for l in result.model.layers if l.name == "lam")
        assert tuple(layer.output.shape) == expected, expression
