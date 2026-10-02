"""
> [!AML-DOC-FILE]
@file       tests/test_project.py
@description Regression tests for the `.nnarch` project format: round-tripping,
             refusing files it cannot fully understand, atomic writes, and importing
             an existing Keras model.
@module     tests.test_project
@exports    (pytest test functions)
@created    2026-09-30
@context    Covers the task 12 gates. These guard user data, so the tests that matter
            most are the ones asserting what happens when something is wrong.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import keras
import pytest

from nnarch import IR_VERSION
from nnarch.catalog import Registry, bootstrap
from nnarch.ir import GraphIR, compile_graph
from nnarch.ir.importers import import_model
from nnarch.ir.project import (
    Codes,
    ProjectMeta,
    Viewport,
    import_keras_json,
    load_project,
    save_project,
    write_project_atomic,
)
from nnarch.ir.schema import Edge, Node, Position


@pytest.fixture(scope="session")
def registry() -> Registry:
    """
    > [!AML-DOC-UNIT]
    The bootstrapped layer catalog.
    @returns the shared Registry
    """
    return bootstrap()


def _graph() -> GraphIR:
    """
    > [!AML-DOC-UNIT]
    A small graph with deliberately awkward positions and parameters.
    @returns the GraphIR
    @context Fractional coordinates and a tuple parameter are what a naive
             serialisation rounds or flattens.
    """
    return GraphIR(name="Round Trip", nodes=[
        Node(id="i", type="keras.Input", name="in",
             params={"shape": [28, 28, 1]}, position=Position(x=120.5, y=40.25)),
        Node(id="c", type="keras.Conv2D", name="conv block",
             params={"filters": 16, "kernel_size": [3, 3], "activation": "relu"},
             position=Position(x=-73.75, y=180.5), notes="first stage"),
        Node(id="d", type="keras.Dropout", name="drop",
             params={"rate": 0.25}, position=Position(x=12.0, y=320.0), disabled=True),
    ], edges=[Edge(id="e1", source="i", target="c"), Edge(id="e2", source="c", target="d")])


def test_round_trip_preserves_everything() -> None:
    """
    > [!AML-DOC-UNIT]
    Saving and reopening must return the same project, down to node positions and
    the canvas view. This is the task's headline gate: anything lost here is work
    the user has to redo.
    """
    graph = _graph()
    viewport = Viewport(x=-33.5, y=12.25, zoom=0.85)
    runs = [{"epoch": 1, "loss": 0.5}, {"epoch": 2, "loss": 0.3}]

    project, diagnostics = load_project(
        save_project(graph, viewport=viewport, runs=runs)
    )

    assert project is not None, [d.message for d in diagnostics]
    assert project.graph.name == graph.name
    assert project.graph.model_dump() == graph.model_dump()
    assert project.viewport.model_dump() == viewport.model_dump()
    assert project.runs == runs
    assert project.meta.created and project.meta.modified
    assert project.meta.ir_version == IR_VERSION


def test_archive_holds_the_documented_entries() -> None:
    """
    > [!AML-DOC-UNIT]
    The format is a zip with named parts, so a user can look inside it with ordinary
    tools rather than needing this program to read their own work.
    """
    archive = zipfile.ZipFile(
        io.BytesIO(
            save_project(
                _graph(),
                runs=[{"epoch": 1}],
                assets={"sample.png": b"\\x89PNG"},
                thumbnail=b"\\x89PNG",
            )
        )
    )
    names = set(archive.namelist())
    assert {"project.json", "meta.json", "thumbnail.png"} <= names
    assert any(name.startswith("runs/") for name in names)
    assert "assets/sample.png" in names


@pytest.mark.parametrize(
    "label,payload,code",
    [
        ("random bytes", b"not a zip at all", Codes.NOT_A_PROJECT),
        ("empty zip", None, Codes.MISSING_GRAPH),
        ("damaged json", None, Codes.MALFORMED_GRAPH),
    ],
)
def test_unreadable_files_are_refused_without_crashing(
    label: str, payload: bytes | None, code: str
) -> None:
    """
    > [!AML-DOC-UNIT]
    A file this build cannot read must come back as a diagnostic, never an
    exception, and must load nothing at all. A half-loaded project would silently
    discard part of the user's work.
    """
    if label == "empty zip":
        buffer = io.BytesIO()
        zipfile.ZipFile(buffer, "w").close()
        payload = buffer.getvalue()
    elif label == "damaged json":
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("project.json", "{ not json")
        payload = buffer.getvalue()

    project, diagnostics = load_project(payload)
    assert project is None
    assert [d.code for d in diagnostics] == [code]
    assert diagnostics[0].severity == "error"


def test_a_project_from_a_newer_build_is_refused() -> None:
    """
    > [!AML-DOC-UNIT]
    A format from the future is refused with an explanation rather than opened
    partially, and the message says the file is untouched.
    """
    archive = save_project(_graph().model_copy(update={"ir_version": IR_VERSION + 50}))
    project, diagnostics = load_project(archive)

    assert project is None
    assert [d.code for d in diagnostics] == [Codes.VERSION_TOO_NEW]
    assert "Update AI Architect" in diagnostics[0].message


def test_atomic_write_keeps_the_previous_file(tmp_path: Path) -> None:
    """
    > [!AML-DOC-UNIT]
    Saving over a project must leave the old one recoverable, and must never leave a
    half-written file in its place.
    """
    path = tmp_path / "demo.nnarch"
    write_project_atomic(path, save_project(_graph()))
    write_project_atomic(
        path, save_project(_graph().model_copy(update={"name": "Second"}))
    )

    assert {entry.name for entry in tmp_path.iterdir()} == {
        "demo.nnarch", "demo.nnarch.bak",
    }
    current, _ = load_project(path.read_bytes())
    backup, _ = load_project((tmp_path / "demo.nnarch.bak").read_bytes())
    assert current is not None and current.graph.name == "Second"
    assert backup is not None and backup.graph.name == "Round Trip"
    assert not list(tmp_path.glob("*.tmp")), "no temporary file may be left behind"


def test_importing_resnet50_reproduces_it_exactly(registry: Registry) -> None:
    """
    > [!AML-DOC-UNIT]
    Importing a real model's JSON must give a graph that compiles back to the same
    model, parameter for parameter.
    @context ResNet50 is the right test because it is large, has skip connections
             and merging layers, and nothing about it was designed with this
             importer in mind.
    """
    model = keras.applications.ResNet50(
        weights=None, input_shape=(64, 64, 3), classes=10
    )
    graph, diagnostics = import_keras_json(model.to_json(), registry)

    assert graph is not None
    assert not [d for d in diagnostics if d.severity == "error"], diagnostics
    assert len(graph.nodes) == len(model.layers)

    result = compile_graph(graph, registry)
    assert result.ok, [d.message for d in result.diagnostics][:3]
    assert result.params_total == model.count_params()


def test_importing_a_model_with_named_inputs(registry: Registry) -> None:
    """
    > [!AML-DOC-UNIT]
    Attention layers receive their tensors by keyword, not position, so the importer
    has to read `inbound_nodes` kwargs as well as args.
    """
    tokens = keras.Input((16, 32))
    attended = keras.layers.MultiHeadAttention(num_heads=2, key_dim=8)(
        query=tokens, value=tokens, key=tokens
    )
    pooled = keras.layers.GlobalAveragePooling1D()(attended)
    model = keras.Model(tokens, keras.layers.Dense(3)(pooled))

    graph, _ = import_keras_json(model.to_json(), registry)
    assert graph is not None

    ports = {
        edge.target_port
        for edge in graph.edges
        if graph.node_map()[edge.target].type == "keras.MultiHeadAttention"
    }
    assert {"query", "value", "key"} <= ports

    result = compile_graph(graph, registry)
    assert result.ok, [d.message for d in result.diagnostics][:3]
    assert result.params_total == model.count_params()


def test_import_rejects_json_that_is_not_a_model(registry: Registry) -> None:
    """
    > [!AML-DOC-UNIT]
    Arbitrary JSON must be refused with a message that says what was expected.
    """
    graph, diagnostics = import_keras_json('{"hello": "world"}', registry)
    assert graph is None
    assert diagnostics[0].code == Codes.IMPORT_NOT_FUNCTIONAL

    graph, diagnostics = import_keras_json("not json at all", registry)
    assert graph is None
    assert diagnostics[0].code == Codes.IMPORT_MALFORMED


def _tiny_sequential() -> keras.Model:
    """
    > [!AML-DOC-UNIT]
    A small sequential classifier, used across the import tests.
    @returns the model
    """
    return keras.Sequential([
        keras.Input((28, 28, 1)),
        keras.layers.Conv2D(8, 3),
        keras.layers.MaxPooling2D(),
        keras.layers.Flatten(),
        keras.layers.Dense(10, activation="softmax"),
    ], name="tiny")


def test_sequential_models_are_chained_on_import(registry: Registry) -> None:
    """
    > [!AML-DOC-UNIT]
    A sequential model records no wiring, because the wiring *is* the declaration
    order. Importing one has to chain its layers, or it arrives as a pile of
    disconnected boxes.
    @context This was a real gap: before it was fixed, importing any sequential model
             produced the right nodes and zero edges.
    """
    model = _tiny_sequential()
    graph, diagnostics = import_keras_json(model.to_json(), registry)

    assert graph is not None
    assert len(graph.edges) == len(graph.nodes) - 1
    assert any(d.code == Codes.IMPORT_SEQUENTIAL_CHAINED for d in diagnostics)

    result = compile_graph(graph, registry)
    assert result.ok, [d.message for d in result.diagnostics][:3]
    assert result.params_total == model.count_params()


def test_keras_archive_rebuilds_the_model_exactly(registry: Registry, tmp_path: Path) -> None:
    """
    > [!AML-DOC-UNIT]
    A `.keras` file carries the exact configuration, so importing one must rebuild
    the architecture parameter for parameter — for a functional model with skip
    connections as much as for a sequential one.
    """
    inputs = keras.Input((32, 32, 3))
    first = keras.layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
    second = keras.layers.Conv2D(16, 3, padding="same")(first)
    merged = keras.layers.Add()([first, second])
    pooled = keras.layers.GlobalAveragePooling2D()(merged)
    model = keras.Model(inputs, keras.layers.Dense(10, activation="softmax")(pooled))

    path = tmp_path / "model.keras"
    model.save(path)

    graph, diagnostics = import_model(path.read_bytes(), "model.keras", registry)
    assert graph is not None
    assert not [d for d in diagnostics if d.severity == "error"]

    result = compile_graph(graph, registry)
    assert result.ok, [d.message for d in result.diagnostics][:3]
    assert result.params_total == model.count_params()
    # The user must be told the weights did not come along.
    assert any("weights" in d.message for d in diagnostics)


def test_a_damaged_keras_file_is_refused(registry: Registry) -> None:
    """
    > [!AML-DOC-UNIT]
    A file that is not a `.keras` archive, or is one without a model inside, must be
    refused with a message that says what was expected.
    """
    graph, diagnostics = import_model(b"not a zip", "model.keras", registry)
    assert graph is None
    assert diagnostics[0].code == Codes.IMPORT_NOT_A_KERAS_FILE

    import io as _io
    import zipfile as _zipfile

    buffer = _io.BytesIO()
    with _zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("something.txt", "hello")
    graph, diagnostics = import_model(buffer.getvalue(), "model.keras", registry)
    assert graph is None
    assert diagnostics[0].code == Codes.IMPORT_NOT_A_KERAS_FILE


@pytest.mark.slow
def test_tflite_import_recovers_the_structure(registry: Registry) -> None:
    """
    > [!AML-DOC-UNIT]
    A `.tflite` file holds the converted graph rather than the original model, so the
    test asserts what that honestly gives: the right layers in the right order with
    the right widths, the conversion artefacts left out, and a warning saying the
    settings were not recoverable.
    """
    import tensorflow as tf

    model = _tiny_sequential()
    flatbuffer = tf.lite.TFLiteConverter.from_keras_model(model).convert()

    graph, diagnostics = import_model(flatbuffer, "model.tflite", registry)
    assert graph is not None

    types = [node.type for node in graph.nodes]
    assert types == [
        "keras.Input",
        "keras.Conv2D",
        "keras.MaxPooling2D",
        "keras.Reshape",
        "keras.Dense",
        "keras.Softmax",
    ]

    by_id = {node.id: node for node in graph.nodes}
    convolution = next(n for n in by_id.values() if n.type == "keras.Conv2D")
    assert convolution.params["filters"] == 8
    assert convolution.params["kernel_size"] == [3, 3]
    assert next(n for n in by_id.values() if n.type == "keras.Dense").params["units"] == 10

    # The shape plumbing a flatten lowers into must not appear as architecture.
    assert not any("strided_slice" in node.id or "pack" in node.id for node in graph.nodes)
    assert any(d.code == Codes.IMPORT_LOSSY and d.severity == "warning" for d in diagnostics)


def test_a_misnamed_model_file_still_opens(registry: Registry, tmp_path: Path) -> None:
    """
    > [!AML-DOC-UNIT]
    A correct file with the wrong extension should still open: the reader falls back
    to recognising the content.
    """
    path = tmp_path / "model.keras"
    _tiny_sequential().save(path)

    graph, _ = import_model(path.read_bytes(), "mystery.bin", registry)
    assert graph is not None and len(graph.nodes) == 5

    graph, diagnostics = import_model(b"\x00\x01 not a model at all", "mystery.bin", registry)
    assert graph is None
    assert diagnostics[0].severity == "error"
