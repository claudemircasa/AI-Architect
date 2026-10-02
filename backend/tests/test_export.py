"""
> [!AML-DOC-FILE]
@file       tests/test_export.py
@description Regression tests for code generation: that the emitted project is valid
             Python, is semantically identical to the compiled model, and runs with
             no access to this tool.
@module     tests.test_export
@exports    (pytest test functions)
@created    2026-09-30
@context    Covers the task 06 gates. The parity tests are the important ones: they
            are what stop a model from working in the app and breaking on export
            [amm: E.4].
"""

from __future__ import annotations

import ast
import subprocess
import sys
import textwrap
import zipfile
from pathlib import Path

import keras
import pytest

from nnarch.catalog import Category, Registry, bootstrap
from nnarch.catalog.introspect import derive_params
from nnarch.catalog.spec import LayerSpec
from nnarch.export import DatasetChoice, ExportOptions, build_zip, export_project
from nnarch.ir import GraphIR, compile_graph
from nnarch.ir.schema import Edge, Node
from nnarch.naming import SHADOWED_IDENTIFIERS

_PROBE_LAYER_ID = "probe.GatedDense"


@pytest.fixture(scope="session")
def registry() -> Registry:
    """
    > [!AML-DOC-UNIT]
    The layer catalog, extended with a fixture layer that stands in for the
    research layers until task 04 provides real ones.
    @returns the shared Registry, including `probe.GatedDense`
    @sideEffects registers one extra spec into the session-wide registry, which the
                 all-layers sweep in test_ir.py then also covers
    """
    shared = bootstrap()
    if shared.get(_PROBE_LAYER_ID) is None:
        shared.register(LayerSpec(
            id=_PROBE_LAYER_ID,
            label="Probe Gated Dense",
            category=Category.RESEARCH_FFN,
            keras_path="fixtures.probe_layers.ProbeGatedDense",
            # A starter value for the mandatory argument, following the same rule
            # the stock catalog applies: a layer dropped on the canvas must be
            # valid on arrival. Task 04's real research layers must do likewise,
            # and the all-layers sweep in test_ir.py enforces it.
            params=derive_params(
                "fixtures.probe_layers.ProbeGatedDense",
                overrides={"units": {"default": 32}},
            ),
            is_research=True,
            # The citation test requires every research spec to name its source;
            # this one says plainly that it has none.
            paper="test fixture, not a published architecture",
        ))
    return shared


def node(node_id: str, node_type: str, **params: object) -> Node:
    """
    > [!AML-DOC-UNIT]
    Build a graph node concisely.
    @param node_id   graph-unique id, also used as the node's label
    @param node_type catalog layer id
    @param params    constructor parameters
    @returns the Node
    """
    return Node(id=node_id, type=node_type, name=node_id, params=dict(params))


def edge(source: str, target: str) -> Edge:
    """
    > [!AML-DOC-UNIT]
    Build a graph edge with a generated id.
    @param source id of the producing node
    @param target id of the consuming node
    @returns the Edge
    """
    return Edge(id=f"{source}->{target}", source=source, target=target)


def _mnist_graph() -> GraphIR:
    """
    > [!AML-DOC-UNIT]
    A small image classifier whose two convolutions deliberately share a label, so
    name deduplication is exercised.
    @returns the GraphIR
    """
    return GraphIR(name="MNIST CNN", nodes=[
        node("inp", "keras.Input", shape=[28, 28, 1]),
        Node(id="c1", type="keras.Conv2D", name="conv block",
             params={"filters": 16, "kernel_size": [3, 3], "activation": "relu"}),
        node("p1", "keras.MaxPooling2D", pool_size=[2, 2]),
        Node(id="c2", type="keras.Conv2D", name="conv block",
             params={"filters": 32, "kernel_size": [3, 3], "activation": "relu"}),
        node("gap", "keras.GlobalAveragePooling2D"),
        node("out", "keras.Dense", units=10, activation="softmax"),
    ], edges=[edge("inp", "c1"), edge("c1", "p1"), edge("p1", "c2"),
              edge("c2", "gap"), edge("gap", "out")])


def test_generated_files_are_valid_python(registry: Registry) -> None:
    """
    > [!AML-DOC-UNIT]
    Every emitted .py file must parse. A syntax error here would only surface once
    the user tried to run their exported project.
    """
    bundle = export_project(_mnist_graph(), registry)
    assert bundle.ok, [d.message for d in bundle.diagnostics]

    emitted = {name for name in bundle.files if name.endswith(".py")}
    assert emitted == {"model.py", "data.py", "train.py"}
    for name in sorted(emitted):
        ast.parse(bundle.files[name], filename=name)


def test_exported_param_count_matches_the_app(registry: Registry, tmp_path: Path) -> None:
    """
    > [!AML-DOC-UNIT]
    The exported model must have exactly the weights the app reported. This is the
    headline parity gate: a mismatch means the generated source drifted from the
    parameters the compiler used [amm: E.4].
    """
    graph = _mnist_graph()
    bundle = export_project(graph, registry)
    (tmp_path / "model.py").write_text(bundle.files["model.py"])

    namespace: dict[str, object] = {}
    exec(compile(bundle.files["model.py"], "model.py", "exec"), namespace)
    exported = namespace["build_model"]()

    assert exported.count_params() == bundle.params
    assert exported.count_params() == compile_graph(graph, registry).params_total


def test_exported_layer_names_match_the_compiled_model(registry: Registry) -> None:
    """
    > [!AML-DOC-UNIT]
    Layer names must be byte-identical between the running model and the exported
    one, including how duplicate user labels are numbered. Names are how a user
    addresses layers afterwards, so a divergence silently breaks their code.
    """
    graph = _mnist_graph()
    bundle = export_project(graph, registry)

    namespace: dict[str, object] = {}
    exec(compile(bundle.files["model.py"], "model.py", "exec"), namespace)
    exported = namespace["build_model"]()
    compiled = compile_graph(graph, registry).model

    assert [layer.name for layer in exported.layers] == [
        layer.name for layer in compiled.layers
    ]
    assert "conv_block" in {layer.name for layer in exported.layers}
    assert "conv_block_2" in {layer.name for layer in exported.layers}


def test_special_calling_conventions_survive_export(registry: Registry) -> None:
    """
    > [!AML-DOC-UNIT]
    Merging, attention and wrapper layers each need a different call shape in the
    generated source. Exporting them and rebuilding must reproduce the compiled
    model's weight count.
    """
    graph = GraphIR(name="Conventions", nodes=[
        node("inp", "keras.Input", shape=[16, 32]),
        node("child", "keras.LSTM", units=8),
        node("bi", "keras.Bidirectional", layer="child"),
        node("mha", "keras.MultiHeadAttention", num_heads=2, key_dim=8),
        node("pool", "keras.GlobalAveragePooling1D"),
        node("cat", "keras.Concatenate"),
        node("lam", "keras.Lambda", function="x * 2.0"),
        node("out", "keras.Dense", units=4, activation="softmax"),
    ], edges=[
        edge("inp", "bi"),
        Edge(id="q", source="inp", target="mha", target_port="query"),
        Edge(id="v", source="inp", target="mha", target_port="value"),
        edge("mha", "pool"),
        Edge(id="c0", source="bi", target="cat", target_port="inputs", order=0),
        Edge(id="c1", source="pool", target="cat", target_port="inputs", order=1),
        edge("cat", "lam"), edge("lam", "out"),
    ])

    bundle = export_project(graph, registry)
    assert bundle.ok, [d.message for d in bundle.diagnostics]

    namespace: dict[str, object] = {}
    exec(compile(bundle.files["model.py"], "model.py", "exec"), namespace)
    exported = namespace["build_model"]()
    assert exported.count_params() == compile_graph(graph, registry).params_total
    assert "layers.Concatenate(name=\"cat\")([" in bundle.files["model.py"]
    assert "query=inp, value=inp" in bundle.files["model.py"]
    assert "lambda x: x * 2.0" in bundle.files["model.py"]


def test_only_used_custom_layers_are_inlined(registry: Registry) -> None:
    """
    > [!AML-DOC-UNIT]
    A custom layer travels with the export, together with the module-level helpers
    it needs, while unrelated classes from the same module stay behind.
    """
    graph = GraphIR(name="Inline Probe", nodes=[
        node("inp", "keras.Input", shape=[16]),
        node("gd", _PROBE_LAYER_ID, units=32),
        node("out", "keras.Dense", units=3, activation="softmax"),
    ], edges=[edge("inp", "gd"), edge("gd", "out")])

    bundle = export_project(graph, registry)
    assert bundle.ok, [d.message for d in bundle.diagnostics]

    custom = bundle.files["custom_layers.py"]
    ast.parse(custom, filename="custom_layers.py")
    assert "class ProbeGatedDense" in custom
    assert "def scaled_gelu" in custom, "the helper the layer calls must come along"
    assert "SCALE_FLOOR = 0.25" in custom, "the constant the helper reads must come along"
    assert "ProbeUnusedDense" not in custom, "an unused class must not be carried over"
    assert custom.count("from __future__") == 1
    assert "from custom_layers import ProbeGatedDense" in bundle.files["model.py"]
    assert "nnarch" not in custom


def test_no_emitted_source_references_this_tool(registry: Registry) -> None:
    """
    > [!AML-DOC-UNIT]
    No generated source file may mention this package, which is what "standalone"
    means in practice. The README's generated-by credit is prose, not an import.
    """
    graph = GraphIR(name="Inline Probe", nodes=[
        node("inp", "keras.Input", shape=[16]),
        node("gd", _PROBE_LAYER_ID, units=32),
    ], edges=[edge("inp", "gd")])

    bundle = export_project(graph, registry)
    for name, content in bundle.files.items():
        if name == "README.md":
            continue
        assert "nnarch" not in content, f"{name} refers to the tool"


def test_keras_defaults_are_elided_but_real_values_are_not(registry: Registry) -> None:
    """
    > [!AML-DOC-UNIT]
    Arguments equal to a layer's own default are left out so the file reads well,
    while anything the user actually chose is written explicitly.
    """
    graph = GraphIR(name="Elision", nodes=[
        node("inp", "keras.Input", shape=[8, 8, 3]),
        node("keep", "keras.Conv2D", filters=8, kernel_size=[3, 3], padding="same"),
        node("plain", "keras.MaxPooling2D", pool_size=[2, 2]),
    ], edges=[edge("inp", "keep"), edge("keep", "plain")])

    source = export_project(graph, registry).files["model.py"]
    assert 'padding="same"' in source, "a non-default choice must be written out"
    assert "filters=8" in source
    assert "strides=" not in source, "Conv2D's default strides must be elided"
    assert "pool_size=" not in source, "MaxPooling2D's default pool_size must be elided"
    assert "sparse=" not in source, "Input's sparse=False is semantically the default"


def test_variables_never_shadow_builtins_but_layer_names_are_untouched(
    registry: Registry,
) -> None:
    """
    > [!AML-DOC-UNIT]
    A layer labelled "input" or "model" is a natural thing for a user to draw. The
    generated Python variable must not shadow a builtin or a name the module binds,
    while the Keras layer name must stay exactly what the user typed.
    @context Renaming the layer instead would show the user a suffix they never asked
             for; PEP 8's trailing underscore fixes the variable without touching the
             model.
    """
    graph = GraphIR(name="Shadow Test", nodes=[
        Node(id="a", type="keras.Input", name="input", params={"shape": [8]}),
        Node(id="b", type="keras.Dense", name="model", params={"units": 16}),
        Node(id="c", type="keras.Dense", name="layers", params={"units": 8}),
        Node(id="d", type="keras.Dense", name="type",
             params={"units": 4, "activation": "softmax"}),
    ], edges=[edge("a", "b"), edge("b", "c"), edge("c", "d")])

    source = export_project(graph, registry).files["model.py"]
    tree = ast.parse(source)
    assigned = {
        target.id
        for statement in ast.walk(tree)
        if isinstance(statement, ast.Assign)
        for target in statement.targets
        if isinstance(target, ast.Name)
    }
    assert not assigned & SHADOWED_IDENTIFIERS, f"shadowing variables: {assigned}"

    namespace: dict[str, object] = {}
    exec(compile(source, "model.py", "exec"), namespace)
    exported = namespace["build_model"]()
    assert [layer.name for layer in exported.layers] == ["input", "model", "layers", "type"]
    assert [layer.name for layer in exported.layers] == [
        layer.name for layer in compile_graph(graph, registry).model.layers
    ]


def test_broken_graph_exports_nothing_and_says_why(registry: Registry) -> None:
    """
    > [!AML-DOC-UNIT]
    An architecture that does not compile must yield no files, so a half-written
    project never reaches disk.
    """
    graph = GraphIR(name="Broken", nodes=[node("d", "keras.Dense", units=4)])

    bundle = export_project(graph, registry)
    assert not bundle.ok
    assert bundle.files == {}
    assert any(d.severity == "error" for d in bundle.diagnostics)


def test_zip_contains_text_and_binary_members(registry: Registry) -> None:
    """
    > [!AML-DOC-UNIT]
    The archive streamed to the save dialog carries the sources and, when asked,
    the serialised model alongside them.
    """
    bundle = export_project(
        _mnist_graph(), registry,
        ExportOptions(include_saved_model=True, include_model_json=True),
    )
    assert "model.json" in bundle.files
    assert "model.keras" in bundle.binaries

    archive = zipfile.ZipFile(__import__("io").BytesIO(build_zip(bundle, root="proj")))
    names = set(archive.namelist())
    assert "proj/model.py" in names
    assert "proj/model.keras" in names
    assert archive.read("proj/model.py").decode() == bundle.files["model.py"]


@pytest.mark.slow
def test_exported_project_trains_without_the_tool(
    registry: Registry, tmp_path: Path
) -> None:
    """
    > [!AML-DOC-UNIT]
    The end-to-end gate: write the project to disk, run its training script in a
    subprocess with this package made unimportable, and require that it trains and
    saves a model. Synthetic data is used so the test needs no network.
    @context Blocking the import proves independence more sharply than a clean
             virtualenv would, and runs in seconds rather than minutes.
    """
    graph = GraphIR(name="Standalone", nodes=[
        node("inp", "keras.Input", shape=[16]),
        node("gd", _PROBE_LAYER_ID, units=16),
        node("out", "keras.Dense", units=3, activation="softmax"),
    ], edges=[edge("inp", "gd"), edge("gd", "out")])

    bundle = export_project(
        graph, registry,
        ExportOptions(dataset=DatasetChoice(kind="synthetic", samples=64), epochs=1),
    )
    assert bundle.ok, [d.message for d in bundle.diagnostics]
    for name, content in bundle.files.items():
        (tmp_path / name).write_text(content)

    runner = tmp_path / "_runner.py"
    runner.write_text(textwrap.dedent('''
        """Run a script with this tool made unimportable."""
        import pathlib
        import runpy
        import sys


        class Blocker:
            """Meta-path finder that refuses anything under the `nnarch` package."""

            def find_spec(self, fullname, path=None, target=None):
                if fullname.split(".")[0] == "nnarch":
                    raise ImportError(f"BLOCKED: {fullname}")
                return None


        sys.meta_path.insert(0, Blocker())
        for name in [n for n in sys.modules if n.split(".")[0] == "nnarch"]:
            del sys.modules[name]

        script, *argv = sys.argv[1:]
        sys.path.insert(0, str(pathlib.Path(script).resolve().parent))
        sys.argv = [script, *argv]
        runpy.run_path(script, run_name="__main__")
    ''').lstrip())

    completed = subprocess.run(
        [sys.executable, str(runner), str(tmp_path / "train.py"), "--epochs", "1"],
        capture_output=True, text=True, timeout=600, cwd=tmp_path,
        env={"PYTHONPATH": "", "PATH": "/usr/bin:/bin", "HOME": str(tmp_path)},
    )
    assert completed.returncode == 0, completed.stderr[-3000:]
    assert "BLOCKED" not in completed.stderr
    assert (tmp_path / "model.keras").exists(), "training did not save a model"
