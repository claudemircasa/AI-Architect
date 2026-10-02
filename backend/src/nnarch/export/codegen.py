"""
> [!AML-DOC-FILE]
@file       export/codegen.py
@description Turns a graph IR into a standalone, runnable TensorFlow project:
             a functional model builder, a training script, a data loader, pinned
             requirements, a README and any custom layer sources it needs.
@module     nnarch.export.codegen
@exports    ExportOptions, ExportBundle, export_project, build_zip
@created    2026-09-30
@context    RISK:MED [amm: E.4]. The graph is compiled here before emission, so the
            exported model is generated from the same parameters the running model
            used and the two cannot silently diverge.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field

from nnarch import __version__
from nnarch.catalog import CallStyle, Diagnostic, Registry
from nnarch.ir.compiler import CompileResult, coerce_params, compile_graph
from nnarch.data import DatasetChoice, DatasetKind, make_adapter
from nnarch.ir.schema import GraphIR

from .inline import collect_inline_sources
from nnarch.naming import python_variable, sanitize_name

from .render import IdentifierPool, render_kwargs, render_value

class ExportOptions(BaseModel):
    """
    > [!AML-DOC-UNIT]
    Choices that shape the generated project.
    @param dataset       data source for the training script
    @param optimizer     Keras optimizer name
    @param learning_rate initial learning rate
    @param loss          Keras loss name; empty means infer it from the output layer
    @param metrics       metric names to track
    @param epochs        default epoch count baked into the script's argparse
    @param batch_size    default batch size baked into the script's argparse
    @param project_name  directory-safe name used in the README and saved filenames
    @param include_saved_model  also write `model.keras`, the untrained architecture
                                and weights in Keras v3 format
    @param include_model_json   also write `model.json`, the architecture as returned
                                by `Model.to_json()`
    """

    dataset: DatasetChoice = Field(default_factory=DatasetChoice)
    optimizer: str = "adam"
    learning_rate: float = 1e-3
    loss: str = ""
    metrics: list[str] = Field(default_factory=lambda: ["accuracy"])
    epochs: int = 5
    batch_size: int = 32
    project_name: str = ""
    include_saved_model: bool = False
    include_model_json: bool = False


@dataclass
class ExportBundle:
    """
    > [!AML-DOC-UNIT]
    The generated project.
    @param files       mapping of relative path to text file contents
    @param binaries    mapping of relative path to binary file contents, used for
                       the optional `model.keras` archive
    @param diagnostics problems found while compiling or emitting
    @param params      weight count of the compiled model, for parity checking
    @sideEffects none
    """

    files: dict[str, str] = field(default_factory=dict)
    binaries: dict[str, bytes] = field(default_factory=dict)
    diagnostics: list[Diagnostic] = field(default_factory=list)
    params: int = 0

    @property
    def ok(self) -> bool:
        """
        > [!AML-DOC-UNIT]
        Whether a usable project was produced.
        @returns True when files were emitted and no diagnostic is an error
        """
        return bool(self.files) and not any(d.severity == "error" for d in self.diagnostics)


def _infer_loss(result: CompileResult, graph: GraphIR, registry: Registry) -> str:
    """
    > [!AML-DOC-UNIT]
    Pick a loss function that matches the model's final layer.
    @param result   the compiled model, read for the output shape
    @param graph    the IR, read for the output node's activation
    @param registry the layer catalog
    @returns a Keras loss name
    @sideEffects none
    @context Guesses from the output head the way a practitioner would: a softmax
             over several units is categorical, a single sigmoid is binary, and
             anything else is treated as regression.
    """
    output_ids = graph.resolved_outputs()
    if not output_ids:
        return "mse"
    node = graph.node_map().get(output_ids[-1])
    shape = result.shapes.get(output_ids[-1])
    if node is None or shape is None:
        return "mse"

    spec = registry.get(node.type)
    activation = node.params.get("activation") or (
        spec.defaults().get("activation") if spec else None
    )
    units = shape.shape[-1] if shape.shape else None

    if activation == "softmax":
        return "sparse_categorical_crossentropy"
    if activation == "sigmoid" and units == 1:
        return "binary_crossentropy"
    if node.type == "keras.Softmax":
        return "sparse_categorical_crossentropy"
    return "mse"


def _emit_model_module(
    graph: GraphIR,
    registry: Registry,
    result: CompileResult,
    inline_symbols: list[str],
) -> tuple[str, list[Diagnostic]]:
    """
    > [!AML-DOC-UNIT]
    Emit `model.py`: a function that rebuilds the architecture with the Keras
    functional API.
    @param graph          the architecture to write out
    @param registry       the layer catalog
    @param result         the compiled model, used only for its node ordering inputs
    @param inline_symbols names defined in `custom_layers.py`, which must be imported
    @returns (source text, diagnostics raised while rendering)
    @raises nothing; a value with no literal form becomes an error diagnostic
    @sideEffects none
    @context Nodes are written in topological order, so the file reads top to bottom
             the way the data flows. Layer names are drawn from the same unseeded
             pool the compiler uses, so both models name their layers identically;
             the Python variable is derived from that name and only differs when it
             would shadow something [amm: E.4].
    """
    diagnostics: list[Diagnostic] = []
    ordered, _ = graph.topological_order()
    node_map = graph.node_map()
    pool = IdentifierPool()
    variables: dict[str, str] = {}
    body: list[str] = []

    for node_id in ordered:
        node = node_map[node_id]
        spec = registry.require(node.type)
        params = coerce_params(spec, node.params)
        params.pop("trainable", None) if params.get("trainable") is True else None

        if node.disabled:
            sources = [
                edge.source for edge in graph.incoming(node_id)
                if edge.source in variables
            ]
            if len(sources) == 1:
                variables[node_id] = variables[sources[0]]
                body.append(f"    # {spec.label} {node.name or node_id!r} is disabled")
            continue

        layer_name = pool.allocate(node.name, node_id)
        variable = python_variable(layer_name)
        variables[node_id] = variable
        constructor = _constructor_source(
            node, spec, params, registry, graph, pool, diagnostics, layer_name=layer_name
        )
        if constructor is None:
            continue

        if spec.call_style is CallStyle.SOURCE:
            body.append(f"    {variable} = {constructor}")
            continue

        edges = [
            edge for edge in graph.incoming(node_id)
            if edge.source in variables and edge.source not in graph.child_node_ids()
        ]
        if spec.call_style is CallStyle.QUERY_VALUE_KEY:
            arguments = ", ".join(
                f"{edge.target_port}={variables[edge.source]}" for edge in edges
            )
        elif spec.call_style is CallStyle.LIST:
            arguments = "[" + ", ".join(variables[edge.source] for edge in edges) + "]"
        elif spec.call_style is CallStyle.RECURRENT:
            # Must match `_arguments` in the compiler exactly: the running model and
            # the generated one are the same model or the export is a lie [amm: E.4].
            sequence = [
                variables[edge.source] for edge in edges
                if (edge.target_port or "input") == "input"
            ]
            state = [
                variables[edge.source] for edge in sorted(
                    (edge for edge in edges if edge.target_port == "initial_state"),
                    key=lambda edge: edge.order,
                )
            ]
            arguments = ", ".join(sequence)
            if state:
                arguments += f", initial_state=[{', '.join(state)}]"
        else:
            arguments = ", ".join(variables[edge.source] for edge in edges)
        body.append(f"    {variable} = {constructor}({arguments})")

    input_vars = [variables[n] for n in graph.resolved_inputs() if n in variables]
    output_vars = [variables[n] for n in graph.resolved_outputs() if n in variables]
    inputs_source = input_vars[0] if len(input_vars) == 1 else "[" + ", ".join(input_vars) + "]"
    outputs_source = output_vars[0] if len(output_vars) == 1 else "[" + ", ".join(output_vars) + "]"
    model_name = sanitize_name(graph.name, "model")

    imports = ["import keras", "from keras import layers"]
    if inline_symbols:
        imports.append(f"from custom_layers import {', '.join(sorted(inline_symbols))}")

    source = "\n".join([
        f'"""{graph.name} — model definition.',
        "",
        f"Generated by AI Architect {__version__}. This file has no dependency on the",
        "tool that produced it; edit it freely.",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        *imports,
        "",
        "",
        "def build_model() -> keras.Model:",
        f'    """Build the {graph.name} architecture."""',
        *body,
        f'    return keras.Model(inputs={inputs_source}, outputs={outputs_source},'
        f' name={render_value(model_name)})',
        "",
        "",
        'if __name__ == "__main__":',
        "    build_model().summary()",
        "",
    ])
    return source, diagnostics


def _constructor_source(
    node: Any,
    spec: Any,
    params: dict[str, Any],
    registry: Registry,
    graph: GraphIR,
    pool: IdentifierPool,
    diagnostics: list[Diagnostic],
    *,
    layer_name: str,
) -> str | None:
    """
    > [!AML-DOC-UNIT]
    Render the call that constructs one layer.
    @param node        the node being written
    @param spec        its catalog spec
    @param params      effective constructor parameters
    @param registry    the layer catalog, for resolving a wrapped child
    @param graph       the owning graph, for resolving a wrapped child
    @param pool        identifier pool, so a child's Keras name stays unique
    @param diagnostics list appended to when a parameter cannot be rendered
    @param layer_name  the allocated name, used for both the Python variable and the
                       Keras `name=` argument so the two models agree [amm: E.4]
    @returns constructor source such as `layers.Conv2D(filters=32, name="c1")`, or
             None when rendering failed
    @sideEffects appends to `diagnostics` and may allocate an identifier
    """
    inline: dict[str, str] = {}

    if isinstance(params.get("function"), str):
        inline["function"] = f"lambda x: {params['function']}"

    child_id = params.get("layer")
    if isinstance(child_id, str):
        child = graph.node_map().get(child_id)
        if child is None:
            diagnostics.append(Diagnostic(
                severity="error", code="export_missing_child", node_id=node.id,
                message=f"{spec.label} wraps node {child_id!r}, which is not in the graph.",
            ))
            return None
        child_spec = registry.require(child.type)
        child_params = coerce_params(child_spec, child.params)
        child_params.pop("trainable", None) if child_params.get("trainable") is True else None
        child_name = pool.allocate(child.name, child.id)
        child_class = _class_reference(child_spec)
        child_args = render_kwargs(child_spec, child_params, name=child_name)
        inline["layer"] = f"{child_class}({child_args})"

    try:
        arguments = render_kwargs(spec, params, name=layer_name, inline=inline)
    except TypeError as exc:
        diagnostics.append(Diagnostic(
            severity="error", code="export_unrenderable_param", node_id=node.id,
            message=f"{spec.label} has a setting that cannot be written to code: {exc}",
        ))
        return None
    return f"{_class_reference(spec)}({arguments})"


def _class_reference(spec: Any) -> str:
    """
    > [!AML-DOC-UNIT]
    Render how a layer class is referred to in the generated file.
    @param spec the layer's catalog spec
    @returns `keras.Input`, `layers.Conv2D` or a bare class name for an inlined
             custom layer
    """
    path = spec.keras_path
    if path == "keras.layers.Input":
        return "keras.Input"
    if path.startswith("keras.layers."):
        return f"layers.{path.rsplit('.', 1)[1]}"
    return path.rsplit(".", 1)[1]


def _emit_data_module(options: ExportOptions, result: CompileResult, graph: GraphIR) -> str:
    """
    > [!AML-DOC-UNIT]
    Emit `data.py` by asking the chosen data source to write its own loader.
    @param options export choices, read for the dataset selection
    @param result  the compiled model, read for the real input and output shapes
    @param graph   the IR, read to locate the input and output nodes
    @returns source text defining `load_data()`
    @sideEffects none
    @context The adapter that feeds the in-app preview is the same object that emits
             this code [amm: E.4], so the exported project cannot preprocess
             differently from what the user saw on screen.
    """
    input_ids = graph.resolved_inputs()
    output_ids = graph.resolved_outputs()
    input_shape = list(result.shapes[input_ids[0]].shape[1:]) if input_ids else [1]
    output_shape = result.shapes[output_ids[-1]].shape[1:] if output_ids else [1]
    classes = int(output_shape[-1]) if output_shape and output_shape[-1] else None

    adapter = make_adapter(options.dataset, input_shape, classes)
    return adapter.codegen()


def _emit_train_module(options: ExportOptions, loss: str) -> str:
    """
    > [!AML-DOC-UNIT]
    Emit `train.py`: an argparse-driven training entry point.
    @param options export choices, which become the script's default arguments
    @param loss    the inferred or explicitly chosen loss name
    @returns source text
    @sideEffects none
    """
    metrics = ", ".join(render_value(metric) for metric in options.metrics)
    return "\n".join([
        '"""Train the model defined in model.py."""',
        "",
        "from __future__ import annotations",
        "",
        "import argparse",
        "",
        "import keras",
        "",
        "from data import load_data",
        "from model import build_model",
        "",
        "",
        "def main() -> int:",
        '    """Parse arguments, train the model and save the result."""',
        "    parser = argparse.ArgumentParser(description=__doc__)",
        f"    parser.add_argument('--epochs', type=int, default={options.epochs})",
        f"    parser.add_argument('--batch-size', type=int, default={options.batch_size})",
        f"    parser.add_argument('--lr', type=float, default={options.learning_rate})",
        f"    parser.add_argument('--optimizer', default={render_value(options.optimizer)})",
        f"    parser.add_argument('--loss', default={render_value(loss)})",
        "    parser.add_argument('--out', default='model.keras',",
        "                        help='where to save the trained model')",
        "    parser.add_argument('--summary', action='store_true',",
        "                        help='print the architecture and exit')",
        "    args = parser.parse_args()",
        "",
        "    model = build_model()",
        "    if args.summary:",
        "        model.summary()",
        "        return 0",
        "",
        "    optimizer = keras.optimizers.get(args.optimizer)",
        "    optimizer.learning_rate = args.lr",
        f"    model.compile(optimizer=optimizer, loss=args.loss, metrics=[{metrics}])",
        "",
        "    (x_train, y_train), (x_test, y_test) = load_data()",
        "    model.fit(",
        "        x_train, y_train,",
        "        epochs=args.epochs,",
        "        batch_size=args.batch_size,",
        "        validation_data=(x_test, y_test),",
        "    )",
        "",
        "    model.save(args.out)",
        "    print(f'saved {args.out}')",
        "    return 0",
        "",
        "",
        'if __name__ == "__main__":',
        "    raise SystemExit(main())",
        "",
    ])


def _emit_readme(graph: GraphIR, options: ExportOptions, params: int, loss: str) -> str:
    """
    > [!AML-DOC-UNIT]
    Emit the exported project's README.
    @param graph   the architecture, for its name and layer count
    @param options export choices, for the documented defaults
    @param params  weight count, so the reader can confirm the model matches
    @param loss    the loss the training script defaults to
    @returns Markdown text
    @sideEffects none
    """
    source = (
        f"the `{options.dataset.name}` dataset"
        if options.dataset.kind is DatasetKind.BUILTIN
        else f"the folder `{options.dataset.path}`"
        if options.dataset.kind is DatasetKind.FOLDER
        else f"{options.dataset.samples} generated samples"
    )
    return "\n".join([
        f"# {graph.name}",
        "",
        f"A Keras model exported from AI Architect {__version__}.",
        f"{len(graph.dataflow_nodes())} layers, {params:,} parameters.",
        "",
        "This project is self-contained: nothing here imports AI Architect.",
        "",
        "## Setup",
        "",
        "```bash",
        "python3.13 -m venv .venv",
        ".venv/bin/python -m pip install -r requirements.txt",
        "```",
        "",
        "TensorFlow 2.21 requires Python 3.13; it has no wheel for 3.14 or newer.",
        "",
        "## Inspect the architecture",
        "",
        "```bash",
        ".venv/bin/python model.py",
        "```",
        "",
        "## Train",
        "",
        "```bash",
        ".venv/bin/python train.py",
        f".venv/bin/python train.py --epochs 20 --batch-size 64 --lr 3e-4",
        "```",
        "",
        f"Trains on {source}, with `{options.optimizer}` and `{loss}`.",
        f"Defaults are {options.epochs} epochs at batch size {options.batch_size}.",
        "The trained model is written to `model.keras`.",
        "",
        "## Files",
        "",
        "| File | Purpose |",
        "|------|---------|",
        "| `model.py` | Builds the architecture. Edit this to change the model. |",
        "| `train.py` | Compiles and fits it. |",
        "| `data.py` | Loads the data. Replace `load_data()` to use your own. |",
        "| `requirements.txt` | Pinned dependencies. |",
        "",
    ])


def export_project(
    graph: GraphIR,
    registry: Registry,
    options: ExportOptions | None = None,
) -> ExportBundle:
    """
    > [!AML-DOC-UNIT]
    Generate a complete, runnable project from a graph.
    @param graph    the architecture to export
    @param registry the layer catalog
    @param options  export choices; defaults to synthetic data and Adam
    @returns ExportBundle with the files, any diagnostics, and the compiled model's
             weight count for parity checking against the app
    @raises nothing; a graph that does not compile comes back as diagnostics with no
            files
    @sideEffects compiles the graph, which imports TensorFlow and builds a model
    @context The graph is compiled first, so emission works from the same effective
             parameters the running model used [amm: E.4], and the reported weight
             count is the real one rather than a recount.
    """
    options = options or ExportOptions()
    result = compile_graph(graph, registry)
    if not result.ok:
        return ExportBundle(diagnostics=result.diagnostics)

    used_specs = [registry.require(node.type) for node in graph.nodes]
    inline_result = collect_inline_sources(used_specs)
    diagnostics = list(result.diagnostics)
    for problem in inline_result.problems:
        diagnostics.append(Diagnostic(
            severity="error", code="export_inline_failed",
            message=f"A custom layer could not be made standalone: {problem}",
        ))
    if any(d.severity == "error" for d in diagnostics):
        return ExportBundle(diagnostics=diagnostics)

    model_source, render_diagnostics = _emit_model_module(
        graph, registry, result, inline_result.symbols
    )
    diagnostics.extend(render_diagnostics)
    if any(d.severity == "error" for d in diagnostics):
        return ExportBundle(diagnostics=diagnostics)

    loss = options.loss or _infer_loss(result, graph, registry)
    files = {
        "model.py": model_source,
        "data.py": _emit_data_module(options, result, graph),
        "train.py": _emit_train_module(options, loss),
        "requirements.txt": "tensorflow==2.21.*\nnumpy>=1.26\n",
        "README.md": _emit_readme(graph, options, result.params_total, loss),
    }
    if inline_result.needed:
        files["custom_layers.py"] = inline_result.source

    binaries: dict[str, bytes] = {}
    if options.include_model_json:
        files["model.json"] = result.model.to_json(indent=2)
    if options.include_saved_model:
        saved = _serialize_model(result.model)
        if saved is None:
            diagnostics.append(Diagnostic(
                severity="warning", code="export_saved_model_failed",
                message=(
                    "The architecture could not be written as model.keras; the "
                    "generated source files are unaffected."
                ),
            ))
        else:
            binaries["model.keras"] = saved

    return ExportBundle(
        files=files, binaries=binaries, diagnostics=diagnostics, params=result.params_total
    )


def _serialize_model(model: Any) -> bytes | None:
    """
    > [!AML-DOC-UNIT]
    Write a model to the Keras v3 archive format and return its bytes.
    @param model the compiled Keras model
    @returns the archive contents, or None when Keras refused to serialise it
    @sideEffects writes to a temporary file, which is removed before returning
    @context Keras only saves to a path, not a buffer, so a temporary file is
             unavoidable. A model holding an unserialisable layer fails here rather
             than at load time in the user's project, which is why the failure is a
             warning attached to the export rather than an exception.
    """
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as directory:
        target = Path(directory) / "model.keras"
        try:
            model.save(target)
        except Exception:
            return None
        return target.read_bytes()


def build_zip(bundle: ExportBundle, root: str = "") -> bytes:
    """
    > [!AML-DOC-UNIT]
    Pack an export bundle into a zip archive.
    @param bundle the generated project
    @param root   optional directory prefix inside the archive
    @returns the archive bytes, ready to stream to the desktop shell's save dialog
    @sideEffects none; the archive is built in memory
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path, content in sorted(bundle.files.items()):
            archive.writestr(f"{root}/{path}" if root else path, content)
        for path, blob in sorted(bundle.binaries.items()):
            archive.writestr(f"{root}/{path}" if root else path, blob)
    return buffer.getvalue()
