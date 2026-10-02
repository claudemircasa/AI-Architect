"""
> [!AML-DOC-FILE]
@file       ir/importers.py
@description Reads existing models into editable graphs: Keras JSON, `.keras`
             archives and TensorFlow Lite flatbuffers.
@module     nnarch.ir.importers
@exports    import_model, import_keras_archive, import_tflite, TFLITE_OP_MAP
@created    2026-10-01
@context    The three formats do not give the same thing, and the diagnostics say so.
            A `.keras` archive carries the exact configuration, so it rebuilds the
            model parameter for parameter. A `.tflite` file carries a graph of
            *converted* operations rather than Keras layers, so what comes back is
            the structure and the shapes — faithful to the file, but usually needing
            work before it compiles.
"""

from __future__ import annotations

import io
import json
import warnings
import zipfile

from nnarch.catalog import Diagnostic, Registry

from .project import Codes, import_keras_json
from .schema import Edge, GraphIR, Node, Position

KERAS_CONFIG_ENTRY = "config.json"

TFLITE_OP_MAP: dict[str, str] = {
    "CONV_2D": "keras.Conv2D",
    "DEPTHWISE_CONV_2D": "keras.DepthwiseConv2D",
    "TRANSPOSE_CONV": "keras.Conv2DTranspose",
    "FULLY_CONNECTED": "keras.Dense",
    "MAX_POOL_2D": "keras.MaxPooling2D",
    "AVERAGE_POOL_2D": "keras.AveragePooling2D",
    "MEAN": "keras.GlobalAveragePooling2D",
    "RESHAPE": "keras.Reshape",
    "SOFTMAX": "keras.Softmax",
    "RELU": "keras.ReLU",
    "RELU6": "keras.ReLU",
    "LEAKY_RELU": "keras.LeakyReLU",
    "PRELU": "keras.PReLU",
    "ELU": "keras.ELU",
    "ADD": "keras.Add",
    "SUB": "keras.Subtract",
    "MUL": "keras.Multiply",
    "MAXIMUM": "keras.Maximum",
    "MINIMUM": "keras.Minimum",
    "CONCATENATION": "keras.Concatenate",
    "PAD": "keras.ZeroPadding2D",
    "PADV2": "keras.ZeroPadding2D",
    "RESIZE_BILINEAR": "keras.UpSampling2D",
    "RESIZE_NEAREST_NEIGHBOR": "keras.UpSampling2D",
    "BATCH_TO_SPACE_ND": "keras.Identity",
    "SPACE_TO_BATCH_ND": "keras.Identity",
    "QUANTIZE": "keras.Identity",
    "DEQUANTIZE": "keras.Identity",
}
"""
> [!AML-DOC-UNIT]
TensorFlow Lite operations mapped to the closest catalog layer.

The mapping is approximate by nature: the converter fuses, lowers and quantises, so
one Keras layer may become several operations and several may become one. Anything
absent from this table is still imported, carrying its operation name, which the
validator then reports as an unknown type rather than silently dropping it.
"""

_TFLITE_PLUMBING = frozenset({"SHAPE", "STRIDED_SLICE", "PACK", "DELEGATE"})
"""
> [!AML-DOC-UNIT]
Operations that are an artefact of conversion rather than part of the architecture.

`DELEGATE` entries are fused kernels the runtime substituted, not operations in the
model. `SHAPE`/`STRIDED_SLICE`/`PACK` is how a flatten over a dynamic batch is
lowered. Keeping them would bury the architecture in plumbing the author never wrote.
"""


def import_keras_archive(
    data: bytes, registry: Registry
) -> tuple[GraphIR | None, list[Diagnostic]]:
    """
    > [!AML-DOC-UNIT]
    Read a `.keras` archive into an editable graph.
    @param data     the archive contents
    @param registry the layer catalog
    @returns (the graph, diagnostics); the graph is None when the file is not a
             `.keras` archive
    @raises nothing; a wrong or damaged file comes back as a diagnostic
    @sideEffects none
    @context A `.keras` file is a zip holding `config.json` — the same document
             `to_json()` produces — alongside the weights. Only the configuration is
             read: this tool edits architectures, and importing trained weights into
             a graph the user is about to change would promise more than it keeps.
    """
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except (zipfile.BadZipFile, OSError):
        return None, [
            Diagnostic(
                severity="error",
                code=Codes.IMPORT_NOT_A_KERAS_FILE,
                message=(
                    "This is not a .keras file. A .keras model is a zip archive; this "
                    "one could not be opened as one."
                ),
            )
        ]

    if KERAS_CONFIG_ENTRY not in archive.namelist():
        return None, [
            Diagnostic(
                severity="error",
                code=Codes.IMPORT_NOT_A_KERAS_FILE,
                message=(
                    f"This archive has no {KERAS_CONFIG_ENTRY}, so it holds no model "
                    f"definition. Saved models from Keras 3 always contain one."
                ),
            )
        ]

    graph, diagnostics = import_keras_json(
        archive.read(KERAS_CONFIG_ENTRY).decode("utf-8", errors="replace"), registry
    )
    if graph is not None:
        graph.meta["imported_from"] = "keras_archive"
        diagnostics.append(
            Diagnostic(
                severity="info",
                code=Codes.IMPORT_LOSSY,
                message=(
                    "The architecture was imported. Trained weights in the file were "
                    "not: this edits architectures, and weights would not survive the "
                    "first change you make."
                ),
            )
        )
    return graph, diagnostics


def _tflite_graph(data: bytes):
    """
    > [!AML-DOC-UNIT]
    Read a TensorFlow Lite flatbuffer's operations and tensors.
    @param data the flatbuffer contents
    @returns (operations, tensor details, input indices, output indices)
    @raises ValueError when the bytes are not a readable TFLite model
    @sideEffects allocates the interpreter's tensors
    @context Uses `tf.lite.Interpreter`, whose deprecation warning is suppressed here
             because nothing in this process can act on it. The replacement lives in
             a separate `ai_edge_litert` package, which this tool does not depend on.
    """
    import tensorflow as tf

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            interpreter = tf.lite.Interpreter(model_content=data)
            interpreter.allocate_tensors()
        except Exception as exc:
            raise ValueError(str(exc)) from exc

    if not hasattr(interpreter, "_get_ops_details"):
        raise ValueError(
            "this build of TensorFlow cannot list a .tflite model's operations"
        )

    return (
        interpreter._get_ops_details(),
        interpreter.get_tensor_details(),
        [detail["index"] for detail in interpreter.get_input_details()],
        [detail["index"] for detail in interpreter.get_output_details()],
    )


def import_tflite(
    data: bytes, registry: Registry
) -> tuple[GraphIR | None, list[Diagnostic]]:
    """
    > [!AML-DOC-UNIT]
    Read a TensorFlow Lite model into an editable graph.
    @param data     the flatbuffer contents
    @param registry the layer catalog
    @returns (the graph, diagnostics); the graph is None when the file cannot be read
    @raises nothing; an unreadable file comes back as a diagnostic
    @sideEffects loads the model into a TFLite interpreter
    @context What comes back is the converted graph, not the original model. The
             converter fuses activations into convolutions, lowers reshapes into
             several operations and may quantise, and the settings that distinguish
             one convolution from another — stride, padding, dilation — live in the
             flatbuffer's builtin options, which the interpreter does not expose.
             Topology and shapes are faithful; layer settings are defaults and will
             usually need correcting. The diagnostics say so rather than letting the
             result look like a round trip.
    """
    try:
        operations, tensors, input_indices, output_indices = _tflite_graph(data)
    except ValueError as exc:
        return None, [
            Diagnostic(
                severity="error",
                code=Codes.IMPORT_MALFORMED,
                message=f"This is not a readable .tflite model: {exc}",
            )
        ]

    by_index = {int(detail["index"]): detail for detail in tensors}
    real_ops = [
        operation
        for operation in operations
        if operation["op_name"] not in _TFLITE_PLUMBING
    ]

    # A tensor is data flowing through the graph only if an operation produces it or
    # it is a model input. Everything else is a stored weight.
    produced_by: dict[int, str] = {}
    nodes: list[Node] = []
    edges: list[Edge] = []
    unmapped: set[str] = set()
    skipped = len(operations) - len(real_ops)

    for order, index in enumerate(input_indices):
        detail = by_index.get(int(index))
        shape = [int(dim) for dim in detail["shape"]] if detail is not None else []
        name = f"input_{order}"
        nodes.append(
            Node(
                id=name,
                type="keras.Input",
                name=name,
                params={"shape": shape[1:], "dtype": "float32"},
                position=Position(x=240.0, y=80.0),
            )
        )
        produced_by[int(index)] = name

    for position, operation in enumerate(real_ops):
        op_name = str(operation["op_name"])
        spec_id = TFLITE_OP_MAP.get(op_name)
        if spec_id is None:
            unmapped.add(op_name)
            spec_id = f"tflite.{op_name}"

        node_id = f"{op_name.lower()}_{position}"
        outputs = [int(index) for index in operation["outputs"]]
        output_detail = by_index.get(outputs[0]) if outputs else None
        output_shape = (
            [int(dim) for dim in output_detail["shape"]] if output_detail is not None else []
        )

        nodes.append(
            Node(
                id=node_id,
                type=spec_id,
                name=f"{op_name.lower()} {position}",
                params=_tflite_params(spec_id, operation, by_index, output_shape),
                position=Position(x=240.0, y=190.0 + position * 110.0),
                notes=f"tflite op {op_name}; output {output_shape}",
            )
        )

        spec = registry.get(spec_id)
        port = spec.inputs[0].name if spec and spec.inputs else "input"
        incoming = [
            int(index) for index in operation["inputs"] if int(index) in produced_by
        ]
        for order, tensor_index in enumerate(incoming):
            source = produced_by[tensor_index]
            edges.append(
                Edge(
                    id=f"{source}->{node_id}:{order}",
                    source=source,
                    target=node_id,
                    target_port=port,
                    order=order,
                )
            )

        for index in outputs:
            produced_by[index] = node_id

    diagnostics: list[Diagnostic] = [
        Diagnostic(
            severity="warning",
            code=Codes.IMPORT_LOSSY,
            message=(
                "A .tflite file holds the converted graph, not the original model. The "
                "topology and the shapes are faithful; strides, padding and pool sizes "
                "live in the file's binary options and were left at their defaults, so "
                "expect to correct them before this compiles."
            ),
        )
    ]
    if skipped:
        diagnostics.append(
            Diagnostic(
                severity="info",
                code=Codes.IMPORT_LOSSY,
                message=(
                    f"{skipped} conversion artefact(s) were left out: fused kernels and "
                    f"the shape plumbing a flatten is lowered into, which the author "
                    f"never wrote."
                ),
            )
        )
    if unmapped:
        diagnostics.append(
            Diagnostic(
                severity="warning",
                code=Codes.IMPORT_UNKNOWN_LAYER,
                message=(
                    f"{len(unmapped)} operation(s) have no equivalent layer and were "
                    f"kept under their own name: {', '.join(sorted(unmapped))}."
                ),
            )
        )

    return (
        GraphIR(
            name="Imported TFLite model",
            nodes=nodes,
            edges=edges,
            meta={"imported_from": "tflite"},
        ),
        diagnostics,
    )


def _tflite_params(
    spec_id: str, operation: dict, by_index: dict, output_shape: list[int]
) -> dict:
    """
    > [!AML-DOC-UNIT]
    Recover what can be recovered of an operation's settings.
    @param spec_id      the catalog layer the operation maps to
    @param operation    the operation's details
    @param by_index     tensor details keyed by index
    @param output_shape shape of the operation's first output
    @returns constructor parameters, as far as the shapes reveal them
    @sideEffects none
    @context Only what the shapes imply: the output width gives a convolution its
             filter count and a dense layer its units, and a weight tensor's shape
             gives the kernel size. Stride and padding are not in reach, so they are
             left unset rather than guessed.
    """
    params: dict = {}
    if not output_shape:
        return params

    if spec_id in ("keras.Conv2D", "keras.Conv2DTranspose"):
        params["filters"] = int(output_shape[-1])
        for index in operation["inputs"]:
            detail = by_index.get(int(index))
            if detail is None:
                continue
            shape = [int(dim) for dim in detail["shape"]]
            # Convolution weights are stored as (out, kh, kw, in).
            if len(shape) == 4 and shape[0] == output_shape[-1]:
                params["kernel_size"] = [shape[1], shape[2]]
                break

    elif spec_id == "keras.DepthwiseConv2D":
        for index in operation["inputs"]:
            detail = by_index.get(int(index))
            if detail is not None and len(detail["shape"]) == 4:
                shape = [int(dim) for dim in detail["shape"]]
                params["kernel_size"] = [shape[1], shape[2]]
                break

    elif spec_id == "keras.Dense":
        params["units"] = int(output_shape[-1])

    elif spec_id == "keras.Reshape":
        params["target_shape"] = [int(dim) for dim in output_shape[1:]]

    return params


def import_model(
    data: bytes, filename: str, registry: Registry
) -> tuple[GraphIR | None, list[Diagnostic]]:
    """
    > [!AML-DOC-UNIT]
    Read a saved model into an editable graph, choosing the reader by file type.
    @param data     the file contents
    @param filename the original name, which selects the reader
    @param registry the layer catalog
    @returns (the graph, diagnostics)
    @raises nothing; every failure is a diagnostic
    @sideEffects may load TensorFlow Lite
    @context Dispatches on the extension, then falls back to sniffing the content, so
             a correct file with the wrong name still opens. The three formats offer
             different fidelity, and each reader says what its own gives.
    """
    lowered = filename.lower()

    if lowered.endswith(".tflite"):
        return import_tflite(data, registry)
    if lowered.endswith(".keras"):
        return import_keras_archive(data, registry)
    if lowered.endswith(".json"):
        return import_keras_json(data.decode("utf-8", errors="replace"), registry)

    # No usable extension: recognise the file by what it is.
    if data[:2] == b"PK":
        return import_keras_archive(data, registry)
    if data[4:8] == b"TFL3":
        return import_tflite(data, registry)
    try:
        return import_keras_json(data.decode("utf-8"), registry)
    except UnicodeDecodeError:
        return None, [
            Diagnostic(
                severity="error",
                code=Codes.IMPORT_MALFORMED,
                message=(
                    "This file is not a model this tool can read. It accepts .keras "
                    "archives, .tflite models, and the JSON from Model.to_json()."
                ),
            )
        ]
