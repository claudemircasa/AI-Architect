"""
> [!AML-DOC-FILE]
@file       viz/tensors.py
@description Captures the real activations a model produces for one sample, and
             turns each into something the editor can draw.
@module     nnarch.viz.tensors
@exports    RenderHint, Activation, ActivationResult, capture_activations
@created    2026-10-01
@context    This is the feature the tool exists for: not a diagram of an
            architecture, but the actual numbers each layer produced. Nothing here
            simulates or approximates — a probe model is built from the compiled
            model and run, and what comes back is what came out of those layers.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from nnarch.catalog import Diagnostic, Registry
from nnarch.data import DatasetAdapter, DatasetError, Modality
from nnarch.ir.compiler import compile_graph
from nnarch.ir.schema import GraphIR

from .encode import MAX_CHANNELS, downsample_1d, normalise, summarise, tile_grid, to_png


class RenderHint(str, Enum):
    """
    > [!AML-DOC-UNIT]
    How the editor should draw one layer's output. Chosen from the tensor's rank and
    the layer's declared modality, because the same numbers mean different things in
    different places: a (batch, time, channels) tensor is a spectrogram in an audio
    model and a token sequence in a language one.
    """

    FEATURE_MAPS = "feature_maps"
    SEQUENCE_HEATMAP = "sequence_heatmap"
    ATTENTION_MATRIX = "attention_matrix"
    VECTOR_BARS = "vector_bars"
    PROBABILITIES = "probabilities"
    WAVEFORM = "waveform"
    SPECTROGRAM = "spectrogram"
    VOLUME_SLICES = "volume_slices"
    SCALAR = "scalar"


class Activation(BaseModel):
    """
    > [!AML-DOC-UNIT]
    One layer's output, ready to draw.
    @param node_id     the node that produced it
    @param label       the layer's name, for the drawer's title
    @param shape       full tensor shape, batch axis included
    @param dtype       element type
    @param hint        how to draw it
    @param stats       min, max, mean, standard deviation and sparsity
    @param tiles       base64 PNGs, one per feature map or attention head
    @param series      a one-dimensional trace, for waveforms and bar charts
    @param labels      axis labels, such as decoded tokens or class names
    @param heatmap     a single base64 PNG, for sequences and spectrograms
    @param truncated   whether channels were left out to keep the payload small
    @param channel_count how many channels the layer really has
    """

    node_id: str
    label: str
    shape: list[int | None]
    dtype: str
    hint: RenderHint
    stats: dict[str, float] = Field(default_factory=dict)
    tiles: list[str] = Field(default_factory=list)
    series: list[float] = Field(default_factory=list)
    labels: list[str] = Field(default_factory=list)
    heatmap: str | None = None
    truncated: bool = False
    channel_count: int = 0


class ActivationResult(BaseModel):
    """
    > [!AML-DOC-UNIT]
    Everything a single preview run produced.
    @param activations  one entry per layer, in topological order
    @param input_preview how the input itself looked, so the user can see what went in
    @param diagnostics  problems encountered
    @param sample_label what the sample was, such as a class name or a filename
    """

    activations: list[Activation] = Field(default_factory=list)
    input_preview: Activation | None = None
    diagnostics: list[Diagnostic] = Field(default_factory=list)
    sample_label: str = ""


class Codes:
    """
    > [!AML-DOC-UNIT]
    Diagnostic identifiers produced while capturing activations. Stable contract.
    """

    GRAPH_INVALID = "activations_graph_invalid"
    NO_DATA = "activations_no_data"
    RUN_FAILED = "activations_run_failed"
    SHAPE_MISMATCH = "activations_shape_mismatch"
    ZEROED_INPUTS = "activations_zeroed_inputs"
    INPUT_TRANSFORMED = "activations_input_transformed"


def _looks_like_attention(values: np.ndarray) -> bool:
    """
    > [!AML-DOC-UNIT]
    Decide whether a rank-3 tensor holds attention weights rather than feature maps.
    @param values the activation for one sample, shaped (rows, columns, channels)
    @returns True when it behaves like attention
    @sideEffects none
    @context Shape alone cannot tell them apart: a 26x26x16 convolution output is
             just as square as an attention matrix over 26 tokens. What does
             distinguish them is that attention weights are a distribution — each
             row sums to one — which a feature map has no reason to do. Checking the
             property rather than the shape is what stopped every square convolution
             from being drawn as attention.
    """
    rows, columns, heads = values.shape
    if rows != columns or rows < 2 or heads > 32:
        return False
    sums = values.sum(axis=1)
    return bool(np.all(np.abs(sums - 1.0) < 1e-3)) and bool(np.all(values >= -1e-6))


def _choose_hint(values: np.ndarray, modality: Modality, is_output: bool) -> RenderHint:
    """
    > [!AML-DOC-UNIT]
    Decide how a tensor should be drawn.
    @param values    the activation for one sample, batch axis removed
    @param modality  the data kind the model is working with
    @param is_output whether this is the model's final layer
    @returns the render hint
    @sideEffects none
    """
    rank = values.ndim

    if rank == 0:
        return RenderHint.SCALAR
    if rank == 1:
        if is_output and values.size <= 1000 and abs(float(values.sum()) - 1.0) < 1e-3:
            return RenderHint.PROBABILITIES
        return RenderHint.VECTOR_BARS
    if rank == 2:
        # (heads, positions) is rare; (time, channels) is the common case.
        if modality is Modality.AUDIO:
            return RenderHint.WAVEFORM if values.shape[-1] == 1 else RenderHint.SPECTROGRAM
        return RenderHint.SEQUENCE_HEATMAP
    if rank == 3:
        if _looks_like_attention(values):
            return RenderHint.ATTENTION_MATRIX
        return RenderHint.FEATURE_MAPS
    if rank == 4:
        return RenderHint.VOLUME_SLICES
    return RenderHint.VECTOR_BARS


def _encode(
    node_id: str,
    label: str,
    tensor: np.ndarray,
    modality: Modality,
    is_output: bool,
    colormap: str,
    class_names: list[str],
) -> Activation:
    """
    > [!AML-DOC-UNIT]
    Turn one raw activation into a drawable record.
    @param node_id     the producing node
    @param label       the layer's name
    @param tensor      the activation including the batch axis
    @param modality    the data kind
    @param is_output   whether this is the final layer
    @param colormap    ramp to render with
    @param class_names labels for the output layer, when the dataset knows them
    @returns the Activation
    @sideEffects none
    """
    shape = [int(dim) for dim in tensor.shape]
    sample = tensor[0] if tensor.ndim > 0 and tensor.shape[0] >= 1 else tensor
    hint = _choose_hint(sample, modality, is_output)

    activation = Activation(
        node_id=node_id,
        label=label,
        shape=shape,
        dtype=str(tensor.dtype),
        hint=hint,
        stats=summarise(tensor),
        channel_count=int(sample.shape[-1]) if sample.ndim >= 1 else 0,
    )

    if hint is RenderHint.FEATURE_MAPS:
        activation.tiles = tile_grid(sample, colormap)
        activation.truncated = sample.shape[-1] > MAX_CHANNELS

    elif hint is RenderHint.ATTENTION_MATRIX:
        heads = min(sample.shape[-1], 8)
        activation.tiles = [
            to_png(normalise(sample[..., head]), colormap) for head in range(heads)
        ]
        activation.truncated = sample.shape[-1] > heads

    elif hint in (RenderHint.SEQUENCE_HEATMAP, RenderHint.SPECTROGRAM):
        activation.heatmap = to_png(normalise(sample.T), colormap)

    elif hint is RenderHint.WAVEFORM:
        activation.series = downsample_1d(sample.ravel())

    elif hint in (RenderHint.VECTOR_BARS, RenderHint.PROBABILITIES):
        activation.series = downsample_1d(sample.ravel(), limit=512)
        # Names are attached whenever the user's vocabulary is exactly as long as this
        # layer is wide, whether or not the values happen to sum to one. Requiring a
        # probability distribution meant a head that emits logits — or any softmax
        # applied in the loss rather than the layer — showed bare indices although its
        # names were sitting right there [E-042].
        if class_names and len(class_names) == sample.size:
            activation.labels = class_names[: len(activation.series)]
        elif hint is RenderHint.PROBABILITIES and class_names:
            activation.labels = class_names[: len(activation.series)]

    elif hint is RenderHint.VOLUME_SLICES:
        middle = sample.shape[0] // 2
        activation.tiles = tile_grid(sample[middle], colormap, limit=16)
        activation.truncated = True

    elif hint is RenderHint.SCALAR:
        activation.series = [float(np.asarray(sample).ravel()[0])]

    return activation



def _input_label(tensor: Any) -> str:
    """
    > [!AML-DOC-UNIT]
    A readable name for one of a model's inputs.
    @param tensor the input KerasTensor
    @returns the layer's name, without Keras's internal suffixes
    @sideEffects none
    """
    return str(getattr(tensor, "name", "input")).split("/")[0].split(":")[0]


def _zeros_like_input(tensor: Any, batch: int) -> Any:
    """
    > [!AML-DOC-UNIT]
    An all-zero array shaped for one of a model's inputs.
    @param tensor the input KerasTensor
    @param batch  how many examples
    @returns a numpy array of zeros
    @sideEffects none
    """
    import numpy as np

    dims = [1 if dim is None else int(dim) for dim in tensor.shape[1:]]
    return np.zeros((batch, *dims), dtype="float32")


def primary_input_index(model: Any, sample: Any = None) -> int:
    """
    > [!AML-DOC-UNIT]
    Which of a model's inputs the sample is meant for.
    @param model  the compiled model
    @param sample the data to feed, or None before any has been chosen
    @returns the index of the input to feed it to
    @sideEffects none
    @context An exact shape match decides it when there is one. Otherwise the widest
             input wins, because the sample is the thing being looked at — an image
             or a spectrogram — and the other inputs of a model like this are the
             scalars and states that steer it [E-036].
    """
    if sample is not None:
        wanted = tuple(int(d) for d in sample.shape[1:])
        for index, tensor in enumerate(model.inputs):
            dims = tuple(int(d) for d in tensor.shape[1:] if d is not None)
            if dims == wanted:
                return index
    sizes = []
    for tensor in model.inputs:
        dims = [int(d) for d in tensor.shape[1:] if d is not None]
        total = 1
        for dim in dims:
            total *= dim
        sizes.append((len(dims), total))
    return sizes.index(max(sizes))

def capture_activations(
    graph: GraphIR,
    registry: Registry,
    adapter: DatasetAdapter,
    *,
    node_ids: list[str] | None = None,
    colormap: str = "viridis",
    labels_by_size: dict[int, list[str]] | None = None,
    labels_by_node: dict[str, list[str]] | None = None,
) -> ActivationResult:
    """
    > [!AML-DOC-UNIT]
    Run one real sample through the model and collect what every layer produced.
    @param graph    the architecture
    @param registry the layer catalog
    @param adapter  where the sample comes from
    @param node_ids restrict capture to these nodes; None captures every layer
    @param colormap ramp to render with
    @param labels_by_size names for an output, keyed by how many of them there are,
                          so a model with several differently-sized outputs gets the
                          right vocabulary on each [E-042]
    @param labels_by_node names chosen for a particular layer by hand, which override
                          the automatic match — the user knows which vocabulary is
                          which when two are the same length and the file does not say
    @returns an ActivationResult holding one entry per layer, in topological order
    @raises nothing; a graph that will not compile or data that will not load comes
            back as diagnostics
    @sideEffects compiles the graph, builds a probe model and runs a forward pass
    @context The probe is `keras.Model(inputs, [every layer's output])` over the
             already-compiled model, so the numbers are exactly the ones the real
             model produces. Nothing is recomputed or approximated.
    """
    import keras

    compiled = compile_graph(graph, registry)
    if not compiled.ok:
        return ActivationResult(
            diagnostics=[
                Diagnostic(
                    severity="error",
                    code=Codes.GRAPH_INVALID,
                    message="The architecture has to be valid before data can flow "
                            "through it. Fix the errors first.",
                ),
                *compiled.diagnostics,
            ]
        )

    model = compiled.model
    wanted = set(node_ids) if node_ids else None
    ordered, _ = graph.topological_order()
    node_map = graph.node_map()

    layers_by_name = {layer.name: layer for layer in model.layers}
    from nnarch.naming import NamePool

    pool = NamePool()
    probe_nodes: list[tuple[str, str, Any]] = []
    for node_id in ordered:
        node = node_map[node_id]
        if node.disabled:
            continue
        name = pool.allocate(node.name, node_id)
        layer = layers_by_name.get(name)
        if layer is None:
            continue
        if wanted is None or node_id in wanted:
            probe_nodes.append((node_id, node.name or node_id, layer.output))

    if not probe_nodes:
        return ActivationResult(
            diagnostics=[
                Diagnostic(severity="info", code=Codes.NO_DATA,
                           message="There are no layers to show yet.")
            ]
        )

    try:
        sample, labels = adapter.sample(1)
    except DatasetError as exc:
        return ActivationResult(
            diagnostics=[
                Diagnostic(severity="error", code=Codes.NO_DATA, message=str(exc))
            ]
        )

    spec = adapter.spec()
    extra: list[Diagnostic] = []

    # What the data source had to do to the file to make it fit. A transform the user
    # did not ask for must be visible, or the picture is being presented as something
    # it is not [E-041].
    from nnarch.data.adapters import take_upload_note

    note = take_upload_note()
    if note:
        extra.append(Diagnostic(severity="info", code=Codes.INPUT_TRANSFORMED, message=note))
    primary = primary_input_index(model, sample)
    expected = [dim for dim in model.inputs[primary].shape[1:]]
    actual = list(sample.shape[1:])
    if any(e is not None and int(e) != int(a) for e, a in zip(expected, actual)):
        return ActivationResult(
            diagnostics=[
                Diagnostic(
                    severity="error",
                    code=Codes.SHAPE_MISMATCH,
                    message=(
                        f"The data is shaped {actual} but this model's input is "
                        f"{[int(d) if d else 'any' for d in expected]}. Change the "
                        f"Input layer, or pick a source that matches it."
                    ),
                )
            ]
        )

    feed: Any = sample
    if len(model.inputs) > 1:
        # A model can take more than one input — a recurrent generator takes its own
        # state back, and this one takes six. The sample goes to the input it fits,
        # and the rest start at zero, which for a state input is not a stand-in but
        # the correct starting condition [E-036]. Said out loud, because a number
        # the user did not supply should never look like one they did.
        feed = [
            sample if index == primary else _zeros_like_input(tensor, len(sample))
            for index, tensor in enumerate(model.inputs)
        ]
        others = [
            _input_label(tensor)
            for index, tensor in enumerate(model.inputs)
            if index != primary
        ]
        extra.append(Diagnostic(
            severity="info", code=Codes.ZEROED_INPUTS,
            message=(
                f"This model takes {len(model.inputs)} inputs. The sample was fed to "
                f"{_input_label(model.inputs[primary])}; the others started at zero: "
                f"{', '.join(others)}."
            ),
        ))

    probe = keras.Model(model.inputs, [output for _, _, output in probe_nodes])
    try:
        outputs = probe.predict(feed, verbose=0)
    except Exception as exc:
        return ActivationResult(
            diagnostics=[
                Diagnostic(
                    severity="error", code=Codes.RUN_FAILED,
                    message=f"The model could not run on this sample: {exc}",
                )
            ]
        )

    if not isinstance(outputs, list):
        outputs = [outputs]

    output_ids = set(graph.resolved_outputs())

    def names_for(node_id: str, values: Any) -> list[str]:
        """
        > [!AML-DOC-UNIT]
        The vocabulary belonging to one layer's output.
        @param node_id the layer
        @param values  its activations
        @returns the names, or the dataset's own when nothing matches
        @sideEffects none
        @context A choice made by hand wins, then a vocabulary exactly as long as the
                 layer is wide. Size is the only thing a vocabulary and a layer share
                 and it carries most cases, but it cannot separate two vocabularies of
                 the same length — which is precisely when somebody has to say which
                 is which, and the reason the manual route exists at all [E-042].
        """
        chosen = (labels_by_node or {}).get(node_id)
        if chosen is not None:
            return chosen
        if not labels_by_size:
            return spec.class_names
        array = np.asarray(values)
        width = int(array.shape[-1]) if array.ndim else 0
        return labels_by_size.get(width) or spec.class_names

    activations = [
        _encode(
            node_id,
            label,
            np.asarray(values),
            spec.modality,
            node_id in output_ids,
            colormap,
            names_for(node_id, values),
        )
        for (node_id, label, _), values in zip(probe_nodes, outputs)
    ]

    sample_label = spec.label
    if labels is not None and len(labels) and spec.class_names:
        index = int(np.asarray(labels).ravel()[0])
        if 0 <= index < len(spec.class_names):
            sample_label = f"{spec.label} — {spec.class_names[index]}"

    return ActivationResult(
        activations=activations,
        input_preview=_encode(
            "__input__", "Input sample", sample, spec.modality, False, colormap,
            spec.class_names,
        ),
        sample_label=sample_label,
        diagnostics=extra,
    )
