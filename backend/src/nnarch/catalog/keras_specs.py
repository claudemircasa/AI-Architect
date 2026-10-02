"""
> [!AML-DOC-FILE]
@file       catalog/keras_specs.py
@description Declarative table covering the whole Keras 3 layer taxonomy. Each row
             names a layer, its palette category, its tensor ranks and its port
             arity; the editable parameters are derived by introspection.
@module     nnarch.catalog.keras_specs
@exports    build_keras_specs
@created    2026-09-30
@context    Verified against keras 3.15.1 / tensorflow 2.21.0 [amm: C]. Layer ids
            are persisted in saved projects: PRESERVE, never rename [amm: E.2].
"""

from __future__ import annotations

from typing import Any, Sequence

from .introspect import derive_params
from .spec import CallStyle, Category, LayerSpec, Modality, ParamSpec, ParamType, PortSpec

_IMAGE = [Modality.IMAGE]
_SEQ = [Modality.SEQUENCE, Modality.TEXT, Modality.AUDIO]
_VOL = [Modality.VOLUMETRIC]
_ANY = [Modality.ANY]

_KERAS_DOCS = "https://keras.io/api/layers"


_STARTER_DEFAULTS: dict[str, dict[str, Any]] = {
    "Dropout": {"rate": 0.2},
    "SpatialDropout1D": {"rate": 0.2},
    "SpatialDropout2D": {"rate": 0.2},
    "SpatialDropout3D": {"rate": 0.2},
    "GaussianDropout": {"rate": 0.2},
    "AlphaDropout": {"rate": 0.2},
    "GaussianNoise": {"stddev": 0.1},
    "AveragePooling1D": {"pool_size": 2},
    "AveragePooling2D": {"pool_size": (2, 2)},
    "AveragePooling3D": {"pool_size": (2, 2, 2)},
    "AdaptiveAveragePooling1D": {"output_size": 16},
    "AdaptiveAveragePooling2D": {"output_size": (7, 7)},
    "AdaptiveAveragePooling3D": {"output_size": (4, 4, 4)},
    "AdaptiveMaxPooling1D": {"output_size": 16},
    "AdaptiveMaxPooling2D": {"output_size": (7, 7)},
    "AdaptiveMaxPooling3D": {"output_size": (4, 4, 4)},
    "Dot": {"axes": -1},
    "GroupQueryAttention": {"head_dim": 64, "num_query_heads": 8, "num_key_value_heads": 2},
    "MultiHeadAttention": {"num_heads": 8, "key_dim": 64},
    "Dense": {"units": 64},
    "Conv1D": {"filters": 32, "kernel_size": 3},
    "Conv2D": {"filters": 32, "kernel_size": (3, 3)},
    "Conv3D": {"filters": 32, "kernel_size": (3, 3, 3)},
    "Conv1DTranspose": {"filters": 32, "kernel_size": 3},
    "Conv2DTranspose": {"filters": 32, "kernel_size": (3, 3)},
    "Conv3DTranspose": {"filters": 32, "kernel_size": (3, 3, 3)},
    "SeparableConv1D": {"filters": 32, "kernel_size": 3},
    "SeparableConv2D": {"filters": 32, "kernel_size": (3, 3)},
    "DepthwiseConv1D": {"kernel_size": 3},
    "DepthwiseConv2D": {"kernel_size": (3, 3)},
    "ConvLSTM1D": {"filters": 32, "kernel_size": 3},
    "ConvLSTM2D": {"filters": 32, "kernel_size": (3, 3)},
    "ConvLSTM3D": {"filters": 32, "kernel_size": (3, 3, 3)},
    "LSTM": {"units": 64},
    "GRU": {"units": 64},
    "SimpleRNN": {"units": 64},
    "Embedding": {"input_dim": 10000, "output_dim": 128},
    "EinsumDense": {"equation": "abc,cd->abd", "output_shape": (None, 64)},
    "Reshape": {"target_shape": (7, 7, 16)},
    "RepeatVector": {"n": 8},
    "Permute": {"dims": (2, 1)},
    "CenterCrop": {"height": 224, "width": 224},
    "RandomCrop": {"height": 224, "width": 224},
    "Resizing": {"height": 224, "width": 224},
    "RandomBrightness": {"factor": 0.2},
    "RandomContrast": {"factor": 0.2},
    "RandomRotation": {"factor": 0.1},
    "RandomSharpness": {"factor": 0.2},
    "RandomZoom": {"height_factor": 0.2},
    "RandomTranslation": {"height_factor": 0.1, "width_factor": 0.1},
    "Rescaling": {"scale": 1.0 / 255.0},
    "Hashing": {"num_bins": 1024},
    "HashedCrossing": {"num_bins": 1024},
    "CategoryEncoding": {"num_tokens": 10},
    "Discretization": {"num_bins": 10},
    "Lambda": {"function": "x"},
}
"""
> [!AML-DOC-UNIT]
Starter values for constructor arguments Keras leaves mandatory. A layer dropped
on the canvas must be immediately valid and runnable, so every required argument
without a Keras default gets a sensible opening value here. The parameter stays
flagged `required` so the property panel still highlights it as the user's to set.
@context Derived from the instantiation gate in task 02: 33 of 112 specs could not
         be constructed from their own defaults before this table existed.
"""


def _spec(
    name: str,
    label: str,
    category: Category,
    *,
    doc_section: str,
    rank_in: int | None = None,
    rank_out: int | None = None,
    min_inputs: int = 1,
    max_inputs: int | None = 1,
    modalities: Sequence[Modality] = _ANY,
    description: str = "",
    tags: Sequence[str] = (),
    overrides: dict[str, dict[str, Any] | None] | None = None,
    inputs: Sequence[PortSpec] | None = None,
    outputs: Sequence[PortSpec] | None = None,
    call_style: CallStyle = CallStyle.SINGLE,
) -> LayerSpec:
    """
    > [!AML-DOC-UNIT]
    Build one LayerSpec for a stock Keras layer, deriving its parameters from the
    constructor signature.
    @param name        Keras class name under `keras.layers`
    @param label       palette display name
    @param category    palette grouping
    @param doc_section keras.io section slug used to build the documentation link
    @param rank_in     required input rank including batch, or None when agnostic
    @param rank_out    produced rank, or None when derived from the input
    @param min_inputs  minimum connected inputs for a valid graph
    @param max_inputs  maximum connected inputs; None means unbounded
    @param modalities  data kinds this layer suits
    @param description palette tooltip text
    @param tags        extra search keywords
    @param overrides   per-parameter field overrides, merged on top of
                       `_STARTER_DEFAULTS` before reaching `derive_params`
    @param inputs      explicit ordered input ports; defaults to a single "input"
    @param outputs     explicit ordered output ports; defaults to a single "output"
    @param call_style  tensor-passing convention used by the compiler and codegen
    @returns the assembled LayerSpec
    @raises ImportError when `keras.layers.<name>` does not exist
    """
    merged: dict[str, dict[str, Any] | None] = {
        param: {"default": value} for param, value in _STARTER_DEFAULTS.get(name, {}).items()
    }
    for param, fields in (overrides or {}).items():
        if fields is None:
            merged[param] = None
        else:
            merged[param] = {**(merged.get(param) or {}), **fields}

    return LayerSpec(
        id=f"keras.{name}",
        label=label,
        category=category,
        keras_path=f"keras.layers.{name}",
        params=derive_params(f"keras.layers.{name}", rank_in=rank_in, overrides=merged),
        inputs=list(inputs) if inputs is not None else [PortSpec(name="input", label="Input", rank=rank_in)],
        outputs=list(outputs) if outputs is not None else [PortSpec(name="output", label="Output", rank=rank_out)],
        min_inputs=min_inputs,
        max_inputs=max_inputs,
        rank_in=rank_in,
        rank_out=rank_out,
        modalities=list(modalities),
        doc_url=f"{_KERAS_DOCS}/{doc_section}",
        description=description,
        tags=[name.lower(), *tags],
        call_style=call_style,
    )


def _core() -> list[LayerSpec]:
    """
    > [!AML-DOC-UNIT]
    Core layers: the graph entry point and the fundamental dense/embedding blocks.
    @returns LayerSpec list for the Core palette category
    """
    section = "core_layers"
    return [
        LayerSpec(
            id="keras.Input",
            label="Input",
            category=Category.CORE,
            keras_path="keras.layers.Input",
            params=[
                ParamSpec(name="shape", type=ParamType.SHAPE, default=[28, 28, 1], required=True,
                          help="Shape of one sample, excluding the batch dimension."),
                ParamSpec(name="dtype", type=ParamType.DTYPE, default="float32",
                          help="Element type of the incoming tensor."),
                ParamSpec(name="batch_size", type=ParamType.INT, default=None, minimum=1,
                          help="Fixed batch size; leave empty to keep it dynamic.", advanced=True),
                ParamSpec(name="sparse", type=ParamType.BOOL, default=False,
                          help="Accept a sparse tensor.", advanced=True),
            ],
            inputs=[],
            outputs=[PortSpec(name="output", label="Tensor")],
            min_inputs=0,
            max_inputs=0,
            modalities=_ANY,
            doc_url=f"{_KERAS_DOCS}/{section}/input",
            description="Entry point of the model. Declares the shape and dtype of the data.",
            tags=["input", "placeholder", "entry", "source"],
            call_style=CallStyle.SOURCE,
        ),
        _spec("Dense", "Dense", Category.CORE, doc_section=f"{section}/dense",
              modalities=_ANY, tags=["fully connected", "linear", "mlp", "neuron"],
              description="Fully connected layer: every input unit connects to every output unit."),
        _spec("EinsumDense", "Einsum Dense", Category.CORE, doc_section=f"{section}/einsum_dense",
              tags=["einsum", "tensor contraction", "projection"],
              description="Arbitrary dense contraction expressed as an Einstein-summation equation."),
        _spec("Activation", "Activation", Category.CORE, doc_section=f"{section}/activation",
              overrides={"activation": {"type": ParamType.ACTIVATION, "default": "relu",
                                        "required": True,
                                        "help": "Non-linearity applied element-wise."}},
              tags=["nonlinearity"],
              description="Applies an activation function element-wise, shape unchanged."),
        _spec("Embedding", "Embedding", Category.CORE, doc_section=f"{section}/embedding",
              rank_in=2, rank_out=3, modalities=[Modality.TEXT, Modality.SEQUENCE],
              tags=["lookup", "token", "vocabulary", "nlp"],
              description="Maps integer token ids to dense trainable vectors."),
        _spec("Masking", "Masking", Category.CORE, doc_section=f"{section}/masking",
              modalities=_SEQ, tags=["padding", "mask"],
              description="Marks timesteps equal to a sentinel value so later layers skip them."),
        _spec("Lambda", "Lambda", Category.CORE, doc_section=f"{section}/lambda",
              tags=["custom", "expression"],
              description="Wraps an arbitrary expression over the input tensor `x`."),
        _spec("Identity", "Identity", Category.CORE, doc_section=f"{section}/identity",
              tags=["passthrough", "noop"],
              description="Passes the tensor through unchanged. Useful as a named waypoint."),
    ]


def _convolution() -> list[LayerSpec]:
    """
    > [!AML-DOC-UNIT]
    Convolution family across 1D, 2D and 3D, including separable, depthwise and
    transposed variants.
    @returns LayerSpec list for the Convolution palette category
    """
    section = "convolution_layers"
    ranks = {1: (3, _SEQ), 2: (4, _IMAGE), 3: (5, _VOL)}
    specs: list[LayerSpec] = []
    for dim, (rank, modality) in ranks.items():
        specs.append(_spec(
            f"Conv{dim}D", f"Conv {dim}D", Category.CONVOLUTION,
            doc_section=f"{section}/convolution{dim}d", rank_in=rank, rank_out=rank,
            modalities=modality, tags=["conv", "cnn", "filter", "kernel"],
            description=f"Standard {dim}D convolution: slides learnable kernels over the input.",
        ))
        specs.append(_spec(
            f"Conv{dim}DTranspose", f"Conv {dim}D Transpose", Category.CONVOLUTION,
            doc_section=f"{section}/convolution{dim}d_transpose", rank_in=rank, rank_out=rank,
            modalities=modality, tags=["deconv", "upsample", "decoder", "generator"],
            description=f"Transposed {dim}D convolution: learns to upsample the spatial size.",
        ))
    for dim in (1, 2):
        rank, modality = ranks[dim]
        specs.append(_spec(
            f"SeparableConv{dim}D", f"Separable Conv {dim}D", Category.CONVOLUTION,
            doc_section=f"{section}/separable_convolution{dim}d", rank_in=rank, rank_out=rank,
            modalities=modality, tags=["separable", "efficient", "mobilenet", "xception"],
            description=f"Depthwise then pointwise {dim}D convolution: far fewer parameters.",
        ))
        specs.append(_spec(
            f"DepthwiseConv{dim}D", f"Depthwise Conv {dim}D", Category.CONVOLUTION,
            doc_section=f"{section}/depthwise_convolution{dim}d", rank_in=rank, rank_out=rank,
            modalities=modality, tags=["depthwise", "efficient", "mobilenet"],
            description=f"Convolves each input channel independently, without mixing channels.",
        ))
    return specs


def _pooling() -> list[LayerSpec]:
    """
    > [!AML-DOC-UNIT]
    Pooling family: windowed, global and adaptive, across 1D/2D/3D.
    @returns LayerSpec list for the Pooling palette category
    """
    section = "pooling_layers"
    ranks = {1: (3, _SEQ), 2: (4, _IMAGE), 3: (5, _VOL)}
    specs: list[LayerSpec] = []
    for dim, (rank, modality) in ranks.items():
        for kind, word in (("Max", "strongest"), ("Average", "mean")):
            specs.append(_spec(
                f"{kind}Pooling{dim}D", f"{kind} Pooling {dim}D", Category.POOLING,
                doc_section=f"{section}/{kind.lower()}_pooling{dim}d",
                rank_in=rank, rank_out=rank, modalities=modality,
                tags=["pool", "downsample", "subsample"],
                description=f"Downsamples by taking the {word} value in each window.",
            ))
            specs.append(_spec(
                f"Global{kind}Pooling{dim}D", f"Global {kind} Pooling {dim}D", Category.POOLING,
                doc_section=f"{section}/global_{kind.lower()}_pooling{dim}d",
                rank_in=rank, rank_out=2, modalities=modality,
                tags=["pool", "global", "flatten", "head"],
                description=f"Collapses all spatial axes to one {word} value per channel.",
            ))
            specs.append(_spec(
                f"Adaptive{kind}Pooling{dim}D", f"Adaptive {kind} Pooling {dim}D", Category.POOLING,
                doc_section=f"{section}/adaptive_{kind.lower()}_pooling{dim}d",
                rank_in=rank, rank_out=rank, modalities=modality,
                tags=["pool", "adaptive", "fixed output"],
                description=f"Pools to an exact output size regardless of the input size.",
            ))
    return specs


def _recurrent() -> list[LayerSpec]:
    """
    > [!AML-DOC-UNIT]
    Recurrent layers plus the two structural wrappers that take a child layer.
    @returns LayerSpec list for the Recurrent palette category
    @context Wrappers expose a LAYER_REF parameter; the compiler resolves it to a
             nested node [amm: E.3].
    """
    section = "recurrent_layers"
    wrapped = {"layer": {"type": ParamType.LAYER_REF, "required": True,
                         "help": "The layer to wrap. Pick a node from the canvas."}}
    specs = [
        _spec("LSTM", "LSTM", Category.RECURRENT, doc_section=f"{section}/lstm",
              rank_in=3, modalities=_SEQ, tags=["rnn", "gate", "memory", "long short-term"],
              # LSTM carries two state tensors, the hidden state and the cell state,
              # so it accepts one more input than the single-state layers do.
              min_inputs=1, max_inputs=3, call_style=CallStyle.RECURRENT,
              inputs=_recurrent_ports(),
              description="Long short-term memory: gated recurrence that resists vanishing gradients."),
        _spec("GRU", "GRU", Category.RECURRENT, doc_section=f"{section}/gru",
              rank_in=3, modalities=_SEQ, tags=["rnn", "gate", "gated recurrent"],
              min_inputs=1, max_inputs=2, call_style=CallStyle.RECURRENT,
              inputs=_recurrent_ports(),
              description="Gated recurrent unit: like LSTM with fewer gates and parameters."),
        _spec("SimpleRNN", "Simple RNN", Category.RECURRENT, doc_section=f"{section}/simple_rnn",
              rank_in=3, modalities=_SEQ, tags=["rnn", "vanilla", "elman"],
              min_inputs=1, max_inputs=2, call_style=CallStyle.RECURRENT,
              inputs=_recurrent_ports(),
              description="Plain recurrent layer where the output feeds back as state."),
        _spec("Bidirectional", "Bidirectional", Category.RECURRENT,
              doc_section=f"{section}/bidirectional", rank_in=3, modalities=_SEQ,
              overrides=wrapped, tags=["wrapper", "forward backward", "birnn"],
              description="Runs a recurrent layer forwards and backwards, then merges both passes."),
        _spec("TimeDistributed", "Time Distributed", Category.RECURRENT,
              doc_section=f"{section}/time_distributed", modalities=_SEQ,
              overrides=wrapped, tags=["wrapper", "per timestep"],
              description="Applies a layer independently to every timestep of a sequence."),
    ]
    for dim, rank in ((1, 4), (2, 5), (3, 6)):
        specs.append(_spec(
            f"ConvLSTM{dim}D", f"Conv LSTM {dim}D", Category.RECURRENT,
            doc_section=f"{section}/conv_lstm{dim}d", rank_in=rank,
            modalities=[Modality.IMAGE, Modality.SEQUENCE],
            tags=["rnn", "conv", "spatiotemporal", "video"],
            description=f"LSTM whose gates are {dim}D convolutions: models space and time jointly.",
        ))
    return specs


def _normalization() -> list[LayerSpec]:
    """
    > [!AML-DOC-UNIT]
    Normalization layers, including RMSNorm as used by modern transformers.
    @returns LayerSpec list for the Normalization palette category
    """
    section = "normalization_layers"
    return [
        _spec("BatchNormalization", "Batch Normalization", Category.NORMALIZATION,
              doc_section=f"{section}/batch_normalization",
              tags=["batchnorm", "bn", "stabilize"],
              description="Normalizes each feature using statistics across the batch."),
        _spec("LayerNormalization", "Layer Normalization", Category.NORMALIZATION,
              doc_section=f"{section}/layer_normalization",
              tags=["layernorm", "ln", "transformer"],
              description="Normalizes each sample across its features. The transformer default."),
        _spec("RMSNormalization", "RMS Normalization", Category.NORMALIZATION,
              doc_section=f"{section}/rms_normalization",
              tags=["rmsnorm", "llama", "transformer", "efficient"],
              description="Scale-only normalization by root-mean-square. Cheaper than LayerNorm."),
        _spec("GroupNormalization", "Group Normalization", Category.NORMALIZATION,
              doc_section=f"{section}/group_normalization",
              tags=["groupnorm", "gn", "small batch"],
              description="Normalizes within channel groups, so it works at tiny batch sizes."),
        _spec("UnitNormalization", "Unit Normalization", Category.NORMALIZATION,
              doc_section=f"{section}/unit_normalization",
              tags=["l2 normalize", "unit norm"],
              description="Rescales each sample to unit L2 norm along the chosen axis."),
        _spec("SpectralNormalization", "Spectral Normalization", Category.NORMALIZATION,
              doc_section=f"{section}/spectral_normalization",
              overrides={"layer": {"type": ParamType.LAYER_REF, "required": True,
                                   "help": "The layer whose kernel gets spectral normalization."}},
              tags=["gan", "lipschitz", "wrapper", "stability"],
              description="Constrains a layer's spectral norm. Standard for stabilizing GANs."),
    ]


def _regularization() -> list[LayerSpec]:
    """
    > [!AML-DOC-UNIT]
    Stochastic regularizers and the activity penalty layer.
    @returns LayerSpec list for the Regularization palette category
    """
    section = "regularization_layers"
    specs = [
        _spec("Dropout", "Dropout", Category.REGULARIZATION, doc_section=f"{section}/dropout",
              tags=["regularize", "overfitting"],
              description="Randomly zeroes a fraction of units during training only."),
        _spec("GaussianDropout", "Gaussian Dropout", Category.REGULARIZATION,
              doc_section=f"{section}/gaussian_dropout", tags=["regularize", "multiplicative noise"],
              description="Multiplies activations by Gaussian noise of mean 1."),
        _spec("GaussianNoise", "Gaussian Noise", Category.REGULARIZATION,
              doc_section=f"{section}/gaussian_noise", tags=["regularize", "additive noise", "augment"],
              description="Adds zero-centred Gaussian noise during training."),
        _spec("AlphaDropout", "Alpha Dropout", Category.REGULARIZATION,
              doc_section=f"{section}/alpha_dropout", tags=["selu", "self-normalizing"],
              description="Dropout that preserves mean and variance. Pairs with SELU networks."),
        _spec("ActivityRegularization", "Activity Regularization", Category.REGULARIZATION,
              doc_section=f"{section}/activity_regularization", tags=["l1", "l2", "sparsity"],
              description="Adds an L1/L2 penalty on the layer's own activations."),
    ]
    for dim, (rank, modality) in {1: (3, _SEQ), 2: (4, _IMAGE), 3: (5, _VOL)}.items():
        specs.append(_spec(
            f"SpatialDropout{dim}D", f"Spatial Dropout {dim}D", Category.REGULARIZATION,
            doc_section=f"{section}/spatial_dropout{dim}d", rank_in=rank, rank_out=rank,
            modalities=modality, tags=["regularize", "channel dropout", "conv"],
            description="Drops entire feature maps instead of individual elements.",
        ))
    return specs


def _attention() -> list[LayerSpec]:
    """
    > [!AML-DOC-UNIT]
    Stock attention layers. These take multiple tensors, so they declare ordered
    query/value/key ports that the compiler passes positionally [amm: E.3].
    @returns LayerSpec list for the Attention palette category
    """
    section = "attention_layers"
    qvk = [
        PortSpec(name="query", label="Query", rank=3),
        PortSpec(name="value", label="Value", rank=3),
        PortSpec(name="key", label="Key", rank=3, optional=True),
    ]
    return [
        _spec("MultiHeadAttention", "Multi-Head Attention", Category.ATTENTION,
              doc_section=f"{section}/multi_head_attention", rank_in=3, rank_out=3,
              min_inputs=2, max_inputs=3, inputs=qvk, modalities=_SEQ,
              tags=["mha", "transformer", "self attention", "cross attention"],
              call_style=CallStyle.QUERY_VALUE_KEY,
              description="The transformer attention block: several attention heads in parallel."),
        _spec("GroupQueryAttention", "Grouped-Query Attention", Category.ATTENTION,
              doc_section=f"{section}/group_query_attention", rank_in=3, rank_out=3,
              min_inputs=2, max_inputs=3, inputs=qvk, modalities=_SEQ,
              tags=["gqa", "llama", "efficient", "kv cache"],
              call_style=CallStyle.QUERY_VALUE_KEY,
              description="Attention where query heads share key/value heads, shrinking the KV cache."),
        _spec("Attention", "Dot-Product Attention", Category.ATTENTION,
              doc_section=f"{section}/attention", rank_in=3, rank_out=3,
              min_inputs=2, max_inputs=3, inputs=qvk, modalities=_SEQ,
              tags=["luong", "dot product", "single head"],
              call_style=CallStyle.LIST,
              description="Single-head dot-product attention (Luong style)."),
        _spec("AdditiveAttention", "Additive Attention", Category.ATTENTION,
              doc_section=f"{section}/additive_attention", rank_in=3, rank_out=3,
              min_inputs=2, max_inputs=3, inputs=qvk, modalities=_SEQ,
              tags=["bahdanau", "additive", "concat score"],
              call_style=CallStyle.LIST,
              description="Additive attention scoring (Bahdanau style)."),
    ]


def _reshaping() -> list[LayerSpec]:
    """
    > [!AML-DOC-UNIT]
    Shape-manipulation layers: reshape, flatten, crop, pad and upsample.
    @returns LayerSpec list for the Reshaping palette category
    """
    section = "reshaping_layers"
    specs = [
        _spec("Reshape", "Reshape", Category.RESHAPING, doc_section=f"{section}/reshape",
              tags=["view", "shape"],
              description="Reinterprets the tensor with a new shape, keeping element count."),
        _spec("Flatten", "Flatten", Category.RESHAPING, doc_section=f"{section}/flatten",
              rank_out=2, tags=["vectorize", "head", "dense bridge"],
              description="Collapses every non-batch axis into one vector."),
        _spec("Permute", "Permute", Category.RESHAPING, doc_section=f"{section}/permute",
              tags=["transpose", "axis order"],
              description="Reorders the tensor axes."),
        _spec("RepeatVector", "Repeat Vector", Category.RESHAPING,
              doc_section=f"{section}/repeat_vector", rank_in=2, rank_out=3,
              tags=["tile", "decoder", "seq2seq"],
              description="Repeats a vector n times to seed a sequence decoder."),
    ]
    for dim, (rank, modality) in {1: (3, _SEQ), 2: (4, _IMAGE), 3: (5, _VOL)}.items():
        specs.append(_spec(
            f"Cropping{dim}D", f"Cropping {dim}D", Category.RESHAPING,
            doc_section=f"{section}/cropping{dim}d", rank_in=rank, rank_out=rank,
            modalities=modality, tags=["trim", "crop", "border"],
            description="Removes rows/columns from the spatial edges.",
        ))
        specs.append(_spec(
            f"UpSampling{dim}D", f"Up Sampling {dim}D", Category.RESHAPING,
            doc_section=f"{section}/up_sampling{dim}d", rank_in=rank, rank_out=rank,
            modalities=modality, tags=["resize", "interpolate", "decoder", "unet"],
            description="Repeats or interpolates values to enlarge the spatial size.",
        ))
        specs.append(_spec(
            f"ZeroPadding{dim}D", f"Zero Padding {dim}D", Category.RESHAPING,
            doc_section=f"{section}/zero_padding{dim}d", rank_in=rank, rank_out=rank,
            modalities=modality, tags=["pad", "border"],
            description="Adds zeros around the spatial edges.",
        ))
    return specs


def _merging() -> list[LayerSpec]:
    """
    > [!AML-DOC-UNIT]
    Layers that combine several tensors into one. These are the skip-connection and
    multi-branch primitives.
    @returns LayerSpec list for the Merging palette category
    """
    section = "merging_layers"
    variadic = {
        "Concatenate": ("Concatenate", "Joins tensors along one axis; all other axes must match.",
                        ["concat", "skip", "unet", "densenet"], 2, None),
        "Add": ("Add", "Element-wise sum. The residual-connection primitive.",
                ["residual", "skip", "resnet", "sum"], 2, None),
        "Average": ("Average", "Element-wise mean of all inputs.", ["mean", "ensemble"], 2, None),
        "Maximum": ("Maximum", "Element-wise maximum of all inputs.", ["max"], 2, None),
        "Minimum": ("Minimum", "Element-wise minimum of all inputs.", ["min"], 2, None),
        "Multiply": ("Multiply", "Element-wise product. Used for gating and attention masks.",
                     ["gate", "hadamard", "product"], 2, None),
        "Subtract": ("Subtract", "Element-wise difference of exactly two tensors.",
                     ["difference", "residual"], 2, 2),
        "Dot": ("Dot", "Contracts two tensors along the chosen axes.",
                ["inner product", "similarity"], 2, 2),
    }
    specs: list[LayerSpec] = []
    for name, (label, description, tags, lo, hi) in variadic.items():
        specs.append(_spec(
            name, label, Category.MERGING, doc_section=f"{section}/{_snake(name)}",
            min_inputs=lo, max_inputs=hi, description=description, tags=tags,
            inputs=[PortSpec(name="inputs", label="Inputs")], call_style=CallStyle.LIST,
        ))
    return specs


def _activation() -> list[LayerSpec]:
    """
    > [!AML-DOC-UNIT]
    Activation functions that carry state or parameters of their own, and therefore
    exist as layers rather than as the `activation` argument.
    @returns LayerSpec list for the Activation palette category
    """
    section = "activation_layers"
    return [
        _spec("ReLU", "ReLU", Category.ACTIVATION, doc_section=f"{section}/relu",
              tags=["rectifier", "nonlinearity"],
              description="Rectified linear unit with optional ceiling and negative slope."),
        _spec("LeakyReLU", "Leaky ReLU", Category.ACTIVATION, doc_section=f"{section}/leaky_relu",
              tags=["rectifier", "dying relu"],
              description="ReLU that lets a small gradient through for negative inputs."),
        _spec("PReLU", "Parametric ReLU", Category.ACTIVATION, doc_section=f"{section}/prelu",
              tags=["rectifier", "learnable"],
              description="Leaky ReLU whose negative slope is learned per channel."),
        _spec("ELU", "ELU", Category.ACTIVATION, doc_section=f"{section}/elu",
              tags=["exponential linear"],
              description="Exponential linear unit: smooth and negative-saturating."),
        _spec("Softmax", "Softmax", Category.ACTIVATION, doc_section=f"{section}/softmax",
              tags=["probability", "classifier", "logits"],
              description="Turns logits into a probability distribution over one axis."),
    ]


def _preprocessing() -> list[LayerSpec]:
    """
    > [!AML-DOC-UNIT]
    Preprocessing and augmentation layers. These live inside the model, so the
    exported code carries identical preprocessing to the in-app preview
    [amm: E.4, task 13].
    @returns LayerSpec list for the Preprocessing palette category
    """
    image = "preprocessing_layers/image_preprocessing"
    augment = "preprocessing_layers/image_augmentation"
    text = "preprocessing_layers/text"
    numeric = "preprocessing_layers/numerical"
    cat = "preprocessing_layers/categorical"
    audio = "preprocessing_layers/audio_preprocessing"
    rows: list[tuple[str, str, str, int | None, Sequence[Modality], str, list[str]]] = [
        ("Rescaling", "Rescaling", image, None, _ANY,
         "Multiplies and offsets every value, e.g. 1/255 to map bytes to 0-1.",
         ["normalize", "scale", "0-1"]),
        ("Resizing", "Resizing", image, 4, _IMAGE,
         "Resizes images to a fixed height and width.", ["resize", "interpolate"]),
        ("CenterCrop", "Center Crop", image, 4, _IMAGE,
         "Crops the central region to a fixed size.", ["crop"]),
        ("Normalization", "Feature Normalization", numeric, None, [Modality.TABULAR, Modality.ANY],
         "Standardizes features using mean and variance learned from data.",
         ["standardize", "z-score", "adapt"]),
        ("Discretization", "Discretization", numeric, None, [Modality.TABULAR],
         "Buckets continuous values into integer bins.", ["bucketize", "bins"]),
        ("RandomFlip", "Random Flip", augment, 4, _IMAGE,
         "Randomly mirrors images during training.", ["augment", "mirror"]),
        ("RandomRotation", "Random Rotation", augment, 4, _IMAGE,
         "Randomly rotates images during training.", ["augment"]),
        ("RandomZoom", "Random Zoom", augment, 4, _IMAGE,
         "Randomly zooms images during training.", ["augment"]),
        ("RandomTranslation", "Random Translation", augment, 4, _IMAGE,
         "Randomly shifts images during training.", ["augment", "shift"]),
        ("RandomCrop", "Random Crop", augment, 4, _IMAGE,
         "Takes a random crop of the given size during training.", ["augment"]),
        ("RandomContrast", "Random Contrast", augment, 4, _IMAGE,
         "Randomly perturbs image contrast during training.", ["augment", "color"]),
        ("RandomBrightness", "Random Brightness", augment, 4, _IMAGE,
         "Randomly perturbs image brightness during training.", ["augment", "color"]),
        ("RandomSharpness", "Random Sharpness", augment, 4, _IMAGE,
         "Randomly sharpens or blurs images during training.", ["augment"]),
        ("RandomGrayscale", "Random Grayscale", augment, 4, _IMAGE,
         "Randomly converts images to grayscale during training.", ["augment", "color"]),
        ("AutoContrast", "Auto Contrast", augment, 4, _IMAGE,
         "Rescales each image to span the full value range.", ["contrast"]),
        ("Solarization", "Solarization", augment, 4, _IMAGE,
         "Inverts pixels above a threshold.", ["augment", "simclr"]),
        ("TextVectorization", "Text Vectorization", text, None, [Modality.TEXT],
         "Turns raw strings into token id sequences using a learned vocabulary.",
         ["tokenize", "nlp", "vocabulary", "adapt"]),
        ("StringLookup", "String Lookup", cat, None, [Modality.TEXT, Modality.TABULAR],
         "Maps strings to integer indices.", ["vocabulary", "categorical"]),
        ("IntegerLookup", "Integer Lookup", cat, None, [Modality.TABULAR],
         "Maps integer values to contiguous indices.", ["vocabulary", "categorical"]),
        ("CategoryEncoding", "Category Encoding", cat, None, [Modality.TABULAR],
         "Encodes integer categories as one-hot, multi-hot or counts.", ["one-hot"]),
        ("Hashing", "Hashing", cat, None, [Modality.TABULAR, Modality.TEXT],
         "Hashes categories into a fixed number of bins.", ["hash trick"]),
        ("HashedCrossing", "Hashed Crossing", cat, None, [Modality.TABULAR],
         "Crosses two categorical features into hashed interaction bins.",
         ["feature cross", "interaction"]),
        ("MelSpectrogram", "Mel Spectrogram", audio, 2, [Modality.AUDIO],
         "Converts a raw waveform (batch, samples) into a mel-scaled spectrogram.",
         ["audio", "stft", "mel", "speech"]),
        ("STFTSpectrogram", "STFT Spectrogram", audio, 3, [Modality.AUDIO],
         "Converts a waveform (batch, samples, channels) into a short-time Fourier "
         "spectrogram. The sample axis must be at least `frame_length` long.",
         ["audio", "stft", "fourier"]),
    ]
    specs: list[LayerSpec] = []
    for name, label, section, rank, modality, description, tags in rows:
        overrides: dict[str, dict[str, Any] | None] = {}
        if name == "HashedCrossing":
            specs.append(_spec(name, label, Category.PREPROCESSING,
                               doc_section=f"{section}/{_snake(name)}", min_inputs=2, max_inputs=2,
                               modalities=modality, description=description, tags=tags,
                               inputs=[PortSpec(name="inputs", label="Features")],
                               call_style=CallStyle.LIST))
            continue
        specs.append(_spec(name, label, Category.PREPROCESSING,
                           doc_section=f"{section}/{_snake(name)}", rank_in=rank,
                           modalities=modality, description=description, tags=tags,
                           overrides=overrides))
    return specs



def _recurrent_ports() -> list[PortSpec]:
    """
    > [!AML-DOC-UNIT]
    The ports a stateful recurrent layer exposes.
    @returns the sequence input and the optional starting state
    @sideEffects none
    @context Keras takes the starting state by keyword rather than as a second
             positional tensor, which is why these layers need a call style of their
             own. The state port is optional: the ordinary case leaves it unconnected
             and Keras starts from zeros [E-032].
    """
    return [
        PortSpec(name="input", label="Sequence", rank=3),
        PortSpec(name="initial_state", label="Initial state", rank=2, optional=True),
    ]

def _snake(name: str) -> str:
    """
    > [!AML-DOC-UNIT]
    Convert a CamelCase class name to the snake_case slug keras.io uses in URLs.
    @param name CamelCase class name, e.g. "RandomFlip"
    @returns snake_case slug, e.g. "random_flip"
    """
    out: list[str] = []
    for index, char in enumerate(name):
        if char.isupper() and index and not name[index - 1].isupper():
            out.append("_")
        out.append(char.lower())
    return "".join(out)


def build_keras_specs() -> list[LayerSpec]:
    """
    > [!AML-DOC-UNIT]
    Assemble the LayerSpec list for every stock Keras layer the tool exposes.
    @returns flat list of LayerSpec across all non-research categories
    @raises ImportError when a declared Keras class is absent, which signals the
            installed Keras version drifted from the one this table was verified
            against [amm: C]
    @sideEffects imports keras, which loads TensorFlow
    """
    specs: list[LayerSpec] = []
    for builder in (
        _core, _convolution, _pooling, _recurrent, _normalization,
        _regularization, _attention, _reshaping, _merging, _activation, _preprocessing,
    ):
        specs.extend(builder())
    return specs
