"""
> [!AML-DOC-FILE]
@file       catalog/introspect.py
@description Derives ParamSpec lists automatically from Keras layer constructor
             signatures, so the catalog covers the whole Keras surface without a
             hand-written form per layer.
@module     nnarch.catalog.introspect
@exports    ACTIVATIONS, INITIALIZERS, REGULARIZERS, CONSTRAINTS, resolve_keras_class,
            derive_params, COMMON_PARAMS
@created    2026-09-30
@context    Backbone of [amm: B.1]. Rules are applied in priority order:
            exact-name table -> suffix rule -> default-value type fallback.
"""

from __future__ import annotations

import importlib
import inspect
from typing import Any

from .spec import ParamSpec, ParamType

ACTIVATIONS: list[str] = [
    "linear", "relu", "relu6", "leaky_relu", "elu", "celu", "selu", "gelu", "silu",
    "swish", "hard_silu", "hard_sigmoid", "hard_tanh", "hard_shrink", "sigmoid",
    "log_sigmoid", "tanh", "tanh_shrink", "softmax", "log_softmax", "sparsemax",
    "softplus", "softsign", "soft_shrink", "sparse_plus", "sparse_sigmoid", "mish",
    "squareplus", "exponential", "glu", "threshold",
]

INITIALIZERS: list[str] = [
    "glorot_uniform", "glorot_normal", "he_normal", "he_uniform", "lecun_normal",
    "lecun_uniform", "random_normal", "random_uniform", "truncated_normal",
    "orthogonal", "variance_scaling", "identity", "zeros", "ones", "constant",
]

REGULARIZERS: list[str | None] = [None, "l1", "l2", "l1_l2", "orthogonal_regularizer"]

CONSTRAINTS: list[str | None] = [None, "max_norm", "min_max_norm", "non_neg", "unit_norm"]

DTYPES: list[str] = [
    "float32", "float16", "bfloat16", "float64", "int32", "int64", "bool",
    "mixed_float16", "mixed_bfloat16",
]

_SKIP: frozenset[str] = frozenset({"self", "kwargs", "args", "name", "dtype", "autocast"})

COMMON_PARAMS: list[ParamSpec] = [
    ParamSpec(
        name="dtype", type=ParamType.DTYPE, default=None, choices=[None, *DTYPES],
        help="Computation dtype. Leave empty to inherit the global policy.",
        advanced=True, group="Common",
    ),
    ParamSpec(
        name="trainable", type=ParamType.BOOL, default=True,
        help="When off, the layer's weights are frozen during training.",
        advanced=True, group="Common",
    ),
]

_NAME_HINTS: dict[str, dict[str, Any]] = {
    "units": {"type": ParamType.INT, "minimum": 1,
              "help": "Number of output neurons."},
    "filters": {"type": ParamType.INT, "minimum": 1,
                "help": "Number of convolution output channels (feature maps)."},
    "kernel_size": {"type": ParamType.INT_TUPLE, "minimum": 1,
                    "help": "Spatial size of the convolution window."},
    "strides": {"type": ParamType.INT_TUPLE, "minimum": 1,
                "help": "Step of the sliding window; >1 downsamples."},
    "dilation_rate": {"type": ParamType.INT_TUPLE, "minimum": 1,
                      "help": "Spacing between kernel taps; enlarges the receptive field."},
    "pool_size": {"type": ParamType.INT_TUPLE, "minimum": 1,
                  "help": "Size of the pooling window."},
    "data_format": {"type": ParamType.ENUM, "choices": [None, "channels_last", "channels_first"],
                    "help": "Axis order of the input tensor.", "advanced": True},
    "groups": {"type": ParamType.INT, "minimum": 1,
               "help": "Split channels into groups convolved independently.", "advanced": True},
    "use_bias": {"type": ParamType.BOOL, "help": "Add a learnable bias vector."},
    "rate": {"type": ParamType.FLOAT, "minimum": 0.0, "maximum": 1.0,
             "help": "Fraction of units dropped during training."},
    "dropout": {"type": ParamType.FLOAT, "minimum": 0.0, "maximum": 1.0,
                "help": "Dropout applied to the input transformation."},
    "recurrent_dropout": {"type": ParamType.FLOAT, "minimum": 0.0, "maximum": 1.0,
                          "help": "Dropout applied to the recurrent state.", "advanced": True},
    "epsilon": {"type": ParamType.FLOAT, "minimum": 0.0,
                "help": "Small constant added to the variance for numerical stability.",
                "advanced": True},
    "momentum": {"type": ParamType.FLOAT, "minimum": 0.0, "maximum": 1.0,
                 "help": "Decay used for the moving statistics.", "advanced": True},
    "center": {"type": ParamType.BOOL, "help": "Learn an additive beta offset."},
    "scale": {"type": ParamType.BOOL, "help": "Learn a multiplicative gamma factor."},
    "num_heads": {"type": ParamType.INT, "minimum": 1,
                  "help": "Number of parallel attention heads."},
    "num_key_value_heads": {"type": ParamType.INT, "minimum": 1,
                            "help": "Number of shared key/value heads (grouped-query attention)."},
    "num_query_heads": {"type": ParamType.INT, "minimum": 1,
                        "help": "Number of query heads; must be a multiple of the key/value heads."},
    "head_dim": {"type": ParamType.INT, "minimum": 1,
                 "help": "Size of each attention head."},
    "output_size": {"type": ParamType.INT_TUPLE, "minimum": 1,
                    "help": "Exact spatial size the layer pools down to."},
    "num_bins": {"type": ParamType.INT, "minimum": 1,
                 "help": "Number of hash or quantile buckets."},
    "num_tokens": {"type": ParamType.INT, "minimum": 1,
                   "help": "Size of the category vocabulary being encoded."},
    "factor": {"type": ParamType.FLOAT,
               "help": "Strength of the random augmentation, as a fraction."},
    "height_factor": {"type": ParamType.FLOAT,
                      "help": "Vertical augmentation strength, as a fraction of the height."},
    "width_factor": {"type": ParamType.FLOAT,
                     "help": "Horizontal augmentation strength, as a fraction of the width."},
    "height": {"type": ParamType.INT, "minimum": 1, "help": "Target height in pixels."},
    "width": {"type": ParamType.INT, "minimum": 1, "help": "Target width in pixels."},
    "offset": {"type": ParamType.FLOAT, "help": "Value added after scaling."},
    "key_dim": {"type": ParamType.INT, "minimum": 1,
                "help": "Per-head size of the query and key projections."},
    "value_dim": {"type": ParamType.INT, "minimum": 1,
                  "help": "Per-head size of the value projection. Defaults to key_dim."},
    "input_dim": {"type": ParamType.INT, "minimum": 1,
                  "help": "Vocabulary size: number of distinct input tokens."},
    "output_dim": {"type": ParamType.INT, "minimum": 1,
                   "help": "Dimensionality of each embedding vector."},
    "mask_zero": {"type": ParamType.BOOL,
                  "help": "Treat token id 0 as padding and propagate a mask."},
    "target_shape": {"type": ParamType.SHAPE,
                     "help": "Output shape excluding the batch dimension."},
    "size": {"type": ParamType.INT_TUPLE, "minimum": 1,
             "help": "Upsampling factor per spatial axis."},
    "interpolation": {"type": ParamType.ENUM,
                      "choices": ["nearest", "bilinear", "bicubic", "lanczos3", "lanczos5"],
                      "help": "Resampling method."},
    "axis": {"type": ParamType.INT, "help": "Axis the operation is applied along."},
    "axes": {"type": ParamType.INT_TUPLE, "help": "Axes the operation is applied along."},
    "seed": {"type": ParamType.INT, "help": "Random seed for reproducibility.", "advanced": True},
    "return_sequences": {"type": ParamType.BOOL,
                         "help": "Output every timestep instead of only the last one."},
    "return_state": {"type": ParamType.BOOL,
                     "help": "Additionally output the final internal state.", "advanced": True},
    "go_backwards": {"type": ParamType.BOOL,
                     "help": "Process the sequence in reverse.", "advanced": True},
    "stateful": {"type": ParamType.BOOL,
                 "help": "Carry the final state of each batch into the next batch.",
                 "advanced": True},
    "unroll": {"type": ParamType.BOOL,
               "help": "Unroll the loop at build time: faster but memory-hungry.",
               "advanced": True},
    "merge_mode": {"type": ParamType.ENUM, "choices": ["concat", "sum", "mul", "ave", None],
                   "help": "How the forward and backward passes are combined."},
    "negative_slope": {"type": ParamType.FLOAT, "minimum": 0.0,
                       "help": "Slope applied to negative inputs."},
    "alpha": {"type": ParamType.FLOAT, "help": "Scaling applied to negative inputs."},
    "theta": {"type": ParamType.FLOAT, "help": "Threshold of the activation."},
    "max_value": {"type": ParamType.FLOAT, "help": "Saturation ceiling; empty means unbounded."},
    "threshold": {"type": ParamType.FLOAT, "help": "Input value below which the output is zero."},
    "stddev": {"type": ParamType.FLOAT, "minimum": 0.0,
               "help": "Standard deviation of the injected noise."},
    "l1": {"type": ParamType.FLOAT, "minimum": 0.0, "help": "L1 penalty factor."},
    "l2": {"type": ParamType.FLOAT, "minimum": 0.0, "help": "L2 penalty factor."},
    "equation": {"type": ParamType.STR,
                 "help": "Einstein-summation expression, e.g. 'abc,cd->abd'."},
    "output_shape": {"type": ParamType.SHAPE,
                     "help": "Output shape excluding the batch dimension."},
    "bias_axes": {"type": ParamType.STR,
                  "help": "Equation letters that get a bias term.", "advanced": True},
    "n": {"type": ParamType.INT, "minimum": 1,
          "help": "Number of times the input vector is repeated."},
    "dims": {"type": ParamType.INT_TUPLE,
             "help": "Permutation of the input axes, 1-indexed and excluding batch."},
    "cropping": {"type": ParamType.ANY, "help": "Units trimmed from each spatial edge."},
    "shape": {"type": ParamType.SHAPE,
              "help": "Input shape excluding the batch dimension."},
    "batch_size": {"type": ParamType.INT, "minimum": 1,
                   "help": "Fixed batch size; empty keeps it dynamic.", "advanced": True},
    "sparse": {"type": ParamType.BOOL, "help": "Accept a sparse tensor.", "advanced": True},
    "normalize": {"type": ParamType.BOOL, "help": "L2-normalize the inputs before combining."},
    "score_mode": {"type": ParamType.ENUM, "choices": ["dot", "concat"],
                   "help": "How attention scores are computed."},
    "use_scale": {"type": ParamType.BOOL, "help": "Learn a scalar scale for the scores."},
    "mask_value": {"type": ParamType.FLOAT, "help": "Value that marks a timestep as masked."},
    "function": {"type": ParamType.TEXT,
                 # The expression itself, not the call that built the layer: pasting
                 # `layers.Lambda(lambda t: ...)(y)` in here is the mistake this
                 # sentence exists to prevent [E-040].
                 "help": "The expression only, over `x` — e.g. `x[:, -1, :]`, not the "
                         "`layers.Lambda(...)` call around it. `ops` and `math` are "
                         "in scope; nothing else is."},
    # --- research layer parameters ---
    "hidden_units": {"type": ParamType.INT, "minimum": 1,
                     "help": "Width of the hidden layer inside the block's MLP."},
    "state_dim": {"type": ParamType.INT, "minimum": 1,
                  "help": "Size of the hidden state each channel carries. The paper's N."},
    "expansion": {"type": ParamType.INT, "minimum": 1,
                  "help": "How much wider the block's inner stream is than its input."},
    "conv_width": {"type": ParamType.INT, "minimum": 1,
                   "help": "Window of the causal convolution that mixes neighbouring tokens."},
    "dt_rank": {"type": ParamType.INT, "minimum": 1,
                "help": "Rank of the step-size projection. Empty uses width/16.",
                "advanced": True},
    "dt_min": {"type": ParamType.FLOAT, "minimum": 0.0,
               "help": "Smallest initial step size, setting the shortest timescale.",
               "advanced": True},
    "dt_max": {"type": ParamType.FLOAT, "minimum": 0.0,
               "help": "Largest initial step size, setting the longest timescale.",
               "advanced": True},
    "hidden_dim": {"type": ParamType.INT, "minimum": 1,
                   "help": "Width of the inner projection."},
    "expert_dim": {"type": ParamType.INT, "minimum": 1,
                   "help": "Hidden width inside each expert."},
    "num_experts": {"type": ParamType.INT, "minimum": 1,
                    "help": "How many experts the layer holds."},
    "top_k": {"type": ParamType.INT, "minimum": 1,
              "help": "How many experts each token is routed to. 1 gives Switch routing."},
    "balance_loss_weight": {"type": ParamType.FLOAT, "minimum": 0.0,
                            "help": "Strength of the loss that stops a few experts "
                                    "taking every token."},
    "router_noise": {"type": ParamType.FLOAT, "minimum": 0.0,
                     "help": "Noise added to routing scores while training, to aid "
                             "exploration.", "advanced": True},
    "slots_per_expert": {"type": ParamType.INT, "minimum": 1,
                         "help": "Input slots each expert receives."},
    "grid_size": {"type": ParamType.INT, "minimum": 1,
                  "help": "Number of spline intervals. More gives finer detail."},
    "spline_order": {"type": ParamType.INT, "minimum": 1,
                     "help": "Degree of the B-splines. 3 is cubic, the usual choice."},
    "grid_range": {"type": ParamType.FLOAT_TUPLE, "arity": 2,
                   "help": "Interval the basis spans. Normalise inputs into it."},
    "num_grids": {"type": ParamType.INT, "minimum": 1,
                  "help": "Number of radial basis centres."},
    "degree": {"type": ParamType.INT, "minimum": 1,
               "help": "Highest polynomial degree. Higher fits sharper functions."},
    "wavelet": {"type": ParamType.ENUM, "choices": ["mexican_hat", "morlet"],
                "help": "Mother wavelet the edge functions are built from."},
    "base_activation": {"type": ParamType.ACTIVATION,
                        "help": "Non-linearity of the residual path kept beside the basis."},
    "rope_dim": {"type": ParamType.INT, "minimum": 1,
                 "help": "Width of the rotary part, carried outside the compression."},
    "kv_latent_dim": {"type": ParamType.INT, "minimum": 1,
                      "help": "Width of the cached key/value latent. This is what "
                              "decides memory use during generation."},
    "query_latent_dim": {"type": ParamType.INT, "minimum": 1,
                         "help": "Width of the query bottleneck. Empty projects directly.",
                         "advanced": True},
    "causal": {"type": ParamType.BOOL,
               "help": "Let a token attend only to earlier tokens. Required for generation."},
    "use_rope": {"type": ParamType.BOOL,
                 "help": "Rotate queries and keys so position is encoded by distance."},
    "base": {"type": ParamType.FLOAT, "minimum": 1.0,
             "help": "Geometric base of the rotary frequency ladder. Raise it for "
                     "longer context."},
    "decay": {"type": ParamType.FLOAT, "minimum": 0.0, "maximum": 1.0,
              "help": "Fraction of the membrane potential kept each step."},
    "surrogate_slope": {"type": ParamType.FLOAT, "minimum": 0.0,
                        "help": "Steepness of the gradient substitute. Higher is more "
                                "faithful but gives a narrower learning window."},
    "reset": {"type": ParamType.ENUM, "choices": ["subtract", "zero"],
              "help": "'subtract' keeps the remainder after firing; 'zero' discards it."},
    "timesteps": {"type": ParamType.INT, "minimum": 1,
                  "help": "Length of the generated spike train."},
    "drop_path": {"type": ParamType.FLOAT, "minimum": 0.0, "maximum": 1.0,
                  "help": "Chance of dropping this residual branch for a sample."},
    "layer_scale": {"type": ParamType.FLOAT, "minimum": 0.0,
                    "help": "Initial value of the learned residual scale. 0 disables it.",
                    "advanced": True},
    "se_ratio": {"type": ParamType.FLOAT, "minimum": 0.0, "maximum": 1.0,
                 "help": "Squeeze-and-excite bottleneck ratio. 0 omits it."},
    "reduction_ratio": {"type": ParamType.INT, "minimum": 1,
                        "help": "Bottleneck ratio of the kernel-generating branch."},
    "patch_size": {"type": ParamType.INT, "minimum": 1,
                   "help": "Side length of each square patch, in pixels."},
    "embed_dim": {"type": ParamType.INT, "minimum": 1,
                  "help": "Width of the vector each patch becomes."},
    "tokens_hidden": {"type": ParamType.INT, "minimum": 1,
                      "help": "Width of the MLP that mixes across tokens."},
    "channels_hidden": {"type": ParamType.INT, "minimum": 1,
                        "help": "Width of the MLP that mixes across channels."},
    "num_seeds": {"type": ParamType.INT, "minimum": 1,
                  "help": "Output vectors the pooling produces. 1 pools to a single vector."},
    "num_latents": {"type": ParamType.INT, "minimum": 1,
                    "help": "Fixed number of latents the input is compressed into."},
    "latent_dim": {"type": ParamType.INT, "minimum": 1,
                   "help": "Width of each latent vector."},
    "num_capsules": {"type": ParamType.INT, "minimum": 1,
                     "help": "Number of output capsules."},
    "capsule_dim": {"type": ParamType.INT, "minimum": 1,
                    "help": "Width of each output capsule's vector."},
    "routing_iterations": {"type": ParamType.INT, "minimum": 1,
                           "help": "Rounds of routing-by-agreement. The paper uses 3."},
    "num_codes": {"type": ParamType.INT, "minimum": 1,
                  "help": "Size of the codebook."},
    "code_dim": {"type": ParamType.INT, "minimum": 1,
                 "help": "Width of each code. Must match the input width."},
    "commitment_cost": {"type": ParamType.FLOAT, "minimum": 0.0,
                        "help": "Weight pulling the encoder towards its chosen code."},
    "rank": {"type": ParamType.INT, "minimum": 1,
             "help": "Rank of the low-rank correction. 4 to 16 suits most layers."},
    "freeze_base": {"type": ParamType.BOOL,
                    "help": "Hold the main weight matrix fixed, which is the point of LoRA."},
    "gate_bias": {"type": ParamType.FLOAT,
                  "help": "Initial gate bias. Negative starts the layer biased towards "
                          "passing its input through."},
    "gate_activation": {"type": ParamType.ACTIVATION,
                        "help": "Non-linearity producing the gate."},
    "aggregator": {"type": ParamType.ENUM, "choices": ["mean", "max", "sum"],
                   "help": "How a node's neighbours are summarised."},
    "concat_heads": {"type": ParamType.BOOL,
                     "help": "Concatenate the heads, or average them. Final layers average."},
    "train_epsilon": {"type": ParamType.BOOL,
                      "help": "Learn the extra weight on a node's own features.",
                      "advanced": True},
    "version": {"type": ParamType.ENUM, "choices": ["v2", "v1"],
                "help": "'v2' fixes the original's static attention; prefer it."},
    "backbone_units": {"type": ParamType.INT, "minimum": 1,
                       "help": "Width of the shared backbone inside the cell."},
    "backbone_layers": {"type": ParamType.INT, "minimum": 1,
                        "help": "Depth of that backbone."},
    "use_layer_norm": {"type": ParamType.BOOL,
                       "help": "Normalise inputs into the basis range first."},
    "use_base": {"type": ParamType.BOOL,
                 "help": "Keep a residual dense path beside the learned basis."},
    "frame_step": {"type": ParamType.INT, "minimum": 1,
                   "help": "Hop between successive analysis windows, in samples."},
    "fft_length": {"type": ParamType.INT, "minimum": 1,
                   "help": "FFT size. Defaults to the smallest power of two that fits the frame."},
    "max_freq": {"type": ParamType.FLOAT, "minimum": 0.0,
                 "help": "Highest frequency included, in hertz."},
    "min_freq": {"type": ParamType.FLOAT, "minimum": 0.0,
                 "help": "Lowest frequency included, in hertz."},
    "output_sequence_length": {"type": ParamType.INT, "minimum": 1,
                               "help": "Pad or truncate the token sequence to this length."},
    "backward_layer": {"type": ParamType.LAYER_REF,
                       "help": "Layer used for the backward pass. Defaults to a copy of the "
                               "forward one.", "advanced": True},
    "output_padding": {"type": ParamType.INT_TUPLE, "minimum": 0,
                       "help": "Extra size added to one side of the output.", "advanced": True},
    "lora_rank": {"type": ParamType.INT, "minimum": 1,
                  "help": "Enable LoRA fine-tuning with this rank. Empty means full training.",
                  "advanced": True},
    "lora_alpha": {"type": ParamType.FLOAT, "minimum": 0.0,
                   "help": "LoRA scaling factor. Defaults to the rank.", "advanced": True},
    "quantization_config": {"type": ParamType.ANY,
                            "help": "Post-training quantization settings.", "advanced": True},
    "gptq_unpacked_column_size": {"type": ParamType.INT, "minimum": 1,
                                  "help": "Column count for GPTQ-packed weights.",
                                  "advanced": True},
    "salt": {"type": ParamType.ANY,
             "help": "Value mixed into the hash so different layers bucket differently.",
             "advanced": True},
    "max_tokens": {"type": ParamType.INT, "minimum": 1,
                   "help": "Vocabulary size cap, counting the padding and out-of-vocabulary slots."},
    "sequence_length": {"type": ParamType.INT, "minimum": 1,
                        "help": "Pad or truncate every sequence to this many tokens."},
    "vocabulary": {"type": ParamType.ANY,
                   "help": "Explicit vocabulary, in place of one learned by `adapt()`.",
                   "advanced": True},
    "idf_weights": {"type": ParamType.ANY,
                    "help": "Inverse document frequencies for TF-IDF output.", "advanced": True},
    "mask_token": {"type": ParamType.ANY,
                   "help": "Token reserved for masked positions.", "advanced": True},
    "oov_token": {"type": ParamType.ANY,
                  "help": "Token standing in for anything outside the vocabulary.",
                  "advanced": True},
    "sliding_window": {"type": ParamType.INT, "minimum": 1,
                       "help": "Limit attention to this many neighbouring positions.",
                       "advanced": True},
    "flash_attention": {"type": ParamType.BOOL,
                        "help": "Use the fused attention kernel when the backend offers one.",
                        "advanced": True},
    "attention_axes": {"type": ParamType.INT_TUPLE,
                       "help": "Axes attention is applied over. Empty means all but the batch "
                               "and feature axes.", "advanced": True},
    "weights": {"type": ParamType.ANY,
                "help": "Initial weight values.", "advanced": True},
    "noise_shape": {"type": ParamType.ANY,
                    "help": "Shape of the dropout mask, for sharing it across axes.",
                    "advanced": True},
    "shared_axes": {"type": ParamType.INT_TUPLE,
                    "help": "Axes that share one learned parameter.", "advanced": True},
    "mask": {"type": ParamType.ANY, "help": "Explicit mask tensor.", "advanced": True},
    "arguments": {"type": ParamType.ANY,
                  "help": "Extra keyword arguments passed to the function.", "advanced": True},
    "renorm_clipping": {"type": ParamType.ANY,
                        "help": "Clipping bounds used by batch renormalization.",
                        "advanced": True},
    "bin_boundaries": {"type": ParamType.ANY,
                       "help": "Explicit bucket edges, in place of learned quantiles.",
                       "advanced": True},
    "mean": {"type": ParamType.ANY,
             "help": "Mean used for normalization, in place of one learned by `adapt()`.",
             "advanced": True},
    "variance": {"type": ParamType.ANY,
                 "help": "Variance used for normalization, in place of one learned by `adapt()`.",
                 "advanced": True},
    "activation": {"type": ParamType.ACTIVATION,
                   "help": "Non-linearity applied to the layer output."},
    "recurrent_activation": {"type": ParamType.ACTIVATION,
                             "help": "Non-linearity applied to the gates.", "advanced": True},
}

_SUFFIX_HINTS: list[tuple[str, dict[str, Any]]] = [
    ("_initializer", {"type": ParamType.INITIALIZER, "choices": INITIALIZERS,
                      "help": "Strategy used to initialize these weights.", "advanced": True}),
    ("_regularizer", {"type": ParamType.REGULARIZER, "choices": REGULARIZERS,
                      "help": "Penalty added to the loss for these weights.", "advanced": True}),
    ("_constraint", {"type": ParamType.CONSTRAINT, "choices": CONSTRAINTS,
                     "help": "Hard constraint enforced after each update.", "advanced": True}),
    ("_activation", {"type": ParamType.ACTIVATION, "help": "Non-linearity applied here."}),
    ("_dropout", {"type": ParamType.FLOAT, "minimum": 0.0, "maximum": 1.0,
                  "help": "Dropout probability.", "advanced": True}),
    ("_dim", {"type": ParamType.INT, "minimum": 1, "help": "Dimensionality."}),
    ("_heads", {"type": ParamType.INT, "minimum": 1, "help": "Number of heads."}),
]


def resolve_keras_class(keras_path: str) -> type:
    """
    > [!AML-DOC-UNIT]
    Import the class named by a dotted path.
    @param keras_path e.g. "keras.layers.Conv2D" or "nnarch.layers.kan.DenseKAN"
    @returns the resolved class object
    @raises ImportError when the module cannot be imported
    @raises AttributeError when the module has no such attribute
    """
    module_path, _, attr = keras_path.rpartition(".")
    return getattr(importlib.import_module(module_path), attr)


def _spatial_arity(param_name: str, rank_in: int | None) -> int | None:
    """
    > [!AML-DOC-UNIT]
    Decide how many numbers a tuple-valued spatial parameter needs.
    @param param_name  the tuple parameter, e.g. "kernel_size"
    @param rank_in     declared input rank of the layer, batch axis included
    @returns element count, or None when it cannot be determined
    @context Conv2D has rank_in 4 (batch, h, w, c) and therefore 2 spatial axes.
    """
    if rank_in is None:
        return None
    if param_name in {"kernel_size", "strides", "dilation_rate", "pool_size", "size"}:
        return max(rank_in - 2, 1)
    return None


def _padding_spec(default: Any) -> dict[str, Any]:
    """
    > [!AML-DOC-UNIT]
    Resolve the overloaded `padding` keyword, which is an enum on convolution and
    pooling layers but a tuple of edge widths on ZeroPadding layers.
    @param default the default value read from the constructor signature
    @returns partial ParamSpec fields appropriate to the observed default
    """
    if isinstance(default, str):
        return {"type": ParamType.ENUM, "choices": ["valid", "same"],
                "help": "'valid' shrinks the output; 'same' keeps the spatial size."}
    return {"type": ParamType.ANY,
            "help": "Number of zeros added at each spatial edge."}


def _scale_spec(default: Any) -> dict[str, Any]:
    """
    > [!AML-DOC-UNIT]
    Resolve the overloaded `scale` keyword: normalization layers use it as a
    boolean gamma switch, while Rescaling uses it as a numeric multiplier.
    @param default the default value read from the constructor signature
    @returns partial ParamSpec fields appropriate to the observed default
    @context Resolving this by name alone silently turned BatchNormalization's
             gamma switch into a number field.
    """
    if isinstance(default, bool):
        return {"type": ParamType.BOOL, "help": "Learn a multiplicative gamma factor."}
    return {"type": ParamType.FLOAT, "help": "Multiplier applied to every value."}


_AMBIGUOUS: dict[str, Any] = {"padding": _padding_spec, "scale": _scale_spec}
"""
> [!AML-DOC-UNIT]
Keywords whose meaning differs between layers and therefore cannot be resolved
from the parameter name alone. Each entry maps the name to a resolver that reads
the constructor's own default to decide.
"""


def _fallback_type(default: Any) -> ParamType:
    """
    > [!AML-DOC-UNIT]
    Guess a widget class from the default value when no naming rule applies.
    @param default value taken from the constructor signature
    @returns the best-matching ParamType, defaulting to ANY
    """
    if isinstance(default, bool):
        return ParamType.BOOL
    if isinstance(default, int):
        return ParamType.INT
    if isinstance(default, float):
        return ParamType.FLOAT
    if isinstance(default, str):
        return ParamType.STR
    if isinstance(default, (tuple, list)):
        return ParamType.INT_TUPLE if all(isinstance(v, int) for v in default) else ParamType.ANY
    return ParamType.ANY


def derive_params(
    keras_path: str,
    *,
    rank_in: int | None = None,
    overrides: dict[str, dict[str, Any] | None] | None = None,
    include_common: bool = True,
) -> list[ParamSpec]:
    """
    > [!AML-DOC-UNIT]
    Build the editable parameter list for a layer by inspecting its constructor.
    @param keras_path      dotted path of the layer class
    @param rank_in         declared input rank, used to size spatial tuples
    @param overrides       per-parameter field overrides; a None value hides the
                           parameter, and a key absent from the signature is added
                           as a brand-new parameter. Only an override may force
                           `required`: otherwise optionality comes from whether the
                           constructor supplies a default, so a name-based hint can
                           never mark an optional argument as mandatory.
    @param include_common  append the shared dtype/trainable controls
    @returns ParamSpec list in constructor order, required parameters first
    @raises ImportError when `keras_path` cannot be resolved
    @sideEffects none
    """
    overrides = overrides or {}
    cls = resolve_keras_class(keras_path)
    signature = inspect.signature(cls.__init__)
    specs: list[ParamSpec] = []

    for name, parameter in signature.parameters.items():
        if name in _SKIP or parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        if name in overrides and overrides[name] is None:
            continue

        has_default = parameter.default is not inspect.Parameter.empty
        default = parameter.default if has_default else None

        if name in _AMBIGUOUS:
            fields = dict(_AMBIGUOUS[name](default))
        elif name in _NAME_HINTS:
            fields = dict(_NAME_HINTS[name])
        else:
            fields = next(
                (dict(hint) for suffix, hint in _SUFFIX_HINTS if name.endswith(suffix)),
                {"type": _fallback_type(default)},
            )

        fields["required"] = not has_default
        fields["default"] = default
        fields.update(overrides.get(name) or {})

        if fields["type"] in (ParamType.INT_TUPLE, ParamType.FLOAT_TUPLE):
            fields.setdefault("arity", _spatial_arity(name, rank_in))

        specs.append(ParamSpec(name=name, **fields))

    known = {s.name for s in specs}
    for name, fields in overrides.items():
        if fields is not None and name not in known:
            specs.append(ParamSpec(name=name, **fields))

    specs.sort(key=lambda s: (not s.required, s.advanced))
    if include_common:
        specs.extend(COMMON_PARAMS)
    return specs
