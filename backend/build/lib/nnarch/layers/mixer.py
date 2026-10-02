"""
> [!AML-DOC-FILE]
@file       layers/mixer.py
@description Token-mixing blocks and the vision-transformer input stages: the
             architectures that asked whether attention was the necessary part.
@module     nnarch.layers.mixer
@exports    PatchEmbedding, ClassToken, AddPositionEmbedding, MLPMixerBlock,
            GatedMLPBlock, FourierMixing
@created    2026-09-30
@context    AML-DOC EXEMPTION (§7): ordinary docstrings, because `export.inline`
            copies these classes into user projects [task 06].
"""

from __future__ import annotations

import keras
from keras import ops


@keras.saving.register_keras_serializable(package="custom_layers")
class PatchEmbedding(keras.layers.Layer):
    """Cuts an image into square patches and projects each one into a vector.

    The input stage of a vision transformer: a strided convolution whose window
    equals its stride, which is exactly "split into non-overlapping patches and
    apply one linear map to each".

    Reference: Dosovitskiy et al. 2020, "An Image is Worth 16x16 Words",
    arXiv:2010.11929.

    Args:
        patch_size: Side length of each square patch, in pixels.
        embed_dim: Width of the vector each patch becomes.
        use_bias: Whether the projection carries a bias term.
    """

    def __init__(
        self, patch_size: int = 16, embed_dim: int = 768, use_bias: bool = True, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.patch_size = int(patch_size)
        self.embed_dim = int(embed_dim)
        self.use_bias = use_bias

    def build(self, input_shape):
        """Create the patch projection."""
        self.projection = keras.layers.Conv2D(
            self.embed_dim,
            kernel_size=self.patch_size,
            strides=self.patch_size,
            padding="valid",
            use_bias=self.use_bias,
            name="projection",
        )
        super().build(input_shape)

    def call(self, inputs):
        """Project each patch and flatten the grid into a sequence."""
        projected = self.projection(inputs)
        shape = ops.shape(projected)
        return ops.reshape(projected, (shape[0], shape[1] * shape[2], self.embed_dim))

    def compute_output_shape(self, input_shape):
        """An image becomes a sequence of patch vectors."""
        _, height, width, _ = input_shape
        count = (
            None
            if height is None or width is None
            else (height // self.patch_size) * (width // self.patch_size)
        )
        return (input_shape[0], count, self.embed_dim)

    def get_config(self):
        """Serialise every constructor argument."""
        return {
            **super().get_config(),
            "patch_size": self.patch_size,
            "embed_dim": self.embed_dim,
            "use_bias": self.use_bias,
        }


@keras.saving.register_keras_serializable(package="custom_layers")
class ClassToken(keras.layers.Layer):
    """Prepends a learned token whose final state represents the whole sequence.

    Borrowed from BERT's [CLS] and used by ViT: rather than pooling at the end,
    one extra token attends to everything and is read out as the summary.

    Reference: Dosovitskiy et al. 2020, arXiv:2010.11929.
    """

    def build(self, input_shape):
        """Create the learned token."""
        self.token = self.add_weight(
            shape=(1, 1, int(input_shape[-1])),
            initializer="random_normal",
            trainable=True,
            name="class_token",
        )
        super().build(input_shape)

    def call(self, inputs):
        """Broadcast the token across the batch and prepend it."""
        batch = ops.shape(inputs)[0]
        token = ops.broadcast_to(self.token, (batch, 1, ops.shape(inputs)[-1]))
        return ops.concatenate([ops.cast(token, inputs.dtype), inputs], axis=1)

    def compute_output_shape(self, input_shape):
        """One extra position at the front."""
        batch, length, width = input_shape
        return (batch, None if length is None else length + 1, width)


@keras.saving.register_keras_serializable(package="custom_layers")
class AddPositionEmbedding(keras.layers.Layer):
    """Adds a learned vector per position, so order carries information.

    Attention is permutation-invariant on its own; without something like this a
    shuffled sequence is indistinguishable from the original.

    Reference: Dosovitskiy et al. 2020, arXiv:2010.11929.

    Args:
        initializer: Initializer for the position table.
    """

    def __init__(self, initializer: str = "random_normal", **kwargs) -> None:
        super().__init__(**kwargs)
        self.initializer = initializer

    def build(self, input_shape):
        """Create one vector per position, sized from the sequence length."""
        _, length, width = input_shape
        if length is None:
            raise ValueError(
                "AddPositionEmbedding needs a fixed sequence length, but the input "
                "length is dynamic. Pad or crop to a fixed length first."
            )
        self.position = self.add_weight(
            shape=(1, int(length), int(width)),
            initializer=self.initializer,
            trainable=True,
            name="position_embedding",
        )
        super().build(input_shape)

    def call(self, inputs):
        """Add the position table to the sequence."""
        return inputs + ops.cast(self.position, inputs.dtype)

    def compute_output_shape(self, input_shape):
        """Shape is unchanged."""
        return input_shape

    def get_config(self):
        """Serialise every constructor argument."""
        return {**super().get_config(), "initializer": self.initializer}


@keras.saving.register_keras_serializable(package="custom_layers")
class MLPMixerBlock(keras.layers.Layer):
    """Mixes tokens and channels with two MLPs, and no attention at all.

    One MLP runs across the token axis, the other across the channel axis. The
    paper's point was that much of a transformer's performance survives
    replacing attention with a plain transpose-and-project.

    Reference: Tolstikhin et al. 2021, "MLP-Mixer: An all-MLP Architecture for
    Vision", arXiv:2105.01601.

    Args:
        tokens_hidden: Width of the token-mixing MLP.
        channels_hidden: Width of the channel-mixing MLP.
        dropout: Dropout applied inside both MLPs.
    """

    def __init__(
        self,
        tokens_hidden: int = 256,
        channels_hidden: int = 512,
        dropout: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.tokens_hidden = int(tokens_hidden)
        self.channels_hidden = int(channels_hidden)
        self.dropout = float(dropout)

    def build(self, input_shape):
        """Create the two mixing MLPs and their normalisations."""
        _, length, width = input_shape
        if length is None:
            raise ValueError(
                "MLPMixerBlock mixes across the token axis, so it needs a fixed "
                "sequence length. Pad or crop to a fixed length first."
            )
        self.token_norm = keras.layers.LayerNormalization(epsilon=1e-6, name="token_norm")
        self.token_mlp = keras.Sequential(
            [
                keras.layers.Dense(self.tokens_hidden, activation="gelu"),
                keras.layers.Dropout(self.dropout),
                keras.layers.Dense(int(length)),
                keras.layers.Dropout(self.dropout),
            ],
            name="token_mlp",
        )
        self.channel_norm = keras.layers.LayerNormalization(
            epsilon=1e-6, name="channel_norm"
        )
        self.channel_mlp = keras.Sequential(
            [
                keras.layers.Dense(self.channels_hidden, activation="gelu"),
                keras.layers.Dropout(self.dropout),
                keras.layers.Dense(int(width)),
                keras.layers.Dropout(self.dropout),
            ],
            name="channel_mlp",
        )
        super().build(input_shape)

    def call(self, inputs, training=False):
        """Mix across tokens, then across channels, each as a residual."""
        mixed = self.token_norm(inputs)
        mixed = ops.transpose(mixed, (0, 2, 1))
        mixed = self.token_mlp(mixed, training=training)
        mixed = ops.transpose(mixed, (0, 2, 1))
        x = inputs + mixed
        return x + self.channel_mlp(self.channel_norm(x), training=training)

    def compute_output_shape(self, input_shape):
        """Shape is unchanged; both stages are residual."""
        return input_shape

    def get_config(self):
        """Serialise every constructor argument."""
        return {
            **super().get_config(),
            "tokens_hidden": self.tokens_hidden,
            "channels_hidden": self.channels_hidden,
            "dropout": self.dropout,
        }


@keras.saving.register_keras_serializable(package="custom_layers")
class GatedMLPBlock(keras.layers.Layer):
    """gMLP: a spatial gating unit in place of attention.

    Projects up, splits the result in two, mixes one half across positions, and
    uses it to gate the other. Competitive with transformers on several tasks
    without any attention head.

    Reference: Liu et al. 2021, "Pay Attention to MLPs", arXiv:2105.08050.

    Args:
        hidden_dim: Width after the input projection, split in half for gating.
        dropout: Dropout applied after the gate.
    """

    def __init__(self, hidden_dim: int = 512, dropout: float = 0.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)

    def build(self, input_shape):
        """Create the projections and the spatial gating weights."""
        _, length, width = input_shape
        if length is None:
            raise ValueError(
                "GatedMLPBlock gates across positions, so it needs a fixed sequence "
                "length. Pad or crop to a fixed length first."
            )
        half = self.hidden_dim // 2
        self.norm = keras.layers.LayerNormalization(epsilon=1e-6, name="norm")
        self.project_in = keras.layers.Dense(
            self.hidden_dim, activation="gelu", name="project_in"
        )
        self.gate_norm = keras.layers.LayerNormalization(epsilon=1e-6, name="gate_norm")
        # The spatial projection is initialised near identity: the paper notes
        # that a near-zero start keeps the block close to a residual at the
        # beginning of training, which is what makes it stable without warmup.
        self.spatial_weight = self.add_weight(
            shape=(int(length), int(length)),
            initializer=keras.initializers.RandomNormal(stddev=1e-3),
            trainable=True,
            name="spatial_weight",
        )
        self.spatial_bias = self.add_weight(
            shape=(int(length),), initializer="ones", trainable=True, name="spatial_bias"
        )
        self.project_out = keras.layers.Dense(int(width), name="project_out")
        self.dropout_layer = (
            keras.layers.Dropout(self.dropout) if self.dropout > 0 else None
        )
        self.half = half
        super().build(input_shape)

    def call(self, inputs, training=False):
        """Project, split, mix one half across positions, and gate the other with it."""
        x = self.project_in(self.norm(inputs))
        value, gate = x[..., : self.half], x[..., self.half :]
        gate = self.gate_norm(gate)
        gate = ops.einsum("bld,lm->bmd", gate, self.spatial_weight)
        gate = gate + ops.reshape(self.spatial_bias, (1, -1, 1))
        gated = value * gate
        if self.dropout_layer is not None:
            gated = self.dropout_layer(gated, training=training)
        return inputs + self.project_out(gated)

    def compute_output_shape(self, input_shape):
        """Shape is unchanged; the block is residual."""
        return input_shape

    def get_config(self):
        """Serialise every constructor argument."""
        return {
            **super().get_config(),
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
        }


@keras.saving.register_keras_serializable(package="custom_layers")
class FourierMixing(keras.layers.Layer):
    """Replaces attention with an unparameterised Fourier transform.

    FNet mixes tokens by taking a 2D DFT and keeping the real part. It has no
    parameters at all, trains much faster than attention, and still reaches most
    of BERT's accuracy, which makes it a useful baseline for how much of a
    transformer is really the attention.

    Reference: Lee-Thorp et al. 2021, "FNet: Mixing Tokens with Fourier
    Transforms", arXiv:2105.03824.
    """

    def call(self, inputs):
        """Transform along the feature axis, then the sequence axis, keep the real part."""
        real = ops.cast(inputs, "float32")
        imaginary = ops.zeros_like(real)
        real, imaginary = ops.fft((real, imaginary))
        real = ops.transpose(real, (0, 2, 1))
        imaginary = ops.transpose(imaginary, (0, 2, 1))
        real, _ = ops.fft((real, imaginary))
        return ops.cast(ops.transpose(real, (0, 2, 1)), inputs.dtype)

    def compute_output_shape(self, input_shape):
        """Shape is unchanged."""
        return input_shape
