"""
> [!AML-DOC-FILE]
@file       layers/misc.py
@description Architectures that do not form a family: capsules, set pooling,
             cross-attention bottlenecks, gated depth, low-rank adaptation and
             vector quantisation.
@module     nnarch.layers.misc
@exports    HighwayDense, LoRADense, VectorQuantizer, SetAttentionPooling,
            PerceiverCrossAttention, CapsuleLayer
@created    2026-09-30
@context    AML-DOC EXEMPTION (§7): ordinary docstrings, because `export.inline`
            copies these classes into user projects [task 06].
"""

from __future__ import annotations

import math

import keras
from keras import ops


@keras.saving.register_keras_serializable(package="custom_layers")
class HighwayDense(keras.layers.Layer):
    """A dense layer with a learned gate deciding how much to transform.

    Predates residual connections and solves the same problem: a gate lets the
    input pass through untouched when transforming it would not help, so depth
    stops being an obstacle.

    Reference: Srivastava et al. 2015, "Highway Networks", arXiv:1505.00387.

    Args:
        activation: Non-linearity of the transform path.
        gate_bias: Initial bias of the gate. A negative value starts the layer
            biased towards carrying the input through, which the paper found
            necessary for very deep stacks.
    """

    def __init__(
        self, activation: str = "relu", gate_bias: float = -1.0, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.activation = activation
        self.gate_bias = float(gate_bias)

    def build(self, input_shape):
        """Create the transform and gate paths, both at the input width."""
        width = int(input_shape[-1])
        self.transform = keras.layers.Dense(
            width, activation=self.activation, name="transform"
        )
        self.gate = keras.layers.Dense(
            width,
            activation="sigmoid",
            bias_initializer=keras.initializers.Constant(self.gate_bias),
            name="gate",
        )
        super().build(input_shape)

    def call(self, inputs):
        """Blend the transformed and untouched paths by the gate."""
        gate = self.gate(inputs)
        return gate * self.transform(inputs) + (1.0 - gate) * inputs

    def compute_output_shape(self, input_shape):
        """Shape is unchanged."""
        return input_shape

    def get_config(self):
        """Serialise every constructor argument."""
        return {
            **super().get_config(),
            "activation": self.activation,
            "gate_bias": self.gate_bias,
        }


@keras.saving.register_keras_serializable(package="custom_layers")
class LoRADense(keras.layers.Layer):
    """A frozen dense layer with a small trainable low-rank correction.

    Fine-tuning a large model means updating every weight. LoRA freezes them and
    learns a rank-r correction instead, which is a tiny fraction of the
    parameters and can be folded back into the original weights at the end, so
    inference costs nothing extra.

    Reference: Hu et al. 2021, "LoRA: Low-Rank Adaptation of Large Language
    Models", arXiv:2106.09685.

    Args:
        units: Width of the output.
        rank: Rank of the correction. 4 to 16 is typical even for very wide
            layers.
        alpha: Scaling of the correction, applied as `alpha / rank`.
        freeze_base: Whether the main weight matrix is held fixed, which is the
            point of the method.
        activation: Non-linearity applied to the sum.
    """

    def __init__(
        self,
        units: int,
        rank: int = 8,
        alpha: float = 16.0,
        freeze_base: bool = True,
        activation: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.units = int(units)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.freeze_base = freeze_base
        self.activation = activation

    def build(self, input_shape):
        """Create the frozen base matrix and the two small adapter matrices."""
        width = int(input_shape[-1])
        self.kernel = self.add_weight(
            shape=(width, self.units),
            initializer=keras.initializers.GlorotUniform(),
            trainable=not self.freeze_base,
            name="kernel",
        )
        self.bias = self.add_weight(
            shape=(self.units,), initializer="zeros", trainable=True, name="bias"
        )
        # A starts random and B starts at zero, so the correction contributes
        # nothing at step one and the layer begins exactly as the frozen one.
        self.lora_a = self.add_weight(
            shape=(width, self.rank),
            initializer=keras.initializers.RandomNormal(stddev=1.0 / math.sqrt(width)),
            trainable=True,
            name="lora_a",
        )
        self.lora_b = self.add_weight(
            shape=(self.rank, self.units),
            initializer="zeros",
            trainable=True,
            name="lora_b",
        )
        self.activation_fn = keras.activations.get(self.activation)
        super().build(input_shape)

    def call(self, inputs):
        """Add the scaled low-rank correction to the frozen projection."""
        base = ops.matmul(inputs, ops.cast(self.kernel, inputs.dtype))
        correction = ops.matmul(
            ops.matmul(inputs, ops.cast(self.lora_a, inputs.dtype)),
            ops.cast(self.lora_b, inputs.dtype),
        )
        return self.activation_fn(
            base + correction * (self.alpha / self.rank) + ops.cast(self.bias, inputs.dtype)
        )

    def compute_output_shape(self, input_shape):
        """Replace the last axis with the output width."""
        return (*input_shape[:-1], self.units)

    def get_config(self):
        """Serialise every constructor argument."""
        return {
            **super().get_config(),
            "units": self.units,
            "rank": self.rank,
            "alpha": self.alpha,
            "freeze_base": self.freeze_base,
            "activation": self.activation,
        }


@keras.saving.register_keras_serializable(package="custom_layers")
class VectorQuantizer(keras.layers.Layer):
    """Snaps each vector to the nearest entry of a learned codebook.

    Turns a continuous representation into a discrete one, which is what lets an
    autoregressive model be trained over images or audio. The nearest-neighbour
    lookup has no gradient, so the gradient is copied straight through to the
    encoder, and two auxiliary losses keep the codebook and the encoder near each
    other.

    Reference: van den Oord et al. 2017, "Neural Discrete Representation
    Learning", arXiv:1711.00937.

    Args:
        num_codes: Size of the codebook.
        code_dim: Width of each code. Must match the input width.
        commitment_cost: Weight of the loss pulling the encoder towards its
            chosen code, which stops the encoder outrunning the codebook.
    """

    def __init__(
        self,
        num_codes: int = 512,
        code_dim: int = 64,
        commitment_cost: float = 0.25,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.num_codes = int(num_codes)
        self.code_dim = int(code_dim)
        self.commitment_cost = float(commitment_cost)

    def build(self, input_shape):
        """Create the codebook."""
        if int(input_shape[-1]) != self.code_dim:
            raise ValueError(
                f"VectorQuantizer expects inputs of width {self.code_dim}, matching "
                f"`code_dim`, but received {input_shape[-1]}. Project the input first, "
                f"or set code_dim={input_shape[-1]}."
            )
        self.codebook = self.add_weight(
            shape=(self.num_codes, self.code_dim),
            initializer=keras.initializers.RandomUniform(-1.0, 1.0),
            trainable=True,
            name="codebook",
        )
        super().build(input_shape)

    def call(self, inputs, training=False):
        """Replace each vector by its nearest code, passing the gradient through."""
        flat = ops.reshape(inputs, (-1, self.code_dim))
        codebook = ops.cast(self.codebook, flat.dtype)

        distances = (
            ops.sum(ops.square(flat), axis=1, keepdims=True)
            - 2.0 * ops.matmul(flat, ops.transpose(codebook))
            + ops.sum(ops.square(codebook), axis=1)
        )
        indices = ops.argmin(distances, axis=1)
        quantised = ops.reshape(
            ops.take(codebook, indices, axis=0), ops.shape(inputs)
        )

        if training:
            codebook_loss = ops.mean(
                ops.square(quantised - ops.stop_gradient(inputs))
            )
            commitment_loss = ops.mean(
                ops.square(ops.stop_gradient(quantised) - inputs)
            )
            self.add_loss(codebook_loss + self.commitment_cost * commitment_loss)

        return inputs + ops.stop_gradient(quantised - inputs)

    def compute_output_shape(self, input_shape):
        """Shape is unchanged."""
        return input_shape

    def get_config(self):
        """Serialise every constructor argument."""
        return {
            **super().get_config(),
            "num_codes": self.num_codes,
            "code_dim": self.code_dim,
            "commitment_cost": self.commitment_cost,
        }


@keras.saving.register_keras_serializable(package="custom_layers")
class SetAttentionPooling(keras.layers.Layer):
    """Pools a set with learned seed vectors attending over its elements.

    Set Transformer's pooling-by-multihead-attention. Unlike mean or max pooling
    it can represent interactions between elements, and unlike a recurrent
    reader it is order-independent, which is what a set requires.

    Reference: Lee et al. 2019, "Set Transformer: A Framework for
    Attention-based Permutation-Invariant Neural Networks", arXiv:1810.00825.

    Args:
        num_seeds: Number of output vectors. 1 pools the set to a single vector.
        num_heads: Number of attention heads.
        key_dim: Width of each head.
    """

    def __init__(
        self, num_seeds: int = 1, num_heads: int = 4, key_dim: int = 32, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.num_seeds = int(num_seeds)
        self.num_heads = int(num_heads)
        self.key_dim = int(key_dim)

    def build(self, input_shape):
        """Create the learned seeds and the attention block."""
        width = int(input_shape[-1])
        self.seeds = self.add_weight(
            shape=(1, self.num_seeds, width),
            initializer="glorot_uniform",
            trainable=True,
            name="seeds",
        )
        self.attention = keras.layers.MultiHeadAttention(
            num_heads=self.num_heads, key_dim=self.key_dim, name="attention"
        )
        self.norm = keras.layers.LayerNormalization(epsilon=1e-6, name="norm")
        super().build(input_shape)

    def call(self, inputs):
        """Let the seeds attend over the set."""
        batch = ops.shape(inputs)[0]
        seeds = ops.broadcast_to(
            ops.cast(self.seeds, inputs.dtype),
            (batch, self.num_seeds, ops.shape(inputs)[-1]),
        )
        return self.norm(self.attention(query=seeds, value=inputs, key=inputs))

    def compute_output_shape(self, input_shape):
        """The set axis becomes the seed count."""
        return (input_shape[0], self.num_seeds, input_shape[-1])

    def get_config(self):
        """Serialise every constructor argument."""
        return {
            **super().get_config(),
            "num_seeds": self.num_seeds,
            "num_heads": self.num_heads,
            "key_dim": self.key_dim,
        }


@keras.saving.register_keras_serializable(package="custom_layers")
class PerceiverCrossAttention(keras.layers.Layer):
    """Compresses an input of any size into a fixed number of latent vectors.

    Attention over the raw input costs the square of its length, which rules out
    images, audio and point clouds. Perceiver cross-attends a small fixed set of
    latents into the input once, and does all further work on the latents, so
    cost is linear in input size and independent of it thereafter.

    Takes one input: the array to compress.

    Reference: Jaegle et al. 2021, "Perceiver: General Perception with Iterative
    Attention", arXiv:2103.03206.

    Args:
        num_latents: Number of latent vectors.
        latent_dim: Width of each latent.
        num_heads: Number of attention heads.
        key_dim: Width of each head.
    """

    def __init__(
        self,
        num_latents: int = 64,
        latent_dim: int = 128,
        num_heads: int = 4,
        key_dim: int = 32,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.num_latents = int(num_latents)
        self.latent_dim = int(latent_dim)
        self.num_heads = int(num_heads)
        self.key_dim = int(key_dim)

    def build(self, input_shape):
        """Create the latent array and the cross-attention block."""
        self.latents = self.add_weight(
            shape=(1, self.num_latents, self.latent_dim),
            initializer=keras.initializers.TruncatedNormal(stddev=0.02),
            trainable=True,
            name="latents",
        )
        self.attention = keras.layers.MultiHeadAttention(
            num_heads=self.num_heads, key_dim=self.key_dim, name="cross_attention"
        )
        self.norm_latents = keras.layers.LayerNormalization(
            epsilon=1e-6, name="norm_latents"
        )
        self.norm_inputs = keras.layers.LayerNormalization(
            epsilon=1e-6, name="norm_inputs"
        )
        self.feed_forward = keras.Sequential(
            [
                keras.layers.LayerNormalization(epsilon=1e-6),
                keras.layers.Dense(self.latent_dim * 2, activation="gelu"),
                keras.layers.Dense(self.latent_dim),
            ],
            name="feed_forward",
        )
        super().build(input_shape)

    def call(self, inputs):
        """Cross-attend the latents into the input, then refine them."""
        batch = ops.shape(inputs)[0]
        latents = ops.broadcast_to(
            ops.cast(self.latents, inputs.dtype), (batch, self.num_latents, self.latent_dim)
        )
        attended = self.attention(
            query=self.norm_latents(latents),
            value=self.norm_inputs(inputs),
            key=self.norm_inputs(inputs),
        )
        latents = latents + attended
        return latents + self.feed_forward(latents)

    def compute_output_shape(self, input_shape):
        """The input length is replaced by the fixed latent count."""
        return (input_shape[0], self.num_latents, self.latent_dim)

    def get_config(self):
        """Serialise every constructor argument."""
        return {
            **super().get_config(),
            "num_latents": self.num_latents,
            "latent_dim": self.latent_dim,
            "num_heads": self.num_heads,
            "key_dim": self.key_dim,
        }


@keras.saving.register_keras_serializable(package="custom_layers")
class CapsuleLayer(keras.layers.Layer):
    """Capsules with dynamic routing: vectors whose length encodes confidence.

    A capsule outputs a vector rather than a scalar: its direction describes the
    entity's properties and its length the probability that it is present.
    Routing-by-agreement then sends each lower capsule's output to whichever
    higher capsule predicts it best, which preserves the spatial relationships a
    max-pool discards.

    Expects (batch, input_capsules, input_dim).

    Reference: Sabour et al. 2017, "Dynamic Routing Between Capsules",
    arXiv:1710.09829.

    Args:
        num_capsules: Number of output capsules.
        capsule_dim: Width of each output capsule.
        routing_iterations: Rounds of routing-by-agreement. The paper uses 3;
            more rarely helps.
    """

    def __init__(
        self,
        num_capsules: int = 10,
        capsule_dim: int = 16,
        routing_iterations: int = 3,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.num_capsules = int(num_capsules)
        self.capsule_dim = int(capsule_dim)
        self.routing_iterations = int(routing_iterations)

    def build(self, input_shape):
        """Create one transformation matrix per input/output capsule pair."""
        _, self.input_capsules, self.input_dim = input_shape
        self.transform = self.add_weight(
            shape=(
                1,
                int(self.input_capsules),
                self.num_capsules,
                self.capsule_dim,
                int(self.input_dim),
            ),
            initializer=keras.initializers.GlorotUniform(),
            trainable=True,
            name="transform",
        )
        super().build(input_shape)

    @staticmethod
    def _squash(vectors, axis=-1):
        """Shrink a vector's length into [0, 1) while keeping its direction."""
        squared = ops.sum(ops.square(vectors), axis=axis, keepdims=True)
        scale = squared / (1.0 + squared) / ops.sqrt(squared + 1e-9)
        return scale * vectors

    def call(self, inputs):
        """Predict each output capsule from each input, then route by agreement."""
        expanded = ops.reshape(
            inputs, (-1, int(self.input_capsules), 1, int(self.input_dim), 1)
        )
        predictions = ops.squeeze(
            ops.matmul(ops.cast(self.transform, inputs.dtype), expanded), -1
        )

        logits = ops.zeros_like(predictions[..., :1])
        outputs = None
        for iteration in range(self.routing_iterations):
            coupling = ops.softmax(logits, axis=2)
            outputs = self._squash(
                ops.sum(coupling * predictions, axis=1, keepdims=True), axis=-1
            )
            if iteration < self.routing_iterations - 1:
                logits = logits + ops.sum(
                    predictions * outputs, axis=-1, keepdims=True
                )

        return ops.squeeze(outputs, 1)

    def compute_output_shape(self, input_shape):
        """A grid of capsules, each a vector."""
        return (input_shape[0], self.num_capsules, self.capsule_dim)

    def get_config(self):
        """Serialise every constructor argument."""
        return {
            **super().get_config(),
            "num_capsules": self.num_capsules,
            "capsule_dim": self.capsule_dim,
            "routing_iterations": self.routing_iterations,
        }
