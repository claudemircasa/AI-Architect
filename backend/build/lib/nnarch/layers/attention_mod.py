"""
> [!AML-DOC-FILE]
@file       layers/attention_mod.py
@description Attention variants from the LLM efficiency literature: rotary and
             ALiBi positions, shared key/value heads, and latent KV compression.
@module     nnarch.layers.attention_mod
@exports    RotaryPositionEmbedding, ALiBiAttention, MultiQueryAttention,
            MultiHeadLatentAttention
@created    2026-09-30
@context    AML-DOC EXEMPTION (§7): ordinary docstrings, because `export.inline`
            copies these classes into user projects [task 06].
"""

from __future__ import annotations

import math

import keras
from keras import ops


def _rotate_half(x):
    """Swap and negate the halves of the last axis, the rotation RoPE applies."""
    half = ops.shape(x)[-1] // 2
    first, second = x[..., :half], x[..., half:]
    return ops.concatenate([-second, first], axis=-1)


def _rope_tables(length, head_dim, base, dtype):
    """Build the cosine and sine tables RoPE multiplies positions by."""
    half = head_dim // 2
    exponent = ops.arange(0, half, dtype="float32") * (2.0 / head_dim)
    inverse = 1.0 / ops.power(ops.cast(base, "float32"), exponent)
    positions = ops.arange(0, length, dtype="float32")
    angles = ops.einsum("i,j->ij", positions, inverse)
    angles = ops.concatenate([angles, angles], axis=-1)
    return ops.cast(ops.cos(angles), dtype), ops.cast(ops.sin(angles), dtype)


@keras.saving.register_keras_serializable(package="custom_layers")
class RotaryPositionEmbedding(keras.layers.Layer):
    """Encodes position by rotating pairs of features, rather than adding a vector.

    RoPE makes the dot product between two positions depend only on their
    distance, so a model can generalise to lengths it never saw. It is the
    position scheme in Llama, Mistral, Qwen and most recent open models.

    Applied to a sequence of shape (batch, length, features); features must be
    even, since the rotation works on pairs.

    Reference: Su et al. 2021, "RoFormer: Enhanced Transformer with Rotary
    Position Embedding", arXiv:2104.09864.

    Args:
        base: Geometric base of the frequency ladder. Larger values stretch the
            usable context; 10000 is the original, and long-context models often
            raise it.
    """

    def __init__(self, base: float = 10000.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.base = float(base)

    def call(self, inputs):
        """Rotate each feature pair by an angle proportional to its position."""
        length = ops.shape(inputs)[1]
        cos, sin = _rope_tables(length, inputs.shape[-1], self.base, inputs.dtype)
        return inputs * cos + _rotate_half(inputs) * sin

    def compute_output_shape(self, input_shape):
        """Shape is unchanged."""
        return input_shape

    def get_config(self):
        """Serialise every constructor argument."""
        return {**super().get_config(), "base": self.base}


@keras.saving.register_keras_serializable(package="custom_layers")
class ALiBiAttention(keras.layers.Layer):
    """Self-attention that encodes position as a linear penalty on distance.

    Instead of position embeddings, ALiBi subtracts a per-head slope times the
    distance between tokens. There is nothing to learn and nothing to
    extrapolate, so a model trained at one length works at longer ones.

    Reference: Press et al. 2021, "Train Short, Test Long: Attention with Linear
    Biases Enables Input Length Extrapolation", arXiv:2108.12409.

    Args:
        num_heads: Number of attention heads.
        head_dim: Width of each head.
        causal: Whether a token may attend only to earlier tokens.
        dropout: Dropout applied to the attention weights.
    """

    def __init__(
        self,
        num_heads: int = 8,
        head_dim: int = 64,
        causal: bool = True,
        dropout: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.causal = causal
        self.dropout = float(dropout)

    def build(self, input_shape):
        """Create the projections and the fixed per-head slopes."""
        width = int(input_shape[-1])
        inner = self.num_heads * self.head_dim
        self.to_query = keras.layers.Dense(inner, use_bias=False, name="query")
        self.to_key = keras.layers.Dense(inner, use_bias=False, name="key")
        self.to_value = keras.layers.Dense(inner, use_bias=False, name="value")
        self.to_output = keras.layers.Dense(width, use_bias=False, name="output")
        self.dropout_layer = (
            keras.layers.Dropout(self.dropout) if self.dropout > 0 else None
        )
        # The paper's slopes: a geometric sequence starting at 2^-(8/n).
        start = 2.0 ** (-8.0 / self.num_heads)
        self.slopes = ops.convert_to_tensor(
            [start ** (index + 1) for index in range(self.num_heads)], dtype="float32"
        )
        super().build(input_shape)

    def _split(self, x, batch, length):
        """Reshape a flat projection into (batch, heads, length, head_dim)."""
        x = ops.reshape(x, (batch, length, self.num_heads, self.head_dim))
        return ops.transpose(x, (0, 2, 1, 3))

    def call(self, inputs, training=False):
        """Attend with a distance penalty added to the scores."""
        shape = ops.shape(inputs)
        batch, length = shape[0], shape[1]

        query = self._split(self.to_query(inputs), batch, length)
        key = self._split(self.to_key(inputs), batch, length)
        value = self._split(self.to_value(inputs), batch, length)

        scores = ops.matmul(query, ops.transpose(key, (0, 1, 3, 2)))
        scores = scores / math.sqrt(self.head_dim)

        positions = ops.arange(0, length, dtype="float32")
        distance = ops.expand_dims(positions, 0) - ops.expand_dims(positions, 1)
        bias = ops.reshape(self.slopes, (1, self.num_heads, 1, 1)) * ops.reshape(
            distance, (1, 1, length, length)
        )
        scores = scores + ops.cast(bias, scores.dtype)

        if self.causal:
            mask = ops.triu(ops.ones((length, length), dtype="bool"), k=1)
            scores = ops.where(
                ops.reshape(mask, (1, 1, length, length)),
                ops.full_like(scores, float("-inf")),
                scores,
            )

        weights = ops.softmax(scores, axis=-1)
        if self.dropout_layer is not None:
            weights = self.dropout_layer(weights, training=training)

        attended = ops.matmul(weights, value)
        attended = ops.transpose(attended, (0, 2, 1, 3))
        attended = ops.reshape(attended, (batch, length, self.num_heads * self.head_dim))
        return self.to_output(attended)

    def compute_output_shape(self, input_shape):
        """Shape is unchanged."""
        return input_shape

    def get_config(self):
        """Serialise every constructor argument."""
        return {
            **super().get_config(),
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "causal": self.causal,
            "dropout": self.dropout,
        }


@keras.saving.register_keras_serializable(package="custom_layers")
class MultiQueryAttention(keras.layers.Layer):
    """Many query heads sharing a single key/value head.

    The extreme end of grouped-query attention: the KV cache shrinks by the head
    count, which is what dominates memory during generation. Cheaper and faster
    to decode, at some cost in head diversity.

    Reference: Shazeer 2019, "Fast Transformer Decoding: One Write-Head is All
    You Need", arXiv:1911.02150.

    Args:
        num_heads: Number of query heads.
        head_dim: Width of each head.
        causal: Whether a token may attend only to earlier tokens.
        dropout: Dropout applied to the attention weights.
        use_rope: Whether to rotate queries and keys with RoPE.
    """

    def __init__(
        self,
        num_heads: int = 8,
        head_dim: int = 64,
        causal: bool = True,
        dropout: float = 0.0,
        use_rope: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.causal = causal
        self.dropout = float(dropout)
        self.use_rope = use_rope

    def build(self, input_shape):
        """Create many query projections but only one key and one value head."""
        width = int(input_shape[-1])
        self.to_query = keras.layers.Dense(
            self.num_heads * self.head_dim, use_bias=False, name="query"
        )
        self.to_key = keras.layers.Dense(self.head_dim, use_bias=False, name="key")
        self.to_value = keras.layers.Dense(self.head_dim, use_bias=False, name="value")
        self.to_output = keras.layers.Dense(width, use_bias=False, name="output")
        self.dropout_layer = (
            keras.layers.Dropout(self.dropout) if self.dropout > 0 else None
        )
        super().build(input_shape)

    def call(self, inputs, training=False):
        """Attend with the shared key/value head broadcast across every query head."""
        shape = ops.shape(inputs)
        batch, length = shape[0], shape[1]

        query = ops.reshape(
            self.to_query(inputs), (batch, length, self.num_heads, self.head_dim)
        )
        query = ops.transpose(query, (0, 2, 1, 3))
        key = ops.expand_dims(self.to_key(inputs), 1)
        value = ops.expand_dims(self.to_value(inputs), 1)

        if self.use_rope:
            cos, sin = _rope_tables(length, self.head_dim, 10000.0, query.dtype)
            cos = ops.reshape(cos, (1, 1, length, self.head_dim))
            sin = ops.reshape(sin, (1, 1, length, self.head_dim))
            query = query * cos + _rotate_half(query) * sin
            key = key * cos + _rotate_half(key) * sin

        scores = ops.matmul(query, ops.transpose(key, (0, 1, 3, 2)))
        scores = scores / math.sqrt(self.head_dim)

        if self.causal:
            mask = ops.triu(ops.ones((length, length), dtype="bool"), k=1)
            scores = ops.where(
                ops.reshape(mask, (1, 1, length, length)),
                ops.full_like(scores, float("-inf")),
                scores,
            )

        weights = ops.softmax(scores, axis=-1)
        if self.dropout_layer is not None:
            weights = self.dropout_layer(weights, training=training)

        attended = ops.matmul(weights, value)
        attended = ops.transpose(attended, (0, 2, 1, 3))
        attended = ops.reshape(attended, (batch, length, self.num_heads * self.head_dim))
        return self.to_output(attended)

    def compute_output_shape(self, input_shape):
        """Shape is unchanged."""
        return input_shape

    def get_config(self):
        """Serialise every constructor argument."""
        return {
            **super().get_config(),
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "causal": self.causal,
            "dropout": self.dropout,
            "use_rope": self.use_rope,
        }


@keras.saving.register_keras_serializable(package="custom_layers")
class MultiHeadLatentAttention(keras.layers.Layer):
    """DeepSeek's MLA: caches a small latent vector instead of full keys and values.

    Keys and values are compressed through a low-rank bottleneck, and only that
    latent is cached during generation, which shrinks the KV cache by an order of
    magnitude. Position is carried by a small separate RoPE part, kept outside
    the compression because rotation and low-rank projection do not commute.

    Reference: DeepSeek-AI 2024, "DeepSeek-V2: A Strong, Economical, and
    Efficient Mixture-of-Experts Language Model", arXiv:2405.04434.

    Args:
        num_heads: Number of attention heads.
        head_dim: Width of the content part of each head.
        rope_dim: Width of the rotary part of each head, carried separately.
        kv_latent_dim: Width of the compressed key/value latent. This is what
            gets cached, so it is the number that decides memory use.
        query_latent_dim: Width of the query bottleneck. Omit to project queries
            directly.
        causal: Whether a token may attend only to earlier tokens.
        dropout: Dropout applied to the attention weights.
    """

    def __init__(
        self,
        num_heads: int = 8,
        head_dim: int = 64,
        rope_dim: int = 32,
        kv_latent_dim: int = 128,
        query_latent_dim: int | None = None,
        causal: bool = True,
        dropout: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.rope_dim = int(rope_dim)
        self.kv_latent_dim = int(kv_latent_dim)
        self.query_latent_dim = query_latent_dim
        self.causal = causal
        self.dropout = float(dropout)

    def build(self, input_shape):
        """Create the compression bottlenecks and the decoupled rotary projection."""
        width = int(input_shape[-1])
        self.compress_kv = keras.layers.Dense(
            self.kv_latent_dim, use_bias=False, name="compress_kv"
        )
        self.kv_norm = keras.layers.LayerNormalization(epsilon=1e-6, name="kv_norm")
        self.expand_key = keras.layers.Dense(
            self.num_heads * self.head_dim, use_bias=False, name="expand_key"
        )
        self.expand_value = keras.layers.Dense(
            self.num_heads * self.head_dim, use_bias=False, name="expand_value"
        )
        # The rotary part is produced straight from the input and shared across
        # heads: rotating a compressed key would not survive decompression.
        self.rope_key = keras.layers.Dense(self.rope_dim, use_bias=False, name="rope_key")

        if self.query_latent_dim:
            self.compress_query = keras.layers.Dense(
                self.query_latent_dim, use_bias=False, name="compress_query"
            )
            self.query_norm = keras.layers.LayerNormalization(
                epsilon=1e-6, name="query_norm"
            )
        else:
            self.compress_query = None
            self.query_norm = None

        self.expand_query = keras.layers.Dense(
            self.num_heads * self.head_dim, use_bias=False, name="expand_query"
        )
        self.rope_query = keras.layers.Dense(
            self.num_heads * self.rope_dim, use_bias=False, name="rope_query"
        )
        self.to_output = keras.layers.Dense(width, use_bias=False, name="output")
        self.dropout_layer = (
            keras.layers.Dropout(self.dropout) if self.dropout > 0 else None
        )
        super().build(input_shape)

    def call(self, inputs, training=False):
        """Compress keys and values, attend over content plus rotary parts, expand back."""
        shape = ops.shape(inputs)
        batch, length = shape[0], shape[1]

        latent = self.kv_norm(self.compress_kv(inputs))
        key = ops.transpose(
            ops.reshape(
                self.expand_key(latent), (batch, length, self.num_heads, self.head_dim)
            ),
            (0, 2, 1, 3),
        )
        value = ops.transpose(
            ops.reshape(
                self.expand_value(latent), (batch, length, self.num_heads, self.head_dim)
            ),
            (0, 2, 1, 3),
        )

        query_source = inputs
        if self.compress_query is not None:
            query_source = self.query_norm(self.compress_query(inputs))
        query = ops.transpose(
            ops.reshape(
                self.expand_query(query_source),
                (batch, length, self.num_heads, self.head_dim),
            ),
            (0, 2, 1, 3),
        )

        rope_q = ops.transpose(
            ops.reshape(
                self.rope_query(query_source),
                (batch, length, self.num_heads, self.rope_dim),
            ),
            (0, 2, 1, 3),
        )
        rope_k = ops.expand_dims(self.rope_key(inputs), 1)

        cos, sin = _rope_tables(length, self.rope_dim, 10000.0, rope_q.dtype)
        cos = ops.reshape(cos, (1, 1, length, self.rope_dim))
        sin = ops.reshape(sin, (1, 1, length, self.rope_dim))
        rope_q = rope_q * cos + _rotate_half(rope_q) * sin
        rope_k = rope_k * cos + _rotate_half(rope_k) * sin

        scores = ops.matmul(query, ops.transpose(key, (0, 1, 3, 2))) + ops.matmul(
            rope_q, ops.transpose(rope_k, (0, 1, 3, 2))
        )
        scores = scores / math.sqrt(self.head_dim + self.rope_dim)

        if self.causal:
            mask = ops.triu(ops.ones((length, length), dtype="bool"), k=1)
            scores = ops.where(
                ops.reshape(mask, (1, 1, length, length)),
                ops.full_like(scores, float("-inf")),
                scores,
            )

        weights = ops.softmax(scores, axis=-1)
        if self.dropout_layer is not None:
            weights = self.dropout_layer(weights, training=training)

        attended = ops.matmul(weights, value)
        attended = ops.transpose(attended, (0, 2, 1, 3))
        attended = ops.reshape(attended, (batch, length, self.num_heads * self.head_dim))
        return self.to_output(attended)

    def compute_output_shape(self, input_shape):
        """Shape is unchanged."""
        return input_shape

    def get_config(self):
        """Serialise every constructor argument."""
        return {
            **super().get_config(),
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "rope_dim": self.rope_dim,
            "kv_latent_dim": self.kv_latent_dim,
            "query_latent_dim": self.query_latent_dim,
            "causal": self.causal,
            "dropout": self.dropout,
        }
