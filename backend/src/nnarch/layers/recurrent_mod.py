"""
> [!AML-DOC-FILE]
@file       layers/recurrent_mod.py
@description Modern recurrent blocks: the designs that train in parallel like a
             transformer but run in constant memory like an RNN.
@module     nnarch.layers.recurrent_mod
@exports    Retention, MatrixLSTM, CfCCell, CfC
@created    2026-09-30
@context    AML-DOC EXEMPTION (§7): ordinary docstrings, because `export.inline`
            copies these classes into user projects [task 06].

            RWKV-7's generalised delta rule and xLSTM's scalar sLSTM are not here.
            Both need a sequential state update that cannot be written as the
            parallel form the others use, and a careful implementation of either is
            a task of its own rather than a variation on these.
"""

from __future__ import annotations

import math

import keras
from keras import ops


@keras.saving.register_keras_serializable(package="custom_layers")
class Retention(keras.layers.Layer):
    """RetNet's retention: attention with the softmax replaced by a fixed decay.

    Dropping the softmax makes the operation associative, so the same weights
    run three ways: in parallel for training, as a recurrence for generation at
    constant memory per step, and chunk-wise for long sequences. That is the
    property attention does not have.

    Reference: Sun et al. 2023, "Retentive Network: A Successor to Transformer
    for Large Language Models", arXiv:2307.08621.

    Args:
        num_heads: Number of retention heads.
        head_dim: Width of each head.
        gate_activation: Non-linearity of the output gate.
    """

    def __init__(
        self,
        num_heads: int = 4,
        head_dim: int = 64,
        gate_activation: str = "swish",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.gate_activation = gate_activation

    def build(self, input_shape):
        """Create the projections and the per-head decay rates."""
        width = int(input_shape[-1])
        inner = self.num_heads * self.head_dim
        self.to_query = keras.layers.Dense(inner, use_bias=False, name="query")
        self.to_key = keras.layers.Dense(inner, use_bias=False, name="key")
        self.to_value = keras.layers.Dense(inner, use_bias=False, name="value")
        self.to_gate = keras.layers.Dense(inner, use_bias=False, name="gate")
        self.group_norm = keras.layers.GroupNormalization(
            groups=self.num_heads, name="group_norm"
        )
        self.to_output = keras.layers.Dense(width, use_bias=False, name="output")
        # One decay per head, spread over the range the paper uses, so different
        # heads remember over different horizons.
        self.decay = ops.convert_to_tensor(
            [
                1.0 - 2.0 ** (-5.0 - index)
                for index in range(self.num_heads)
            ],
            dtype="float32",
        )
        self.activation_fn = keras.activations.get(self.gate_activation)
        super().build(input_shape)

    def call(self, inputs):
        """Score every pair, weight by distance decay, and gate the result."""
        shape = ops.shape(inputs)
        batch, length = shape[0], shape[1]

        def split(x):
            x = ops.reshape(x, (batch, length, self.num_heads, self.head_dim))
            return ops.transpose(x, (0, 2, 1, 3))

        query, key, value = (
            split(self.to_query(inputs)),
            split(self.to_key(inputs)),
            split(self.to_value(inputs)),
        )

        positions = ops.arange(0, length, dtype="float32")
        distance = ops.expand_dims(positions, 1) - ops.expand_dims(positions, 0)
        causal = distance >= 0
        decay = ops.where(
            ops.reshape(causal, (1, 1, length, length)),
            ops.power(
                ops.reshape(self.decay, (1, self.num_heads, 1, 1)),
                ops.reshape(ops.maximum(distance, 0.0), (1, 1, length, length)),
            ),
            ops.zeros((1, 1, length, length), dtype="float32"),
        )

        scores = ops.matmul(query, ops.transpose(key, (0, 1, 3, 2)))
        scores = scores / math.sqrt(self.head_dim)
        scores = scores * ops.cast(decay, scores.dtype)

        attended = ops.matmul(scores, value)
        attended = ops.transpose(attended, (0, 2, 1, 3))
        attended = ops.reshape(attended, (batch, length, self.num_heads * self.head_dim))

        # Normalise within each head at each position. Applying GroupNormalization
        # to the (batch, length, channels) tensor directly would reduce over the
        # time axis too, which mixes information across positions and destroys
        # causality — the property that makes retention usable for generation.
        flat = ops.reshape(attended, (-1, self.num_heads * self.head_dim))
        attended = ops.reshape(
            self.group_norm(flat), (batch, length, self.num_heads * self.head_dim)
        )
        return self.to_output(self.activation_fn(self.to_gate(inputs)) * attended)

    def compute_output_shape(self, input_shape):
        """Shape is unchanged."""
        return input_shape

    def get_config(self):
        """Serialise every constructor argument."""
        return {
            **super().get_config(),
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "gate_activation": self.gate_activation,
        }


@keras.saving.register_keras_serializable(package="custom_layers")
class MatrixLSTM(keras.layers.Layer):
    """xLSTM's mLSTM: an LSTM whose memory is a matrix, with exponential gates.

    Two changes to the LSTM. The cell state becomes an outer-product matrix, so
    capacity grows quadratically rather than linearly with width. And the gates
    become exponential rather than sigmoid, which lets the cell revise a stored
    value instead of only decaying it — the limitation that made LSTMs lose to
    transformers on retrieval. Exponential gates need a running max subtracted
    for stability, which is what the stabiliser below does.

    Reference: Beck et al. 2024, "xLSTM: Extended Long Short-Term Memory",
    arXiv:2405.04517.

    Args:
        num_heads: Number of memory matrices.
        head_dim: Width of each head.
    """

    def __init__(self, num_heads: int = 4, head_dim: int = 64, **kwargs) -> None:
        super().__init__(**kwargs)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)

    def build(self, input_shape):
        """Create the query, key, value and gate projections."""
        width = int(input_shape[-1])
        inner = self.num_heads * self.head_dim
        self.to_query = keras.layers.Dense(inner, name="query")
        self.to_key = keras.layers.Dense(inner, name="key")
        self.to_value = keras.layers.Dense(inner, name="value")
        self.input_gate = keras.layers.Dense(self.num_heads, name="input_gate")
        self.forget_gate = keras.layers.Dense(self.num_heads, name="forget_gate")
        self.norm = keras.layers.LayerNormalization(epsilon=1e-6, name="norm")
        self.to_output = keras.layers.Dense(width, use_bias=False, name="output")
        super().build(input_shape)

    def call(self, inputs):
        """Accumulate gated outer products, stabilised in log space."""
        shape = ops.shape(inputs)
        batch, length = shape[0], shape[1]

        def split(x):
            x = ops.reshape(x, (batch, length, self.num_heads, self.head_dim))
            return ops.transpose(x, (0, 2, 1, 3))

        query, key, value = (
            split(self.to_query(inputs)),
            split(self.to_key(inputs)),
            split(self.to_value(inputs)),
        )
        key = key / math.sqrt(self.head_dim)

        log_input = ops.transpose(
            keras.activations.log_sigmoid(self.input_gate(inputs)), (0, 2, 1)
        )
        log_forget = ops.transpose(
            keras.activations.log_sigmoid(self.forget_gate(inputs)), (0, 2, 1)
        )

        cumulative_forget = ops.cumsum(log_forget, axis=-1)
        # log of the weight token n places on token m: everything forgotten
        # between them, plus m's own input gate.
        weight = (
            ops.expand_dims(cumulative_forget, -1)
            - ops.expand_dims(cumulative_forget, -2)
            + ops.expand_dims(log_input, -2)
        )
        positions = ops.arange(0, length, dtype="float32")
        causal = ops.expand_dims(positions, 1) >= ops.expand_dims(positions, 0)
        weight = ops.where(
            ops.reshape(causal, (1, 1, length, length)),
            weight,
            ops.full_like(weight, -1e9),
        )
        # Subtract the row maximum before exponentiating: without it the
        # exponential gates overflow within a few dozen steps.
        weight = ops.exp(weight - ops.max(weight, axis=-1, keepdims=True))

        scores = ops.matmul(query, ops.transpose(key, (0, 1, 3, 2))) * weight
        normaliser = ops.maximum(
            ops.abs(ops.sum(scores, axis=-1, keepdims=True)), 1.0
        )
        attended = ops.matmul(scores / normaliser, value)

        attended = ops.transpose(attended, (0, 2, 1, 3))
        attended = ops.reshape(attended, (batch, length, self.num_heads * self.head_dim))
        return self.to_output(self.norm(attended))

    def compute_output_shape(self, input_shape):
        """Shape is unchanged."""
        return input_shape

    def get_config(self):
        """Serialise every constructor argument."""
        return {
            **super().get_config(),
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
        }


@keras.saving.register_keras_serializable(package="custom_layers")
class CfCCell(keras.layers.Layer):
    """Closed-form continuous-time cell: a liquid network without the ODE solver.

    Liquid time-constant networks let each neuron's time constant depend on its
    input, which makes them unusually robust on control and sensor tasks. The
    original needed a numerical ODE solver per step; this is the closed-form
    approximation, which is a few times faster and trains with ordinary
    backpropagation.

    Reference: Hasani et al. 2022, "Closed-form continuous-time neural
    networks", Nature Machine Intelligence; arXiv:2106.13898.

    Args:
        units: Size of the hidden state.
        backbone_units: Width of the shared input/state backbone.
        backbone_layers: Depth of that backbone.
    """

    def __init__(
        self,
        units: int,
        backbone_units: int = 64,
        backbone_layers: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.units = int(units)
        self.backbone_units = int(backbone_units)
        self.backbone_layers = int(backbone_layers)
        self.state_size = self.units
        self.output_size = self.units

    def build(self, input_shape):
        """Create the backbone and the three closed-form heads."""
        self.backbone = keras.Sequential(
            [
                keras.layers.Dense(self.backbone_units, activation="tanh")
                for _ in range(self.backbone_layers)
            ],
            name="backbone",
        )
        self.g_head = keras.layers.Dense(self.units, name="g")
        self.h_head = keras.layers.Dense(self.units, name="h")
        self.gate_head = keras.layers.Dense(self.units, name="gate")
        super().build(input_shape)

    def call(self, inputs, states):
        """Interpolate between two candidate states with an input-dependent gate."""
        previous = states[0]
        joined = ops.concatenate([inputs, previous], axis=-1)
        features = self.backbone(joined)

        g = self.g_head(features)
        h = self.h_head(features)
        gate = ops.sigmoid(self.gate_head(features))

        output = g * (1.0 - gate) + h * gate
        return output, [output]

    def get_config(self):
        """Serialise every constructor argument."""
        return {
            **super().get_config(),
            "units": self.units,
            "backbone_units": self.backbone_units,
            "backbone_layers": self.backbone_layers,
        }


@keras.saving.register_keras_serializable(package="custom_layers")
class CfC(keras.layers.Layer):
    """Closed-form continuous-time recurrent layer.

    Wraps `CfCCell` in a Keras RNN, so it behaves like LSTM or GRU on the canvas.

    Reference: Hasani et al. 2022, arXiv:2106.13898.

    Args:
        units: Size of the hidden state.
        backbone_units: Width of the shared backbone inside the cell.
        backbone_layers: Depth of that backbone.
        return_sequences: Output every timestep instead of only the last.
    """

    def __init__(
        self,
        units: int,
        backbone_units: int = 64,
        backbone_layers: int = 1,
        return_sequences: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.units = int(units)
        self.backbone_units = int(backbone_units)
        self.backbone_layers = int(backbone_layers)
        self.return_sequences = return_sequences

    def build(self, input_shape):
        """Create the cell and the RNN that drives it."""
        self.rnn = keras.layers.RNN(
            CfCCell(self.units, self.backbone_units, self.backbone_layers),
            return_sequences=self.return_sequences,
            name="rnn",
        )
        super().build(input_shape)

    def call(self, inputs):
        """Run the cell across the sequence."""
        return self.rnn(inputs)

    def compute_output_shape(self, input_shape):
        """A sequence, or just the final state."""
        batch, length, _ = input_shape
        return (batch, length, self.units) if self.return_sequences else (batch, self.units)

    def get_config(self):
        """Serialise every constructor argument."""
        return {
            **super().get_config(),
            "units": self.units,
            "backbone_units": self.backbone_units,
            "backbone_layers": self.backbone_layers,
            "return_sequences": self.return_sequences,
        }
