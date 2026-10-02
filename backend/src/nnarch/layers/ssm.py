"""
> [!AML-DOC-FILE]
@file       layers/ssm.py
@description Selective state-space blocks: the sub-quadratic alternative to
             attention that reads a sequence in linear time.
@module     nnarch.layers.ssm
@exports    MambaBlock, S4DBlock
@created    2026-09-30
@context    AML-DOC EXEMPTION (§7): ordinary docstrings, because `export.inline`
            copies these classes into user projects [task 06].

            The recurrence is solved with `keras.ops.associative_scan`, which costs
            O(L log L) and holds O(L) memory. The obvious alternatives are both
            unusable here: a Python loop over timesteps unrolls into a graph the
            size of the sequence, and the decay-matrix formulation needs an
            (L x L x channels x state) tensor.
"""

from __future__ import annotations

import math

import keras
from keras import ops


def _linear_recurrence(decay, update, static_shape):
    """Solve h_t = decay_t * h_{t-1} + update_t along axis 1, in parallel.

    Composing two steps of a first-order linear recurrence gives another one, so
    the whole sequence can be resolved by an associative scan rather than a loop.

    `static_shape` is the trailing (length, channels, state) triple. The scan
    returns a tensor whose static shape is unknown, which stops any following
    Dense layer from building, so it is restored here.
    """

    def combine(left, right):
        left_decay, left_update = left
        right_decay, right_update = right
        return (
            left_decay * right_decay,
            right_decay * left_update + right_update,
        )

    _, states = ops.associative_scan(combine, (decay, update), axis=1)
    return ops.reshape(states, (-1, *static_shape))


@keras.saving.register_keras_serializable(package="custom_layers")
class MambaBlock(keras.layers.Layer):
    """Mamba: a state-space block whose dynamics depend on the input.

    Earlier state-space models applied the same recurrence to every token, which
    let them run in linear time but meant they could not choose what to keep.
    Mamba makes the step size and the input and output projections functions of
    the token itself, so the model can hold on to one thing and forget another,
    and still reads the sequence in linear time rather than attention's
    quadratic.

    Reference: Gu & Dao 2023, "Mamba: Linear-Time Sequence Modeling with
    Selective State Spaces", arXiv:2312.00752.

    Args:
        state_dim: Size of the hidden state per channel. The paper's `N`; 16 is
            the usual value.
        expansion: Width multiplier of the inner stream. The paper uses 2.
        conv_width: Width of the causal depthwise convolution that mixes
            neighbouring tokens before the recurrence.
        dt_rank: Rank of the step-size projection. Defaults to the model width
            divided by 16, as in the paper.
        use_bias: Whether the input and output projections carry bias terms.
    """

    def __init__(
        self,
        state_dim: int = 16,
        expansion: int = 2,
        conv_width: int = 4,
        dt_rank: int | None = None,
        use_bias: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.state_dim = int(state_dim)
        self.expansion = int(expansion)
        self.conv_width = int(conv_width)
        self.dt_rank = dt_rank
        self.use_bias = use_bias

    def build(self, input_shape):
        """Create the projections, the causal convolution and the state matrix."""
        width = int(input_shape[-1])
        if input_shape[1] is None:
            raise ValueError(
                "MambaBlock needs a fixed sequence length: the parallel scan's output "
                "shape has to be restored statically for the next layer to build. "
                "Pad or crop to a fixed length first."
            )
        self.length = int(input_shape[1])
        self.inner_dim = width * self.expansion
        self.rank = self.dt_rank if self.dt_rank else max(1, math.ceil(width / 16))

        self.in_projection = keras.layers.Dense(
            self.inner_dim * 2, use_bias=self.use_bias, name="in_projection"
        )
        # DepthwiseConv1D offers only "same" and "valid", so causality is obtained
        # by padding the past explicitly and convolving with no further padding.
        # Any "same" padding would let a token see its own future.
        self.causal_pad = keras.layers.ZeroPadding1D(
            padding=(self.conv_width - 1, 0), name="causal_pad"
        )
        self.conv = keras.layers.DepthwiseConv1D(
            kernel_size=self.conv_width,
            padding="valid",
            depth_multiplier=1,
            name="causal_conv",
        )
        self.x_projection = keras.layers.Dense(
            self.rank + self.state_dim * 2, use_bias=False, name="x_projection"
        )
        self.dt_projection = keras.layers.Dense(
            self.inner_dim, use_bias=True, name="dt_projection"
        )
        self.out_projection = keras.layers.Dense(
            width, use_bias=self.use_bias, name="out_projection"
        )

        # A is parameterised in log space and negated, which keeps the
        # continuous-time system stable however the weights move. The S4D-Lin
        # initialisation gives each state a distinct decay rate.
        a_init = [
            [math.log(n + 1) for n in range(self.state_dim)] for _ in range(self.inner_dim)
        ]
        self.a_log = self.add_weight(
            shape=(self.inner_dim, self.state_dim),
            initializer=keras.initializers.Constant(
                keras.ops.convert_to_numpy(ops.convert_to_tensor(a_init))
            ),
            trainable=True,
            name="a_log",
        )
        self.skip = self.add_weight(
            shape=(self.inner_dim,), initializer="ones", trainable=True, name="skip"
        )
        super().build(input_shape)

    def call(self, inputs):
        """Project, convolve causally, run the selective recurrence, then gate and project back."""
        projected = self.in_projection(inputs)
        x, gate = ops.split(projected, 2, axis=-1)

        x = ops.silu(self.conv(self.causal_pad(x)))

        parameters = self.x_projection(x)
        delta, b_matrix, c_matrix = ops.split(
            parameters, [self.rank, self.rank + self.state_dim], axis=-1
        )
        delta = ops.softplus(self.dt_projection(delta))

        a_matrix = -ops.exp(ops.cast(self.a_log, x.dtype))

        # Discretise: the zero-order hold of a diagonal system is an exponential.
        decay = ops.exp(ops.expand_dims(delta, -1) * a_matrix)
        update = (
            ops.expand_dims(delta, -1)
            * ops.expand_dims(b_matrix, 2)
            * ops.expand_dims(x, -1)
        )

        states = _linear_recurrence(
            decay, update, (self.length, self.inner_dim, self.state_dim)
        )
        y = ops.sum(states * ops.expand_dims(c_matrix, 2), axis=-1)
        y = y + x * ops.cast(self.skip, x.dtype)

        return self.out_projection(y * ops.silu(gate))

    def compute_output_shape(self, input_shape):
        """Shape is unchanged; the block projects back to the model width."""
        return input_shape

    def get_config(self):
        """Serialise every constructor argument."""
        return {
            **super().get_config(),
            "state_dim": self.state_dim,
            "expansion": self.expansion,
            "conv_width": self.conv_width,
            "dt_rank": self.dt_rank,
            "use_bias": self.use_bias,
        }


@keras.saving.register_keras_serializable(package="custom_layers")
class S4DBlock(keras.layers.Layer):
    """Diagonal structured state space: the same recurrence for every token.

    S4D keeps one fixed set of dynamics rather than deriving them per token, so
    it has far fewer moving parts than Mamba and is the right thing when the
    signal is genuinely time-invariant, such as raw audio or sensor traces. The
    diagonal parameterisation is what made S4 simple enough to implement in a
    few lines without losing its long-range behaviour.

    Reference: Gu et al. 2022, "On the Parameterization and Initialization of
    Diagonal State Space Models", arXiv:2206.11893.

    Args:
        state_dim: Size of the hidden state per channel.
        dt_min: Smallest step size at initialisation.
        dt_max: Largest step size at initialisation. The spread across channels
            is what gives the block its range of timescales.
    """

    def __init__(
        self,
        state_dim: int = 64,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.state_dim = int(state_dim)
        self.dt_min = float(dt_min)
        self.dt_max = float(dt_max)

    def build(self, input_shape):
        """Create the diagonal dynamics, the input and output maps, and the step sizes."""
        self.channels = int(input_shape[-1])
        if input_shape[1] is None:
            raise ValueError(
                "S4DBlock needs a fixed sequence length: the parallel scan's output "
                "shape has to be restored statically for the next layer to build. "
                "Pad or crop to a fixed length first."
            )
        self.length = int(input_shape[1])

        a_init = [
            [math.log(n + 1) for n in range(self.state_dim)] for _ in range(self.channels)
        ]
        self.a_log = self.add_weight(
            shape=(self.channels, self.state_dim),
            initializer=keras.initializers.Constant(
                keras.ops.convert_to_numpy(ops.convert_to_tensor(a_init))
            ),
            trainable=True,
            name="a_log",
        )
        self.b_matrix = self.add_weight(
            shape=(self.channels, self.state_dim),
            initializer=keras.initializers.RandomNormal(stddev=0.5),
            trainable=True,
            name="b",
        )
        self.c_matrix = self.add_weight(
            shape=(self.channels, self.state_dim),
            initializer=keras.initializers.RandomNormal(stddev=0.5),
            trainable=True,
            name="c",
        )
        self.skip = self.add_weight(
            shape=(self.channels,), initializer="ones", trainable=True, name="skip"
        )
        # Log-uniform step sizes, so different channels respond over different
        # horizons without having to learn that from scratch.
        span = math.log(self.dt_max) - math.log(self.dt_min)
        dt_init = [
            math.log(self.dt_min) + span * index / max(self.channels - 1, 1)
            for index in range(self.channels)
        ]
        self.log_dt = self.add_weight(
            shape=(self.channels,),
            initializer=keras.initializers.Constant(
                keras.ops.convert_to_numpy(ops.convert_to_tensor(dt_init))
            ),
            trainable=True,
            name="log_dt",
        )
        super().build(input_shape)

    def call(self, inputs):
        """Run one fixed diagonal recurrence per channel over the sequence."""
        shape = ops.shape(inputs)
        batch, length = shape[0], shape[1]

        step = ops.exp(ops.cast(self.log_dt, inputs.dtype))
        a_matrix = -ops.exp(ops.cast(self.a_log, inputs.dtype))

        decay = ops.exp(ops.expand_dims(step, -1) * a_matrix)
        decay = ops.broadcast_to(
            ops.reshape(decay, (1, 1, self.channels, self.state_dim)),
            (batch, length, self.channels, self.state_dim),
        )
        update = ops.expand_dims(inputs, -1) * ops.reshape(
            ops.expand_dims(step, -1) * ops.cast(self.b_matrix, inputs.dtype),
            (1, 1, self.channels, self.state_dim),
        )

        states = _linear_recurrence(
            decay, update, (self.length, self.channels, self.state_dim)
        )
        y = ops.sum(
            states
            * ops.reshape(
                ops.cast(self.c_matrix, inputs.dtype), (1, 1, self.channels, self.state_dim)
            ),
            axis=-1,
        )
        return y + inputs * ops.cast(self.skip, inputs.dtype)

    def compute_output_shape(self, input_shape):
        """Shape is unchanged."""
        return input_shape

    def get_config(self):
        """Serialise every constructor argument."""
        return {
            **super().get_config(),
            "state_dim": self.state_dim,
            "dt_min": self.dt_min,
            "dt_max": self.dt_max,
        }
