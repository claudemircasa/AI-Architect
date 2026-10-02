"""
> [!AML-DOC-FILE]
@file       layers/neuro.py
@description Spiking neurons: units that communicate with discrete events rather
             than continuous values.
@module     nnarch.layers.neuro
@exports    LIFCell, SpikingLIF, PoissonEncoder
@created    2026-09-30
@context    AML-DOC EXEMPTION (§7): ordinary docstrings, because `export.inline`
            copies these classes into user projects [task 06].

            A spike is a step function, whose derivative is zero everywhere and
            undefined at the threshold, so gradient descent has nothing to follow.
            These layers use a surrogate gradient: the forward pass emits a real
            spike, the backward pass sees a smooth approximation. That substitution
            is what made deep spiking networks trainable at all.
"""

from __future__ import annotations

import keras
from keras import ops


def _spike_with_surrogate(membrane, threshold, slope):
    """Emit a hard spike forwards while passing a smooth gradient backwards.

    The straight-through construction: the smooth term carries the derivative,
    and the difference between the hard and smooth values is detached, so the
    forward value is exactly the step function.
    """
    hard = ops.cast(membrane >= threshold, membrane.dtype)
    smooth = ops.sigmoid(slope * (membrane - threshold))
    return smooth + ops.stop_gradient(hard - smooth)


@keras.saving.register_keras_serializable(package="custom_layers")
class LIFCell(keras.layers.Layer):
    """Leaky integrate-and-fire neuron: the workhorse of spiking networks.

    Charge accumulates on a membrane, leaks away over time, and when it crosses
    a threshold the neuron emits a spike and resets. Because the output is a
    binary event, inference on neuromorphic hardware costs an addition where a
    conventional network costs a multiply.

    Reference: Neftci et al. 2019, "Surrogate Gradient Learning in Spiking
    Neural Networks", arXiv:1901.09948.

    Args:
        units: Number of neurons.
        decay: Fraction of the membrane potential retained each step.
        threshold: Potential at which the neuron fires.
        surrogate_slope: Steepness of the smooth gradient substitute. Higher is
            closer to the true step but gives a narrower window of useful
            gradient.
        reset: `subtract` removes exactly the threshold after firing and keeps
            the remainder, which preserves information; `zero` discards it.
    """

    def __init__(
        self,
        units: int,
        decay: float = 0.9,
        threshold: float = 1.0,
        surrogate_slope: float = 10.0,
        reset: str = "subtract",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.units = int(units)
        self.decay = float(decay)
        self.threshold = float(threshold)
        self.surrogate_slope = float(surrogate_slope)
        self.reset = reset
        self.state_size = [self.units, self.units]
        self.output_size = self.units

    def build(self, input_shape):
        """Create the synaptic weights feeding the membrane."""
        self.synapse = keras.layers.Dense(self.units, name="synapse")
        super().build(input_shape)

    def call(self, inputs, states):
        """Integrate the input, leak, fire if above threshold, then reset."""
        membrane, _ = states
        membrane = self.decay * membrane + self.synapse(inputs)
        spikes = _spike_with_surrogate(membrane, self.threshold, self.surrogate_slope)

        if self.reset == "zero":
            membrane = membrane * (1.0 - ops.stop_gradient(spikes))
        else:
            membrane = membrane - self.threshold * ops.stop_gradient(spikes)

        return spikes, [membrane, spikes]

    def get_config(self):
        """Serialise every constructor argument."""
        return {
            **super().get_config(),
            "units": self.units,
            "decay": self.decay,
            "threshold": self.threshold,
            "surrogate_slope": self.surrogate_slope,
            "reset": self.reset,
        }


@keras.saving.register_keras_serializable(package="custom_layers")
class SpikingLIF(keras.layers.Layer):
    """A layer of leaky integrate-and-fire neurons over a time axis.

    Expects a sequence (batch, timesteps, features) and emits spikes for each
    step. Feed it with `PoissonEncoder` when the data is static.

    Reference: Neftci et al. 2019, arXiv:1901.09948.

    Args:
        units: Number of neurons.
        decay: Fraction of the membrane potential retained each step.
        threshold: Potential at which the neuron fires.
        surrogate_slope: Steepness of the smooth gradient substitute.
        reset: `subtract` or `zero`.
        return_sequences: Emit every timestep's spikes, or only the last.
    """

    def __init__(
        self,
        units: int,
        decay: float = 0.9,
        threshold: float = 1.0,
        surrogate_slope: float = 10.0,
        reset: str = "subtract",
        return_sequences: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.units = int(units)
        self.decay = float(decay)
        self.threshold = float(threshold)
        self.surrogate_slope = float(surrogate_slope)
        self.reset = reset
        self.return_sequences = return_sequences

    def build(self, input_shape):
        """Create the cell and the RNN that steps it through time."""
        self.rnn = keras.layers.RNN(
            LIFCell(
                self.units, self.decay, self.threshold, self.surrogate_slope, self.reset
            ),
            return_sequences=self.return_sequences,
            name="rnn",
        )
        super().build(input_shape)

    def call(self, inputs):
        """Run the neurons across the time axis."""
        return self.rnn(inputs)

    def compute_output_shape(self, input_shape):
        """A spike train, or the final step's spikes."""
        batch, length, _ = input_shape
        return (batch, length, self.units) if self.return_sequences else (batch, self.units)

    def get_config(self):
        """Serialise every constructor argument."""
        return {
            **super().get_config(),
            "units": self.units,
            "decay": self.decay,
            "threshold": self.threshold,
            "surrogate_slope": self.surrogate_slope,
            "reset": self.reset,
            "return_sequences": self.return_sequences,
        }


@keras.saving.register_keras_serializable(package="custom_layers")
class PoissonEncoder(keras.layers.Layer):
    """Turns a static input into a spike train whose rate encodes its value.

    Spiking networks consume events, not numbers, so an image has to become a
    sequence first. Each feature fires at a probability equal to its value, and
    over enough timesteps the spike count recovers the original magnitude. This
    is rate coding, the simplest and most robust of the encodings.

    Expects values in [0, 1]. Produces (batch, timesteps, features).

    Reference: Tavanaei et al. 2019, "Deep Learning in Spiking Neural Networks",
    arXiv:1804.08150.

    Args:
        timesteps: Length of the generated spike train. Longer gives a more
            faithful rate but costs proportionally more compute.
        seed: Seed for reproducible spike sampling.
    """

    def __init__(self, timesteps: int = 16, seed: int | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.timesteps = int(timesteps)
        self.seed = seed
        self.seed_generator = keras.random.SeedGenerator(seed)

    def call(self, inputs, training=False):
        """Sample a Bernoulli spike per feature per timestep.

        Outside training the expected rate is emitted directly, so evaluation is
        deterministic rather than varying run to run.
        """
        rate = ops.clip(inputs, 0.0, 1.0)
        repeated = ops.repeat(ops.expand_dims(rate, 1), self.timesteps, axis=1)
        if not training:
            return repeated
        noise = keras.random.uniform(ops.shape(repeated), seed=self.seed_generator)
        hard = ops.cast(noise < repeated, repeated.dtype)
        # Straight-through again: the sampling step has no useful derivative, so
        # the rate carries the gradient.
        return repeated + ops.stop_gradient(hard - repeated)

    def compute_output_shape(self, input_shape):
        """A time axis is inserted after the batch."""
        return (input_shape[0], self.timesteps, *input_shape[1:])

    def get_config(self):
        """Serialise every constructor argument."""
        return {
            **super().get_config(),
            "timesteps": self.timesteps,
            "seed": self.seed,
        }
