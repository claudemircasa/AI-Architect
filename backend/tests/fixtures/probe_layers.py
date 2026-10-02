"""
> [!AML-DOC-FILE]
@file       tests/fixtures/probe_layers.py
@description A stand-in custom layer module used to exercise source inlining while
             the real research layers are still unwritten.
@module     fixtures.probe_layers
@exports    SCALE_FLOOR, scaled_gelu, ProbeGatedDense, ProbeUnusedDense
@created    2026-09-30
@context    Shaped to test the hard parts of `export.inline`: a class that depends
            on a module-level constant and a module-level helper, alongside a second
            class that must NOT be carried into the export.

            AML-DOC EXEMPTION (§7): the units below carry ordinary docstrings rather
            than `[!AML-DOC-UNIT]` blocks. This module's source is copied verbatim
            into exported user projects by `export.inline`, so AML markers here would
            leak this project's documentation convention into files handed to users.
            Research layers in `nnarch/layers/*` [task 04] inherit the same exemption
            for the same reason.
"""

from __future__ import annotations

import keras

SCALE_FLOOR = 0.25


def scaled_gelu(x):
    """Apply GELU, then floor the scale. A module-level helper the layer depends on."""
    return keras.ops.gelu(x) * (1.0 - SCALE_FLOOR) + x * SCALE_FLOOR


@keras.saving.register_keras_serializable(package="probe")
class ProbeGatedDense(keras.layers.Layer):
    """A dense projection with a learned gate, for testing inlining."""

    def __init__(self, units: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.units = units
        self.value = keras.layers.Dense(units, use_bias=False)
        self.gate = keras.layers.Dense(units, activation="sigmoid")

    def call(self, inputs):
        """Gate the projected value and pass it through the helper activation."""
        return scaled_gelu(self.value(inputs)) * self.gate(inputs)

    def compute_output_shape(self, input_shape):
        """Report the output shape, which replaces the last axis with `units`."""
        return (*input_shape[:-1], self.units)

    def get_config(self):
        """Serialise the layer so a saved model can be reloaded."""
        return {**super().get_config(), "units": self.units}


@keras.saving.register_keras_serializable(package="probe")
class ProbeUnusedDense(keras.layers.Layer):
    """A second layer that no test model uses, so it must not reach the export."""

    def __init__(self, units: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.units = units
        self.dense = keras.layers.Dense(units)

    def call(self, inputs):
        """Project the input."""
        return self.dense(inputs)

    def get_config(self):
        """Serialise the layer."""
        return {**super().get_config(), "units": self.units}
