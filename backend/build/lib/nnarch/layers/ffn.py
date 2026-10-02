"""
> [!AML-DOC-FILE]
@file       layers/ffn.py
@description Gated feed-forward blocks: the GLU family that replaced the plain
             two-layer MLP in modern transformers.
@module     nnarch.layers.ffn
@exports    GLUFeedForward, SwiGLU, GeGLU
@created    2026-09-30
@context    AML-DOC EXEMPTION (§7): the classes below carry ordinary docstrings, not
            `[!AML-DOC-UNIT]` blocks, because `export.inline` copies their source
            verbatim into user projects [task 06]. This module docstring is not
            copied, so it keeps the file-level block.
            Nothing here may import from `nnarch`: an exported project has no such
            package.
"""

from __future__ import annotations

import keras


@keras.saving.register_keras_serializable(package="custom_layers")
class GLUFeedForward(keras.layers.Layer):
    """Gated-linear-unit feed-forward block.

    Splits the usual expand-then-project MLP into two parallel projections and
    multiplies them, so one branch gates the other. Shazeer (2020) found every
    gated variant beat the plain MLP, and they are now the default in Llama,
    PaLM and Mistral.

    Reference: Shazeer 2020, "GLU Variants Improve Transformer", arXiv:2002.05202.

    Args:
        hidden_dim: Width of the inner projection. Commonly 8/3 of the model
            width, so the parameter count matches a plain MLP of 4x width.
        output_dim: Width of the output. Defaults to the input width.
        activation: Non-linearity applied to the gate branch. `silu` gives
            SwiGLU, `gelu` gives GeGLU, `sigmoid` gives the original GLU.
        dropout: Dropout applied after the gate, before the projection.
        use_bias: Whether the projections carry bias terms. Modern transformers
            usually leave these off.
    """

    def __init__(
        self,
        hidden_dim: int,
        output_dim: int | None = None,
        activation: str = "silu",
        dropout: float = 0.0,
        use_bias: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = output_dim
        self.activation = activation
        self.dropout = float(dropout)
        self.use_bias = use_bias

    def build(self, input_shape):
        """Create the gate, value and output projections."""
        width = int(input_shape[-1])
        out = self.output_dim if self.output_dim is not None else width
        self.gate_projection = keras.layers.Dense(
            self.hidden_dim, use_bias=self.use_bias, name="gate"
        )
        self.value_projection = keras.layers.Dense(
            self.hidden_dim, use_bias=self.use_bias, name="value"
        )
        self.output_projection = keras.layers.Dense(
            out, use_bias=self.use_bias, name="project"
        )
        self.activation_fn = keras.activations.get(self.activation)
        self.dropout_layer = (
            keras.layers.Dropout(self.dropout) if self.dropout > 0.0 else None
        )
        super().build(input_shape)

    def call(self, inputs, training=False):
        """Gate one projection with the other, then project back down."""
        gated = self.activation_fn(self.gate_projection(inputs)) * self.value_projection(
            inputs
        )
        if self.dropout_layer is not None:
            gated = self.dropout_layer(gated, training=training)
        return self.output_projection(gated)

    def compute_output_shape(self, input_shape):
        """Replace the last axis with the output width."""
        out = self.output_dim if self.output_dim is not None else input_shape[-1]
        return (*input_shape[:-1], out)

    def get_config(self):
        """Serialise every constructor argument."""
        return {
            **super().get_config(),
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "activation": self.activation,
            "dropout": self.dropout,
            "use_bias": self.use_bias,
        }


@keras.saving.register_keras_serializable(package="custom_layers")
class SwiGLU(GLUFeedForward):
    """GLU feed-forward with a SiLU gate, as used by Llama and Mistral.

    Reference: Shazeer 2020, arXiv:2002.05202.

    Args:
        hidden_dim: Width of the inner projection.
        output_dim: Width of the output. Defaults to the input width.
        dropout: Dropout applied after the gate.
        use_bias: Whether the projections carry bias terms.
    """

    def __init__(
        self,
        hidden_dim: int,
        output_dim: int | None = None,
        dropout: float = 0.0,
        use_bias: bool = False,
        **kwargs,
    ) -> None:
        kwargs.pop("activation", None)
        super().__init__(
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            activation="silu",
            dropout=dropout,
            use_bias=use_bias,
            **kwargs,
        )

    def get_config(self):
        """Serialise without the fixed activation."""
        config = super().get_config()
        config.pop("activation", None)
        return config


@keras.saving.register_keras_serializable(package="custom_layers")
class GeGLU(GLUFeedForward):
    """GLU feed-forward with a GELU gate, as used by PaLM and T5 v1.1.

    Reference: Shazeer 2020, arXiv:2002.05202.

    Args:
        hidden_dim: Width of the inner projection.
        output_dim: Width of the output. Defaults to the input width.
        dropout: Dropout applied after the gate.
        use_bias: Whether the projections carry bias terms.
    """

    def __init__(
        self,
        hidden_dim: int,
        output_dim: int | None = None,
        dropout: float = 0.0,
        use_bias: bool = False,
        **kwargs,
    ) -> None:
        kwargs.pop("activation", None)
        super().__init__(
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            activation="gelu",
            dropout=dropout,
            use_bias=use_bias,
            **kwargs,
        )

    def get_config(self):
        """Serialise without the fixed activation."""
        config = super().get_config()
        config.pop("activation", None)
        return config
