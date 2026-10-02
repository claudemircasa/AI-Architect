"""
> [!AML-DOC-FILE]
@file       layers/kan.py
@description Kolmogorov-Arnold layers: networks that learn the activation functions
             on the edges instead of weights on them.
@module     nnarch.layers.kan
@exports    DenseKAN, ChebyshevKAN, FastKAN, WaveletKAN
@created    2026-09-30
@context    AML-DOC EXEMPTION (§7): ordinary docstrings, because `export.inline`
            copies these classes into user projects [task 06].

            All four follow the Efficient-KAN formulation: evaluate the basis once
            and contract it with a single weight matrix. The original paper's
            per-edge expansion materialises an (in x out x basis) tensor and is
            orders of magnitude slower for no change in what is computed.
"""

from __future__ import annotations

import math

import keras
from keras import ops


@keras.saving.register_keras_serializable(package="custom_layers")
class DenseKAN(keras.layers.Layer):
    """Kolmogorov-Arnold layer with learnable B-spline activations on each edge.

    A Dense layer learns a number per connection and applies a fixed
    non-linearity at the node. A KAN layer inverts that: each connection carries
    a learned univariate function, built from B-splines, and the node simply
    sums. The appeal is interpretability, since each edge's function can be
    plotted and read.

    Reference: Liu et al. 2024, "KAN: Kolmogorov-Arnold Networks",
    arXiv:2404.19756.

    Args:
        units: Number of outputs.
        grid_size: Number of spline intervals. More gives finer detail and more
            parameters.
        spline_order: Degree of the B-splines. 3 is cubic, the usual choice.
        grid_range: Interval the grid spans. Inputs outside it still work but
            lose resolution, so normalise your inputs.
        base_activation: Non-linearity of the residual Dense path, which the
            paper keeps alongside the spline path for trainability.
        use_bias: Whether the output carries a bias term.
    """

    def __init__(
        self,
        units: int,
        grid_size: int = 5,
        spline_order: int = 3,
        grid_range: tuple[float, float] = (-1.0, 1.0),
        base_activation: str = "silu",
        use_bias: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.units = int(units)
        self.grid_size = int(grid_size)
        self.spline_order = int(spline_order)
        self.grid_range = tuple(grid_range)
        self.base_activation = base_activation
        self.use_bias = use_bias

    def build(self, input_shape):
        """Create the fixed knot grid and the base and spline weight matrices."""
        self.in_features = int(input_shape[-1])
        self.basis_size = self.grid_size + self.spline_order

        step = (self.grid_range[1] - self.grid_range[0]) / self.grid_size
        knots = [
            self.grid_range[0] + step * (index - self.spline_order)
            for index in range(self.grid_size + 2 * self.spline_order + 1)
        ]
        self.grid = self.add_weight(
            shape=(1, len(knots)),
            initializer=keras.initializers.Constant(
                keras.ops.convert_to_numpy(ops.convert_to_tensor([knots]))
            ),
            trainable=False,
            name="grid",
        )

        self.base_weight = self.add_weight(
            shape=(self.in_features, self.units),
            initializer=keras.initializers.GlorotUniform(),
            trainable=True,
            name="base_weight",
        )
        self.spline_weight = self.add_weight(
            shape=(self.in_features * self.basis_size, self.units),
            initializer=keras.initializers.RandomNormal(
                stddev=0.1 / math.sqrt(self.in_features)
            ),
            trainable=True,
            name="spline_weight",
        )
        self.bias = (
            self.add_weight(
                shape=(self.units,), initializer="zeros", trainable=True, name="bias"
            )
            if self.use_bias
            else None
        )
        self.activation_fn = keras.activations.get(self.base_activation)
        super().build(input_shape)

    def b_splines(self, inputs):
        """Evaluate every B-spline basis function at each input, by Cox-de Boor recursion.

        Returns a tensor of shape (batch, in_features, grid_size + spline_order).
        """
        grid = ops.cast(self.grid, inputs.dtype)
        x = ops.expand_dims(inputs, -1)

        bases = ops.cast(
            ops.logical_and(x >= grid[:, :-1], x < grid[:, 1:]), inputs.dtype
        )
        for order in range(1, self.spline_order + 1):
            left_numerator = x - grid[:, : -(order + 1)]
            left_denominator = grid[:, order:-1] - grid[:, : -(order + 1)]
            right_numerator = grid[:, order + 1 :] - x
            right_denominator = grid[:, order + 1 :] - grid[:, 1:-order]
            bases = (left_numerator / left_denominator) * bases[..., :-1] + (
                right_numerator / right_denominator
            ) * bases[..., 1:]
        return bases

    def call(self, inputs):
        """Sum the residual Dense path and the contracted spline path."""
        flat = ops.reshape(inputs, (-1, self.in_features))
        base = ops.matmul(self.activation_fn(flat), self.base_weight)
        splines = ops.reshape(
            self.b_splines(flat), (-1, self.in_features * self.basis_size)
        )
        output = base + ops.matmul(splines, self.spline_weight)
        if self.bias is not None:
            output = output + self.bias
        return ops.reshape(output, (*ops.shape(inputs)[:-1], self.units))

    def compute_output_shape(self, input_shape):
        """Replace the last axis with the output width."""
        return (*input_shape[:-1], self.units)

    def get_config(self):
        """Serialise every constructor argument."""
        return {
            **super().get_config(),
            "units": self.units,
            "grid_size": self.grid_size,
            "spline_order": self.spline_order,
            "grid_range": list(self.grid_range),
            "base_activation": self.base_activation,
            "use_bias": self.use_bias,
        }


@keras.saving.register_keras_serializable(package="custom_layers")
class ChebyshevKAN(keras.layers.Layer):
    """Kolmogorov-Arnold layer using Chebyshev polynomials instead of splines.

    Chebyshev polynomials have a recurrence that costs two multiplies per degree
    and need no knot grid, which makes this the cheapest KAN variant to evaluate.
    Inputs are squashed with tanh first, because the polynomials are only
    well-behaved on [-1, 1].

    Reference: SS et al. 2024, "Chebyshev Polynomial-Based Kolmogorov-Arnold
    Networks", arXiv:2405.07200.

    Args:
        units: Number of outputs.
        degree: Highest polynomial degree. Higher fits sharper functions.
    """

    def __init__(self, units: int, degree: int = 4, **kwargs) -> None:
        super().__init__(**kwargs)
        self.units = int(units)
        self.degree = int(degree)

    def build(self, input_shape):
        """Create one coefficient per input, output and degree."""
        self.in_features = int(input_shape[-1])
        self.coefficients = self.add_weight(
            shape=(self.in_features, self.units, self.degree + 1),
            initializer=keras.initializers.RandomNormal(
                stddev=1.0 / math.sqrt(self.in_features * (self.degree + 1))
            ),
            trainable=True,
            name="coefficients",
        )
        super().build(input_shape)

    def call(self, inputs):
        """Build the polynomial basis by recurrence, then contract it with the coefficients."""
        flat = ops.reshape(inputs, (-1, self.in_features))
        x = ops.tanh(flat)

        terms = [ops.ones_like(x), x]
        for _ in range(2, self.degree + 1):
            terms.append(2.0 * x * terms[-1] - terms[-2])
        basis = ops.stack(terms[: self.degree + 1], axis=-1)

        output = ops.einsum("bid,iod->bo", basis, ops.cast(self.coefficients, basis.dtype))
        return ops.reshape(output, (*ops.shape(inputs)[:-1], self.units))

    def compute_output_shape(self, input_shape):
        """Replace the last axis with the output width."""
        return (*input_shape[:-1], self.units)

    def get_config(self):
        """Serialise every constructor argument."""
        return {**super().get_config(), "units": self.units, "degree": self.degree}


@keras.saving.register_keras_serializable(package="custom_layers")
class FastKAN(keras.layers.Layer):
    """Kolmogorov-Arnold layer with Gaussian radial basis functions.

    Replaces the B-spline basis with Gaussians on a fixed grid. A third-order
    B-spline is closely approximated by a Gaussian, so accuracy is similar while
    the forward pass is a single exponential rather than a recursion.

    Reference: Li 2024, "Kolmogorov-Arnold Networks are Radial Basis Function
    Networks", arXiv:2405.06721.

    Args:
        units: Number of outputs.
        num_grids: Number of radial basis centres.
        grid_range: Interval the centres span.
        use_layer_norm: Whether to normalise inputs into the grid range first.
            The paper recommends it, since the centres are fixed.
    """

    def __init__(
        self,
        units: int,
        num_grids: int = 8,
        grid_range: tuple[float, float] = (-2.0, 2.0),
        use_layer_norm: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.units = int(units)
        self.num_grids = int(num_grids)
        self.grid_range = tuple(grid_range)
        self.use_layer_norm = use_layer_norm

    def build(self, input_shape):
        """Create the fixed centres and the contraction weights."""
        self.in_features = int(input_shape[-1])
        span = self.grid_range[1] - self.grid_range[0]
        self.denominator = span / (self.num_grids - 1)
        centres = [
            self.grid_range[0] + self.denominator * index for index in range(self.num_grids)
        ]
        self.centres = self.add_weight(
            shape=(self.num_grids,),
            initializer=keras.initializers.Constant(
                keras.ops.convert_to_numpy(ops.convert_to_tensor(centres))
            ),
            trainable=False,
            name="centres",
        )
        self.norm = (
            keras.layers.LayerNormalization(name="input_norm")
            if self.use_layer_norm
            else None
        )
        self.spline_weight = self.add_weight(
            shape=(self.in_features * self.num_grids, self.units),
            initializer=keras.initializers.RandomNormal(
                stddev=1.0 / math.sqrt(self.in_features * self.num_grids)
            ),
            trainable=True,
            name="spline_weight",
        )
        super().build(input_shape)

    def call(self, inputs):
        """Evaluate the Gaussians and contract them with the weights."""
        x = self.norm(inputs) if self.norm is not None else inputs
        flat = ops.reshape(x, (-1, self.in_features))
        distance = ops.expand_dims(flat, -1) - ops.cast(self.centres, flat.dtype)
        basis = ops.exp(-ops.square(distance / self.denominator))
        basis = ops.reshape(basis, (-1, self.in_features * self.num_grids))
        output = ops.matmul(basis, self.spline_weight)
        return ops.reshape(output, (*ops.shape(inputs)[:-1], self.units))

    def compute_output_shape(self, input_shape):
        """Replace the last axis with the output width."""
        return (*input_shape[:-1], self.units)

    def get_config(self):
        """Serialise every constructor argument."""
        return {
            **super().get_config(),
            "units": self.units,
            "num_grids": self.num_grids,
            "grid_range": list(self.grid_range),
            "use_layer_norm": self.use_layer_norm,
        }


@keras.saving.register_keras_serializable(package="custom_layers")
class WaveletKAN(keras.layers.Layer):
    """Kolmogorov-Arnold layer with a wavelet basis.

    Wavelets are localised in both position and frequency, so each edge function
    can capture a sharp local feature and a broad trend at once. The paper
    reports better robustness to noisy inputs than the spline basis.

    Reference: Bozorgasl & Chen 2024, "Wav-KAN: Wavelet Kolmogorov-Arnold
    Networks", arXiv:2405.12832.

    Args:
        units: Number of outputs.
        wavelet: Mother wavelet, `mexican_hat` or `morlet`.
        use_base: Whether to keep a residual Dense path alongside the wavelets.
    """

    def __init__(
        self,
        units: int,
        wavelet: str = "mexican_hat",
        use_base: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.units = int(units)
        self.wavelet = wavelet
        self.use_base = use_base

    def build(self, input_shape):
        """Create per-edge translation and scale, plus the output weights."""
        self.in_features = int(input_shape[-1])
        shape = (self.in_features, self.units)
        self.translation = self.add_weight(
            shape=shape, initializer="zeros", trainable=True, name="translation"
        )
        self.scale = self.add_weight(
            shape=shape, initializer="ones", trainable=True, name="scale"
        )
        self.wavelet_weight = self.add_weight(
            shape=shape,
            initializer=keras.initializers.GlorotUniform(),
            trainable=True,
            name="wavelet_weight",
        )
        self.base = (
            keras.layers.Dense(self.units, activation="silu", name="base")
            if self.use_base
            else None
        )
        super().build(input_shape)

    def call(self, inputs):
        """Evaluate the wavelet at each edge's own translation and scale."""
        flat = ops.reshape(inputs, (-1, self.in_features))
        x = (ops.expand_dims(flat, -1) - self.translation) / self.scale

        if self.wavelet == "morlet":
            # Real Morlet: a cosine carrier under a Gaussian envelope.
            response = ops.cos(5.0 * x) * ops.exp(-0.5 * ops.square(x))
        else:
            # Mexican hat, the second derivative of a Gaussian.
            squared = ops.square(x)
            response = (
                (2.0 / (math.sqrt(3.0) * math.pi**0.25))
                * (1.0 - squared)
                * ops.exp(-0.5 * squared)
            )

        output = ops.sum(response * self.wavelet_weight, axis=1)
        if self.base is not None:
            output = output + self.base(flat)
        return ops.reshape(output, (*ops.shape(inputs)[:-1], self.units))

    def compute_output_shape(self, input_shape):
        """Replace the last axis with the output width."""
        return (*input_shape[:-1], self.units)

    def get_config(self):
        """Serialise every constructor argument."""
        return {
            **super().get_config(),
            "units": self.units,
            "wavelet": self.wavelet,
            "use_base": self.use_base,
        }
