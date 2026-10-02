"""
> [!AML-DOC-FILE]
@file       layers/conv_mod.py
@description Modern convolutional blocks: the designs that kept convolution
             competitive with vision transformers.
@module     nnarch.layers.conv_mod
@exports    DropPath, SqueezeExcite, ConvNeXtBlock, MBConv, Involution
@created    2026-09-30
@context    AML-DOC EXEMPTION (§7): ordinary docstrings, because `export.inline`
            copies these classes into user projects [task 06]. No import from
            `nnarch` is permitted here.
"""

from __future__ import annotations

import keras
from keras import ops


@keras.saving.register_keras_serializable(package="custom_layers")
class DropPath(keras.layers.Layer):
    """Stochastic depth: drops an entire residual branch for a whole sample.

    Unlike dropout, which zeroes individual activations, this removes the branch
    for the sample altogether, so the network trains as an ensemble of shallower
    networks. Standard in ConvNeXt, ViT and EfficientNet training recipes.

    Reference: Huang et al. 2016, "Deep Networks with Stochastic Depth",
    arXiv:1603.09382.

    Args:
        rate: Probability of dropping the branch for a given sample.
        seed: Seed for reproducible sampling.
    """

    def __init__(self, rate: float = 0.1, seed: int | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.rate = float(rate)
        self.seed = seed
        self.seed_generator = keras.random.SeedGenerator(seed)

    def call(self, inputs, training=False):
        """Zero the whole sample with probability `rate`, rescaling what survives."""
        if not training or self.rate == 0.0:
            return inputs
        keep = 1.0 - self.rate
        shape = [ops.shape(inputs)[0]] + [1] * (len(inputs.shape) - 1)
        mask = keras.random.uniform(shape, seed=self.seed_generator) < keep
        return inputs / keep * ops.cast(mask, inputs.dtype)

    def compute_output_shape(self, input_shape):
        """Shape is unchanged."""
        return input_shape

    def get_config(self):
        """Serialise every constructor argument."""
        return {**super().get_config(), "rate": self.rate, "seed": self.seed}


@keras.saving.register_keras_serializable(package="custom_layers")
class SqueezeExcite(keras.layers.Layer):
    """Channel attention: learns which feature maps matter for this input.

    Pools each channel to a single number, passes those through a bottleneck, and
    uses the result to rescale the channels. Cheap, and it won ILSVRC 2017.

    Reference: Hu et al. 2017, "Squeeze-and-Excitation Networks",
    arXiv:1709.01507.

    Args:
        ratio: Bottleneck width as a fraction of the channel count.
        activation: Non-linearity inside the bottleneck.
        gate_activation: Non-linearity producing the channel weights.
    """

    def __init__(
        self,
        ratio: float = 0.25,
        activation: str = "relu",
        gate_activation: str = "sigmoid",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.ratio = float(ratio)
        self.activation = activation
        self.gate_activation = gate_activation

    def build(self, input_shape):
        """Create the bottleneck sized from the channel count."""
        channels = int(input_shape[-1])
        hidden = max(1, int(channels * self.ratio))
        self.reduce = keras.layers.Dense(hidden, activation=self.activation, name="reduce")
        self.expand = keras.layers.Dense(
            channels, activation=self.gate_activation, name="expand"
        )
        self.spatial_axes = tuple(range(1, len(input_shape) - 1))
        super().build(input_shape)

    def call(self, inputs):
        """Pool, bottleneck, and rescale the channels."""
        pooled = ops.mean(inputs, axis=self.spatial_axes, keepdims=True)
        weights = self.expand(self.reduce(pooled))
        return inputs * weights

    def compute_output_shape(self, input_shape):
        """Shape is unchanged."""
        return input_shape

    def get_config(self):
        """Serialise every constructor argument."""
        return {
            **super().get_config(),
            "ratio": self.ratio,
            "activation": self.activation,
            "gate_activation": self.gate_activation,
        }


@keras.saving.register_keras_serializable(package="custom_layers")
class ConvNeXtBlock(keras.layers.Layer):
    """The ConvNeXt residual block: a convnet rebuilt with transformer habits.

    A 7x7 depthwise convolution for mixing space, layer normalisation, then an
    inverted bottleneck MLP mixing channels, with a learned per-channel scale on
    the residual. Matches vision transformers on ImageNet without attention.

    Reference: Liu et al. 2022, "A ConvNet for the 2020s", arXiv:2201.03545.

    Args:
        kernel_size: Size of the depthwise convolution window.
        expansion: Width multiplier of the inverted bottleneck.
        drop_path: Stochastic depth rate for the residual branch.
        layer_scale: Initial value of the learned residual scale. Use 0 to
            disable it.
    """

    def __init__(
        self,
        kernel_size: int = 7,
        expansion: int = 4,
        drop_path: float = 0.0,
        layer_scale: float = 1e-6,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.kernel_size = int(kernel_size)
        self.expansion = int(expansion)
        self.drop_path = float(drop_path)
        self.layer_scale = float(layer_scale)

    def build(self, input_shape):
        """Create the depthwise convolution and the inverted bottleneck."""
        channels = int(input_shape[-1])
        self.depthwise = keras.layers.DepthwiseConv2D(
            self.kernel_size, padding="same", name="depthwise"
        )
        self.norm = keras.layers.LayerNormalization(epsilon=1e-6, name="norm")
        self.expand = keras.layers.Dense(
            channels * self.expansion, activation="gelu", name="expand"
        )
        self.project = keras.layers.Dense(channels, name="project")
        self.stochastic_depth = DropPath(self.drop_path) if self.drop_path > 0 else None
        if self.layer_scale > 0:
            self.scale = self.add_weight(
                shape=(channels,),
                initializer=keras.initializers.Constant(self.layer_scale),
                trainable=True,
                name="layer_scale",
            )
        else:
            self.scale = None
        super().build(input_shape)

    def call(self, inputs, training=False):
        """Mix space depthwise, mix channels through the bottleneck, then add back."""
        residual = self.project(self.expand(self.norm(self.depthwise(inputs))))
        if self.scale is not None:
            residual = residual * self.scale
        if self.stochastic_depth is not None:
            residual = self.stochastic_depth(residual, training=training)
        return inputs + residual

    def compute_output_shape(self, input_shape):
        """Shape is unchanged; the block is residual."""
        return input_shape

    def get_config(self):
        """Serialise every constructor argument."""
        return {
            **super().get_config(),
            "kernel_size": self.kernel_size,
            "expansion": self.expansion,
            "drop_path": self.drop_path,
            "layer_scale": self.layer_scale,
        }


@keras.saving.register_keras_serializable(package="custom_layers")
class MBConv(keras.layers.Layer):
    """Inverted residual block with squeeze-and-excitation.

    Expands the channels, convolves depthwise, recalibrates with channel
    attention, then projects back down. The building block of MobileNetV2 and
    EfficientNet, and the reason both are cheap at a given accuracy.

    Reference: Sandler et al. 2018, "MobileNetV2", arXiv:1801.04381; Tan & Le
    2019, "EfficientNet", arXiv:1905.11946.

    Args:
        filters: Output channel count.
        expansion: Width multiplier of the inner depthwise stage.
        kernel_size: Size of the depthwise convolution window.
        strides: Stride of the depthwise convolution. A stride above 1 disables
            the residual connection, since the shapes no longer match.
        se_ratio: Squeeze-and-excitation bottleneck ratio. Use 0 to omit it.
        drop_path: Stochastic depth rate, applied only when residual.
    """

    def __init__(
        self,
        filters: int,
        expansion: int = 4,
        kernel_size: int = 3,
        strides: int = 1,
        se_ratio: float = 0.25,
        drop_path: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.filters = int(filters)
        self.expansion = int(expansion)
        self.kernel_size = int(kernel_size)
        self.strides = int(strides)
        self.se_ratio = float(se_ratio)
        self.drop_path = float(drop_path)

    def build(self, input_shape):
        """Create the expand, depthwise, attention and project stages."""
        channels = int(input_shape[-1])
        hidden = channels * self.expansion
        self.residual = self.strides == 1 and channels == self.filters

        self.expand = (
            keras.Sequential(
                [
                    keras.layers.Conv2D(hidden, 1, use_bias=False),
                    keras.layers.BatchNormalization(),
                    keras.layers.Activation("swish"),
                ],
                name="expand",
            )
            if self.expansion != 1
            else None
        )
        self.depthwise = keras.Sequential(
            [
                keras.layers.DepthwiseConv2D(
                    self.kernel_size, strides=self.strides, padding="same", use_bias=False
                ),
                keras.layers.BatchNormalization(),
                keras.layers.Activation("swish"),
            ],
            name="depthwise",
        )
        self.excite = SqueezeExcite(self.se_ratio) if self.se_ratio > 0 else None
        self.project = keras.Sequential(
            [
                keras.layers.Conv2D(self.filters, 1, use_bias=False),
                keras.layers.BatchNormalization(),
            ],
            name="project",
        )
        self.stochastic_depth = (
            DropPath(self.drop_path) if self.drop_path > 0 and self.residual else None
        )
        super().build(input_shape)

    def call(self, inputs, training=False):
        """Expand, convolve depthwise, recalibrate, project, and add back if possible."""
        x = inputs if self.expand is None else self.expand(inputs, training=training)
        x = self.depthwise(x, training=training)
        if self.excite is not None:
            x = self.excite(x)
        x = self.project(x, training=training)
        if self.stochastic_depth is not None:
            x = self.stochastic_depth(x, training=training)
        return inputs + x if self.residual else x

    def compute_output_shape(self, input_shape):
        """Spatial size follows the stride; the channel count becomes `filters`."""
        batch, height, width, _ = input_shape
        if self.strides > 1:
            height = None if height is None else (height + self.strides - 1) // self.strides
            width = None if width is None else (width + self.strides - 1) // self.strides
        return (batch, height, width, self.filters)

    def get_config(self):
        """Serialise every constructor argument."""
        return {
            **super().get_config(),
            "filters": self.filters,
            "expansion": self.expansion,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "se_ratio": self.se_ratio,
            "drop_path": self.drop_path,
        }


@keras.saving.register_keras_serializable(package="custom_layers")
class Involution(keras.layers.Layer):
    """Convolution turned inside out: kernels generated per position, shared across channels.

    A convolution uses one kernel everywhere and a different one per channel.
    Involution inverts both: the kernel is produced from the pixel itself, so it
    varies across space, and it is shared across channels within a group. Fewer
    parameters, and a receptive field that adapts to content.

    Reference: Li et al. 2021, "Involution: Inverting the Inherence of
    Convolution for Visual Recognition", arXiv:2103.06255.

    Args:
        kernel_size: Size of the generated window.
        groups: Number of channel groups sharing a kernel.
        reduction_ratio: Bottleneck ratio of the kernel-generating branch.
        strides: Stride of the operation.
    """

    def __init__(
        self,
        kernel_size: int = 3,
        groups: int = 1,
        reduction_ratio: int = 4,
        strides: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.kernel_size = int(kernel_size)
        self.groups = int(groups)
        self.reduction_ratio = int(reduction_ratio)
        self.strides = int(strides)

    def build(self, input_shape):
        """Create the kernel-generating branch and the patch extractor."""
        self.channels = int(input_shape[-1])
        hidden = max(1, self.channels // self.reduction_ratio)
        self.pool = (
            keras.layers.AveragePooling2D(self.strides, self.strides, padding="same")
            if self.strides > 1
            else None
        )
        self.reduce = keras.layers.Conv2D(hidden, 1, name="reduce")
        self.norm = keras.layers.BatchNormalization(name="norm")
        self.span = keras.layers.Conv2D(
            self.kernel_size * self.kernel_size * self.groups, 1, name="span"
        )
        super().build(input_shape)

    def call(self, inputs):
        """Generate a kernel per position, then apply it to that position's patch."""
        source = inputs if self.pool is None else self.pool(inputs)
        kernel = self.span(keras.activations.relu(self.norm(self.reduce(source))))

        shape = ops.shape(kernel)
        batch, height, width = shape[0], shape[1], shape[2]
        kernel = ops.reshape(
            kernel, (batch, height, width, self.kernel_size**2, self.groups, 1)
        )

        patches = ops.image.extract_patches(
            inputs,
            size=(self.kernel_size, self.kernel_size),
            strides=(self.strides, self.strides),
            dilation_rate=1,
            padding="same",
        )
        patches = ops.reshape(
            patches,
            (
                batch,
                height,
                width,
                self.kernel_size**2,
                self.groups,
                self.channels // self.groups,
            ),
        )

        output = ops.sum(patches * kernel, axis=3)
        return ops.reshape(output, (batch, height, width, self.channels))

    def compute_output_shape(self, input_shape):
        """Spatial size follows the stride; the channel count is unchanged."""
        batch, height, width, channels = input_shape
        if self.strides > 1:
            height = None if height is None else (height + self.strides - 1) // self.strides
            width = None if width is None else (width + self.strides - 1) // self.strides
        return (batch, height, width, channels)

    def get_config(self):
        """Serialise every constructor argument."""
        return {
            **super().get_config(),
            "kernel_size": self.kernel_size,
            "groups": self.groups,
            "reduction_ratio": self.reduction_ratio,
            "strides": self.strides,
        }
