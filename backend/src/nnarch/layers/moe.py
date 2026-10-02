"""
> [!AML-DOC-FILE]
@file       layers/moe.py
@description Mixture-of-experts blocks: many feed-forward experts, of which each
             token uses only a few.
@module     nnarch.layers.moe
@exports    SparseMoE, SoftMoE
@created    2026-09-30
@context    AML-DOC EXEMPTION (§7): ordinary docstrings, because `export.inline`
            copies these classes into user projects [task 06].

            Both run every expert over every token and mask afterwards. That is the
            honest shape for a prototyping tool: real sparse dispatch needs a
            gather/scatter kernel, and the saving only appears at a scale this tool
            is not for. The parameter count, the routing behaviour and the exported
            code are all faithful; only the speed differs.
"""

from __future__ import annotations

import keras
from keras import ops


@keras.saving.register_keras_serializable(package="custom_layers")
class SparseMoE(keras.layers.Layer):
    """Routes each token to its top-k experts out of many.

    Capacity grows with the expert count while the work per token stays fixed,
    which is how Mixtral and DeepSeek reach hundreds of billions of parameters at
    the cost of a much smaller dense model. A router scores the experts per
    token, the best `top_k` run, and their outputs are combined by the router's
    own weights.

    An auxiliary loss pushes the router to spread load evenly. Without it routing
    collapses: a few experts win early, get all the gradient, and the rest never
    train.

    Reference: Shazeer et al. 2017, "Outrageously Large Neural Networks",
    arXiv:1701.06538; Fedus et al. 2021, "Switch Transformers",
    arXiv:2101.03961. Set `top_k=1` for Switch routing.

    Args:
        num_experts: How many experts to hold.
        expert_dim: Hidden width inside each expert.
        top_k: How many experts each token uses.
        activation: Non-linearity inside the experts.
        balance_loss_weight: Strength of the load-balancing auxiliary loss.
        router_noise: Standard deviation of noise added to the router's scores
            during training, which helps exploration early on.
    """

    def __init__(
        self,
        num_experts: int = 8,
        expert_dim: int = 256,
        top_k: int = 2,
        activation: str = "gelu",
        balance_loss_weight: float = 0.01,
        router_noise: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.num_experts = int(num_experts)
        self.expert_dim = int(expert_dim)
        self.top_k = int(top_k)
        self.activation = activation
        self.balance_loss_weight = float(balance_loss_weight)
        self.router_noise = float(router_noise)

    def build(self, input_shape):
        """Create the router and one expert per slot, batched into single tensors."""
        width = int(input_shape[-1])
        self.router = keras.layers.Dense(self.num_experts, use_bias=False, name="router")
        # The experts live in one stacked tensor rather than a list of Dense
        # layers, so the forward pass is two einsums instead of `num_experts`
        # separate matmuls.
        self.expert_in = self.add_weight(
            shape=(self.num_experts, width, self.expert_dim),
            initializer=keras.initializers.GlorotUniform(),
            trainable=True,
            name="expert_in",
        )
        self.expert_in_bias = self.add_weight(
            shape=(self.num_experts, self.expert_dim),
            initializer="zeros",
            trainable=True,
            name="expert_in_bias",
        )
        self.expert_out = self.add_weight(
            shape=(self.num_experts, self.expert_dim, width),
            initializer=keras.initializers.GlorotUniform(),
            trainable=True,
            name="expert_out",
        )
        self.expert_out_bias = self.add_weight(
            shape=(self.num_experts, width),
            initializer="zeros",
            trainable=True,
            name="expert_out_bias",
        )
        self.activation_fn = keras.activations.get(self.activation)
        self.seed_generator = keras.random.SeedGenerator(None)
        super().build(input_shape)

    def call(self, inputs, training=False):
        """Score the experts, keep the best few per token, and combine their outputs."""
        logits = self.router(inputs)
        if training and self.router_noise > 0:
            logits = logits + keras.random.normal(
                ops.shape(logits), stddev=self.router_noise, seed=self.seed_generator
            )
        probabilities = ops.softmax(logits, axis=-1)

        top_values, top_indices = ops.top_k(probabilities, k=self.top_k)
        # Renormalise over the chosen experts so the combination is a convex one.
        top_values = top_values / (ops.sum(top_values, axis=-1, keepdims=True) + 1e-9)

        mask = ops.sum(
            ops.one_hot(top_indices, self.num_experts, dtype=probabilities.dtype)
            * ops.expand_dims(top_values, -1),
            axis=-2,
        )

        hidden = self.activation_fn(
            ops.einsum("...d,edh->...eh", inputs, ops.cast(self.expert_in, inputs.dtype))
            + ops.cast(self.expert_in_bias, inputs.dtype)
        )
        expert_output = ops.einsum(
            "...eh,ehd->...ed", hidden, ops.cast(self.expert_out, inputs.dtype)
        ) + ops.cast(self.expert_out_bias, inputs.dtype)

        if training and self.balance_loss_weight > 0:
            self.add_loss(self._balance_loss(probabilities, top_indices))

        return ops.sum(expert_output * ops.expand_dims(mask, -1), axis=-2)

    def _balance_loss(self, probabilities, top_indices):
        """Penalise routing that concentrates tokens on a few experts.

        The Switch Transformer formulation: the dot product of the fraction of
        tokens dispatched to each expert with the mean router probability for
        that expert, which is minimised when both are uniform.
        """
        axes = tuple(range(len(probabilities.shape) - 1))
        dispatched = ops.mean(
            ops.max(
                ops.one_hot(top_indices, self.num_experts, dtype=probabilities.dtype),
                axis=-2,
            ),
            axis=axes,
        )
        mean_probability = ops.mean(probabilities, axis=axes)
        return (
            self.balance_loss_weight
            * self.num_experts
            * ops.sum(dispatched * mean_probability)
        )

    def compute_output_shape(self, input_shape):
        """Shape is unchanged; experts map the width back to itself."""
        return input_shape

    def get_config(self):
        """Serialise every constructor argument."""
        return {
            **super().get_config(),
            "num_experts": self.num_experts,
            "expert_dim": self.expert_dim,
            "top_k": self.top_k,
            "activation": self.activation,
            "balance_loss_weight": self.balance_loss_weight,
            "router_noise": self.router_noise,
        }


@keras.saving.register_keras_serializable(package="custom_layers")
class SoftMoE(keras.layers.Layer):
    """Mixture of experts with no discrete routing at all.

    Each expert receives a weighted average of the tokens, over learned slots,
    and the outputs are weighted back onto the tokens. Because nothing is
    chosen, there is no load-balancing loss, no token dropping and no
    non-differentiable step, which makes it far better behaved than top-k
    routing at the cost of being dense.

    Reference: Puigcerver et al. 2023, "From Sparse to Soft Mixtures of
    Experts", arXiv:2308.00951.

    Args:
        num_experts: How many experts to hold.
        expert_dim: Hidden width inside each expert.
        slots_per_expert: Input slots each expert receives.
        activation: Non-linearity inside the experts.
    """

    def __init__(
        self,
        num_experts: int = 8,
        expert_dim: int = 256,
        slots_per_expert: int = 1,
        activation: str = "gelu",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.num_experts = int(num_experts)
        self.expert_dim = int(expert_dim)
        self.slots_per_expert = int(slots_per_expert)
        self.activation = activation

    def build(self, input_shape):
        """Create the slot parameters and the stacked experts."""
        width = int(input_shape[-1])
        self.slots = self.num_experts * self.slots_per_expert
        self.phi = self.add_weight(
            shape=(width, self.slots),
            initializer=keras.initializers.GlorotUniform(),
            trainable=True,
            name="phi",
        )
        self.expert_in = self.add_weight(
            shape=(self.num_experts, width, self.expert_dim),
            initializer=keras.initializers.GlorotUniform(),
            trainable=True,
            name="expert_in",
        )
        self.expert_out = self.add_weight(
            shape=(self.num_experts, self.expert_dim, width),
            initializer=keras.initializers.GlorotUniform(),
            trainable=True,
            name="expert_out",
        )
        self.activation_fn = keras.activations.get(self.activation)
        super().build(input_shape)

    def call(self, inputs):
        """Mix tokens into slots, run the experts, and mix the results back."""
        logits = ops.einsum("bld,ds->bls", inputs, ops.cast(self.phi, inputs.dtype))

        dispatch = ops.softmax(logits, axis=1)
        slot_inputs = ops.einsum("bld,bls->bsd", inputs, dispatch)
        slot_inputs = ops.reshape(
            slot_inputs,
            (-1, self.num_experts, self.slots_per_expert, ops.shape(inputs)[-1]),
        )

        hidden = self.activation_fn(
            ops.einsum(
                "besd,edh->besh", slot_inputs, ops.cast(self.expert_in, inputs.dtype)
            )
        )
        slot_outputs = ops.einsum(
            "besh,ehd->besd", hidden, ops.cast(self.expert_out, inputs.dtype)
        )
        slot_outputs = ops.reshape(
            slot_outputs, (-1, self.slots, ops.shape(inputs)[-1])
        )

        combine = ops.softmax(logits, axis=-1)
        return ops.einsum("bls,bsd->bld", combine, slot_outputs)

    def compute_output_shape(self, input_shape):
        """Shape is unchanged."""
        return input_shape

    def get_config(self):
        """Serialise every constructor argument."""
        return {
            **super().get_config(),
            "num_experts": self.num_experts,
            "expert_dim": self.expert_dim,
            "slots_per_expert": self.slots_per_expert,
            "activation": self.activation,
        }
