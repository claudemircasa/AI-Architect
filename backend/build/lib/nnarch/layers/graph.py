"""
> [!AML-DOC-FILE]
@file       layers/graph.py
@description Graph neural network layers, over a dense adjacency matrix.
@module     nnarch.layers.graph
@exports    GCNConv, GATConv, GraphSAGEConv, GINConv, GlobalAttentionPool
@created    2026-09-30
@context    AML-DOC EXEMPTION (§7): ordinary docstrings, because `export.inline`
            copies these classes into user projects [task 06].

            Every layer here takes two tensors: node features (batch, nodes,
            features) and a dense adjacency (batch, nodes, nodes). Dense adjacency
            costs O(N^2) and real graph libraries use edge lists instead, but it
            keeps the layers plain Keras with no custom data structure, which is
            what lets them be wired on a canvas and exported as ordinary code.
"""

from __future__ import annotations

import keras
from keras import ops


def _normalise_adjacency(adjacency):
    """Symmetrically normalise an adjacency matrix with self-loops added.

    Produces D^-1/2 (A + I) D^-1/2, the propagation rule from the GCN paper,
    which keeps activations from growing with node degree.
    """
    nodes = ops.shape(adjacency)[-1]
    identity = ops.eye(nodes, dtype=adjacency.dtype)
    adjacency = adjacency + identity
    degree = ops.sum(adjacency, axis=-1)
    inverse_sqrt = ops.where(
        degree > 0, ops.rsqrt(ops.maximum(degree, 1e-12)), ops.zeros_like(degree)
    )
    return adjacency * ops.expand_dims(inverse_sqrt, -1) * ops.expand_dims(inverse_sqrt, -2)


@keras.saving.register_keras_serializable(package="custom_layers")
class GCNConv(keras.layers.Layer):
    """Graph convolution: average each node with its neighbours, then project.

    The layer that made graph networks practical. One pass mixes a node with its
    immediate neighbours, so stacking k of them gives each node a view of
    everything within k hops.

    Takes two inputs: node features and a dense adjacency matrix.

    Reference: Kipf & Welling 2016, "Semi-Supervised Classification with Graph
    Convolutional Networks", arXiv:1609.02907.

    Args:
        units: Width of the output features.
        activation: Non-linearity applied after the projection.
        use_bias: Whether the projection carries a bias term.
    """

    def __init__(
        self, units: int, activation: str | None = "relu", use_bias: bool = True, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.units = int(units)
        self.activation = activation
        self.use_bias = use_bias

    def build(self, input_shape):
        """Create the node projection."""
        self.projection = keras.layers.Dense(
            self.units, activation=self.activation, use_bias=self.use_bias, name="project"
        )
        super().build(input_shape)

    def call(self, inputs):
        """Propagate features along normalised edges, then project."""
        nodes, adjacency = inputs
        return self.projection(ops.matmul(_normalise_adjacency(adjacency), nodes))

    def compute_output_shape(self, input_shape):
        """Node count is unchanged; features become `units`."""
        node_shape, _ = input_shape
        return (*node_shape[:-1], self.units)

    def get_config(self):
        """Serialise every constructor argument."""
        return {
            **super().get_config(),
            "units": self.units,
            "activation": self.activation,
            "use_bias": self.use_bias,
        }


@keras.saving.register_keras_serializable(package="custom_layers")
class GATConv(keras.layers.Layer):
    """Graph attention: learn how much each neighbour matters.

    GCN weights neighbours by degree alone. GAT learns the weights, so a node can
    attend to the neighbours that matter for the task. `GATv2` fixes a limitation
    of the original where the ranking of neighbours was the same for every node,
    which the authors called static attention.

    Takes two inputs: node features and a dense adjacency matrix.

    Reference: Veličković et al. 2017, "Graph Attention Networks",
    arXiv:1710.10903; Brody et al. 2021, "How Attentive are Graph Attention
    Networks?", arXiv:2105.14491.

    Args:
        units: Width of each attention head's output.
        num_heads: Number of attention heads.
        concat_heads: Concatenate the heads, or average them. The last layer of
            a network usually averages.
        dropout: Dropout applied to the attention coefficients.
        negative_slope: Slope of the LeakyReLU in the attention score.
        version: `v2` for GATv2, `v1` for the original.
    """

    def __init__(
        self,
        units: int,
        num_heads: int = 4,
        concat_heads: bool = True,
        dropout: float = 0.0,
        negative_slope: float = 0.2,
        version: str = "v2",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.units = int(units)
        self.num_heads = int(num_heads)
        self.concat_heads = concat_heads
        self.dropout = float(dropout)
        self.negative_slope = float(negative_slope)
        self.version = version

    def build(self, input_shape):
        """Create the per-head projection and the attention vector."""
        self.projection = keras.layers.Dense(
            self.units * self.num_heads, use_bias=False, name="project"
        )
        self.attention = self.add_weight(
            shape=(self.num_heads, self.units),
            initializer=keras.initializers.GlorotUniform(),
            trainable=True,
            name="attention",
        )
        self.dropout_layer = (
            keras.layers.Dropout(self.dropout) if self.dropout > 0 else None
        )
        super().build(input_shape)

    def call(self, inputs, training=False):
        """Score every edge, soft-max over neighbours, and aggregate."""
        nodes, adjacency = inputs
        shape = ops.shape(nodes)
        batch, count = shape[0], shape[1]

        projected = ops.reshape(
            self.projection(nodes), (batch, count, self.num_heads, self.units)
        )
        source = ops.expand_dims(projected, 2)
        target = ops.expand_dims(projected, 1)

        if self.version == "v2":
            # GATv2 applies the non-linearity before the attention vector, which
            # is what makes the ranking depend on the querying node.
            scores = ops.sum(
                ops.cast(self.attention, projected.dtype)
                * keras.activations.leaky_relu(
                    source + target, negative_slope=self.negative_slope
                ),
                axis=-1,
            )
        else:
            scores = keras.activations.leaky_relu(
                ops.sum(
                    ops.cast(self.attention, projected.dtype) * (source + target), axis=-1
                ),
                negative_slope=self.negative_slope,
            )

        mask = ops.expand_dims(adjacency > 0, -1)
        scores = ops.where(mask, scores, ops.full_like(scores, -1e9))
        weights = ops.softmax(scores, axis=2)
        if self.dropout_layer is not None:
            weights = self.dropout_layer(weights, training=training)

        attended = ops.einsum("bijh,bjhu->bihu", weights, projected)
        if self.concat_heads:
            return ops.reshape(attended, (batch, count, self.num_heads * self.units))
        return ops.mean(attended, axis=2)

    def compute_output_shape(self, input_shape):
        """Features become `units` times the head count, or just `units` if averaged."""
        node_shape, _ = input_shape
        width = self.units * self.num_heads if self.concat_heads else self.units
        return (*node_shape[:-1], width)

    def get_config(self):
        """Serialise every constructor argument."""
        return {
            **super().get_config(),
            "units": self.units,
            "num_heads": self.num_heads,
            "concat_heads": self.concat_heads,
            "dropout": self.dropout,
            "negative_slope": self.negative_slope,
            "version": self.version,
        }


@keras.saving.register_keras_serializable(package="custom_layers")
class GraphSAGEConv(keras.layers.Layer):
    """Keeps a node's own features separate from its neighbours' summary.

    GCN mixes a node into the same average as its neighbours. GraphSAGE
    concatenates the two instead, so the node's own identity survives the
    aggregation, which matters when node features are informative on their own.

    Takes two inputs: node features and a dense adjacency matrix.

    Reference: Hamilton et al. 2017, "Inductive Representation Learning on Large
    Graphs", arXiv:1706.02216.

    Args:
        units: Width of the output features.
        aggregator: How neighbours are summarised: `mean`, `max` or `sum`.
        activation: Non-linearity applied after the projection.
        normalize: L2-normalise the output, as the paper does.
    """

    def __init__(
        self,
        units: int,
        aggregator: str = "mean",
        activation: str | None = "relu",
        normalize: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.units = int(units)
        self.aggregator = aggregator
        self.activation = activation
        self.normalize = normalize

    def build(self, input_shape):
        """Create the projection over the concatenated self and neighbour features."""
        self.projection = keras.layers.Dense(
            self.units, activation=self.activation, name="project"
        )
        super().build(input_shape)

    def call(self, inputs):
        """Summarise the neighbours, concatenate with self, then project."""
        nodes, adjacency = inputs
        adjacency = ops.cast(adjacency, nodes.dtype)

        if self.aggregator == "max":
            masked = ops.expand_dims(adjacency, -1) * ops.expand_dims(nodes, 1)
            neighbours = ops.max(
                ops.where(
                    ops.expand_dims(adjacency, -1) > 0, masked, ops.full_like(masked, -1e9)
                ),
                axis=2,
            )
            neighbours = ops.where(
                ops.expand_dims(ops.sum(adjacency, -1) > 0, -1),
                neighbours,
                ops.zeros_like(neighbours),
            )
        else:
            summed = ops.matmul(adjacency, nodes)
            if self.aggregator == "mean":
                degree = ops.maximum(ops.sum(adjacency, axis=-1, keepdims=True), 1.0)
                neighbours = summed / degree
            else:
                neighbours = summed

        output = self.projection(ops.concatenate([nodes, neighbours], axis=-1))
        if self.normalize:
            output = output / (
                ops.sqrt(ops.sum(ops.square(output), axis=-1, keepdims=True)) + 1e-9
            )
        return output

    def compute_output_shape(self, input_shape):
        """Node count is unchanged; features become `units`."""
        node_shape, _ = input_shape
        return (*node_shape[:-1], self.units)

    def get_config(self):
        """Serialise every constructor argument."""
        return {
            **super().get_config(),
            "units": self.units,
            "aggregator": self.aggregator,
            "activation": self.activation,
            "normalize": self.normalize,
        }


@keras.saving.register_keras_serializable(package="custom_layers")
class GINConv(keras.layers.Layer):
    """The most expressive message passing a graph network can do.

    The paper proves that sum aggregation followed by an MLP is as powerful as
    the Weisfeiler-Lehman graph isomorphism test, and that mean and max
    aggregation are strictly weaker: they cannot tell apart graphs that differ
    only in multiplicity.

    Takes two inputs: node features and a dense adjacency matrix.

    Reference: Xu et al. 2018, "How Powerful are Graph Neural Networks?",
    arXiv:1810.00826.

    Args:
        units: Width of the MLP's output.
        hidden_units: Width of the MLP's hidden layer. Defaults to `units`.
        epsilon: Initial extra weight on a node's own features.
        train_epsilon: Whether that weight is learned.
    """

    def __init__(
        self,
        units: int,
        hidden_units: int | None = None,
        epsilon: float = 0.0,
        train_epsilon: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.units = int(units)
        self.hidden_units = hidden_units
        self.epsilon = float(epsilon)
        self.train_epsilon = train_epsilon

    def build(self, input_shape):
        """Create the MLP and the self-weight."""
        hidden = self.hidden_units if self.hidden_units else self.units
        self.mlp = keras.Sequential(
            [
                keras.layers.Dense(hidden, activation="relu"),
                keras.layers.BatchNormalization(),
                keras.layers.Dense(self.units, activation="relu"),
            ],
            name="mlp",
        )
        self.eps = self.add_weight(
            shape=(),
            initializer=keras.initializers.Constant(self.epsilon),
            trainable=self.train_epsilon,
            name="epsilon",
        )
        super().build(input_shape)

    def call(self, inputs, training=False):
        """Sum the neighbours, add the weighted self, then apply the MLP."""
        nodes, adjacency = inputs
        aggregated = ops.matmul(ops.cast(adjacency, nodes.dtype), nodes)
        combined = (1.0 + ops.cast(self.eps, nodes.dtype)) * nodes + aggregated
        return self.mlp(combined, training=training)

    def compute_output_shape(self, input_shape):
        """Node count is unchanged; features become `units`."""
        node_shape, _ = input_shape
        return (*node_shape[:-1], self.units)

    def get_config(self):
        """Serialise every constructor argument."""
        return {
            **super().get_config(),
            "units": self.units,
            "hidden_units": self.hidden_units,
            "epsilon": self.epsilon,
            "train_epsilon": self.train_epsilon,
        }


@keras.saving.register_keras_serializable(package="custom_layers")
class GlobalAttentionPool(keras.layers.Layer):
    """Pools a set of nodes into one vector, weighting each node by a learned gate.

    Mean or max pooling treats every node alike. This learns which nodes carry
    the graph's label, which is usually a small subset.

    Takes one input: node features.

    Reference: Li et al. 2015, "Gated Graph Sequence Neural Networks",
    arXiv:1511.05493.

    Args:
        units: Width of the pooled output.
    """

    def __init__(self, units: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.units = int(units)

    def build(self, input_shape):
        """Create the gate and the value projection."""
        self.gate = keras.layers.Dense(1, activation="sigmoid", name="gate")
        self.value = keras.layers.Dense(self.units, name="value")
        super().build(input_shape)

    def call(self, inputs):
        """Weight each node by its gate, then sum."""
        return ops.sum(self.gate(inputs) * self.value(inputs), axis=1)

    def compute_output_shape(self, input_shape):
        """The node axis is collapsed."""
        return (input_shape[0], self.units)

    def get_config(self):
        """Serialise every constructor argument."""
        return {**super().get_config(), "units": self.units}
