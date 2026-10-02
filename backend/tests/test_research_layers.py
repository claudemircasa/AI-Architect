"""
> [!AML-DOC-FILE]
@file       tests/test_research_layers.py
@description Regression tests for the research-frontier layers: that each one
             trains, serialises, and leaves the tool behind when exported.
@module     tests.test_research_layers
@exports    (pytest test functions)
@created    2026-09-30
@context    Covers the task 04 verification gates. The causality tests are the ones
            that matter most: a sequence model that leaks the future still trains
            happily and only fails once someone tries to generate with it.
"""

from __future__ import annotations

import ast

import keras
import numpy as np
import pytest

from nnarch.catalog import Registry, bootstrap
from nnarch.catalog.introspect import resolve_keras_class
from nnarch.export import DatasetChoice, ExportOptions, export_project
from nnarch.ir import GraphIR, compile_graph
from nnarch.ir.schema import Edge, Node


@pytest.fixture(scope="session")
def registry() -> Registry:
    """
    > [!AML-DOC-UNIT]
    The bootstrapped catalog, including the research families.
    @returns the shared Registry
    """
    return bootstrap()


def node(node_id: str, node_type: str, **params: object) -> Node:
    """
    > [!AML-DOC-UNIT]
    Build a graph node concisely.
    @param node_id   graph-unique id, also used as the label
    @param node_type catalog layer id
    @param params    constructor parameters
    @returns the Node
    """
    return Node(id=node_id, type=node_type, name=node_id, params=dict(params))


def edge(source: str, target: str) -> Edge:
    """
    > [!AML-DOC-UNIT]
    Build a graph edge with a generated id.
    @param source id of the producing node
    @param target id of the consuming node
    @returns the Edge
    """
    return Edge(id=f"{source}->{target}", source=source, target=target)


def test_every_research_layer_is_cited_and_resolvable(registry: Registry) -> None:
    """
    > [!AML-DOC-UNIT]
    Every research layer must name the paper it implements and resolve to a real
    class. A layer without a citation is not reproducible by the person using it.
    """
    research = [spec for spec in registry.all() if spec.is_research]
    assert len(research) >= 40, f"only {len(research)} research layers registered"

    uncited = [spec.id for spec in research if not spec.paper]
    assert not uncited, f"research layers without a citation: {uncited}"

    for spec in research:
        resolve_keras_class(spec.keras_path)


def test_every_research_layer_round_trips_its_config(registry: Registry) -> None:
    """
    > [!AML-DOC-UNIT]
    `from_config(get_config())` must reproduce the configuration exactly, or a saved
    project reopens as a different model.
    """
    failures: list[str] = []
    for spec in registry.all():
        if not spec.is_research:
            continue
        cls = resolve_keras_class(spec.keras_path)
        try:
            layer = cls(**spec.defaults())
            config = layer.get_config()
            if cls.from_config(config).get_config() != config:
                failures.append(f"{spec.id}: config differs after round trip")
        except Exception as exc:
            failures.append(f"{spec.id}: {type(exc).__name__}: {exc}")
    assert not failures, "\n".join(failures)


@pytest.mark.parametrize(
    "layer_id,params",
    [
        ("research.MambaBlock", {"state_dim": 8}),
        ("research.Retention", {"num_heads": 2, "head_dim": 8}),
        ("research.MatrixLSTM", {"num_heads": 2, "head_dim": 8}),
        ("research.ALiBiAttention", {"num_heads": 2, "head_dim": 8}),
        ("research.MultiQueryAttention", {"num_heads": 2, "head_dim": 8}),
        ("research.MultiHeadLatentAttention",
         {"num_heads": 2, "head_dim": 8, "rope_dim": 4, "kv_latent_dim": 16}),
    ],
)
def test_sequence_layers_do_not_see_the_future(
    registry: Registry, layer_id: str, params: dict
) -> None:
    """
    > [!AML-DOC-UNIT]
    A causal layer's output at time t must not change when the input after t does.
    @context This is the gate that found a real bug: `Retention` applied
             GroupNormalization across the time axis, which mixed every position
             together. It trained perfectly well and would only have failed once
             someone used it to generate.
    """
    cls = resolve_keras_class(registry.require(layer_id).keras_path)
    layer = cls(**params)
    inputs = keras.Input((12, 16))
    model = keras.Model(inputs, layer(inputs))

    original = np.random.randn(1, 12, 16).astype("float32")
    perturbed = original.copy()
    perturbed[0, 6:, :] += 5.0

    before = np.abs(model.predict(original, verbose=0)[:, :6]
                    - model.predict(perturbed, verbose=0)[:, :6]).max()
    after = np.abs(model.predict(original, verbose=0)[:, 6:]
                   - model.predict(perturbed, verbose=0)[:, 6:]).max()

    assert before < 1e-5, f"{layer_id} leaks the future: output changed by {before}"
    assert after > 1e-3, f"{layer_id} ignored the change entirely: {after}"


def test_spiking_neurons_emit_actual_spikes(registry: Registry) -> None:
    """
    > [!AML-DOC-UNIT]
    A spiking layer must output discrete events, not a smooth approximation of
    them. The surrogate gradient belongs in the backward pass only.
    """
    from nnarch.layers.neuro import PoissonEncoder, SpikingLIF

    inputs = keras.Input((32,))
    spikes = SpikingLIF(units=16)(PoissonEncoder(timesteps=8)(inputs))
    model = keras.Model(inputs, spikes)

    output = np.asarray(
        keras.ops.convert_to_numpy(model(np.random.rand(4, 32).astype("float32"),
                                         training=True))
    )
    assert set(np.unique(output).tolist()) <= {0.0, 1.0}, "spikes must be binary"


def test_lora_freezes_the_base_and_starts_as_a_no_op() -> None:
    """
    > [!AML-DOC-UNIT]
    LoRA's whole point is that the base weights do not move and the adapter starts
    at zero, so the layer begins identical to the frozen one it adapts.
    """
    from nnarch.layers.misc import LoRADense

    layer = LoRADense(units=16, rank=4)
    inputs = keras.Input((8,))
    model = keras.Model(inputs, layer(inputs))

    assert [weight.name for weight in layer.non_trainable_weights] == ["kernel"]
    assert "lora_a" in [weight.name for weight in layer.trainable_weights]

    sample = np.random.randn(4, 8).astype("float32")
    base_only = np.asarray(
        keras.ops.convert_to_numpy(keras.ops.matmul(sample, layer.kernel))
    )
    assert np.allclose(np.asarray(model.predict(sample, verbose=0)), base_only, atol=1e-5)


def test_frontier_model_exports_and_carries_only_what_it_uses(registry: Registry) -> None:
    """
    > [!AML-DOC-UNIT]
    The task 04 export gate: a model built from research layers must generate a
    project that carries their source, excludes the layers it does not use, and
    mentions this tool nowhere.
    """
    graph = GraphIR(name="Frontier Net", nodes=[
        node("tokens", "keras.Input", shape=[32], dtype="int32"),
        node("embed", "keras.Embedding", input_dim=1000, output_dim=32),
        node("mamba", "research.MambaBlock", state_dim=8),
        node("moe", "research.SparseMoE", num_experts=4, expert_dim=32, top_k=2),
        node("pool", "keras.GlobalAveragePooling1D"),
        node("kan", "research.DenseKAN", units=16),
        node("head", "keras.Dense", units=4, activation="softmax"),
    ], edges=[edge("tokens", "embed"), edge("embed", "mamba"), edge("mamba", "moe"),
              edge("moe", "pool"), edge("pool", "kan"), edge("kan", "head")])

    bundle = export_project(
        graph, registry,
        ExportOptions(dataset=DatasetChoice(kind="synthetic", samples=32), epochs=1),
    )
    assert bundle.ok, [d.message for d in bundle.diagnostics]
    assert bundle.params == compile_graph(graph, registry).params_total

    custom = bundle.files["custom_layers.py"]
    ast.parse(custom, filename="custom_layers.py")

    for carried in ("class MambaBlock", "class SparseMoE", "class DenseKAN",
                    "def _linear_recurrence"):
        assert carried in custom, f"{carried} must travel with the export"
    for excluded in ("S4DBlock", "SoftMoE", "FastKAN", "WaveletKAN", "Retention"):
        assert excluded not in custom, f"{excluded} is unused and must not be carried"

    for name, content in bundle.files.items():
        if name == "README.md":
            continue
        assert "nnarch" not in content, f"{name} mentions the tool"


# Input shapes that satisfy each research layer's declared rank and any width
# constraint it documents.
_PROBE_SHAPES: dict[str, list[int]] = {
    "research.VectorQuantizer": [12, 64],
    "research.GCNConv": [12, 12],
    "research.GATConv": [12, 12],
    "research.GraphSAGEConv": [12, 12],
    "research.GINConv": [12, 12],
    "research.PoissonEncoder": [16],
    "research.PatchEmbedding": [32, 32, 3],
    "research.SqueezeExcite": [8, 8, 16],
    "research.ConvNeXtBlock": [8, 8, 16],
    "research.MBConv": [8, 8, 16],
    "research.Involution": [8, 8, 16],
}


def test_every_research_layer_trains(registry: Registry) -> None:
    """
    > [!AML-DOC-UNIT]
    Each research layer must survive a training step inside a real model.
    @context Compiling only proves the forward pass builds. Training is what proves
             gradients reach the weights, which is the thing most likely to be wrong
             in these layers: several depend on a surrogate gradient or a
             straight-through estimator, and getting one backwards gives a layer
             that runs perfectly and learns nothing.
    """
    failures: list[str] = []

    for spec in registry.all():
        if not spec.is_research:
            continue

        shape = _PROBE_SHAPES.get(spec.id, [12, 16] if spec.rank_in == 3 else [12, 12])
        nodes = [node("inp", "keras.Input", shape=shape)]
        edges: list[Edge] = []

        if len(spec.inputs) >= 2 and spec.min_inputs >= 2:
            nodes.append(node("lay", spec.id))
            edges += [
                Edge(id=f"e{index}", source="inp", target="lay", target_port=port.name)
                for index, port in enumerate(spec.inputs[: spec.min_inputs])
            ]
        else:
            nodes.append(node("lay", spec.id))
            edges.append(edge("inp", "lay"))

        nodes.append(node("flat", "keras.Flatten"))
        nodes.append(node("head", "keras.Dense", units=3, activation="softmax"))
        edges += [edge("lay", "flat"), edge("flat", "head")]

        result = compile_graph(GraphIR(name="probe", nodes=nodes, edges=edges), registry)
        if not result.ok:
            reason = next((d.message for d in result.diagnostics if d.severity == "error"), "?")
            failures.append(f"{spec.id}: will not compile: {reason[:90]}")
            continue

        model = result.model
        model.compile("adam", "sparse_categorical_crossentropy")
        sample = np.random.rand(4, *shape).astype("float32")
        labels = np.random.randint(0, 3, 4)
        try:
            # Keyed by path, not position: some inner layers only create their
            # weights on the first call, so the list grows during `fit` and a
            # positional comparison pairs unrelated tensors.
            def snapshot():
                """
                > [!AML-DOC-UNIT]
                Read every trainable weight, keyed by its path.
                @returns mapping of weight path to its current values
                """
                return {
                    weight.path: np.asarray(keras.ops.convert_to_numpy(weight))
                    for weight in model.trainable_weights
                }

            before = snapshot()
            model.fit(sample, labels, epochs=1, verbose=0)
            after = snapshot()

            shared = set(before) & set(after)
            if shared and not any(
                not np.allclose(before[key], after[key]) for key in shared
            ):
                failures.append(f"{spec.id}: trained but no weight moved")
        except Exception as exc:
            failures.append(f"{spec.id}: {type(exc).__name__}: {str(exc)[:90]}")

    assert not failures, "research layers that failed to train:\n" + "\n".join(
        f"  {line}" for line in failures
    )
