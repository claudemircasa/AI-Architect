"""
> [!AML-DOC-FILE]
@file        src/nnarch/optimize/rules.py
@description The rewrite rules: what can be removed, and what can be swapped.
@module      nnarch.optimize.rules
@exports     RuleKind, Proposal, analyse_rules, apply_rule
@created     2026-10-02
@context     Every rule says which of two things it is. An `EXACT` rule is one whose
             result computes the same function as the original — removing a layer that
             passes its input through unchanged. A `SUBSTITUTION` puts a different
             layer in the same place because it fills the same role more cheaply, and
             the model it leaves has to be trained again [E-054].

             The rules only ever rewrite a chain: a node with one thing feeding it and
             one thing reading it. A node in the middle of a branch is left alone,
             because reconnecting a fork correctly needs to know what the fork meant.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from nnarch.catalog import Registry
from nnarch.ir.schema import Edge, GraphIR, Node


class RuleKind(str, Enum):
    """
    > [!AML-DOC-UNIT]
    What a proposal does to the model.

    - `EXACT`: the result computes the same function. Safe on a trained model.
    - `SUBSTITUTION`: a different layer in the same role, usually with fewer weights.
      The model changes and has to be trained again.
    """

    EXACT = "exact"
    SUBSTITUTION = "substitution"


class Proposal(BaseModel):
    """
    > [!AML-DOC-UNIT]
    One change the optimiser can make.
    @param id        stable identifier, so the editor can ask for this one
    @param rule      which rule produced it
    @param kind      exact or substitution
    @param title     one line naming the change
    @param detail    why it is worth making, and what it costs
    @param node_ids  the layers it touches
    """

    id: str
    rule: str
    kind: RuleKind
    title: str
    detail: str
    node_ids: list[str] = Field(default_factory=list)


def _single_chain(graph: GraphIR, node_id: str) -> tuple[Edge, Edge] | None:
    """
    > [!AML-DOC-UNIT]
    The one edge in and the one edge out, when a node sits in a plain chain.
    @param graph   the architecture
    @param node_id the node
    @returns (incoming, outgoing), or None when it branches or is an endpoint
    @sideEffects none
    @context Rewriting a node that forks means deciding what the fork meant, and
             nothing here knows that. A forked node is left alone rather than guessed
             at.
    """
    incoming = graph.incoming(node_id)
    outgoing = graph.outgoing(node_id)
    if len(incoming) != 1 or len(outgoing) != 1:
        return None
    # Nothing else may read the producer through this path, or removing the node
    # would silently re-route a second consumer.
    return incoming[0], outgoing[0]


def _drop_node(graph: GraphIR, node_id: str) -> GraphIR:
    """
    > [!AML-DOC-UNIT]
    Remove a chained node, joining what fed it to what read it.
    @param graph   the architecture
    @param node_id the node to remove
    @returns a new graph without it
    @raises ValueError when the node is not in a plain chain
    @sideEffects none; the original is untouched
    """
    chain = _single_chain(graph, node_id)
    if chain is None:
        raise ValueError(f"{node_id} does not sit in a plain chain")
    before, after = chain

    edges = [
        edge for edge in graph.edges
        if edge.id not in {before.id, after.id}
    ]
    edges.append(
        Edge(
            id=f"{before.source}->{after.target}:joined",
            source=before.source,
            source_port=before.source_port,
            target=after.target,
            target_port=after.target_port,
            order=after.order,
        )
    )
    return graph.model_copy(update={
        "nodes": [node for node in graph.nodes if node.id != node_id],
        "edges": edges,
    })


def _replace_node(graph: GraphIR, node_id: str, type_id: str, params: dict[str, Any]) -> GraphIR:
    """
    > [!AML-DOC-UNIT]
    Put a different layer in the same place.
    @param graph   the architecture
    @param node_id the node to replace
    @param type_id the catalog id of what replaces it
    @param params  its parameters
    @returns a new graph with the swap made
    @sideEffects none
    """
    nodes = [
        node.model_copy(update={"type": type_id, "params": params})
        if node.id == node_id else node
        for node in graph.nodes
    ]
    return graph.model_copy(update={"nodes": nodes})


#: Layers that pass their input through untouched, whatever they are configured with.
_PASS_THROUGH = {"keras.Identity"}


def _is_inert(node: Node) -> bool:
    """
    > [!AML-DOC-UNIT]
    Whether a layer provably does nothing to its input.
    @param node the layer
    @returns True when removing it cannot change the model's output
    @sideEffects none
    @context Each case is one a model picks up by being built and edited rather than
             by being written badly: an Identity left behind by an import, a Dropout
             turned down to nothing, an activation set to linear.
    """
    if node.type in _PASS_THROUGH:
        return True
    if node.type == "keras.Dropout" and float(node.params.get("rate") or 0) == 0.0:
        return True
    if node.type == "keras.Activation":
        activation = node.params.get("activation")
        return activation in (None, "", "linear")
    if node.type == "keras.Lambda":
        return str(node.params.get("function", "")).strip() == "x"
    return False


def analyse_rules(graph: GraphIR, registry: Registry) -> list[Proposal]:
    """
    > [!AML-DOC-UNIT]
    Every change the rules can see in this graph.
    @param graph    the architecture
    @param registry the layer catalog, used to check a replacement exists
    @returns the proposals, exact ones first
    @sideEffects none
    @context Nothing is applied here. A proposal is a claim about what would happen,
             and what would actually happen is measured afterwards by building both
             models [E-054].
    """
    proposals: list[Proposal] = []
    by_id = graph.node_map()

    for node in graph.nodes:
        chained = _single_chain(graph, node.id) is not None

        if chained and _is_inert(node):
            proposals.append(Proposal(
                id=f"drop:{node.id}",
                rule="drop-inert",
                kind=RuleKind.EXACT,
                title=f"Remove {node.name}",
                detail=(
                    f"{node.type.split('.')[-1]} passes its input through unchanged as "
                    f"it is configured, so removing it cannot change what the model "
                    f"computes."
                ),
                node_ids=[node.id],
            ))

        if chained and node.type == "keras.LSTM" and registry.get("keras.GRU"):
            proposals.append(Proposal(
                id=f"gru:{node.id}",
                rule="lstm-to-gru",
                kind=RuleKind.SUBSTITUTION,
                title=f"{node.name}: LSTM → GRU",
                detail=(
                    "A GRU has three gates where an LSTM has four, so about a quarter "
                    "fewer weights for the same units. It is a different layer, not a "
                    "cheaper spelling of the same one: the model has to be retrained."
                ),
                node_ids=[node.id],
            ))

        if (
            chained
            and node.type == "keras.Conv2D"
            and registry.get("keras.SeparableConv2D")
        ):
            kernel = node.params.get("kernel_size")
            size = kernel if isinstance(kernel, int) else (kernel or [0])[0]
            if int(size or 0) >= 3:
                proposals.append(Proposal(
                    id=f"sep:{node.id}",
                    rule="conv-to-separable",
                    kind=RuleKind.SUBSTITUTION,
                    title=f"{node.name}: Conv2D → SeparableConv2D",
                    detail=(
                        "A separable convolution factors the kernel into a per-channel "
                        "pass and a 1×1 mix, which is where most of MobileNet's size "
                        "saving comes from. Fewer weights and a different function: "
                        "the model has to be retrained."
                    ),
                    node_ids=[node.id],
                ))

        if node.type == "keras.Flatten":
            outgoing = graph.outgoing(node.id)
            incoming = graph.incoming(node.id)
            if len(outgoing) == 1 and len(incoming) == 1:
                consumer = by_id.get(outgoing[0].target)
                if (
                    consumer is not None
                    and consumer.type == "keras.Dense"
                    and registry.get("keras.GlobalAveragePooling2D")
                ):
                    proposals.append(Proposal(
                        id=f"gap:{node.id}",
                        rule="flatten-to-pooling",
                        kind=RuleKind.SUBSTITUTION,
                        title=f"{node.name}: Flatten → Global Average Pooling",
                        detail=(
                            "Flattening a feature map into a Dense layer is usually "
                            "where a convolutional model keeps most of its weights: "
                            "the Dense that follows is as wide as the whole map. "
                            "Averaging each channel instead collapses it to one number "
                            "per channel. This is the single largest saving available "
                            "in most image models, and the largest change: the model "
                            "has to be retrained."
                        ),
                        node_ids=[node.id, consumer.id],
                    ))

    # Repetition is reported rather than rewritten: identical blocks are a fact about
    # the architecture, and what to do about them is a decision nothing here can make.
    seen: dict[tuple[str, str], list[str]] = {}
    for node in graph.nodes:
        signature = (node.type, repr(sorted(node.params.items())))
        seen.setdefault(signature, []).append(node.id)
    for (type_id, _), ids in seen.items():
        if len(ids) >= 3:
            proposals.append(Proposal(
                id=f"repeat:{type_id}:{len(ids)}",
                rule="repeated-block",
                kind=RuleKind.EXACT,
                title=f"{len(ids)} identical {type_id.split('.')[-1]} layers",
                detail=(
                    "Configured identically. Nothing is changed by saying so — weight "
                    "sharing is not something this editor can express — but a repeated "
                    "block is usually where a model's size is, and where a smaller "
                    "one would start."
                ),
                node_ids=ids,
            ))

    proposals.sort(key=lambda p: (p.kind is RuleKind.SUBSTITUTION, p.id))
    return proposals


def apply_rule(graph: GraphIR, proposal: Proposal, registry: Registry) -> GraphIR:
    """
    > [!AML-DOC-UNIT]
    Carry out one proposal.
    @param graph    the architecture
    @param proposal what to do
    @param registry the layer catalog
    @returns the rewritten graph
    @raises ValueError when the proposal no longer fits the graph
    @sideEffects none; a new graph is returned and the original left alone
    """
    node_id = proposal.node_ids[0] if proposal.node_ids else ""
    node = graph.node_map().get(node_id)

    if proposal.rule == "repeated-block":
        return graph

    if node is None:
        raise ValueError(f"{node_id} is no longer in the graph")

    if proposal.rule == "drop-inert":
        return _drop_node(graph, node_id)

    if proposal.rule == "lstm-to-gru":
        keep = {
            key: value for key, value in node.params.items()
            # The gate-specific settings have no meaning on a GRU and are dropped
            # rather than carried across under a name that happens to match.
            if key not in {"unit_forget_bias"}
        }
        return _replace_node(graph, node_id, "keras.GRU", keep)

    if proposal.rule == "conv-to-separable":
        return _replace_node(graph, node_id, "keras.SeparableConv2D", dict(node.params))

    if proposal.rule == "flatten-to-pooling":
        return _replace_node(graph, node_id, "keras.GlobalAveragePooling2D", {})

    raise ValueError(f"no such rule: {proposal.rule}")
