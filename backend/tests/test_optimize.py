"""
> [!AML-DOC-FILE]
@file       tests/test_optimize.py
@description Regression tests for the architecture optimiser: what it proposes, what
             it refuses to touch, and that an exact rewrite really is exact.
@module     tests.test_optimize
@exports    (pytest test functions)
@created    2026-10-02
@context    The distinction between an exact rewrite and a substitution is the whole
            feature [E-054]. The test that matters most runs both models on the same
            input and compares their outputs: a rule labelled `exact` that changes
            what the model computes would be worse than no optimiser at all, because
            it would be trusted.
"""

from __future__ import annotations

import numpy as np
import pytest

from nnarch.catalog import Registry, bootstrap
from nnarch.ir import GraphIR, compile_graph
from nnarch.ir.schema import Edge, Node
from nnarch.optimize import analyse, apply_proposals
from nnarch.optimize.rules import RuleKind, analyse_rules, apply_rule


@pytest.fixture(scope="module")
def registry() -> Registry:
    """
    > [!AML-DOC-UNIT]
    The catalog, built once.
    @returns the bootstrapped registry
    """
    return bootstrap()


def _node(node_id: str, type_id: str, **params: object) -> Node:
    """
    > [!AML-DOC-UNIT]
    A graph node.
    @param node_id identifier and display name
    @param type_id catalog layer id
    @param params  layer parameters
    @returns the node
    """
    return Node(id=node_id, type=type_id, name=node_id, params=dict(params))


def _chain(*pairs: tuple[str, str]) -> list[Edge]:
    """
    > [!AML-DOC-UNIT]
    Edges joining the given pairs in order.
    @param pairs (source, target) pairs
    @returns the edges
    """
    return [
        Edge(id=f"e{index}", source=source, target=target)
        for index, (source, target) in enumerate(pairs)
    ]


def _smelly() -> GraphIR:
    """
    > [!AML-DOC-UNIT]
    A small image model carrying every kind of layer the rules look for.
    @returns the graph
    """
    return GraphIR(
        name="smelly",
        nodes=[
            _node("in", "keras.Input", shape=[16, 16, 3]),
            _node("conv", "keras.Conv2D", filters=8, kernel_size=3, padding="same"),
            _node("idle", "keras.Identity"),
            _node("drop0", "keras.Dropout", rate=0.0),
            _node("linear", "keras.Activation", activation="linear"),
            _node("flat", "keras.Flatten"),
            _node("head", "keras.Dense", units=4, activation="softmax"),
        ],
        edges=_chain(
            ("in", "conv"), ("conv", "idle"), ("idle", "drop0"),
            ("drop0", "linear"), ("linear", "flat"), ("flat", "head"),
        ),
    )


def test_an_exact_rewrite_computes_the_same_thing(registry: Registry) -> None:
    """
    > [!AML-DOC-UNIT]
    Removing the inert layers leaves a model with identical outputs.
    @param registry the catalog
    @raises AssertionError when an "exact" rule changes the function
    @context The test the whole distinction rests on. A rule labelled exact that moved
             an output would be worse than no optimiser, because the label is what
             makes it safe to use without retraining.
    """
    graph = _smelly()
    exact = [
        p.id for p in analyse_rules(graph, registry)
        if p.kind is RuleKind.EXACT and p.rule != "repeated-block"
    ]
    assert len(exact) == 3, exact

    rewritten, _ = apply_proposals(graph, registry, exact)
    before = compile_graph(graph, registry).model
    after = compile_graph(rewritten, registry).model
    assert before is not None and after is not None

    # The same weights are needed on both sides, or the comparison is meaningless:
    # the layers that survive are copied across by name.
    by_name = {layer.name: layer for layer in before.layers}
    for layer in after.layers:
        twin = by_name.get(layer.name)
        if twin is not None and twin.get_weights():
            layer.set_weights(twin.get_weights())

    sample = np.random.default_rng(0).standard_normal((2, 16, 16, 3)).astype("float32")
    np.testing.assert_allclose(
        before.predict(sample, verbose=0), after.predict(sample, verbose=0),
        rtol=1e-5, atol=1e-5,
    )
    assert len(after.layers) == len(before.layers) - 3


def test_a_substitution_is_labelled_as_one(registry: Registry) -> None:
    """
    > [!AML-DOC-UNIT]
    Swapping a layer for a cheaper one is never called exact.
    @param registry the catalog
    @raises AssertionError when a model-changing rule claims to be safe
    """
    proposals = {p.rule: p for p in analyse_rules(_smelly(), registry)}
    assert proposals["flatten-to-pooling"].kind is RuleKind.SUBSTITUTION
    assert "retrain" in proposals["flatten-to-pooling"].detail.lower()


def test_a_branching_layer_is_left_alone(registry: Registry) -> None:
    """
    > [!AML-DOC-UNIT]
    A node that forks is not rewritten.
    @param registry the catalog
    @raises AssertionError when a fork is proposed for removal
    @context Reconnecting a fork means deciding what the fork meant, and nothing in
             the rules knows that. An inert layer feeding two consumers stays.
    """
    graph = GraphIR(
        name="forked",
        nodes=[
            _node("in", "keras.Input", shape=[8]),
            _node("idle", "keras.Identity"),
            _node("a", "keras.Dense", units=4),
            _node("b", "keras.Dense", units=4),
            _node("join", "keras.Add"),
        ],
        edges=[
            Edge(id="e1", source="in", target="idle"),
            Edge(id="e2", source="idle", target="a"),
            Edge(id="e3", source="idle", target="b"),
            Edge(id="e4", source="a", target="join", target_port="inputs", order=0),
            Edge(id="e5", source="b", target="join", target_port="inputs", order=1),
        ],
    )
    assert compile_graph(graph, registry).model is not None
    assert not [
        p for p in analyse_rules(graph, registry) if p.rule == "drop-inert"
    ], "a forked layer was proposed for removal"


def test_the_saving_is_measured_not_claimed(registry: Registry) -> None:
    """
    > [!AML-DOC-UNIT]
    Each proposal reports the model it actually leaves.
    @param registry the catalog
    @raises AssertionError when the numbers do not match a built model
    @context Not a table of expected savings: both models are built, and the count
             shown is the one Keras reported for the second [E-054].
    """
    report = analyse(_smelly(), registry, time_it=False)
    assert report.before.parameters > 0

    swap = next(
        outcome for outcome in report.outcomes
        if outcome.proposal.rule == "flatten-to-pooling"
    )
    assert swap.after is not None
    assert swap.after.parameters < report.before.parameters

    rewritten = apply_rule(_smelly(), swap.proposal, registry)
    built = compile_graph(rewritten, registry).model
    assert built is not None
    assert swap.after.parameters == int(built.count_params())


def test_a_difference_inside_the_noise_is_not_reported_as_one(
    registry: Registry,
) -> None:
    """
    > [!AML-DOC-UNIT]
    A timing change smaller than the measurement's own spread is called what it is.
    @param registry the catalog
    @raises AssertionError when noise is presented as a speed-up
    @context Removing an Identity layer cannot measurably change anything at batch
             one, and a tool that reported "0.2 ms faster" would be dressing noise up
             as a result [E-055].
    """
    report = analyse(_smelly(), registry, time_it=True)
    timed = [o for o in report.outcomes if o.speed]
    assert timed, "nothing was timed"

    removals = [o for o in timed if o.proposal.rule == "drop-inert"]
    assert removals
    assert all(
        "no measurable difference" in (o.speed or "") for o in removals
    ), [o.speed for o in removals]


def test_repetition_is_reported_without_being_changed(registry: Registry) -> None:
    """
    > [!AML-DOC-UNIT]
    Identical layers are pointed out, and applying that proposal changes nothing.
    @param registry the catalog
    @raises AssertionError when an observation silently edits the graph
    @context "Group the identical layers" is a reasonable thing to want and an
             unreasonable thing to do silently: weight sharing is not something this
             IR can express, so the honest move is to say where the repetition is.
    """
    graph = GraphIR(
        name="repeated",
        nodes=[
            _node("in", "keras.Input", shape=[8]),
            *[_node(f"d{i}", "keras.Dense", units=8, activation="relu") for i in range(4)],
        ],
        edges=_chain(("in", "d0"), ("d0", "d1"), ("d1", "d2"), ("d2", "d3")),
    )
    note = next(
        p for p in analyse_rules(graph, registry) if p.rule == "repeated-block"
    )
    assert len(note.node_ids) == 4

    unchanged, _ = apply_proposals(graph, registry, [note.id])
    assert len(unchanged.nodes) == len(graph.nodes)
    assert len(unchanged.edges) == len(graph.edges)


def test_a_proposal_that_no_longer_applies_is_skipped(registry: Registry) -> None:
    """
    > [!AML-DOC-UNIT]
    Applying two changes where the first undoes the second is reported, not crashed.
    @param registry the catalog
    @raises AssertionError when a stale proposal corrupts the graph
    @context Each proposal is re-derived against the graph as it stands after the ones
             before it, because removing one layer can make a later proposal refer to
             something that is no longer there.
    """
    graph = _smelly()
    exact = [
        p.id for p in analyse_rules(graph, registry)
        if p.kind is RuleKind.EXACT and p.rule != "repeated-block"
    ]
    rewritten, diagnostics = apply_proposals(graph, registry, [*exact, *exact])

    assert compile_graph(rewritten, registry).model is not None
    assert any("no longer applies" in d.message for d in diagnostics)
