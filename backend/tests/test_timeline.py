"""
> [!AML-DOC-FILE]
@file       tests/test_timeline.py
@description Regression tests for running an architecture over time: the loops it
             closes between steps, and whether closing them changes anything.
@module     tests.test_timeline
@exports    (pytest test functions)
@created    2026-10-02
@context    Some architectures are a process rather than a function, and one forward
            pass shows one tick of something that only means anything as a sequence
            [E-047]. The ablation tests matter most: a recurrence that does nothing
            looks exactly like one that works, and the only way to tell is to cut it
            and compare. The model this was built for turns out to ignore its own
            state entirely, which nothing in the tool could show before.
"""

from __future__ import annotations

import numpy as np
import pytest

from nnarch.catalog import Registry, bootstrap
from nnarch.ir import GraphIR, compile_graph
from nnarch.ir.schema import Edge, Node
from nnarch.viz.timeline import FeedbackPair, run_timeline, suggest_feedback


@pytest.fixture(scope="module")
def registry() -> Registry:
    """
    > [!AML-DOC-UNIT]
    The catalog, built once.
    @returns the bootstrapped registry
    """
    return bootstrap()


def _recurrent(carry: bool) -> GraphIR:
    """
    > [!AML-DOC-UNIT]
    A model with a state input and a state output.
    @param carry whether the state actually reaches the prediction
    @returns the graph
    @context With `carry` false the state is wired all the way through and multiplied
             by zero on the way. That is the shape of a model trained with its state
             input always zero: the connection is there, the gradient never was, and
             the weights learned to ignore it. Leaving the state dangling instead
             would not do — it has to reach the output and change nothing, or the test
             proves something easier than the real case.
    """
    nodes = [
        Node(id="seq", type="keras.Input", name="seq", params={"shape": [4, 3]}),
        Node(id="state_in", type="keras.Input", name="state_in", params={"shape": [6]}),
        Node(id="gru", type="keras.GRU", name="gru",
             params={"units": 6, "return_state": True}),
        Node(id="state_out", type="keras.Identity", name="state_out", params={}),
        Node(id="head", type="keras.Dense", name="head",
             params={"units": 4, "activation": "softmax"}),
    ]
    edges = [
        Edge(id="e1", source="seq", target="gru", target_port="input"),
        Edge(id="e3", source="gru", source_port="output_1", target="state_out"),
    ]
    if carry:
        edges.append(
            Edge(id="e2", source="state_in", target="gru", target_port="initial_state")
        )
        edges.append(Edge(id="e4", source="gru", target="head"))
    else:
        nodes.append(
            Node(id="deaf", type="keras.Lambda", name="deaf",
                 params={"function": "x * 0.0"})
        )
        nodes.append(Node(id="mix", type="keras.Add", name="mix", params={}))
        edges.append(Edge(id="e2", source="state_in", target="deaf"))
        edges.append(Edge(id="e5", source="gru", target="mix",
                          target_port="inputs", order=0))
        edges.append(Edge(id="e6", source="deaf", target="mix",
                          target_port="inputs", order=1))
        edges.append(Edge(id="e4", source="mix", target="head"))
    return GraphIR(name="recurrent", nodes=nodes, edges=edges)


def _frames(count: int):
    """
    > [!AML-DOC-UNIT]
    A different window per step, so the run is a walk rather than a repeat.
    @param count how many steps will be asked for
    @returns a callable the timeline uses to fetch each step's input
    """
    rng = np.random.default_rng(0)
    windows = [rng.standard_normal((1, 4, 3)).astype("float32") for _ in range(count)]

    def frame(index: int):
        return windows[index % len(windows)], f"step {index}"

    return frame


def test_a_loop_is_suggested_from_the_shapes(registry: Registry) -> None:
    """
    > [!AML-DOC-UNIT]
    An output that fits an input exactly is offered as a feedback pair.
    @param registry the catalog
    @raises AssertionError when the obvious loop is not found
    """
    graph = _recurrent(carry=True)
    model = compile_graph(graph, registry).model
    assert model is not None

    pairs = suggest_feedback(model, graph.resolved_outputs())
    assert FeedbackPair(source="state_out", target="state_in") in pairs


def test_an_ambiguous_loop_is_not_guessed(registry: Registry) -> None:
    """
    > [!AML-DOC-UNIT]
    Two inputs of the same shape mean nothing is suggested.
    @param registry the catalog
    @raises AssertionError when a coin is flipped on the user's behalf
    @context Shape is weak evidence. Where it cannot decide, the honest move is to
             offer nothing rather than one of two equally likely loops.
    """
    graph = GraphIR(
        name="ambiguous",
        nodes=[
            Node(id="a", type="keras.Input", name="a", params={"shape": [5]}),
            Node(id="b", type="keras.Input", name="b", params={"shape": [5]}),
            Node(id="join", type="keras.Add", name="join", params={}),
            Node(id="head", type="keras.Dense", name="head", params={"units": 5}),
        ],
        edges=[
            Edge(id="e1", source="a", target="join", target_port="inputs", order=0),
            Edge(id="e2", source="b", target="join", target_port="inputs", order=1),
            Edge(id="e3", source="join", target="head"),
        ],
    )
    model = compile_graph(graph, registry).model
    assert model is not None
    assert suggest_feedback(model, graph.resolved_outputs()) == []


def test_a_run_produces_one_record_per_step(registry: Registry) -> None:
    """
    > [!AML-DOC-UNIT]
    The sequence is as long as it was asked to be, and each step is its own.
    @param registry the catalog
    @raises AssertionError when steps are missing or identical
    """
    graph = _recurrent(carry=True)
    result = run_timeline(
        graph, registry, steps=5, frame=_frames(5),
        labels_by_size={4: ["n", "e", "s", "w"]}, ablate=False,
    )
    assert len(result.steps) == 5
    assert [step.step for step in result.steps] == [0, 1, 2, 3, 4]
    assert all(step.predictions["head"] for step in result.steps)


def test_an_inert_loop_is_reported_as_inert(registry: Registry) -> None:
    """
    > [!AML-DOC-UNIT]
    A feedback connection that changes nothing is said to change nothing.
    @param registry the catalog
    @raises AssertionError when an ignored state passes for a working one
    @context This is the test the whole feature exists for. The model that prompted it
             carries its state forward and ignores it, which its own training code
             admits in a comment and which no single forward pass could reveal.
    """
    graph = _recurrent(carry=False)
    # Declared rather than suggested: this graph has two outputs of the same width, so
    # nothing is proposed automatically and the user would say which loop they meant.
    result = run_timeline(
        graph, registry, steps=4, frame=_frames(4), ablate=True,
        feedback=[FeedbackPair(source="state_out", target="state_in")],
    )

    assert result.feedback, "the declared loop was dropped"
    assert result.ablation is not None
    assert "changed nothing" in result.ablation, result.ablation
    assert any(
        d.code == "timeline_feedback_inert" for d in result.diagnostics
    ), [d.code for d in result.diagnostics]


def test_a_live_loop_is_reported_as_live(registry: Registry) -> None:
    """
    > [!AML-DOC-UNIT]
    A state that reaches the prediction is seen to matter.
    @param registry the catalog
    @raises AssertionError when a working recurrence is called inert
    @context The complement of the test above, and the one that stops the check being
             a function that always says "inert" and is right by luck.
    """
    graph = _recurrent(carry=True)
    result = run_timeline(graph, registry, steps=6, frame=_frames(6), ablate=True)

    assert result.ablation is not None
    assert "changed nothing" not in result.ablation, result.ablation


def test_a_watched_layer_is_recorded_across_the_run(registry: Registry) -> None:
    """
    > [!AML-DOC-UNIT]
    A watched layer yields one row per step.
    @param registry the catalog
    @raises AssertionError when the trace is the wrong length or constant
    """
    graph = _recurrent(carry=True)
    result = run_timeline(
        graph, registry, steps=4, frame=_frames(4), watch=["gru"], ablate=False,
    )
    trace = next(t for t in result.traces if t.node_id == "gru")
    assert len(trace.values) == 4
    assert trace.channels == 6
    assert any(row != trace.values[0] for row in trace.values[1:]), "the trace never moved"


def test_what_was_held_at_zero_is_declared(registry: Registry) -> None:
    """
    > [!AML-DOC-UNIT]
    Inputs nobody supplied are named.
    @param registry the catalog
    @raises AssertionError when made-up inputs pass silently
    @context Same rule as the single-sample preview: a number the user did not supply
             must never look like one they did [E-036].
    """
    graph = GraphIR(
        name="extra-input",
        nodes=[
            Node(id="seq", type="keras.Input", name="seq", params={"shape": [4, 3]}),
            Node(id="knob", type="keras.Input", name="knob", params={"shape": [2]}),
            Node(id="flat", type="keras.Flatten", name="flat", params={}),
            Node(id="join", type="keras.Concatenate", name="join", params={}),
            Node(id="head", type="keras.Dense", name="head", params={"units": 3}),
        ],
        edges=[
            Edge(id="e1", source="seq", target="flat"),
            Edge(id="e2", source="flat", target="join", target_port="inputs", order=0),
            Edge(id="e3", source="knob", target="join", target_port="inputs", order=1),
            Edge(id="e4", source="join", target="head"),
        ],
    )
    result = run_timeline(graph, registry, steps=2, frame=_frames(2), ablate=False)
    said = " ".join(d.message for d in result.diagnostics)
    assert "knob" in said and "zero" in said, said
