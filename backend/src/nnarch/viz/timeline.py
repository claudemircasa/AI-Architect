"""
> [!AML-DOC-FILE]
@file        src/nnarch/viz/timeline.py
@description Run a model repeatedly over a moving window, feeding chosen outputs back
             into chosen inputs, and record what changes from step to step.
@module      nnarch.viz.timeline
@exports     FeedbackPair, TimelineStep, TimelineTrace, TimelineResult, run_timeline
@created     2026-10-02
@context     Some architectures are not a function of their input; they are a process.
             A free-running generator is handed the state it left off with and hands
             back the state to resume from, and a single forward pass shows one tick
             of something that only means anything as a sequence [E-047].

             The ablation is not an extra: a recurrence that changes nothing is the
             most common way for this kind of model to be quietly broken, and running
             the same steps with the loop cut is the only way to see it. The model
             this was built for turns out to ignore its own state entirely — which
             its training code says in a comment, and which nothing in the tool could
             have shown before.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from pydantic import BaseModel, Field

from nnarch.catalog import Diagnostic, Registry
from nnarch.ir.compiler import compile_graph
from nnarch.ir.schema import GraphIR


class FeedbackPair(BaseModel):
    """
    > [!AML-DOC-UNIT]
    One output wired back into one input between steps.
    @param source the output node
    @param target the input node it feeds on the next step
    """

    source: str
    target: str


class Prediction(BaseModel):
    """
    > [!AML-DOC-UNIT]
    What one output said at one step.
    @param name  the class's name, or its index when there is no vocabulary
    @param value the score
    """

    name: str
    value: float


class TimelineStep(BaseModel):
    """
    > [!AML-DOC-UNIT]
    One tick of the run.
    @param step        its position, from zero
    @param predictions the leading classes of each output that has any
    @param note        what the data source did to produce this step's input
    """

    step: int
    predictions: dict[str, list[Prediction]] = Field(default_factory=dict)
    note: str | None = None


class TimelineTrace(BaseModel):
    """
    > [!AML-DOC-UNIT]
    One layer's activations across every step, for drawing as a heat map.
    @param node_id  the layer
    @param label    its name
    @param values   step by channel, downsampled on the channel axis
    @param channels how many channels there really were
    """

    node_id: str
    label: str
    values: list[list[float]] = Field(default_factory=list)
    channels: int = 0


class TimelineResult(BaseModel):
    """
    > [!AML-DOC-UNIT]
    Everything a run over time produced.
    @param steps       one entry per tick
    @param traces      the watched layers
    @param feedback    the loops that were closed
    @param ablation    what cutting those loops changed, when it was measured
    @param diagnostics anything worth saying
    """

    steps: list[TimelineStep] = Field(default_factory=list)
    traces: list[TimelineTrace] = Field(default_factory=list)
    feedback: list[FeedbackPair] = Field(default_factory=list)
    ablation: str | None = None
    diagnostics: list[Diagnostic] = Field(default_factory=list)


def suggest_feedback(model: Any, output_ids: list[str]) -> list[FeedbackPair]:
    """
    > [!AML-DOC-UNIT]
    Propose the loops a model's shapes allow.
    @param model      the compiled model
    @param output_ids its output nodes, in the order the model returns them
    @returns the pairs whose shapes match exactly and unambiguously
    @sideEffects none
    @context Shape is the only evidence available, and it is weak evidence: it is
             offered as a suggestion the user confirms, never closed silently. A pair
             is dropped when either side could match more than one thing, because a
             guess between two loops is worse than asking.
    """
    inputs = [
        (str(tensor.name).split("/")[0], tuple(tensor.shape[1:]))
        for tensor in model.inputs
    ]
    outputs = [
        (node_id, tuple(tensor.shape[1:]))
        for node_id, tensor in zip(output_ids, model.outputs)
    ]

    pairs: list[FeedbackPair] = []
    for node_id, shape in outputs:
        fits = [name for name, in_shape in inputs if in_shape == shape]
        rival = [other for other, out_shape in outputs if out_shape == shape]
        if len(fits) == 1 and len(rival) == 1:
            pairs.append(FeedbackPair(source=node_id, target=fits[0]))
    return pairs


def _top(values: np.ndarray, names: list[str], limit: int) -> list[Prediction]:
    """
    > [!AML-DOC-UNIT]
    The leading entries of one output.
    @param values the output, batch axis removed
    @param names  its vocabulary, possibly empty
    @param limit  how many to keep
    @returns the leaders, highest first
    @sideEffects none
    """
    flat = np.asarray(values).ravel()
    order = np.argsort(flat)[::-1][:limit]
    return [
        Prediction(
            name=names[int(i)] if int(i) < len(names) else str(int(i)),
            value=float(flat[int(i)]),
        )
        for i in order
    ]


def run_timeline(
    graph: GraphIR,
    registry: Registry,
    *,
    steps: int,
    frame: Callable[[int], tuple[np.ndarray, str | None]],
    feedback: list[FeedbackPair] | None = None,
    watch: list[str] | None = None,
    labels_by_size: dict[int, list[str]] | None = None,
    labels_by_node: dict[str, list[str]] | None = None,
    ablate: bool = True,
    top_k: int = 3,
) -> TimelineResult:
    """
    > [!AML-DOC-UNIT]
    Run the model step by step, closing the chosen loops between steps.
    @param graph          the architecture
    @param registry       the layer catalog
    @param steps          how many ticks to run
    @param frame          called with a step number, returns that step's primary input
                          and a sentence describing where it came from
    @param feedback       loops to close; suggested from the shapes when not given
    @param watch          layers to record across the run
    @param labels_by_size names keyed by class count
    @param labels_by_node names chosen for a particular layer
    @param ablate         also run with the loops cut, to see whether they matter
    @param top_k          how many leading classes to keep per output per step
    @returns the run
    @raises nothing; a graph that will not compile comes back as diagnostics
    @sideEffects compiles the graph and runs one forward pass per step, twice when
                 ablating
    @context Inputs that are neither the primary one nor fed by a loop are held at
             zero throughout, which is what they start at. That is stated rather than
             assumed, because a number nobody supplied must not look like one they did
             [E-036].
    """
    compiled = compile_graph(graph, registry)
    if compiled.model is None:
        return TimelineResult(
            diagnostics=[
                d for d in compiled.diagnostics if d.severity == "error"
            ] or [Diagnostic(severity="error", code="timeline_invalid",
                             message="The architecture has to compile before it can be run.")]
        )

    model = compiled.model
    output_ids = [node_id for node_id in graph.resolved_outputs()]
    input_names = [str(tensor.name).split("/")[0] for tensor in model.inputs]

    loops = feedback if feedback is not None else suggest_feedback(model, output_ids)
    loops = [
        pair for pair in loops
        if pair.target in input_names and pair.source in output_ids
    ]

    # The input the moving window feeds. Everything else is a loop or a zero.
    from nnarch.viz.tensors import primary_input_index

    primary = primary_input_index(model)
    diagnostics: list[Diagnostic] = []

    watched = [node for node in (watch or []) if node]
    probe = None
    probe_ids: list[str] = []
    if watched:
        from nnarch.naming import NamePool

        pool = NamePool()
        by_name = {layer.name: layer for layer in model.layers}
        ordered, _ = graph.topological_order()
        node_map = graph.node_map()
        tensors = []
        for node_id in ordered:
            node = node_map[node_id]
            name = pool.allocate(node.name, node_id)
            if node_id in watched and name in by_name:
                tensors.append(by_name[name].output)
                probe_ids.append(node_id)
        if tensors:
            import keras

            probe = keras.Model(model.inputs, tensors)

    def one_pass(
        close_loops: bool,
    ) -> tuple[list[TimelineStep], list[list[np.ndarray]], list[list[np.ndarray]]]:
        """
        > [!AML-DOC-UNIT]
        Run the whole sequence once.
        @param close_loops whether to feed the outputs back
        @returns the steps, the watched activations per step, and every raw output per
                 step — the last so two runs can be compared on their numbers rather
                 than only on which class came out on top
        @sideEffects runs the model `steps` times
        """
        held: dict[str, np.ndarray] = {}
        records: list[TimelineStep] = []
        traces: list[list[np.ndarray]] = []
        raw: list[list[np.ndarray]] = []

        for index in range(steps):
            sample, note = frame(index)
            feed = []
            for position, name in enumerate(input_names):
                if position == primary:
                    feed.append(sample)
                elif close_loops and name in held:
                    feed.append(held[name])
                else:
                    dims = [1 if d is None else int(d) for d in model.inputs[position].shape[1:]]
                    feed.append(np.zeros((len(sample), *dims), dtype="float32"))

            produced = model.predict(feed if len(feed) > 1 else feed[0], verbose=0)
            if not isinstance(produced, list):
                produced = [produced]

            raw.append([np.asarray(p) for p in produced])
            by_output = dict(zip(output_ids, produced))
            for pair in loops:
                if pair.source in by_output:
                    held[pair.target] = np.asarray(by_output[pair.source])

            predictions: dict[str, list[Prediction]] = {}
            for node_id, values in by_output.items():
                array = np.asarray(values)
                width = int(array.shape[-1]) if array.ndim else 0
                names = (labels_by_node or {}).get(node_id)
                if names is None:
                    names = (labels_by_size or {}).get(width, [])
                if names:
                    predictions[node_id] = _top(array[0], names, top_k)
            records.append(TimelineStep(step=index, predictions=predictions, note=note))

            if probe is not None:
                seen = probe.predict(feed if len(feed) > 1 else feed[0], verbose=0)
                traces.append([np.asarray(t) for t in (seen if isinstance(seen, list) else [seen])])

        return records, traces, raw

    records, traces, values = one_pass(close_loops=bool(loops))

    if ablate and loops:
        cut, _, cut_values = one_pass(close_loops=False)

        # Two questions, and they are not the same one. Whether the numbers moved at
        # all says whether the loop is connected to anything; whether the leading
        # class moved says whether that mattered to the answer. Reporting only the
        # second would call a loop inert whenever a softmax happened to stay put,
        # which is a different and much weaker claim than the one being made.
        drift = 0.0
        for before, after in zip(values, cut_values):
            for left, right in zip(before, after):
                if left.shape == right.shape and left.size:
                    drift = max(drift, float(np.max(np.abs(left - right))))

        changed = sum(
            1
            for a, b in zip(records, cut)
            if {k: v[0].name for k, v in a.predictions.items()}
            != {k: v[0].name for k, v in b.predictions.items()}
        )

        if drift <= 1e-6:
            ablation = (
                f"Cutting the feedback changed nothing: across all {len(records)} "
                f"steps every output was identical with the loop closed and with it "
                f"held at zero. This model is not using its own state."
            )
            diagnostics.append(Diagnostic(
                severity="warning", code="timeline_feedback_inert", message=ablation,
            ))
        elif changed == 0:
            ablation = (
                f"The feedback moves the outputs — by up to {drift:.3g} — but never "
                f"enough to change the leading class in these {len(records)} steps."
            )
        else:
            ablation = (
                f"The feedback matters: {changed} of {len(records)} steps predicted "
                f"something different with the loop cut."
            )
    else:
        ablation = None

    built: list[TimelineTrace] = []
    for position, node_id in enumerate(probe_ids):
        per_step = []
        channels = 0
        for step_values in traces:
            array = np.asarray(step_values[position])[0]
            channels = int(array.size)
            flat = array.ravel()
            if flat.size > 128:
                edges = np.linspace(0, flat.size, 129, dtype=int)
                flat = np.array([flat[a:b].mean() for a, b in zip(edges[:-1], edges[1:])])
            per_step.append([float(v) for v in flat])
        label = graph.node_map()[node_id].name or node_id
        built.append(TimelineTrace(node_id=node_id, label=label,
                                   values=per_step, channels=channels))

    if len(model.inputs) > 1:
        fed = {pair.target for pair in loops}
        zeroed = [n for i, n in enumerate(input_names) if i != primary and n not in fed]
        if zeroed:
            diagnostics.append(Diagnostic(
                severity="info", code="timeline_zeroed_inputs",
                message=(
                    f"{input_names[primary]} moved with the window and "
                    f"{', '.join(sorted(fed)) or 'nothing'} came from the loop; "
                    f"{', '.join(zeroed)} stayed at zero throughout."
                ),
            ))

    return TimelineResult(
        steps=records, traces=built, feedback=loops,
        ablation=ablation, diagnostics=diagnostics,
    )
