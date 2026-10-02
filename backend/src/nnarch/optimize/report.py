"""
> [!AML-DOC-FILE]
@file        src/nnarch/optimize/report.py
@description Builds both models and measures them, so a proposal's cost is read off a
             clock rather than argued from a table.
@module      nnarch.optimize.report
@exports     Measurement, ProposalOutcome, OptimizeReport, analyse, apply_proposals
@created     2026-10-02
@context     A parameter count can be computed; latency cannot. Whether a separable
             convolution is faster than the one it replaces depends on the shapes, the
             hardware and what the runtime chooses to fuse, and the answer is often
             no — a smaller model with more layers can be slower. So both models are
             built and both are timed, on this machine, and the number shown is the
             one that was observed [E-054].

             Timing is the median of several passes after a warm-up, because the first
             pass through a Keras model pays for tracing and would make every
             "optimisation" look like a catastrophe.
"""

from __future__ import annotations

import statistics
import time
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from nnarch.catalog import Diagnostic, Registry
from nnarch.ir.compiler import compile_graph
from nnarch.ir.schema import GraphIR

from .rules import Proposal, RuleKind, analyse_rules, apply_rule

#: Passes to time, after the warm-up.
_RUNS = 7

#: Passes to throw away first, which pay for tracing and allocation.
_WARMUP = 2


class Measurement(BaseModel):
    """
    > [!AML-DOC-UNIT]
    What a model costs, as measured.
    @param parameters      total weights
    @param trainable       weights that train
    @param layers          how many layers it has
    @param latency_ms      median time for one forward pass on this machine
    @param latency_spread  the gap between the fastest and slowest timed pass
    """

    parameters: int = 0
    trainable: int = 0
    layers: int = 0
    latency_ms: float | None = None
    latency_spread: float | None = None


class ProposalOutcome(BaseModel):
    """
    > [!AML-DOC-UNIT]
    A proposal and what it would actually do.
    @param proposal the change
    @param after    the model it leaves, measured
    @param speed    what the timing supports saying, which is often nothing
    @param error    why it could not be measured, when it could not
    """

    proposal: Proposal
    after: Measurement | None = None
    speed: str | None = None
    error: str | None = None


def _speed_verdict(before: Measurement, after: Measurement) -> str | None:
    """
    > [!AML-DOC-UNIT]
    What the two timings actually support saying.
    @param before the model as it stands
    @param after  the model the proposal leaves
    @returns a sentence, or None when nothing was timed
    @sideEffects none
    @context A difference smaller than the spread of the measurements is not a
             difference; printing it as one would dress noise up as a result. Most
             proposals on a small model land here, and saying so is the honest
             outcome rather than a disappointing one [E-055].
    """
    if before.latency_ms is None or after.latency_ms is None:
        return None
    change = after.latency_ms - before.latency_ms
    noise = max(before.latency_spread or 0.0, after.latency_spread or 0.0)
    if abs(change) <= noise:
        return (
            f"no measurable difference — the change of {change:+.2f} ms is inside the "
            f"{noise:.2f} ms spread of the measurements"
        )
    percent = 100.0 * change / before.latency_ms
    return (
        f"{'slower' if change > 0 else 'faster'} by {abs(change):.2f} ms "
        f"({abs(percent):.0f}%)"
    )


class OptimizeReport(BaseModel):
    """
    > [!AML-DOC-UNIT]
    Everything the optimiser found.
    @param before      the model as it stands
    @param outcomes    each proposal, measured
    @param combined    what applying every exact proposal together would give
    @param diagnostics anything worth saying
    """

    before: Measurement = Field(default_factory=Measurement)
    outcomes: list[ProposalOutcome] = Field(default_factory=list)
    combined: Measurement | None = None
    diagnostics: list[Diagnostic] = Field(default_factory=list)


def _measure(graph: GraphIR, registry: Registry, *, time_it: bool = True) -> Measurement:
    """
    > [!AML-DOC-UNIT]
    Build a graph and read its size and speed.
    @param graph    the architecture
    @param registry the layer catalog
    @param time_it  whether to run the model; False counts weights only
    @returns the measurement
    @raises ValueError when the graph will not compile
    @sideEffects builds a Keras model and, when timing, runs it several times
    """
    result = compile_graph(graph, registry)
    if result.model is None:
        reasons = [d.message for d in result.diagnostics if d.severity == "error"]
        raise ValueError(reasons[0] if reasons else "the graph would not compile")

    model = result.model
    measurement = Measurement(
        parameters=int(model.count_params()),
        trainable=int(sum(int(np.prod(w.shape)) for w in model.trainable_weights)),
        layers=len(model.layers),
    )
    if not time_it:
        return measurement

    feed = [
        np.zeros(
            (1, *[1 if d is None else int(d) for d in tensor.shape[1:]]),
            dtype="float32",
        )
        for tensor in model.inputs
    ]
    one = feed if len(feed) > 1 else feed[0]

    for _ in range(_WARMUP):
        model.predict(one, verbose=0)

    timings = []
    for _ in range(_RUNS):
        started = time.perf_counter()
        model.predict(one, verbose=0)
        timings.append((time.perf_counter() - started) * 1000.0)

    measurement.latency_ms = round(statistics.median(timings), 2)
    measurement.latency_spread = round(max(timings) - min(timings), 2)
    return measurement


def analyse(
    graph: GraphIR,
    registry: Registry,
    *,
    time_it: bool = True,
) -> OptimizeReport:
    """
    > [!AML-DOC-UNIT]
    Find every change worth proposing, and measure each one.
    @param graph    the architecture
    @param registry the layer catalog
    @param time_it  whether to time the models; False reports sizes only, which is
                    much faster on a large graph
    @returns the report
    @raises nothing; a graph that will not compile comes back as diagnostics
    @sideEffects builds one model per proposal, and runs each several times when timing
    """
    diagnostics: list[Diagnostic] = []
    try:
        before = _measure(graph, registry, time_it=time_it)
    except ValueError as exc:
        return OptimizeReport(diagnostics=[
            Diagnostic(severity="error", code="optimize_invalid", message=str(exc))
        ])

    proposals = analyse_rules(graph, registry)
    outcomes: list[ProposalOutcome] = []

    for proposal in proposals:
        if proposal.rule == "repeated-block":
            # An observation, not a change: there is nothing to build and nothing to
            # time, and inventing a measurement for it would be the kind of number
            # that looks like evidence.
            outcomes.append(ProposalOutcome(proposal=proposal))
            continue
        try:
            rewritten = apply_rule(graph, proposal, registry)
            after = _measure(rewritten, registry, time_it=time_it)
            outcomes.append(ProposalOutcome(
                proposal=proposal,
                after=after,
                speed=_speed_verdict(before, after),
            ))
        except Exception as exc:  # noqa: BLE001 - a bad rule must not sink the report
            outcomes.append(ProposalOutcome(proposal=proposal, error=str(exc)))

    exact = [o.proposal for o in outcomes
             if o.proposal.kind is RuleKind.EXACT and o.proposal.rule != "repeated-block"]
    combined = None
    if exact:
        try:
            together = graph
            for proposal in exact:
                together = apply_rule(together, proposal, registry)
            combined = _measure(together, registry, time_it=time_it)
        except Exception as exc:  # noqa: BLE001
            diagnostics.append(Diagnostic(
                severity="warning", code="optimize_combine_failed",
                message=f"The safe changes could not all be applied together: {exc}",
            ))

    if not proposals:
        diagnostics.append(Diagnostic(
            severity="info", code="optimize_nothing_found",
            message=(
                "Nothing to propose. The rules look for layers that pass their input "
                "through unchanged, and for places where a cheaper layer fills the "
                "same role."
            ),
        ))

    return OptimizeReport(
        before=before, outcomes=outcomes, combined=combined, diagnostics=diagnostics,
    )


def apply_proposals(
    graph: GraphIR,
    registry: Registry,
    ids: list[str],
) -> tuple[GraphIR, list[Diagnostic]]:
    """
    > [!AML-DOC-UNIT]
    Carry out the chosen proposals.
    @param graph    the architecture
    @param registry the layer catalog
    @param ids      which proposals, by id
    @returns the rewritten graph and anything worth saying
    @raises nothing; a proposal that no longer fits is reported and skipped
    @sideEffects none
    @context Each is applied against the graph as it stands after the ones before it,
             and re-derived rather than trusted: removing one layer can make a second
             proposal meaningless, and applying it anyway would corrupt the graph.
    """
    diagnostics: list[Diagnostic] = []
    wanted = list(ids)
    current = graph

    for identifier in wanted:
        available = {p.id: p for p in analyse_rules(current, registry)}
        proposal = available.get(identifier)
        if proposal is None:
            diagnostics.append(Diagnostic(
                severity="info", code="optimize_no_longer_applies",
                message=f"'{identifier}' no longer applies and was skipped.",
            ))
            continue
        try:
            current = apply_rule(current, proposal, registry)
        except ValueError as exc:
            diagnostics.append(Diagnostic(
                severity="warning", code="optimize_apply_failed",
                message=f"'{identifier}' could not be applied: {exc}",
            ))

    return current, diagnostics
