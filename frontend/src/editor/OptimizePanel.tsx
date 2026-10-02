/**
 * > [!AML-DOC-FILE]
 * @file        src/editor/OptimizePanel.tsx
 * @description Proposes changes to an architecture and shows what each one actually
 *              costs, measured on this machine.
 * @module      frontend/editor/OptimizePanel
 * @exports     OptimizePanel
 * @created     2026-10-02
 * @context     The two kinds of proposal are kept apart on screen because the
 *              difference between them is the whole point: an exact rewrite leaves a
 *              model that computes the same function, a substitution leaves a
 *              different model that has to be retrained [E-054]. Mixing them into one
 *              ranked list of savings would be the most useful-looking way to mislead
 *              somebody.
 *
 *              Latency is reported only where the measurement supports it. Most
 *              proposals on a small model turn out to change nothing you can time,
 *              and saying so is the result [E-055].
 */

import { useCallback, useState } from "react";

import { applyProposals, EngineError, optimizeGraph } from "../api/client";
import type { Measurement, OptimizeReport, ProposalOutcome } from "../api/types";
import { useGraph } from "../store/graph";

/**
 * > [!AML-DOC-UNIT]
 * A count with thousands separators, or a dash.
 * @param value the number
 * @returns the text
 */
function count(value: number | null | undefined): string {
  return value === null || value === undefined ? "—" : value.toLocaleString("en-US");
}

/**
 * > [!AML-DOC-UNIT]
 * One proposal, with what it would do to the model.
 * @param outcome  the proposal and its measurement
 * @param before   the model as it stands
 * @param chosen   whether it is selected
 * @param onToggle called when it is selected or deselected
 * @returns the row
 */
function Row({
  outcome,
  before,
  chosen,
  onToggle,
}: {
  outcome: ProposalOutcome;
  before: Measurement;
  chosen: boolean;
  onToggle: () => void;
}) {
  const { proposal, after, speed, error } = outcome;
  const observation = proposal.rule === "repeated-block";
  const delta = after ? after.parameters - before.parameters : 0;
  const share = before.parameters ? (100 * delta) / before.parameters : 0;

  return (
    <li className="min-w-0 rounded border border-line p-2">
      <label className="flex min-w-0 items-start gap-2">
        {!observation && !error && (
          <input
            type="checkbox"
            className="mt-0.5 shrink-0"
            checked={chosen}
            onChange={onToggle}
          />
        )}
        <span className="min-w-0 flex-1">
          <span className="block text-[11px] font-semibold text-ink-0">
            {proposal.title}
          </span>
          <span className="mt-0.5 block text-[10px] leading-snug text-ink-2">
            {proposal.detail}
          </span>

          {error && (
            <span className="mt-1 block text-[10px] leading-snug text-danger">
              {error}
            </span>
          )}

          {after && (
            <span className="mt-1.5 flex flex-wrap items-baseline gap-x-3 gap-y-0.5">
              <span
                className={`font-mono text-[10px] ${
                  delta < 0 ? "text-good" : delta > 0 ? "text-warn" : "text-ink-2"
                }`}
              >
                {delta === 0 ? "no change in" : `${count(delta)}`} params
                {delta !== 0 && ` (${share.toFixed(1)}%)`}
              </span>
              <span className="font-mono text-[10px] text-ink-2">
                {after.layers} layers
              </span>
              {speed && (
                <span className="text-[10px] text-ink-2">{speed}</span>
              )}
            </span>
          )}
        </span>
      </label>
    </li>
  );
}

/**
 * > [!AML-DOC-UNIT]
 * The panel, shown over the editor.
 * @param onClose called when it should go away
 * @returns the panel
 * @sideEffects asks the engine to build and time one model per proposal, and replaces
 *              the graph when changes are applied
 */
export function OptimizePanel({ onClose }: { onClose: () => void }) {
  const [report, setReport] = useState<OptimizeReport | null>(null);
  const [chosen, setChosen] = useState<string[]>([]);
  const [running, setRunning] = useState(false);
  const [failed, setFailed] = useState<string | null>(null);
  const [measure, setMeasure] = useState(true);
  const [applied, setApplied] = useState<string | null>(null);

  const look = useCallback(async () => {
    setRunning(true);
    setFailed(null);
    setApplied(null);
    try {
      const found = await optimizeGraph(useGraph.getState().toIR(), measure);
      setReport(found);
      setChosen([]);
    } catch (error) {
      setFailed(
        error instanceof EngineError
          ? error.message
          : "The engine could not be reached.",
      );
    } finally {
      setRunning(false);
    }
  }, [measure]);

  const apply = useCallback(async () => {
    setRunning(true);
    try {
      const { graph } = await applyProposals(useGraph.getState().toIR(), chosen);
      // Rebuilt through the store's own loader, so the canvas, the undo history and
      // the engine all see one change rather than three different ones.
      const { graphFromIR } = await import("../store/persistence");
      useGraph.getState().commit();
      useGraph.getState().replaceGraph(graphFromIR(graph), useGraph.getState().name);
      setApplied(`${chosen.length} change${chosen.length === 1 ? "" : "s"} applied.`);
      setReport(null);
      setChosen([]);
    } catch (error) {
      setFailed(
        error instanceof EngineError ? error.message : "The changes could not be applied.",
      );
    } finally {
      setRunning(false);
    }
  }, [chosen]);

  const exact = report?.outcomes.filter(
    (outcome) => outcome.proposal.kind === "exact" && outcome.proposal.rule !== "repeated-block",
  ) ?? [];
  const swaps = report?.outcomes.filter(
    (outcome) => outcome.proposal.kind === "substitution",
  ) ?? [];
  const notes = report?.outcomes.filter(
    (outcome) => outcome.proposal.rule === "repeated-block",
  ) ?? [];

  const toggle = (id: string) =>
    setChosen((previous) =>
      previous.includes(id) ? previous.filter((other) => other !== id) : [...previous, id],
    );

  return (
    <div className="absolute inset-0 z-50 flex items-center justify-center bg-surface-0/80 p-6">
      <div className="flex max-h-full w-full max-w-xl flex-col rounded border border-line bg-surface-1">
        <div className="flex shrink-0 items-baseline justify-between gap-2 border-b border-line p-3">
          <h2 className="text-[13px] font-semibold text-ink-0">Optimise architecture</h2>
          <button
            type="button"
            onClick={onClose}
            disabled={running}
            className="text-[11px] text-ink-2 hover:text-ink-0 disabled:opacity-40"
          >
            Close
          </button>
        </div>

        <div className="min-h-0 flex-1 overflow-y-auto p-3">
          {!report && !failed && (
            <>
              <p className="text-[11px] leading-relaxed text-ink-2">
                Looks for layers that pass their input through unchanged, and for
                places where a cheaper layer fills the same role. Each proposal is
                built and timed, so what you see is what it did here rather than what
                it usually does.
              </p>
              <label className="mt-3 flex items-center gap-2 text-[10px] text-ink-2">
                <input
                  type="checkbox"
                  checked={measure}
                  onChange={(event) => setMeasure(event.target.checked)}
                />
                Time each model as well as counting its weights
              </label>
              {applied && (
                <p className="mt-3 text-[11px] text-good">{applied}</p>
              )}
            </>
          )}

          {failed && (
            <p className="rounded border border-danger/40 bg-danger/10 p-2 text-[11px] text-danger">
              {failed}
            </p>
          )}

          {report && (
            <div className="space-y-3">
              <div className="rounded border border-line bg-surface-2 p-2">
                <p className="text-[10px] uppercase tracking-wide text-ink-2">As it stands</p>
                <p className="mt-0.5 font-mono text-[11px] text-ink-1">
                  {count(report.before.parameters)} params · {report.before.layers} layers
                  {report.before.latency_ms !== null &&
                    ` · ${report.before.latency_ms} ms ± ${report.before.latency_spread}`}
                </p>
              </div>

              {exact.length > 0 && (
                <section>
                  <h3 className="text-[11px] font-semibold text-good">
                    Safe — the model computes the same thing
                  </h3>
                  <ul className="mt-1.5 space-y-1.5">
                    {exact.map((outcome) => (
                      <Row
                        key={outcome.proposal.id}
                        outcome={outcome}
                        before={report.before}
                        chosen={chosen.includes(outcome.proposal.id)}
                        onToggle={() => toggle(outcome.proposal.id)}
                      />
                    ))}
                  </ul>
                  {report.combined && (
                    <p className="mt-1.5 font-mono text-[10px] text-ink-2">
                      all together: {count(report.combined.parameters)} params ·{" "}
                      {report.combined.layers} layers
                      {report.combined.latency_ms !== null &&
                        ` · ${report.combined.latency_ms} ms`}
                    </p>
                  )}
                </section>
              )}

              {swaps.length > 0 && (
                <section>
                  <h3 className="text-[11px] font-semibold text-warn">
                    Different model — has to be retrained
                  </h3>
                  <ul className="mt-1.5 space-y-1.5">
                    {swaps.map((outcome) => (
                      <Row
                        key={outcome.proposal.id}
                        outcome={outcome}
                        before={report.before}
                        chosen={chosen.includes(outcome.proposal.id)}
                        onToggle={() => toggle(outcome.proposal.id)}
                      />
                    ))}
                  </ul>
                </section>
              )}

              {notes.length > 0 && (
                <section>
                  <h3 className="text-[11px] font-semibold text-ink-1">Worth knowing</h3>
                  <ul className="mt-1.5 space-y-1.5">
                    {notes.map((outcome) => (
                      <Row
                        key={outcome.proposal.id}
                        outcome={outcome}
                        before={report.before}
                        chosen={false}
                        onToggle={() => undefined}
                      />
                    ))}
                  </ul>
                </section>
              )}

              {report.diagnostics.map((diagnostic, index) => (
                <p
                  // eslint-disable-next-line react/no-array-index-key
                  key={index}
                  className="text-[10px] leading-snug text-ink-2"
                >
                  {diagnostic.message}
                </p>
              ))}
            </div>
          )}
        </div>

        <div className="flex shrink-0 items-center gap-2 border-t border-line p-3">
          <button
            type="button"
            onClick={() => void look()}
            disabled={running}
            className="rounded border border-line bg-surface-2 px-3 py-1.5 text-[11px]
                       text-ink-1 hover:text-ink-0 disabled:opacity-40"
          >
            {running && !report ? "Measuring…" : report ? "Look again" : "Look for changes"}
          </button>
          {report && (
            <button
              type="button"
              onClick={() => void apply()}
              disabled={running || chosen.length === 0}
              className="ml-auto rounded border border-accent/60 bg-surface-2 px-3 py-1.5
                         text-[11px] text-ink-0 hover:bg-surface-3 disabled:opacity-40"
            >
              Apply {chosen.length || ""}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
