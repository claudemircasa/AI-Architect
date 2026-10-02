/**
 * > [!AML-DOC-FILE]
 * @file        src/viz/TimePanel.tsx
 * @description Runs an architecture over time and shows what it predicted at each
 *              step, how its layers moved, and whether its feedback did anything.
 * @module      frontend/viz/TimePanel
 * @exports     TimePanel
 * @created     2026-10-02
 * @context     Some architectures are not a function of their input; they are a
 *              process. A free-running generator is handed the state it left off with
 *              and hands back the state to resume from, so a single forward pass shows
 *              one tick of something that only means anything as a sequence [E-047].
 *
 *              The ablation sits at the top rather than at the bottom, because a
 *              recurrence that changes nothing is the commonest way for this kind of
 *              model to be quietly wrong, and it is the first thing worth knowing.
 */

import { useCallback, useMemo, useRef, useState } from "react";

import { EngineError, runTimeline } from "../api/client";
import type { TimelineResult } from "../api/types";
import { useActivations } from "../store/activations";
import { useGraph } from "../store/graph";

/** How many steps to offer, and what to start at. */
const STEP_CHOICES = [4, 8, 16, 32, 64];

/** The tallest a single layer's map may be, so several of them still fit together. */
const TRACE_MAX_HEIGHT = 160;

/**
 * > [!AML-DOC-UNIT]
 * One output's prediction at every step, as a strip of names.
 * @param steps  the run
 * @param nodeId which output to draw
 * @param label  its name
 * @returns the strip
 * @context Repeats are drawn without their name so a held prediction reads as one
 *          stretch rather than as eight separate claims, which is how a chord that
 *          lasts two bars actually looks.
 */
function Strip({
  steps,
  nodeId,
  label,
}: {
  steps: TimelineResult["steps"];
  nodeId: string;
  label: string;
}) {
  const cells = steps.map((step) => step.predictions[nodeId]?.[0] ?? null);
  const high = Math.max(...cells.map((cell) => cell?.value ?? 0), 1e-9);

  return (
    <div className="min-w-0">
      <p className="text-[10px] text-ink-2">{label}</p>
      <div className="mt-0.5 flex gap-px overflow-hidden rounded-sm">
        {cells.map((cell, index) => {
          const same = index > 0 && cells[index - 1]?.name === cell?.name;
          return (
            <div
              // eslint-disable-next-line react/no-array-index-key
              key={index}
              className="min-w-0 flex-1 px-0.5 py-1"
              style={{
                background: cell
                  ? `color-mix(in srgb, var(--color-accent) ${Math.round(
                      (cell.value / high) * 70 + 10,
                    )}%, var(--color-surface-2))`
                  : "var(--color-surface-2)",
              }}
              title={
                cell
                  ? `step ${index}: ${cell.name} ${cell.value.toFixed(3)}`
                  : `step ${index}`
              }
            >
              <span className="block truncate text-center text-[9px] text-ink-0">
                {same ? "" : (cell?.name ?? "")}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

/**
 * > [!AML-DOC-UNIT]
 * One layer's activations across the run, as a heat map of steps by channel.
 * @param trace the recorded layer
 * @returns the map
 */
function Trace({ trace }: { trace: TimelineResult["traces"][number] }) {
  const flat = trace.values.flat();
  const low = Math.min(...flat);
  const high = Math.max(...flat, low + 1e-9);
  const width = trace.values[0]?.length ?? 1;

  return (
    <div className="min-w-0">
      <p className="text-[10px] text-ink-2">
        {trace.label}{" "}
        <span className="font-mono">
          {trace.values.length} steps × {trace.channels}
        </span>
      </p>
      <svg
        viewBox={`0 0 ${width} ${trace.values.length}`}
        preserveAspectRatio="none"
        className="mt-0.5 w-full rounded-sm"
        // Six pixels a step up to a ceiling: the viewBox scales, so a long run
        // squashes its rows rather than pushing everything below it off the panel.
        style={{ height: Math.min(TRACE_MAX_HEIGHT, Math.max(40, trace.values.length * 6)) }}
        role="img"
        aria-label={`${trace.label} across ${trace.values.length} steps`}
      >
        {trace.values.map((row, step) =>
          row.map((value, channel) => (
            <rect
              // eslint-disable-next-line react/no-array-index-key
              key={`${step}-${channel}`}
              x={channel}
              y={step}
              width={1}
              height={1}
              fill="var(--color-accent)"
              opacity={(value - low) / (high - low)}
            />
          )),
        )}
      </svg>
    </div>
  );
}

/**
 * > [!AML-DOC-UNIT]
 * The panel.
 * @returns the controls and whatever the last run produced
 * @sideEffects runs the model once per step, twice when comparing against a cut loop
 */
export function TimePanel() {
  const compiled = useGraph((state) => state.compiled);
  const nodes = useGraph((state) => state.nodes);
  const edges = useGraph((state) => state.edges);
  const labels = useActivations((state) => state.labels);
  const labelFile = useActivations((state) => state.labelFile);

  const [steps, setSteps] = useState(8);
  const [watch, setWatch] = useState<string[]>([]);
  const [ablate, setAblate] = useState(true);
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<TimelineResult | null>(null);
  const [running, setRunning] = useState(false);
  const [failed, setFailed] = useState<string | null>(null);
  const picker = useRef<HTMLInputElement>(null);

  const outputs = useMemo(() => {
    const consumed = new Set(edges.map((edge) => edge.source));
    return nodes
      .filter((node) => !consumed.has(node.id))
      .map((node) => ({ id: node.id, label: node.data.name }));
  }, [nodes, edges]);

  const run = useCallback(async () => {
    setRunning(true);
    setFailed(null);
    try {
      setResult(
        await runTimeline(useGraph.getState().toIR(), {
          steps,
          watch,
          labels,
          ablate,
          file,
        }),
      );
    } catch (error) {
      setFailed(
        error instanceof EngineError
          ? error.message
          : "The engine could not be reached.",
      );
    } finally {
      setRunning(false);
    }
  }, [steps, watch, labels, ablate, file]);

  if (!compiled) {
    return (
      <div className="p-3">
        <p className="text-[11px] leading-snug text-ink-2">
          A model has to compile before it can be run over time.
        </p>
      </div>
    );
  }

  return (
    // The same shape as the other panels: a header that stays put and a region that
    // scrolls. Without `h-full` this div is sized by its own contents, so its
    // `overflow-y-auto` never had anything to scroll and the results simply grew past
    // the bottom of the window — which is what watching several layers at once made
    // obvious [E-049].
    <div className="flex h-full min-w-0 flex-col">
      <div className="shrink-0 border-b border-line p-3">
        <p className="text-[11px] leading-relaxed text-ink-2">
        Runs the model once per step, moving through a recording and feeding whichever
        outputs loop back into inputs. For an architecture that carries state, this is
        the only reading that means anything.
      </p>

      <div className="mt-3 space-y-2">
        <label className="flex items-center justify-between gap-2 text-[10px] text-ink-2">
          Steps
          <select
            className="rounded border border-line bg-surface-2 px-1 py-0.5 text-[10px] text-ink-1"
            value={steps}
            onChange={(event) => setSteps(Number(event.target.value))}
          >
            {STEP_CHOICES.map((count) => (
              <option key={count} value={count}>
                {count}
              </option>
            ))}
          </select>
        </label>

        <div className="flex items-center justify-between gap-2">
          <span className="min-w-0 truncate text-[10px] text-ink-2">
            {file ? file.name : "no recording — generated input"}
          </span>
          <button
            type="button"
            onClick={() => picker.current?.click()}
            className="shrink-0 rounded border border-line bg-surface-2 px-2 py-0.5
                       text-[10px] text-ink-1 hover:text-ink-0"
          >
            {file ? "Replace…" : "Recording…"}
          </button>
          <input
            ref={picker}
            type="file"
            className="hidden"
            accept="audio/*,image/*,.npy"
            onChange={(event) => {
              const chosen = event.target.files?.[0] ?? null;
              event.target.value = "";
              setFile(chosen);
            }}
          />
        </div>

        <label className="flex items-center gap-2 text-[10px] text-ink-2">
          <input
            type="checkbox"
            checked={ablate}
            onChange={(event) => setAblate(event.target.checked)}
          />
          Also run with the feedback cut, to see whether it matters
        </label>

        {outputs.length > 0 && (
          <details className="rounded border border-line">
            <summary className="cursor-pointer px-2 py-1 text-[10px] text-ink-2">
              Watch layers ({watch.length})
            </summary>
            <div className="max-h-40 overflow-y-auto px-2 pb-2">
              {nodes.map((node) => (
                <label
                  key={node.id}
                  className="flex items-center gap-1.5 py-0.5 text-[10px] text-ink-1"
                >
                  <input
                    type="checkbox"
                    checked={watch.includes(node.id)}
                    onChange={(event) =>
                      setWatch((previous) =>
                        event.target.checked
                          ? [...previous, node.id]
                          : previous.filter((id) => id !== node.id),
                      )
                    }
                  />
                  <span className="min-w-0 truncate">{node.data.name}</span>
                </label>
              ))}
            </div>
          </details>
        )}

        <button
          type="button"
          onClick={() => void run()}
          disabled={running}
          className="w-full rounded border border-line bg-surface-2 px-3 py-1.5
                     text-[11px] text-ink-1 hover:text-ink-0 disabled:opacity-40"
        >
          {running ? `Running ${steps} steps…` : "Run over time"}
        </button>
      </div>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto p-3">
      {failed && (
        <p className="mt-3 rounded border border-danger/40 bg-danger/10 p-2 text-[11px] text-danger">
          {failed}
        </p>
      )}

      {result && (
        <div className="mt-4 space-y-3">
          {/*
            The first thing worth knowing: whether the loop did anything at all. A
            recurrence that changes nothing looks exactly like one that works until
            somebody cuts it and compares.
          */}
          {result.ablation && (
            <p
              className={`rounded border p-2 text-[10px] leading-snug ${
                result.ablation.startsWith("Cutting")
                  ? "border-warn/40 bg-warn/10 text-warn"
                  : "border-good/40 bg-good/10 text-good"
              }`}
            >
              {result.ablation}
            </p>
          )}

          {result.feedback.length > 0 && (
            <p className="text-[10px] leading-snug text-ink-2">
              Fed back between steps:{" "}
              {result.feedback.map((pair) => `${pair.source} → ${pair.target}`).join(", ")}
            </p>
          )}

          {outputs
            .filter((output) =>
              result.steps.some((step) => step.predictions[output.id]?.length),
            )
            .map((output) => (
              <Strip
                key={output.id}
                steps={result.steps}
                nodeId={output.id}
                label={output.label}
              />
            ))}

          {result.traces.map((trace) => (
            <Trace key={trace.node_id} trace={trace} />
          ))}

          {result.steps[0]?.note && (
            <p className="text-[10px] leading-snug text-ink-2">{result.steps[0].note}</p>
          )}

          {result.diagnostics.map((diagnostic, index) => (
            <p
              // eslint-disable-next-line react/no-array-index-key
              key={index}
              className={`text-[10px] leading-snug ${
                diagnostic.severity === "warning" ? "text-warn" : "text-ink-2"
              }`}
            >
              {diagnostic.message}
            </p>
          ))}

          {!labelFile && (
            <p className="text-[10px] leading-snug text-ink-2">
              Load a names file under Data to read the predictions as names rather than
              indices.
            </p>
          )}
        </div>
      )}
      </div>
    </div>
  );
}
