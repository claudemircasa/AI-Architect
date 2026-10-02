/**
 * > [!AML-DOC-FILE]
 * @file        src/viz/TrainPanel.tsx
 * @description The training pane: set a run going on the current architecture and
 *              watch its loss and accuracy as they come in.
 * @module      frontend/viz/TrainPanel
 * @exports     TrainPanel, SERIES_COLOURS
 * @created     2026-10-01
 * @context     Training is explicit and interruptible [task 07]. The point of
 *              watching a curve is usually to decide a run is not worth finishing,
 *              so Stop is as prominent as Start once a run is going.
 */

import { useActivations } from "../store/activations";
import { useGraph } from "../store/graph";
import { useTraining } from "../store/training";
import { MetricChart } from "./MetricChart";

/**
 * Categorical slots 1 and 2 of the project's chart palette, stepped per surface.
 * Both pairs clear CVD separation, the lightness band and the chroma floor against
 * their own surface; the light pair's contrast warning is relieved by the direct
 * labels the chart always draws.
 */
export const SERIES_COLOURS = {
  training: "var(--color-series-1)",
  validation: "var(--color-series-2)",
};

const CONTROL =
  "w-full rounded bg-surface-2 border border-line px-2 py-1 text-[12px] text-ink-0 " +
  "outline-none focus:border-accent focus:ring-1 focus:ring-accent/40";

/**
 * > [!AML-DOC-UNIT]
 * The training pane.
 * @returns the run settings, the progress readout and the two metric charts
 * @sideEffects opens a training connection to the engine when started
 */
export function TrainPanel() {
  const training = useTraining();
  const compiled = useGraph((state) => state.compiled);
  const parameters = useGraph((state) => state.paramsTotal);
  const dataset = useActivations((state) => state.choice);

  const epochs = training.history.map((point) => point.epoch);
  const hasValidation = training.history.some((point) => point.valLoss !== null);

  return (
    <div className="flex h-full flex-col">
      <div className="space-y-2 border-b border-line p-3">
        <div className="grid grid-cols-2 gap-2">
          <label className="block">
            <span className="mb-1 block text-[10px] uppercase tracking-wide text-ink-2">
              Epochs
            </span>
            <input
              type="number"
              min={1}
              className={CONTROL}
              value={training.epochs}
              disabled={training.running}
              onChange={(event) =>
                training.configure({ epochs: Math.max(1, Number(event.target.value) || 1) })
              }
            />
          </label>
          <label className="block">
            <span className="mb-1 block text-[10px] uppercase tracking-wide text-ink-2">
              Batch size
            </span>
            <input
              type="number"
              min={1}
              className={CONTROL}
              value={training.batchSize}
              disabled={training.running}
              onChange={(event) =>
                training.configure({
                  batchSize: Math.max(1, Number(event.target.value) || 1),
                })
              }
            />
          </label>
          <label className="block">
            <span className="mb-1 block text-[10px] uppercase tracking-wide text-ink-2">
              Optimizer
            </span>
            <select
              className={CONTROL}
              value={training.optimizer}
              disabled={training.running}
              onChange={(event) => training.configure({ optimizer: event.target.value })}
            >
              {["adam", "adamw", "sgd", "rmsprop", "nadam", "adagrad"].map((name) => (
                <option key={name} value={name}>
                  {name}
                </option>
              ))}
            </select>
          </label>
          <label className="block">
            <span className="mb-1 block text-[10px] uppercase tracking-wide text-ink-2">
              Learning rate
            </span>
            <input
              type="number"
              step="0.0001"
              min={0}
              className={CONTROL}
              value={training.learningRate}
              disabled={training.running}
              onChange={(event) =>
                training.configure({ learningRate: Number(event.target.value) || 0.001 })
              }
            />
          </label>
        </div>

        <p className="text-[10px] leading-snug text-ink-2">
          Trains on whichever input the Data tab is set to. Generated data proves the
          architecture runs; it cannot learn anything.
        </p>

        <div className="flex gap-2 pt-1">
          {training.running ? (
            <button
              type="button"
              onClick={training.stop}
              className="flex-1 rounded border border-danger/60 bg-surface-2 px-2 py-1.5
                         text-[12px] text-danger hover:border-danger"
            >
              Stop
            </button>
          ) : (
            <button
              type="button"
              disabled={!compiled}
              onClick={() => training.start(dataset)}
              className="flex-1 rounded border border-line bg-surface-2 px-2 py-1.5
                         text-[12px] text-ink-0 hover:border-accent
                         disabled:opacity-40 disabled:hover:border-line"
              title={compiled ? "Train this architecture" : "The architecture has to be valid first"}
            >
              Train
            </button>
          )}
        </div>
      </div>

      <div className="min-h-0 flex-1 space-y-3 overflow-y-auto p-3">
        {training.message && (
          <p className="text-[11px] text-ink-1">{training.message}</p>
        )}

        {training.running && (
          <div>
            <div className="flex items-baseline justify-between text-[10px] text-ink-2">
              <span>
                epoch {Math.max(training.currentEpoch, 1)} of {training.totalEpochs}
              </span>
              <span className="font-mono">
                {training.batchLoss === null
                  ? ""
                  : `loss ${training.batchLoss.toFixed(4)}`}
              </span>
            </div>
            <div className="mt-1 h-1 overflow-hidden rounded bg-surface-2">
              <div
                className="h-full rounded bg-accent transition-[width] duration-150"
                style={{ width: `${Math.min(training.progress * 100, 100)}%` }}
              />
            </div>
            <p className="mt-1 text-[10px] text-ink-2">
              {training.elapsed.toFixed(1)}s elapsed
            </p>
          </div>
        )}

        <MetricChart
          title="Loss"
          series={[
            {
              label: "training",
              values: training.history.map((point) => point.loss),
              colour: SERIES_COLOURS.training,
            },
            ...(hasValidation
              ? [
                  {
                    label: "validation",
                    values: training.history.map((point) => point.valLoss),
                    colour: SERIES_COLOURS.validation,
                  },
                ]
              : []),
          ]}
        />

        {training.history.some((point) => point.accuracy !== null) && (
          <MetricChart
            title="Accuracy"
            format={(value) => `${(value * 100).toFixed(1)}%`}
            series={[
              {
                label: "training",
                values: training.history.map((point) => point.accuracy),
                colour: SERIES_COLOURS.training,
              },
              ...(hasValidation
                ? [
                    {
                      label: "validation",
                      values: training.history.map((point) => point.valAccuracy),
                      colour: SERIES_COLOURS.validation,
                    },
                  ]
                : []),
            ]}
          />
        )}

        {training.history.length > 0 && (
          <details className="rounded border border-line">
            <summary className="cursor-pointer px-2 py-1 text-[10px] uppercase tracking-wide text-ink-2">
              The numbers ({training.history.length} epochs)
            </summary>
            <table className="w-full border-t border-line text-[10px]">
              <thead>
                <tr className="text-ink-2">
                  <th className="px-2 py-1 text-left font-normal">epoch</th>
                  <th className="px-2 py-1 text-right font-normal">loss</th>
                  <th className="px-2 py-1 text-right font-normal">val loss</th>
                  <th className="px-2 py-1 text-right font-normal">val acc</th>
                </tr>
              </thead>
              <tbody className="font-mono text-ink-1">
                {training.history.map((point) => (
                  <tr key={point.epoch} className="border-t border-line/50">
                    <td className="px-2 py-0.5">{point.epoch}</td>
                    <td className="px-2 py-0.5 text-right">{point.loss.toFixed(4)}</td>
                    <td className="px-2 py-0.5 text-right">
                      {point.valLoss?.toFixed(4) ?? "—"}
                    </td>
                    <td className="px-2 py-0.5 text-right">
                      {point.valAccuracy === null
                        ? "—"
                        : `${(point.valAccuracy * 100).toFixed(1)}%`}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </details>
        )}

        {epochs.length === 0 && !training.running && (
          <p className="text-[11px] leading-relaxed text-ink-2">
            {compiled
              ? `${parameters.toLocaleString("en-US")} parameters ready to train.`
              : "Finish the architecture first."}
          </p>
        )}
      </div>
    </div>
  );
}
