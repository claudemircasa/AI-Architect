/**
 * > [!AML-DOC-FILE]
 * @file        src/editor/SidePanel.tsx
 * @description The right-hand pane, switching between editing a layer's settings and
 *              watching data move through the model.
 * @module      frontend/editor/SidePanel
 * @exports     SidePanel
 * @created     2026-10-01
 * @context     One pane with two tabs rather than two panes: the canvas is where the
 *              work happens and deserves the width, and the two views are read at
 *              different moments rather than together.
 */

import { useState } from "react";

import { useActivations } from "../store/activations";
import { useTraining } from "../store/training";
import { DataPanel } from "../viz/DataPanel";
import { TimePanel } from "../viz/TimePanel";
import { TrainPanel } from "../viz/TrainPanel";
import { PropertyPanel } from "./PropertyPanel";

type Tab = "properties" | "data" | "time" | "train";

/**
 * > [!AML-DOC-UNIT]
 * The right-hand pane.
 * @returns the tab strip and whichever panel is selected
 * @sideEffects none beyond the stores the panels use
 */
export function SidePanel() {
  const [tab, setTab] = useState<Tab>("properties");
  const captured = useActivations((state) => state.activations.length);
  const training = useTraining((state) => state.running);

  return (
    <aside className="flex h-full w-80 shrink-0 flex-col border-l border-line bg-surface-1">
      <div className="flex shrink-0 border-b border-line">
        {(
          [
            ["properties", "Settings"],
            ["data", "Data"],
            ["time", "Time"],
            ["train", "Train"],
          ] as const
        ).map(([value, label]) => (
          <button
            key={value}
            type="button"
            onClick={() => setTab(value)}
            className={`flex-1 border-b-2 px-3 py-1.5 text-[11px] ${
              tab === value
                ? "border-accent text-ink-0"
                : "border-transparent text-ink-2 hover:text-ink-1"
            }`}
          >
            {label}
            {value === "data" && captured > 0 && (
              <span className="ml-1 text-[9px] text-ink-2">{captured}</span>
            )}
            {value === "train" && training && (
              <span
                aria-label="training in progress"
                className="ml-1 inline-block h-1.5 w-1.5 rounded-full bg-good align-middle"
              />
            )}
          </button>
        ))}
      </div>

      <div className="min-h-0 flex-1">
        {tab === "properties" && <PropertyPanel embedded />}
        {tab === "data" && <DataPanel />}
        {tab === "time" && <TimePanel />}
        {tab === "train" && <TrainPanel />}
      </div>
    </aside>
  );
}
