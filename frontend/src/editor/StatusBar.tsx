/**
 * > [!AML-DOC-FILE]
 * @file        src/editor/StatusBar.tsx
 * @description Bottom bar: engine state, parameter count, and the diagnostics the
 *              engine reported, each clickable to select the layer it concerns.
 * @module      frontend/editor/StatusBar
 * @exports     StatusBar
 * @created     2026-09-30
 * @context     Reads `/health`-derived state from the catalog store, so a stopped
 *              engine is visible rather than appearing as an app that does nothing.
 */

import { useState } from "react";

import { useCatalog, type EngineState } from "../store/catalog";
import { useGraph } from "../store/graph";

/** How many diagnostics the expanded list draws before summarising the remainder. */
const DIAGNOSTIC_LIMIT = 40;

const ENGINE_LABELS: Record<EngineState, { text: string; colour: string }> = {
  connecting: { text: "connecting", colour: "bg-ink-2" },
  warming: { text: "loading TensorFlow", colour: "bg-warn" },
  ready: { text: "engine ready", colour: "bg-good" },
  unreachable: { text: "engine not running", colour: "bg-danger" },
  failed: { text: "engine failed", colour: "bg-danger" },
  "needs-setup": { text: "engine not installed", colour: "bg-warn" },
};

/**
 * > [!AML-DOC-UNIT]
 * The status bar.
 * @returns the bar, with an expandable diagnostics list
 * @sideEffects selects a node when its diagnostic is clicked
 */
export function StatusBar() {
  const engine = useCatalog((state) => state.engine);
  const engineMessage = useCatalog((state) => state.engineMessage);
  const versions = useCatalog((state) => state.engineVersions);
  const diagnostics = useGraph((state) => state.diagnostics);
  const paramsTotal = useGraph((state) => state.paramsTotal);
  const compiled = useGraph((state) => state.compiled);
  const validating = useGraph((state) => state.validating);
  const nodeCount = useGraph((state) => state.nodes.length);
  const setSelection = useGraph((state) => state.setSelection);
  const [expanded, setExpanded] = useState(false);

  const errors = diagnostics.filter((d) => d.severity === "error");
  const warnings = diagnostics.filter((d) => d.severity === "warning");
  const badge = ENGINE_LABELS[engine];

  return (
    <footer className="shrink-0 border-t border-line bg-surface-1 text-[11px]">
      {expanded && diagnostics.length > 0 && (
        <div className="max-h-40 overflow-y-auto border-b border-line">
          {/*
            Only the first few are rendered. An imported model can produce over a
            thousand diagnostics, and a list that long is both slow to draw and no
            more useful than its beginning — the count beside it carries the scale.
          */}
          {diagnostics.slice(0, DIAGNOSTIC_LIMIT).map((diagnostic, index) => (
            <button
              type="button"
              // eslint-disable-next-line react/no-array-index-key
              key={`${diagnostic.code}-${index}`}
              onClick={() => diagnostic.node_id && setSelection([diagnostic.node_id])}
              className="flex w-full items-start gap-2 px-3 py-1 text-left hover:bg-surface-2"
            >
              <span
                className={`mt-1 h-1.5 w-1.5 shrink-0 rounded-full ${
                  diagnostic.severity === "error"
                    ? "bg-danger"
                    : diagnostic.severity === "warning"
                      ? "bg-warn"
                      : "bg-ink-2"
                }`}
              />
              <span className="text-ink-1">{diagnostic.message}</span>
              {diagnostic.node_id && (
                <span className="ml-auto shrink-0 font-mono text-[10px] text-ink-2">
                  {diagnostic.node_id}
                </span>
              )}
            </button>
          ))}
          {diagnostics.length > DIAGNOSTIC_LIMIT && (
            <p className="px-3 py-1.5 text-[10px] text-ink-2">
              and {(diagnostics.length - DIAGNOSTIC_LIMIT).toLocaleString("en-US")} more.
              Fixing the first ones usually clears many of the rest.
            </p>
          )}
        </div>
      )}

      <div className="flex items-center gap-3 px-3 py-1.5">
        <span className="flex items-center gap-1.5" title={engineMessage ?? undefined}>
          <span className={`h-2 w-2 rounded-full ${badge.colour}`} />
          <span className="text-ink-1">{badge.text}</span>
        </span>

        {versions.tensorflow && (
          <span className="text-ink-2">
            tf {String(versions.tensorflow)} · keras {String(versions.keras)}
          </span>
        )}

        <span className="ml-auto flex items-center gap-3">
          <span className="text-ink-2">{nodeCount} layers</span>
          {nodeCount > 0 && (
            <span className={compiled ? "text-good" : "text-ink-2"}>
              {validating
                ? "validating…"
                : compiled
                  ? `${paramsTotal.toLocaleString("en-US")} parameters`
                  : "not valid"}
            </span>
          )}
          {diagnostics.length > 0 && (
            <button
              type="button"
              onClick={() => setExpanded((value) => !value)}
              className="flex items-center gap-2 rounded px-1.5 py-0.5 hover:bg-surface-2"
            >
              {errors.length > 0 && (
                <span className="text-danger">
                  {errors.length.toLocaleString("en-US")} errors
                </span>
              )}
              {warnings.length > 0 && (
                <span className="text-warn">
                  {warnings.length.toLocaleString("en-US")} warnings
                </span>
              )}
              <span className="text-ink-2">{expanded ? "▾" : "▴"}</span>
            </button>
          )}
        </span>
      </div>
    </footer>
  );
}
