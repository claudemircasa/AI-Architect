/**
 * > [!AML-DOC-FILE]
 * @file        src/shell/BackendPanel.tsx
 * @description What the Backend Components menu item opens: which runtime is in use,
 *              and the way to install it again.
 * @module      frontend/shell/BackendPanel
 * @exports     BackendPanel
 * @created     2026-10-02
 * @context     The runtime is one directory that the app owns, so maintaining it
 *              means replacing it rather than patching it: a reinstall fetches the
 *              current Python and the current TensorFlow and swaps the lot in one
 *              move, which is the same path first launch takes [E-045].
 */

import { useCallback, useEffect, useState } from "react";

import { shellEngineStatus } from "../api/client";
import { useCatalog } from "../store/catalog";

/** Matches `SETUP_EVENT` in Rust. */
const PROGRESS_EVENT = "setup://progress";

/**
 * > [!AML-DOC-UNIT]
 * The backend panel, shown over the editor.
 * @param onClose called when it should go away
 * @returns the panel
 * @sideEffects a reinstall downloads about a gigabyte and restarts the app
 */
export function BackendPanel({ onClose }: { onClose: () => void }) {
  const versions = useCatalog((state) => state.engineVersions);
  const [steps, setSteps] = useState<string[]>([]);
  const [running, setRunning] = useState(false);
  const [failed, setFailed] = useState<string | null>(null);
  const status = shellEngineStatus();

  useEffect(() => {
    let stop: (() => void) | null = null;
    let cancelled = false;
    void (async () => {
      const { listen } = await import("@tauri-apps/api/event");
      const unlisten = await listen<string>(PROGRESS_EVENT, (event) =>
        setSteps((previous) => [...previous, event.payload]),
      );
      if (cancelled) unlisten();
      else stop = unlisten;
    })();
    return () => {
      cancelled = true;
      stop?.();
    };
  }, []);

  const reinstall = useCallback(async () => {
    setRunning(true);
    setFailed(null);
    setSteps([]);
    try {
      const { invoke } = await import("@tauri-apps/api/core");
      await invoke<string>("install_runtime");
      await invoke("relaunch");
    } catch (error) {
      setFailed(error instanceof Error ? error.message : String(error));
      setRunning(false);
    }
  }, []);

  return (
    <div className="absolute inset-0 z-50 flex items-center justify-center bg-surface-0/80 p-8">
      <div className="w-full max-w-md rounded border border-line bg-surface-1 p-4">
        <div className="flex items-baseline justify-between gap-2">
          <h2 className="text-[13px] font-semibold text-ink-0">Backend components</h2>
          <button
            type="button"
            onClick={onClose}
            disabled={running}
            className="text-[11px] text-ink-2 hover:text-ink-0 disabled:opacity-40"
          >
            Close
          </button>
        </div>

        <dl className="mt-3 space-y-1">
          {[
            ["Python", versions.python],
            ["TensorFlow", versions.tensorflow],
            ["Keras", versions.keras],
            ["Engine", versions.nnarch],
            ["Layers", versions.layer_count],
          ].map(([label, value]) => (
            <div key={String(label)} className="flex items-baseline justify-between gap-2">
              <dt className="text-[11px] text-ink-2">{label}</dt>
              <dd className="font-mono text-[11px] text-ink-1">{value ?? "—"}</dd>
            </div>
          ))}
        </dl>

        {status?.runtime && (
          <p className="mt-2 break-all font-mono text-[9px] leading-snug text-ink-2">
            {status.runtime}
          </p>
        )}

        {steps.length > 0 && (
          <ol className="mt-3 space-y-1">
            {steps.map((step, index) => (
              // eslint-disable-next-line react/no-array-index-key
              <li key={index} className="flex items-baseline gap-2 text-[11px]">
                <span className={index === steps.length - 1 && running ? "text-accent" : "text-good"}>
                  {index === steps.length - 1 && running ? "›" : "✓"}
                </span>
                <span className="min-w-0 text-ink-1">{step}</span>
              </li>
            ))}
          </ol>
        )}

        {failed && (
          <p className="mt-3 rounded border border-danger/40 bg-danger/10 p-2 text-[11px] leading-snug text-danger">
            {failed}
          </p>
        )}

        <button
          type="button"
          onClick={() => void reinstall()}
          disabled={running}
          className="mt-4 w-full rounded border border-line bg-surface-2 px-3 py-2
                     text-[12px] text-ink-1 hover:text-ink-0 disabled:opacity-40"
        >
          {running ? "Reinstalling…" : "Reinstall with the current versions"}
        </button>
        <p className="mt-2 text-[10px] leading-relaxed text-ink-2">
          Downloads Python and TensorFlow again and replaces what is installed. Your
          projects are untouched; the app restarts when it finishes.
        </p>
      </div>
    </div>
  );
}
