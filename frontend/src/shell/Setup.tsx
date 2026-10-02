/**
 * > [!AML-DOC-FILE]
 * @file        src/shell/Setup.tsx
 * @description The first-launch screen: installs the engine's Python runtime and
 *              reports what it is doing while it does it.
 * @module      frontend/shell/Setup
 * @exports     Setup
 * @created     2026-10-02
 * @context     The application ships without Python or TensorFlow, which takes the
 *              download from 326MB to about 8MB [E-045]. The cost is that the first
 *              launch has to fetch them, and a progress bar that only spins would
 *              turn several minutes into several minutes of doubt — so each step says
 *              what it is, and the long one says that it is the long one.
 */

import { useCallback, useEffect, useState } from "react";

import { openExternal } from "./external";

/** What the shell emits per installation step, matching `SETUP_EVENT` in Rust. */
const PROGRESS_EVENT = "setup://progress";

/**
 * > [!AML-DOC-UNIT]
 * The setup screen.
 * @param onInstalled called once the runtime is in place
 * @returns the screen
 * @sideEffects downloads about a gigabyte when the user starts it
 */
export function Setup({ onInstalled }: { onInstalled: () => void }) {
  const [steps, setSteps] = useState<string[]>([]);
  const [running, setRunning] = useState(false);
  const [failed, setFailed] = useState<string | null>(null);
  const [done, setDone] = useState(false);

  useEffect(() => {
    let stop: (() => void) | null = null;
    let cancelled = false;
    void (async () => {
      const { listen } = await import("@tauri-apps/api/event");
      const unlisten = await listen<string>(PROGRESS_EVENT, (event) => {
        setSteps((previous) => [...previous, event.payload]);
      });
      if (cancelled) unlisten();
      else stop = unlisten;
    })();
    return () => {
      cancelled = true;
      stop?.();
    };
  }, []);

  const install = useCallback(async () => {
    setRunning(true);
    setFailed(null);
    setSteps([]);
    try {
      const { invoke } = await import("@tauri-apps/api/core");
      await invoke<string>("install_runtime");
      setDone(true);
      onInstalled();
    } catch (error) {
      setFailed(error instanceof Error ? error.message : String(error));
    } finally {
      setRunning(false);
    }
  }, [onInstalled]);

  return (
    <div className="flex min-h-0 flex-1 items-center justify-center bg-surface-0 p-8">
      <div className="w-full max-w-md">
        <h1 className="text-[15px] font-semibold text-ink-0">AI Architect</h1>
        <p className="mt-1 text-[11px] leading-relaxed text-ink-2">
          One thing before you start. The engine runs on Python and TensorFlow, which
          are about a gigabyte together — too much to put in a download most people
          would never unpack. They are fetched now, once, and kept for every later
          launch.
        </p>

        {!running && !done && !failed && (
          <button
            type="button"
            onClick={() => void install()}
            className="mt-4 w-full rounded border border-accent/60 bg-surface-2 px-3 py-2
                       text-[12px] text-ink-0 hover:bg-surface-3"
          >
            Install the engine
          </button>
        )}

        {(running || done) && (
          <ol className="mt-4 space-y-1.5">
            {steps.map((step, index) => (
              <li
                // eslint-disable-next-line react/no-array-index-key
                key={index}
                className="flex items-baseline gap-2 text-[11px] leading-snug"
              >
                <span
                  className={
                    index === steps.length - 1 && running
                      ? "text-accent"
                      : "text-good"
                  }
                >
                  {index === steps.length - 1 && running ? "›" : "✓"}
                </span>
                <span className="min-w-0 text-ink-1">{step}</span>
              </li>
            ))}
            {running && steps.length === 0 && (
              <li className="text-[11px] text-ink-2">Starting…</li>
            )}
          </ol>
        )}

        {running && (
          <p className="mt-3 text-[10px] leading-relaxed text-ink-2">
            Downloading TensorFlow takes a few minutes on most connections. You can
            leave this window; it will not stop.
          </p>
        )}

        {failed && (
          <div className="mt-4 rounded border border-danger/40 bg-danger/10 p-2">
            <p className="text-[11px] leading-snug text-danger">{failed}</p>
            <button
              type="button"
              onClick={() => void install()}
              className="mt-2 rounded border border-line bg-surface-2 px-2 py-1
                         text-[11px] text-ink-1 hover:text-ink-0"
            >
              Try again
            </button>
          </div>
        )}

        {done && (
          <p className="mt-3 text-[11px] text-good">
            Installed. Restarting to pick it up…
          </p>
        )}

        <p className="mt-6 border-t border-line pt-3 text-[10px] leading-relaxed text-ink-2">
          Nothing is sent anywhere: the download comes from python-build-standalone
          and PyPI, and the engine only ever listens on this machine.{" "}
          <button
            type="button"
            onClick={() => void openExternal("https://governor.ltd")}
            className="text-accent hover:underline"
          >
            governor.ltd
          </button>
        </p>
      </div>
    </div>
  );
}
