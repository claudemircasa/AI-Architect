/**
 * > [!AML-DOC-FILE]
 * @file        src/App.tsx
 * @description Application shell: loads the catalog, then lays out the editor as
 *              palette, canvas, property panel and status bar.
 * @module      frontend/App
 * @exports     App
 * @created     2026-09-30
 * @context     Shows an onboarding panel instead of an empty editor when the engine
 *              is unreachable, so a stopped backend never looks like a broken app
 *              [task 08 item 3].
 */

import { ReactFlowProvider } from "@xyflow/react";
import { useEffect, useState } from "react";

import { engineBaseUrl, insideShell, shellEngineStatus } from "./api/client";
import { Canvas } from "./editor/Canvas";
import { GraphView } from "./graphview/GraphView";
import { Palette } from "./editor/Palette";
import { SidePanel } from "./editor/SidePanel";
import { StatusBar } from "./editor/StatusBar";
import { Toolbar } from "./editor/Toolbar";
import { OptimizePanel } from "./editor/OptimizePanel";
import { BackendPanel } from "./shell/BackendPanel";
import { Setup } from "./shell/Setup";
import { useCatalog } from "./store/catalog";
import { useGraph } from "./store/graph";
import {
  clearAutosave,
  readAutosave,
  restoreAutosave,
  startAutosave,
} from "./store/persistence";
import { applyViewport, currentViewport } from "./editor/viewport";

/**
 * > [!AML-DOC-UNIT]
 * Panel shown when the engine cannot be reached or failed to start.
 * @param message the engine's own explanation, when it gave one
 * @returns copy-pasteable setup instructions
 */
function EngineDown({ message }: { message: string | null }) {
  const shell = shellEngineStatus();
  return (
    <div className="flex h-full items-center justify-center p-8">
      <div className="max-w-xl">
        <h1 className="text-[15px] font-semibold text-ink-0">The engine is not running</h1>
        <p className="mt-2 text-[12px] leading-relaxed text-ink-1">
          AI Architect needs its Python engine for the layer catalog, shape inference and
          export. Nothing was answering on{" "}
          <span className="font-mono text-ink-0">{engineBaseUrl()}</span>.
        </p>
        {message && (
          <p className="mt-2 rounded border border-line bg-surface-1 p-2 font-mono text-[11px] text-danger">
            {message}
          </p>
        )}
        {shell && shell.searched.length > 0 && (
          <details className="mt-3 rounded border border-line bg-surface-1">
            <summary className="cursor-pointer px-2 py-1 text-[10px] uppercase tracking-wide text-ink-2">
              Where the app looked ({shell.searched.length})
            </summary>
            <ul className="border-t border-line p-2 font-mono text-[10px] text-ink-2">
              {shell.searched.map((path) => (
                <li key={path} className="truncate">
                  {path}
                </li>
              ))}
            </ul>
          </details>
        )}
        <p className="mt-4 text-[11px] font-semibold uppercase tracking-wide text-ink-2">
          Start it
        </p>
        <pre className="mt-1 overflow-x-auto rounded border border-line bg-surface-1 p-3 font-mono text-[11px] text-ink-1">
{`./scripts/run-engine.sh`}
        </pre>
        <p className="mt-4 text-[11px] font-semibold uppercase tracking-wide text-ink-2">
          First time here
        </p>
        <pre className="mt-1 overflow-x-auto rounded border border-line bg-surface-1 p-3 font-mono text-[11px] text-ink-1">
{`brew install python@3.13
./scripts/setup-backend.sh`}
        </pre>
        <p className="mt-2 text-[11px] leading-relaxed text-ink-2">
          TensorFlow 2.21 has no build for Python 3.14, so the engine pins 3.13. The setup
          script checks this before installing anything.
        </p>
        <button
          type="button"
          onClick={() => useCatalog.getState().load()}
          className="mt-4 rounded border border-line bg-surface-2 px-3 py-1.5 text-[12px]
                     text-ink-1 hover:text-ink-0"
        >
          Try again
        </button>
      </div>
    </div>
  );
}

/**
 * > [!AML-DOC-UNIT]
 * The application root.
 * @returns the editor, a loading state, or the engine-down panel
 * @sideEffects starts the catalog load once on mount
 */
export function App() {
  const engine = useCatalog((state) => state.engine);
  const message = useCatalog((state) => state.engineMessage);
  const [recovery, setRecovery] = useState(() => readAutosave());
  const view = useGraph((state) => state.view);

  useEffect(() => {
    void useCatalog.getState().load();
  }, []);

  useEffect(() => startAutosave(currentViewport), []);

  // The window title is where a desktop application says which document it is
  // showing, and it is the only place the project's name needs to appear now that
  // the name comes from the file it was saved to.
  //
  // Every hook stays above the early returns below. Having these two under the
  // engine-down branch meant the hook order changed the moment the engine came up,
  // which React will not have.
  const projectName = useGraph((state) => state.name);
  const unsaved = useGraph((state) => state.dirty);
  useEffect(() => {
    const title = `${projectName}${unsaved ? " — edited" : ""} — AI Architect`;
    document.title = title;
    if (!insideShell()) return;
    void import("@tauri-apps/api/window")
      .then(({ getCurrentWindow }) => getCurrentWindow().setTitle(title))
      .catch(() => {
        /* a title is a courtesy; losing it must not break the editor */
      });
  }, [projectName, unsaved]);

  // Opened from the application menu, which knows nothing about the editor.
  const [backendOpen, setBackendOpen] = useState(false);
  const [optimizeOpen, setOptimizeOpen] = useState(false);
  useEffect(() => {
    const backend = () => setBackendOpen(true);
    const optimize = () => setOptimizeOpen(true);
    window.addEventListener("nnarch:backend", backend);
    window.addEventListener("nnarch:optimize", optimize);
    return () => {
      window.removeEventListener("nnarch:backend", backend);
      window.removeEventListener("nnarch:optimize", optimize);
    };
  }, []);

  if (engine === "needs-setup") {
    return (
      <Setup
        onInstalled={() => {
          void import("@tauri-apps/api/core").then(({ invoke }) => invoke("relaunch"));
        }}
      />
    );
  }

  if (engine === "unreachable" || engine === "failed") {
    return <EngineDown message={message} />;
  }

  return (
    <div className="relative flex h-full flex-col">
      {backendOpen && <BackendPanel onClose={() => setBackendOpen(false)} />}
      {optimizeOpen && <OptimizePanel onClose={() => setOptimizeOpen(false)} />}
      <Toolbar />
      {recovery && (
        <div className="flex items-center gap-3 border-b border-line bg-warn/10 px-3 py-2">
          <span className="text-[12px] text-ink-0">
            Unsaved work from {new Date(recovery.savedAt).toLocaleString()} was
            recovered: <span className="font-semibold">{recovery.name}</span>,{" "}
            {recovery.nodes.length} layers.
          </span>
          <button
            type="button"
            className="rounded border border-line bg-surface-2 px-2 py-0.5 text-[11px]
                       text-ink-1 hover:text-ink-0"
            onClick={() => {
              restoreAutosave(recovery);
              applyViewport(recovery.viewport);
              setRecovery(null);
            }}
          >
            Restore it
          </button>
          <button
            type="button"
            className="rounded px-2 py-0.5 text-[11px] text-ink-2 hover:text-ink-1"
            onClick={() => {
              clearAutosave();
              setRecovery(null);
            }}
          >
            Discard
          </button>
        </div>
      )}
      <div className="flex min-h-0 flex-1">
        <Palette />
        <ReactFlowProvider>
          {/*
            The canvas stays mounted while the graph view is on screen: React Flow
            loses its viewport when unmounted, so switching views and back would
            otherwise reset where the user was looking.
          */}
          <div className={view === "architecture" ? "flex min-w-0 flex-1" : "hidden"}>
            <Canvas />
          </div>
          {view === "graph" && <GraphView />}
          <SidePanel />
        </ReactFlowProvider>
      </div>
      <StatusBar />
    </div>
  );
}
