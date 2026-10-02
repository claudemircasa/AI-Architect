/**
 * > [!AML-DOC-FILE]
 * @file        src/editor/Toolbar.tsx
 * @description Top bar: the project's name, the view switch, and the quick tools.
 * @module      frontend/editor/Toolbar
 * @exports     Toolbar
 * @created     2026-09-30
 * @context     Inside the desktop shell the standard commands live in the system
 *              menu bar, where the platform puts them, and this bar keeps only what
 *              a hand reaches for repeatedly — undo, tidy, fit, the view switch — as
 *              icons. A browser has no menu bar, so there the same commands appear
 *              here as well; both routes call `runCommand`, so there is one
 *              implementation and not two [E-029].
 *
 *              The project's name is shown, not typed. A project is named when it is
 *              saved, by the save dialog, which is the moment it acquires a place to
 *              be named after [task 08].
 */

import { useCallback, useEffect, useRef, useState } from "react";

import { EngineError, exportProject } from "../api/client";
import { runCommand, type CommandId } from "../shell/commands";
import { nativeMenuOwnsShortcuts, useNativeMenu } from "../shell/menu";
import { useGraph } from "../store/graph";
import { importFromKeras, openFromFile } from "../store/persistence";
import { applyViewport, fitGraphInView } from "./viewport";

/**
 * > [!AML-DOC-UNIT]
 * One icon button.
 * @param glyph    the mark to draw, as SVG path data
 * @param title    tooltip, which also carries the keyboard shortcut
 * @param onClick  what it does
 * @param disabled whether it is unavailable
 * @param active   whether it shows as the current choice
 * @returns the button element
 */
function Tool({
  glyph,
  title,
  onClick,
  disabled = false,
  active = false,
}: {
  glyph: string;
  title: string;
  onClick: () => void;
  disabled?: boolean;
  active?: boolean;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      title={title}
      aria-label={title}
      aria-pressed={active}
      className={`flex h-7 w-7 items-center justify-center rounded border transition-colors
        ${
          active
            ? "border-accent/60 bg-surface-3 text-ink-0"
            : "border-transparent text-ink-2 hover:bg-surface-2 hover:text-ink-0"
        }
        disabled:cursor-default disabled:opacity-30 disabled:hover:bg-transparent`}
    >
      <svg
        viewBox="0 0 20 20"
        className="h-4 w-4"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
        strokeLinejoin="round"
        aria-hidden="true"
      >
        <path d={glyph} />
      </svg>
    </button>
  );
}

/** Icon paths, drawn on the same 20×20 grid at the same weight. */
const ICONS = {
  undo: "M7 7 4 10l3 3M4 10h8a4 4 0 0 1 0 8h-1",
  redo: "m13 7 3 3-3 3m3-3H8a4 4 0 0 0 0 8h1",
  tidy: "M10 3v4m0 0H5v3m5-3h5v3M3 10h4v3H3zm6 0h2v3H9zm6 0h2v3h-2z",
  fit: "M4 7V4h3M16 7V4h-3M4 13v3h3m9-3v3h-3",
  architecture: "M4 4h5v5H4zm7 7h5v5h-5zM6.5 9v2m0 0h7m0 0v0",
  graph: "M6 5a2 2 0 1 0 0 4 2 2 0 0 0 0-4Zm8 6a2 2 0 1 0 0 4 2 2 0 0 0 0-4ZM7.6 8.4l4.8 4.2",
  save: "M4 4h9l3 3v9H4zM7 4v4h6M7 16v-4h6v4",
  open: "M3 6h5l2 2h7v8H3zm0 0v10",
  add: "M10 5v10M5 10h10",
  download: "M10 4v8m0 0 3-3m-3 3-3-3M4 16h12",
  upload: "M10 16V8m0 0 3 3M10 8 7 11M4 4h12",
} as const;

/**
 * > [!AML-DOC-UNIT]
 * The top bar.
 * @returns the project name, the view switch and the quick tools
 * @sideEffects runs editor commands; export downloads an archive
 */
export function Toolbar() {
  const name = useGraph((state) => state.name);
  const undo = useGraph((state) => state.undo);
  const redo = useGraph((state) => state.redo);
  const past = useGraph((state) => state.past.length);
  const future = useGraph((state) => state.future.length);
  const compiled = useGraph((state) => state.compiled);
  const tidy = useGraph((state) => state.tidy);
  const view = useGraph((state) => state.view);
  const setView = useGraph((state) => state.setView);
  const nodeCount = useGraph((state) => state.nodes.length);
  const dirty = useGraph((state) => state.dirty);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const openInput = useRef<HTMLInputElement>(null);
  const importInput = useRef<HTMLInputElement>(null);

  const inShell = nativeMenuOwnsShortcuts();

  /**
   * > [!AML-DOC-UNIT]
   * Open the hidden file input, for a browser with no native dialog.
   * @param kind which picker to open
   * @sideEffects opens the operating system's file chooser
   */
  const fallback = useCallback((kind: "project" | "model") => {
    (kind === "project" ? openInput : importInput).current?.click();
  }, []);

  useNativeMenu(setMessage, fallback);

  // Export is the one command the menu cannot carry out by itself: it needs the
  // dataset choice this component holds, so the menu asks and this answers.
  useEffect(() => {
    const onAsked = () => void onExport();
    window.addEventListener("nnarch:export", onAsked);
    return () => window.removeEventListener("nnarch:export", onAsked);
  });

  /**
   * > [!AML-DOC-UNIT]
   * Run a command and report whatever it says.
   * @param id the command
   * @sideEffects sets the busy flag and the status message
   */
  async function invoke(id: CommandId) {
    setBusy(true);
    try {
      setMessage(await runCommand(id, fallback));
    } catch (error) {
      if (error instanceof EngineError) {
        const detail = error.detail as { diagnostics?: { message: string }[] } | null;
        setMessage(detail?.diagnostics?.[0]?.message ?? error.message);
      } else {
        setMessage(error instanceof Error ? error.message : "something went wrong");
      }
    } finally {
      setBusy(false);
    }
  }

  /**
   * > [!AML-DOC-UNIT]
   * Read a file the user chose through a hidden input.
   * @param event  the change event
   * @param handle what to do with the file
   * @sideEffects clears the input so choosing the same file twice still fires
   */
  function onPicked(
    event: React.ChangeEvent<HTMLInputElement>,
    handle: (file: File) => Promise<string>,
  ) {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    setBusy(true);
    handle(file)
      .then(setMessage)
      .catch((error: unknown) =>
        setMessage(error instanceof Error ? error.message : "could not read that file"),
      )
      .finally(() => setBusy(false));
  }

  /**
   * > [!AML-DOC-UNIT]
   * Export the current graph as a standalone project.
   * @sideEffects calls the engine, then downloads the archive
   */
  async function onExport() {
    setBusy(true);
    setMessage(null);
    try {
      const { bytes, filename } = await exportProject(useGraph.getState().toIR(), {
        dataset: { kind: "synthetic" },
      });
      const url = URL.createObjectURL(
        new Blob([bytes as BlobPart], { type: "application/zip" }),
      );
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = filename;
      anchor.click();
      URL.revokeObjectURL(url);
      setMessage(`exported ${filename}`);
    } catch (error) {
      if (error instanceof EngineError) {
        const detail = error.detail as { diagnostics?: { message: string }[] } | null;
        setMessage(detail?.diagnostics?.[0]?.message ?? "export failed");
      } else {
        setMessage(error instanceof Error ? error.message : "export failed");
      }
    } finally {
      setBusy(false);
    }
  }

  return (
    <header className="flex shrink-0 items-center gap-2 border-b border-line bg-surface-1 px-3 py-1.5">
      <span className="text-[12px] font-semibold tracking-tight text-ink-0">AI Architect</span>

      <span className="ml-1 flex items-baseline gap-1.5">
        <span className="text-[12px] text-ink-1" title="Named when the project is saved">
          {name}
        </span>
        {dirty && (
          <span className="h-1.5 w-1.5 rounded-full bg-ink-2" title="Unsaved changes" />
        )}
      </span>

      <span className="ml-3 flex items-center gap-0.5 rounded border border-line bg-surface-2 p-0.5">
        <Tool
          glyph={ICONS.architecture}
          title="Architecture (⌘1)"
          onClick={() => setView("architecture")}
          active={view === "architecture"}
        />
        <Tool
          glyph={ICONS.graph}
          title="Graph (⌘2)"
          onClick={() => setView("graph")}
          active={view === "graph"}
        />
      </span>

      <span className="ml-2 flex items-center gap-0.5">
        <Tool glyph={ICONS.undo} title="Undo (⌘Z)" onClick={undo} disabled={past === 0} />
        <Tool glyph={ICONS.redo} title="Redo (⇧⌘Z)" onClick={redo} disabled={future === 0} />
        <Tool
          glyph={ICONS.tidy}
          title="Tidy layout (⌘L)"
          onClick={() => {
            tidy();
            requestAnimationFrame(fitGraphInView);
          }}
          disabled={nodeCount === 0}
        />
        <Tool
          glyph={ICONS.fit}
          title="Fit to window (⌘0)"
          onClick={() =>
            window.dispatchEvent(new CustomEvent("nnarch:viewport", { detail: "view.fit" }))
          }
          disabled={nodeCount === 0}
        />
      </span>

      {/*
        A browser has no menu bar, so the standard commands appear here instead. In
        the shell they are in the File menu and this group is left out, which is what
        keeps the bar to the quick tools it is meant to hold.
      */}
      {!inShell && (
        <span className="ml-2 flex items-center gap-0.5 border-l border-line pl-2">
          <Tool glyph={ICONS.add} title="New project (⌘N)" onClick={() => void invoke("file.new")} />
          <Tool glyph={ICONS.open} title="Open project (⌘O)" onClick={() => void invoke("file.open")} />
          <Tool
            glyph={ICONS.save}
            title="Save (⌘S)"
            onClick={() => void invoke("file.save")}
            disabled={busy || nodeCount === 0}
          />
          <Tool
            glyph={ICONS.upload}
            title="Import a .keras, .tflite or to_json() model (⌘I)"
            onClick={() => void invoke("file.import")}
          />
        </span>
      )}

      <input
        ref={openInput}
        type="file"
        accept=".nnarch,application/zip"
        className="hidden"
        onChange={(event) =>
          onPicked(event, async (file) => {
            const { diagnostics, viewport } = await openFromFile(file);
            const blocking = diagnostics.find((d) => d.severity === "error");
            if (blocking) return blocking.message;
            if (viewport) applyViewport(viewport);
            return `opened ${file.name}`;
          })
        }
      />
      <input
        ref={importInput}
        type="file"
        accept=".keras,.tflite,.json,application/json,application/octet-stream"
        className="hidden"
        onChange={(event) =>
          onPicked(event, async (file) => {
            const diagnostics = await importFromKeras(file);
            requestAnimationFrame(fitGraphInView);
            const blocking = diagnostics.find((d) => d.severity === "error");
            if (blocking) return blocking.message;
            // A lossy import is worth saying out loud: a .tflite graph usually needs
            // correcting, and silence would read as a faithful round trip.
            const warning = diagnostics.find((d) => d.severity === "warning");
            return warning ? warning.message : `imported ${file.name}`;
          })
        }
      />

      <span className="ml-auto flex items-center gap-2">
        {message && <span className="max-w-xs truncate text-[10px] text-ink-2">{message}</span>}
        <button
          type="button"
          onClick={onExport}
          disabled={busy || !compiled}
          title={compiled ? "Download a standalone trainable project (⌘E)" : "Fix the errors first"}
          className="flex items-center gap-1.5 rounded border border-line bg-surface-2 px-2 py-1
                     text-[11px] text-ink-1 hover:text-ink-0 disabled:opacity-40"
        >
          <svg viewBox="0 0 20 20" className="h-3.5 w-3.5" fill="none" stroke="currentColor"
               strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
            <path d={ICONS.download} />
          </svg>
          {busy ? "Working…" : "Export"}
        </button>
      </span>
    </header>
  );
}
