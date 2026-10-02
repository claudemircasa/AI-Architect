/**
 * > [!AML-DOC-FILE]
 * @file        src/store/persistence.ts
 * @description Saving and opening `.nnarch` projects, importing Keras models, and
 *              the autosave that survives a reload or a crash.
 * @module      frontend/store/persistence
 * @exports     saveToFile, openFromFile, importFromKeras, readAutosave,
 *              clearAutosave, startAutosave, AUTOSAVE_INTERVAL_MS
 * @created     2026-09-30
 * @context     RISK:MED user data [amm: B.2]. Autosave lives in `localStorage`
 *              rather than on disk: the desktop shell's file access is unavailable
 *              [task 08], and the browser has nowhere else that survives a reload.
 *              It is a recovery net, never the record — losing it must cost the user
 *              nothing they had saved.
 */

import type { Diagnostic, GraphIR, Viewport } from "../api/types";
import { importKerasModel, insideShell, openProject, saveProject } from "../api/client";
import { useGraph, type LayerEdge, type LayerNode } from "./graph";

const AUTOSAVE_KEY = "nnarch.autosave.v1";
export const AUTOSAVE_INTERVAL_MS = 60_000;

interface Autosave {
  savedAt: string;
  name: string;
  nodes: LayerNode[];
  edges: LayerEdge[];
  viewport: Viewport;
}

/**
 * > [!AML-DOC-UNIT]
 * Hand the user a file the browser downloads.
 * @param bytes    file contents
 * @param filename name offered in the save dialog
 * @param type     media type
 * @sideEffects creates and revokes an object URL, and clicks a hidden link
 */
function download(bytes: Uint8Array, filename: string, type: string): void {
  const url = URL.createObjectURL(new Blob([bytes as BlobPart], { type }));
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}

/**
 * > [!AML-DOC-UNIT]
 * Save the current project to a `.nnarch` file.
 * @param viewport the canvas view to store with it
 * @returns the filename that was offered
 * @raises EngineError when the engine refuses the request
 * @sideEffects downloads a file and clears the dirty flag
 */
export async function saveToFile(viewport: Viewport, askWhere = false): Promise<string> {
  const state = useGraph.getState();
  const { bytes, filename } = await saveProject(state.toIR(), viewport);

  if (!insideShell()) {
    download(bytes, filename, "application/zip");
    useGraph.setState({ dirty: false });
    return filename;
  }

  // In the shell a project has a real place on disk. The first save asks where,
  // and the name the user types there becomes the project's name — which is the
  // only moment a project is ever named, so there is no field to type it into
  // beforehand [task 08].
  const { save } = await import("@tauri-apps/plugin-dialog");
  const { writeFile } = await import("@tauri-apps/plugin-fs");

  let target = askWhere ? null : savedPath;
  if (!target) {
    target = await save({
      defaultPath: savedPath ?? filename,
      filters: [{ name: "AI Architect project", extensions: ["nnarch"] }],
    });
  }
  if (!target) return "save cancelled";

  await writeFile(target, bytes);
  savedPath = target;
  useGraph.setState({ name: projectName(target), dirty: false });
  return `saved ${projectName(target)}`;
}

/** Where this project was last written, so a plain Save does not ask again. */
let savedPath: string | null = null;

/**
 * > [!AML-DOC-UNIT]
 * The project name a file path implies.
 * @param path full path to a `.nnarch` file
 * @returns the base name without its extension
 * @sideEffects none
 */
function projectName(path: string): string {
  const base = path.split("/").pop() ?? path;
  return base.replace(/\.nnarch$/i, "");
}

/**
 * > [!AML-DOC-UNIT]
 * Forget where this project lives, so the next save asks again.
 * @sideEffects clears the remembered path
 */
export function forgetSavedPath(): void {
  savedPath = null;
}

/**
 * > [!AML-DOC-UNIT]
 * Ask the user for a file, using the platform's own dialog when there is one.
 * @param kind which kind of file to ask for
 * @returns the chosen file, or null when the user cancelled
 * @raises never; a cancelled dialog is a null, not an error
 * @sideEffects reads the chosen file from disk
 * @context Returns a `File` either way, so everything downstream — the engine call,
 *          the diagnostics, the error paths — is one implementation rather than one
 *          for the browser and a second for the shell.
 */
export async function pickFile(kind: "project" | "model"): Promise<File | null> {
  if (!insideShell()) return null;

  const { open } = await import("@tauri-apps/plugin-dialog");
  const { readFile } = await import("@tauri-apps/plugin-fs");

  const filters =
    kind === "project"
      ? [{ name: "AI Architect project", extensions: ["nnarch"] }]
      : [{ name: "Model", extensions: ["keras", "tflite", "json"] }];

  const chosen = await open({ multiple: false, filters });
  if (typeof chosen !== "string") return null;

  const bytes = await readFile(chosen);
  const file = new File([bytes as BlobPart], chosen.split("/").pop() ?? "model");
  if (kind === "project") savedPath = chosen;
  return file;
}

/**
 * > [!AML-DOC-UNIT]
 * Rebuild canvas nodes and edges from a stored graph.
 * @param graph the architecture as it was saved
 * @returns nodes and edges in the shape React Flow expects
 * @sideEffects none
 * @context Shapes and diagnostics are deliberately not restored: they are the
 *          engine's answer about this graph, and asking again is both cheap and the
 *          only way to be sure they still hold.
 */
export function graphFromIR(graph: GraphIR): { nodes: LayerNode[]; edges: LayerEdge[] } {
  return {
    nodes: graph.nodes.map((node) => ({
      id: node.id,
      type: "layer" as const,
      position: node.position,
      selected: false,
      data: {
        typeId: node.type,
        name: node.name,
        params: node.params,
        disabled: node.disabled,
        shape: null,
        diagnostics: [],
      },
    })),
    edges: graph.edges.map((edge) => ({
      id: edge.id,
      source: edge.source,
      sourceHandle: edge.source_port,
      target: edge.target,
      targetHandle: edge.target_port,
      data: { order: edge.order },
    })),
  };
}

/**
 * > [!AML-DOC-UNIT]
 * Open a `.nnarch` file the user chose.
 * @param file the archive
 * @returns the diagnostics the engine reported, and the viewport to restore
 * @raises EngineError only on transport failure
 * @sideEffects replaces the whole graph when the file loaded, and leaves the current
 *              work untouched when it did not
 */
export async function openFromFile(
  file: File,
): Promise<{ diagnostics: Diagnostic[]; viewport: Viewport | null }> {
  const { project, diagnostics } = await openProject(file);
  if (!project) return { diagnostics, viewport: null };
  useGraph.getState().replaceGraph(graphFromIR(project.graph), project.graph.name);
  return { diagnostics, viewport: project.viewport };
}

/**
 * > [!AML-DOC-UNIT]
 * Import an existing saved model.
 * @param file a `.keras` archive, a `.tflite` model, or `Model.to_json()` output
 * @returns the diagnostics the engine reported, which say how much of the original
 *          that format was able to carry
 * @sideEffects replaces the whole graph when the file could be read
 */
export async function importFromKeras(file: File): Promise<Diagnostic[]> {
  const { graph, diagnostics } = await importKerasModel(file);
  if (graph) useGraph.getState().replaceGraph(graphFromIR(graph), graph.name);
  return diagnostics;
}

/**
 * > [!AML-DOC-UNIT]
 * Read the autosave left by a previous session.
 * @returns the stored snapshot, or null when there is none or it is unreadable
 * @sideEffects none
 * @context Every access is guarded: `localStorage` throws in a private window and
 *          returns nothing when site data was cleared, and neither is a reason for
 *          the editor to fail to start.
 */
export function readAutosave(): Autosave | null {
  try {
    const raw = window.localStorage.getItem(AUTOSAVE_KEY);
    return raw ? (JSON.parse(raw) as Autosave) : null;
  } catch {
    return null;
  }
}

/**
 * > [!AML-DOC-UNIT]
 * Restore a previously autosaved snapshot onto the canvas.
 * @param snapshot the stored snapshot
 * @sideEffects replaces the whole graph
 */
export function restoreAutosave(snapshot: Autosave): void {
  useGraph
    .getState()
    .replaceGraph({ nodes: snapshot.nodes, edges: snapshot.edges }, snapshot.name);
  useGraph.setState({ dirty: true });
}

/**
 * > [!AML-DOC-UNIT]
 * Discard the autosave.
 * @sideEffects removes the stored snapshot
 */
export function clearAutosave(): void {
  try {
    window.localStorage.removeItem(AUTOSAVE_KEY);
  } catch {
    /* nothing to clear is not a failure */
  }
}

/**
 * > [!AML-DOC-UNIT]
 * Begin autosaving the canvas.
 * @param getViewport reads the current canvas view at the moment of each save
 * @returns a function that stops autosaving
 * @sideEffects writes to `localStorage` on a timer and when the window loses focus,
 *              and registers two event listeners
 */
export function startAutosave(getViewport: () => Viewport): () => void {
  function write(): void {
    const state = useGraph.getState();
    if (state.nodes.length === 0) return;
    const snapshot: Autosave = {
      savedAt: new Date().toISOString(),
      name: state.name,
      nodes: state.nodes,
      edges: state.edges,
      viewport: getViewport(),
    };
    try {
      window.localStorage.setItem(AUTOSAVE_KEY, JSON.stringify(snapshot));
    } catch {
      /* a full or unavailable store must not interrupt editing */
    }
  }

  const timer = window.setInterval(write, AUTOSAVE_INTERVAL_MS);
  window.addEventListener("blur", write);
  window.addEventListener("beforeunload", write);
  return () => {
    window.clearInterval(timer);
    window.removeEventListener("blur", write);
    window.removeEventListener("beforeunload", write);
  };
}
