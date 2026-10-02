/**
 * > [!AML-DOC-FILE]
 * @file        src/shell/commands.ts
 * @description The editor's commands, in one place, so the native menu and the
 *              toolbar invoke the same code rather than two versions of it.
 * @module      frontend/shell/commands
 * @exports     COMMAND_IDS, runCommand, type CommandId, type CommandReport
 * @created     2026-10-01
 * @context     The menu bar is built in Rust and knows nothing about the graph; it
 *              emits the id of whatever was chosen. This is where an id becomes an
 *              action. Keeping the toolbar on the same table is what stops the two
 *              drifting — the failure the export path already taught us once, when
 *              the generated code and the running model named layers differently
 *              [E-029, amm E.4].
 */

import { fitGraphInView, currentViewport, applyViewport } from "../editor/viewport";
import {
  forgetSavedPath,
  importFromKeras,
  openFromFile,
  pickFile,
  saveToFile,
} from "../store/persistence";
import { useGraph } from "../store/graph";
import { openExternal } from "./external";

/** Every command the menu can send. */
export const COMMAND_IDS = [
  "file.new",
  "file.open",
  "file.save",
  "file.saveAs",
  "file.import",
  "file.export",
  "edit.undo",
  "edit.redo",
  "edit.delete",
  "edit.duplicate",
  "view.architecture",
  "view.graph",
  "view.tidy",
  "view.fit",
  "view.zoomIn",
  "view.zoomOut",
  "help.guide",
  "help.keras",
  "help.tensorflow",
  "help.governor",
  "app.backend",
  "model.optimize",
] as const;

export type CommandId = (typeof COMMAND_IDS)[number];

/** What a command wants said in the status line, if anything. */
export type CommandReport = string | null;

const DOCS: Partial<Record<CommandId, string>> = {
  "help.guide": "https://keras.io/guides/functional_api/",
  "help.keras": "https://keras.io/api/layers/",
  "help.tensorflow": "https://www.tensorflow.org/guide",
  "help.governor": "https://governor.ltd",
};

/**
 * > [!AML-DOC-UNIT]
 * Whether the user is typing, in which case editing commands belong to the field.
 * @returns True when a text control has focus
 * @sideEffects none
 */
function typing(): boolean {
  const element = document.activeElement as HTMLElement | null;
  if (!element) return false;
  return (
    element.tagName === "INPUT" ||
    element.tagName === "TEXTAREA" ||
    element.isContentEditable
  );
}

/**
 * > [!AML-DOC-UNIT]
 * Carry out one command.
 * @param id       which command
 * @param fallback for the browser, where there is no native file dialog: asked to
 *                 open a file picker of the given kind
 * @returns a line to show the user, or null when there is nothing to say
 * @raises never; a failure becomes the message it returns
 * @sideEffects depends entirely on the command: most mutate the graph store, and the
 *              file commands touch the disk
 */
export async function runCommand(
  id: CommandId,
  fallback?: (kind: "project" | "model") => void,
): Promise<CommandReport> {
  const store = useGraph.getState();

  switch (id) {
    case "file.new":
      store.replaceGraph({ nodes: [], edges: [] }, "Untitled");
      forgetSavedPath();
      return "new project";

    case "file.open":
    case "file.import": {
      const kind = id === "file.open" ? "project" : "model";
      const file = await pickFile(kind);
      if (!file) {
        // No native dialog here, so hand it back to the hidden file input.
        fallback?.(kind);
        return null;
      }
      if (kind === "project") {
        const { diagnostics, viewport } = await openFromFile(file);
        const blocking = diagnostics.find((d) => d.severity === "error");
        if (blocking) return blocking.message;
        if (viewport) applyViewport(viewport);
        return `opened ${file.name}`;
      }
      const diagnostics = await importFromKeras(file);
      requestAnimationFrame(fitGraphInView);
      const blocking = diagnostics.find((d) => d.severity === "error");
      if (blocking) return blocking.message;
      const warning = diagnostics.find((d) => d.severity === "warning");
      return warning ? warning.message : `imported ${file.name}`;
    }

    case "file.save":
      return saveToFile(currentViewport(), false);

    case "file.saveAs":
      return saveToFile(currentViewport(), true);

    case "file.export":
      // Owned by the toolbar, which holds the dataset choice the export needs.
      window.dispatchEvent(new CustomEvent("nnarch:export"));
      return null;

    case "edit.undo":
      if (typing()) return null;
      store.undo();
      return null;

    case "edit.redo":
      if (typing()) return null;
      store.redo();
      return null;

    case "edit.delete":
      if (typing()) return null;
      store.deleteSelection();
      return null;

    case "edit.duplicate":
      if (typing()) return null;
      store.duplicateSelection();
      return null;

    case "view.architecture":
      store.setView("architecture");
      return null;

    case "view.graph":
      store.setView("graph");
      return null;

    case "view.tidy":
      store.tidy();
      requestAnimationFrame(fitGraphInView);
      return null;

    case "view.fit":
    case "view.zoomIn":
    case "view.zoomOut":
      // The graph view runs its own transform, so whichever surface is on screen
      // answers rather than the canvas always answering.
      window.dispatchEvent(new CustomEvent("nnarch:viewport", { detail: id }));
      return null;

    case "help.guide":
    case "help.keras":
    case "help.tensorflow":
    case "help.governor":
      await openExternal(DOCS[id] ?? "");
      return null;

    case "model.optimize":
      window.dispatchEvent(new CustomEvent("nnarch:optimize"));
      return null;

    case "app.backend":
      // Reinstalling is the only maintenance there is: the runtime is one directory,
      // and replacing it wholesale is simpler and safer than patching it in place.
      window.dispatchEvent(new CustomEvent("nnarch:backend"));
      return null;

    default:
      return null;
  }
}
