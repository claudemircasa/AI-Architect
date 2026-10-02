/**
 * > [!AML-DOC-FILE]
 * @file        src/shell/menu.ts
 * @description Listen to the native menu bar and run what it asks for.
 * @module      frontend/shell/menu
 * @exports     useNativeMenu, nativeMenuOwnsShortcuts
 * @created     2026-10-01
 * @context     The menu is built in Rust, where the platform wants it, and emits the
 *              id of the chosen item. Its accelerators are registered with the
 *              operating system, so inside the shell they never reach the page —
 *              which means the page's own handler for those keys must stand down, or
 *              one press would undo twice [E-031].
 */

import { useEffect } from "react";

import { insideShell } from "../api/client";
import { runCommand, type CommandId } from "./commands";

/** The event the shell emits, matching `menu::ITEM_EVENT` in Rust. */
const ITEM_EVENT = "menu://item";

/**
 * > [!AML-DOC-UNIT]
 * Whether the native menu holds the keyboard shortcuts.
 * @returns True inside the desktop shell, where the menu bar exists
 * @sideEffects none
 */
export function nativeMenuOwnsShortcuts(): boolean {
  return insideShell();
}

/**
 * > [!AML-DOC-UNIT]
 * Subscribe to the native menu for as long as the editor is mounted.
 * @param report   called with whatever a command wants said in the status line
 * @param fallback opens a hidden file input, for commands with no native dialog
 * @sideEffects registers a listener with the shell and removes it on unmount
 */
export function useNativeMenu(
  report: (message: string | null) => void,
  fallback: (kind: "project" | "model") => void,
): void {
  useEffect(() => {
    if (!insideShell()) return undefined;

    let stop: (() => void) | null = null;
    let cancelled = false;

    void (async () => {
      const { listen } = await import("@tauri-apps/api/event");
      const unlisten = await listen<string>(ITEM_EVENT, (event) => {
        void runCommand(event.payload as CommandId, fallback)
          .then(report)
          .catch((error: unknown) =>
            report(error instanceof Error ? error.message : String(error)),
          );
      });
      if (cancelled) unlisten();
      else stop = unlisten;
    })();

    return () => {
      cancelled = true;
      stop?.();
    };
  }, [report, fallback]);
}
