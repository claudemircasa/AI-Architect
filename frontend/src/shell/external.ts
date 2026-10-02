/**
 * > [!AML-DOC-FILE]
 * @file        src/shell/external.ts
 * @description Open a link outside the application.
 * @module      frontend/shell/external
 * @exports     openExternal
 * @created     2026-10-01
 * @context     An `<a href>` to the Keras documentation does nothing inside the
 *              desktop shell. The page is served from `tauri://`, and the web view
 *              refuses to navigate away from the application it is showing — it does
 *              not fail, it simply ignores the click, which is why the links looked
 *              dead rather than broken [E-030]. The shell has to hand the URL to the
 *              operating system instead, which is what the opener plugin does.
 */

import { insideShell } from "../api/client";

/**
 * > [!AML-DOC-UNIT]
 * Open a URL in the user's own browser.
 * @param url the address to open; only `https:` is allowed through
 * @returns resolves once the system has been asked
 * @sideEffects opens a browser window
 * @raises never; a failure is reported to the console rather than thrown, because
 *         nothing the editor is doing depends on the link having opened
 */
export async function openExternal(url: string): Promise<void> {
  // The capability grants `https://*` and nothing else; checking here too means a
  // bad `doc_url` is refused with an explanation rather than silently dropped.
  if (!/^https:\/\//i.test(url)) {
    console.warn(`refusing to open a non-https link: ${url}`);
    return;
  }

  if (!insideShell()) {
    window.open(url, "_blank", "noopener,noreferrer");
    return;
  }

  try {
    const { openUrl } = await import("@tauri-apps/plugin-opener");
    await openUrl(url);
  } catch (error) {
    console.error(`could not open ${url}: ${String(error)}`);
  }
}
