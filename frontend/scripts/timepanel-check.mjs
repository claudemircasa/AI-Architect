/**
 * > [!AML-DOC-FILE]
 * @file        scripts/timepanel-check.mjs
 * @description Drives the Time panel in a real browser and asserts that its results
 *              stay inside it however many layers are watched.
 * @module      frontend/scripts/timepanel-check
 * @exports     (executable script; exits non-zero on failure)
 * @created     2026-10-02
 * @context     The panel grew past the bottom of the window as soon as more than one
 *              layer was watched: it had `overflow-y-auto` but no height, so there
 *              was never anything for it to scroll [E-049]. That is geometry, and
 *              geometry can only be checked by reading what a browser computed.
 *              Builds its own small recurrent model, so the check needs no files from
 *              outside the repository. Requires the engine on :8756 and the dev
 *              server on :5173.
 */

import { chromium } from "playwright-core";

const SHOTS = process.argv[2] ?? "./time-shots";
const failures = [];

/**
 * > [!AML-DOC-UNIT]
 * Record an assertion result.
 * @param label     what was checked
 * @param condition whether it held
 * @param detail    observed value, shown either way
 */
function check(label, condition, detail) {
  const line = `${condition ? "PASS" : "FAIL"}  ${label}${detail ? ` — ${detail}` : ""}`;
  console.log(line);
  if (!condition) failures.push(line);
}

const browser = await chromium.launch({ channel: "chrome" });
const page = await browser.newPage({ viewport: { width: 1500, height: 900 } });
page.on("pageerror", (error) => failures.push(`page error: ${error.message}`));
await page.goto("http://localhost:5173/", { waitUntil: "networkidle" });
await page.waitForFunction(() => window.__graphStore, null, { timeout: 20000 });

// A model that carries state: a sequence in, a GRU that takes a starting state and
// hands one back, and a head. The smallest thing the Time panel is actually for.
await page.evaluate(() => {
  const graph = window.__graphStore.getState();
  const node = (id, typeId, params) => ({
    id, type: "layer", position: { x: 240, y: 0 }, selected: false,
    data: { typeId, name: id, params, disabled: false, shape: null, diagnostics: [] },
  });
  const edge = (id, source, target, targetHandle, sourceHandle) => ({
    id, source, target,
    sourceHandle: sourceHandle ?? "output",
    targetHandle: targetHandle ?? "input",
    data: { order: 0 },
  });
  graph.replaceGraph({
    nodes: [
      node("seq", "keras.Input", { shape: [6, 4] }),
      node("state_in", "keras.Input", { shape: [8] }),
      node("gru", "keras.GRU", { units: 8, return_state: true }),
      node("state_out", "keras.Identity", {}),
      node("dense", "keras.Dense", { units: 8, activation: "relu" }),
      node("head", "keras.Dense", { units: 5, activation: "softmax" }),
    ],
    edges: [
      edge("e1", "seq", "gru"),
      edge("e2", "state_in", "gru", "initial_state"),
      edge("e3", "gru", "state_out", "input", "output_1"),
      edge("e4", "gru", "dense"),
      edge("e5", "dense", "head"),
    ],
  }, "Stateful");
});
await page.waitForFunction(() => window.__graphStore.getState().compiled, null, { timeout: 60000 });
check("the stateful model compiles", true);

await page.getByRole("button", { name: /^Time$/ }).click();
await page.waitForTimeout(600);

// Every layer watched, which is the case that burst the panel open.
const watched = await page.evaluate(() => {
  document.querySelector("details > summary")?.click();
  let count = 0;
  for (const label of document.querySelectorAll("details label")) {
    const box = label.querySelector("input");
    if (box && !box.checked) { box.click(); count += 1; }
  }
  return count;
});
check("every layer can be watched at once", watched >= 6, `${watched} watched`);

await page.selectOption("select", "32");
await page.getByRole("button", { name: /Run over time/ }).click();
await page.waitForFunction(
  () => document.body.innerText.includes("steps ×"),
  null,
  { timeout: 180000 },
);
await page.waitForTimeout(800);

const geometry = await page.evaluate(() => {
  const aside = [...document.querySelectorAll("aside")]
    .find((el) => el.textContent?.includes("Run over time"));
  const panel = aside.querySelector("div.flex.h-full.min-w-0.flex-col");
  const scroller = panel.querySelector(':scope > [class*="overflow-y-auto"]');
  const asideBox = aside.getBoundingClientRect();
  const scrollBox = scroller.getBoundingClientRect();
  const traces = [...scroller.querySelectorAll("svg")];
  return {
    clips: ["auto", "scroll"].includes(getComputedStyle(scroller).overflowY),
    within: Math.round(scrollBox.bottom) <= Math.round(asideBox.bottom) + 1,
    scrolls: scroller.scrollHeight > scroller.clientHeight,
    content: scroller.scrollHeight,
    viewport: scroller.clientHeight,
    pageGrew: document.documentElement.scrollHeight > window.innerHeight,
    traces: traces.length,
    tallest: Math.max(...traces.map((t) => Math.round(t.getBoundingClientRect().height))),
  };
});

check("the results region clips its contents", geometry.clips);
check("it ends inside the panel, not below it", geometry.within);
check("it scrolls rather than growing", geometry.scrolls,
  `${geometry.content}px of content in ${geometry.viewport}px`);
check("the page itself did not grow", !geometry.pageGrew);
check("every watched layer was drawn", geometry.traces >= 6, `${geometry.traces} traces`);
check("a long run does not make one layer tower over the rest", geometry.tallest <= 160,
  `tallest ${geometry.tallest}px for 32 steps`);

await page.screenshot({ path: `${SHOTS}/time-01.png` });
await browser.close();

console.log(`\nscreenshots in ${SHOTS}`);
if (failures.length) { console.log(`\n${failures.length} FAILED`); process.exit(1); }
console.log("\nall time-panel checks passed");
