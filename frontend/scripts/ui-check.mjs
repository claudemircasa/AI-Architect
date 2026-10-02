/**
 * > [!AML-DOC-FILE]
 * @file        scripts/ui-check.mjs
 * @description Drives the editor in a real browser and asserts what a user would
 *              see: layers placed, edges drawn, shapes arriving from the engine,
 *              errors surfacing, and undo restoring state.
 * @module      frontend/scripts/ui-check
 * @exports     (executable script; exits non-zero on failure)
 * @created     2026-09-30
 * @context     Covers the task 09 verification gates, which cannot be checked by
 *              typechecking. Uses the installed Google Chrome through
 *              `playwright-core`, so no browser download is needed. Requires the
 *              engine on :8756 and the dev server on :5173.
 */

import { mkdirSync } from "node:fs";
import { chromium } from "playwright-core";

const SHOTS = process.argv[2] ?? "./ui-shots";
// `localhost`, not `127.0.0.1`: vite binds IPv4 or IPv6 depending on the run, and
// a check that only knows one of them fails for a reason that is not the product.
const APP = "http://localhost:5173/";
const failures = [];
const notes = [];

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

/**
 * > [!AML-DOC-UNIT]
 * Connect two nodes by dragging from one handle to the other, the way a user does.
 * @param page   the Playwright page
 * @param source id of the producing node
 * @param target id of the consuming node
 * @param port   target handle id, for layers with several named inputs
 */
async function connect(page, source, target, port) {
  const from = page.locator(`[data-id="${source}"] .react-flow__handle-bottom`);
  const to = port
    ? page.locator(`[data-id="${target}"] [data-handleid="${port}"]`)
    : page.locator(`[data-id="${target}"] .react-flow__handle-top`);
  const a = await from.boundingBox();
  const b = await to.boundingBox();
  await page.mouse.move(a.x + a.width / 2, a.y + a.height / 2);
  await page.mouse.down();
  await page.mouse.move(b.x + b.width / 2, b.y + b.height / 2, { steps: 12 });
  await page.mouse.up();
}

/**
 * > [!AML-DOC-UNIT]
 * Read the shape chip each node is showing.
 * @param page the Playwright page
 * @returns list of "<layer name> -> <chip text>"
 */
async function chips(page) {
  return page.locator(".react-flow__node").evaluateAll((elements) =>
    elements.map((element) => {
      const name = element.querySelector(".font-semibold")?.textContent ?? "?";
      const chip = element.querySelector(".font-mono")?.textContent ?? "-";
      return `${name} -> ${chip}`;
    }),
  );
}

mkdirSync(SHOTS, { recursive: true });
const browser = await chromium.launch({ channel: "chrome" });
const page = await browser.newPage({ viewport: { width: 1600, height: 1000 } });
const consoleErrors = [];
page.on("console", (message) => message.type() === "error" && consoleErrors.push(message.text()));
page.on("pageerror", (error) => consoleErrors.push(`pageerror: ${error.message}`));
page.on("response", (response) => {
  if (response.status() >= 400) consoleErrors.push(`${response.status()} ${response.url()}`);
});

await page.goto(APP, { waitUntil: "networkidle" });

await page.waitForSelector("text=engine ready", { timeout: 90000 });
const paletteCount = (await page.textContent("aside >> text=/of \\d+ layers/")).trim();
check("catalog loads and the palette fills", /of 1\d\d layers/.test(paletteCount), paletteCount);

// The research families must be reachable and placeable like any other layer.
await page.fill('input[placeholder="Search layers…"]', "mamba");
const mambaVisible = await page.locator('button[data-layer-id="research.MambaBlock"]').count();
check("research layers appear in the palette", mambaVisible === 1, `${mambaVisible} match for "mamba"`);

await page.fill('input[placeholder="Search layers…"]', "kolmogorov");
const kanHits = await page.locator("aside button[data-layer-id^='research.']").count();
check("searching by concept finds them", kanHits >= 1, `${kanHits} matches for "kolmogorov"`);
await page.fill('input[placeholder="Search layers…"]', "");

/**
 * > [!AML-DOC-UNIT]
 * Add a layer by searching the palette and clicking its entry.
 * @param page    the Playwright page
 * @param search  text typed into the palette filter
 * @param layerId catalog id of the entry to click
 */
async function addLayer(page, search, layerId) {
  await page.fill('input[placeholder="Search layers…"]', search);
  await page.click(`button[data-layer-id="${layerId}"]`);
}

await addLayer(page, "input", "keras.Input");
await addLayer(page, "conv 2d", "keras.Conv2D");
await addLayer(page, "max pooling 2d", "keras.MaxPooling2D");
await addLayer(page, "flatten", "keras.Flatten");
await addLayer(page, "dense", "keras.Dense");
await page.fill('input[placeholder="Search layers…"]', "");

const ids = await page
  .locator(".react-flow__node")
  .evaluateAll((els) => els.map((el) => el.getAttribute("data-id")));
check("five layers placed from the palette", ids.length === 5, ids.join(", "));

await page.waitForTimeout(1500);
check(
  "clicking layers in sequence chains them together",
  (await page.locator(".react-flow__edge").count()) === 4,
  `${await page.locator(".react-flow__edge").count()} edges`,
);

await page.getByRole("button", { name: /^Tidy layout/ }).click();
await page.waitForTimeout(800);
const tidied = await page.evaluate(() =>
  window.__graphStore.getState().nodes.map((node) => node.position.y),
);
check("tidy layout ranks the nodes top to bottom",
  tidied.every((y, index) => index === 0 || y > tidied[index - 1]),
  tidied.join(", "));

const built = await chips(page);
notes.push(...built.map((line) => `    ${line}`));
check(
  "shape chips carry values the engine inferred",
  built.some((line) => line.includes("26 × 26 × 32")) &&
    built.some((line) => line.includes("13 × 13 × 32")) &&
    built.some((line) => line.includes("5408")),
  built.join(" | "),
);

const footer = (await page.textContent("footer")).replace(/\s+/g, " ").trim();
check("status bar reports a compiled model with a parameter count",
  /parameters/.test(footer) && !/not valid/.test(footer), footer);
await page.screenshot({ path: `${SHOTS}/ui-01-built.png` });

await page.click(`[data-id="${ids[1]}"]`);
await page.waitForTimeout(300);
const panel = (await page.textContent("aside:last-of-type")).replace(/\s+/g, " ");
check("property panel shows the selected layer's own settings",
  panel.includes("keras.Conv2D") && panel.includes("filters") && panel.includes("kernel size"));
await page.screenshot({ path: `${SHOTS}/ui-02-properties.png` });

await page.locator('aside:last-of-type input[type="number"]').first().fill("96");
await page.waitForTimeout(1600);
const reshaped = await chips(page);
check(
  "changing filters updates this layer and everything downstream",
  reshaped.some((line) => line.includes("26 × 26 × 96")) &&
    reshaped.some((line) => line.includes("16224")),
  reshaped.join(" | "),
);
await page.screenshot({ path: `${SHOTS}/ui-03-reshaped.png` });

// Place a loose layer with nothing selected, then wire it by dragging handles, so
// manual connection is covered as well as chaining.
await page.locator(".react-flow__pane").click({ position: { x: 40, y: 40 } });
await addLayer(page, "conv 2d", "keras.Conv2D");
await page.fill('input[placeholder="Search layers…"]', "");
const withExtra = await page
  .locator(".react-flow__node")
  .evaluateAll((els) => els.map((el) => el.getAttribute("data-id")));
check("a layer added with nothing selected stays loose", withExtra.length === 6 &&
  (await page.locator(".react-flow__edge").count()) === 4);

const loose = withExtra[withExtra.length - 1];
const denseId = ids[4];
await connect(page, denseId, loose);
await page.waitForTimeout(1800);
const badFooter = (await page.textContent("footer")).replace(/\s+/g, " ").trim();
check("connecting a convolution to a flat vector is refused by the engine",
  (await page.locator(".react-flow__node .ring-danger").count()) >= 1);
check("and the failure is reported in the status bar", /error/.test(badFooter),
  badFooter.slice(0, 120));
await page.screenshot({ path: `${SHOTS}/ui-04-error.png` });

const history = () =>
  page.evaluate(() => {
    const state = window.__graphStore.getState();
    return { past: state.past.length, future: state.future.length, edges: state.edges.length };
  });

const beforeUndo = await history();
await page.locator(".react-flow__pane").click({ position: { x: 40, y: 40 } });
await page.keyboard.press("Meta+z");
await page.waitForTimeout(1600);
const afterUndo = await history();
const codesAfterUndo = await page.evaluate(() =>
  window.__graphStore.getState().diagnostics.map((d) => d.code),
);
check(
  "undo removes the bad edge and the shape error with it",
  afterUndo.edges === beforeUndo.edges - 1 && !codesAfterUndo.includes("layer_call_failed"),
  `edges ${beforeUndo.edges} -> ${afterUndo.edges}, diagnostics now [${codesAfterUndo.join(", ")}]`,
);

await page.keyboard.press("Shift+Meta+z");
await page.waitForTimeout(1600);
const afterRedo = await history();
check("redo puts it back", afterRedo.edges === beforeUndo.edges,
  `edges ${afterUndo.edges} -> ${afterRedo.edges}`);

await page.keyboard.press("Meta+z");
await page.waitForTimeout(1200);
check("selecting a layer does not create an empty undo step",
  (await history()).edges === beforeUndo.edges - 1);

// --- a graph large enough that drawing every layer would not do ------------------
await page.evaluate(() => {
  const graph = window.__graphStore.getState();
  const catalog = window.__catalogStore.getState();
  const dense = catalog.specs.get("keras.Dense");
  const input = catalog.specs.get("keras.Input");
  const nodes = [];
  const edges = [];
  const COUNT = 600;
  for (let index = 0; index < COUNT; index += 1) {
    const spec = index === 0 ? input : dense;
    nodes.push({
      id: `n${index}`,
      type: "layer",
      position: { x: 240, y: index * 110 },
      selected: false,
      data: {
        typeId: spec.id,
        name: `layer ${index}`,
        params: index === 0 ? { shape: [8] } : { units: 8 },
        disabled: false,
        shape: null,
        diagnostics: [],
      },
    });
    if (index > 0) {
      edges.push({
        id: `e${index}`,
        source: `n${index - 1}`,
        target: `n${index}`,
        sourceHandle: "output",
        targetHandle: "input",
        data: { order: 0 },
      });
    }
  }
  graph.replaceGraph({ nodes, edges }, "Large graph");
});
await page.waitForTimeout(3000);

const drawn = await page.locator(".react-flow__node").count();
const held = await page.evaluate(() => window.__graphStore.getState().nodes.length);
check("a large graph is folded rather than drawn layer by layer",
  held === 600 && drawn > 0 && drawn < 120, `holds ${held}, draws ${drawn}`);

const visible = await page.evaluate(() =>
  [...document.querySelectorAll(".react-flow__node")]
    .filter((node) => getComputedStyle(node).visibility !== "hidden").length,
);
check("and what it draws is actually visible", visible > 0,
  `${visible} of ${drawn} not hidden`);

check("the full graph is still what gets sent to the engine",
  (await page.evaluate(() => window.__graphStore.getState().toIR().nodes.length)) === 600);

// Click a box that is actually on screen. Only what is visible is mounted, so the
// first in document order may be far above the viewport, and clicking it would then
// open layers the window is not looking at.
const opened = await page.evaluate(() => {
  const onScreen = [...document.querySelectorAll('button[title*="folded into one"]')]
    .find((box) => {
      const rect = box.getBoundingClientRect();
      return rect.top > 80 && rect.bottom < window.innerHeight - 40;
    });
  const title = onScreen?.getAttribute("title") ?? null;
  onScreen?.dispatchEvent(new MouseEvent("click", { bubbles: true }));
  return title;
});
await page.waitForTimeout(2000);
const expandedCount = await page.evaluate(
  () => window.__graphStore.getState().expandedGroups.size,
);
check("opening a folded box unfolds it", expandedCount === 1, opened ?? "none on screen");
const mix = await page.evaluate(() => {
  const all = [...document.querySelectorAll(".react-flow__node")];
  const folded = all.filter((node) => node.querySelector('button[title*="folded"]'));
  return { total: all.length, folded: folded.length, plain: all.length - folded.length };
});
check("and the layers inside it are now drawn individually",
  mix.plain > 0, `${mix.plain} plain layers beside ${mix.folded} folded boxes`);

const frame = await page.evaluate(async () => {
  const pane = document.querySelector(".react-flow__pane");
  const frames = [];
  let last = performance.now();
  for (let step = 0; step < 30; step += 1) {
    pane.dispatchEvent(new WheelEvent("wheel", { deltaY: 30, bubbles: true, cancelable: true }));
    await new Promise((resolve) =>
      requestAnimationFrame(() => {
        const now = performance.now();
        frames.push(now - last);
        last = now;
        resolve(1);
      }),
    );
  }
  frames.sort((a, b) => a - b);
  return Math.round(frames[15]);
});
check("and it still moves smoothly", frame < 50, `${frame}ms per frame`);

check("no failed requests or uncaught errors in the browser",
  consoleErrors.length === 0, consoleErrors.join(" | ") || "none");

console.log("\nshape chips observed:");
for (const note of notes) console.log(note);
console.log(`\nscreenshots written to ${SHOTS}`);
await browser.close();

if (failures.length) {
  console.error(`\n${failures.length} check(s) failed`);
  process.exit(1);
}
console.log("\nall checks passed");
