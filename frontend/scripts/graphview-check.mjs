/**
 * > [!AML-DOC-FILE]
 * @file        scripts/graphview-check.mjs
 * @description Drives the analytical graph view in a real browser and asserts what a
 *              user would see: circular marks sized by weight, working zoom and pan,
 *              fit that lands every layer on screen, and folding for a huge model.
 * @module      frontend/scripts/graphview-check
 * @exports     (executable script; exits non-zero on failure)
 * @created     2026-10-01
 * @context     The companion to `ui-check.mjs`, for the view it does not cover. Pan
 *              and zoom are geometry, and geometry can only be checked by reading the
 *              transform a real browser computed [E-023]. Requires the engine on :8756
 *              and the dev server on :5173.
 */

import { mkdirSync } from "node:fs";
import { chromium } from "playwright-core";

const SHOTS = process.argv[2] ?? "./gv-shots";
mkdirSync(SHOTS, { recursive: true });
const fails = [];
const check = (label, ok, detail) => {
  console.log(`${ok ? "PASS" : "FAIL"}  ${label}${detail ? ` — ${detail}` : ""}`);
  if (!ok) fails.push(label);
};

const browser = await chromium.launch({ channel: "chrome" });
const page = await browser.newPage({ viewport: { width: 1600, height: 1000 } });
page.on("pageerror", (e) => fails.push(`page error: ${e.message}`));
await page.goto("http://localhost:5173/", { waitUntil: "networkidle" });
await page.waitForFunction(() => window.__graphStore, null, { timeout: 20000 });

// A small model with a skip connection, so edges and the critical path both matter.
await page.evaluate(() => {
  const g = window.__graphStore.getState();
  const mk = (id, typeId, params, x, y) => ({
    id, type: "layer", position: { x, y }, selected: false,
    data: { typeId, name: id, params, disabled: false, shape: null, diagnostics: [] },
  });
  g.replaceGraph({
    nodes: [
      mk("inp", "keras.Input", { shape: [32, 32, 3] }, 0, 0),
      mk("c1", "keras.Conv2D", { filters: 32, kernel_size: 3, padding: "same" }, 0, 120),
      mk("c2", "keras.Conv2D", { filters: 32, kernel_size: 3, padding: "same" }, 0, 240),
      mk("add", "keras.Add", {}, 0, 360),
      mk("pool", "keras.GlobalAveragePooling2D", {}, 0, 480),
      mk("out", "keras.Dense", { units: 10, activation: "softmax" }, 0, 600),
    ],
    edges: [
      ...[["e1", "inp", "c1"], ["e2", "c1", "c2"], ["e3", "c2", "add"],
          ["e4", "c1", "add"], ["e5", "add", "pool"], ["e6", "pool", "out"]].map(
        ([id, source, target], order) => ({
          id, source, target,
          sourceHandle: "output",
          // Add takes a list, and its port is named for that.
          targetHandle: target === "add" ? "inputs" : "input",
          data: { order },
        }),
      ),
    ],
  }, "Residual block");
});

// The circle sizes come from parameter counts the engine computes, so wait for them
// rather than for a duration: a timeout that is merely long enough today is an
// instrument that will lie later [E-011].
await page.waitForFunction(
  () => window.__graphStore.getState().nodes.every((n) => n.data.shape),
  null,
  { timeout: 30000 },
);
const weighted = await page.evaluate(() =>
  Object.fromEntries(window.__graphStore.getState().nodes.map(
    (n) => [n.id, n.data.shape?.params ?? 0])));
check("the engine reported parameter counts to size the circles with",
  weighted.c1 > 0 && weighted.c2 > 0, JSON.stringify(weighted));

await page.evaluate(() => window.__graphStore.getState().setView("graph"));
await page.waitForTimeout(600);

const svg = page.locator('svg[aria-label*="view of"]');
const zoom = (title) => page.getByRole("group", { name: "Graph zoom" }).getByTitle(title);
check("the graph view renders its own svg surface", await svg.count() === 1);

const circles = await page.locator('svg[aria-label*="view of"] circle').count();
check("nodes are drawn as circles", circles === 6, `${circles} circles`);
const rects = await page.locator('svg[aria-label*="view of"] rect').count();
check("no rectangular node marks remain", rects === 0, `${rects} rects`);

const transform = () =>
  page.evaluate(() =>
    document.querySelector('svg[aria-label*="view of"] > g').getAttribute("transform"));
const scaleOf = (t) => Number(/scale\(([\d.]+)\)/.exec(t)?.[1]);
const xOf = (t) => Number(/translate\(([-\d.]+) /.exec(t)?.[1]);

const t0 = await transform();
check("the surface carries a pan/zoom transform", /translate\(.*scale\(/.test(t0), t0);

// wheel zoom
const box = await svg.boundingBox();
await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
await page.mouse.wheel(0, -300);
await page.waitForTimeout(250);
const t1 = await transform();
check("scrolling zooms in", scaleOf(t1) > scaleOf(t0), `${scaleOf(t0)} -> ${scaleOf(t1)}`);
await page.mouse.wheel(0, 600);
await page.waitForTimeout(250);
const t2 = await transform();
check("scrolling the other way zooms out", scaleOf(t2) < scaleOf(t1), `${scaleOf(t1)} -> ${scaleOf(t2)}`);

// drag to pan
await page.mouse.move(box.x + 120, box.y + box.height - 80);
await page.mouse.down();
await page.mouse.move(box.x + 420, box.y + box.height - 160, { steps: 14 });
await page.mouse.up();
await page.waitForTimeout(250);
const t3 = await transform();
check("dragging pans the diagram", Math.abs(xOf(t3) - xOf(t2)) > 200,
  `x ${xOf(t2)} -> ${xOf(t3)}`);
check("panning did not change the zoom", scaleOf(t3) === scaleOf(t2));

// the +/- / fit buttons
await zoom("Zoom in").click();
await page.waitForTimeout(250);
const t4 = await transform();
check("the + button zooms in", scaleOf(t4) > scaleOf(t3), `${scaleOf(t3)} -> ${scaleOf(t4)}`);
await zoom("Zoom out").click();
await page.waitForTimeout(250);
check("the − button zooms out", scaleOf(await transform()) < scaleOf(t4));
await zoom("Fit the whole graph").click();
await page.waitForTimeout(300);
const tf = await transform();

// After a fit, every circle must actually be inside the visible surface.
const inside = await page.evaluate(() => {
  const s = document.querySelector('svg[aria-label*="view of"]');
  const b = s.getBoundingClientRect();
  return [...s.querySelectorAll("circle")].every((c) => {
    const r = c.getBoundingClientRect();
    return r.top >= b.top - 1 && r.bottom <= b.bottom + 1
      && r.left >= b.left - 1 && r.right <= b.right + 1;
  });
});
check("fit brings every layer inside the viewport", inside, `scale ${scaleOf(tf)}`);
check("fit does not shrink the graph to specks", scaleOf(tf) > 0.2, `scale ${scaleOf(tf)}`);

const readout = await page.locator("text=/^\\d+%$/").first().textContent();
check("the zoom level is shown to the user", /%$/.test(readout ?? ""), readout);

// circle area should track the parameter count: conv layers carry weights, pooling does not
const radii = await page.evaluate(() => {
  const out = {};
  for (const g of document.querySelectorAll('svg[aria-label*="view of"] g > g')) {
    const name = g.querySelector("title")?.textContent?.split(" ·")[0];
    const r = g.querySelector("circle")?.getAttribute("r");
    if (name) out[name] = Number(r);
  }
  return out;
});
check("a layer with more parameters gets a bigger circle",
  radii.c2 > radii.c1 && radii.c1 > radii.pool,
  `c2=${radii.c2?.toFixed(1)} c1=${radii.c1?.toFixed(1)} pool=${radii.pool?.toFixed(1)}`);

// clicking still selects on the canvas
await page.locator('svg[aria-label*="view of"] circle').nth(2).click();
await page.waitForTimeout(200);
const sel = await page.evaluate(() => [...window.__graphStore.getState().selection]);
check("clicking a circle selects that layer in the editor", sel.length === 1, sel.join(","));

await page.screenshot({ path: `${SHOTS}/gv-01-dataflow.png` });

// the skip edge must be dashed, and must not hide behind the chain it bypasses
const dashed = await page.locator('svg[aria-label*="view of"] path[stroke-dasharray]').count();
check("the skip connection is drawn dashed", dashed >= 1, `${dashed} dashed`);
const bow = await page.evaluate(() => {
  const skip = document.querySelector('svg[aria-label*="view of"] path[stroke-dasharray]');
  const chain = [...document.querySelectorAll('svg[aria-label*="view of"] path')]
    .find((p) => !p.hasAttribute("stroke-dasharray"));
  const box = (el) => { const b = el.getBBox(); return { x: b.x, width: b.width }; };
  return { skip: box(skip), chain: box(chain) };
});
check("the skip is bowed clear of the layers it bypasses", bow.skip.width > 20,
  `skip spans ${bow.skip.width.toFixed(0)}px across, chain ${bow.chain.width.toFixed(0)}px`);
check("the skip bows away from the labels, not through them",
  bow.skip.x < bow.chain.x, `skip x ${bow.skip.x.toFixed(0)}, chain x ${bow.chain.x.toFixed(0)}`);

// volume mode and matrix mode still render
await page.getByRole("button", { name: "Tensor volume" }).click();
await page.waitForTimeout(400);
check("tensor-volume mode still draws circles",
  (await page.locator('svg[aria-label*="view of"] circle').count()) === 6);
await page.screenshot({ path: `${SHOTS}/gv-02-volume.png` });

await page.getByRole("button", { name: "Connections" }).click();
await page.waitForTimeout(400);
check("the connection matrix still renders", (await page.locator("table td").count()) === 36);
await page.screenshot({ path: `${SHOTS}/gv-03-matrix.png` });

await page.getByRole("button", { name: "Dataflow" }).click();
await page.waitForTimeout(300);

// --- and now a graph far too large to draw one mark per layer -------------------
await page.evaluate(() => {
  const g = window.__graphStore.getState();
  const node = (i) => ({
    id: `n${i}`, type: "layer", position: { x: 240, y: i * 110 }, selected: false,
    data: {
      typeId: i === 0 ? "keras.Input" : "keras.Dense",
      name: `n${i}`,
      params: i === 0 ? { shape: [64] } : { units: 64 },
      disabled: false, shape: null, diagnostics: [],
    },
  });
  const nodes = [node(0)];
  const edges = [];
  for (let i = 1; i < 900; i += 1) {
    nodes.push(node(i));
    edges.push({
      id: `e${i}`, source: `n${i - 1}`, target: `n${i}`,
      sourceHandle: "output", targetHandle: "input", data: { order: 0 },
    });
  }
  g.replaceGraph({ nodes, edges }, "Large graph");
});
await page.waitForTimeout(1200);
await zoom("Fit the whole graph").click();
await page.waitForTimeout(400);
const shown = await page.locator('svg[aria-label*="view of"] circle').count();
check("a huge graph is folded in the graph view too", shown > 0 && shown < 200,
  `${shown} marks for 900 layers`);

// A deep chain must not fit only by becoming a thread of specks [E-022, E-024].
const deepScale = scaleOf(await transform());
check("a deep chain stays legible rather than shrinking to specks", deepScale > 0.4,
  `scale ${deepScale.toFixed(2)}`);
const columns = await page.evaluate(() => {
  const xs = new Set();
  for (const c of document.querySelectorAll('svg[aria-label*="view of"] circle')) {
    xs.add(Math.round(Number(c.closest("g").getAttribute("transform").match(/translate\(([-\d.]+)/)[1]) / 50));
  }
  return xs.size;
});
check("the chain is wrapped across several columns", columns > 2, `${columns} columns`);
const allIn = await page.evaluate(() => {
  const s = document.querySelector('svg[aria-label*="view of"]');
  const b = s.getBoundingClientRect();
  return [...s.querySelectorAll("circle")].every((c) => {
    const r = c.getBoundingClientRect();
    return r.top >= b.top - 1 && r.bottom <= b.bottom + 1;
  });
});
check("every wrapped column is on screen after a fit", allIn);
check("the engine still receives every layer",
  (await page.evaluate(() => window.__graphStore.getState().toIR().nodes.length)) === 900);

const t = Date.now();
for (let i = 0; i < 6; i += 1) {
  await page.mouse.move(box.x + 300 + i * 40, box.y + 400);
  await page.mouse.wheel(0, i % 2 ? 200 : -200);
}
await page.waitForTimeout(100);
check("zooming a folded large graph stays responsive", Date.now() - t < 3000,
  `${Date.now() - t}ms for 6 zoom steps`);
await page.screenshot({ path: `${SHOTS}/gv-04-large.png` });

await browser.close();
console.log(`\nshots in ${SHOTS}`);
if (fails.length) { console.log(`\n${fails.length} FAILED`); process.exit(1); }
console.log("\nall graph-view checks passed");
