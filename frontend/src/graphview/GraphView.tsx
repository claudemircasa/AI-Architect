/**
 * > [!AML-DOC-FILE]
 * @file        src/graphview/GraphView.tsx
 * @description Analytical readings of the same architecture: a layered dataflow
 *              diagram, a tensor-volume view, and a connectivity matrix.
 * @module      frontend/graphview/GraphView
 * @exports     GraphView
 * @created     2026-10-01
 * @context     Read-only, and over the same IR the editor holds [task 10]. Clicking
 *              anything selects it on the canvas, so the two views stay one thing
 *              rather than two copies that can disagree.
 *
 *              Pan and zoom are implemented here rather than borrowed from React
 *              Flow: this is a drawing, not an editor, and a plain transform over one
 *              SVG group stays smooth at thousands of marks where mounting a
 *              component per node does not.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { findChains, GROUPING_THRESHOLD, projectGraph } from "../editor/grouping";
import { useActivations } from "../store/activations";
import { categoryAccent, useCatalog } from "../store/catalog";
import { useGraph } from "../store/graph";
import { analyseGraph, type AnalysedNode } from "./analysis";

type Mode = "dataflow" | "volume" | "matrix";
type Overlay = "shape" | "parameters" | "volume";

const ROW_GAP = 76;
const COLUMN_GAP = 190;
const MIN_RADIUS = 7;
const MAX_RADIUS = 26;
const MIN_ZOOM = 0.08;
const MAX_ZOOM = 4;
// Below this a depth-ordered column is taller than any screen, so it is wrapped.
const WRAP_MIN_DEPTH = 24;
const WRAP_MAX_WIDTH = 2;

/**
 * > [!AML-DOC-UNIT]
 * Format a count compactly, so a million parameters does not overrun a label.
 * @param value the number
 * @returns a short string such as "23.6M"
 */
function compact(value: number): string {
  if (value >= 1e9) return `${(value / 1e9).toFixed(1)}B`;
  if (value >= 1e6) return `${(value / 1e6).toFixed(1)}M`;
  if (value >= 1e3) return `${(value / 1e3).toFixed(1)}k`;
  return String(value);
}

/**
 * > [!AML-DOC-UNIT]
 * The analytical views.
 * @returns the mode switcher and whichever diagram is selected
 * @sideEffects selects nodes in the editor when they are clicked, and downloads an
 *              SVG when the export button is used
 */
export function GraphView() {
  const allNodes = useGraph((state) => state.nodes);
  const allEdges = useGraph((state) => state.edges);
  const grouping = useGraph((state) => state.grouping);
  const expandedGroups = useGraph((state) => state.expandedGroups);
  const setSelection = useGraph((state) => state.setSelection);
  const byNode = useActivations((state) => state.byNode);
  const specs = useCatalog((state) => state.specs);

  const [mode, setMode] = useState<Mode>("dataflow");
  const [overlay, setOverlay] = useState<Overlay>("parameters");
  const [hover, setHover] = useState<string | null>(null);
  const [view, setView] = useState({ x: 0, y: 0, k: 1 });
  const dragging = useRef<{ x: number; y: number; vx: number; vy: number } | null>(null);
  const surface = useRef<SVGSVGElement>(null);
  const [size, setSize] = useState({ width: 1000, height: 700 });

  // The layout needs the viewport's proportions, not just its existence: how many
  // columns a deep chain wraps into is decided by the shape of the space it has.
  useEffect(() => {
    const element = surface.current;
    if (!element) return undefined;
    const observer = new ResizeObserver((entries) => {
      const { width, height } = entries[0]?.contentRect ?? { width: 0, height: 0 };
      if (width > 0 && height > 0) setSize({ width, height });
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, [mode]);

  // The same folding the canvas uses, for the same reason: a converted model holds
  // thousands of operations, and a mark for each is unreadable before it is slow.
  const { nodes, edges } = useMemo(() => {
    if (!grouping || allNodes.length < GROUPING_THRESHOLD) {
      return { nodes: allNodes, edges: allEdges };
    }
    const groups = findChains(allNodes, allEdges, (node) =>
      specs.get(node.data.typeId)?.label ?? node.data.typeId.split(".").pop() ?? "layer",
    );
    return projectGraph(allNodes, allEdges, groups, expandedGroups);
  }, [allNodes, allEdges, grouping, expandedGroups, specs]);

  const analysis = useMemo(() => analyseGraph(nodes, edges, byNode), [nodes, edges, byNode]);

  /**
   * > [!AML-DOC-UNIT]
   * Place every layer, wrapping a deep chain into columns.
   * @returns the position of each node, the diagram's extent, and how many columns
   * @sideEffects none
   * @context Depth ordering alone puts a ninety-layer stack in one column, which only
   *          fits on screen once it has been shrunk to a thread of specks — the same
   *          failure as E-022, in a different view. A chain is therefore folded into
   *          columns chosen so the drawing has roughly the viewport's proportions,
   *          which is what newspaper columns do and for the same reason. Graphs that
   *          branch keep the single ordering, because wrapping would cut their
   *          branches apart.
   */
  const layout = useMemo(() => {
    const rows = analysis.byDepth;
    const widest = Math.max(1, ...rows.map((row) => row.length));
    const band = widest * COLUMN_GAP;
    const aspect = Math.max(0.2, size.height / Math.max(size.width, 1));
    const columns =
      widest <= WRAP_MAX_WIDTH && rows.length >= WRAP_MIN_DEPTH
        ? Math.max(1, Math.round(Math.sqrt((rows.length * ROW_GAP) / (band * aspect))))
        : 1;
    const perColumn = Math.ceil(rows.length / columns);

    const positions = new Map<string, { x: number; y: number; column: number }>();
    rows.forEach((row, depth) => {
      const column = Math.floor(depth / perColumn);
      const slot = depth % perColumn;
      row.forEach((node, index) => {
        const spread = (index - (row.length - 1) / 2) * COLUMN_GAP;
        positions.set(node.id, {
          x: column * band + band / 2 + spread,
          y: 48 + slot * ROW_GAP,
          column,
        });
      });
    });

    return {
      positions,
      columns,
      // The right margin is the labels, which sit outside the last circle.
      width: columns * band + 150,
      height: Math.min(perColumn, rows.length) * ROW_GAP + 96,
    };
  }, [analysis, size]);

  const positions = layout.positions;
  const extent = layout;

  const maxParameters = Math.max(1, ...analysis.nodes.map((node) => node.parameters));

  /**
   * > [!AML-DOC-UNIT]
   * How big to draw a node, given what the overlay is measuring.
   * @param node the analysed node
   * @returns a radius in diagram units
   * @sideEffects none
   * @context Area carries the quantity, not radius: doubling a radius quadruples the
   *          ink, which reads as four times the value rather than twice. Taking the
   *          square root makes the circle's area proportional to what it stands for.
   */
  function radiusOf(node: AnalysedNode): number {
    if (overlay === "shape") return 11;
    const value = overlay === "parameters" ? node.parameters : node.volume;
    const scale = overlay === "parameters" ? maxParameters : analysis.maxVolume;
    const fraction = Math.sqrt(Math.max(value, 0) / Math.max(scale, 1));
    return MIN_RADIUS + fraction * (MAX_RADIUS - MIN_RADIUS);
  }

  /**
   * > [!AML-DOC-UNIT]
   * What the overlay prints beside a node.
   * @param node the analysed node
   * @returns the label text
   */
  function overlayLabel(node: AnalysedNode): string {
    if (overlay === "parameters") return node.parameters ? compact(node.parameters) : "—";
    if (overlay === "volume") return compact(node.volume);
    return node.shape.length > 1
      ? node.shape.slice(1).map((dim) => dim ?? "?").join("×")
      : "—";
  }

  const zoomBy = useCallback((factor: number, about?: { x: number; y: number }) => {
    setView((current) => {
      const next = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, current.k * factor));
      const box = surface.current?.getBoundingClientRect();
      const px = about?.x ?? (box ? box.width / 2 : 0);
      const py = about?.y ?? (box ? box.height / 2 : 0);
      // Keep whatever is under the pointer under the pointer.
      return {
        k: next,
        x: px - ((px - current.x) * next) / current.k,
        y: py - ((py - current.y) * next) / current.k,
      };
    });
  }, []);

  const fit = useCallback(() => {
    const box = surface.current?.getBoundingClientRect();
    if (!box) return;
    const k = Math.min(
      MAX_ZOOM,
      Math.max(MIN_ZOOM, Math.min(box.width / extent.width, box.height / extent.height) * 0.9),
    );
    setView({
      k,
      x: (box.width - extent.width * k) / 2,
      y: Math.max(12, (box.height - extent.height * k) / 2),
    });
  }, [extent]);

  // The View menu drives whichever surface is on screen, so this one listens while
  // it is the one being shown.
  useEffect(() => {
    function onAsked(event: Event) {
      const which = (event as CustomEvent<string>).detail;
      if (which === "view.fit") fit();
      else if (which === "view.zoomIn") zoomBy(1.25);
      else if (which === "view.zoomOut") zoomBy(1 / 1.25);
    }
    window.addEventListener("nnarch:viewport", onAsked);
    return () => window.removeEventListener("nnarch:viewport", onAsked);
  }, [fit, zoomBy]);

  // Fit whenever the drawing itself changes shape, so switching to this view or
  // loading a model never opens on an empty patch of canvas the user has to hunt
  // around. Panning does not change the extent, so it does not fight the user.
  const fitted = useRef("");
  useEffect(() => {
    const signature = `${mode}:${extent.width}x${extent.height}:${analysis.nodes.length}`;
    if (fitted.current === signature) return;
    fitted.current = signature;
    fit();
  }, [mode, extent, analysis.nodes.length, fit]);

  /**
   * > [!AML-DOC-UNIT]
   * Download the current diagram as an SVG file.
   * @sideEffects serialises the live SVG and triggers a download
   */
  function exportSvg(): void {
    const element = surface.current;
    if (!element) return;
    const source = new XMLSerializer().serializeToString(element);
    const url = URL.createObjectURL(
      new Blob([`<?xml version="1.0" encoding="UTF-8"?>\n${source}`], {
        type: "image/svg+xml",
      }),
    );
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `${useGraph.getState().name || "graph"}-${mode}.svg`;
    anchor.click();
    URL.revokeObjectURL(url);
  }

  if (allNodes.length === 0) {
    return (
      <div className="flex min-w-0 flex-1 items-center justify-center">
        <p className="max-w-sm text-center text-[11px] leading-relaxed text-ink-2">
          Nothing to analyse yet. Build an architecture on the canvas and these views
          will read its structure.
        </p>
      </div>
    );
  }

  return (
    <div className="flex min-w-0 flex-1 flex-col bg-surface-0">
      <div className="flex shrink-0 items-center gap-3 border-b border-line px-3 py-1.5">
        <span className="flex items-center gap-1">
          {(
            [
              ["dataflow", "Dataflow"],
              ["volume", "Tensor volume"],
              ["matrix", "Connections"],
            ] as const
          ).map(([value, label]) => (
            <button
              key={value}
              type="button"
              onClick={() => setMode(value)}
              className={`rounded px-2 py-0.5 text-[11px] ${
                mode === value ? "bg-surface-2 text-ink-0" : "text-ink-2 hover:text-ink-1"
              }`}
            >
              {label}
            </button>
          ))}
        </span>

        {mode !== "matrix" && (
          <label className="flex items-center gap-1 text-[10px] text-ink-2">
            Size by
            <select
              className="rounded border border-line bg-surface-2 px-1 py-0.5 text-[10px] text-ink-1"
              value={overlay}
              onChange={(event) => setOverlay(event.target.value as Overlay)}
            >
              <option value="parameters">parameters</option>
              <option value="volume">tensor size</option>
              <option value="shape">nothing (equal circles)</option>
            </select>
          </label>
        )}

        <span className="ml-auto flex items-center gap-3 text-[10px] text-ink-2">
          <span>
            {analysis.nodes.length} shown
            {nodes.length < allNodes.length ? ` of ${allNodes.length}` : ""}
          </span>
          <span>{compact(analysis.totalParameters)} params</span>
          <span>depth {analysis.byDepth.length}</span>
          {analysis.skipCount > 0 && (
            <span className="text-cat-merging">{analysis.skipCount} skip</span>
          )}
          <button
            type="button"
            onClick={exportSvg}
            className="rounded border border-line bg-surface-2 px-2 py-0.5 text-ink-1 hover:text-ink-0"
          >
            Export SVG
          </button>
        </span>
      </div>

      <div className="relative min-h-0 flex-1">
        {mode === "matrix" ? (
          <div className="h-full overflow-auto p-4">
            <ConnectionMatrix
              analysis={analysis}
              hover={hover}
              onHover={setHover}
              onSelect={(id) => setSelection([id])}
            />
          </div>
        ) : (
          <>
            <svg
              ref={surface}
              xmlns="http://www.w3.org/2000/svg"
              className="h-full w-full touch-none"
              style={{ cursor: dragging.current ? "grabbing" : "grab" }}
              role="img"
              aria-label={`${mode} view of ${analysis.nodes.length} layers`}
              onWheel={(event) => {
                event.preventDefault();
                const box = surface.current?.getBoundingClientRect();
                zoomBy(event.deltaY < 0 ? 1.12 : 1 / 1.12, {
                  x: event.clientX - (box?.left ?? 0),
                  y: event.clientY - (box?.top ?? 0),
                });
              }}
              onPointerDown={(event) => {
                (event.target as Element).setPointerCapture?.(event.pointerId);
                dragging.current = {
                  x: event.clientX,
                  y: event.clientY,
                  vx: view.x,
                  vy: view.y,
                };
              }}
              onPointerMove={(event) => {
                const drag = dragging.current;
                if (!drag) return;
                setView((current) => ({
                  ...current,
                  x: drag.vx + (event.clientX - drag.x),
                  y: drag.vy + (event.clientY - drag.y),
                }));
              }}
              onPointerUp={() => {
                dragging.current = null;
              }}
              onPointerLeave={() => {
                dragging.current = null;
              }}
            >
              <g transform={`translate(${view.x} ${view.y}) scale(${view.k})`}>
                {analysis.edges.map((edge) => {
                  const from = positions.get(edge.source);
                  const to = positions.get(edge.target);
                  if (!from || !to) return null;
                  const lit = hover === edge.source || hover === edge.target;
                  // A skip drawn straight down lands exactly on top of the chain it
                  // bypasses, which hides the one thing it exists to show, so it is
                  // bowed aside instead — to the left, because the labels are to the
                  // right and a bow through them trades one collision for another. An
                  // edge that crosses a column seam
                  // leaves sideways for the same reason: so it reads as going across.
                  const seam = from.column !== to.column;
                  const bow = edge.isSkip ? 34 + Math.abs(to.y - from.y) * 0.18 : 0;
                  const mid = (from.y + to.y) / 2;
                  const path = seam
                    ? `M ${from.x} ${from.y} C ${from.x + COLUMN_GAP * 0.6} ${from.y}, ` +
                      `${to.x - COLUMN_GAP * 0.6} ${to.y}, ${to.x} ${to.y}`
                    : bow
                      ? `M ${from.x} ${from.y} C ${from.x - bow} ${mid}, ` +
                        `${to.x - bow} ${mid}, ${to.x} ${to.y}`
                      : `M ${from.x} ${from.y} C ${from.x} ${mid}, ${to.x} ${mid}, ${to.x} ${to.y}`;
                  const thickness =
                    mode === "volume"
                      ? Math.max(1, (edge.volume / analysis.maxVolume) * 16)
                      : edge.onCriticalPath
                        ? 1.8
                        : 1.1;
                  return (
                    <path
                      key={edge.id}
                      d={path}
                      fill="none"
                      stroke={edge.isSkip ? "var(--color-cat-merging)" : "var(--color-line)"}
                      strokeWidth={thickness}
                      strokeOpacity={lit ? 1 : 0.7}
                      strokeDasharray={edge.isSkip ? "5 4" : undefined}
                    >
                      <title>
                        {edge.source} → {edge.target}
                        {edge.isSkip ? " (skip connection)" : ""} · {compact(edge.volume)} values
                      </title>
                    </path>
                  );
                })}

                {analysis.nodes.map((node) => {
                  const at = positions.get(node.id);
                  if (!at) return null;
                  const spec = specs.get(node.typeId);
                  const accent = spec
                    ? categoryAccent(spec.category)
                    : "var(--color-cat-core)";
                  const lit = hover === node.id;
                  const radius = radiusOf(node);
                  return (
                    <g
                      key={node.id}
                      transform={`translate(${at.x} ${at.y})`}
                      onMouseEnter={() => setHover(node.id)}
                      onMouseLeave={() => setHover(null)}
                      onClick={() => setSelection([node.id])}
                      style={{ cursor: "pointer" }}
                    >
                      <circle
                        r={radius}
                        fill={accent}
                        fillOpacity={node.disabled ? 0.2 : lit ? 0.95 : 0.7}
                        stroke={
                          lit || node.onCriticalPath ? accent : "var(--color-surface-0)"
                        }
                        strokeWidth={lit ? 2.5 : node.onCriticalPath ? 1.8 : 1.5}
                      />
                      <text
                        x={radius + 7}
                        y="-1"
                        fill="var(--color-ink-0)"
                        fontSize="11"
                        fontFamily="ui-sans-serif, system-ui"
                      >
                        {node.name.length > 22 ? `${node.name.slice(0, 21)}…` : node.name}
                      </text>
                      <text
                        x={radius + 7}
                        y="11"
                        fill="var(--color-ink-2)"
                        fontSize="9.5"
                        fontFamily="ui-monospace, monospace"
                      >
                        {overlayLabel(node)}
                      </text>
                      <title>
                        {node.name} · {node.typeId} · {compact(node.parameters)} parameters
                        {node.onCriticalPath ? " · on the longest path" : ""}
                      </title>
                    </g>
                  );
                })}
              </g>
            </svg>

            <div
              role="group"
              aria-label="Graph zoom"
              className="absolute bottom-3 left-3 flex flex-col overflow-hidden
                         rounded border border-line bg-surface-1"
            >
              {(
                [
                  ["+", "Zoom in", () => zoomBy(1.25)],
                  ["−", "Zoom out", () => zoomBy(1 / 1.25)],
                  ["⤢", "Fit the whole graph", fit],
                ] as const
              ).map(([glyph, title, action]) => (
                <button
                  key={title}
                  type="button"
                  title={title}
                  onClick={action}
                  className="border-b border-line px-2 py-1 text-[12px] text-ink-1
                             last:border-b-0 hover:bg-surface-2 hover:text-ink-0"
                >
                  {glyph}
                </button>
              ))}
            </div>

            <span className="absolute bottom-3 right-3 rounded border border-line
                             bg-surface-1/90 px-2 py-0.5 text-[10px] text-ink-2">
              {Math.round(view.k * 100)}%
            </span>
          </>
        )}
      </div>

      <p className="shrink-0 border-t border-line px-3 py-1.5 text-[10px] text-ink-2">
        {mode === "volume"
          ? "Edge thickness is how many values cross it, so a bottleneck shows as a narrowing."
          : mode === "matrix"
            ? "A filled cell means the row's layer feeds the column's."
            : "Circle area is the quantity, not its radius. Dashed edges skip over a layer, which is what a residual connection does. Drag to move, scroll to zoom, click a layer to select it on the canvas."}
      </p>
    </div>
  );
}

/**
 * > [!AML-DOC-UNIT]
 * Connectivity as a matrix, which reads the wiring of a dense graph that a
 * node-and-edge diagram turns into a thicket.
 * @param analysis the graph analysis
 * @param hover    the layer under the pointer
 * @param onHover  called as the pointer moves
 * @param onSelect called when a layer is clicked
 * @returns the matrix element
 */
function ConnectionMatrix({
  analysis,
  hover,
  onHover,
  onSelect,
}: {
  analysis: ReturnType<typeof analyseGraph>;
  hover: string | null;
  onHover: (id: string | null) => void;
  onSelect: (id: string) => void;
}) {
  const ordered = [...analysis.nodes].sort((a, b) => a.depth - b.depth);
  const links = new Set(analysis.edges.map((edge) => `${edge.source}→${edge.target}`));
  const skips = new Set(
    analysis.edges.filter((e) => e.isSkip).map((e) => `${e.source}→${e.target}`),
  );
  const cell = ordered.length > 40 ? 10 : 16;

  return (
    <div className="inline-block">
      <table className="border-collapse">
        <tbody>
          {ordered.map((row) => (
            <tr key={row.id}>
              <th
                scope="row"
                className="whitespace-nowrap pr-2 text-right text-[10px] font-normal text-ink-2"
                onMouseEnter={() => onHover(row.id)}
                onMouseLeave={() => onHover(null)}
              >
                <button
                  type="button"
                  onClick={() => onSelect(row.id)}
                  className={hover === row.id ? "text-ink-0" : "hover:text-ink-1"}
                >
                  {row.name.length > 18 ? `${row.name.slice(0, 17)}…` : row.name}
                </button>
              </th>
              {ordered.map((column) => {
                const key = `${row.id}→${column.id}`;
                const linked = links.has(key);
                const skip = skips.has(key);
                return (
                  <td
                    key={column.id}
                    style={{ width: cell, height: cell }}
                    className="border border-line/40 p-0"
                    title={linked ? `${row.name} feeds ${column.name}` : undefined}
                    onMouseEnter={() => onHover(column.id)}
                    onMouseLeave={() => onHover(null)}
                  >
                    {linked && (
                      <div
                        className="h-full w-full"
                        style={{
                          background: skip
                            ? "var(--color-cat-merging)"
                            : "var(--color-accent)",
                          opacity: hover === row.id || hover === column.id ? 1 : 0.7,
                        }}
                      />
                    )}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
