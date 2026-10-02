/**
 * > [!AML-DOC-FILE]
 * @file        src/editor/Palette.tsx
 * @description The layer palette: every catalog layer, grouped by category and
 *              searchable, draggable onto the canvas.
 * @module      frontend/editor/Palette
 * @exports     Palette
 * @created     2026-09-30
 * @context     Rendered entirely from the fetched catalog [amm: E.2]. Search covers
 *              the label, the description and the spec's tags, so "residual" finds
 *              Add and "mobilenet" finds the separable convolutions.
 */

import { useMemo, useState } from "react";
import { createPortal } from "react-dom";

import type { LayerSpec } from "../api/types";
import { categoryAccent, useCatalog } from "../store/catalog";
import { LayerGlyph } from "./LayerGlyph";
import { useGraph } from "../store/graph";
import { placementPosition } from "./viewport";

/** MIME type used to carry a layer id through an HTML5 drag. */
export const LAYER_DRAG_TYPE = "application/x-nnarch-layer";

/**
 * > [!AML-DOC-UNIT]
 * One palette entry. Draggable onto the canvas, and clickable for the same effect so
 * the palette is usable without a pointer drag.
 * @param spec the layer this row represents
 * @returns the row element
 */
function PaletteItem({ spec }: { spec: LayerSpec }) {
  const accent = categoryAccent(spec.category);
  const addLayer = useGraph((state) => state.addLayer);
  // Where to put the card, in screen coordinates. It is drawn into the body rather
  // than beside the row because the palette scrolls, and a scrolling box clips its own
  // absolutely-positioned children — the card was in the DOM and painted nowhere
  // [E-053].
  const [at, setAt] = useState<{ x: number; y: number } | null>(null);
  return (
    <div
      onMouseEnter={(event) => {
        const box = event.currentTarget.getBoundingClientRect();
        setAt({ x: box.right + 8, y: box.top });
      }}
      onMouseLeave={() => setAt(null)}
    >
    <button
      type="button"
      draggable
      data-layer-id={spec.id}
      onDragStart={(event) => {
        event.dataTransfer.setData(LAYER_DRAG_TYPE, spec.id);
        event.dataTransfer.effectAllowed = "copy";
      }}
      onClick={() => {
        const state = useGraph.getState();
        const selected = state.nodes.filter((node) => node.selected);
        const appendTo = selected.length === 1 ? (selected[0]?.id ?? null) : null;
        addLayer(spec, placementPosition(state.nodes.length), appendTo);
      }}
      className="group flex w-full items-center gap-2 rounded px-2 py-1 text-left hover:bg-surface-2"
      title={`${spec.description || spec.label}\n\nClick to add below the selected layer, or drag onto the canvas.`}
    >
      <span className="h-3 w-0.5 shrink-0 rounded" style={{ background: accent }} />
      <span className="truncate text-[12px] text-ink-0">{spec.label}</span>
      {spec.is_research && (
        <span className="ml-auto shrink-0 text-[9px] uppercase text-ink-2">paper</span>
      )}
    </button>

    {/*
      Shown beside the row rather than inside it, because the palette is narrow and a
      drawing squeezed into it would be the picture that says nothing [E-052]. The
      native tooltip still carries the text for anyone who never hovers long enough.
    */}
    {at && createPortal(
      <div
        role="tooltip"
        style={{
          left: at.x,
          // Nudged up when it would run off the bottom, so the last layer in a long
          // category is as readable as the first.
          top: Math.min(at.y, Math.max(8, window.innerHeight - 220)),
        }}
        className="pointer-events-none fixed z-50 w-56 rounded border border-line
                   bg-surface-1 p-2 shadow-lg"
      >
        <p className="text-[11px] font-semibold text-ink-0">{spec.label}</p>
        <div className="mt-1.5">
          <LayerGlyph typeId={spec.id} category={spec.category} />
        </div>
        {spec.description && (
          <p className="mt-1.5 text-[10px] leading-snug text-ink-1">{spec.description}</p>
        )}
        {spec.paper && (
          <p className="mt-1 text-[9px] leading-snug text-ink-2">{spec.paper}</p>
        )}
      </div>,
      document.body,
    )}
    </div>
  );
}

/**
 * > [!AML-DOC-UNIT]
 * The palette pane.
 * @returns the searchable, categorised list of placeable layers
 * @sideEffects none; dragging carries only the layer id, and the drop handler on the
 *              canvas is what mutates the graph
 */
export function Palette() {
  const catalog = useCatalog((state) => state.catalog);
  const [query, setQuery] = useState("");
  const [collapsed, setCollapsed] = useState<Set<string>>(new Set());

  const sections = useMemo(() => {
    if (!catalog) return [];
    const needle = query.trim().toLowerCase();
    if (!needle) return catalog.categories;
    return catalog.categories
      .map((category) => ({
        ...category,
        layers: category.layers.filter((layer) =>
          [layer.label, layer.description, ...layer.tags]
            .join(" ")
            .toLowerCase()
            .includes(needle),
        ),
      }))
      .filter((category) => category.layers.length > 0);
  }, [catalog, query]);

  const total = sections.reduce((sum, section) => sum + section.layers.length, 0);

  return (
    <aside className="flex h-full w-64 shrink-0 flex-col border-r border-line bg-surface-1">
      <div className="border-b border-line p-2">
        <input
          className="w-full rounded bg-surface-2 border border-line px-2 py-1 text-[12px]
                     text-ink-0 outline-none placeholder:text-ink-2
                     focus:border-accent focus:ring-1 focus:ring-accent/40"
          placeholder="Search layers…"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
        />
        <p className="mt-1 text-[10px] text-ink-2">
          {catalog ? `${total} of ${catalog.count} layers` : "loading catalog…"}
        </p>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto p-1">
        {sections.map((section) => {
          const isOpen = !collapsed.has(section.id) || query.trim().length > 0;
          return (
            <section key={section.id} className="mb-1">
              <button
                type="button"
                className="flex w-full items-center gap-1 rounded px-2 py-1 text-left
                           text-[10px] font-semibold uppercase tracking-wide
                           text-ink-2 hover:text-ink-1"
                onClick={() =>
                  setCollapsed((previous) => {
                    const next = new Set(previous);
                    if (next.has(section.id)) next.delete(section.id);
                    else next.add(section.id);
                    return next;
                  })
                }
              >
                <span className="w-2">{isOpen ? "−" : "+"}</span>
                <span className="truncate">{section.label}</span>
                <span className="ml-auto font-normal">{section.layers.length}</span>
              </button>
              {isOpen && (
                <div className="pl-1">
                  {section.layers.map((layer) => (
                    <PaletteItem key={layer.id} spec={layer} />
                  ))}
                </div>
              )}
            </section>
          );
        })}
        {catalog && total === 0 && (
          <p className="p-3 text-[11px] text-ink-2">No layer matches “{query}”.</p>
        )}
      </div>

      <p className="border-t border-line px-2 py-1.5 text-[10px] text-ink-2">
        Click to chain onto the selected layer, or drag to place it freely.
      </p>
    </aside>
  );
}
