/**
 * > [!AML-DOC-FILE]
 * @file        src/editor/PropertyPanel.tsx
 * @description Right-hand pane: edits the selected layer's name and every parameter
 *              its catalog spec declares.
 * @module      frontend/editor/PropertyPanel
 * @exports     PropertyPanel
 * @created     2026-09-30
 * @context     Contains no per-layer knowledge [amm: E.2]. It walks the spec's
 *              `params` and hands each one to `ParamField`, which picks the control
 *              from the declared type.
 */

import { useMemo } from "react";

import { useCatalog } from "../store/catalog";
import { useGraph } from "../store/graph";
import { LayerGlyph } from "./LayerGlyph";
import { openExternal } from "../shell/external";
import { ParamField } from "./widgets/ParamField";

/**
 * > [!AML-DOC-UNIT]
 * The property pane for the current selection.
 * @param embedded render as a pane body, without its own border and width, when the
 *                 side panel already provides them
 * @returns the editor for a single selected layer, a count when several are
 *          selected, or guidance when nothing is
 * @sideEffects writes through to the graph store on every edit, which triggers the
 *              debounced revalidation and updates downstream shapes
 */
export function PropertyPanel({ embedded = false }: { embedded?: boolean }) {
  const nodes = useGraph((state) => state.nodes);
  const updateParam = useGraph((state) => state.updateParam);
  const renameNode = useGraph((state) => state.renameNode);
  const toggleDisabled = useGraph((state) => state.toggleDisabled);
  const specs = useCatalog((state) => state.specs);
  const vocabularies = useCatalog((state) => state.catalog?.vocabularies);

  const selected = nodes.filter((node) => node.selected);
  const node = selected.length === 1 ? selected[0] : undefined;
  const spec = node ? specs.get(node.data.typeId) : undefined;

  const nodeOptions = useMemo(
    () =>
      nodes
        .filter((candidate) => candidate.id !== node?.id)
        .map((candidate) => ({
          id: candidate.id,
          label: `${candidate.data.name} (${candidate.id})`,
        })),
    [nodes, node?.id],
  );

  const groups = useMemo(() => {
    if (!spec) return { main: [], advanced: [] };
    return {
      main: spec.params.filter((param) => !param.advanced),
      advanced: spec.params.filter((param) => param.advanced),
    };
  }, [spec]);

  if (selected.length > 1) {
    return (
      <div className={embedded ? "p-3" : "w-80 shrink-0 border-l border-line bg-surface-1 p-3"}>
        <p className="text-[12px] text-ink-1">{selected.length} layers selected</p>
        <p className="mt-1 text-[11px] text-ink-2">
          Select a single layer to edit its settings.
        </p>
      </div>
    );
  }

  if (!node || !spec || !vocabularies) {
    return (
      <div className={embedded ? "p-3" : "w-80 shrink-0 border-l border-line bg-surface-1 p-3"}>
        <p className="text-[12px] font-semibold text-ink-1">Nothing selected</p>
        <p className="mt-1 text-[11px] leading-relaxed text-ink-2">
          Drag a layer from the palette, then click it to edit its settings. Start with an
          Input layer so the model has somewhere for data to enter.
        </p>
      </div>
    );
  }

  const errors = node.data.diagnostics.filter((d) => d.severity === "error");

  return (
    <div className={embedded ? "flex h-full flex-col" : "flex w-80 shrink-0 flex-col border-l border-line bg-surface-1"}>
      <div className="border-b border-line p-3">
        <input
          className="w-full rounded bg-surface-2 border border-line px-2 py-1 text-[13px]
                     font-semibold text-ink-0 outline-none
                     focus:border-accent focus:ring-1 focus:ring-accent/40"
          value={node.data.name}
          onChange={(event) => renameNode(node.id, event.target.value)}
        />
        <div className="mt-1.5 flex items-center gap-2 text-[10px] text-ink-2">
          <span className="font-mono">{spec.id}</span>
          <span>·</span>
          <span className="font-mono">{node.id}</span>
        </div>
        {/*
          The drawing sits above the sentence: what a layer does to a tensor's shape
          is a spatial fact, and the prose is there to qualify it [E-052].
        */}
        <div className="mt-2">
          <LayerGlyph typeId={spec.id} category={spec.category} caption={false} />
        </div>
        {spec.description && (
          <p className="mt-2 text-[11px] leading-snug text-ink-1">{spec.description}</p>
        )}
        <div className="mt-2 flex items-center gap-2">
          {spec.doc_url && (
            <button
              type="button"
              onClick={() => void openExternal(spec.doc_url as string)}
              className="text-[10px] text-accent hover:underline"
            >
              Keras docs ↗
            </button>
          )}
          {spec.paper && (
            <span className="text-[10px] text-ink-2" title={spec.paper}>
              {spec.paper}
            </span>
          )}
        </div>
        <button
          type="button"
          onClick={() => toggleDisabled(node.id)}
          className="mt-2 rounded border border-line bg-surface-2 px-2 py-0.5 text-[10px]
                     text-ink-1 hover:text-ink-0"
          title="Pass the input straight through, without removing the layer"
        >
          {node.data.disabled ? "Enable layer" : "Disable layer (ablate)"}
        </button>
      </div>

      {errors.length > 0 && (
        <div className="border-b border-line bg-danger/10 px-3 py-2">
          {errors.map((diagnostic) => (
            <p key={diagnostic.code} className="text-[11px] leading-snug text-danger">
              {diagnostic.message}
            </p>
          ))}
        </div>
      )}

      {node.data.shape && (
        <div className="border-b border-line px-3 py-2">
          <p className="text-[10px] uppercase tracking-wide text-ink-2">Output</p>
          <p className="mt-0.5 font-mono text-[12px] text-ink-0">
            {node.data.shape.shape.map((d) => (d === null ? "?" : d)).join(" × ")}
          </p>
          <p className="text-[10px] text-ink-2">
            {node.data.shape.dtype} · {node.data.shape.params.toLocaleString("en-US")} weights
          </p>
        </div>
      )}

      <div className="min-h-0 flex-1 space-y-3 overflow-y-auto p-3">
        {groups.main.map((param) => (
          <ParamField
            key={param.name}
            spec={param}
            value={node.data.params[param.name] ?? null}
            vocabularies={vocabularies}
            nodeOptions={nodeOptions}
            onChange={(value) => updateParam(node.id, param.name, value)}
          />
        ))}

        {groups.advanced.length > 0 && (
          <details className="rounded border border-line">
            <summary className="cursor-pointer px-2 py-1 text-[10px] uppercase tracking-wide text-ink-2">
              Advanced ({groups.advanced.length})
            </summary>
            <div className="space-y-3 border-t border-line p-2">
              {groups.advanced.map((param) => (
                <ParamField
                  key={param.name}
                  spec={param}
                  value={node.data.params[param.name] ?? null}
                  vocabularies={vocabularies}
                  nodeOptions={nodeOptions}
                  onChange={(value) => updateParam(node.id, param.name, value)}
                />
              ))}
            </div>
          </details>
        )}
      </div>
    </div>
  );
}
