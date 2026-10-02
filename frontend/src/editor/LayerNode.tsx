/**
 * > [!AML-DOC-FILE]
 * @file        src/editor/LayerNode.tsx
 * @description Canvas renderer for a placed layer: title, category accent, the
 *              output shape the engine inferred, and an error ring when the engine
 *              complained about it.
 * @module      frontend/editor/LayerNode
 * @exports     LayerNodeView
 * @created     2026-09-30
 * @context     Ports are drawn from `LayerSpec.inputs`, so an attention layer grows
 *              its query/value/key handles without this file knowing what attention
 *              is [amm: E.2].
 *
 *              When a sample has been run, each node also carries a thumbnail of
 *              what it produced, so the data flow is legible on the architecture
 *              itself rather than only in a side panel [task 11].
 */

import { Handle, Position, type NodeProps } from "@xyflow/react";
import { memo } from "react";

import type { ShapeInfo } from "../api/types";
import { useActivations } from "../store/activations";
import { categoryAccent, useCatalog } from "../store/catalog";
import type { LayerNode } from "../store/graph";
import { ActivationThumbnail } from "../viz/ActivationView";

/**
 * > [!AML-DOC-UNIT]
 * Format an inferred shape for the chip on the node.
 * @param shape the shape the engine reported, or null before validation
 * @returns text such as "26 x 26 x 32", with the batch axis dropped as noise
 */
function formatShape(shape: ShapeInfo | null): string {
  if (!shape) return "—";
  const dims = shape.shape.slice(1);
  if (dims.length === 0) return "scalar";
  return dims.map((dim) => (dim === null ? "?" : String(dim))).join(" × ");
}

/**
 * > [!AML-DOC-UNIT]
 * Summarise the parameters worth seeing at a glance, without opening the panel.
 * @param params the node's current parameters
 * @returns a short string such as "filters 32 · kernel 3×3", or "" when nothing
 *          notable is set
 */
function summarise(params: Record<string, unknown>): string {
  const parts: string[] = [];
  const show = (key: string, label: string) => {
    const value = params[key];
    if (value === null || value === undefined) return;
    parts.push(`${label} ${Array.isArray(value) ? value.join("×") : String(value)}`);
  };
  show("units", "units");
  show("filters", "filters");
  show("kernel_size", "kernel");
  show("pool_size", "pool");
  show("rate", "rate");
  show("num_heads", "heads");
  show("output_dim", "dim");
  return parts.slice(0, 2).join(" · ");
}

/**
 * > [!AML-DOC-UNIT]
 * Render one layer on the canvas.
 * @param id       the node's id, shown when the layer has been renamed
 * @param data     the node's editor state: type, name, params, shape, diagnostics
 * @param selected whether React Flow considers it selected
 * @returns the node element
 * @sideEffects none
 */

/**
 * > [!AML-DOC-UNIT]
 * The output ports a node actually has, which its parameters can add to.
 * @param declared the ports the catalog declares
 * @param params   the node's current parameters
 * @returns every output port, in order
 * @sideEffects none
 * @context Nearly every layer's ports are fixed, and the catalog is where they
 *          belong. A recurrent layer is the exception: `return_state` decides at
 *          configuration time whether it also hands back its state, and LSTM hands
 *          back two states where GRU hands back one. Drawing only the declared port
 *          meant an imported free-running model lost the state connection on arrival
 *          — the edge was in the graph and had nowhere to land [E-034].
 */
function outputPorts(
  declared: { name: string; label: string }[],
  params: Record<string, unknown>,
  typeId: string,
): { name: string; label: string }[] {
  if (params.return_state !== true) return declared;
  // LSTM carries two states, the hidden state and the cell state; GRU and SimpleRNN
  // carry one.
  const extra = [{ name: "output_1", label: "State" }];
  if (typeId.endsWith(".LSTM")) extra.push({ name: "output_2", label: "Cell state" });
  return [...declared, ...extra];
}

export const LayerNodeView = memo(function LayerNodeView({
  id,
  data,
  selected,
}: NodeProps<LayerNode>) {
  const spec = useCatalog((state) => state.specs.get(data.typeId));
  const activation = useActivations((state) => state.byNode[id]);
  const accent = spec ? categoryAccent(spec.category) : "var(--color-cat-core)";
  const errors = data.diagnostics.filter((diagnostic) => diagnostic.severity === "error");
  const warnings = data.diagnostics.filter((diagnostic) => diagnostic.severity === "warning");
  const summary = summarise(data.params);

  const ring = errors.length
    ? "ring-2 ring-danger"
    : warnings.length
      ? "ring-2 ring-warn"
      : selected
        ? "ring-2 ring-accent"
        : "ring-1 ring-line";

  const inputs = spec?.inputs ?? [];
  const outputs = outputPorts(spec?.outputs ?? [], data.params, data.typeId);

  return (
    <div
      className={`min-w-[168px] rounded-lg bg-surface-1 ${ring} ${
        data.disabled ? "opacity-45" : ""
      }`}
      title={errors[0]?.message ?? warnings[0]?.message ?? spec?.description ?? ""}
    >
      {inputs.map((port, index) => (
        <Handle
          key={port.name}
          id={port.name}
          type="target"
          position={Position.Top}
          isConnectable
          style={{
            left: `${((index + 1) / (inputs.length + 1)) * 100}%`,
            background: "var(--color-surface-3)",
            borderColor: "var(--color-line)",
          }}
          title={inputs.length > 1 ? port.label : undefined}
        />
      ))}

      <div className="h-1 rounded-t-lg" style={{ background: accent }} />

      <div className="px-2.5 pt-1.5 pb-2">
        <div className="flex items-baseline justify-between gap-2">
          <span className="truncate text-[12px] font-semibold text-ink-0">{data.name}</span>
          {data.disabled && (
            <span className="shrink-0 text-[9px] uppercase tracking-wide text-ink-2">off</span>
          )}
        </div>
        <div className="truncate text-[10px] text-ink-2" style={{ color: accent }}>
          {spec?.label ?? data.typeId}
        </div>

        {summary && <div className="mt-1 truncate text-[10px] text-ink-1">{summary}</div>}

        <div className="mt-1.5 flex items-center gap-1.5">
          <span
            className="rounded bg-surface-2 px-1.5 py-0.5 font-mono text-[10px] text-ink-1"
            title={
              data.shape
                ? `${data.shape.dtype}, ${data.shape.params.toLocaleString("en-US")} weights`
                : "not validated yet"
            }
          >
            {formatShape(data.shape)}
          </span>
          {data.shape && data.shape.params > 0 && (
            <span className="text-[9px] text-ink-2">
              {data.shape.params.toLocaleString("en-US")}w
            </span>
          )}
        </div>

        {activation && <ActivationThumbnail activation={activation} />}

        {inputs.length > 1 && (
          <div className="mt-1.5 flex gap-1">
            {inputs.map((port) => (
              <span
                key={port.name}
                className="rounded bg-surface-2 px-1 text-[9px] text-ink-2"
                title={`input port: ${port.name}`}
              >
                {port.name}
              </span>
            ))}
          </div>
        )}
      </div>

      {outputs.map((port, index) => (
        <Handle
          key={port.name}
          id={port.name}
          type="source"
          position={Position.Bottom}
          isConnectable
          style={{
            left: `${((index + 1) / (outputs.length + 1)) * 100}%`,
            background: accent,
            borderColor: "var(--color-line)",
          }}
          title={outputs.length > 1 ? port.label : undefined}
        />
      ))}
    </div>
  );
});
