/**
 * > [!AML-DOC-FILE]
 * @file        src/editor/GroupNode.tsx
 * @description Canvas renderer for a folded run of layers.
 * @module      frontend/editor/GroupNode
 * @exports     GroupNodeView
 * @created     2026-10-01
 * @context     Stands in for several layers at once [task 09, large graphs]. It says
 *              how many it holds and what they are, so folding summarises rather than
 *              hides, and clicking it opens the run.
 */

import { Handle, Position, type NodeProps } from "@xyflow/react";
import { memo } from "react";

import type { LayerNode } from "../store/graph";
import { useGraph } from "../store/graph";

/**
 * > [!AML-DOC-UNIT]
 * Render a folded run.
 * @param id       the group's id, used to open it
 * @param data     the run's summary, size, output shape and gathered diagnostics
 * @param selected whether React Flow considers it selected
 * @returns the node element
 * @sideEffects opens the run when clicked
 */
export const GroupNodeView = memo(function GroupNodeView({
  id,
  data,
  selected,
}: NodeProps<LayerNode>) {
  const expandGroup = useGraph((state) => state.expandGroup);
  const errors = data.diagnostics.filter((d) => d.severity === "error").length;
  const size = Number(data.groupSize ?? 0);
  const shape = data.shape?.shape;

  return (
    <button
      type="button"
      onClick={() => expandGroup(id)}
      title={`${size} layers folded into one. Click to open them.`}
      className={`min-w-[172px] rounded-lg bg-surface-1 text-left ${
        errors ? "ring-2 ring-danger" : selected ? "ring-2 ring-accent" : "ring-1 ring-line"
      }`}
      style={{
        // A single offset edge rather than a stack of shadows and a gradient: this
        // node is drawn dozens of times at once on a large graph, and compositing
        // cost per node is what sets the frame rate while panning.
        borderTop: "3px double var(--color-ink-2)",
      }}
    >
      <Handle type="target" position={Position.Top} id="input" />

      <div className="px-2.5 pt-1.5 pb-2">
        <div className="flex items-baseline justify-between gap-2">
          <span className="truncate text-[12px] font-semibold text-ink-0">
            {size} layers
          </span>
          <span className="shrink-0 text-[9px] uppercase tracking-wide text-ink-2">
            folded
          </span>
        </div>

        <div className="mt-0.5 truncate text-[10px] text-ink-1" title={String(data.groupSummary ?? "")}>
          {String(data.groupSummary ?? "")}
        </div>

        <div className="mt-1.5 flex items-center gap-1.5">
          <span className="rounded bg-surface-2 px-1.5 py-0.5 font-mono text-[10px] text-ink-1">
            {shape && shape.length > 1
              ? shape.slice(1).map((dim) => dim ?? "?").join(" × ")
              : "—"}
          </span>
          {errors > 0 && (
            <span className="text-[9px] text-danger">{errors} errors inside</span>
          )}
        </div>

        <div className="mt-1 text-[9px] text-ink-2">click to open</div>
      </div>

      <Handle type="source" position={Position.Bottom} id="output" />
    </button>
  );
});
