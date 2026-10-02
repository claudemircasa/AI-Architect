/**
 * > [!AML-DOC-FILE]
 * @file        src/editor/Canvas.tsx
 * @description The editing canvas: drop targets, connections, selection, and the
 *              keyboard shortcuts that act on them.
 * @module      frontend/editor/Canvas
 * @exports     Canvas
 * @created     2026-09-30
 * @context     React Flow owns pan, zoom and hit-testing; the graph store owns
 *              meaning [amm: B.6]. Connections the graph cannot express are refused
 *              at the cursor by `isValidConnection`, while the engine still checks
 *              independently [task 03].
 *
 *              `fitViewOptions.maxZoom` is capped at 1: an unconstrained fitView on
 *              an empty or single-node canvas zooms to the maximum, which left
 *              newly added layers outside the visible area.
 *
 *              Past `GROUPING_THRESHOLD` nodes the canvas draws a folded projection
 *              and mounts only what is on screen. Both are presentation: the store
 *              keeps the whole graph, so nothing downstream sees a folded model.
 */

import {
  Background,
  BackgroundVariant,
  Controls,
  MiniMap,
  ReactFlow,
  useReactFlow,
  type NodeTypes,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { useCallback, useEffect, useMemo, useRef } from "react";

import { categoryAccent, useCatalog } from "../store/catalog";
import { nativeMenuOwnsShortcuts } from "../shell/menu";
import { useGraph, type LayerNode } from "../store/graph";
import { GroupNodeView } from "./GroupNode";
import { findChains, GROUPING_THRESHOLD, projectGraph } from "./grouping";
import { LayerNodeView } from "./LayerNode";
import { LAYER_DRAG_TYPE } from "./Palette";
import { centreOn, fitGraphInView, setFlowInstance, zoomCanvasBy } from "./viewport";

const NODE_TYPES: NodeTypes = { layer: LayerNodeView, group: GroupNodeView };
const SNAP_GRID: [number, number] = [16, 16];

/**
 * > [!AML-DOC-UNIT]
 * The canvas pane.
 * @returns the React Flow surface with its background, controls and minimap
 * @sideEffects binds window-level key handlers while mounted, and mutates the graph
 *              store on drop, connect, move and delete
 */
export function Canvas() {
  const allNodes = useGraph((state) => state.nodes);
  const allEdges = useGraph((state) => state.edges);
  const grouping = useGraph((state) => state.grouping);
  const expandedGroups = useGraph((state) => state.expandedGroups);
  const onNodesChange = useGraph((state) => state.onNodesChange);
  const onEdgesChange = useGraph((state) => state.onEdgesChange);
  const connect = useGraph((state) => state.connect);
  const isValidConnection = useGraph((state) => state.isValidConnection);
  const addLayer = useGraph((state) => state.addLayer);
  const specs = useCatalog((state) => state.specs);
  const { screenToFlowPosition } = useReactFlow();

  // Long straight runs are folded into one box each. This is drawing only: the
  // store still holds every node, so validation, export and saving are unaffected.
  // Without it a converted recurrent model arrives as thousands of gate operations
  // and the canvas becomes both unusably slow and a worse picture than one box
  // saying how many there are.
  const { nodes, edges } = useMemo(() => {
    if (!grouping || allNodes.length < GROUPING_THRESHOLD) {
      return { nodes: allNodes, edges: allEdges };
    }
    const groups = findChains(allNodes, allEdges, (node) =>
      specs.get(node.data.typeId)?.label ?? node.data.typeId.split(".").pop() ?? "layer",
    );
    return projectGraph(allNodes, allEdges, groups, expandedGroups);
  }, [allNodes, allEdges, grouping, expandedGroups, specs]);
  const surface = useRef<HTMLDivElement>(null);

  useEffect(() => () => setFlowInstance(null), []);

  // Frame the graph whenever the drawn node count changes by a lot, which is what
  // happens after an import and after folding is switched. Fitting only at the
  // moment of import is too early: the folded projection, and the positions that
  // come with it, do not exist until the graph has been through the store.
  // Move to whatever was just opened. The layers inside a folded box are laid out
  // by the projection, not at the positions the store holds, so only the canvas
  // knows where they ended up — and a click that appears to do nothing is worse
  // than no click at all.
  const lastExpanded = useRef(0);
  useEffect(() => {
    if (expandedGroups.size <= lastExpanded.current) {
      lastExpanded.current = expandedGroups.size;
      return;
    }
    lastExpanded.current = expandedGroups.size;
    const opened = [...expandedGroups].pop();
    if (!opened) return;
    const first = opened.replace("group:", "");
    const target = nodes.find((node) => node.id === first);
    if (target) centreOn(target.position);
  }, [expandedGroups, nodes]);

  const lastFitted = useRef(0);
  useEffect(() => {
    const drawn = nodes.length;
    if (drawn === 0) return;
    const previous = lastFitted.current;
    if (previous === 0 || Math.abs(drawn - previous) > Math.max(4, previous * 0.25)) {
      lastFitted.current = drawn;
      const timer = window.setTimeout(fitGraphInView, 60);
      return () => window.clearTimeout(timer);
    }
    lastFitted.current = drawn;
    return undefined;
  }, [nodes.length]);

  const onDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();
      const layerId = event.dataTransfer.getData(LAYER_DRAG_TYPE);
      const spec = specs.get(layerId);
      if (!spec) return;
      addLayer(spec, screenToFlowPosition({ x: event.clientX, y: event.clientY }));
    },
    [addLayer, screenToFlowPosition, specs],
  );

  useEffect(() => {
    /**
     * > [!AML-DOC-UNIT]
     * Handle the editor's keyboard shortcuts.
     * @param event the keydown event
     * @sideEffects mutates the graph store; ignored while a text field has focus so
     *              typing a layer name cannot delete the selection
     */
    function onKeyDown(event: KeyboardEvent) {
      const target = event.target as HTMLElement | null;
      if (
        target &&
        (target.tagName === "INPUT" ||
          target.tagName === "TEXTAREA" ||
          target.tagName === "SELECT" ||
          target.isContentEditable)
      ) {
        return;
      }

      const store = useGraph.getState();
      const meta = event.metaKey || event.ctrlKey;

      // Inside the shell these keys are menu accelerators, registered with the
      // operating system. Handling them here as well would undo twice on one
      // press, so the menu is left to own them [E-031]. Copy and paste are not in
      // that set: the menu's are the platform's text commands, and these are the
      // canvas's, which only this side can tell apart.
      if (nativeMenuOwnsShortcuts()) {
        const owned =
          (meta && ["z", "d"].includes(event.key.toLowerCase())) ||
          event.key === "Backspace" ||
          event.key === "Delete";
        if (owned) return;
      }

      if (meta && event.key.toLowerCase() === "z") {
        event.preventDefault();
        if (event.shiftKey) store.redo();
        else store.undo();
        return;
      }
      if (meta && event.key.toLowerCase() === "a") {
        event.preventDefault();
        store.selectAll();
        return;
      }
      if (meta && event.key.toLowerCase() === "d") {
        event.preventDefault();
        store.duplicateSelection();
        return;
      }
      if (meta && event.key.toLowerCase() === "c") {
        store.copySelection();
        return;
      }
      if (meta && event.key.toLowerCase() === "v") {
        store.paste();
        return;
      }
      if (event.key === "Backspace" || event.key === "Delete") {
        event.preventDefault();
        store.deleteSelection();
      }
    }

    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

  // The View menu's fit and zoom, when the canvas is the surface on screen.
  useEffect(() => {
    function onAsked(event: Event) {
      if (useGraph.getState().view !== "architecture") return;
      const which = (event as CustomEvent<string>).detail;
      if (which === "view.fit") fitGraphInView();
      else if (which === "view.zoomIn") zoomCanvasBy(1.25);
      else if (which === "view.zoomOut") zoomCanvasBy(1 / 1.25);
    }
    window.addEventListener("nnarch:viewport", onAsked);
    return () => window.removeEventListener("nnarch:viewport", onAsked);
  }, []);

  return (
    <div ref={surface} className="relative min-w-0 flex-1">
      <ReactFlow<LayerNode>
        nodes={nodes}
        edges={edges}
        nodeTypes={NODE_TYPES}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={connect}
        isValidConnection={isValidConnection}
        onDrop={onDrop}
        onDragOver={(event) => {
          event.preventDefault();
          event.dataTransfer.dropEffect = "copy";
        }}
        // Only what is on screen is mounted. A large graph otherwise puts thousands
        // of elements in the DOM, which is what makes panning crawl.
        onlyRenderVisibleElements={allNodes.length > GROUPING_THRESHOLD}
        snapToGrid
        snapGrid={SNAP_GRID}
        selectionOnDrag
        panOnDrag={[1, 2]}
        panOnScroll
        selectNodesOnDrag={false}
        deleteKeyCode={null}
        multiSelectionKeyCode={["Meta", "Shift"]}
        onInit={setFlowInstance}
        fitView
        // A floor as well as a ceiling. A tall graph cannot be made to fit a short
        // window without shrinking its boxes to specks — an imported model came out
        // at 18x14 pixels each, which is an empty canvas as far as anyone can tell.
        // Below this zoom the view anchors at the top instead and the user scrolls,
        // which is what a long column is for.
        fitViewOptions={{ maxZoom: 1, minZoom: 0.45, padding: 0.2 }}
        minZoom={0.1}
        maxZoom={2.5}
        defaultEdgeOptions={{ animated: false, style: { strokeWidth: 1.5 } }}
        proOptions={{ hideAttribution: true }}
      >
        <Background
          variant={BackgroundVariant.Dots}
          gap={16}
          size={1}
          color="var(--color-surface-2)"
        />
        <Controls
          showInteractive={false}
          className="!border-line !bg-surface-1 [&_button]:!border-line
                     [&_button]:!bg-surface-2 [&_button]:!fill-ink-1"
        />
        <MiniMap
          pannable
          zoomable
          maskColor="color-mix(in oklab, var(--color-surface-0) 70%, transparent)"
          className="!border !border-line !bg-surface-1"
          nodeColor={(node) => {
            const spec = specs.get((node.data as LayerNode["data"]).typeId);
            return spec ? categoryAccent(spec.category) : "var(--color-cat-core)";
          }}
        />
      </ReactFlow>

      {allNodes.length >= GROUPING_THRESHOLD && (
        <div className="pointer-events-none absolute left-3 top-3 z-10">
          <div className="pointer-events-auto flex items-center gap-2 rounded border border-line
                          bg-surface-1/95 px-2 py-1 text-[10px] text-ink-2">
            <span>
              {allNodes.length.toLocaleString("en-US")} layers
              {grouping && nodes.length < allNodes.length
                ? ` · showing ${nodes.length.toLocaleString("en-US")}`
                : ""}
            </span>
            <button
              type="button"
              className="rounded px-1.5 py-0.5 hover:bg-surface-2 hover:text-ink-1"
              onClick={() => useGraph.getState().setGrouping(!grouping)}
              title="Fold long straight runs of layers into one box each"
            >
              {grouping ? "unfold all" : "fold runs"}
            </button>
            {grouping && expandedGroups.size > 0 && (
              <button
                type="button"
                className="rounded px-1.5 py-0.5 hover:bg-surface-2 hover:text-ink-1"
                onClick={() => useGraph.getState().collapseAll()}
              >
                re-fold {expandedGroups.size}
              </button>
            )}
          </div>
        </div>
      )}

      {nodes.length === 0 && (
        <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
          <div className="max-w-sm text-center">
            <p className="text-[13px] font-semibold text-ink-1">Empty canvas</p>
            <p className="mt-1 text-[11px] leading-relaxed text-ink-2">
              Drag an <span className="text-ink-1">Input</span> layer from the palette to
              start, then add layers and connect them by dragging between the dots.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
