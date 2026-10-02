/**
 * > [!AML-DOC-FILE]
 * @file        src/store/graph.ts
 * @description Editor state: the canvas graph, selection, undo history, and the
 *              debounced round-trip that annotates nodes with real shapes.
 * @module      frontend/store/graph
 * @exports     useGraph, type LayerNodeData, type LayerNode, type LayerEdge
 * @created     2026-09-30
 * @context     RISK:MED [amm: B.6]. Undo/redo invariants live here. The canvas holds
 *              the authoritative state and `toIR()` projects it onto the wire format
 *              [amm: E.1]; shapes are never computed locally, only received.
 */

import {
  addEdge,
  applyEdgeChanges,
  applyNodeChanges,
  type Connection,
  type Edge,
  type EdgeChange,
  type Node,
  type NodeChange,
} from "@xyflow/react";
import { create } from "zustand";

import { validateGraph } from "../api/client";
import { autoLayout } from "../editor/layout";
import type { Diagnostic, GraphIR, LayerSpec, ShapeInfo } from "../api/types";
import { useCatalog } from "./catalog";

const IR_VERSION = 1;
const VALIDATE_DEBOUNCE_MS = 250;

/** Above this many nodes, validation waits longer between edits. */
const LARGE_GRAPH_NODES = 240;

/** Debounce used for a large graph, whose IR is megabytes on the wire. */
const LARGE_GRAPH_DEBOUNCE_MS = 1_200;
const HISTORY_LIMIT = 120;

/** Per-node editor state carried in the React Flow node's `data`. */
export interface LayerNodeData extends Record<string, unknown> {
  typeId: string;
  /** Set only on a folded group node: what the run is made of. */
  groupSummary?: string;
  /** Set only on a folded group node: how many layers it stands for. */
  groupSize?: number;
  name: string;
  params: Record<string, unknown>;
  disabled: boolean;
  shape: ShapeInfo | null;
  diagnostics: Diagnostic[];
}

export type LayerNode = Node<LayerNodeData, "layer" | "group">;
export type LayerEdge = Edge;

interface Snapshot {
  nodes: LayerNode[];
  edges: LayerEdge[];
}

/** Which reading of the project is on screen. */
export type ViewMode = "architecture" | "graph";

interface GraphStore {
  name: string;
  view: ViewMode;
  setView: (view: ViewMode) => void;
  /** Ids of folded runs the user has opened. */
  expandedGroups: Set<string>;
  /** Whether long straight runs are folded at all. */
  grouping: boolean;
  expandGroup: (id: string) => void;
  collapseAll: () => void;
  setGrouping: (grouping: boolean) => void;
  nodes: LayerNode[];
  edges: LayerEdge[];
  selection: string[];
  diagnostics: Diagnostic[];
  paramsTotal: number;
  trainableParams: number;
  compiled: boolean;
  validating: boolean;
  dirty: boolean;
  past: Snapshot[];
  future: Snapshot[];
  clipboard: Snapshot | null;

  onNodesChange: (changes: NodeChange<LayerNode>[]) => void;
  onEdgesChange: (changes: EdgeChange<LayerEdge>[]) => void;
  connect: (connection: Connection) => void;
  isValidConnection: (connection: Connection | Edge) => boolean;

  addLayer: (
    spec: LayerSpec,
    position: { x: number; y: number },
    appendTo?: string | null,
  ) => string;
  tidy: () => void;
  updateParam: (nodeId: string, param: string, value: unknown) => void;
  renameNode: (nodeId: string, name: string) => void;
  toggleDisabled: (nodeId: string) => void;
  deleteSelection: () => void;
  duplicateSelection: () => void;
  copySelection: () => void;
  paste: () => void;
  selectAll: () => void;
  setSelection: (ids: string[]) => void;
  setName: (name: string) => void;
  replaceGraph: (snapshot: Snapshot, name: string) => void;

  undo: () => void;
  redo: () => void;
  commit: () => void;

  toIR: () => GraphIR;
  validate: () => void;
}

let validateTimer: ReturnType<typeof setTimeout> | null = null;
let validateAbort: AbortController | null = null;
let nodeCounter = 0;

/**
 * > [!AML-DOC-UNIT]
 * Mint a graph-unique node id.
 * @param typeId catalog layer id, whose last segment seeds a readable prefix
 * @returns an id such as "conv2d_3"
 * @sideEffects increments a module-level counter
 */
function nextNodeId(typeId: string): string {
  nodeCounter += 1;
  const leaf = typeId.split(".").pop() ?? "node";
  return `${leaf.toLowerCase()}_${nodeCounter}`;
}

/**
 * > [!AML-DOC-UNIT]
 * Starting parameters for a freshly dropped layer.
 * @param spec the layer's catalog spec
 * @returns the spec's non-null defaults, so the node is valid on arrival
 * @sideEffects none
 */
function initialParams(spec: LayerSpec): Record<string, unknown> {
  const params: Record<string, unknown> = {};
  for (const param of spec.params) {
    if (param.default !== null && param.default !== undefined) {
      params[param.name] = param.default;
    }
  }
  return params;
}

/**
 * > [!AML-DOC-UNIT]
 * Count the edges already landing on one input port.
 * @param edges     every edge in the graph
 * @param nodeId    the consuming node
 * @param port      the port name, or null to count all of the node's inputs
 * @returns the number of connections
 * @sideEffects none
 */
function incomingCount(edges: LayerEdge[], nodeId: string, port?: string | null): number {
  return edges.filter(
    (edge) =>
      edge.target === nodeId && (port == null || (edge.targetHandle ?? "input") === port),
  ).length;
}

/**
 * > [!AML-DOC-UNIT]
 * Test whether adding an edge would create a cycle.
 * @param edges  the current edges
 * @param source id of the would-be producing node
 * @param target id of the would-be consuming node
 * @returns True when `source` is already reachable from `target`
 * @sideEffects none
 * @context Checked in the editor so an invalid connection is refused at the cursor
 *          rather than becoming a diagnostic a moment later. The backend still
 *          detects cycles independently [task 03].
 */
function wouldCycle(edges: LayerEdge[], source: string, target: string): boolean {
  const successors = new Map<string, string[]>();
  for (const edge of edges) {
    const list = successors.get(edge.source) ?? [];
    list.push(edge.target);
    successors.set(edge.source, list);
  }
  const seen = new Set<string>();
  const stack = [target];
  while (stack.length) {
    const current = stack.pop() as string;
    if (current === source) return true;
    if (seen.has(current)) continue;
    seen.add(current);
    stack.push(...(successors.get(current) ?? []));
  }
  return false;
}

/**
 * > [!AML-DOC-UNIT]
 * Editor store. The canvas reads and writes this; every mutation that a user would
 * expect to undo calls `commit()` first to snapshot the previous state.
 */
export const useGraph = create<GraphStore>((set, get) => ({
  name: "Untitled",
  view: "architecture",
  expandedGroups: new Set<string>(),
  grouping: true,
  nodes: [],
  edges: [],
  selection: [],
  diagnostics: [],
  paramsTotal: 0,
  trainableParams: 0,
  compiled: false,
  validating: false,
  dirty: false,
  past: [],
  future: [],
  clipboard: null,

  /**
   * > [!AML-DOC-UNIT]
   * Switch between the editing canvas and the analytical views.
   * @param view which reading to show
   * @sideEffects none beyond the store; both read the same graph
   */
  setView: (view) => set({ view }),

  /**
   * > [!AML-DOC-UNIT]
   * Open one folded run.
   * @param id the group's id
   * @sideEffects changes only what is drawn; the graph itself is untouched
   */
  expandGroup: (id) =>
    set({ expandedGroups: new Set([...get().expandedGroups, id]) }),

  /**
   * > [!AML-DOC-UNIT]
   * Fold every run again.
   * @sideEffects changes only what is drawn
   */
  collapseAll: () => set({ expandedGroups: new Set<string>() }),

  /**
   * > [!AML-DOC-UNIT]
   * Turn folding on or off.
   * @param grouping whether long straight runs should fold
   * @sideEffects changes only what is drawn
   */
  setGrouping: (grouping) => set({ grouping, expandedGroups: new Set<string>() }),

  /**
   * > [!AML-DOC-UNIT]
   * Push the current graph onto the undo stack and drop the redo stack.
   * @sideEffects trims history to the most recent HISTORY_LIMIT snapshots
   */
  commit: () => {
    // Captured activations describe the graph as it was; once it changes they are
    // about a model that no longer exists, and showing them would be a lie.
    void import("./activations").then(({ useActivations }) => {
      if (useActivations.getState().activations.length) useActivations.getState().clear();
    });
    const { nodes, edges, past } = get();
    set({
      past: [...past, { nodes, edges }].slice(-HISTORY_LIMIT),
      future: [],
      dirty: true,
    });
  },

  /**
   * > [!AML-DOC-UNIT]
   * Apply React Flow node changes.
   * @param changes the change set React Flow produced
   * @sideEffects snapshots history before a change that is not a drag in progress,
   *              and revalidates when the graph's meaning changed
   * @context Position changes while `dragging` is true are deliberately not
   *          snapshotted, so one drag is one undo step rather than dozens.
   */
  onNodesChange: (changes) => {
    const structural = changes.some(
      (change) =>
        change.type === "remove" ||
        change.type === "add" ||
        (change.type === "position" && change.dragging === false),
    );
    if (structural) get().commit();
    set({ nodes: applyNodeChanges(changes, get().nodes) });
    if (changes.some((change) => change.type === "remove")) get().validate();
    if (changes.some((change) => change.type === "select")) {
      set({ selection: get().nodes.filter((node) => node.selected).map((node) => node.id) });
    }
  },

  /**
   * > [!AML-DOC-UNIT]
   * Apply React Flow edge changes.
   * @param changes the change set React Flow produced
   * @sideEffects snapshots history on removal and revalidates
   */
  onEdgesChange: (changes) => {
    if (changes.some((change) => change.type === "remove")) {
      get().commit();
      set({ edges: applyEdgeChanges(changes, get().edges) });
      get().validate();
      return;
    }
    set({ edges: applyEdgeChanges(changes, get().edges) });
  },

  /**
   * > [!AML-DOC-UNIT]
   * Refuse connections the graph cannot express, at the moment the user draws them.
   * @param connection the proposed connection
   * @returns True when the edge is allowed
   * @sideEffects none
   */
  isValidConnection: (connection) => {
    const { source, target, targetHandle } = connection as Connection;
    if (!source || !target || source === target) return false;

    const node = get().nodes.find((candidate) => candidate.id === target);
    const spec = node ? useCatalog.getState().spec(node.data.typeId) : undefined;
    if (!spec) return false;

    if (spec.max_inputs !== null) {
      if (incomingCount(get().edges, target) >= spec.max_inputs) return false;
    }
    if (spec.inputs.length > 1 && targetHandle) {
      if (incomingCount(get().edges, target, targetHandle) >= 1) return false;
    }
    return !wouldCycle(get().edges, source, target);
  },

  /**
   * > [!AML-DOC-UNIT]
   * Add a connection the user drew.
   * @param connection the accepted connection
   * @sideEffects snapshots history and revalidates
   * @context `order` records the position among edges sharing a target port, which
   *          is what makes residual connections unambiguous [task 03].
   */
  connect: (connection) => {
    get().commit();
    const port = connection.targetHandle ?? "input";
    const order = incomingCount(get().edges, connection.target, port);
    set({
      edges: addEdge(
        { ...connection, id: `e${Date.now()}_${order}`, data: { order } },
        get().edges,
      ),
    });
    get().validate();
  },

  /**
   * > [!AML-DOC-UNIT]
   * Place a new layer on the canvas.
   * @param spec     the layer to instantiate
   * @param position canvas coordinates to place it at, used when it is not appended
   * @param appendTo id of a node to chain this one after, which also positions it
   *                 directly below that node
   * @returns the new node's id
   * @sideEffects snapshots history, selects the new node, may add an edge, and
   *              revalidates
   * @context Appending exists because building a stack is the common case: clicking
   *          five layers in a row should produce a connected chain, not five loose
   *          boxes the user must then wire by hand.
   */
  addLayer: (spec, position, appendTo) => {
    get().commit();
    const id = nextNodeId(spec.id);
    const parent = appendTo ? get().nodes.find((node) => node.id === appendTo) : undefined;
    const placement = parent
      ? {
          x: parent.position.x,
          y: parent.position.y + (parent.measured?.height ?? 92) + 56,
        }
      : position;

    const node: LayerNode = {
      id,
      type: "layer",
      position: placement,
      selected: true,
      data: {
        typeId: spec.id,
        name: spec.label,
        params: initialParams(spec),
        disabled: false,
        shape: null,
        diagnostics: [],
      },
    };

    const edges = [...get().edges];
    if (parent && spec.min_inputs > 0 && spec.inputs.length > 0) {
      const port = spec.inputs[0]?.name ?? "input";
      edges.push({
        id: `e${Date.now()}_auto`,
        source: parent.id,
        sourceHandle: "output",
        target: id,
        targetHandle: port,
        data: { order: 0 },
      });
    }

    set({
      nodes: [...get().nodes.map((n) => ({ ...n, selected: false })), node],
      edges,
      selection: [id],
    });
    get().validate();
    return id;
  },

  /**
   * > [!AML-DOC-UNIT]
   * Arrange the graph top-to-bottom so it can be read.
   * @sideEffects snapshots history and rewrites every node position
   */
  tidy: () => {
    if (get().nodes.length === 0) return;
    get().commit();
    set({ nodes: autoLayout(get().nodes, get().edges) });
  },

  /**
   * > [!AML-DOC-UNIT]
   * Change one parameter of one node.
   * @param nodeId the node to edit
   * @param param  the parameter name
   * @param value  the new value
   * @sideEffects snapshots history and revalidates, so downstream shapes update
   */
  updateParam: (nodeId, param, value) => {
    get().commit();
    set({
      nodes: get().nodes.map((node) =>
        node.id === nodeId
          ? { ...node, data: { ...node.data, params: { ...node.data.params, [param]: value } } }
          : node,
      ),
    });
    get().validate();
  },

  /**
   * > [!AML-DOC-UNIT]
   * Rename a node.
   * @param nodeId the node to rename
   * @param name   the new label
   * @sideEffects snapshots history and revalidates, since the name reaches the model
   */
  renameNode: (nodeId, name) => {
    get().commit();
    set({
      nodes: get().nodes.map((node) =>
        node.id === nodeId ? { ...node, data: { ...node.data, name } } : node,
      ),
    });
    get().validate();
  },

  /**
   * > [!AML-DOC-UNIT]
   * Ablate or restore a node without deleting it.
   * @param nodeId the node to toggle
   * @sideEffects snapshots history and revalidates
   */
  toggleDisabled: (nodeId) => {
    get().commit();
    set({
      nodes: get().nodes.map((node) =>
        node.id === nodeId
          ? { ...node, data: { ...node.data, disabled: !node.data.disabled } }
          : node,
      ),
    });
    get().validate();
  },

  /**
   * > [!AML-DOC-UNIT]
   * Delete the selected nodes and any edges touching them.
   * @sideEffects snapshots history and revalidates
   */
  deleteSelection: () => {
    const doomed = new Set(get().nodes.filter((node) => node.selected).map((n) => n.id));
    const selectedEdges = new Set(get().edges.filter((edge) => edge.selected).map((e) => e.id));
    if (!doomed.size && !selectedEdges.size) return;
    get().commit();
    set({
      nodes: get().nodes.filter((node) => !doomed.has(node.id)),
      edges: get().edges.filter(
        (edge) =>
          !selectedEdges.has(edge.id) && !doomed.has(edge.source) && !doomed.has(edge.target),
      ),
      selection: [],
    });
    get().validate();
  },

  /**
   * > [!AML-DOC-UNIT]
   * Copy the selection into the in-app clipboard.
   * @sideEffects none beyond the store
   */
  copySelection: () => {
    const nodes = get().nodes.filter((node) => node.selected);
    const ids = new Set(nodes.map((node) => node.id));
    set({
      clipboard: {
        nodes,
        edges: get().edges.filter((edge) => ids.has(edge.source) && ids.has(edge.target)),
      },
    });
  },

  /**
   * > [!AML-DOC-UNIT]
   * Paste the clipboard, offset so it does not land exactly on the original.
   * @sideEffects snapshots history, selects the pasted nodes and revalidates
   */
  paste: () => {
    const clipboard = get().clipboard;
    if (!clipboard?.nodes.length) return;
    get().commit();
    const remap = new Map<string, string>();
    const nodes = clipboard.nodes.map((node) => {
      const id = nextNodeId(node.data.typeId);
      remap.set(node.id, id);
      return {
        ...node,
        id,
        selected: true,
        position: { x: node.position.x + 40, y: node.position.y + 40 },
        data: { ...node.data, shape: null, diagnostics: [] },
      };
    });
    const edges = clipboard.edges.map((edge, index) => ({
      ...edge,
      id: `e${Date.now()}_p${index}`,
      source: remap.get(edge.source) as string,
      target: remap.get(edge.target) as string,
    }));
    set({
      nodes: [...get().nodes.map((node) => ({ ...node, selected: false })), ...nodes],
      edges: [...get().edges, ...edges],
      selection: nodes.map((node) => node.id),
    });
    get().validate();
  },

  /**
   * > [!AML-DOC-UNIT]
   * Copy then immediately paste the selection.
   * @sideEffects as `copySelection` and `paste`
   */
  duplicateSelection: () => {
    get().copySelection();
    get().paste();
  },

  /**
   * > [!AML-DOC-UNIT]
   * Select every node.
   * @sideEffects does not touch history, since selection is not undoable
   */
  selectAll: () => {
    set({
      nodes: get().nodes.map((node) => ({ ...node, selected: true })),
      selection: get().nodes.map((node) => node.id),
    });
  },

  /**
   * > [!AML-DOC-UNIT]
   * Select an explicit set of nodes, as when clicking through from another view.
   * @param ids node ids to select
   */
  setSelection: (ids) => {
    const wanted = new Set(ids);
    set({
      nodes: get().nodes.map((node) => ({ ...node, selected: wanted.has(node.id) })),
      selection: ids,
    });
  },

  /**
   * > [!AML-DOC-UNIT]
   * Rename the project.
   * @param name the new project name
   * @sideEffects marks the project dirty
   */
  setName: (name) => set({ name, dirty: true }),

  /**
   * > [!AML-DOC-UNIT]
   * Replace the whole graph, as when opening a project.
   * @param snapshot the nodes and edges to load
   * @param name     the project name
   * @sideEffects clears history and revalidates; the loaded graph is not dirty
   */
  replaceGraph: (snapshot, name) => {
    nodeCounter = snapshot.nodes.length;
    set({
      ...snapshot,
      name,
      expandedGroups: new Set<string>(),
      selection: [],
      past: [],
      future: [],
      dirty: false,
      diagnostics: [],
    });
    get().validate();
  },

  /**
   * > [!AML-DOC-UNIT]
   * Step back one change.
   * @sideEffects moves the current state onto the redo stack and revalidates
   */
  undo: () => {
    const { past, nodes, edges, future } = get();
    const previous = past[past.length - 1];
    if (!previous) return;
    set({
      ...previous,
      past: past.slice(0, -1),
      future: [...future, { nodes, edges }].slice(-HISTORY_LIMIT),
      dirty: true,
    });
    get().validate();
  },

  /**
   * > [!AML-DOC-UNIT]
   * Step forward one undone change.
   * @sideEffects moves the current state back onto the undo stack and revalidates
   */
  redo: () => {
    const { future, nodes, edges, past } = get();
    const next = future[future.length - 1];
    if (!next) return;
    set({
      ...next,
      future: future.slice(0, -1),
      past: [...past, { nodes, edges }].slice(-HISTORY_LIMIT),
      dirty: true,
    });
    get().validate();
  },

  /**
   * > [!AML-DOC-UNIT]
   * Project the canvas onto the wire format the engine expects.
   * @returns the graph IR
   * @sideEffects none
   * @context Shapes and diagnostics are editor-only annotations and are deliberately
   *          not sent back [amm: E.1].
   */
  toIR: () => ({
    ir_version: IR_VERSION,
    name: get().name,
    nodes: get().nodes.map((node) => ({
      id: node.id,
      type: node.data.typeId,
      name: node.data.name,
      params: node.data.params,
      position: { x: Math.round(node.position.x), y: Math.round(node.position.y) },
      notes: null,
      disabled: node.data.disabled,
    })),
    edges: get().edges.map((edge) => ({
      id: edge.id,
      source: edge.source,
      source_port: edge.sourceHandle ?? "output",
      target: edge.target,
      target_port: edge.targetHandle ?? "input",
      order: typeof edge.data?.order === "number" ? edge.data.order : 0,
    })),
    inputs: [],
    outputs: [],
    meta: {},
  }),

  /**
   * > [!AML-DOC-UNIT]
   * Ask the engine to validate the graph, debounced, and write the shapes and
   * diagnostics it returns back onto the nodes.
   * @sideEffects cancels any in-flight validation, schedules a new one 250ms out,
   *              and mutates node data when the response arrives
   * @context Debounced because this fires on every keystroke in the property panel.
   *          The previous request is aborted so a slow earlier response cannot
   *          overwrite a newer one [task 09]. The wait scales with the graph, since
   *          a large one costs megabytes per round trip.
   */
  validate: () => {
    if (validateTimer) clearTimeout(validateTimer);
    // A thousand-layer graph is megabytes of JSON per request, so it is checked
    // less eagerly. Small graphs keep the responsive 250ms, which is what makes
    // editing feel immediate.
    const delay =
      get().nodes.length > LARGE_GRAPH_NODES
        ? LARGE_GRAPH_DEBOUNCE_MS
        : VALIDATE_DEBOUNCE_MS;
    validateTimer = setTimeout(() => {
      if (get().nodes.length === 0) {
        set({ diagnostics: [], paramsTotal: 0, trainableParams: 0, compiled: false });
        return;
      }
      validateAbort?.abort();
      const controller = new AbortController();
      validateAbort = controller;
      set({ validating: true });

      validateGraph(get().toIR(), controller.signal)
        .then((response) => {
          const byNode = new Map<string, Diagnostic[]>();
          for (const diagnostic of response.diagnostics) {
            if (!diagnostic.node_id) continue;
            const list = byNode.get(diagnostic.node_id) ?? [];
            list.push(diagnostic);
            byNode.set(diagnostic.node_id, list);
          }
          set({
            nodes: get().nodes.map((node) => ({
              ...node,
              data: {
                ...node.data,
                shape: response.shapes[node.id] ?? null,
                diagnostics: byNode.get(node.id) ?? [],
              },
            })),
            diagnostics: response.diagnostics,
            paramsTotal: response.params_total,
            trainableParams: response.trainable_params,
            compiled: response.compiled,
            validating: false,
          });
        })
        .catch((error: unknown) => {
          if (error instanceof DOMException && error.name === "AbortError") return;
          set({ validating: false });
        });
    }, delay);
  },
}));

// Exposed for the browser-driven UI checks in scripts/ui-check.mjs, which need to
// assert on history depth rather than only on what is rendered.
if (typeof window !== "undefined") {
  (window as unknown as { __graphStore?: typeof useGraph }).__graphStore = useGraph;
}
