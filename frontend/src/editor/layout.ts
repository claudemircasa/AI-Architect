/**
 * > [!AML-DOC-FILE]
 * @file        src/editor/layout.ts
 * @description Arranges the graph top-to-bottom with dagre, so a model drawn in any
 *              order can be made readable in one action.
 * @module      frontend/editor/layout
 * @exports     autoLayout, NODE_WIDTH, NODE_HEIGHT
 * @created     2026-09-30
 * @context     Serves task 09 item 2. Data flows downward here because that is how
 *              the generated `model.py` reads, and because layer stacks are tall and
 *              narrow rather than wide.
 */

import dagre from "@dagrejs/dagre";

import type { LayerEdge, LayerNode } from "../store/graph";

/** Assumed node size, used only for spacing; React Flow measures the real size. */
export const NODE_WIDTH = 190;
export const NODE_HEIGHT = 92;

/**
 * > [!AML-DOC-UNIT]
 * Compute tidy positions for every node.
 * @param nodes the current nodes, whose measured sizes are used when available
 * @param edges the current edges, which define the ranking
 * @returns the nodes with new positions; the input array is not mutated
 * @sideEffects none
 * @context Disconnected nodes are laid out by dagre too, in their own components, so
 *          a half-built graph still tidies sensibly rather than piling up at the
 *          origin.
 */
export function autoLayout(nodes: LayerNode[], edges: LayerEdge[]): LayerNode[] {
  if (nodes.length === 0) return nodes;

  const graph = new dagre.graphlib.Graph();
  graph.setDefaultEdgeLabel(() => ({}));
  graph.setGraph({ rankdir: "TB", ranksep: 64, nodesep: 36, marginx: 24, marginy: 24 });

  for (const node of nodes) {
    graph.setNode(node.id, {
      width: node.measured?.width ?? NODE_WIDTH,
      height: node.measured?.height ?? NODE_HEIGHT,
    });
  }
  for (const edge of edges) {
    if (graph.hasNode(edge.source) && graph.hasNode(edge.target)) {
      graph.setEdge(edge.source, edge.target);
    }
  }

  dagre.layout(graph);

  return nodes.map((node) => {
    const placed = graph.node(node.id);
    if (!placed) return node;
    return {
      ...node,
      position: {
        x: Math.round(placed.x - (node.measured?.width ?? NODE_WIDTH) / 2),
        y: Math.round(placed.y - (node.measured?.height ?? NODE_HEIGHT) / 2),
      },
    };
  });
}
