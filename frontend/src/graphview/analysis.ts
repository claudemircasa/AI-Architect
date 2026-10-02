/**
 * > [!AML-DOC-FILE]
 * @file        src/graphview/analysis.ts
 * @description Reads structure out of a graph: layering, the longest path, skip
 *              connections, and how much data moves along each edge.
 * @module      frontend/graphview/analysis
 * @exports     analyseGraph, type GraphAnalysis, type AnalysedNode, type AnalysedEdge
 * @created     2026-10-01
 * @context     Everything here is derived from the shapes the engine inferred, never
 *              from shape arithmetic done in the browser [task 03]. The numbers shown
 *              are the ones the compiled model really has.
 */

import type { Activation } from "../api/types";
import type { LayerEdge, LayerNode } from "../store/graph";

/** A node with the structural facts the views draw. */
export interface AnalysedNode {
  id: string;
  name: string;
  typeId: string;
  category: string;
  depth: number;
  parameters: number;
  shape: (number | null)[];
  /** Elements in one sample's output, which is what the sankey widths encode. */
  volume: number;
  onCriticalPath: boolean;
  disabled: boolean;
}

/** An edge with the volume of data that crosses it. */
export interface AnalysedEdge {
  id: string;
  source: string;
  target: string;
  volume: number;
  /** True when the edge jumps over at least one layer, as a residual does. */
  isSkip: boolean;
  onCriticalPath: boolean;
}

/** Everything the analytical views need. */
export interface GraphAnalysis {
  nodes: AnalysedNode[];
  edges: AnalysedEdge[];
  byDepth: AnalysedNode[][];
  totalParameters: number;
  maxVolume: number;
  criticalPathLength: number;
  skipCount: number;
}

/**
 * > [!AML-DOC-UNIT]
 * Count the elements in one sample's output.
 * @param shape the inferred shape, batch axis included
 * @returns the product of every fixed axis, with dynamic axes treated as 1
 * @sideEffects none
 */
function volumeOf(shape: (number | null)[]): number {
  return shape
    .slice(1)
    .reduce<number>((total, dim) => total * (dim && dim > 0 ? dim : 1), 1);
}

/**
 * > [!AML-DOC-UNIT]
 * Work out the structure of a graph.
 * @param nodes       the canvas nodes
 * @param edges       the canvas edges
 * @param activations captured activations, used only to name the shapes when a node
 *                    has not been validated
 * @returns the analysis the views draw from
 * @sideEffects none
 * @context Depth is the longest path from any source, not the shortest. That is what
 *          makes a residual branch sit beside the layers it skips rather than
 *          collapsing onto them.
 */
export function analyseGraph(
  nodes: LayerNode[],
  edges: LayerEdge[],
  activations: Record<string, Activation> = {},
): GraphAnalysis {
  const incoming = new Map<string, string[]>();
  const outgoing = new Map<string, string[]>();
  for (const node of nodes) {
    incoming.set(node.id, []);
    outgoing.set(node.id, []);
  }
  for (const edge of edges) {
    incoming.get(edge.target)?.push(edge.source);
    outgoing.get(edge.source)?.push(edge.target);
  }

  // Longest-path layering, computed over a topological order.
  const depth = new Map<string, number>();
  const indegree = new Map(nodes.map((node) => [node.id, incoming.get(node.id)?.length ?? 0]));
  const ready = nodes.filter((node) => (indegree.get(node.id) ?? 0) === 0).map((n) => n.id);
  const order: string[] = [];
  while (ready.length) {
    const id = ready.shift() as string;
    order.push(id);
    depth.set(id, depth.get(id) ?? 0);
    for (const next of outgoing.get(id) ?? []) {
      depth.set(next, Math.max(depth.get(next) ?? 0, (depth.get(id) ?? 0) + 1));
      const remaining = (indegree.get(next) ?? 1) - 1;
      indegree.set(next, remaining);
      if (remaining === 0) ready.push(next);
    }
  }
  for (const node of nodes) if (!depth.has(node.id)) depth.set(node.id, 0);

  // The critical path is the longest chain, walked back from the deepest node.
  const deepest = [...depth.entries()].sort((a, b) => b[1] - a[1])[0];
  const critical = new Set<string>();
  if (deepest) {
    let current: string | undefined = deepest[0];
    while (current !== undefined) {
      critical.add(current);
      const parents: string[] = incoming.get(current) ?? [];
      const deepestParent = parents
        .slice()
        .sort((a: string, b: string) => (depth.get(b) ?? 0) - (depth.get(a) ?? 0))[0];
      // A cycle cannot reach here, but a guard keeps a malformed graph from looping.
      current = deepestParent !== undefined && !critical.has(deepestParent)
        ? deepestParent
        : undefined;
    }
  }

  const analysedNodes: AnalysedNode[] = nodes.map((node) => {
    const shape = node.data.shape?.shape ?? activations[node.id]?.shape ?? [];
    return {
      id: node.id,
      name: node.data.name || node.id,
      typeId: node.data.typeId,
      category: node.data.typeId.split(".")[0] ?? "keras",
      depth: depth.get(node.id) ?? 0,
      parameters: node.data.shape?.params ?? 0,
      shape,
      volume: volumeOf(shape),
      onCriticalPath: critical.has(node.id),
      disabled: node.data.disabled,
    };
  });

  const byId = new Map(analysedNodes.map((node) => [node.id, node]));

  const analysedEdges: AnalysedEdge[] = edges.map((edge) => {
    const from = byId.get(edge.source);
    const to = byId.get(edge.target);
    return {
      id: edge.id,
      source: edge.source,
      target: edge.target,
      volume: from?.volume ?? 0,
      // An edge spanning more than one layer of depth jumped over something.
      isSkip: !!from && !!to && to.depth - from.depth > 1,
      onCriticalPath: critical.has(edge.source) && critical.has(edge.target),
    };
  });

  const maxDepth = Math.max(0, ...analysedNodes.map((node) => node.depth));
  const byDepth: AnalysedNode[][] = Array.from({ length: maxDepth + 1 }, () => []);
  for (const node of analysedNodes) byDepth[node.depth]?.push(node);

  return {
    nodes: analysedNodes,
    edges: analysedEdges,
    byDepth,
    totalParameters: analysedNodes.reduce((total, node) => total + node.parameters, 0),
    maxVolume: Math.max(1, ...analysedNodes.map((node) => node.volume)),
    criticalPathLength: critical.size,
    skipCount: analysedEdges.filter((edge) => edge.isSkip).length,
  };
}
