/**
 * > [!AML-DOC-FILE]
 * @file        src/editor/grouping.ts
 * @description Folds long straight runs of layers into single collapsible nodes, so
 *              a graph with thousands of them can be read and drawn.
 * @module      frontend/editor/grouping
 * @exports     LayerGroup, GROUPING_THRESHOLD, MIN_CHAIN, findChains, projectGraph
 * @created     2026-10-01
 * @context     Grouping is presentation only. The store keeps every node, and
 *              `toIR()` still sends the whole graph, so validation, export and saving
 *              are unaffected by what is folded on screen. Collapsing the data would
 *              change the model; collapsing the drawing changes only what you look at.
 *
 *              The case this exists for: a converted `.tflite` recurrent model
 *              arrives as thousands of gate operations — the same cell unrolled once
 *              per timestep. Drawing each one is both unusably slow and a worse
 *              picture of the architecture than one box saying how many there are.
 */

import type { LayerEdge, LayerNode } from "../store/graph";

/** Above this many nodes, groups start collapsed. */
export const GROUPING_THRESHOLD = 240;

/** Shortest run worth folding. Below this, folding hides more than it saves. */
export const MIN_CHAIN = 6;

/** Declared box size, so React Flow need not measure to know what is on screen. */
const NODE_WIDTH = 176;
const NODE_HEIGHT = 88;
const GROUP_HEIGHT = 80;

/** Boxes the first view aims for, whatever the model's size. */
export const TARGET_BOXES = 48;

/** Smallest cap, so a moderately large graph is not folded coarser than it needs. */
export const MIN_REGION = 12;

/**
 * > [!AML-DOC-UNIT]
 * How many layers to fold into one box, for a graph of this size.
 * @param nodeCount how many layers the graph holds
 * @returns the largest run to fold
 * @sideEffects none
 * @context Scaled rather than fixed, so the first view is always about the same
 *          number of boxes whether the model has three hundred layers or three
 *          thousand. A fixed cap gave 191 boxes for a converted recurrent model,
 *          which is better than 3,212 but still more than anyone scans — and still
 *          slow, because at the zoom that fits them they are all on screen at once.
 *          Without any cap the whole model folds into a single box, which hides
 *          everything.
 */
export function regionSize(nodeCount: number): number {
  return Math.max(MIN_REGION, Math.ceil(nodeCount / TARGET_BOXES));
}

/** A straight run of layers that can be shown as one node. */
export interface LayerGroup {
  id: string;
  members: string[];
  /** What the run is made of, e.g. "Multiply ×4 · Add ×3". */
  summary: string;
}

/**
 * > [!AML-DOC-UNIT]
 * Find the straight runs in a graph.
 * @param nodes     the canvas nodes
 * @param edges     the canvas edges
 * @param labelOf   reads a node's display type, used to summarise a run
 * @param minLength shortest run to return
 * @returns the runs, longest first
 * @sideEffects none
 * @context Folds *regions*, not straight lines. A region is a stretch of the graph
 *          that nothing enters except at its first node and nothing leaves except at
 *          its last — so folding it cannot hide a connection to the rest of the
 *          model, even though the region may branch and rejoin inside.
 *
 *          Straight lines were the first attempt and were nearly useless here: a
 *          converted recurrent cell is full of two-input gates, so almost no node has
 *          a single predecessor, and folding caught 50 nodes out of 3,212. Regions
 *          catch the cell itself, which is what repeats.
 *
 *          Boundaries are chosen rather than searched for: the number of edges
 *          spanning each position is computed in two linear passes, and each box
 *          ends at the narrowest point within reach. Searching window by window was
 *          the first version and was quadratic — it turned a 5-second pan into an
 *          84-second one. Requiring a *perfect* boundary was the second, and folded
 *          the entire model into one box, since a model with one input and one
 *          output has no interior point where nothing crosses.
 */
export function findChains(
  nodes: LayerNode[],
  edges: LayerEdge[],
  labelOf: (node: LayerNode) => string,
  minLength: number = MIN_CHAIN,
): LayerGroup[] {
  const order = topologicalOrder(nodes, edges);
  const count = order.length;
  const position = new Map(order.map((id, index) => [id, index]));
  const byId = new Map(nodes.map((node) => [node.id, node]));

  // How many edges span each boundary. A difference array turns this into two
  // linear passes instead of counting edges at every position.
  const delta = new Array<number>(count + 2).fill(0);
  for (const edge of edges) {
    const from = position.get(edge.source);
    const to = position.get(edge.target);
    if (from === undefined || to === undefined || to <= from) continue;
    delta[from + 1] = (delta[from + 1] ?? 0) + 1;
    delta[to + 1] = (delta[to + 1] ?? 0) - 1;
  }
  const crossing = new Array<number>(count + 1).fill(0);
  let running = 0;
  for (let index = 0; index <= count; index += 1) {
    running += delta[index] ?? 0;
    crossing[index] = running;
  }

  // Cut where the graph is narrowest: the fewest connections span the boundary, so
  // the box that results is the most self-contained one available there.
  const maxRegion = regionSize(count);
  const groups: LayerGroup[] = [];
  let start = 0;
  while (start < count) {
    const limit = Math.min(start + maxRegion, count);
    if (limit - start < minLength) break;

    // Search for the seam only in the back half of the allowed span, so boxes stay
    // near the target size. Searching the whole span finds an early narrow point
    // almost immediately and produces three times as many boxes as intended —
    // which is slower, because at the zoom that fits them they are all on screen.
    const earliest = Math.max(start + minLength, limit - Math.ceil(maxRegion * 0.45));
    let cut = limit;
    let narrowest = Infinity;
    for (let index = earliest; index <= limit; index += 1) {
      const width = crossing[index] ?? Infinity;
      if (width < narrowest) {
        narrowest = width;
        cut = index;
      }
    }

    const members = order.slice(start, cut) as string[];
    if (members.length >= minLength) {
      const counts = new Map<string, number>();
      for (const id of members) {
        const member = byId.get(id);
        if (!member) continue;
        const label = labelOf(member);
        counts.set(label, (counts.get(label) ?? 0) + 1);
      }
      groups.push({
        id: `group:${members[0]}`,
        members,
        summary: [...counts.entries()]
          .sort((a, b) => b[1] - a[1])
          .slice(0, 3)
          .map(([label, n]) => (n > 1 ? `${label} ×${n}` : label))
          .join(" · "),
      });
    }
    start = cut;
  }

  return groups;
}

/**
 * > [!AML-DOC-UNIT]
 * Order the nodes so each follows everything that feeds it.
 * @param nodes the canvas nodes
 * @param edges the canvas edges
 * @returns node ids in topological order, with any cycle appended at the end
 * @sideEffects none
 */
function topologicalOrder(nodes: LayerNode[], edges: LayerEdge[]): string[] {
  const indegree = new Map(nodes.map((node) => [node.id, 0]));
  const successors = new Map<string, string[]>(nodes.map((node) => [node.id, []]));
  for (const edge of edges) {
    if (!indegree.has(edge.target) || !successors.has(edge.source)) continue;
    indegree.set(edge.target, (indegree.get(edge.target) ?? 0) + 1);
    successors.get(edge.source)?.push(edge.target);
  }
  const ready = nodes.filter((node) => indegree.get(node.id) === 0).map((n) => n.id);
  const order: string[] = [];
  while (ready.length) {
    const id = ready.shift() as string;
    order.push(id);
    for (const next of successors.get(id) ?? []) {
      const remaining = (indegree.get(next) ?? 1) - 1;
      indegree.set(next, remaining);
      if (remaining === 0) ready.push(next);
    }
  }
  const placed = new Set(order);
  for (const node of nodes) if (!placed.has(node.id)) order.push(node.id);
  return order;
}

/**
 * > [!AML-DOC-UNIT]
 * Build the node and edge lists the canvas should draw.
 * @param nodes    every node in the graph
 * @param edges    every edge in the graph
 * @param groups   the runs that may be folded
 * @param expanded ids of groups the user has opened
 * @returns the projected nodes and edges
 * @sideEffects none
 * @context A folded run becomes one node sitting where its first member was, and the
 *          edges into and out of the run are redirected to it. Edges *inside* a
 *          folded run are dropped, since both their ends are now the same box.
 */
export function projectGraph(
  nodes: LayerNode[],
  edges: LayerEdge[],
  groups: LayerGroup[],
  expanded: Set<string>,
): { nodes: LayerNode[]; edges: LayerEdge[] } {
  const folded = groups.filter((group) => !expanded.has(group.id));
  if (folded.length === 0) return { nodes, edges };

  const byId = new Map(nodes.map((node) => [node.id, node]));
  /** Maps every hidden member to the group node that now stands for it. */
  const representative = new Map<string, string>();
  for (const group of folded) {
    for (const member of group.members) representative.set(member, group.id);
  }

  const projected: LayerNode[] = [];
  const emitted = new Set<string>();

  for (const node of nodes) {
    const groupId = representative.get(node.id);
    if (!groupId) {
      projected.push(node);
      continue;
    }
    if (emitted.has(groupId)) continue;
    emitted.add(groupId);

    const group = folded.find((candidate) => candidate.id === groupId) as LayerGroup;
    const first = byId.get(group.members[0] as string);
    const last = byId.get(group.members[group.members.length - 1] as string);
    projected.push({
      id: group.id,
      type: "group",
      position: first?.position ?? { x: 0, y: 0 },
      selected: false,
      data: {
        typeId: "nnarch.group",
        name: `${group.members.length} layers`,
        params: {},
        disabled: false,
        // The run's output is the last member's output, which is what leaves the box.
        shape: last?.data.shape ?? null,
        diagnostics: group.members.flatMap(
          (member) => byId.get(member)?.data.diagnostics ?? [],
        ),
        groupSummary: group.summary,
        groupSize: group.members.length,
      },
    } as LayerNode);
  }

  // Folded boxes are laid out afresh rather than left where their first member sat.
  // An imported model spreads its layers over hundreds of thousands of pixels, so
  // inheriting those positions leaves the folded view just as sprawling — and
  // measurably slower than not folding at all, because more boxes then fall inside
  // the viewport at the zoom that fits them.
  const spacing = 104;
  const compact = projected.map((node, index) => ({
    ...node,
    position: { x: node.position.x, y: index * spacing },
    // Declared rather than measured. React Flow hides a node it has not measured
    // when only the visible ones are rendered, and a hidden node is never measured —
    // so the whole canvas comes up blank. Stating the size breaks that circle.
    width: NODE_WIDTH,
    height: node.type === "group" ? GROUP_HEIGHT : NODE_HEIGHT,
  }));

  const projectedEdges: LayerEdge[] = [];
  const seen = new Set<string>();
  for (const edge of edges) {
    const source = representative.get(edge.source) ?? edge.source;
    const target = representative.get(edge.target) ?? edge.target;
    if (source === target) continue;
    const key = `${source}->${target}`;
    if (seen.has(key)) continue;
    seen.add(key);
    projectedEdges.push({
      ...edge,
      id: `p:${key}`,
      source,
      target,
      sourceHandle: representative.has(edge.source) ? "output" : edge.sourceHandle,
      targetHandle: representative.has(edge.target) ? "input" : edge.targetHandle,
    });
  }

  return { nodes: compact, edges: projectedEdges };
}
