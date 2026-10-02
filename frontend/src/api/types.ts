/**
 * > [!AML-DOC-FILE]
 * @file        src/api/types.ts
 * @description TypeScript mirror of the backend's catalog and graph schemas.
 * @module      frontend/api/types
 * @exports     ParamType, ParamSpec, PortSpec, LayerSpec, Catalog, Diagnostic,
 *              GraphNode, GraphEdge, GraphIR, ShapeInfo, ValidateResponse
 * @created     2026-09-30
 * @context     Hand-written counterpart to `nnarch.catalog.spec` and
 *              `nnarch.ir.schema` [amm: E.1, E.2]. These names are the wire
 *              contract, so a rename here must be matched on the backend.
 */

/** Widget class of a layer parameter; mirrors `nnarch.catalog.spec.ParamType`. */
export type ParamType =
  | "int"
  | "float"
  | "bool"
  | "str"
  | "enum"
  | "int_tuple"
  | "float_tuple"
  | "shape"
  | "activation"
  | "initializer"
  | "regularizer"
  | "constraint"
  | "dtype"
  | "layer_ref"
  | "text"
  | "any";

/** How a layer receives its input tensors; mirrors `nnarch.catalog.spec.CallStyle`. */
export type CallStyle = "source" | "single" | "list" | "query_value_key";

/** One editable constructor argument of a layer. */
export interface ParamSpec {
  name: string;
  type: ParamType;
  default: unknown;
  choices: unknown[] | null;
  minimum: number | null;
  maximum: number | null;
  arity: number | null;
  help: string;
  required: boolean;
  advanced: boolean;
  group: string | null;
}

/** A named tensor slot on a layer. */
export interface PortSpec {
  name: string;
  label: string;
  rank: number | null;
  optional: boolean;
}

/** Complete description of a placeable layer. */
export interface LayerSpec {
  id: string;
  label: string;
  category: string;
  keras_path: string;
  params: ParamSpec[];
  inputs: PortSpec[];
  outputs: PortSpec[];
  min_inputs: number;
  max_inputs: number | null;
  rank_in: number | null;
  rank_out: number | null;
  modalities: string[];
  paper: string | null;
  doc_url: string | null;
  description: string;
  tags: string[];
  call_style: CallStyle;
  is_research: boolean;
}

/** One palette section. */
export interface CatalogCategory {
  id: string;
  label: string;
  layers: LayerSpec[];
}

/** Vocabularies backing the searchable select controls. */
export interface Vocabularies {
  activations: string[];
  initializers: (string | null)[];
  regularizers: (string | null)[];
  constraints: (string | null)[];
  dtypes: string[];
}

/** Response of `GET /catalog`. */
export interface Catalog {
  count: number;
  categories: CatalogCategory[];
  vocabularies: Vocabularies;
}

/** A validation result attached to the graph, a node or an edge. */
export interface Diagnostic {
  severity: "error" | "warning" | "info";
  code: string;
  message: string;
  node_id: string | null;
  edge_id: string | null;
}

/** Canvas coordinates of a node. */
export interface Position {
  x: number;
  y: number;
}

/** One placed layer, as persisted. */
export interface GraphNode {
  id: string;
  type: string;
  name: string;
  params: Record<string, unknown>;
  position: Position;
  notes: string | null;
  disabled: boolean;
}

/** A tensor flowing between two ports. */
export interface GraphEdge {
  id: string;
  source: string;
  source_port: string;
  target: string;
  target_port: string;
  order: number;
}

/** A complete architecture; mirrors `nnarch.ir.schema.GraphIR`. */
export interface GraphIR {
  ir_version: number;
  name: string;
  nodes: GraphNode[];
  edges: GraphEdge[];
  inputs: string[];
  outputs: string[];
  meta: Record<string, unknown>;
}

/** Shape and dtype Keras reported for a node's output. */
export interface ShapeInfo {
  shape: (number | null)[];
  dtype: string;
  params: number;
}

/** Response of `POST /graph/validate`. */
export interface ValidateResponse {
  compiled: boolean;
  diagnostics: Diagnostic[];
  shapes: Record<string, ShapeInfo>;
  params_total: number;
  trainable_params: number;
}

/** Response of `GET /health`. */
export interface Health {
  status: string;
  engine: string;
  ir_version: number;
  python: string;
  warm: "cold" | "warming" | "warm" | "failed";
  error: string | null;
  versions: Record<string, string | number>;
}


/** Canvas pan and zoom, stored with a project so reopening restores the view. */
export interface Viewport {
  x: number;
  y: number;
  zoom: number;
}

/** Facts about the saved file rather than the model; mirrors `ProjectMeta`. */
export interface ProjectMeta {
  name: string;
  created: string;
  modified: string;
  tool_version: string;
  ir_version: number;
}

/** A loaded `.nnarch` project; mirrors `nnarch.ir.project.Project`. */
export interface ProjectFile {
  graph: GraphIR;
  viewport: Viewport;
  meta: ProjectMeta;
  runs: Record<string, unknown>[];
}


/** How the editor should draw one layer's output; mirrors `nnarch.viz.RenderHint`. */
export type RenderHint =
  | "feature_maps"
  | "sequence_heatmap"
  | "attention_matrix"
  | "vector_bars"
  | "probabilities"
  | "waveform"
  | "spectrogram"
  | "volume_slices"
  | "scalar";

/** Summary statistics of one activation. */
export interface ActivationStats {
  min: number;
  max: number;
  mean: number;
  std: number;
  sparsity: number;
}

/** One layer's output, ready to draw; mirrors `nnarch.viz.Activation`. */
export interface Activation {
  node_id: string;
  label: string;
  shape: (number | null)[];
  dtype: string;
  hint: RenderHint;
  stats: ActivationStats;
  tiles: string[];
  series: number[];
  labels: string[];
  heatmap: string | null;
  truncated: boolean;
  channel_count: number;
}

/** Response of `POST /graph/activations`. */
export interface ActivationResult {
  activations: Activation[];
  input_preview: Activation | null;
  diagnostics: Diagnostic[];
  sample_label: string;
}

/** Where a sample comes from; mirrors `nnarch.data.DatasetChoice`. */
export interface DatasetChoice {
  kind: "synthetic" | "builtin" | "upload" | "folder";
  name: string;
  path: string;
  samples: number;
  seed: number;
  pattern: "noise" | "ones" | "zeros" | "ramp" | "checkerboard" | "sine";
  text: string;
  validation_split: number;
}

/** One dataset Keras can download. */
export interface BuiltinDataset {
  name: string;
  label: string;
  classes: number;
  modality: string;
  shape: number[] | null;
  class_names: string[];
}

/** Response of `GET /datasets`. */
export interface DatasetCatalog {
  builtin: BuiltinDataset[];
  patterns: string[];
  colormaps: string[];
}

/** One vocabulary read out of a labels file. */
export interface LabelVocabulary {
  /** The key it was stored under, or the file's name for a bare list. */
  name: string;
  /** How many names it holds, which is what matches it to a layer. */
  count: number;
  /** True when it was stored name-to-index and had to be turned round. */
  inverted: boolean;
  /** Its own names, so choosing between two of the same length means something. */
  names: string[];
}

/** What the engine made of a labels file. */
export interface LabelFileResult {
  filename: string;
  /** One sentence naming each vocabulary and its size. */
  note: string;
  vocabularies: LabelVocabulary[];
  /** The names, keyed by how many there are; JSON has no integer keys. */
  labels: Record<string, string[]>;
}

/** One output's leading class at one step of a run over time. */
export interface Prediction {
  name: string;
  value: number;
}

/** One tick of a run over time. */
export interface TimelineStep {
  step: number;
  predictions: Record<string, Prediction[]>;
  note: string | null;
}

/** One layer's activations across every step. */
export interface TimelineTrace {
  node_id: string;
  label: string;
  /** Step by channel, downsampled on the channel axis. */
  values: number[][];
  channels: number;
}

/** An output wired back into an input between steps. */
export interface FeedbackPair {
  source: string;
  target: string;
}

/** Everything a run over time produced. */
export interface TimelineResult {
  steps: TimelineStep[];
  traces: TimelineTrace[];
  feedback: FeedbackPair[];
  /** What cutting the loops changed, when it was measured. */
  ablation: string | null;
  diagnostics: Diagnostic[];
}

/** What a model costs, as measured rather than estimated. */
export interface Measurement {
  parameters: number;
  trainable: number;
  layers: number;
  latency_ms: number | null;
  latency_spread: number | null;
}

/** One change the optimiser can make. */
export interface Proposal {
  id: string;
  rule: string;
  /** `exact` leaves the same function; `substitution` leaves a different model. */
  kind: "exact" | "substitution";
  title: string;
  detail: string;
  node_ids: string[];
}

/** A proposal and what it would actually do. */
export interface ProposalOutcome {
  proposal: Proposal;
  after: Measurement | null;
  /** What the timings support saying, which is often that they support nothing. */
  speed: string | null;
  error: string | null;
}

/** Everything the optimiser found. */
export interface OptimizeReport {
  before: Measurement;
  outcomes: ProposalOutcome[];
  combined: Measurement | null;
  diagnostics: Diagnostic[];
}
