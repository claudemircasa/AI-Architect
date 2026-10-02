/**
 * > [!AML-DOC-FILE]
 * @file        src/api/client.ts
 * @description HTTP client for the backend engine.
 * @module      frontend/api/client
 * @exports     engineBaseUrl, getHealth, getCatalog, validateGraph, exportProject,
 *              EngineError
 * @created     2026-09-30
 * @context     The engine's port is fixed at 8756 in development and injected by
 *              the desktop shell at runtime [amm: E.5], because Tauri picks the
 *              port when it spawns the sidecar.
 */

import type {
  ActivationResult,
  Catalog,
  DatasetCatalog,
  DatasetChoice,
  Diagnostic,
  GraphIR,
  Health,
  LabelFileResult,
  OptimizeReport,
  TimelineResult,
  ProjectFile,
  ValidateResponse,
} from "./types";

/** Port the engine uses when started by hand, and the fallback outside the shell. */
const DEFAULT_PORT = 8756;

/** Shape of the desktop shell's `engine_status` command. */
export interface EngineStatus {
  port: number | null;
  running: boolean;
  error: string | null;
  searched: string[];
  /** True on a first launch, when Python and TensorFlow have yet to be installed. */
  needs_setup: boolean;
  /** The interpreter in use, once there is one. */
  runtime: string | null;
}

let resolvedPort: number = DEFAULT_PORT;
let shellStatus: EngineStatus | null = null;
let transport: typeof fetch | null = null;

/** How long to wait for the desktop shell's bridge before assuming a browser. */
const SHELL_BRIDGE_TIMEOUT_MS = 3000;

/**
 * > [!AML-DOC-UNIT]
 * Whether the desktop shell's bridge is present right now.
 * @returns True when Tauri has injected its bridge
 * @sideEffects none
 */
export function insideShell(): boolean {
  return typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
}

/**
 * > [!AML-DOC-UNIT]
 * Wait for the desktop shell's bridge to appear.
 * @returns True when the bridge arrived, False when this is an ordinary browser
 * @sideEffects polls for up to three seconds
 * @context The bridge is injected by the shell's own startup script, which can land
 *          after React has mounted and called this. Testing for it once, at whatever
 *          moment the first request happens, made the app decide it was in a browser
 *          and fall back to the web view's `fetch` — which the `tauri://` scheme then
 *          blocks as mixed content, so no request reached the engine and nothing was
 *          logged anywhere.
 */
async function waitForShell(): Promise<boolean> {
  if (insideShell()) return true;
  const deadline = Date.now() + SHELL_BRIDGE_TIMEOUT_MS;
  while (Date.now() < deadline) {
    await new Promise((resolve) => setTimeout(resolve, 50));
    if (insideShell()) return true;
  }
  return false;
}

/**
 * > [!AML-DOC-UNIT]
 * Record a line in the desktop shell's log, where the engine's output also goes.
 * @param level   severity label
 * @param message text to record
 * @returns resolves once the shell has been told, or immediately in a browser
 * @sideEffects none in a browser beyond a console write
 * @context The packaged app has no console the user can open, so without this a
 *          frontend failure inside it leaves no trace anywhere.
 */
export async function logToShell(level: string, message: string): Promise<void> {
  if (!insideShell()) {
    console[level === "error" ? "error" : "log"](message);
    return;
  }
  try {
    const { invoke } = await import("@tauri-apps/api/core");
    await invoke("log_message", { level, message });
  } catch {
    /* the shell is the only sink; losing a log line must never break the app */
  }
}

/**
 * > [!AML-DOC-UNIT]
 * Choose how HTTP requests are made.
 * @returns the browser's own `fetch`, or the desktop shell's when running inside it
 * @sideEffects caches the chosen transport
 * @context Inside the shell the page is served from the `tauri://` scheme, which
 *          counts as secure, and a secure page may not fetch plain `http://`. Such
 *          a request is dropped as mixed content before it leaves the web view, so
 *          nothing reaches the engine and nothing appears in its log. Tauri's HTTP
 *          plugin performs the request in Rust instead, where that rule does not
 *          apply, and its capability scope confines it to loopback addresses.
 */
async function getTransport(): Promise<typeof fetch> {
  if (transport) return transport;
  if (!(await waitForShell())) {
    transport = window.fetch.bind(window);
    return transport;
  }
  try {
    const module = await import("@tauri-apps/plugin-http");
    transport = module.fetch as unknown as typeof fetch;
    void logToShell("info", "using the shell HTTP transport");
  } catch (error) {
    void logToShell(
      "error",
      `the shell HTTP transport could not be loaded, falling back to the web view's: ${
        error instanceof Error ? error.message : String(error)
      }`,
    );
    transport = window.fetch.bind(window);
  }
  return transport;
}

/**
 * > [!AML-DOC-UNIT]
 * Ask the desktop shell which port it started the engine on, and remember it.
 * @returns the shell's engine status, or null when running in a plain browser
 * @sideEffects caches the port for every later request
 * @context The shell picks a free port rather than a fixed one, so the frontend has
 *          to be told. An earlier attempt injected it as a page variable through
 *          Tauri's `initialization_script`, which never executed; asking over the
 *          command bridge is explicit and observable instead of depending on
 *          script-injection timing [amm: E.5].
 */
export async function resolveEngine(): Promise<EngineStatus | null> {
  if (!(await waitForShell())) return null;
  try {
    const { invoke } = await import("@tauri-apps/api/core");
    const status = await invoke<EngineStatus>("engine_status");
    shellStatus = status;
    if (status.port) resolvedPort = status.port;
    return status;
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error);
    void logToShell("error", `engine_status failed: ${detail}`);
    shellStatus = {
      port: null, running: false, error: detail, searched: [],
      needs_setup: false, runtime: null,
    };
    return shellStatus;
  }
}

/**
 * > [!AML-DOC-UNIT]
 * The shell's last reported engine status, for the onboarding panel.
 * @returns the status, or null when running in a browser or before resolution
 */
export function shellEngineStatus(): EngineStatus | null {
  return shellStatus;
}

/**
 * > [!AML-DOC-UNIT]
 * Resolve the engine's base URL.
 * @returns origin such as `http://127.0.0.1:8756`
 * @sideEffects none; reads the port cached by `resolveEngine`
 */
export function engineBaseUrl(): string {
  return `http://127.0.0.1:${resolvedPort}`;
}

/** An error carrying the engine's own response, so callers can show its detail. */
export class EngineError extends Error {
  readonly status: number;
  readonly detail: unknown;

  constructor(status: number, detail: unknown, message: string) {
    super(message);
    this.name = "EngineError";
    this.status = status;
    this.detail = detail;
  }
}

/**
 * > [!AML-DOC-UNIT]
 * Send a request to the engine and decode its JSON response.
 * @param path   path beginning with a slash
 * @param init   fetch options; a body object is JSON-encoded by the caller
 * @param signal optional abort signal, used to cancel superseded validations
 * @returns the decoded response body
 * @raises EngineError when the engine answers with a non-2xx status
 * @raises TypeError when the engine is unreachable
 */
async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const send = await getTransport();
  const response = await send(`${engineBaseUrl()}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!response.ok) {
    let detail: unknown = null;
    try {
      detail = (await response.json()).detail ?? null;
    } catch {
      detail = await response.text().catch(() => null);
    }
    throw new EngineError(
      response.status,
      detail,
      `engine returned ${response.status} for ${path}`,
    );
  }
  return (await response.json()) as T;
}

/**
 * > [!AML-DOC-UNIT]
 * Probe engine liveness. Answers immediately, even while TensorFlow is loading.
 * @returns the health payload, including the TensorFlow warm-up state
 * @raises TypeError when the engine is not listening
 */
export function getHealth(): Promise<Health> {
  return request<Health>("/health");
}

/**
 * > [!AML-DOC-UNIT]
 * Fetch the full layer catalog. Blocks until TensorFlow has finished loading.
 * @returns the catalog: category tree, layer specs and enum vocabularies
 * @raises EngineError 503 when the engine failed to warm up
 */
export function getCatalog(): Promise<Catalog> {
  return request<Catalog>("/catalog");
}

/**
 * > [!AML-DOC-UNIT]
 * Validate a graph and get the real output shape of every layer.
 * @param graph  the architecture to check
 * @param signal abort signal, so a superseded edit does not overwrite a newer result
 * @returns diagnostics, per-node shapes and parameter totals
 * @raises EngineError 503 when the engine failed to warm up
 * @context A malformed graph still answers 200 with diagnostics, so this rejects
 *          only on transport or engine failure [amm: E.1].
 */
export function validateGraph(
  graph: GraphIR,
  signal?: AbortSignal,
): Promise<ValidateResponse> {
  return request<ValidateResponse>("/graph/validate", {
    method: "POST",
    body: JSON.stringify(graph),
    signal,
  });
}

/**
 * > [!AML-DOC-UNIT]
 * The data sources the editor can offer.
 * @returns the builtin datasets, synthetic patterns and colour ramps
 * @raises EngineError 503 when the engine failed to warm up
 */
export function getDatasets(): Promise<DatasetCatalog> {
  return request<DatasetCatalog>("/datasets");
}

/**
 * > [!AML-DOC-UNIT]
 * Run one sample through the model and get what every layer produced.
 * @param graph    the architecture to run
 * @param dataset  where the sample comes from
 * @param colormap ramp to render feature maps and heatmaps with
 * @returns per-layer tiles, traces and statistics
 * @raises EngineError 503 when the engine failed to warm up
 * @context A graph that will not compile answers 200 with diagnostics, since that
 *          is an ordinary state while editing.
 */
export function captureActivations(
  graph: GraphIR,
  dataset: DatasetChoice,
  colormap: string,
  labels: Record<string, string[]> = {},
  labelAssignments: Record<string, string[]> = {},
): Promise<ActivationResult> {
  return request<ActivationResult>("/graph/activations", {
    method: "POST",
    body: JSON.stringify({
      graph, dataset, colormap, labels,
      label_assignments: labelAssignments,
      node_ids: [],
    }),
  });
}

/**
 * > [!AML-DOC-UNIT]
 * Read a file of output names.
 * @param file a JSON, text or CSV file of labels
 * @returns the vocabularies it holds, keyed by how many names each has
 * @raises EngineError when nothing in the file reads as a list of names
 * @context Parsed once by the engine and then held here, travelling back with each
 *          capture, so the engine keeps no state about which model is whose [E-042].
 */
export async function parseLabelFile(file: File): Promise<LabelFileResult> {
  const body = new FormData();
  body.append("file", file, file.name);
  const send = await getTransport();
  const response = await send(`${engineBaseUrl()}/labels/parse`, { method: "POST", body });
  if (!response.ok) {
    const detail: unknown = await response.json().catch(() => null);
    const message =
      detail && typeof detail === "object" && "detail" in detail
        ? String((detail as { detail: unknown }).detail)
        : `the engine refused that labels file (${response.status})`;
    throw new EngineError(response.status, detail, message);
  }
  return (await response.json()) as LabelFileResult;
}

/**
 * > [!AML-DOC-UNIT]
 * Run an uploaded file through the model.
 * @param graph    the architecture to run
 * @param file     the image, audio clip, text or tensor to feed
 * @param colormap ramp to render with
 * @returns per-layer tiles, traces and statistics
 * @raises EngineError when the engine refuses the request
 */
export async function captureFromUpload(
  graph: GraphIR,
  file: File,
  colormap: string,
  labels: Record<string, string[]> = {},
  labelAssignments: Record<string, string[]> = {},
): Promise<ActivationResult> {
  const body = new FormData();
  body.append("graph", JSON.stringify(graph));
  body.append("colormap", colormap);
  body.append("labels", JSON.stringify(labels));
  body.append("label_assignments", JSON.stringify(labelAssignments));
  body.append("file", file, file.name);
  const send = await getTransport();
  const response = await send(`${engineBaseUrl()}/graph/activations/upload`, {
    method: "POST",
    body,
  });
  if (!response.ok) {
    throw new EngineError(response.status, null, `engine returned ${response.status}`);
  }
  return (await response.json()) as ActivationResult;
}

/**
 * > [!AML-DOC-UNIT]
 * Serialise the current project into a `.nnarch` archive.
 * @param graph    the architecture to store
 * @param viewport canvas pan and zoom
 * @param meta     file metadata; the engine fills in the timestamps
 * @returns the archive bytes and the filename the engine suggested
 * @raises EngineError when the engine refuses the request
 */
export async function saveProject(
  graph: GraphIR,
  viewport: { x: number; y: number; zoom: number },
  meta: Record<string, unknown> = {},
): Promise<{ bytes: Uint8Array; filename: string }> {
  const send = await getTransport();
  const response = await send(`${engineBaseUrl()}/project/save`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ graph, viewport, meta }),
  });
  if (!response.ok) {
    throw new EngineError(response.status, null, "the project could not be saved");
  }
  const disposition = response.headers.get("Content-Disposition") ?? "";
  const match = /filename="(.+)"/.exec(disposition);
  return {
    bytes: new Uint8Array(await response.arrayBuffer()),
    filename: match?.[1] ?? "project.nnarch",
  };
}

/**
 * > [!AML-DOC-UNIT]
 * Read a `.nnarch` file back into a project.
 * @param file the archive the user chose
 * @returns {project, diagnostics}; `project` is null when it could not be read
 * @raises EngineError only on transport failure; an unreadable file comes back as
 *         diagnostics, because that is a thing the user did rather than a fault
 */
export async function openProject(
  file: File,
): Promise<{ project: ProjectFile | null; diagnostics: Diagnostic[] }> {
  return uploadToEngine("/project/open", file);
}

/**
 * > [!AML-DOC-UNIT]
 * Turn a saved model into an editable graph.
 * @param file a `.keras` archive, a `.tflite` model, or `Model.to_json()` output
 * @returns {graph, diagnostics}; `graph` is null only when the file is unreadable
 */
export async function importKerasModel(
  file: File,
): Promise<{ graph: GraphIR | null; diagnostics: Diagnostic[] }> {
  return uploadToEngine("/project/import", file);
}

/**
 * > [!AML-DOC-UNIT]
 * Post one file to an engine route as multipart form data.
 * @param path the engine route
 * @param file the file to send
 * @returns the decoded JSON response
 * @raises EngineError when the engine answers with a non-2xx status
 */
async function uploadToEngine<T>(path: string, file: File): Promise<T> {
  const body = new FormData();
  body.append("file", file, file.name);
  const send = await getTransport();
  const response = await send(`${engineBaseUrl()}${path}`, { method: "POST", body });
  if (!response.ok) {
    throw new EngineError(response.status, null, `engine returned ${response.status}`);
  }
  return (await response.json()) as T;
}

/**
 * > [!AML-DOC-UNIT]
 * Generate a standalone project and return the zip archive.
 * @param graph   the architecture to export
 * @param options generation options, matching `nnarch.export.ExportOptions`
 * @returns the archive bytes and the filename the engine suggested
 * @raises EngineError 422 carrying diagnostics when the graph cannot be exported
 */
export async function exportProject(
  graph: GraphIR,
  options: Record<string, unknown> = {},
): Promise<{ bytes: Uint8Array; filename: string }> {
  const send = await getTransport();
  const response = await send(`${engineBaseUrl()}/export`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ graph, options }),
  });
  if (!response.ok) {
    const detail = await response.json().catch(() => null);
    throw new EngineError(response.status, detail?.detail ?? null, "export failed");
  }
  const disposition = response.headers.get("Content-Disposition") ?? "";
  const match = /filename="(.+)"/.exec(disposition);
  return {
    bytes: new Uint8Array(await response.arrayBuffer()),
    filename: match?.[1] ?? "model.zip",
  };
}

/**
 * > [!AML-DOC-UNIT]
 * Run the model repeatedly over a moving window.
 * @param graph    the architecture
 * @param options  how many steps, what to watch, the names, and the recording
 * @returns what each step predicted, the watched layers, and what the loops did
 * @raises EngineError when the engine refuses the request
 * @context Multipart because the recording goes with it: a run over time is a walk
 *          through one file, and splitting the file from the walk would mean holding
 *          it somewhere between the two [E-047].
 */
export async function runTimeline(
  graph: GraphIR,
  options: {
    steps: number;
    watch?: string[];
    labels?: Record<string, string[]>;
    labelAssignments?: Record<string, string[]>;
    feedback?: { source: string; target: string }[] | null;
    ablate?: boolean;
    file?: File | null;
  },
): Promise<TimelineResult> {
  const body = new FormData();
  body.append("graph", JSON.stringify(graph));
  body.append("steps", String(options.steps));
  body.append("watch", JSON.stringify(options.watch ?? []));
  body.append("labels", JSON.stringify(options.labels ?? {}));
  body.append("label_assignments", JSON.stringify(options.labelAssignments ?? {}));
  body.append("ablate", options.ablate === false ? "false" : "true");
  if (options.feedback) body.append("feedback", JSON.stringify(options.feedback));
  if (options.file) body.append("file", options.file, options.file.name);

  const send = await getTransport();
  const response = await send(`${engineBaseUrl()}/graph/timeline`, { method: "POST", body });
  if (!response.ok) {
    const detail: unknown = await response.json().catch(() => null);
    throw new EngineError(response.status, detail, `the engine refused the run (${response.status})`);
  }
  return (await response.json()) as TimelineResult;
}

/**
 * > [!AML-DOC-UNIT]
 * Ask what could be simplified or swapped in an architecture.
 * @param graph   the architecture
 * @param measure whether to time the models; off reports sizes only and is far
 *                faster on a large graph
 * @returns every proposal, each measured against the model as it stands
 * @raises EngineError when the engine refuses the request
 */
export function optimizeGraph(graph: GraphIR, measure = true): Promise<OptimizeReport> {
  return request<OptimizeReport>("/graph/optimize", {
    method: "POST",
    body: JSON.stringify({ graph, measure_latency: measure }),
  });
}

/**
 * > [!AML-DOC-UNIT]
 * Carry out the chosen proposals.
 * @param graph the architecture
 * @param ids   which proposals, in order
 * @returns the rewritten graph and anything worth saying
 * @raises EngineError when the engine refuses the request
 * @context The engine rewrites and hands the result back rather than the editor doing
 *          it, so the rules live in one place and the two cannot drift [amm: E.4].
 */
export function applyProposals(
  graph: GraphIR,
  ids: string[],
): Promise<{ graph: GraphIR; diagnostics: Diagnostic[] }> {
  return request<{ graph: GraphIR; diagnostics: Diagnostic[] }>("/graph/optimize/apply", {
    method: "POST",
    body: JSON.stringify({ graph, proposal_ids: ids }),
  });
}
