/**
 * > [!AML-DOC-FILE]
 * @file        src/store/activations.ts
 * @description Holds the real activations captured from the model, and the choice
 *              of what to feed it.
 * @module      frontend/store/activations
 * @exports     useActivations
 * @created     2026-10-01
 * @context     Capture is explicit rather than automatic: it runs a forward pass
 *              over every layer, which is far too expensive to repeat on each
 *              keystroke the way validation does [task 05]. The results are cleared
 *              when the graph changes, so what is on screen always belongs to the
 *              architecture on screen.
 */

import { create } from "zustand";

import {
  captureActivations,
  captureFromUpload,
  parseLabelFile,
  getDatasets,
  EngineError,
} from "../api/client";
import type {
  Activation,
  DatasetCatalog,
  DatasetChoice,
  Diagnostic,
  LabelFileResult,
} from "../api/types";
import { useGraph } from "./graph";

const DEFAULT_CHOICE: DatasetChoice = {
  kind: "synthetic",
  name: "mnist",
  path: "",
  samples: 512,
  seed: 0,
  pattern: "noise",
  text: "",
  validation_split: 0.2,
};

interface ActivationStore {
  catalog: DatasetCatalog | null;
  choice: DatasetChoice;
  colormap: string;
  activations: Activation[];
  byNode: Record<string, Activation>;
  inputPreview: Activation | null;
  sampleLabel: string;
  diagnostics: Diagnostic[];
  running: boolean;
  focused: string | null;
  /** Output names the user supplied, keyed by how many of them there are. */
  labels: Record<string, string[]>;
  /** What the engine made of that file, for the panel to show. */
  labelFile: LabelFileResult | null;
  /**
   * Which vocabulary the user chose for a layer, by node id. The value is the
   * vocabulary's name, or "" for "no names here"; a layer not mentioned is matched
   * automatically.
   */
  labelChoice: Record<string, string>;

  loadCatalog: () => Promise<void>;
  setChoice: (choice: Partial<DatasetChoice>) => void;
  setColormap: (colormap: string) => void;
  focus: (nodeId: string | null) => void;
  run: () => Promise<void>;
  runUpload: (file: File) => Promise<void>;
  loadLabels: (file: File) => Promise<void>;
  clearLabels: () => void;
  chooseLabels: (nodeId: string, vocabulary: string | null) => void;
  clear: () => void;
}

/**
 * > [!AML-DOC-UNIT]
 * Activation store. Holds the captured tensors, the data source, and which layer
 * the user is looking at in detail.
 */
/**
 * > [!AML-DOC-UNIT]
 * Turn the user's per-layer choices into the name lists the engine expects.
 * @param choice     node id to vocabulary name, or "" for none
 * @param labelFile  the parsed file the names come from
 * @returns node id to names, ready to send
 * @sideEffects none
 */
function resolveChoices(
  choice: Record<string, string>,
  labelFile: LabelFileResult | null,
): Record<string, string[]> {
  const out: Record<string, string[]> = {};
  for (const [nodeId, name] of Object.entries(choice)) {
    if (!name) {
      // Chosen as "none", which is a decision and not an absence: an empty list
      // stops the automatic match from putting something there anyway.
      out[nodeId] = [];
      continue;
    }
    // This vocabulary's own names, never the one that happens to share its size:
    // picking `chord2idx` over `idx2chord` has to actually pick it [E-042].
    const found = labelFile?.vocabularies.find((v) => v.name === name);
    if (found?.names.length) out[nodeId] = found.names;
  }
  return out;
}

export const useActivations = create<ActivationStore>((set, get) => ({
  catalog: null,
  choice: DEFAULT_CHOICE,
  colormap: "viridis",
  activations: [],
  byNode: {},
  inputPreview: null,
  sampleLabel: "",
  diagnostics: [],
  running: false,
  focused: null,
  labels: {},
  labelFile: null,
  labelChoice: {},

  /**
   * > [!AML-DOC-UNIT]
   * Fetch the available data sources once.
   * @sideEffects calls `/datasets`
   */
  loadCatalog: async () => {
    if (get().catalog) return;
    try {
      set({ catalog: await getDatasets() });
    } catch {
      /* the panel degrades to synthetic-only, which always works */
    }
  },

  /**
   * > [!AML-DOC-UNIT]
   * Change part of the data source selection.
   * @param choice the fields to change
   */
  setChoice: (choice) => set({ choice: { ...get().choice, ...choice } }),

  /**
   * > [!AML-DOC-UNIT]
   * Change the colour ramp. Existing results keep their old colours until the next
   * run, since the ramp is applied when the images are made, on the engine.
   * @param colormap the ramp name
   */
  setColormap: (colormap) => set({ colormap }),

  /**
   * > [!AML-DOC-UNIT]
   * Open one layer's detail, or close it.
   * @param nodeId the layer to look at, or null to close
   */
  focus: (nodeId) => set({ focused: nodeId }),

  /**
   * > [!AML-DOC-UNIT]
   * Discard captured activations.
   * @sideEffects clears everything the renderers draw
   */
  clear: () =>
    set({
      activations: [],
      byNode: {},
      inputPreview: null,
      sampleLabel: "",
      diagnostics: [],
      focused: null,
    }),

  /**
   * > [!AML-DOC-UNIT]
   * Run the current graph on a sample from the chosen source.
   * @sideEffects calls the engine, which compiles the graph and runs a forward pass
   */
  /**
   * > [!AML-DOC-UNIT]
   * Read a file of output names and keep it for every later run.
   * @param file a JSON, text or CSV file of labels
   * @sideEffects asks the engine to parse it, then re-runs if something is on screen
   * @context Held here rather than in the engine, and sent back with each capture, so
   *          the engine stays stateless [E-042].
   */
  loadLabels: async (file) => {
    try {
      const parsed = await parseLabelFile(file);
      set({ labels: parsed.labels, labelFile: parsed, labelChoice: {} });
      if (get().activations.length) await get().run();
    } catch (error) {
      set({
        diagnostics: [
          {
            severity: "error",
            code: "labels_unreadable",
            message:
              error instanceof EngineError
                ? error.message
                : "That labels file could not be read.",
            node_id: null,
            edge_id: null,
          },
        ],
      });
    }
  },

  /**
   * > [!AML-DOC-UNIT]
   * Go back to showing plain indices.
   * @sideEffects drops the labels and re-runs if something is on screen
   */
  clearLabels: () => {
    set({ labels: {}, labelFile: null, labelChoice: {} });
    if (get().activations.length) void get().run();
  },

  /**
   * > [!AML-DOC-UNIT]
   * Put a particular vocabulary on a particular layer.
   * @param nodeId     the layer
   * @param vocabulary the vocabulary's name, "" for no names, or null for automatic
   * @sideEffects re-runs if something is on screen, so the choice shows immediately
   * @context Size matches most outputs, but it cannot tell two vocabularies of the
   *          same length apart — and this file has three such pairs. That is exactly
   *          when somebody has to say which is which [E-042].
   */
  chooseLabels: (nodeId, vocabulary) => {
    const next = { ...get().labelChoice };
    if (vocabulary === null) delete next[nodeId];
    else next[nodeId] = vocabulary;
    set({ labelChoice: next });
    if (get().activations.length) void get().run();
  },

  run: async () => {
    set({ running: true, diagnostics: [] });
    try {
      const result = await captureActivations(
        useGraph.getState().toIR(),
        get().choice,
        get().colormap,
        get().labels,
        resolveChoices(get().labelChoice, get().labelFile),
      );
      set({
        activations: result.activations,
        byNode: Object.fromEntries(result.activations.map((a) => [a.node_id, a])),
        inputPreview: result.input_preview,
        sampleLabel: result.sample_label,
        diagnostics: result.diagnostics,
      });
    } catch (error) {
      set({
        diagnostics: [
          {
            severity: "error",
            code: "activations_request_failed",
            message:
              error instanceof EngineError
                ? `The engine refused the request (${error.status}).`
                : "The engine could not be reached.",
            node_id: null,
            edge_id: null,
          },
        ],
      });
    } finally {
      set({ running: false });
    }
  },

  /**
   * > [!AML-DOC-UNIT]
   * Run the current graph on a file the user chose.
   * @param file the image, audio clip, text or tensor to feed
   * @sideEffects uploads the file and runs a forward pass
   */
  runUpload: async (file) => {
    set({ running: true, diagnostics: [] });
    try {
      const result = await captureFromUpload(
        useGraph.getState().toIR(),
        file,
        get().colormap,
        get().labels,
        resolveChoices(get().labelChoice, get().labelFile),
      );
      set({
        activations: result.activations,
        byNode: Object.fromEntries(result.activations.map((a) => [a.node_id, a])),
        inputPreview: result.input_preview,
        sampleLabel: result.sample_label || file.name,
        diagnostics: result.diagnostics,
        choice: { ...get().choice, kind: "upload" },
      });
    } catch (error) {
      set({
        diagnostics: [
          {
            severity: "error",
            code: "activations_request_failed",
            message:
              error instanceof EngineError
                ? `The engine refused the upload (${error.status}).`
                : "The engine could not be reached.",
            node_id: null,
            edge_id: null,
          },
        ],
      });
    } finally {
      set({ running: false });
    }
  },
}));
