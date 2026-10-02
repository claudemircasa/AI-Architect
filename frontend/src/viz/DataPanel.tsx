/**
 * > [!AML-DOC-FILE]
 * @file        src/viz/DataPanel.tsx
 * @description The data pane: choose what to feed the model, run it, and read what
 *              every layer produced.
 * @module      frontend/viz/DataPanel
 * @exports     DataPanel
 * @created     2026-10-01
 * @context     The feature the tool exists for [task 11]. Everything shown here is
 *              a real forward pass through the compiled model, so a layer that looks
 *              dead on screen is dead in the model.
 */

import { useEffect, useMemo, useRef } from "react";

import { useActivations } from "../store/activations";
import { useGraph } from "../store/graph";
import { ActivationView } from "./ActivationView";

const CONTROL =
  "w-full rounded bg-surface-2 border border-line px-2 py-1 text-[12px] text-ink-0 " +
  "outline-none focus:border-accent focus:ring-1 focus:ring-accent/40";

/**
 * > [!AML-DOC-UNIT]
 * The data pane.
 * @returns the source picker, the run control, and one card per layer
 * @sideEffects loads the dataset catalog once, and runs forward passes on demand
 */
export function DataPanel() {
  const {
    catalog,
    choice,
    colormap,
    activations,
    inputPreview,
    sampleLabel,
    diagnostics,
    running,
    focused,
    loadCatalog,
    setChoice,
    setColormap,
    focus,
    run,
    runUpload,
    labelFile,
    loadLabels,
    clearLabels,
    labelChoice,
    chooseLabels,
  } = useActivations();
  const compiled = useGraph((state) => state.compiled);
  // Selected as the store holds them, then derived here: a selector that builds a
  // new array or object every call never compares equal to the last one, so the
  // component re-renders forever [E-039].
  const graphDiagnostics = useGraph((state) => state.diagnostics);
  const nodes = useGraph((state) => state.nodes);
  const setSelection = useGraph((state) => state.setSelection);

  const blocking = useMemo(
    () => graphDiagnostics.filter((diagnostic) => diagnostic.severity === "error"),
    [graphDiagnostics],
  );
  const nodeNames = useMemo(
    () => new Map(nodes.map((node) => [node.id, node.data.name])),
    [nodes],
  );
  const upload = useRef<HTMLInputElement>(null);
  const labels = useRef<HTMLInputElement>(null);

  // The layers nothing reads from: a model's outputs, which is what a mapping is for.
  // Taken from the captured activations so the width is the one actually produced
  // rather than one inferred from the graph.
  const edges = useGraph((state) => state.edges);
  const outputs = useMemo(() => {
    const consumed = new Set(edges.map((edge) => edge.source));
    return activations
      .filter((activation) => !consumed.has(activation.node_id))
      .map((activation) => ({
        nodeId: activation.node_id,
        label: activation.label,
        width: Number(activation.shape[activation.shape.length - 1] ?? 0),
      }))
      .filter((output) => output.width > 1);
  }, [activations, edges]);

  useEffect(() => {
    void loadCatalog();
  }, [loadCatalog]);

  const errors = diagnostics.filter((d) => d.severity === "error");

  return (
    <div className="flex h-full flex-col">
      <div className="space-y-2 border-b border-line p-3">
        <label className="block">
          <span className="mb-1 block text-[10px] uppercase tracking-wide text-ink-2">
            Input
          </span>
          <select
            className={CONTROL}
            value={choice.kind}
            onChange={(event) =>
              setChoice({ kind: event.target.value as typeof choice.kind })
            }
          >
            <option value="synthetic">Generated tensor</option>
            <option value="builtin">Built-in dataset</option>
            <option value="upload">A file I choose</option>
            <option value="folder">A folder on this machine</option>
          </select>
        </label>

        {choice.kind === "synthetic" && (
          <label className="block">
            <span className="mb-1 block text-[10px] uppercase tracking-wide text-ink-2">
              Pattern
            </span>
            <select
              className={CONTROL}
              value={choice.pattern}
              onChange={(event) =>
                setChoice({ pattern: event.target.value as typeof choice.pattern })
              }
            >
              {(catalog?.patterns ?? ["noise"]).map((pattern) => (
                <option key={pattern} value={pattern}>
                  {pattern}
                </option>
              ))}
            </select>
            <p className="mt-1 text-[10px] text-ink-2">
              Shaped to this model's input. Needs no download, so it always works.
            </p>
          </label>
        )}

        {choice.kind === "builtin" && (
          <label className="block">
            <span className="mb-1 block text-[10px] uppercase tracking-wide text-ink-2">
              Dataset
            </span>
            <select
              className={CONTROL}
              value={choice.name}
              onChange={(event) => setChoice({ name: event.target.value })}
            >
              {(catalog?.builtin ?? []).map((dataset) => (
                <option key={dataset.name} value={dataset.name}>
                  {dataset.label}
                </option>
              ))}
            </select>
            <p className="mt-1 text-[10px] text-ink-2">
              Downloaded on first use and cached.
            </p>
          </label>
        )}

        {choice.kind === "folder" && (
          <label className="block">
            <span className="mb-1 block text-[10px] uppercase tracking-wide text-ink-2">
              Folder
            </span>
            <input
              className={CONTROL}
              placeholder="/path/to/images"
              value={choice.path}
              onChange={(event) => setChoice({ path: event.target.value })}
            />
            <p className="mt-1 text-[10px] text-ink-2">
              One subfolder per class, each holding that class's files.
            </p>
          </label>
        )}

        <label className="block">
          <span className="mb-1 block text-[10px] uppercase tracking-wide text-ink-2">
            Colours
          </span>
          <select
            className={CONTROL}
            value={colormap}
            onChange={(event) => setColormap(event.target.value)}
          >
            {(catalog?.colormaps ?? ["viridis"]).map((name) => (
              <option key={name} value={name}>
                {name}
              </option>
            ))}
          </select>
        </label>

        <div className="flex gap-2 pt-1">
          <button
            type="button"
            disabled={running || !compiled}
            onClick={() => {
              if (choice.kind === "upload") upload.current?.click();
              else void run();
            }}
            className="flex-1 rounded border border-line bg-surface-2 px-2 py-1.5
                       text-[12px] text-ink-0 hover:border-accent
                       disabled:opacity-40 disabled:hover:border-line"
            title={
              compiled
                ? "Run one sample through the model"
                : "The architecture has to be valid first"
            }
          >
            {running ? "Running…" : "Run data through"}
          </button>
        </div>
        <input
          ref={upload}
          type="file"
          className="hidden"
          accept="image/*,audio/*,.txt,.csv,.npy"
          onChange={(event) => {
            const file = event.target.files?.[0];
            event.target.value = "";
            if (file) void runUpload(file);
          }}
        />

        {/*
          A model's outputs are indices, and an index is not an answer: class 47
          means nothing without the list that says what 47 is. The list comes from
          whatever file the training pipeline built, and each vocabulary in it is
          matched to the output with that many classes [E-042].
        */}
        <div className="mt-1 border-t border-line pt-2">
          <div className="flex items-center justify-between gap-2">
            <span className="text-[10px] uppercase tracking-wide text-ink-2">
              Output names
            </span>
            <span className="flex items-center gap-1">
              {labelFile && (
                <button
                  type="button"
                  onClick={clearLabels}
                  className="text-[10px] text-ink-2 hover:text-ink-0"
                  title="Go back to showing plain indices"
                >
                  clear
                </button>
              )}
              <button
                type="button"
                onClick={() => labels.current?.click()}
                className="rounded border border-line bg-surface-2 px-2 py-0.5
                           text-[10px] text-ink-1 hover:text-ink-0"
              >
                {labelFile ? "Replace…" : "Load file…"}
              </button>
            </span>
          </div>

          {labelFile ? (
            <div className="mt-1 space-y-1.5">
              <p className="text-[10px] leading-snug text-ink-2">
                {labelFile.vocabularies.length} vocabular
                {labelFile.vocabularies.length === 1 ? "y" : "ies"} in{" "}
                <span className="text-ink-1">{labelFile.filename}</span>, matched to
                each output by how many classes it has.
              </p>

              {/*
                Matching by size carries most outputs, and cannot separate two
                vocabularies of the same length — this file has three such pairs. That
                is exactly where somebody has to say which is which, so every output is
                listed with what it was given and a way to change it [E-042].
              */}
              {outputs.length > 0 && (
                <ul className="space-y-1">
                  {outputs.map((output) => {
                    const chosen = labelChoice[output.nodeId];
                    const fits = labelFile.vocabularies.filter(
                      (vocabulary) => vocabulary.count === output.width,
                    );
                    return (
                      <li key={output.nodeId} className="min-w-0">
                        <div className="flex items-baseline justify-between gap-2">
                          <span className="min-w-0 truncate text-[10px] text-ink-1">
                            {output.label}
                          </span>
                          <span className="shrink-0 font-mono text-[9px] text-ink-2">
                            {output.width}
                          </span>
                        </div>
                        <select
                          className="mt-0.5 w-full rounded border border-line bg-surface-2
                                     px-1 py-0.5 text-[10px] text-ink-1"
                          value={chosen === undefined ? "__auto__" : chosen || "__none__"}
                          onChange={(event) => {
                            const value = event.target.value;
                            chooseLabels(
                              output.nodeId,
                              value === "__auto__" ? null : value === "__none__" ? "" : value,
                            );
                          }}
                        >
                          <option value="__auto__">
                            {fits.length
                              ? `automatic — ${fits[0]?.name}`
                              : "automatic — nothing fits"}
                          </option>
                          <option value="__none__">no names</option>
                          {labelFile.vocabularies.map((vocabulary) => (
                            <option key={vocabulary.name} value={vocabulary.name}>
                              {vocabulary.name} ({vocabulary.count})
                              {vocabulary.count === output.width ? "" : " — wrong size"}
                              {vocabulary.inverted ? " · inverted" : ""}
                            </option>
                          ))}
                        </select>
                      </li>
                    );
                  })}
                </ul>
              )}
            </div>
          ) : (
            <p className="mt-1 text-[10px] leading-snug text-ink-2">
              A JSON, text or CSV file of class names, so outputs read as names rather
              than indices. Several vocabularies in one file are fine.
            </p>
          )}

          <input
            ref={labels}
            type="file"
            className="hidden"
            accept=".json,.txt,.csv,.tsv"
            onChange={(event) => {
              const file = event.target.files?.[0];
              event.target.value = "";
              if (file) void loadLabels(file);
            }}
          />
        </div>

        {/*
          Naming what is blocking, rather than only that something is. A 76-layer
          import can be two typed expressions away from running, and "finish the
          architecture first" gave no way of knowing which two [E-038].
        */}
        {!compiled && (
          <div className="space-y-1">
            <p className="text-[10px] leading-snug text-ink-2">
              {blocking.length === 0
                ? "Data can only flow through a model that compiles."
                : `${blocking.length} thing${blocking.length > 1 ? "s" : ""} to fix before data can flow:`}
            </p>
            {blocking.slice(0, 4).map((diagnostic, index) => (
              <button
                // eslint-disable-next-line react/no-array-index-key
                key={index}
                type="button"
                disabled={!diagnostic.node_id}
                onClick={() => diagnostic.node_id && setSelection([diagnostic.node_id])}
                className="block w-full text-left text-[10px] leading-snug text-danger
                           hover:underline disabled:cursor-default disabled:no-underline"
              >
                {diagnostic.node_id && (
                  <span className="text-ink-1">
                    {nodeNames.get(diagnostic.node_id) ?? diagnostic.node_id}:{" "}
                  </span>
                )}
                {diagnostic.message}
              </button>
            ))}
            {blocking.length > 4 && (
              <p className="text-[10px] text-ink-2">
                and {blocking.length - 4} more, listed in the status bar.
              </p>
            )}
          </div>
        )}
      </div>

      {errors.length > 0 && (
        <div className="border-b border-line bg-danger/10 px-3 py-2">
          {errors.map((diagnostic, index) => (
            // eslint-disable-next-line react/no-array-index-key
            <p key={index} className="text-[11px] leading-snug text-danger">
              {diagnostic.message}
            </p>
          ))}
        </div>
      )}

      <div className="min-h-0 flex-1 overflow-y-auto p-3">
        {activations.length === 0 && errors.length === 0 && (
          <p className="text-[11px] leading-relaxed text-ink-2">
            Nothing has been run yet. Choose an input and press{" "}
            <span className="text-ink-1">Run data through</span> to watch a real
            sample move through every layer.
          </p>
        )}

        {sampleLabel && (
          <p className="mb-2 text-[11px] text-ink-1">
            Sample: <span className="text-ink-0">{sampleLabel}</span>
          </p>
        )}

        {inputPreview && (
          <section className="mb-3 min-w-0 overflow-hidden rounded border border-line p-2">
            <h3 className="mb-1.5 text-[11px] font-semibold text-ink-0">
              What went in
            </h3>
            <ActivationView activation={inputPreview} compact />
          </section>
        )}

        {activations.map((activation) => {
          const open = focused === activation.node_id;
          return (
            <section
              key={activation.node_id}
              className={`mb-2 min-w-0 overflow-hidden rounded border p-2 ${
                open ? "border-accent" : "border-line"
              }`}
            >
              <button
                type="button"
                onClick={() => focus(open ? null : activation.node_id)}
                className="flex w-full items-baseline justify-between gap-2 text-left"
              >
                <span className="truncate text-[11px] font-semibold text-ink-0">
                  {activation.label}
                </span>
                <span className="shrink-0 font-mono text-[10px] text-ink-2">
                  {activation.shape.slice(1).join(" × ")}
                </span>
              </button>

              <div className="mt-1.5">
                <ActivationView activation={activation} compact={!open} />
              </div>

              {open && (
                <dl className="mt-2 grid grid-cols-2 gap-x-3 gap-y-0.5 text-[10px]">
                  {(
                    [
                      ["min", activation.stats.min],
                      ["max", activation.stats.max],
                      ["mean", activation.stats.mean],
                      ["std", activation.stats.std],
                    ] as const
                  ).map(([label, value]) => (
                    <div key={label} className="flex justify-between">
                      <dt className="text-ink-2">{label}</dt>
                      <dd className="font-mono text-ink-1">{value.toFixed(4)}</dd>
                    </div>
                  ))}
                  <div className="col-span-2 flex justify-between">
                    <dt className="text-ink-2">dead values</dt>
                    <dd
                      className={`font-mono ${
                        activation.stats.sparsity > 0.95 ? "text-warn" : "text-ink-1"
                      }`}
                      title={
                        activation.stats.sparsity > 0.95
                          ? "Almost everything here is zero: this layer is passing nothing on."
                          : undefined
                      }
                    >
                      {(activation.stats.sparsity * 100).toFixed(1)}%
                    </dd>
                  </div>
                </dl>
              )}
            </section>
          );
        })}
      </div>
    </div>
  );
}
