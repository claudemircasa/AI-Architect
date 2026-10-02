/**
 * > [!AML-DOC-FILE]
 * @file        src/store/training.ts
 * @description Drives a training run over the engine's WebSocket and keeps the
 *              metrics it streams back.
 * @module      frontend/store/training
 * @exports     useTraining
 * @created     2026-10-01
 * @context     The socket is held in a module variable rather than in the store,
 *              because it is a live resource rather than state to render, and
 *              putting it in the store would make every metric update re-render
 *              anything watching it [task 07].
 */

import { create } from "zustand";

import { engineBaseUrl } from "../api/client";
import type { DatasetChoice } from "../api/types";
import { useActivations } from "./activations";
import { useGraph } from "./graph";

/** One recorded point on the curves. */
export interface EpochPoint {
  epoch: number;
  loss: number;
  accuracy: number | null;
  valLoss: number | null;
  valAccuracy: number | null;
}

interface TrainingStore {
  running: boolean;
  epochs: number;
  batchSize: number;
  learningRate: number;
  optimizer: string;
  totalEpochs: number;
  stepsPerEpoch: number;
  currentEpoch: number;
  progress: number;
  batchLoss: number | null;
  elapsed: number;
  history: EpochPoint[];
  message: string | null;
  lossName: string;

  configure: (patch: Partial<TrainingStore>) => void;
  start: (dataset: DatasetChoice) => void;
  stop: () => void;
  reset: () => void;
}

let socket: WebSocket | null = null;

/**
 * > [!AML-DOC-UNIT]
 * Training store. Holds the run's settings, its progress, and the curve so far.
 */
export const useTraining = create<TrainingStore>((set, get) => ({
  running: false,
  epochs: 5,
  batchSize: 32,
  learningRate: 0.001,
  optimizer: "adam",
  totalEpochs: 0,
  stepsPerEpoch: 0,
  currentEpoch: 0,
  progress: 0,
  batchLoss: null,
  elapsed: 0,
  history: [],
  message: null,
  lossName: "",

  /**
   * > [!AML-DOC-UNIT]
   * Change the run's settings.
   * @param patch the fields to change
   */
  configure: (patch) => set(patch),

  /**
   * > [!AML-DOC-UNIT]
   * Forget the previous run.
   * @sideEffects clears the curve and the progress readouts
   */
  reset: () =>
    set({
      history: [],
      currentEpoch: 0,
      progress: 0,
      batchLoss: null,
      elapsed: 0,
      message: null,
    }),

  /**
   * > [!AML-DOC-UNIT]
   * Begin training the current graph.
   * @param dataset where the data comes from
   * @sideEffects opens a WebSocket to the engine and starts a run on it
   * @context Captured activations are cleared first: they were produced by the
   *          untrained weights, and leaving them on screen beside a falling loss
   *          curve would suggest they reflect the trained model.
   */
  start: (dataset) => {
    if (get().running) return;
    useActivations.getState().clear();
    get().reset();

    const url = `${engineBaseUrl().replace(/^http/, "ws")}/train`;
    socket = new WebSocket(url);
    set({ running: true, message: "connecting…" });

    socket.onopen = () => {
      socket?.send(
        JSON.stringify({
          graph: useGraph.getState().toIR(),
          dataset,
          epochs: get().epochs,
          batch_size: get().batchSize,
          learning_rate: get().learningRate,
          optimizer: get().optimizer,
        }),
      );
      set({ message: "preparing the data…" });
    };

    socket.onmessage = (event) => {
      const payload = JSON.parse(event.data as string);
      switch (payload.type) {
        case "started":
          set({
            totalEpochs: payload.epochs,
            stepsPerEpoch: payload.steps_per_epoch,
            lossName: payload.loss,
            message: `${payload.samples.toLocaleString("en-US")} samples · ${payload.loss}`,
          });
          break;
        case "batch":
          set({
            batchLoss: payload.metrics.loss ?? null,
            progress: payload.steps_per_epoch
              ? payload.batch / payload.steps_per_epoch
              : 0,
            elapsed: payload.elapsed,
          });
          break;
        case "epoch":
          set({
            currentEpoch: payload.epoch,
            progress: 0,
            elapsed: payload.elapsed,
            history: [
              ...get().history,
              {
                epoch: payload.epoch,
                loss: payload.metrics.loss ?? 0,
                accuracy: payload.metrics.accuracy ?? null,
                valLoss: payload.metrics.val_loss ?? null,
                valAccuracy: payload.metrics.val_accuracy ?? null,
              },
            ],
          });
          break;
        case "finished":
          set({
            running: false,
            message: payload.stopped ? "stopped" : "finished",
          });
          socket?.close();
          socket = null;
          break;
        case "error":
          set({ running: false, message: payload.message });
          socket?.close();
          socket = null;
          break;
        default:
          break;
      }
    };

    socket.onerror = () => set({ running: false, message: "the engine closed the connection" });
    socket.onclose = () => {
      socket = null;
      if (get().running) set({ running: false });
    };
  },

  /**
   * > [!AML-DOC-UNIT]
   * Ask the engine to stop at the next batch.
   * @sideEffects sends a message on the socket; the run ends within a batch
   */
  stop: () => {
    socket?.send("stop");
    set({ message: "stopping…" });
  },
}));
