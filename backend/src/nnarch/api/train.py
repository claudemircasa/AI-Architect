"""
> [!AML-DOC-FILE]
@file       api/train.py
@description In-app training: fits a model on a worker thread and streams metrics
             over a WebSocket while it runs.
@module     nnarch.api.train
@exports    TrainRequest, TrainingSession, run_training
@created    2026-10-01
@context    Training blocks for minutes, so it runs on a worker thread and reports
            through a queue [task 07]. The user can stop it at any point, which
            matters because the usual reason to watch a loss curve is to decide the
            run is not worth finishing.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Any

from pydantic import BaseModel, Field

from nnarch.catalog import Registry
from nnarch.data import DatasetChoice, DatasetError, make_adapter
from nnarch.ir.compiler import compile_graph
from nnarch.ir.schema import GraphIR

THROTTLE_SECONDS = 0.1
"""Smallest gap between batch updates. Ten a second is more than the eye follows."""


class TrainRequest(BaseModel):
    """
    > [!AML-DOC-UNIT]
    What to train and how.
    @param graph         the architecture
    @param dataset       where the data comes from
    @param optimizer     Keras optimizer name
    @param learning_rate initial learning rate
    @param loss          Keras loss name; empty infers one from the output layer
    @param metrics       metrics to track
    @param epochs        how many passes over the data
    @param batch_size    examples per step
    """

    graph: GraphIR
    dataset: DatasetChoice = Field(default_factory=DatasetChoice)
    optimizer: str = "adam"
    learning_rate: float = 1e-3
    loss: str = ""
    metrics: list[str] = Field(default_factory=lambda: ["accuracy"])
    epochs: int = 5
    batch_size: int = 32


class TrainingSession:
    """
    > [!AML-DOC-UNIT]
    One training run, and the channel it reports through.
    @sideEffects owns a worker thread and a bounded queue of messages
    @context The queue is bounded so a client that stops reading cannot make the
             process grow without limit; when it fills, the oldest batch update is
             dropped, which is the right thing to lose.
    """

    def __init__(self) -> None:
        self.messages: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=256)
        self.stop_requested = threading.Event()
        self.finished = threading.Event()
        self.thread: threading.Thread | None = None

    def emit(self, message: dict[str, Any]) -> None:
        """
        > [!AML-DOC-UNIT]
        Queue a message for the client.
        @param message the payload to send
        @sideEffects drops the oldest pending message when the queue is full
        """
        try:
            self.messages.put_nowait(message)
        except queue.Full:
            try:
                self.messages.get_nowait()
                self.messages.put_nowait(message)
            except queue.Empty:
                pass

    def stop(self) -> None:
        """
        > [!AML-DOC-UNIT]
        Ask the run to end at the next batch boundary.
        @sideEffects sets the stop flag the callback reads
        """
        self.stop_requested.set()


def _infer_loss(model: Any, graph: GraphIR, registry: Registry) -> str:
    """
    > [!AML-DOC-UNIT]
    Pick a loss that matches the model's final layer.
    @param model    the compiled model, read for the output width
    @param graph    the IR, read for the output node's activation
    @param registry the layer catalog
    @returns a Keras loss name
    @context The same rule the exporter uses, so training in the app and training in
             an exported project default to the same objective [amm: E.4].
    """
    output_ids = graph.resolved_outputs()
    if not output_ids:
        return "mse"
    node = graph.node_map().get(output_ids[-1])
    if node is None:
        return "mse"
    spec = registry.get(node.type)
    activation = node.params.get("activation") or (
        spec.defaults().get("activation") if spec else None
    )
    units = model.outputs[0].shape[-1]
    if activation == "softmax" or node.type == "keras.Softmax":
        return "sparse_categorical_crossentropy"
    if activation == "sigmoid" and units == 1:
        return "binary_crossentropy"
    return "mse"


def run_training(request: TrainRequest, registry: Registry, session: TrainingSession) -> None:
    """
    > [!AML-DOC-UNIT]
    Compile, fit, and report progress until the run ends or is stopped.
    @param request  what to train
    @param registry the layer catalog
    @param session  the channel to report through
    @returns None; everything is communicated through `session`
    @raises nothing; a failure becomes an `error` message so the client always learns
            why the curve stopped
    @sideEffects trains a model, which is slow and allocates heavily
    """
    import keras

    try:
        compiled = compile_graph(request.graph, registry)
        if not compiled.ok:
            session.emit({
                "type": "error",
                "message": "The architecture has to be valid before it can be trained.",
                "diagnostics": [d.model_dump(mode="json") for d in compiled.diagnostics],
            })
            return

        model = compiled.model
        shape = [int(dim) for dim in model.inputs[0].shape[1:] if dim]
        classes = (
            int(model.outputs[0].shape[-1]) if model.outputs[0].shape[-1] else None
        )

        try:
            adapter = make_adapter(request.dataset, shape, classes)
            (x_train, y_train), (x_test, y_test) = adapter.training_data()
        except DatasetError as exc:
            session.emit({"type": "error", "message": str(exc)})
            return

        loss = request.loss or _infer_loss(model, request.graph, registry)
        optimizer = keras.optimizers.get(request.optimizer)
        optimizer.learning_rate = request.learning_rate
        model.compile(optimizer=optimizer, loss=loss, metrics=list(request.metrics))

        steps = max(1, len(x_train) // request.batch_size)
        session.emit({
            "type": "started",
            "epochs": request.epochs,
            "steps_per_epoch": steps,
            "samples": int(len(x_train)),
            "loss": loss,
            "optimizer": request.optimizer,
            "parameters": compiled.params_total,
        })

        class Reporter(keras.callbacks.Callback):
            """
            > [!AML-DOC-UNIT]
            Streams progress to the client and honours a stop request.
            @sideEffects queues messages on the session; sets `stop_training` when
                         the user asks to stop
            """

            def __init__(self) -> None:
                super().__init__()
                self.last_sent = 0.0
                self.started = time.monotonic()

            def on_train_batch_end(self, batch, logs=None):
                """
                > [!AML-DOC-UNIT]
                Report a batch, at most ten times a second.
                @param batch index of the batch just finished
                @param logs  the metrics Keras computed for it
                @sideEffects queues a message; ends the run when stop was requested
                @context Throttled because a batch can finish in milliseconds and
                         nobody reads a number that changes faster than the eye.
                """
                if session.stop_requested.is_set():
                    self.model.stop_training = True
                    return
                now = time.monotonic()
                if now - self.last_sent < THROTTLE_SECONDS:
                    return
                self.last_sent = now
                session.emit({
                    "type": "batch",
                    "batch": int(batch),
                    "steps_per_epoch": steps,
                    "metrics": {k: float(v) for k, v in (logs or {}).items()},
                    "elapsed": now - self.started,
                })

            def on_epoch_end(self, epoch, logs=None):
                """
                > [!AML-DOC-UNIT]
                Report the epoch, including validation metrics.
                @param epoch zero-based index of the epoch just finished
                @param logs  training and validation metrics for it
                @sideEffects queues a message; ends the run when stop was requested
                """
                session.emit({
                    "type": "epoch",
                    "epoch": int(epoch) + 1,
                    "metrics": {k: float(v) for k, v in (logs or {}).items()},
                    "elapsed": time.monotonic() - self.started,
                })
                if session.stop_requested.is_set():
                    self.model.stop_training = True

        history = model.fit(
            x_train,
            y_train,
            validation_data=(x_test, y_test),
            epochs=request.epochs,
            batch_size=request.batch_size,
            verbose=0,
            callbacks=[Reporter()],
        )

        session.emit({
            "type": "finished",
            "stopped": session.stop_requested.is_set(),
            "history": {k: [float(v) for v in values] for k, values in history.history.items()},
        })

    except Exception as exc:
        session.emit({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
    finally:
        session.finished.set()
