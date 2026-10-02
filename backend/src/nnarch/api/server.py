"""
> [!AML-DOC-FILE]
@file       api/server.py
@description FastAPI application exposing the architecture engine to the desktop
             frontend: catalog, graph validation, activations, export and training.
@module     nnarch.api.server
@exports    app, create_app, main
@created    2026-09-30
@context    Runs as a Tauri sidecar on port 8756 and prints a readiness handshake
            on stdout [amm: A.3, E.1, E.5]. TensorFlow is imported lazily so the
            process answers /health long before the ~8s Keras import completes
            [task 01].
"""

from __future__ import annotations

import argparse
import contextlib
import os
import queue
import sys
import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel, Field

from nnarch import IR_VERSION, __version__
from nnarch.data import DatasetChoice
from nnarch.export.codegen import ExportOptions
from nnarch.ir.project import ProjectMeta, Viewport
from nnarch.ir.schema import GraphIR


class ActivationRequest(BaseModel):
    """
    > [!AML-DOC-UNIT]
    Body of a `POST /graph/activations` call.
    @param graph    the architecture to run
    @param dataset  where the sample comes from
    @param node_ids restrict capture to these nodes; empty captures every layer
    @param colormap ramp to render feature maps and heatmaps with
    @param labels   output names keyed by class count, as JSON has no integer keys
    """

    graph: GraphIR
    dataset: DatasetChoice = Field(default_factory=DatasetChoice)
    node_ids: list[str] = Field(default_factory=list)
    colormap: str = "viridis"
    #: Output names keyed by how many there are, from a labels file the user chose.
    #: Sent with the request rather than held by the engine, so the engine stays
    #: stateless and two windows cannot label each other's models [E-042].
    labels: dict[str, list[str]] = Field(default_factory=dict)
    #: Names chosen for a particular layer by hand, keyed by node id. An empty list
    #: means "no names on this one", which is different from not mentioning it.
    label_assignments: dict[str, list[str]] = Field(default_factory=dict)


class OptimizeRequest(BaseModel):
    """
    > [!AML-DOC-UNIT]
    Body of a `POST /graph/optimize` call.
    @param graph            the architecture to look at
    @param measure_latency  whether to time the models; off is much faster on a large
                            graph, and reports sizes only
    """

    graph: GraphIR
    measure_latency: bool = True


class ApplyRequest(BaseModel):
    """
    > [!AML-DOC-UNIT]
    Body of a `POST /graph/optimize/apply` call.
    @param graph        the architecture to rewrite
    @param proposal_ids which proposals to carry out, in order
    """

    graph: GraphIR
    proposal_ids: list[str] = Field(default_factory=list)


class SaveRequest(BaseModel):
    """
    > [!AML-DOC-UNIT]
    Body of a `POST /project/save` call.
    @param graph    the architecture to store
    @param viewport canvas pan and zoom, so reopening restores the view
    @param meta     file metadata; timestamps are filled in by the engine
    @param runs     training histories to keep alongside the model
    """

    graph: GraphIR
    viewport: Viewport = Field(default_factory=Viewport)
    meta: ProjectMeta = Field(default_factory=ProjectMeta)
    runs: list[dict[str, Any]] = Field(default_factory=list)


class ExportRequest(BaseModel):
    """
    > [!AML-DOC-UNIT]
    Body of a `POST /export` call.
    @param graph   the architecture to turn into a project
    @param options generation choices; sensible defaults mean the caller may omit it
    """

    graph: GraphIR
    options: ExportOptions = Field(default_factory=ExportOptions)

PORT = 8756
READY_SENTINEL = "NNARCH_READY"

REQUIRED_PYTHON = (3, 13)

ALLOWED_ORIGINS = [
    "tauri://localhost",
    "http://tauri.localhost",
    "https://tauri.localhost",
]
"""Custom schemes the desktop shell serves the app from; these have no port."""

LOOPBACK_ORIGIN_PATTERN = r"^https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$"
"""
> [!AML-DOC-UNIT]
Any loopback origin, on any port.

The port cannot be fixed on either side: the desktop shell asks the operating system
for a free port so it never collides with an engine already running by hand, and a
developer's dev server moves too. Pinning 5173 meant the app could resolve the right
engine port and still have every request refused.

This is not a widening of exposure: the engine binds to 127.0.0.1 only, so an origin
that can reach it is already running on this machine.
"""

WARM_TIMEOUT_SECONDS = 240.0

_warm_lock = threading.Lock()
_warm_done = threading.Event()
_warm_state: dict[str, Any] = {"status": "cold", "error": None, "versions": {}}


class PythonVersionError(RuntimeError):
    """
    > [!AML-DOC-UNIT]
    Raised when the interpreter cannot host TensorFlow.
    @context ERL E-001: TensorFlow 2.21 publishes no cp314 wheel, so running under
             python3.14 fails at import with a confusing message. Failing early
             with an actionable one is the documented resolution path.
    """


PARENT_PID_ENV = "NNARCH_PARENT_PID"
PARENT_POLL_SECONDS = 2.0


def watch_parent(pid: int) -> None:
    """
    > [!AML-DOC-UNIT]
    Exit the process once the supervising parent is gone.
    @param pid process id of the supervisor, as passed in `NNARCH_PARENT_PID`
    @returns None; the function only returns by ending the process
    @sideEffects polls every two seconds and calls `os._exit` when the parent dies
    @context The desktop shell kills the engine on a clean quit, but a crash or a
             force-quit never reaches that handler and would leave the engine
             holding its port, so the next launch could not bind. Watching the
             parent makes the engine's lifetime genuinely bounded by the shell's,
             regardless of how the shell ends.
             `os._exit` is deliberate: this runs on a daemon thread and a normal
             exit would wait for uvicorn's own shutdown, which the missing parent
             may block.
    """
    while True:
        time.sleep(PARENT_POLL_SECONDS)
        try:
            os.kill(pid, 0)
        except (ProcessLookupError, PermissionError):
            os._exit(0)


def start_parent_watch() -> int | None:
    """
    > [!AML-DOC-UNIT]
    Begin watching the supervising process, when one identified itself.
    @returns the parent pid being watched, or None when running unsupervised
    @sideEffects spawns a daemon thread
    @context Opt-in through an environment variable, so running the engine by hand
             from a terminal behaves normally.
    """
    raw = os.environ.get(PARENT_PID_ENV)
    if not raw or not raw.isdigit():
        return None
    pid = int(raw)
    threading.Thread(target=watch_parent, args=(pid,), name="nnarch-parent-watch",
                     daemon=True).start()
    return pid


def assert_python_supported() -> None:
    """
    > [!AML-DOC-UNIT]
    Verify the running interpreter can import TensorFlow.
    @returns None
    @raises PythonVersionError when the minor version has no TensorFlow wheel
    @sideEffects none
    """
    if sys.version_info[:2] != REQUIRED_PYTHON:
        running = ".".join(str(part) for part in sys.version_info[:3])
        wanted = ".".join(str(part) for part in REQUIRED_PYTHON)
        raise PythonVersionError(
            f"AI Architect needs Python {wanted}.x but is running {running}. "
            f"TensorFlow 2.21 publishes no wheel for other versions. "
            f"Recreate the environment with: "
            f"/opt/homebrew/bin/python{wanted} -m venv backend/.venv"
        )


def _run_warm_up() -> None:
    """
    > [!AML-DOC-UNIT]
    Import TensorFlow and build the layer catalog. Runs exactly once, on the
    background warm-up thread.
    @returns None
    @raises nothing; failures are captured into the shared warm state so every
            endpoint can report them instead of crashing the process
    @sideEffects imports tensorflow and keras (~8s), populates the layer registry,
                 and signals `_warm_done` so waiting requests can proceed
    """
    try:
        assert_python_supported()
        import keras
        import tensorflow as tf

        from nnarch.catalog import bootstrap

        count = len(bootstrap().all())
        _warm_state.update(
            status="warm",
            error=None,
            versions={
                "tensorflow": tf.__version__,
                "keras": keras.__version__,
                "python": ".".join(str(part) for part in sys.version_info[:3]),
                "nnarch": __version__,
                "ir_version": IR_VERSION,
                "layer_count": count,
            },
        )
    except Exception as exc:
        _warm_state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        _warm_done.set()


def start_warm_up() -> None:
    """
    > [!AML-DOC-UNIT]
    Begin warming the engine if it has not started already.
    @returns None
    @sideEffects spawns at most one daemon thread per process
    """
    with _warm_lock:
        if _warm_state["status"] != "cold":
            return
        _warm_state["status"] = "warming"
    threading.Thread(target=_run_warm_up, name="nnarch-warmup", daemon=True).start()


def warm_state() -> dict[str, Any]:
    """
    > [!AML-DOC-UNIT]
    Snapshot the warm state without waiting.
    @returns copy of {status, error, versions}
    """
    return dict(_warm_state)


def warm_up(timeout: float | None = WARM_TIMEOUT_SECONDS) -> dict[str, Any]:
    """
    > [!AML-DOC-UNIT]
    Ensure the engine is warm, waiting for the background import to finish.
    @param timeout seconds to wait; None waits indefinitely
    @returns the warm-state snapshot, whose status is "warming" only when the
             timeout elapsed first
    @sideEffects starts the warm-up thread on first call; blocks the calling
                 request thread until the import settles
    @context Fixes the contract mismatch where a concurrent caller received
             "warming" instead of waiting, which made /catalog answer 503 during
             the first seconds after boot.
    """
    start_warm_up()
    _warm_done.wait(timeout)
    return warm_state()


def _labels_by_size(labels: dict[str, list[str]]) -> dict[int, list[str]]:
    """
    > [!AML-DOC-UNIT]
    Turn the request's label map back into integer keys.
    @param labels names keyed by class count, as strings
    @returns the same keyed by int
    @sideEffects none
    @context JSON has no integer keys, so the count makes the round trip as a string
             and is read back here rather than anywhere further in.
    """
    out: dict[int, list[str]] = {}
    for key, names in labels.items():
        try:
            out[int(key)] = [str(name) for name in names]
        except (TypeError, ValueError):
            continue
    return out

def _build_adapter(request: "ActivationRequest") -> tuple[Any, dict[str, Any] | None]:
    """
    > [!AML-DOC-UNIT]
    Resolve a request's data source against the model it will feed.
    @param request the activation request
    @returns (adapter, None) on success, or (None, diagnostic) when the source
             cannot provide what the model needs
    @sideEffects compiles the graph to learn its input and output shapes
    """
    from nnarch.catalog import registry
    from nnarch.data import DatasetError, make_adapter
    from nnarch.ir.compiler import compile_graph

    compiled = compile_graph(request.graph, registry)
    if not compiled.ok:
        return None, {
            "severity": "error",
            "code": "activations_graph_invalid",
            "message": "The architecture has to be valid before data can flow through it.",
            "node_id": None,
            "edge_id": None,
        }

    # A model may take several inputs; the data source feeds the one the sample is
    # for, chosen by the same rule the preview itself uses [E-036].
    from nnarch.viz.tensors import primary_input_index

    primary = primary_input_index(compiled.model)
    shape = [int(dim) for dim in compiled.model.inputs[primary].shape[1:] if dim]
    outputs = compiled.model.outputs[0].shape
    classes = int(outputs[-1]) if outputs[-1] else None

    try:
        return make_adapter(request.dataset, shape, classes), None
    except DatasetError as exc:
        return None, {
            "severity": "error", "code": "activations_no_data",
            "message": str(exc), "node_id": None, "edge_id": None,
        }


def create_app() -> FastAPI:
    """
    > [!AML-DOC-UNIT]
    Build the FastAPI application and register every route.
    @returns the configured FastAPI instance
    @sideEffects starts a daemon thread that warms TensorFlow on startup
    """
    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        """
        > [!AML-DOC-UNIT]
        Start warming TensorFlow as soon as the server boots, so the first real
        request rarely has to wait.
        @returns async context yielding once the warm-up thread is running
        @sideEffects spawns the warm-up daemon thread
        """
        start_warm_up()
        yield

    app = FastAPI(
        title="AI Architect Engine",
        version=__version__,
        description="Keras 3 / TensorFlow neural architecture prototyping engine.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_origin_regex=LOOPBACK_ORIGIN_PATTERN,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> dict[str, Any]:
        """
        > [!AML-DOC-UNIT]
        Liveness probe. Answers immediately, without waiting for TensorFlow.
        @returns {status, engine, ir_version, warm} where `warm` reports the
                 TensorFlow import state: cold, warming, warm or failed
        """
        return {
            "status": "ok",
            "engine": __version__,
            "ir_version": IR_VERSION,
            "python": ".".join(str(part) for part in sys.version_info[:3]),
            "warm": _warm_state["status"],
            "error": _warm_state["error"],
            "versions": dict(_warm_state["versions"]),
        }

    @app.get("/ready")
    def ready() -> dict[str, Any]:
        """
        > [!AML-DOC-UNIT]
        Readiness probe that blocks until TensorFlow is imported and the catalog is
        built. The desktop shell polls this to switch its status badge to green.
        @returns the warm-state dict once settled
        """
        return warm_up()

    @app.get("/catalog")
    def catalog() -> dict[str, Any]:
        """
        > [!AML-DOC-UNIT]
        Full layer catalog: the category tree, every LayerSpec and the enum
        vocabularies the property panel needs.
        @returns catalog JSON as produced by `Registry.as_json`
        @raises fastapi.HTTPException 503 when the engine failed to warm up
        """
        from fastapi import HTTPException

        from nnarch.catalog import registry

        state = warm_up()
        if state["status"] != "warm":
            raise HTTPException(status_code=503, detail=state["error"] or "engine warming up")
        return registry.as_json()

    @app.post("/graph/validate")
    def validate_graph(graph: "GraphIR") -> dict[str, Any]:
        """
        > [!AML-DOC-UNIT]
        Validate an architecture and report the shape of every layer's output.
        @param graph the architecture as graph IR, posted as JSON
        @returns {diagnostics, shapes, params_total, trainable_params, compiled}
                 where `shapes` maps node id to its inferred output shape, dtype and
                 weight count, and `compiled` says whether a model was produced
        @raises fastapi.HTTPException 503 when the engine failed to warm up
        @sideEffects builds a throwaway Keras model to obtain the shapes
        @context The editor calls this on a debounce while the user edits, so a
                 malformed graph must answer 200 with diagnostics rather than 4xx
                 [amm: E.1, task 09]. Shapes come from Keras itself, never from
                 reimplemented shape math.
        """
        from fastapi import HTTPException

        from nnarch.catalog import registry
        from nnarch.ir import compile_graph

        state = warm_up()
        if state["status"] != "warm":
            raise HTTPException(status_code=503, detail=state["error"] or "engine warming up")

        result = compile_graph(graph, registry)
        return {
            "compiled": result.ok,
            "diagnostics": [d.model_dump(mode="json") for d in result.diagnostics],
            "shapes": {
                node_id: shape.model_dump(mode="json")
                for node_id, shape in result.shapes.items()
            },
            "params_total": result.params_total,
            "trainable_params": result.trainable_params,
        }

    @app.post("/export")
    def export_graph(request: "ExportRequest") -> Any:
        """
        > [!AML-DOC-UNIT]
        Generate a standalone trainable project from an architecture.
        @param request the graph plus the export options
        @returns a zip archive as a streaming response, named after the project, or
                 a 422 JSON body listing the diagnostics when the graph cannot be
                 exported
        @raises fastapi.HTTPException 503 when the engine failed to warm up, or 422
                carrying the diagnostics when generation failed
        @sideEffects compiles the graph and builds the archive in memory
        @context The desktop shell streams this straight into a native save dialog
                 [amm: E.1, task 08]. Unlike `/graph/validate`, a failure here is a
                 4xx: the user asked for a file and there is none to give them.
        """
        from fastapi import HTTPException
        from fastapi.responses import StreamingResponse

        from nnarch.catalog import registry
        from nnarch.export import build_zip, export_project
        from nnarch.naming import sanitize_name

        state = warm_up()
        if state["status"] != "warm":
            raise HTTPException(status_code=503, detail=state["error"] or "engine warming up")

        bundle = export_project(request.graph, registry, request.options)
        if not bundle.ok:
            raise HTTPException(status_code=422, detail={
                "message": "The architecture could not be exported.",
                "diagnostics": [d.model_dump(mode="json") for d in bundle.diagnostics],
            })

        folder = sanitize_name(request.options.project_name or request.graph.name, "model")
        archive = build_zip(bundle, root=folder)
        return StreamingResponse(
            iter([archive]),
            media_type="application/zip",
            headers={
                "Content-Disposition": f'attachment; filename="{folder}.zip"',
                "Content-Length": str(len(archive)),
                "X-Nnarch-Params": str(bundle.params),
            },
        )

    @app.post("/project/save")
    def project_save(request: "SaveRequest") -> Any:
        """
        > [!AML-DOC-UNIT]
        Serialise a project into a `.nnarch` archive.
        @param request the graph plus the viewport and metadata to store with it
        @returns the archive as a download named after the project
        @sideEffects none; the archive is built in memory and never written here,
                     so the caller decides where it lands
        """
        from fastapi.responses import StreamingResponse

        from nnarch.ir.project import PROJECT_SUFFIX, save_project
        from nnarch.naming import sanitize_name

        archive = save_project(
            request.graph, viewport=request.viewport, meta=request.meta, runs=request.runs
        )
        name = sanitize_name(request.graph.name, "project")
        return StreamingResponse(
            iter([archive]),
            media_type="application/zip",
            headers={
                "Content-Disposition": f'attachment; filename="{name}{PROJECT_SUFFIX}"',
                "Content-Length": str(len(archive)),
            },
        )

    @app.post("/project/open")
    async def project_open(file: UploadFile = File(...)) -> dict[str, Any]:
        """
        > [!AML-DOC-UNIT]
        Read an uploaded `.nnarch` archive back into a project.
        @param file the uploaded archive
        @returns {project, diagnostics}; `project` is null when it could not be read
        @sideEffects none
        @context Answers 200 even for an unreadable file, with the reason in the
                 diagnostics, so the editor can show a message and carry on rather
                 than treating it as a transport failure [amm: B.2].
        """
        from nnarch.ir.project import load_project

        project, diagnostics = load_project(await file.read())
        return {
            "project": project.model_dump(mode="json") if project else None,
            "diagnostics": [d.model_dump(mode="json") for d in diagnostics],
        }

    @app.post("/project/import")
    async def project_import(file: UploadFile = File(...)) -> dict[str, Any]:
        """
        > [!AML-DOC-UNIT]
        Turn a saved model into an editable graph.
        @param file a `.keras` archive, a `.tflite` model, or the JSON from
                    `Model.to_json()`
        @returns {graph, diagnostics}; `graph` is null only when the file cannot be
                 read at all
        @raises fastapi.HTTPException 503 when the engine failed to warm up
        @sideEffects reads the upload, and loads TensorFlow Lite for a `.tflite` file
        @context The three formats carry different amounts of the original, and the
                 diagnostics say which this one gave. A `.keras` archive rebuilds the
                 architecture exactly; a `.tflite` file gives the converted graph,
                 whose settings mostly live in binary options the reader cannot see.
        """
        from fastapi import HTTPException

        from nnarch.catalog import registry
        from nnarch.ir.importers import import_model

        state = warm_up()
        if state["status"] != "warm":
            raise HTTPException(status_code=503, detail=state["error"] or "engine warming up")

        graph, diagnostics = import_model(
            await file.read(), file.filename or "model", registry
        )
        return {
            "graph": graph.model_dump(mode="json") if graph else None,
            "diagnostics": [d.model_dump(mode="json") for d in diagnostics],
        }

    @app.post("/graph/activations")
    def graph_activations(request: "ActivationRequest") -> dict[str, Any]:
        """
        > [!AML-DOC-UNIT]
        Run one real sample through the model and return what every layer produced.
        @param request the graph, the data source, and rendering choices
        @returns an ActivationResult: per-layer tiles, traces and statistics
        @raises fastapi.HTTPException 503 when the engine failed to warm up
        @sideEffects compiles the graph, builds a probe model and runs a forward pass
        @context Answers 200 with diagnostics when the graph will not compile, since
                 that is an ordinary state while editing rather than a failed request
                 [amm: E.1].
        """
        from fastapi import HTTPException

        from nnarch.catalog import registry
        from nnarch.viz import capture_activations

        state = warm_up()
        if state["status"] != "warm":
            raise HTTPException(status_code=503, detail=state["error"] or "engine warming up")

        adapter, problem = _build_adapter(request)
        if problem is not None:
            return {"activations": [], "input_preview": None, "sample_label": "",
                    "diagnostics": [problem]}

        result = capture_activations(
            request.graph, registry, adapter,
            node_ids=request.node_ids, colormap=request.colormap,
            labels_by_size=_labels_by_size(request.labels),
            labels_by_node=request.label_assignments,
        )
        return result.model_dump(mode="json")

    @app.post("/graph/activations/upload")
    async def graph_activations_upload(
        graph: str = Form(...),
        colormap: str = Form("viridis"),
        labels: str = Form(""),
        label_assignments: str = Form(""),
        file: UploadFile = File(...),
    ) -> dict[str, Any]:
        """
        > [!AML-DOC-UNIT]
        Run an uploaded file through the model and return what every layer produced.
        @param graph    the architecture, as a JSON string in the form
        @param colormap ramp to render with
        @param file     the image, audio clip, text or tensor to feed
        @returns an ActivationResult
        @raises fastapi.HTTPException 503 when the engine failed to warm up, or 422
                when the graph JSON is unreadable
        @sideEffects reads the upload and runs a forward pass
        @context Multipart rather than JSON because the file is binary; the graph
                 rides along as a form field so this stays one round trip.
        """
        import json

        from fastapi import HTTPException

        from nnarch.catalog import registry
        from nnarch.data import DatasetChoice, DatasetKind, DatasetError, make_adapter
        from nnarch.ir.compiler import compile_graph
        from nnarch.viz import capture_activations

        state = warm_up()
        if state["status"] != "warm":
            raise HTTPException(status_code=503, detail=state["error"] or "engine warming up")

        try:
            parsed = GraphIR.model_validate(json.loads(graph))
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"unreadable graph: {exc}") from exc

        compiled = compile_graph(parsed, registry)
        if not compiled.ok:
            return {"activations": [], "input_preview": None, "sample_label": "",
                    "diagnostics": [d.model_dump(mode="json") for d in compiled.diagnostics]}

        # The same rule as `_build_adapter`: a model may take several inputs, and the
        # file belongs to the one it fits [E-036]. This second call site was reading
        # `inputs[0]` for a while after the first was fixed.
        from nnarch.viz.tensors import primary_input_index

        primary = primary_input_index(compiled.model)
        shape = [int(dim) for dim in compiled.model.inputs[primary].shape[1:] if dim]
        try:
            adapter = make_adapter(
                DatasetChoice(kind=DatasetKind.UPLOAD), shape, None,
                payload=await file.read(), filename=file.filename or "upload",
            )
        except DatasetError as exc:
            return {"activations": [], "input_preview": None, "sample_label": "",
                    "diagnostics": [{"severity": "error", "code": "upload_failed",
                                     "message": str(exc), "node_id": None, "edge_id": None}]}

        return capture_activations(
            parsed, registry, adapter, colormap=colormap,
            labels_by_size=_labels_by_size(json.loads(labels) if labels else {}),
            labels_by_node=json.loads(label_assignments) if label_assignments else {},
        ).model_dump(mode="json")

    @app.post("/labels/parse")
    async def labels_parse(file: UploadFile = File(...)) -> dict[str, Any]:
        """
        > [!AML-DOC-UNIT]
        Read a labels file and return the vocabularies in it.
        @param file the labels file: JSON, text or CSV
        @returns the vocabularies, keyed by how many names each holds, and a sentence
                 describing what was found
        @raises fastapi.HTTPException 422 when nothing in it reads as a list of names
        @sideEffects none; the engine keeps nothing, and the editor sends the result
                     back with each activation request [E-042]
        """
        from fastapi import HTTPException

        from nnarch.data.labels import match_to_size, parse_label_file

        payload = await file.read()
        try:
            parsed = parse_label_file(payload, file.filename or "labels")
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        return {
            "filename": parsed.filename,
            "note": parsed.note,
            # Each vocabulary carries its own names, not just its size. Two of the
            # same length are a real case — this is the only way a choice between
            # them can mean anything [E-042].
            "vocabularies": [
                {
                    "name": v.name,
                    "count": len(v.names),
                    "inverted": v.inverted,
                    "names": v.names,
                }
                for v in parsed.vocabularies
            ],
            "labels": {str(size): names for size, names in match_to_size(parsed).items()},
        }

    @app.post("/graph/timeline")
    async def graph_timeline(
        graph: str = Form(...),
        steps: int = Form(8),
        feedback: str = Form(""),
        watch: str = Form(""),
        labels: str = Form(""),
        label_assignments: str = Form(""),
        ablate: bool = Form(True),
        file: UploadFile | None = File(None),
    ) -> dict[str, Any]:
        """
        > [!AML-DOC-UNIT]
        Run a model repeatedly over a moving window and report what changed.
        @param graph             the architecture, as JSON
        @param steps             how many ticks
        @param feedback          loops to close, as JSON [{source, target}]; omitted
                                 means the shapes decide
        @param watch             node ids to record across the run, as a JSON list
        @param labels            output names keyed by class count
        @param label_assignments names chosen for a particular layer
        @param ablate            also run with the loops cut
        @param file              the recording to walk through; synthetic noise when
                                 absent, which exercises the loop but says nothing
        @returns a TimelineResult
        @raises fastapi.HTTPException 503 when the engine failed to warm up, 422 when
                the graph cannot be read
        @sideEffects compiles the graph and runs one forward pass per step, twice when
                     ablating
        """
        import json

        from fastapi import HTTPException

        from nnarch.catalog import registry
        from nnarch.data.adapters import decode_upload, take_upload_note
        from nnarch.ir.compiler import compile_graph
        from nnarch.viz.tensors import primary_input_index
        from nnarch.viz.timeline import FeedbackPair, run_timeline

        state = warm_up()
        if state["status"] != "warm":
            raise HTTPException(status_code=503, detail=state["error"] or "engine warming up")

        try:
            parsed = GraphIR.model_validate(json.loads(graph))
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"unreadable graph: {exc}") from exc

        compiled = compile_graph(parsed, registry)
        if not compiled.ok:
            return {"steps": [], "traces": [], "feedback": [], "ablation": None,
                    "diagnostics": [d.model_dump(mode="json") for d in compiled.diagnostics]}

        primary = primary_input_index(compiled.model)
        shape = [int(dim) for dim in compiled.model.inputs[primary].shape[1:] if dim]

        payload = await file.read() if file is not None else b""
        filename = (file.filename if file is not None else "") or ""

        import numpy as np

        def frame(index: int) -> tuple[Any, str | None]:
            """
            > [!AML-DOC-UNIT]
            The primary input for one step.
            @param index which step
            @returns the array and a sentence about where it came from
            @sideEffects reads from the uploaded payload
            """
            if not payload:
                # No recording, so the window cannot move. The loop still runs, which
                # is worth seeing, and the caption says the input is standing still.
                rng = np.random.default_rng(index)
                return (
                    rng.standard_normal((1, *shape)).astype("float32"),
                    f"step {index}: generated noise, not a recording",
                )
            return decode_upload(payload, filename, shape, step=index), take_upload_note()

        pairs = None
        if feedback:
            pairs = [FeedbackPair.model_validate(entry) for entry in json.loads(feedback)]

        result = run_timeline(
            parsed, registry,
            steps=max(1, min(int(steps), 128)),
            frame=frame,
            feedback=pairs,
            watch=json.loads(watch) if watch else [],
            labels_by_size=_labels_by_size(json.loads(labels) if labels else {}),
            labels_by_node=json.loads(label_assignments) if label_assignments else {},
            ablate=bool(ablate),
        )
        return result.model_dump(mode="json")

    @app.post("/graph/optimize")
    def graph_optimize(request: "OptimizeRequest") -> dict[str, Any]:
        """
        > [!AML-DOC-UNIT]
        Find the changes worth proposing for an architecture, and measure each.
        @param request the graph, and whether to time the models
        @returns an OptimizeReport
        @raises fastapi.HTTPException 503 when the engine failed to warm up
        @sideEffects builds one model per proposal, and runs each several times when
                     timing
        """
        from fastapi import HTTPException

        from nnarch.catalog import registry
        from nnarch.optimize import analyse

        state = warm_up()
        if state["status"] != "warm":
            raise HTTPException(status_code=503, detail=state["error"] or "engine warming up")

        return analyse(request.graph, registry, time_it=request.measure_latency).model_dump(
            mode="json"
        )

    @app.post("/graph/optimize/apply")
    def graph_optimize_apply(request: "ApplyRequest") -> dict[str, Any]:
        """
        > [!AML-DOC-UNIT]
        Carry out the chosen proposals and hand back the rewritten architecture.
        @param request the graph and which proposals to apply
        @returns the new graph and anything worth saying
        @raises fastapi.HTTPException 503 when the engine failed to warm up
        @sideEffects none; the caller decides whether to keep what comes back
        """
        from fastapi import HTTPException

        from nnarch.catalog import registry
        from nnarch.optimize import apply_proposals

        state = warm_up()
        if state["status"] != "warm":
            raise HTTPException(status_code=503, detail=state["error"] or "engine warming up")

        graph, diagnostics = apply_proposals(request.graph, registry, request.proposal_ids)
        return {
            "graph": graph.model_dump(mode="json"),
            "diagnostics": [d.model_dump(mode="json") for d in diagnostics],
        }

    @app.websocket("/train")
    async def train(socket: WebSocket) -> None:
        """
        > [!AML-DOC-UNIT]
        Train a model, streaming metrics while it runs.
        @param socket the client connection
        @returns None
        @sideEffects starts a worker thread that trains a model, and forwards its
                     messages until the run ends or the client disconnects
        @context The first message from the client is the request; after that any
                 message means stop. Training runs on a worker thread, so the event
                 loop stays free to notice that stop and to keep the socket alive
                 [task 07].
        """
        import asyncio

        from nnarch.api.train import TrainRequest, TrainingSession, run_training
        from nnarch.catalog import registry

        await socket.accept()

        state = warm_up()
        if state["status"] != "warm":
            await socket.send_json({"type": "error", "message": state["error"] or "engine warming up"})
            await socket.close()
            return

        try:
            request = TrainRequest.model_validate(await socket.receive_json())
        except Exception as exc:
            await socket.send_json({"type": "error", "message": f"unreadable request: {exc}"})
            await socket.close()
            return

        session = TrainingSession()
        session.thread = threading.Thread(
            target=run_training, args=(request, registry, session),
            name="nnarch-training", daemon=True,
        )
        session.thread.start()

        async def watch_for_stop() -> None:
            """
            > [!AML-DOC-UNIT]
            Treat any message from the client as a request to stop.
            @returns None; runs until the training thread finishes or the socket closes
            @sideEffects sets the session's stop flag
            @context Reading the socket is also how a disconnect is noticed, so a
                     client that closes its window stops the run rather than leaving
                     it training against nobody.
            """
            try:
                while not session.finished.is_set():
                    await socket.receive_text()
                    session.stop()
            except Exception:
                session.stop()

        watcher = asyncio.create_task(watch_for_stop())
        try:
            while True:
                try:
                    message = session.messages.get_nowait()
                except queue.Empty:
                    if session.finished.is_set():
                        break
                    await asyncio.sleep(0.05)
                    continue
                await socket.send_json(message)
            while not session.messages.empty():
                await socket.send_json(session.messages.get_nowait())
        except WebSocketDisconnect:
            session.stop()
        finally:
            watcher.cancel()
            session.stop()
            with contextlib.suppress(Exception):
                await socket.close()

    return app


app = create_app()


def main(argv: list[str] | None = None) -> int:
    """
    > [!AML-DOC-UNIT]
    Console entry point: run the engine under uvicorn and emit the sidecar
    handshake line the desktop shell waits for.
    @param argv command-line arguments; defaults to sys.argv[1:]
    @returns process exit code
    @sideEffects binds a TCP port and blocks until the server stops, and starts the
                 parent watchdog when `NNARCH_PARENT_PID` names a supervisor
    """
    parser = argparse.ArgumentParser(description="AI Architect backend engine")
    parser.add_argument("--host", default="127.0.0.1", help="interface to bind")
    parser.add_argument("--port", type=int, default=PORT, help="port to bind")
    parser.add_argument("--reload", action="store_true", help="auto-reload on code change")
    args = parser.parse_args(argv)

    import uvicorn

    watched = start_parent_watch()
    if watched is not None:
        print(f"supervised by pid {watched}", flush=True)

    print(f"{READY_SENTINEL} port={args.port}", flush=True)
    uvicorn.run(
        "nnarch.api.server:app" if args.reload else app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
