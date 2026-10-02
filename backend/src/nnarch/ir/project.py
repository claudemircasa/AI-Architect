"""
> [!AML-DOC-FILE]
@file       ir/project.py
@description The `.nnarch` project format: saving, loading, migrating, and importing
             an existing Keras model into an editable graph.
@module     nnarch.ir.project
@exports    Viewport, ProjectMeta, Project, PROJECT_SUFFIX, save_project,
            load_project, write_project_atomic, import_keras_json, MIGRATIONS
@created    2026-09-30
@context    RISK:HIGH user data [amm: B.2]. Two rules govern everything here.
            A project this build cannot fully understand is refused rather than
            partially loaded, because silently dropping half of someone's work is
            worse than refusing to open it. And a save is atomic, because a crash
            mid-write must not destroy the file it was replacing.
"""

from __future__ import annotations

import io
import json
import os
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field, ValidationError

from nnarch import IR_VERSION, __version__
from nnarch.catalog import Diagnostic, Registry

from .schema import OUTPUT_PORT, Edge, GraphIR, Node, Position

PROJECT_SUFFIX = ".nnarch"

_PROJECT_ENTRY = "project.json"
_META_ENTRY = "meta.json"
_RUNS_PREFIX = "runs/"
_ASSETS_PREFIX = "assets/"
_THUMBNAIL_ENTRY = "thumbnail.png"


class Codes:
    """
    > [!AML-DOC-UNIT]
    Diagnostic identifiers produced when loading or importing. Stable, and part of
    the contract tests and documentation rely on.
    """

    NOT_A_PROJECT = "project_not_a_zip"
    MISSING_GRAPH = "project_missing_graph"
    MALFORMED_GRAPH = "project_malformed_graph"
    VERSION_TOO_NEW = "project_version_too_new"
    NO_MIGRATION = "project_no_migration_path"
    MIGRATED = "project_migrated"
    IMPORT_UNKNOWN_LAYER = "import_unknown_layer"
    IMPORT_NOT_FUNCTIONAL = "import_not_functional"
    IMPORT_MALFORMED = "import_malformed_json"
    IMPORT_SEQUENTIAL_CHAINED = "import_sequential_chained"
    IMPORT_NOT_A_KERAS_FILE = "import_not_a_keras_file"
    IMPORT_LOSSY = "import_lossy"
    IMPORT_COMPILED_LAMBDA = "import_compiled_lambda"


class Viewport(BaseModel):
    """
    > [!AML-DOC-UNIT]
    Where the canvas was looking when the project was saved, so reopening restores
    the view rather than resetting it.
    @param x    horizontal pan
    @param y    vertical pan
    @param zoom zoom factor
    """

    x: float = 0.0
    y: float = 0.0
    zoom: float = 1.0


class ProjectMeta(BaseModel):
    """
    > [!AML-DOC-UNIT]
    Facts about the file rather than about the model.
    @param name         project name, mirroring `GraphIR.name`
    @param created      ISO-8601 timestamp of first save
    @param modified     ISO-8601 timestamp of this save
    @param tool_version version of AI Architect that wrote it
    @param ir_version   graph format version, duplicated here so the version can be
                        read without parsing the whole graph
    """

    name: str = "Untitled"
    created: str = ""
    modified: str = ""
    tool_version: str = __version__
    ir_version: int = IR_VERSION


class Project(BaseModel):
    """
    > [!AML-DOC-UNIT]
    A loaded project: the architecture, the view, and the file's own metadata.
    @param graph    the architecture
    @param viewport canvas pan and zoom
    @param meta     file metadata
    @param runs     training histories recorded against this project [task 07]
    """

    graph: GraphIR
    viewport: Viewport = Field(default_factory=Viewport)
    meta: ProjectMeta = Field(default_factory=ProjectMeta)
    runs: list[dict[str, Any]] = Field(default_factory=list)


def _now() -> str:
    """
    > [!AML-DOC-UNIT]
    Current time as an ISO-8601 string in UTC.
    @returns timestamp such as "2026-09-30T18:04:11+00:00"
    """
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


MIGRATIONS: dict[int, Callable[[dict[str, Any]], dict[str, Any]]] = {}
"""
> [!AML-DOC-UNIT]
Migrations keyed by the version they upgrade *from*. Append-only: a released
migration is never edited, because files written by that release still exist.

Empty today, since only format v1 has shipped. The machinery is here so the first
format change is a matter of adding one function, and so `load_project` already has
the behaviour tested.
"""


def _migrate(payload: dict[str, Any]) -> tuple[dict[str, Any], list[Diagnostic]]:
    """
    > [!AML-DOC-UNIT]
    Bring a stored graph up to the current format version.
    @param payload the raw `project.json` contents
    @returns (upgraded payload, diagnostics describing what happened)
    @raises nothing; a version that cannot be reached becomes an error diagnostic
    @sideEffects none; the input dict is not mutated
    """
    diagnostics: list[Diagnostic] = []
    version = int(payload.get("graph", {}).get("ir_version", payload.get("ir_version", 1)))

    if version > IR_VERSION:
        return payload, [
            Diagnostic(
                severity="error",
                code=Codes.VERSION_TOO_NEW,
                message=(
                    f"This project was saved in format v{version}, but this build "
                    f"understands up to v{IR_VERSION}. Update AI Architect to open it. "
                    f"Nothing has been loaded, so the file is untouched."
                ),
            )
        ]

    current = dict(payload)
    while version < IR_VERSION:
        migration = MIGRATIONS.get(version)
        if migration is None:
            return current, [
                Diagnostic(
                    severity="error",
                    code=Codes.NO_MIGRATION,
                    message=(
                        f"This project uses format v{version} and there is no way to "
                        f"upgrade it to v{IR_VERSION}. The file has not been changed."
                    ),
                )
            ]
        current = migration(current)
        version += 1
        diagnostics.append(
            Diagnostic(
                severity="info",
                code=Codes.MIGRATED,
                message=f"Project upgraded to graph format v{version}.",
            )
        )
    return current, diagnostics


def save_project(
    graph: GraphIR,
    *,
    viewport: Viewport | None = None,
    meta: ProjectMeta | None = None,
    runs: list[dict[str, Any]] | None = None,
    assets: dict[str, bytes] | None = None,
    thumbnail: bytes | None = None,
) -> bytes:
    """
    > [!AML-DOC-UNIT]
    Serialise a project into `.nnarch` archive bytes.
    @param graph     the architecture to store
    @param viewport  canvas pan and zoom
    @param meta      file metadata; timestamps are filled in here
    @param runs      training histories to keep alongside the model
    @param assets    sample inputs the user attached, by filename
    @param thumbnail PNG preview of the canvas
    @returns the archive contents
    @sideEffects none; the archive is built in memory
    """
    viewport = viewport or Viewport()
    meta = meta or ProjectMeta(name=graph.name)
    meta = meta.model_copy(
        update={
            "name": graph.name,
            "created": meta.created or _now(),
            "modified": _now(),
            "tool_version": __version__,
            "ir_version": graph.ir_version,
        }
    )

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            _PROJECT_ENTRY,
            json.dumps(
                {
                    "graph": graph.model_dump(mode="json"),
                    "viewport": viewport.model_dump(mode="json"),
                },
                indent=2,
            ),
        )
        archive.writestr(_META_ENTRY, json.dumps(meta.model_dump(mode="json"), indent=2))
        for index, run in enumerate(runs or []):
            archive.writestr(f"{_RUNS_PREFIX}{index:04d}.json", json.dumps(run, indent=2))
        for name, blob in (assets or {}).items():
            archive.writestr(f"{_ASSETS_PREFIX}{name}", blob)
        if thumbnail is not None:
            archive.writestr(_THUMBNAIL_ENTRY, thumbnail)
    return buffer.getvalue()


def load_project(data: bytes) -> tuple[Project | None, list[Diagnostic]]:
    """
    > [!AML-DOC-UNIT]
    Read `.nnarch` archive bytes back into a project.
    @param data the archive contents
    @returns (the project, diagnostics) with the project None whenever any
             diagnostic is an error
    @raises nothing for any input: a corrupt or foreign file comes back as a
            diagnostic, so the app stays usable
    @sideEffects none
    @context Nothing is returned partially. Either the whole project loads or the
             caller gets None and an explanation [amm: B.2].
    """
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except (zipfile.BadZipFile, OSError):
        return None, [
            Diagnostic(
                severity="error",
                code=Codes.NOT_A_PROJECT,
                message=(
                    "This file is not an AI Architect project. A .nnarch file is a zip "
                    "archive; this one could not be opened as one."
                ),
            )
        ]

    names = set(archive.namelist())
    if _PROJECT_ENTRY not in names:
        return None, [
            Diagnostic(
                severity="error",
                code=Codes.MISSING_GRAPH,
                message=(
                    f"This archive has no {_PROJECT_ENTRY}, so it holds no architecture. "
                    f"It may be a different kind of zip file."
                ),
            )
        ]

    try:
        payload = json.loads(archive.read(_PROJECT_ENTRY))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return None, [
            Diagnostic(
                severity="error",
                code=Codes.MALFORMED_GRAPH,
                message=f"The project file inside this archive is damaged: {exc}",
            )
        ]

    payload, diagnostics = _migrate(payload)
    if any(diagnostic.severity == "error" for diagnostic in diagnostics):
        return None, diagnostics

    try:
        graph = GraphIR.model_validate(payload.get("graph", {}))
        viewport = Viewport.model_validate(payload.get("viewport", {}))
    except ValidationError as exc:
        return None, [
            *diagnostics,
            Diagnostic(
                severity="error",
                code=Codes.MALFORMED_GRAPH,
                message=(
                    f"The architecture in this project does not match the expected "
                    f"shape: {exc.error_count()} problem(s) in project.json."
                ),
            ),
        ]

    meta = ProjectMeta(name=graph.name)
    if _META_ENTRY in names:
        try:
            meta = ProjectMeta.model_validate(json.loads(archive.read(_META_ENTRY)))
        except (json.JSONDecodeError, ValidationError, UnicodeDecodeError):
            diagnostics.append(
                Diagnostic(
                    severity="warning",
                    code=Codes.MALFORMED_GRAPH,
                    message="The project's metadata is unreadable; defaults were used.",
                )
            )

    runs: list[dict[str, Any]] = []
    for name in sorted(entry for entry in names if entry.startswith(_RUNS_PREFIX)):
        try:
            runs.append(json.loads(archive.read(name)))
        except (json.JSONDecodeError, UnicodeDecodeError):
            diagnostics.append(
                Diagnostic(
                    severity="warning",
                    code=Codes.MALFORMED_GRAPH,
                    message=f"Training history {name} is unreadable and was skipped.",
                )
            )

    return Project(graph=graph, viewport=viewport, meta=meta, runs=runs), diagnostics


def write_project_atomic(path: Path, data: bytes, *, keep_backup: bool = True) -> Path:
    """
    > [!AML-DOC-UNIT]
    Write project bytes to disk without risking the file already there.
    @param path        destination `.nnarch` path
    @param data        archive contents
    @param keep_backup move any existing file aside to `<name>.nnarch.bak` first
    @returns the path written
    @raises OSError when the directory cannot be written
    @sideEffects creates a temporary file next to the target, then renames it over
                 the destination
    @context The write goes to a temporary file in the same directory and is renamed
             into place, which is atomic on a POSIX filesystem. A crash halfway
             through therefore leaves the previous file intact rather than a
             truncated one.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")

    with open(temporary, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())

    if keep_backup and path.exists():
        backup = path.with_suffix(path.suffix + ".bak")
        backup.unlink(missing_ok=True)
        path.replace(backup)

    temporary.replace(path)
    return path



def _output_port(index: int) -> str:
    """
    > [!AML-DOC-UNIT]
    Name one of a layer's outputs.
    @param index position in what the layer returned
    @returns the port name an edge carries
    @sideEffects none
    @context The first output keeps the ordinary name, so every single-output layer
             in every existing project is unaffected and nothing needs migrating.
    """
    return OUTPUT_PORT if index == 0 else f"{OUTPUT_PORT}_{index}"

def _tensor_sources(value: Any, into: list[tuple[str, int]]) -> None:
    """
    > [!AML-DOC-UNIT]
    Collect the producing layer names from a Keras `inbound_nodes` argument.
    @param value a serialised argument, possibly a nested list of tensors
    @param into  list the found (layer name, output index) pairs are appended to
    @sideEffects appends to `into`
    @context Keras records each input as a `__keras_tensor__` whose `keras_history`
             names the layer that produced it. Merging layers nest their inputs one
             list deeper, which is why this recurses.

             The third element of that history is *which* output of the layer was
             taken, and it was being discarded. A recurrent layer with
             `return_state=True` returns its output and its state, and a free-running
             model is built by feeding that state back — so dropping the index made
             exactly this kind of model impossible to represent [E-034].
    """
    if isinstance(value, list):
        for item in value:
            _tensor_sources(item, into)
        return
    if isinstance(value, dict):
        if value.get("class_name") == "__keras_tensor__":
            history = value.get("config", {}).get("keras_history")
            if isinstance(history, list) and history:
                index = int(history[2]) if len(history) > 2 else 0
                into.append((str(history[0]), index))
            return
        for item in value.values():
            _tensor_sources(item, into)


def import_keras_json(text: str, registry: Registry) -> tuple[GraphIR | None, list[Diagnostic]]:
    """
    > [!AML-DOC-UNIT]
    Turn a Keras model's JSON configuration into an editable graph.
    @param text     output of `Model.to_json()`, or the `config.json` inside a
                    `.keras` archive, which is the same document
    @param registry the layer catalog, used to recognise each class
    @returns (the graph, diagnostics) with the graph None only when the JSON cannot
             be read at all
    @raises nothing; unrecognised layers become nodes whose type the validator then
            reports, with their original configuration preserved so nothing is lost
    @sideEffects none
    @context Handles both model kinds. A functional model records its wiring in each
             layer's `inbound_nodes`; a sequential one records nothing, because the
             wiring *is* the order, so its layers are chained here instead. Without
             that, a sequential model imported as a pile of disconnected layers.
    """
    try:
        document = json.loads(text)
    except (json.JSONDecodeError, TypeError) as exc:
        return None, [
            Diagnostic(
                severity="error",
                code=Codes.IMPORT_MALFORMED,
                message=f"This is not valid Keras model JSON: {exc}",
            )
        ]

    config = document.get("config", {})
    layers = config.get("layers")
    if not isinstance(layers, list):
        return None, [
            Diagnostic(
                severity="error",
                code=Codes.IMPORT_NOT_FUNCTIONAL,
                message=(
                    "Only functional and sequential models can be imported. This JSON "
                    "describes no layer graph."
                ),
            )
        ]

    diagnostics: list[Diagnostic] = []
    nodes: list[Node] = []
    edges: list[Edge] = []
    unknown: set[str] = set()
    compiled_lambdas: list[tuple[str, Any]] = []
    wired = any(entry.get("inbound_nodes") for entry in layers)
    previous: str | None = None

    for index, entry in enumerate(layers):
        class_name = entry.get("class_name", "")
        layer_config = dict(entry.get("config", {}))
        name = str(
            entry.get("name")
            or layer_config.get("name")
            or f"{class_name or 'layer'}_{index}"
        )

        # Keras calls the entry point InputLayer; the catalog calls it Input.
        spec_id = "keras.Input" if class_name == "InputLayer" else f"keras.{class_name}"
        spec = registry.get(spec_id)

        params = {key: value for key, value in layer_config.items() if key != "name"}
        if spec_id == "keras.Input":
            shape = params.pop("batch_shape", None) or params.pop("shape", None)
            if isinstance(shape, list):
                params["shape"] = list(shape[1:])

        # A Lambda saved by Keras carries its function as a marshalled code object.
        # Running it would mean executing code out of a model file, which is exactly
        # the thing a file format should never talk anyone into, so the body is
        # dropped and the user is asked for the expression instead [E-035]. The
        # declared output shape is kept as the hint for what it has to produce.
        if class_name == "Lambda" and isinstance(params.get("function"), dict):
            compiled_lambdas.append((name, params.get("output_shape")))
            params["function"] = ""

        if spec is None:
            # A layer the catalog does not have — somebody's own class, or one from a
            # newer Keras. Keeping its name as the node type meant the whole model
            # refused to compile over one layer, so nothing about it could be seen or
            # run. It becomes an explicit pass-through instead: the architecture
            # around it works, the substitution is named in the diagnostics, and what
            # the layer was is kept on the node so nothing is silently lost [E-033].
            unknown.add(class_name)
            params = {"original_type": class_name, "original_params": params}
            spec_id = "keras.Identity"

        nodes.append(
            Node(
                id=name,
                type=spec_id,
                name=name,
                params=params,
                position=Position(x=240.0, y=80.0 + index * 110.0),
            )
        )

        port = spec.inputs[0].name if spec and spec.inputs else "input"

        if wired:
            for inbound in entry.get("inbound_nodes") or []:
                sources: list[tuple[str, int]] = []
                _tensor_sources(inbound.get("args", []), sources)
                for order, (source, which) in enumerate(sources):
                    edges.append(
                        Edge(
                            id=f"{source}->{name}:{order}",
                            source=source,
                            source_port=_output_port(which),
                            target=name,
                            target_port=port,
                            order=order,
                        )
                    )
                for keyword, value in (inbound.get("kwargs") or {}).items():
                    named: list[tuple[str, int]] = []
                    _tensor_sources(value, named)
                    for source, which in named:
                        edges.append(
                            Edge(
                                id=f"{source}->{name}:{keyword}",
                                source=source,
                                source_port=_output_port(which),
                                target=name,
                                target_port=keyword,
                            )
                        )
        elif previous is not None:
            edges.append(
                Edge(
                    id=f"{previous}->{name}",
                    source=previous,
                    target=name,
                    target_port=port,
                )
            )
        previous = name

    if not wired and len(nodes) > 1:
        diagnostics.append(
            Diagnostic(
                severity="info",
                code=Codes.IMPORT_SEQUENTIAL_CHAINED,
                message=(
                    "This is a sequential model, which stores no wiring, so its layers "
                    "were chained in the order they were declared."
                ),
            )
        )

    if unknown:
        diagnostics.append(
            Diagnostic(
                severity="warning",
                code=Codes.IMPORT_UNKNOWN_LAYER,
                message=(
                    f"{len(unknown)} layer type(s) are not in the catalog and were "
                    f"replaced by a pass-through so the rest of the model still runs: "
                    f"{', '.join(sorted(unknown))}. Each kept its name and its original "
                    f"settings, so you can see where it was and put it back."
                ),
            )
        )

    if compiled_lambdas:
        described = ", ".join(
            f"{name} (needs to produce {shape})" if shape else name
            for name, shape in compiled_lambdas
        )
        diagnostics.append(
            Diagnostic(
                severity="warning",
                code=Codes.IMPORT_COMPILED_LAMBDA,
                message=(
                    f"{len(compiled_lambdas)} Lambda layer(s) store their function as "
                    f"compiled code, which is not executed: a model file is not a "
                    f"place to accept code from. Type each one's expression to make "
                    f"the model run — {described}."
                ),
            )
        )

    graph = GraphIR(
        ir_version=IR_VERSION,
        name=str(config.get("name") or "Imported model"),
        nodes=nodes,
        edges=edges,
        meta={"imported_from": "keras_json"},
    )
    return graph, diagnostics
