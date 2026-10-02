# AI Architect

Design neural network architectures visually, watch real data move through them layer
by layer, and export a standalone trainable TensorFlow project.

The editor is a desktop application. It carries a Python engine that compiles every
graph with Keras itself — shapes, parameter counts and activations are what TensorFlow
reports, not what a diagram guesses.

![The editor, with a 76-layer model imported from a .keras file. Each layer draws what
it just produced; the panel on the right maps the model's eight outputs to the
vocabularies in its own labels file.](docs/screenshots/editor-canvas.png)

---

## What it does

- **154 layers.** 112 stock Keras layers, derived from their constructors so the
  parameter list is the real one, plus 42 implementations of research-frontier
  architectures — state-space models, KAN variants, modern attention, mixture of
  experts, spiking and graph layers — each citing the paper it comes from.
- **A visual editor** with live validation. Shape errors appear on the layer that
  caused them, as the engine's own message.
- **Analytical graph views** — a layered dataflow diagram, tensor volumes, and a
  connectivity matrix — for reading a model's structure rather than editing it.
- **Real activations.** Feed an image, a recording, a tensor or a built-in dataset and
  see what each layer actually produced: feature maps, attention matrices,
  spectrograms, class scores.
- **Runs over time.** For architectures that carry state, step through a recording with
  the feedback loops closed, and see whether cutting them changes anything.
- **An architecture optimiser** that proposes changes and measures each by building and
  timing it, keeping exact rewrites apart from substitutions that need retraining.
- **Training in the app**, with live loss and accuracy curves.
- **Export** to a folder of plain Python that trains on its own. Custom layers travel
  as source; nothing imports this tool.
- **Import** somebody else's model from a `.keras` archive, a `.tflite` file or
  `to_json()` output.

![The same model in the graph view, sized by tensor volume. Circle area is the
parameter count, edge thickness is how many values cross, and a dashed edge is a
connection that skips over a layer.](docs/screenshots/graph-view-tensor-volume.png)

---

## Requirements

| | Version | Why |
|---|---|---|
| Python | **3.13.x exactly** | TensorFlow 2.21 publishes no wheel for 3.14 or newer. |
| Node | 20 or newer | Vite 6 and the Tauri CLI. |
| Rust | 1.90 or newer | Required by Tauri 2.12. `rustup update stable`. |
| macOS | 13 or newer | The desktop shell and the packaging script are macOS-only today. The engine and the editor are not. |

A packaged build needs none of this: it downloads its own Python and TensorFlow on
first launch.

---

## Running from source

### 1. Set up the engine

```bash
./scripts/setup-backend.sh
```

Creates `backend/.venv` with TensorFlow, Keras, FastAPI and the engine itself,
installed in editable mode. It refuses to continue on the wrong Python rather than
failing later with a confusing error.

To point it at a particular interpreter:

```bash
PYTHON_BIN="$HOME/.pyenv/versions/3.13.1/bin/python" ./scripts/setup-backend.sh
```

### 2. Start the engine

```bash
./scripts/run-engine.sh
```

Serves on `http://127.0.0.1:8756`. The first request pays for importing TensorFlow,
which takes several seconds; `GET /ready` waits for it rather than answering "not
yet".

### 3. Start the editor

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. The editor talks to the engine on 8756.

---

## Building the desktop application

```bash
cd desktop && npm install
cd .. && ./scripts/package-app.sh
open "desktop/src-tauri/target/release/bundle/macos/AI Architect.app"
```

Produces an `.app` of about 8 MB and a disk image of about 3 MB. The application does
not carry Python or TensorFlow — a gigabyte is too much to put in a download most
people would never unpack. A first launch opens on a setup screen and installs them
into `~/Library/Application Support/ltd.governor.ai-architect/runtime`, outside the
bundle, so they survive replacing the application. **Backend Components…** in the
application menu shows what is installed and reinstalls it.

> **Testing a first launch means leaving this directory.** The shell falls back to
> `backend/.venv` by walking up from its own location, so a bundle sitting inside this
> repository finds the development engine and never shows the setup screen. Copy the
> `.app` elsewhere first.

To build the runtime ahead of time instead of downloading it at launch:

```bash
./scripts/bundle-runtime.sh                 # into desktop/src-tauri/runtime
./scripts/install-runtime.sh <dir> backend  # into a directory of your choosing
```

For development with hot reloading:

```bash
cd desktop && npm run tauri dev
```

> A `cargo build` debug binary loads the interface from the dev server at
> `localhost:5173` and shows a blank window without it. That is the debug profile, not
> a fault; the debug binary now says so at startup. Use the release build above, or
> start `npm run dev` first.

---

## Tests

```bash
backend/.venv/bin/python -m pytest backend/tests -q
```

88 tests covering the catalog, the graph compiler, code generation, the research
layers, project persistence, multi-input and multi-output models, audio input, output
labelling, runs over time, and the optimiser.

The editor is checked in a real browser, because layout and geometry cannot be checked
any other way. With the engine and the dev server running:

```bash
cd frontend
npm run ui-check          # the editor canvas
npm run graphview-check   # zoom, pan and layout of the graph view
npm run timepanel-check   # the Time panel's layout
```

Each uses the installed Google Chrome through `playwright-core`; no browser is
downloaded. Each writes screenshots — read them.

---

## How it fits together

```
backend/                 the engine: FastAPI over a Keras compiler
  src/nnarch/catalog/      layer specs, derived from Keras constructors
  src/nnarch/ir/           graph schema, validation, compiler, import/export
  src/nnarch/layers/       the research layers, one module per family
  src/nnarch/export/       code generation into a standalone project
  src/nnarch/viz/          activations, runs over time
  src/nnarch/optimize/     rewrite rules and their measurement
  src/nnarch/data/         dataset adapters, uploads, output names
  src/nnarch/api/          HTTP and WebSocket routes
  tests/
frontend/                the editor: React, TypeScript, Vite, React Flow
  src/editor/              palette, canvas, property panel, optimiser
  src/graphview/           the analytical views
  src/viz/                 activations, metrics, data and time panels
  src/shell/               the bridge to the desktop shell
  scripts/                 browser-driven checks
desktop/src-tauri/       the Tauri shell: engine lifecycle, native menus, window
scripts/                 setup, run, runtime installation, packaging
```

### The decision that shapes everything

The engine never reimplements Keras. A layer's parameters come from its constructor
signature, its output shape comes from building it symbolically, and its activations
come from running it. When TensorFlow changes, the catalog changes with it.

The same rule governs export: the compiler and the code generator read one shared
table describing how each layer is called, so the model that runs in the editor and
the model the exported file builds cannot drift apart.

---

## Importing a model

| Format | What comes back |
|---|---|
| `.keras` | The architecture, functional or sequential, rebuilt parameter for parameter. Weights are not imported — this edits architectures, and weights would not survive the first change. |
| `.json` | The same, from `Model.to_json()`. |
| `.tflite` | The *converted* graph. Layers, order and shapes are faithful; strides, padding and pool sizes live in the file's binary options and come back at their defaults. |

Two things an import cannot bring back, and says so. A layer the catalog does not have
arrives as a pass-through, keeping its original class name and settings on the node so
you can see where it was. And a `Lambda` stores its function as compiled code, which is
not executed: a model file is not a place to accept code from. Each one is listed with
the output shape its expression has to produce.

---

## Contributing

Contributions are welcome under the licence below.

### About the docblocks

Nearly every file and function here carries a structured comment that looks like this:

```python
"""
> [!AML-DOC-UNIT]
Whether a layer provably does nothing to its input.
@param node the layer
@returns True when removing it cannot change the model's output
@sideEffects none
@context Each case is one a model picks up by being built and edited rather than
         by being written badly: an Identity left behind by an import, a Dropout
         turned down to nothing, an activation set to linear.
"""
```

This is **AML**, a documentation format proprietary to Governor Ltda, applied across
147 files here. **You do not need to learn it, and nothing is enforced.** Write new
code's comments however you prefer. If you would rather match the surrounding style,
copy a header from any nearby file and change its contents — that is all anyone does.

Two things worth knowing if you read the existing comments:

- Bracketed codes such as `[E-042]`, `[amm: E.4]` or `[task 06]` point at Governor's
  internal engineering notes, which are not part of this repository. They are
  provenance, not instructions. Ignore them.
- The `@context` lines are the ones worth reading. They say *why* something is the way
  it is, usually because the obvious alternative was tried first and failed.

---

## License

Copyright (c) 2026 Governor Ltda. All rights reserved.

Proprietary, with collaboration permitted. You may use, study, modify and share this
software for personal, educational, academic and research purposes, and contribute
changes back. **Commercial exploitation is reserved to Governor Ltda** and requires
prior written authorisation. See [LICENSE](LICENSE).

For commercial authorisation: [governor.ltd](https://governor.ltd)
