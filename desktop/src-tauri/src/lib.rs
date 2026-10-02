//! > [!AML-DOC-FILE]
//! @file        src-tauri/src/lib.rs
//! @description Desktop shell: owns the backend engine's lifetime, waits for its
//!              readiness handshake, and hands the port to the web view.
//! @module      desktop/lib
//! @exports     run, EngineState, engine_status
//! @created     2026-09-30
//! @context     [amm: E.5] the shell spawns the engine, reads `NNARCH_READY port=`
//!              from its stdout, and kills it when the app exits. An installed copy
//!              runs the Python runtime bundled in its own `Resources`, so it needs
//!              nothing on the machine; a development build falls back to the venv
//!              that `setup-backend.sh` creates [task 08].

use std::io::{BufRead, BufReader};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use serde::Serialize;
use tauri::{Emitter, Manager, WindowEvent};

mod menu;

/// Seconds to wait for the engine to print its readiness line before giving up.
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(20);

/// Prefix the engine prints on stdout once it is listening.
const READY_PREFIX: &str = "NNARCH_READY port=";

/// The interpreter inside a runtime directory. The real file is named for its
/// version: `python3` beside it is a symlink, and a symlink is the first thing a copy
/// between file systems is liable to lose.
const PYTHON_IN_RUNTIME: &str = "python/bin/python3.13";

/// Where an installed runtime lives, under the app's own data directory. Keeping it
/// out of the `.app` is what lets the installer be seven megabytes instead of three
/// hundred, and it survives replacing the application [E-045].
const RUNTIME_DIR: &str = "runtime";

/// Path of the development engine executable, relative to the project root.
const ENGINE_RELATIVE: &str = "backend/.venv/bin/nnarch-engine";

/// How to start the engine: a program and the arguments that precede `--port`.
#[derive(Clone, Debug)]
struct EngineCommand {
    program: PathBuf,
    args: Vec<String>,
}

impl EngineCommand {
    /// > [!AML-DOC-UNIT]
    /// The bundled runtime, run as a module.
    ///
    /// @param python the interpreter inside the app's resources
    /// @returns the command that starts the engine from it
    /// @context Not the `nnarch-engine` console script beside it: pip writes the
    ///          interpreter's path into that script's shebang when it installs, and
    ///          that path stops existing the moment the runtime is copied into a
    ///          bundle. `-m nnarch` depends only on the interpreter being invoked,
    ///          which is the one thing the shell knows for certain [E-026].
    fn bundled(python: PathBuf) -> Self {
        Self { program: python, args: vec!["-m".into(), "nnarch".into()] }
    }

    /// > [!AML-DOC-UNIT]
    /// The development venv's console script.
    ///
    /// @param engine path to `backend/.venv/bin/nnarch-engine`
    /// @returns the command that starts it
    fn console_script(engine: PathBuf) -> Self {
        Self { program: engine, args: Vec::new() }
    }
}

/// What the web view is told about the engine.
#[derive(Clone, Debug, Serialize)]
pub struct EngineStatus {
    /// Port the engine is listening on, when it started.
    pub port: Option<u16>,
    /// Whether the shell managed to start it.
    pub running: bool,
    /// Human-readable reason it could not be started.
    pub error: Option<String>,
    /// Where the shell looked for the engine, shown in the onboarding panel.
    pub searched: Vec<String>,
    /// True when nothing is installed yet and the setup screen should be shown.
    pub needs_setup: bool,
    /// What the installed runtime reports about itself, once there is one.
    pub runtime: Option<String>,
}

/// Holds the spawned engine so it can be killed when the app closes.
pub struct EngineState {
    child: Mutex<Option<Child>>,
    status: EngineStatus,
}

impl EngineState {
    /// > [!AML-DOC-UNIT]
    /// Stop the engine if the shell started one.
    ///
    /// @sideEffects kills the child process; safe to call more than once
    /// @context Without this, closing the window leaves a `uvicorn` process holding
    ///          port 8756, which makes the next launch fail to bind [task 08 gate].
    pub fn shutdown(&self) {
        if let Ok(mut guard) = self.child.lock() {
            if let Some(mut child) = guard.take() {
                let _ = child.kill();
                let _ = child.wait();
            }
        }
    }
}

/// > [!AML-DOC-UNIT]
/// Walk upward from a starting directory looking for the engine executable.
///
/// @param start directory to begin from, typically the executable's own directory
/// @returns the engine path and every directory that was tried
/// @sideEffects none; only reads directory entries
/// @context In development the binary sits under `desktop/src-tauri/target`, and in
///          a bundle it sits inside the `.app`, so neither can assume a fixed depth.
fn find_engine(
    data_dir: &Path,
    resources: &Path,
    start: &Path,
) -> (Option<EngineCommand>, Vec<String>) {
    let mut searched = Vec::new();

    // What the app installed for itself, which is the only branch that ever matches
    // for somebody who downloaded the application and nothing else.
    let installed = data_dir.join(RUNTIME_DIR).join(PYTHON_IN_RUNTIME);
    searched.push(installed.display().to_string());
    if installed.is_file() {
        return (Some(EngineCommand::bundled(installed)), searched);
    }

    // A build that chose to carry its runtime inside the bundle instead.
    let bundled = resources.join(RUNTIME_DIR).join(PYTHON_IN_RUNTIME);
    searched.push(bundled.display().to_string());
    if bundled.is_file() {
        return (Some(EngineCommand::bundled(bundled)), searched);
    }

    let mut current = Some(start);
    while let Some(directory) = current {
        let candidate = directory.join(ENGINE_RELATIVE);
        searched.push(candidate.display().to_string());
        if candidate.is_file() {
            return (Some(EngineCommand::console_script(candidate)), searched);
        }
        current = directory.parent();
    }
    (None, searched)
}

/// > [!AML-DOC-UNIT]
/// Ask the operating system for a port nothing is using.
///
/// @returns a free TCP port on the loopback interface
/// @raises never; falls back to the engine's own default when probing fails
/// @sideEffects binds and immediately releases a socket
/// @context Letting the shell choose means a developer who already has an engine
///          running by hand on 8756 can still open the app, and it makes the port
///          injection genuinely load-bearing rather than always agreeing with the
///          frontend's fallback.
fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .and_then(|listener| listener.local_addr())
        .map(|address| address.port())
        .unwrap_or(8756)
}

/// > [!AML-DOC-UNIT]
/// Start the engine and block until it announces the port it bound.
///
/// @param engine path to the engine executable
/// @returns the running child and the port it reported, or the reason it failed
/// @raises never panics; every failure becomes an `Err(String)` the onboarding panel
///         can show
/// @sideEffects spawns a process and reads its stdout until the handshake arrives;
///              the child is told this process's id so it can outlive nothing
fn spawn_engine(engine: &EngineCommand) -> Result<(Child, u16), String> {
    let port = free_port();
    let mut child = Command::new(&engine.program)
        .args(&engine.args)
        .arg("--port")
        .arg(port.to_string())
        // The engine exits by itself if this process disappears, so a crash or a
        // force-quit cannot leave it holding the port. A clean quit still kills it
        // directly, through `EngineState::shutdown`.
        .env("NNARCH_PARENT_PID", std::process::id().to_string())
        .stdout(Stdio::piped())
        // Inherited rather than piped: a piped stderr nobody reads fills its 64KB
        // buffer and then blocks the engine mid-log. Inheriting also puts uvicorn's
        // access log wherever the shell's own output goes, which is what you want
        // when diagnosing a launch.
        .stderr(Stdio::inherit())
        .spawn()
        .map_err(|error| format!("could not start {}: {error}", engine.program.display()))?;

    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| "the engine produced no stdout".to_string())?;

    let deadline = Instant::now() + HANDSHAKE_TIMEOUT;
    let mut reader = BufReader::new(stdout);
    let mut line = String::new();

    while Instant::now() < deadline {
        line.clear();
        match reader.read_line(&mut line) {
            Ok(0) => break,
            Ok(_) => {
                if let Some(rest) = line.trim().strip_prefix(READY_PREFIX) {
                    if let Ok(port) = rest.trim().parse::<u16>() {
                        // Keep draining stdout so a full pipe buffer cannot block
                        // the engine once the app is running.
                        std::thread::spawn(move || {
                            let mut sink = String::new();
                            while reader.read_line(&mut sink).unwrap_or(0) > 0 {
                                sink.clear();
                            }
                        });
                        return Ok((child, port));
                    }
                }
            }
            Err(error) => return Err(format!("could not read the engine's output: {error}")),
        }
    }

    let _ = child.kill();
    Err(format!(
        "the engine did not report a port within {} seconds",
        HANDSHAKE_TIMEOUT.as_secs()
    ))
}

/// > [!AML-DOC-UNIT]
/// Report the engine's state to the web view.
///
/// @param state the shell's engine state
/// @returns the status, including where the engine was looked for when it is missing
/// Largest window the editor is designed for, in logical pixels.
const PREFERRED_WIDTH: f64 = 1600.0;
const PREFERRED_HEIGHT: f64 = 1000.0;

/// How much of the screen to take when the preferred size does not fit.
const SCREEN_FRACTION: f64 = 0.92;

/// > [!AML-DOC-UNIT]
/// An opening size that fits the display it will open on.
///
/// @param app the application handle, for the monitor
/// @returns width and height in logical pixels
/// @sideEffects none
/// @context A monitor's size is reported in physical pixels, and a window is asked
///          for in logical ones, so the scale factor has to be divided out or a
///          Retina display reports twice the room it has. When no monitor can be
///          read, the preferred size is used: being wrong about an absent display is
///          better than opening something tiny.
fn preferred_size<R: tauri::Runtime>(app: &tauri::AppHandle<R>) -> (f64, f64) {
    let Ok(Some(monitor)) = app.primary_monitor() else {
        return (PREFERRED_WIDTH, PREFERRED_HEIGHT);
    };
    let scale = monitor.scale_factor();
    let size = monitor.size();
    let available_width = f64::from(size.width) / scale * SCREEN_FRACTION;
    let available_height = f64::from(size.height) / scale * SCREEN_FRACTION;
    (
        PREFERRED_WIDTH.min(available_width).max(640.0),
        PREFERRED_HEIGHT.min(available_height).max(480.0),
    )
}

#[tauri::command]
fn engine_status(state: tauri::State<'_, EngineState>) -> EngineStatus {
    state.status.clone()
}

/// > [!AML-DOC-UNIT]
/// Write a line from the web view into the shell's log.
///
/// @param level   "info", "warn" or "error"
/// @param message the text to record
/// @sideEffects writes to the shell's stderr, alongside the engine's own output
/// @context The web view has no console a user can reach, so a frontend failure
///          inside the packaged app would otherwise be invisible. Routing it here
///          puts it in the same log as the engine's.
/// Event carrying one line of the installer's progress to the setup screen.
pub const SETUP_EVENT: &str = "setup://progress";

/// > [!AML-DOC-UNIT]
/// Download and install the engine's Python runtime.
///
/// @param app the application handle, for its paths and for emitting progress
/// @returns the installed interpreter's path, or the reason it could not be installed
/// @sideEffects downloads about a gigabyte, writes it under the app's data directory,
///              and emits a `setup://progress` event per step
/// @context Runs the bundled script rather than reimplementing it: downloading,
///          unpacking and running pip are things a shell does well, and this is the
///          same sequence the build-time script already proves out [E-045]. The
///          script installs beside the target and moves it into place at the end, so
///          an interrupted download cannot leave a half-built runtime behind.
#[tauri::command]
async fn install_runtime(app: tauri::AppHandle) -> Result<String, String> {
    let resources = app.path().resource_dir().map_err(|e| e.to_string())?;
    let data_dir = app.path().app_data_dir().map_err(|e| e.to_string())?;
    std::fs::create_dir_all(&data_dir).map_err(|e| e.to_string())?;

    let script = resources.join("install-runtime.sh");
    let engine_src = resources.join("engine-src");
    if !script.is_file() {
        return Err(format!("the installer is missing from the app: {}", script.display()));
    }

    let target = data_dir.join(RUNTIME_DIR);
    let mut child = Command::new("/bin/bash")
        .arg(&script)
        .arg(&target)
        .arg(&engine_src)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| format!("could not start the installer: {error}"))?;

    let stdout = child.stdout.take().ok_or("the installer produced no output")?;
    let emitter = app.clone();
    let reader = std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            if let Some(step) = line.strip_prefix("STEP ") {
                let _ = emitter.emit(SETUP_EVENT, step.to_string());
            }
            eprintln!("[nnarch:setup] {line}");
        }
    });

    // Drained rather than inherited: pip writes a great deal here, and an undrained
    // pipe fills its buffer and stops the process mid-install [E-007].
    let stderr = child.stderr.take();
    let tail = std::thread::spawn(move || {
        let mut last = String::new();
        if let Some(stream) = stderr {
            for line in BufReader::new(stream).lines().map_while(Result::ok) {
                eprintln!("[nnarch:setup] {line}");
                if !line.trim().is_empty() {
                    last = line;
                }
            }
        }
        last
    });

    let finished = child.wait().map_err(|error| error.to_string())?;
    let _ = reader.join();
    let last_error = tail.join().unwrap_or_default();

    if !finished.success() {
        return Err(if last_error.is_empty() {
            format!("the installer stopped with status {finished}")
        } else {
            last_error
        });
    }

    let python = target.join(PYTHON_IN_RUNTIME);
    if !python.is_file() {
        return Err("the installer finished but left no interpreter".to_string());
    }
    Ok(python.display().to_string())
}

/// > [!AML-DOC-UNIT]
/// Restart the application, which is how a freshly installed runtime is picked up.
///
/// @param app the application handle
/// @sideEffects stops the engine and relaunches the process
/// @context The engine is started once, at setup, so the simplest honest way to use
///          a runtime that did not exist then is to start again.
#[tauri::command]
fn relaunch(app: tauri::AppHandle) {
    if let Some(state) = app.try_state::<EngineState>() {
        state.shutdown();
    }
    app.restart();
}

#[tauri::command]
fn log_message(level: String, message: String) {
    eprintln!("[webview {level}] {message}");
}

/// > [!AML-DOC-UNIT]
/// Warn loudly when a debug build is pointed at a dev server that is not running.
///
/// @sideEffects writes to stderr
/// @context A debug build loads `build.devUrl`, and if nothing is serving it the web
///          view renders an empty page with no error anywhere. A blank window is the
///          least diagnosable failure this app can have, so it says so instead.
///          Release builds serve the bundled frontend and never reach this.
#[cfg(debug_assertions)]
fn warn_if_dev_server_missing() {
    use std::net::TcpStream;
    use std::time::Duration;

    const DEV_SERVER: &str = "127.0.0.1:5173";

    let reachable = "localhost:5173"
        .parse()
        .ok()
        .and_then(|address| TcpStream::connect_timeout(&address, Duration::from_millis(400)).ok())
        .is_some()
        || DEV_SERVER
            .parse()
            .ok()
            .and_then(|address| {
                TcpStream::connect_timeout(&address, Duration::from_millis(400)).ok()
            })
            .is_some();

    if !reachable {
        eprintln!(
            "\n\
             ──────────────────────────────────────────────────────────────────\n\
             AI Architect: this is a DEBUG build, which loads the interface from\n\
             the dev server at http://localhost:5173 — and nothing is listening\n\
             there, so the window will be blank.\n\
             \n\
             Start the dev server in another terminal:\n\
               cd frontend && npm run dev\n\
             \n\
             Or build the standalone app, which carries the interface inside it:\n\
               cd desktop && npm run tauri build\n\
             ──────────────────────────────────────────────────────────────────\n"
        );
    }
}

/// > [!AML-DOC-UNIT]
/// Build and run the desktop application.
///
/// @sideEffects starts the engine, opens the window, and stops the engine on exit
/// @context The engine is started before the window is created so the port can be
///          injected as `window.__NNARCH_PORT__`, which is what the frontend's API
///          client reads. A failure here is not fatal: the window still opens and
///          shows the onboarding panel [task 08 item 3].
pub fn run() {
    #[cfg(debug_assertions)]
    warn_if_dev_server_missing();

    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_store::Builder::new().build())
        .plugin(tauri_plugin_http::init())
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![
            engine_status,
            log_message,
            install_runtime,
            relaunch
        ])
        .setup(|app| {
            // The menu bar is built here rather than declared in the configuration
            // because its items need ids the web view can act on, and because macOS
            // wants the application submenu first [E-029].
            let handle = app.handle();
            app.set_menu(menu::build(handle)?)?;
            app.on_menu_event(|app, event| {
                let id = event.id().0.clone();
                if let Err(error) = app.emit(menu::ITEM_EVENT, &id) {
                    eprintln!("[nnarch] could not deliver menu item {id}: {error}");
                }
            });

            let resources = app.path().resource_dir().unwrap_or_else(|_| PathBuf::from("."));
            let start = std::env::current_exe()
                .ok()
                .and_then(|exe| exe.parent().map(Path::to_path_buf))
                .unwrap_or_else(|| resources.clone());

            let data_dir = app
                .path()
                .app_data_dir()
                .unwrap_or_else(|_| PathBuf::from("."));
            let _ = std::fs::create_dir_all(&data_dir);

            let (found, searched) = find_engine(&data_dir, &resources, &start);
            let (child, status) = match found {
                // Nothing installed yet. This is the ordinary first launch, not a
                // failure, so it is reported as something to do rather than as an
                // error: the window opens on the setup screen [E-045].
                None => (
                    None,
                    EngineStatus {
                        port: None,
                        running: false,
                        error: None,
                        searched,
                        needs_setup: true,
                        runtime: None,
                    },
                ),
                Some(engine) => match spawn_engine(&engine) {
                    Ok((child, port)) => (
                        Some(child),
                        EngineStatus {
                            port: Some(port),
                            running: true,
                            error: None,
                            searched: vec![engine.program.display().to_string()],
                            needs_setup: false,
                            runtime: Some(engine.program.display().to_string()),
                        },
                    ),
                    Err(error) => (
                        None,
                        EngineStatus {
                            port: None,
                            running: false,
                            error: Some(error),
                            searched: vec![engine.program.display().to_string()],
                            needs_setup: false,
                            runtime: Some(engine.program.display().to_string()),
                        },
                    ),
                },
            };

            app.manage(EngineState {
                child: Mutex::new(child),
                status,
            });

            // The window is created here rather than declared in tauri.conf.json so
            // the engine is already running, and its port already known, before any
            // frontend code executes. The frontend reads that port by calling the
            // `engine_status` command.
            // Asking for a window larger than the display does not give a large
            // window: it gives one the size of the screen, placed by a `center()`
            // that computed a negative origin, so it opens looking full screen and
            // pushed off to one side [E-051]. The size is fitted to the monitor
            // before it is asked for.
            let (width, height) = preferred_size(app.handle());

            tauri::WebviewWindowBuilder::new(
                app.handle(),
                "main",
                tauri::WebviewUrl::default(),
            )
            .title("AI Architect")
            .inner_size(width, height)
            .min_inner_size(900.0, 600.0)
            .resizable(true)
            .center()
            .theme(Some(tauri::Theme::Dark))
            .build()?;

            Ok(())
        })
        .on_window_event(|window, event| {
            if let WindowEvent::Destroyed = event {
                window.state::<EngineState>().shutdown();
            }
        })
        .build(tauri::generate_context!())
        .expect("the desktop shell could not start")
        .run(|app, event| {
            if let tauri::RunEvent::ExitRequested { .. } | tauri::RunEvent::Exit = event {
                app.state::<EngineState>().shutdown();
            }
        });
}
