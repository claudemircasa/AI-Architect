//! > [!AML-DOC-FILE]
//! @file        src-tauri/src/menu.rs
//! @description The application's native menu bar, and the events it sends the editor.
//! @module      desktop/menu
//! @exports     build, ITEM_EVENT
//! @created     2026-10-01
//! @context     The commands a desktop application is expected to have — New, Open,
//!              Save, Undo, the view switch, Help — belong in the system menu bar,
//!              where the platform puts them and where their keyboard shortcuts are
//!              discoverable. The toolbar keeps only the quick tools, as icons.
//!
//!              Every custom item here does nothing on its own: it emits its id to
//!              the web view, which owns the graph and is the only place that knows
//!              what "Undo" currently means. The menu is a way in, not a second
//!              implementation [E-029].

use tauri::menu::{
    AboutMetadata, Menu, MenuItem, PredefinedMenuItem, Submenu, SubmenuBuilder,
};
use tauri::{AppHandle, Runtime};

/// Event name carrying the id of whichever menu item was chosen.
pub const ITEM_EVENT: &str = "menu://item";

/// > [!AML-DOC-UNIT]
/// One custom item, with its accelerator.
///
/// @param app   handle the item belongs to
/// @param id    the identifier the web view receives
/// @param label the text shown in the menu
/// @param key   accelerator such as `CmdOrCtrl+S`, or none
/// @returns the item
/// @raises tauri::Error when the item cannot be created
fn item<R: Runtime>(
    app: &AppHandle<R>,
    id: &str,
    label: &str,
    key: Option<&str>,
) -> tauri::Result<MenuItem<R>> {
    MenuItem::with_id(app, id, label, true, key)
}

/// > [!AML-DOC-UNIT]
/// Build the whole menu bar.
///
/// @param app handle to build against
/// @returns the menu, ready to be set on the application
/// @raises tauri::Error when any item cannot be created
/// @sideEffects none; setting it is the caller's business
/// @context The first submenu is the application menu, which macOS names after the
///          bundle whatever is put in it, and which must carry Quit and the window
///          commands or the standard shortcuts stop working. Cut, Copy, Paste and
///          Select All are the predefined items rather than custom ones, because the
///          text fields in the property panel need the real platform behaviour.
pub fn build<R: Runtime>(app: &AppHandle<R>) -> tauri::Result<Menu<R>> {
    // The About panel is where the people who made it are named, so it carries the
    // copyright, the contact and the licences rather than a version number alone.
    let about = PredefinedMenuItem::about(
        app,
        Some("About AI Architect"),
        Some(AboutMetadata {
            version: Some(env!("CARGO_PKG_VERSION").into()),
            comments: Some(
                "Visual prototyping for TensorFlow architectures.\n\n\
                 Design neural network architectures, watch real data move through \
                 them, and export standalone trainable projects."
                    .into(),
            ),
            copyright: Some("© 2026 Governor Ltda. All rights reserved.".into()),
            authors: Some(vec!["Governor Ltda".into()]),
            website: Some("https://governor.ltd".into()),
            website_label: Some("governor.ltd".into()),
            ..Default::default()
        }),
    )?;

    let app_menu = Submenu::with_items(
        app,
        "AI Architect",
        true,
        &[
            &about,
            &PredefinedMenuItem::separator(app)?,
            &item(app, "app.backend", "Backend Components…", None)?,
            &PredefinedMenuItem::separator(app)?,
            &PredefinedMenuItem::hide(app, None)?,
            &PredefinedMenuItem::hide_others(app, None)?,
            &PredefinedMenuItem::show_all(app, None)?,
            &PredefinedMenuItem::separator(app)?,
            &PredefinedMenuItem::quit(app, None)?,
        ],
    )?;

    let file = Submenu::with_items(
        app,
        "File",
        true,
        &[
            &item(app, "file.new", "New Project", Some("CmdOrCtrl+N"))?,
            &item(app, "file.open", "Open Project…", Some("CmdOrCtrl+O"))?,
            &PredefinedMenuItem::separator(app)?,
            &item(app, "file.save", "Save", Some("CmdOrCtrl+S"))?,
            &item(app, "file.saveAs", "Save As…", Some("Shift+CmdOrCtrl+S"))?,
            &PredefinedMenuItem::separator(app)?,
            &item(app, "file.import", "Import Model…", Some("CmdOrCtrl+I"))?,
            &item(app, "file.export", "Export Trainable Project…", Some("CmdOrCtrl+E"))?,
            &PredefinedMenuItem::separator(app)?,
            &PredefinedMenuItem::close_window(app, None)?,
        ],
    )?;

    let edit = Submenu::with_items(
        app,
        "Edit",
        true,
        &[
            &item(app, "edit.undo", "Undo", Some("CmdOrCtrl+Z"))?,
            &item(app, "edit.redo", "Redo", Some("Shift+CmdOrCtrl+Z"))?,
            &PredefinedMenuItem::separator(app)?,
            &PredefinedMenuItem::cut(app, None)?,
            &PredefinedMenuItem::copy(app, None)?,
            &PredefinedMenuItem::paste(app, None)?,
            &PredefinedMenuItem::select_all(app, None)?,
            &PredefinedMenuItem::separator(app)?,
            &item(app, "edit.delete", "Delete Selected Layers", Some("Backspace"))?,
            &item(app, "edit.duplicate", "Duplicate", Some("CmdOrCtrl+D"))?,
            &PredefinedMenuItem::separator(app)?,
            &item(app, "model.optimize", "Optimise Architecture…", None)?,
        ],
    )?;

    let view = Submenu::with_items(
        app,
        "View",
        true,
        &[
            &item(app, "view.architecture", "Architecture", Some("CmdOrCtrl+1"))?,
            &item(app, "view.graph", "Graph", Some("CmdOrCtrl+2"))?,
            &PredefinedMenuItem::separator(app)?,
            &item(app, "view.tidy", "Tidy Layout", Some("CmdOrCtrl+L"))?,
            &item(app, "view.fit", "Fit to Window", Some("CmdOrCtrl+0"))?,
            &item(app, "view.zoomIn", "Zoom In", Some("CmdOrCtrl+Plus"))?,
            &item(app, "view.zoomOut", "Zoom Out", Some("CmdOrCtrl+-"))?,
            &PredefinedMenuItem::separator(app)?,
            &PredefinedMenuItem::fullscreen(app, None)?,
        ],
    )?;

    let window = SubmenuBuilder::new(app, "Window")
        .minimize()
        .maximize()
        .separator()
        .close_window()
        .build()?;

    let help = Submenu::with_items(
        app,
        "Help",
        true,
        &[
            &item(app, "help.guide", "AI Architect Guide", None)?,
            &PredefinedMenuItem::separator(app)?,
            &item(app, "help.keras", "Keras Layer Reference", None)?,
            &item(app, "help.tensorflow", "TensorFlow Guide", None)?,
            &PredefinedMenuItem::separator(app)?,
            &item(app, "help.governor", "Governor Ltda — governor.ltd", None)?,
        ],
    )?;

    Menu::with_items(app, &[&app_menu, &file, &edit, &view, &window, &help])
}
