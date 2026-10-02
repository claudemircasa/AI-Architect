//! > [!AML-DOC-FILE]
//! @file        src-tauri/src/main.rs
//! @description Executable entry point for the desktop shell.
//! @module      desktop/main
//! @exports     main
//! @created     2026-09-30
//! @context     [amm: A.1] desktop entry point. The console subsystem is suppressed
//!              on Windows release builds so no terminal flashes behind the window.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    ai_architect_lib::run()
}
