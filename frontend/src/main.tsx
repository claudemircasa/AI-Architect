/**
 * > [!AML-DOC-FILE]
 * @file        src/main.tsx
 * @description Mounts the React application.
 * @module      frontend/main
 * @exports     (entry point)
 * @created     2026-09-30
 * @context     [amm: A.2] frontend entry point.
 */

import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import { App } from "./App";
import "./index.css";

const container = document.getElementById("root");
if (!container) throw new Error("#root is missing from index.html");

createRoot(container).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
