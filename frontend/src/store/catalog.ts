/**
 * > [!AML-DOC-FILE]
 * @file        src/store/catalog.ts
 * @description Loads the layer catalog once and indexes it for the palette, the
 *              property panel and the canvas nodes.
 * @module      frontend/store/catalog
 * @exports     useCatalog, categoryAccent, CATEGORY_ACCENTS
 * @created     2026-09-30
 * @context     The catalog is the single source of truth for what can be placed and
 *              how it is edited [amm: E.2]. Nothing here knows about individual
 *              layer types, so a layer added on the backend appears with no
 *              frontend change.
 */

import { create } from "zustand";

import { getCatalog, getHealth, logToShell, resolveEngine } from "../api/client";
import type { Catalog, LayerSpec, Vocabularies } from "../api/types";

/** Engine connection state, surfaced as a badge in the status bar. */
export type EngineState =
  | "connecting"
  | "warming"
  | "ready"
  | "unreachable"
  | "failed"
  /** Nothing is installed yet: a first launch, not a fault [E-045]. */
  | "needs-setup";

interface CatalogStore {
  catalog: Catalog | null;
  specs: Map<string, LayerSpec>;
  engine: EngineState;
  engineMessage: string | null;
  engineVersions: Record<string, string | number>;
  load: () => Promise<void>;
  spec: (id: string) => LayerSpec | undefined;
  vocabularies: () => Vocabularies | null;
}

const EMPTY_VOCABULARIES: Vocabularies = {
  activations: [],
  initializers: [],
  regularizers: [],
  constraints: [],
  dtypes: [],
};

/**
 * > [!AML-DOC-UNIT]
 * Catalog store. Holds the fetched catalog, an id-keyed index of every layer spec,
 * and the engine's connection state.
 */
export const useCatalog = create<CatalogStore>((set, get) => ({
  catalog: null,
  specs: new Map(),
  engine: "connecting",
  engineMessage: null,
  engineVersions: {},

  /**
   * > [!AML-DOC-UNIT]
   * Wait for the engine to warm up, then fetch and index the catalog.
   * @returns resolves once the catalog is loaded or the engine has given up
   * @sideEffects asks the desktop shell for the engine's port, then polls `/health`
   *              until TensorFlow has loaded, and finally calls `/catalog`
   * @context `/health` is polled rather than `/ready` being awaited, so the status
   *          badge can show progress instead of the app appearing frozen for the
   *          seconds TensorFlow takes to import [task 01].
   */
  load: async () => {
    const shell = await resolveEngine();
    if (shell?.needs_setup) {
      // A first launch. There is nothing wrong, there is something to do.
      set({ engine: "needs-setup", engineMessage: null });
      return;
    }
    if (shell && !shell.running) {
      set({
        engine: "failed",
        engineMessage:
          shell.error ??
          "The desktop shell could not start the engine. Run ./scripts/setup-backend.sh",
      });
      return;
    }

    for (let attempt = 0; attempt < 240; attempt += 1) {
      try {
        const health = await getHealth();
        set({ engineVersions: health.versions });
        if (health.warm === "failed") {
          set({ engine: "failed", engineMessage: health.error });
          return;
        }
        if (health.warm === "warm") break;
        set({ engine: "warming", engineMessage: null });
      } catch (error) {
        void logToShell(
          "error",
          `health probe failed: ${error instanceof Error ? error.message : String(error)}`,
        );
        set({
          engine: "unreachable",
          engineMessage: "The engine is not responding. Start it with ./scripts/run-engine.sh",
        });
      }
      await new Promise((resolve) => setTimeout(resolve, 500));
    }

    try {
      const catalog = await getCatalog();
      const specs = new Map<string, LayerSpec>();
      for (const category of catalog.categories) {
        for (const layer of category.layers) specs.set(layer.id, layer);
      }
      set({ catalog, specs, engine: "ready", engineMessage: null });
    } catch (error) {
      set({
        engine: "failed",
        engineMessage: error instanceof Error ? error.message : String(error),
      });
    }
  },

  /**
   * > [!AML-DOC-UNIT]
   * Look up one layer spec.
   * @param id catalog layer id, e.g. "keras.Conv2D"
   * @returns the spec, or undefined before the catalog has loaded
   */
  spec: (id: string) => get().specs.get(id),

  /**
   * > [!AML-DOC-UNIT]
   * The enum vocabularies backing activation and initializer selects.
   * @returns the vocabularies, or null before the catalog has loaded
   */
  vocabularies: () => get().catalog?.vocabularies ?? EMPTY_VOCABULARIES,
}));

/** Accent colour per palette category, keyed by the category id's first segment. */
export const CATEGORY_ACCENTS: Record<string, string> = {
  core: "var(--color-cat-core)",
  convolution: "var(--color-cat-convolution)",
  pooling: "var(--color-cat-pooling)",
  recurrent: "var(--color-cat-recurrent)",
  normalization: "var(--color-cat-normalization)",
  regularization: "var(--color-cat-regularization)",
  attention: "var(--color-cat-attention)",
  reshaping: "var(--color-cat-reshaping)",
  merging: "var(--color-cat-merging)",
  activation: "var(--color-cat-activation)",
  preprocessing: "var(--color-cat-preprocessing)",
  research: "var(--color-cat-research)",
};

/**
 * > [!AML-DOC-UNIT]
 * Resolve the accent colour for a category.
 * @param category category id such as "convolution" or "research/ssm"
 * @returns a CSS colour value, falling back to the core accent
 * @sideEffects none
 * @context Research categories are namespaced (`research/kan`), so only the first
 *          segment is looked up and every research family shares one hue.
 */
export function categoryAccent(category: string): string {
  const root = category.split("/")[0] ?? "core";
  return CATEGORY_ACCENTS[root] ?? "var(--color-cat-core)";
}

// Exposed for the browser-driven UI checks in scripts/, which need to place layers
// by id rather than by clicking through the palette.
if (typeof window !== "undefined") {
  (window as unknown as { __catalogStore?: typeof useCatalog }).__catalogStore = useCatalog;
}
