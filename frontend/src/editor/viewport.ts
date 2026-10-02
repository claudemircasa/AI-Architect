/**
 * > [!AML-DOC-FILE]
 * @file        src/editor/viewport.ts
 * @description Shares the live React Flow instance with components that sit outside
 *              the canvas, so they can place nodes where the user is looking.
 * @module      frontend/editor/viewport
 * @exports     setFlowInstance, placementPosition
 * @created     2026-09-30
 * @context     The palette is rendered outside `ReactFlowProvider`, so it cannot use
 *              `useReactFlow`. Without this, a clicked layer was placed at fixed
 *              canvas coordinates and could land outside the visible area, which
 *              looked as though the click had done nothing.
 */

/**
 * > [!AML-DOC-UNIT]
 * The only capability this module needs from React Flow. Narrowing it here avoids
 * threading the canvas's node and edge generics through every caller, which is the
 * whole reason the full `ReactFlowInstance` type is awkward to store.
 */
interface CoordinateMapper {
  screenToFlowPosition: (position: { x: number; y: number }) => { x: number; y: number };
  fitView: (options?: {
    padding?: number;
    maxZoom?: number;
    minZoom?: number;
    duration?: number;
  }) => unknown;
  getViewport: () => { x: number; y: number; zoom: number };
  setViewport: (
    viewport: { x: number; y: number; zoom: number },
    options?: { duration?: number },
  ) => unknown;
  setCenter?: (
    x: number,
    y: number,
    options?: { zoom?: number; duration?: number },
  ) => unknown;
}

let instance: CoordinateMapper | null = null;

/**
 * > [!AML-DOC-UNIT]
 * Record the canvas instance once React Flow has initialised.
 * @param value the live instance, or null on unmount
 * @sideEffects replaces the module-level reference
 */
export function setFlowInstance(value: CoordinateMapper | null): void {
  instance = value;
}

/**
 * > [!AML-DOC-UNIT]
 * Choose where a click-added layer should land: inside the visible area, stepped
 * down and right so consecutive additions form a readable column.
 * @param existingCount how many layers are already on the canvas
 * @returns canvas coordinates
 * @sideEffects none
 * @context Falls back to fixed coordinates before the canvas has initialised, which
 *          only happens on the very first render.
 */
export function placementPosition(existingCount: number): { x: number; y: number } {
  const step = { x: (existingCount % 3) * 32, y: (existingCount % 6) * 104 };
  if (!instance) return { x: 240 + step.x, y: 80 + step.y };

  const topLeft = instance.screenToFlowPosition({ x: 0, y: 0 });
  const bottomRight = instance.screenToFlowPosition({
    x: window.innerWidth,
    y: window.innerHeight,
  });
  return {
    x: topLeft.x + (bottomRight.x - topLeft.x) * 0.28 + step.x,
    y: topLeft.y + (bottomRight.y - topLeft.y) * 0.12 + step.y,
  };
}


/**
 * > [!AML-DOC-UNIT]
 * Frame the whole graph in the viewport.
 * @sideEffects animates the canvas viewport; does nothing before the canvas exists
 * @context Called after a tidy, and after a large import changes what is drawn.
 *          The zoom has a floor: a graph taller than the window cannot be fitted
 *          without shrinking its boxes past legibility, so below that floor the view
 *          anchors at the top and the graph is scrolled instead.
 */
export function fitGraphInView(): void {
  instance?.fitView({ padding: 0.2, maxZoom: 1, minZoom: 0.45, duration: 280 });
}


/**
 * > [!AML-DOC-UNIT]
 * Read the canvas's current pan and zoom.
 * @returns the viewport, or the identity view before the canvas exists
 * @sideEffects none
 */
export function currentViewport(): { x: number; y: number; zoom: number } {
  return instance?.getViewport() ?? { x: 0, y: 0, zoom: 1 };
}

/**
 * > [!AML-DOC-UNIT]
 * Restore a saved pan and zoom.
 * @param viewport the view to apply
 * @sideEffects moves the canvas; does nothing before it exists
 */
export function applyViewport(viewport: { x: number; y: number; zoom: number }): void {
  instance?.setViewport(viewport, { duration: 200 });
}


/**
 * > [!AML-DOC-UNIT]
 * Bring a point of the canvas into the middle of the window.
 * @param position where to centre on
 * @param zoom     zoom to settle at
 * @sideEffects moves the canvas; does nothing before it exists
 * @context Used when a folded box is opened. The layers inside it land wherever the
 *          layout puts them, which may be off-screen, and a click that appears to do
 *          nothing is worse than no click at all.
 */
export function centreOn(position: { x: number; y: number }, zoom = 0.8): void {
  instance?.setCenter?.(position.x, position.y, { zoom, duration: 300 });
}

/**
 * > [!AML-DOC-UNIT]
 * Zoom the canvas about its centre.
 * @param factor more than one zooms in, less than one zooms out
 * @sideEffects moves the canvas viewport
 * @context Clamped to the same range the canvas itself allows, so a menu command
 *          cannot reach a zoom the scroll wheel refuses to [E-022].
 */
export function zoomCanvasBy(factor: number): void {
  if (!instance) return;
  const current = instance.getViewport();
  const zoom = Math.min(2.5, Math.max(0.05, current.zoom * factor));
  // Keep the centre of the view where it is: the canvas is in screen coordinates,
  // so holding the midpoint fixed means scaling the offset about it.
  const centre = { x: window.innerWidth / 2, y: window.innerHeight / 2 };
  instance.setViewport(
    {
      zoom,
      x: centre.x - ((centre.x - current.x) * zoom) / current.zoom,
      y: centre.y - ((centre.y - current.y) * zoom) / current.zoom,
    },
    { duration: 120 },
  );
}
