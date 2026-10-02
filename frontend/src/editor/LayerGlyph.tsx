/**
 * > [!AML-DOC-FILE]
 * @file        src/editor/LayerGlyph.tsx
 * @description A small drawing of what each kind of layer does to a tensor.
 * @module      frontend/editor/LayerGlyph
 * @exports     LayerGlyph
 * @created     2026-10-02
 * @context     A sentence says what a layer is for; a picture says what it does to the
 *              shape of the thing passing through it. Convolution sliding a window,
 *              pooling collapsing a grid, attention relating every position to every
 *              other — these are spatial facts, and prose is a poor way to carry them
 *              [E-052].
 *
 *              Chosen by layer id first and by category second, so a type the catalog
 *              gains tomorrow still draws something true about its family rather than
 *              nothing at all.
 */

import type { ReactElement } from "react";

/** The palette the drawings use, taken from the same tokens the canvas uses. */
const INK = "var(--color-ink-1)";
const DIM = "var(--color-ink-2)";

/**
 * > [!AML-DOC-UNIT]
 * A grid of cells, used as the stand-in for a feature map.
 * @param x        left edge
 * @param y        top edge
 * @param cols     columns
 * @param rows     rows
 * @param cell     cell size
 * @param fill     colour
 * @param opacity  how strongly to draw it
 * @returns the cells
 */
function grid(
  x: number,
  y: number,
  cols: number,
  rows: number,
  cell: number,
  fill: string,
  opacity = 0.35,
) {
  const cells = [];
  for (let row = 0; row < rows; row += 1) {
    for (let col = 0; col < cols; col += 1) {
      cells.push(
        <rect
          key={`${row}-${col}`}
          x={x + col * cell}
          y={y + row * cell}
          width={cell - 1}
          height={cell - 1}
          fill={fill}
          opacity={opacity}
        />,
      );
    }
  }
  return cells;
}

/** One drawing per family, on a shared 96×40 stage. */
const GLYPHS: Record<string, { draw: () => ReactElement; caption: string }> = {
  convolution: {
    caption: "A window slides across the input; each position becomes one value.",
    draw: () => (
      <>
        {grid(2, 4, 6, 6, 5, INK, 0.25)}
        <rect x="2" y="4" width="14" height="14" fill="none" stroke="var(--color-accent)" strokeWidth="1.5" />
        <path d="M36 20 L50 20" stroke={DIM} strokeWidth="1" markerEnd="url(#gl-arrow)" />
        {grid(56, 7, 5, 5, 5, "var(--color-accent)", 0.4)}
      </>
    ),
  },
  pooling: {
    caption: "Blocks of the input collapse to one value each: the grid gets smaller.",
    draw: () => (
      <>
        {grid(2, 4, 6, 6, 5, INK, 0.25)}
        <rect x="2" y="4" width="10" height="10" fill="none" stroke="var(--color-accent)" strokeWidth="1.5" />
        <path d="M36 20 L50 20" stroke={DIM} strokeWidth="1" markerEnd="url(#gl-arrow)" />
        {grid(58, 12, 3, 3, 5, "var(--color-accent)", 0.45)}
      </>
    ),
  },
  core: {
    caption: "Every input is connected to every output.",
    draw: () => (
      <>
        {[8, 16, 24, 32].map((y) => (
          <circle key={y} cx="16" cy={y} r="2.5" fill={INK} opacity="0.6" />
        ))}
        {[12, 20, 28].map((y) => (
          <circle key={y} cx="76" cy={y} r="2.5" fill="var(--color-accent)" />
        ))}
        {[8, 16, 24, 32].flatMap((from) =>
          [12, 20, 28].map((to) => (
            <line
              key={`${from}-${to}`}
              x1="18" y1={from} x2="74" y2={to}
              stroke={DIM} strokeWidth="0.5" opacity="0.5"
            />
          )),
        )}
      </>
    ),
  },
  recurrent: {
    caption: "Each step reads the one before it: the layer carries state forward.",
    draw: () => (
      <>
        {[14, 38, 62].map((x, index) => (
          <g key={x}>
            <rect x={x} y="14" width="16" height="14" rx="2" fill={INK} opacity="0.3" />
            {index < 2 && (
              <path d={`M${x + 16} 21 L${x + 22} 21`} stroke="var(--color-accent)"
                    strokeWidth="1.5" markerEnd="url(#gl-arrow)" />
            )}
            <path d={`M${x + 8} 28 L${x + 8} 34`} stroke={DIM} strokeWidth="1" />
          </g>
        ))}
        <path d="M22 34 L78 34" stroke={DIM} strokeWidth="1" strokeDasharray="2 2" />
      </>
    ),
  },
  attention: {
    caption: "Every position weighs every other; the matrix is what it decided.",
    draw: () => (
      <>
        {Array.from({ length: 6 }, (_, row) =>
          Array.from({ length: 6 }, (_, col) => (
            <rect
              key={`${row}-${col}`}
              x={30 + col * 6} y={5 + row * 5} width="5" height="4"
              fill="var(--color-accent)"
              opacity={0.15 + 0.7 * Math.exp(-Math.abs(row - col) / 1.5)}
            />
          )),
        )}
        {[8, 16, 24, 32].map((y) => (
          <circle key={y} cx="10" cy={y} r="2" fill={INK} opacity="0.6" />
        ))}
        {[8, 16, 24, 32].map((y) => (
          <circle key={`r${y}`} cx="86" cy={y} r="2" fill={INK} opacity="0.6" />
        ))}
      </>
    ),
  },
  normalization: {
    caption: "The spread is rescaled: the same shape, centred and evened out.",
    draw: () => (
      <>
        {[6, 22, 10, 34, 14, 28, 8].map((h, index) => (
          <rect key={`a${index}`} x={4 + index * 5} y={36 - h} width="3" height={h}
                fill={INK} opacity="0.35" />
        ))}
        <path d="M44 20 L54 20" stroke={DIM} strokeWidth="1" markerEnd="url(#gl-arrow)" />
        {[16, 22, 18, 24, 19, 21, 17].map((h, index) => (
          <rect key={`b${index}`} x={60 + index * 5} y={36 - h} width="3" height={h}
                fill="var(--color-accent)" opacity="0.6" />
        ))}
      </>
    ),
  },
  regularization: {
    caption: "Some connections are dropped at random while training.",
    draw: () => (
      <>
        {[10, 20, 30].map((y) => (
          <circle key={y} cx="16" cy={y} r="2.5" fill={INK} opacity="0.6" />
        ))}
        {[10, 20, 30].map((y) => (
          <circle key={`o${y}`} cx="76" cy={y} r="2.5" fill="var(--color-accent)" />
        ))}
        {[10, 20, 30].flatMap((from, i) =>
          [10, 20, 30].map((to, j) => (
            <line
              key={`${from}-${to}`} x1="18" y1={from} x2="74" y2={to}
              stroke={(i + j) % 3 === 0 ? "var(--color-danger)" : DIM}
              strokeWidth="0.6"
              strokeDasharray={(i + j) % 3 === 0 ? "2 2" : undefined}
              opacity={(i + j) % 3 === 0 ? 0.8 : 0.5}
            />
          )),
        )}
      </>
    ),
  },
  reshaping: {
    caption: "The same values, arranged differently.",
    draw: () => (
      <>
        {grid(6, 8, 4, 4, 6, INK, 0.3)}
        <path d="M36 20 L50 20" stroke={DIM} strokeWidth="1" markerEnd="url(#gl-arrow)" />
        {grid(56, 17, 8, 2, 5, "var(--color-accent)", 0.4)}
      </>
    ),
  },
  merging: {
    caption: "Two streams come in, one goes out.",
    draw: () => (
      <>
        <rect x="4" y="4" width="20" height="12" rx="2" fill={INK} opacity="0.35" />
        <rect x="4" y="24" width="20" height="12" rx="2" fill={INK} opacity="0.35" />
        <path d="M24 10 Q42 10 44 20" stroke={DIM} strokeWidth="1.2" fill="none" />
        <path d="M24 30 Q42 30 44 20" stroke={DIM} strokeWidth="1.2" fill="none" />
        <path d="M44 20 L56 20" stroke={DIM} strokeWidth="1.2" markerEnd="url(#gl-arrow)" />
        <rect x="62" y="12" width="26" height="16" rx="2" fill="var(--color-accent)" opacity="0.5" />
      </>
    ),
  },
  activation: {
    caption: "A curve applied to every value on its own.",
    draw: () => (
      <>
        <path d="M8 36 L88 36" stroke={DIM} strokeWidth="0.8" />
        <path d="M48 4 L48 38" stroke={DIM} strokeWidth="0.8" />
        <path d="M8 36 L48 36 L88 6" stroke="var(--color-accent)" strokeWidth="2" fill="none" />
      </>
    ),
  },
  preprocessing: {
    caption: "The input is prepared before the model proper sees it.",
    draw: () => (
      <>
        {grid(4, 8, 4, 4, 6, DIM, 0.3)}
        <path d="M34 20 L46 20" stroke={DIM} strokeWidth="1" markerEnd="url(#gl-arrow)" />
        <rect x="50" y="8" width="16" height="24" rx="2" fill="none"
              stroke="var(--color-accent)" strokeWidth="1.2" strokeDasharray="3 2" />
        <path d="M68 20 L78 20" stroke={DIM} strokeWidth="1" markerEnd="url(#gl-arrow)" />
        {grid(80, 12, 2, 3, 5, "var(--color-accent)", 0.45)}
      </>
    ),
  },
  "research/ssm": {
    caption: "State sweeps along the sequence in one pass, instead of attending.",
    draw: () => (
      <>
        <path d="M6 30 Q26 6 46 22 T88 14" stroke="var(--color-accent)" strokeWidth="2" fill="none" />
        {[6, 26, 46, 66, 86].map((x) => (
          <circle key={x} cx={x} cy="36" r="1.8" fill={INK} opacity="0.5" />
        ))}
        <path d="M6 36 L88 36" stroke={DIM} strokeWidth="0.6" />
      </>
    ),
  },
  "research/kan": {
    caption: "The function sits on the edge, not in the node.",
    draw: () => (
      <>
        {[12, 28].map((y) => <circle key={y} cx="12" cy={y} r="2.5" fill={INK} opacity="0.6" />)}
        {[20].map((y) => <circle key={y} cx="84" cy={y} r="2.5" fill="var(--color-accent)" />)}
        <path d="M14 12 Q34 2 48 16 T82 19" stroke="var(--color-accent)" strokeWidth="1.4" fill="none" />
        <path d="M14 28 Q34 40 48 24 T82 21" stroke="var(--color-accent)" strokeWidth="1.4" fill="none" opacity="0.6" />
      </>
    ),
  },
  "research/moe": {
    caption: "A gate picks which few experts see each token.",
    draw: () => (
      <>
        <circle cx="12" cy="20" r="3" fill={INK} opacity="0.6" />
        {[6, 16, 26, 36].map((y, index) => (
          <g key={y}>
            <rect x="54" y={y - 3} width="30" height="7" rx="1.5"
                  fill={index === 1 || index === 3 ? "var(--color-accent)" : INK}
                  opacity={index === 1 || index === 3 ? 0.65 : 0.2} />
            <line x1="16" y1="20" x2="52" y2={y} strokeWidth="0.8"
                  stroke={index === 1 || index === 3 ? "var(--color-accent)" : DIM}
                  opacity={index === 1 || index === 3 ? 0.9 : 0.3} />
          </g>
        ))}
      </>
    ),
  },
};

/** Families that borrow another's drawing rather than inventing a worse one. */
const BORROWS: Record<string, string> = {
  "research/attention": "attention",
  "research/recurrent": "recurrent",
  "research/conv": "convolution",
  "research/ffn": "core",
  "research/mixer": "core",
  "research/graph": "merging",
  "research/neuro": "recurrent",
  "research/misc": "core",
};

/**
 * > [!AML-DOC-UNIT]
 * A drawing of what this kind of layer does.
 * @param typeId   the catalog id, which decides the drawing before the category does
 * @param category the layer's family, used when the id has no drawing of its own
 * @param caption  whether to print the sentence under it
 * @returns the figure, or null when nothing truthful can be drawn
 * @sideEffects none
 * @context Returns null rather than a generic box: a picture that says nothing is
 *          worse than no picture, because it still asks to be read [E-052].
 */
export function LayerGlyph({
  typeId,
  category,
  caption = true,
}: {
  typeId?: string;
  category: string;
  caption?: boolean;
}) {
  const leaf = typeId?.split(".").pop() ?? "";
  const key =
    (leaf && leaf in GLYPHS && leaf) ||
    (category in GLYPHS && category) ||
    BORROWS[category] ||
    "";
  const glyph = key ? GLYPHS[key] : undefined;
  if (!glyph) return null;

  return (
    <figure className="m-0">
      <svg
        viewBox="0 0 96 40"
        className="w-full rounded-sm bg-surface-2"
        style={{ maxHeight: 64 }}
        role="img"
        aria-label={glyph.caption}
      >
        <defs>
          <marker id="gl-arrow" viewBox="0 0 6 6" refX="5" refY="3"
                  markerWidth="5" markerHeight="5" orient="auto">
            <path d="M0 0 L6 3 L0 6 z" fill={DIM} />
          </marker>
        </defs>
        {glyph.draw()}
      </svg>
      {caption && (
        <figcaption className="mt-1 text-[10px] leading-snug text-ink-2">
          {glyph.caption}
        </figcaption>
      )}
    </figure>
  );
}
