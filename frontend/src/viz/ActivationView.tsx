/**
 * > [!AML-DOC-FILE]
 * @file        src/viz/ActivationView.tsx
 * @description Draws one layer's real output, picking the form from the hint the
 *              engine attached to it.
 * @module      frontend/viz/ActivationView
 * @exports     ActivationView, ActivationThumbnail
 * @created     2026-10-01
 * @context     Nothing here decides *what* a tensor means; the engine does that when
 *              it captures [task 05]. This file only draws, so a new kind of data
 *              needs a new hint on the engine and one more branch here.
 */

import type { Activation } from "../api/types";

/**
 * > [!AML-DOC-UNIT]
 * How many bars can carry a readable name under them.
 * @context Past this, a name gets a couple of pixels and truncates to nothing, which
 *          is what a 170-class output did: the names were there and none could be
 *          read [E-043].
 */
const NAMEABLE_BARS = 16;

/** How many of the top predictions to list when there are too many to label. */
const TOP_N = 6;

/**
 * > [!AML-DOC-UNIT]
 * A compact bar chart, used for flattened vectors and for class probabilities.
 * @param values the numbers to draw
 * @param labels optional names, one per value
 * @param height chart height in pixels
 * @returns the chart element
 * @context Drawn as one SVG rather than a row of elements. A div per bar needs a
 *          minimum width to exist at all, so five hundred of them demand a thousand
 *          pixels and burst out of a three-hundred-pixel panel; a viewBox scales to
 *          whatever width it is given [E-044].
 */
function Bars({
  values,
  labels,
  height = 64,
}: {
  values: number[];
  labels?: string[];
  height?: number;
}) {
  if (values.length === 0) return null;
  const low = Math.min(0, ...values);
  const high = Math.max(...values, low + 1e-9);
  const span = high - low || 1;
  const best = values.indexOf(Math.max(...values));
  const named = labels && labels.length >= values.length;

  const ranked = named
    ? values
        .map((value, index) => ({ value, index }))
        .sort((a, b) => b.value - a.value)
        .slice(0, TOP_N)
    : [];

  return (
    <div className="min-w-0">
      <svg
        viewBox={`0 0 ${values.length} 100`}
        preserveAspectRatio="none"
        className="w-full"
        style={{ height }}
        role="img"
        aria-label={`${values.length} values`}
      >
        {values.map((value, index) => {
          const tall = Math.max(((value - low) / span) * 100, 0.5);
          return (
            <rect
              // eslint-disable-next-line react/no-array-index-key
              key={index}
              x={index}
              y={100 - tall}
              width={1}
              height={tall}
              fill={index === best ? "var(--color-good)" : "var(--color-accent)"}
              opacity={index === best ? 1 : 0.6}
            >
              <title>{`${labels?.[index] ?? index}: ${value.toFixed(4)}`}</title>
            </rect>
          );
        })}
      </svg>

      {named && values.length <= NAMEABLE_BARS && (
        <div className="mt-1 flex gap-px text-[9px] text-ink-2">
          {values.map((_, index) => (
            // eslint-disable-next-line react/no-array-index-key
            <span
              key={index}
              className={`min-w-0 flex-1 truncate text-center ${
                index === best ? "font-semibold text-good" : ""
              }`}
            >
              {labels?.[index]}
            </span>
          ))}
        </div>
      )}

      {/*
        Too many to label one by one, so the ones that matter are listed instead.
        A ranked few is what anybody wanted from a 170-class head anyway.
      */}
      {named && values.length > NAMEABLE_BARS && (
        <ol className="mt-1.5 space-y-0.5">
          {ranked.map((entry, position) => (
            <li
              key={entry.index}
              className="flex items-baseline justify-between gap-2 text-[10px]"
            >
              <span
                className={`min-w-0 truncate ${
                  position === 0 ? "font-semibold text-good" : "text-ink-1"
                }`}
                title={labels?.[entry.index]}
              >
                {labels?.[entry.index]}
              </span>
              <span className="shrink-0 font-mono text-[10px] text-ink-2">
                {entry.value.toFixed(3)}
              </span>
            </li>
          ))}
        </ol>
      )}
    </div>
  );
}

/**
 * > [!AML-DOC-UNIT]
 * A waveform drawn as a polyline, which stays readable at thousands of points.
 * @param values the trace
 * @returns the chart element
 */
function Waveform({ values }: { values: number[] }) {
  if (values.length === 0) return null;
  const limit = Math.max(...values.map(Math.abs), 1e-9);
  const points = values
    .map((value, index) => {
      const x = (index / (values.length - 1 || 1)) * 100;
      const y = 50 - (value / limit) * 48;
      return `${x.toFixed(3)},${y.toFixed(3)}`;
    })
    .join(" ");

  return (
    <svg viewBox="0 0 100 100" preserveAspectRatio="none" className="h-20 w-full">
      <line
        x1="0"
        y1="50"
        x2="100"
        y2="50"
        stroke="var(--color-line)"
        strokeWidth="0.3"
      />
      <polyline
        points={points}
        fill="none"
        stroke="var(--color-accent)"
        strokeWidth="0.4"
        vectorEffect="non-scaling-stroke"
      />
    </svg>
  );
}

/**
 * > [!AML-DOC-UNIT]
 * A grid of feature maps, one image per channel.
 * @param tiles     base64 images
 * @param truncated whether channels were left out
 * @param total     how many channels the layer really has
 * @param columns   tiles per row
 * @returns the grid element
 */
function TileGrid({
  tiles,
  truncated,
  total,
  columns,
}: {
  tiles: string[];
  truncated: boolean;
  total: number;
  columns: number;
}) {
  return (
    <div>
      <div
        className="grid gap-1"
        style={{ gridTemplateColumns: `repeat(${columns}, minmax(0, 1fr))` }}
      >
        {tiles.map((tile, index) => (
          // eslint-disable-next-line react/no-array-index-key
          <img
            key={index}
            src={tile}
            alt={`channel ${index}`}
            title={`channel ${index}`}
            className="aspect-square w-full rounded-sm bg-surface-2"
            style={{ imageRendering: "pixelated" }}
          />
        ))}
      </div>
      {truncated && (
        <p className="mt-1 text-[10px] text-ink-2">
          Showing {tiles.length} of {total} channels.
        </p>
      )}
    </div>
  );
}

/**
 * > [!AML-DOC-UNIT]
 * Draw one layer's output in full.
 * @param activation what the engine captured for this layer
 * @param compact    render smaller, for the side panel rather than the drawer
 * @returns the visualisation
 * @sideEffects none
 */
export function ActivationView({
  activation,
  compact = false,
}: {
  activation: Activation;
  compact?: boolean;
}) {
  const columns = compact ? 6 : 8;

  switch (activation.hint) {
    case "feature_maps":
    case "volume_slices":
      return (
        <TileGrid
          tiles={activation.tiles}
          truncated={activation.truncated}
          total={activation.channel_count}
          columns={columns}
        />
      );

    case "attention_matrix":
      return (
        <div>
          <div
            className="grid gap-1"
            style={{
              gridTemplateColumns: `repeat(${Math.min(4, columns)}, minmax(0, 1fr))`,
            }}
          >
            {activation.tiles.map((tile, index) => (
              // eslint-disable-next-line react/no-array-index-key
              <figure key={index} className="m-0">
                <img
                  src={tile}
                  alt={`attention head ${index}`}
                  className="aspect-square w-full rounded-sm bg-surface-2"
                  style={{ imageRendering: "pixelated" }}
                />
                <figcaption className="text-center text-[9px] text-ink-2">
                  head {index}
                </figcaption>
              </figure>
            ))}
          </div>
          <p className="mt-1 text-[10px] text-ink-2">
            Rows are query positions, columns are keys. Each row sums to 1.
          </p>
        </div>
      );

    case "sequence_heatmap":
    case "spectrogram":
      return (
        <div>
          {activation.heatmap && (
            <img
              src={activation.heatmap}
              alt={activation.label}
              className="w-full rounded-sm bg-surface-2"
              style={{ imageRendering: "pixelated", minHeight: 48 }}
            />
          )}
          <p className="mt-1 text-[10px] text-ink-2">
            {activation.hint === "spectrogram"
              ? "Frequency upward, time rightward."
              : "Features upward, position rightward."}
          </p>
        </div>
      );

    case "waveform":
      return <Waveform values={activation.series} />;

    case "probabilities":
      return (
        <Bars
          values={activation.series}
          labels={
            activation.labels.length
              ? activation.labels
              : activation.series.map((_, index) => String(index))
          }
          height={compact ? 48 : 80}
        />
      );

    case "vector_bars":
      // The engine attaches names whenever a vocabulary fits this layer's width, and
      // throwing them away here is why a working mapping showed nothing [E-043].
      return (
        <Bars
          values={activation.series}
          labels={activation.labels.length ? activation.labels : undefined}
          height={compact ? 40 : 64}
        />
      );

    case "scalar":
      return (
        <p className="font-mono text-[18px] text-ink-0">
          {activation.series[0]?.toFixed(4) ?? "—"}
        </p>
      );

    default:
      return (
        <p className="text-[11px] text-ink-2">Nothing to draw for this layer.</p>
      );
  }
}

/**
 * > [!AML-DOC-UNIT]
 * A small image standing in for a layer's output, drawn inside its canvas node so
 * the data flow is visible on the architecture itself.
 * @param activation what the engine captured for this layer
 * @returns the thumbnail, or null when this kind of output has no useful one
 * @sideEffects none
 */
export function ActivationThumbnail({ activation }: { activation: Activation }) {
  const image = activation.tiles[0] ?? activation.heatmap;
  if (image) {
    return (
      <img
        src={image}
        alt=""
        className="mt-1.5 h-12 w-full rounded-sm bg-surface-2 object-contain"
        style={{ imageRendering: "pixelated" }}
      />
    );
  }
  if (activation.series.length > 1) {
    return (
      <div className="mt-1.5">
        <Bars values={activation.series.slice(0, 64)} height={24} />
      </div>
    );
  }
  return null;
}
