/**
 * > [!AML-DOC-FILE]
 * @file        src/viz/MetricChart.tsx
 * @description A compact line chart for a training metric, with a crosshair and a
 *              tooltip.
 * @module      frontend/viz/MetricChart
 * @exports     MetricChart
 * @created     2026-10-01
 * @context     One chart per measure, never two scales on one axis: loss and
 *              accuracy live on different ranges, so they get separate charts.
 *              The series colours are the validated categorical slots 1 and 2 from
 *              the project's chart palette, which clear CVD separation and the
 *              lightness band on both surfaces. Both series are directly labelled,
 *              which is also the relief the light-mode contrast warning requires, so
 *              identity never rests on colour alone.
 */

import { useRef, useState } from "react";

const WIDTH = 100;
const HEIGHT = 46;
const PADDING = 2;

/** One line on the chart. */
export interface Series {
  label: string;
  values: (number | null)[];
  colour: string;
}

/**
 * > [!AML-DOC-UNIT]
 * Map a value onto the chart's vertical axis.
 * @param value the value
 * @param low   smallest value across every series
 * @param high  largest value across every series
 * @returns the y coordinate in the chart's own units
 */
function scale(value: number, low: number, high: number): number {
  const span = high - low || 1;
  return HEIGHT - PADDING - ((value - low) / span) * (HEIGHT - PADDING * 2);
}

/**
 * > [!AML-DOC-UNIT]
 * Draw one metric over epochs.
 * @param title  what the chart measures
 * @param series the lines to draw
 * @param format how to render a value in the tooltip and the direct labels
 * @returns the chart, or a placeholder before there is anything to draw
 * @sideEffects none
 */
export function MetricChart({
  title,
  series,
  format = (value: number) => value.toFixed(4),
}: {
  title: string;
  series: Series[];
  format?: (value: number) => string;
}) {
  const [hover, setHover] = useState<number | null>(null);
  const surface = useRef<SVGSVGElement>(null);

  const present = series.filter((line) => line.values.some((v) => v !== null));
  const points = Math.max(...present.map((line) => line.values.length), 0);

  if (points < 1) {
    return (
      <figure className="m-0">
        <figcaption className="mb-1 text-[10px] uppercase tracking-wide text-ink-2">
          {title}
        </figcaption>
        <div className="flex h-[70px] items-center justify-center rounded border border-line">
          <span className="text-[10px] text-ink-2">no epochs yet</span>
        </div>
      </figure>
    );
  }

  const all = present.flatMap((line) => line.values.filter((v): v is number => v !== null));
  const low = Math.min(...all);
  const high = Math.max(...all);
  const step = points > 1 ? (WIDTH - PADDING * 2) / (points - 1) : 0;

  return (
    <figure className="m-0">
      <figcaption className="mb-1 flex items-baseline justify-between gap-2">
        <span className="text-[10px] uppercase tracking-wide text-ink-2">{title}</span>
        <span className="flex gap-2">
          {present.map((line) => {
            const last = [...line.values].reverse().find((v) => v !== null);
            return (
              <span key={line.label} className="flex items-center gap-1 text-[10px]">
                <span
                  aria-hidden
                  className="inline-block h-[2px] w-3 rounded"
                  style={{ background: line.colour }}
                />
                <span className="text-ink-2">{line.label}</span>
                <span className="font-mono text-ink-1">
                  {last === undefined || last === null ? "—" : format(last)}
                </span>
              </span>
            );
          })}
        </span>
      </figcaption>

      <svg
        ref={surface}
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        preserveAspectRatio="none"
        className="h-[70px] w-full rounded border border-line"
        role="img"
        aria-label={`${title} over ${points} epochs`}
        onMouseMove={(event) => {
          const box = surface.current?.getBoundingClientRect();
          if (!box || points < 2) return;
          const ratio = (event.clientX - box.left) / box.width;
          setHover(Math.round(ratio * (points - 1)));
        }}
        onMouseLeave={() => setHover(null)}
      >
        {present.map((line) => {
          const path = line.values
            .map((value, index) =>
              value === null
                ? null
                : `${PADDING + index * step},${scale(value, low, high)}`,
            )
            .filter((entry): entry is string => entry !== null)
            .join(" ");
          return (
            <polyline
              key={line.label}
              points={path}
              fill="none"
              stroke={line.colour}
              strokeWidth="2"
              strokeLinejoin="round"
              strokeLinecap="round"
              vectorEffect="non-scaling-stroke"
            />
          );
        })}

        {hover !== null && hover >= 0 && hover < points && (
          <g>
            <line
              x1={PADDING + hover * step}
              y1={0}
              x2={PADDING + hover * step}
              y2={HEIGHT}
              stroke="var(--color-line)"
              strokeWidth="1"
              vectorEffect="non-scaling-stroke"
            />
            {present.map((line) => {
              const value = line.values[hover];
              if (value === null || value === undefined) return null;
              return (
                <circle
                  key={line.label}
                  cx={PADDING + hover * step}
                  cy={scale(value, low, high)}
                  r="3"
                  fill={line.colour}
                  stroke="var(--color-surface-1)"
                  strokeWidth="2"
                  vectorEffect="non-scaling-stroke"
                />
              );
            })}
          </g>
        )}
      </svg>

      <div className="mt-0.5 flex h-3 items-center justify-between text-[9px] text-ink-2">
        {hover !== null && hover >= 0 && hover < points ? (
          <span className="truncate">
            epoch {hover + 1}:{" "}
            {present
              .map((line) => {
                const value = line.values[hover];
                return value === null || value === undefined
                  ? null
                  : `${line.label} ${format(value)}`;
              })
              .filter(Boolean)
              .join(" · ")}
          </span>
        ) : (
          <>
            <span>epoch 1</span>
            <span>epoch {points}</span>
          </>
        )}
      </div>
    </figure>
  );
}
