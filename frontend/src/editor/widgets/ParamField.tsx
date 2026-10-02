/**
 * > [!AML-DOC-FILE]
 * @file        src/editor/widgets/ParamField.tsx
 * @description Renders the right input control for any layer parameter, chosen from
 *              the parameter's declared type rather than from its layer.
 * @module      frontend/editor/widgets/ParamField
 * @exports     ParamField
 * @created     2026-09-30
 * @context     This file is the whole of [amm: E.2]. Every layer's property form is
 *              produced here from its `ParamSpec` list, so adding a layer type on the
 *              backend needs no frontend change. A new control is only ever needed
 *              when a new `ParamType` is introduced.
 */

import { useMemo, useState } from "react";

import type { ParamSpec, Vocabularies } from "../../api/types";

interface ParamFieldProps {
  spec: ParamSpec;
  value: unknown;
  vocabularies: Vocabularies;
  nodeOptions: { id: string; label: string }[];
  onChange: (value: unknown) => void;
}

const LABEL_CLASS = "block text-[11px] font-medium text-ink-1 mb-1";
const INPUT_CLASS =
  "w-full rounded bg-surface-2 border border-line px-2 py-1 text-[12px] text-ink-0 " +
  "outline-none focus:border-accent focus:ring-1 focus:ring-accent/40 " +
  "disabled:opacity-50 placeholder:text-ink-2";

/**
 * > [!AML-DOC-UNIT]
 * Turn a snake_case parameter name into a readable label.
 * @param name the parameter name from the spec
 * @returns the name with underscores replaced by spaces
 */
function humanize(name: string): string {
  return name.replace(/_/g, " ");
}

/**
 * > [!AML-DOC-UNIT]
 * Coerce whatever the IR holds into a list of numbers for the tuple editors.
 * @param value  the current parameter value
 * @param arity  expected element count, when the spec declares one
 * @returns a numeric list of at least `arity` entries
 */
function asNumberList(value: unknown, arity: number | null): (number | null)[] {
  const list = Array.isArray(value)
    ? value.map((item) => (typeof item === "number" ? item : null))
    : typeof value === "number"
      ? Array.from({ length: arity ?? 1 }, () => value)
      : [];
  const size = arity ?? Math.max(list.length, 1);
  return Array.from({ length: size }, (_, index) => list[index] ?? null);
}

/**
 * > [!AML-DOC-UNIT]
 * A searchable single-select, used for the long vocabularies such as the 31
 * activation functions where a plain dropdown is unpleasant to scan.
 * @param options  the allowed values; null renders as "(none)"
 * @param value    the current value
 * @param onChange called with the chosen value
 * @returns the control
 */
function SearchableSelect({
  options,
  value,
  onChange,
}: {
  options: (string | null)[];
  value: unknown;
  onChange: (value: unknown) => void;
}) {
  const [query, setQuery] = useState("");
  const filtered = useMemo(() => {
    const needle = query.trim().toLowerCase();
    if (!needle) return options;
    return options.filter((option) => (option ?? "none").toLowerCase().includes(needle));
  }, [options, query]);

  return (
    <div className="space-y-1">
      {options.length > 8 && (
        <input
          className={INPUT_CLASS}
          placeholder="filter…"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
        />
      )}
      <select
        className={INPUT_CLASS}
        value={value === null || value === undefined ? "__none__" : String(value)}
        onChange={(event) =>
          onChange(event.target.value === "__none__" ? null : event.target.value)
        }
      >
        {!filtered.includes(null) && <option value="__none__">(none)</option>}
        {filtered.map((option) => (
          <option key={option ?? "__none__"} value={option ?? "__none__"}>
            {option ?? "(none)"}
          </option>
        ))}
      </select>
    </div>
  );
}

/**
 * > [!AML-DOC-UNIT]
 * Render the control for one parameter.
 * @param spec         the parameter's declared type, bounds, choices and help text
 * @param value        its current value
 * @param vocabularies enum vocabularies from the catalog
 * @param nodeOptions  canvas nodes, offered when the parameter references a layer
 * @param onChange     called with the new value
 * @returns the labelled control, with help text and a required marker
 * @sideEffects none; the caller owns the state
 */
export function ParamField({
  spec,
  value,
  vocabularies,
  nodeOptions,
  onChange,
}: ParamFieldProps) {
  const control = () => {
    switch (spec.type) {
      case "bool":
        return (
          <button
            type="button"
            role="switch"
            aria-checked={value === true}
            onClick={() => onChange(!(value === true))}
            className={`relative h-5 w-9 rounded-full transition-colors ${
              value === true ? "bg-accent" : "bg-surface-3"
            }`}
          >
            <span
              className={`absolute top-0.5 h-4 w-4 rounded-full bg-surface-0 transition-transform ${
                value === true ? "translate-x-4.5" : "translate-x-0.5"
              }`}
            />
          </button>
        );

      case "int":
      case "float":
        return (
          <input
            type="number"
            className={INPUT_CLASS}
            value={typeof value === "number" ? value : ""}
            min={spec.minimum ?? undefined}
            max={spec.maximum ?? undefined}
            step={spec.type === "int" ? 1 : "any"}
            placeholder={spec.required ? "required" : "default"}
            onChange={(event) => {
              const raw = event.target.value;
              if (raw === "") return onChange(null);
              const parsed = spec.type === "int" ? parseInt(raw, 10) : parseFloat(raw);
              onChange(Number.isNaN(parsed) ? null : parsed);
            }}
          />
        );

      case "enum":
        return (
          <select
            className={INPUT_CLASS}
            value={value === null || value === undefined ? "__none__" : String(value)}
            onChange={(event) =>
              onChange(event.target.value === "__none__" ? null : event.target.value)
            }
          >
            {(spec.choices ?? []).map((choice) => (
              <option key={String(choice)} value={choice === null ? "__none__" : String(choice)}>
                {choice === null ? "(default)" : String(choice)}
              </option>
            ))}
          </select>
        );

      case "activation":
        return (
          <SearchableSelect
            options={[null, ...vocabularies.activations]}
            value={value}
            onChange={onChange}
          />
        );
      case "initializer":
        return (
          <SearchableSelect
            options={vocabularies.initializers}
            value={value}
            onChange={onChange}
          />
        );
      case "regularizer":
        return (
          <SearchableSelect
            options={vocabularies.regularizers}
            value={value}
            onChange={onChange}
          />
        );
      case "constraint":
        return (
          <SearchableSelect options={vocabularies.constraints} value={value} onChange={onChange} />
        );
      case "dtype":
        return (
          <SearchableSelect
            options={[null, ...vocabularies.dtypes]}
            value={value}
            onChange={onChange}
          />
        );

      case "int_tuple":
      case "float_tuple": {
        const list = asNumberList(value, spec.arity);
        return (
          <div className="flex gap-1">
            {list.map((item, index) => (
              <input
                // eslint-disable-next-line react/no-array-index-key
                key={index}
                type="number"
                className={INPUT_CLASS}
                value={item ?? ""}
                min={spec.minimum ?? undefined}
                step={spec.type === "int_tuple" ? 1 : "any"}
                onChange={(event) => {
                  const next = [...list];
                  const raw = event.target.value;
                  next[index] =
                    raw === ""
                      ? null
                      : spec.type === "int_tuple"
                        ? parseInt(raw, 10)
                        : parseFloat(raw);
                  onChange(next.map((entry) => entry ?? 0));
                }}
              />
            ))}
          </div>
        );
      }

      case "shape": {
        const dims = Array.isArray(value) ? (value as (number | null)[]) : [];
        return (
          <div className="space-y-1">
            <div className="flex flex-wrap gap-1">
              {dims.map((dim, index) => (
                <input
                  // eslint-disable-next-line react/no-array-index-key
                  key={index}
                  type="number"
                  className={`${INPUT_CLASS} w-16`}
                  value={dim ?? ""}
                  placeholder="any"
                  min={1}
                  onChange={(event) => {
                    const next = [...dims];
                    const raw = event.target.value;
                    next[index] = raw === "" ? null : parseInt(raw, 10);
                    onChange(next);
                  }}
                />
              ))}
              <button
                type="button"
                className="rounded border border-line bg-surface-2 px-2 text-ink-1 hover:text-ink-0"
                onClick={() => onChange([...dims, 1])}
                title="Add a dimension"
              >
                +
              </button>
              {dims.length > 0 && (
                <button
                  type="button"
                  className="rounded border border-line bg-surface-2 px-2 text-ink-1 hover:text-ink-0"
                  onClick={() => onChange(dims.slice(0, -1))}
                  title="Remove the last dimension"
                >
                  −
                </button>
              )}
            </div>
            <p className="text-[10px] text-ink-2">
              {dims.length === 0
                ? "no dimensions"
                : `rank ${dims.length}, batch axis excluded`}
            </p>
          </div>
        );
      }

      case "layer_ref":
        return (
          <select
            className={INPUT_CLASS}
            value={typeof value === "string" ? value : "__none__"}
            onChange={(event) =>
              onChange(event.target.value === "__none__" ? null : event.target.value)
            }
          >
            <option value="__none__">(pick a layer)</option>
            {nodeOptions.map((option) => (
              <option key={option.id} value={option.id}>
                {option.label}
              </option>
            ))}
          </select>
        );

      case "text":
        return (
          <textarea
            className={`${INPUT_CLASS} font-mono`}
            rows={2}
            value={typeof value === "string" ? value : ""}
            placeholder="x * 2.0"
            onChange={(event) => onChange(event.target.value)}
          />
        );

      default:
        return (
          <input
            className={`${INPUT_CLASS} font-mono`}
            value={
              value === null || value === undefined
                ? ""
                : typeof value === "string"
                  ? value
                  : JSON.stringify(value)
            }
            placeholder={spec.required ? "required" : "default"}
            onChange={(event) => {
              const raw = event.target.value;
              if (raw === "") return onChange(null);
              try {
                onChange(JSON.parse(raw));
              } catch {
                onChange(raw);
              }
            }}
          />
        );
    }
  };

  return (
    <div>
      <label className={LABEL_CLASS}>
        <span className="font-mono">{humanize(spec.name)}</span>
        {spec.required && <span className="ml-1 text-danger">*</span>}
      </label>
      {control()}
      {spec.help && <p className="mt-1 text-[10px] leading-snug text-ink-2">{spec.help}</p>}
    </div>
  );
}
