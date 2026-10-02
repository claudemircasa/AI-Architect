"""
> [!AML-DOC-FILE]
@file       export/render.py
@description Primitives for turning IR values into readable Python source: literal
             rendering, identifier sanitisation and Keras default elision.
@module     nnarch.export.render
@exports    render_value, render_kwargs, IdentifierPool, keras_signature_defaults
@created    2026-09-30
@context    RISK:MED [amm: E.4]. The kwargs these helpers emit must be semantically
            identical to what `ir.compiler.coerce_params` passes at runtime, or a
            model will work in the app and break once exported.
"""

from __future__ import annotations

import inspect
from functools import lru_cache
from typing import Any

from nnarch.catalog.introspect import resolve_keras_class
from nnarch.catalog.spec import LayerSpec
from nnarch.naming import NamePool, sanitize_name



def render_value(value: Any) -> str:
    """
    > [!AML-DOC-UNIT]
    Render a parameter value as a Python literal.
    @param value the value taken from the IR, already coerced by the compiler's rules
    @returns source text that evaluates back to an equal value
    @raises TypeError when the value has no literal form, which signals a parameter
            type the catalog declares but codegen has not been taught
    @sideEffects none
    """
    if value is None or isinstance(value, bool):
        return repr(value)
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(value, tuple):
        inner = ", ".join(render_value(item) for item in value)
        return f"({inner},)" if len(value) == 1 else f"({inner})"
    if isinstance(value, list):
        return "[" + ", ".join(render_value(item) for item in value) + "]"
    if isinstance(value, dict):
        pairs = ", ".join(f"{render_value(k)}: {render_value(v)}" for k, v in value.items())
        return "{" + pairs + "}"
    raise TypeError(f"no literal form for {type(value).__name__}: {value!r}")


@lru_cache(maxsize=512)
def keras_signature_defaults(keras_path: str) -> dict[str, Any]:
    """
    > [!AML-DOC-UNIT]
    Read the constructor defaults a layer class declares for itself.
    @param keras_path dotted import path of the layer class
    @returns mapping of parameter name to its own default, omitting parameters that
             have none
    @raises ImportError when the path cannot be resolved
    @sideEffects none beyond an import that has already happened during bootstrap
    @context Used to leave out arguments the user never changed, so the exported
             file reads like something a person wrote. Elision is safe only against
             the class's own default, never against the catalog's starter value,
             which may deliberately differ.
    """
    signature = inspect.signature(resolve_keras_class(keras_path).__init__)
    return {
        name: parameter.default
        for name, parameter in signature.parameters.items()
        if parameter.default is not inspect.Parameter.empty
    }


_NEUTRAL_VALUES: dict[str, dict[str, Any]] = {
    "keras.layers.Input": {"sparse": False},
}
"""
> [!AML-DOC-UNIT]
Values that are semantically identical to a class's own default but are not equal to
it, so plain equality cannot elide them. `keras.Input` defaults `sparse` to None,
which behaves exactly as False, and the catalog surfaces False so the property panel
can show a switch.

Deliberately narrow. `dtype="float32"` is NOT listed: it looks redundant but is not,
because a None dtype defers to the global mixed-precision policy, and dropping it
would change the model under a non-default policy.
"""


def _is_keras_default(keras_path: str, name: str, value: Any) -> bool:
    """
    > [!AML-DOC-UNIT]
    Decide whether a value can be omitted because the class already defaults to it.
    @param keras_path dotted import path of the layer class
    @param name       parameter name
    @param value      effective value the compiler would pass
    @returns True when omitting the argument leaves behaviour unchanged
    @sideEffects none
    @context Checks `_NEUTRAL_VALUES` first, then plain equality against the class's
             own signature default. Never elides against the catalog's starter
             default, which may deliberately differ from Keras'.
    """
    if _NEUTRAL_VALUES.get(keras_path, {}).get(name, object()) == value:
        return True
    defaults = keras_signature_defaults(keras_path)
    if name not in defaults:
        return False
    default = defaults[name]
    if type(default) is not type(value) and not (
        isinstance(default, (int, float)) and isinstance(value, (int, float))
    ):
        return False
    return bool(default == value)


def render_kwargs(
    spec: LayerSpec,
    params: dict[str, Any],
    *,
    name: str | None = None,
    inline: dict[str, str] | None = None,
) -> str:
    """
    > [!AML-DOC-UNIT]
    Render a layer's constructor arguments.
    @param spec   catalog spec of the layer being constructed
    @param params effective parameters, as produced by `ir.compiler.coerce_params`
    @param name   value for the Keras `name=` argument, appended last when given
    @param inline pre-rendered source for parameters that have no literal form,
                  such as a Lambda's expression or a wrapper's child layer
    @returns comma-separated argument source, empty when everything was elided
    @raises TypeError when a parameter has neither a literal form nor an `inline`
            entry
    @sideEffects none
    @context Arguments equal to the class's own default are dropped for readability;
             every other value is emitted verbatim, so the exported model is
             constructed with exactly the arguments the compiler used [amm: E.4].
    """
    inline = inline or {}
    parts: list[str] = []
    for param_name, value in params.items():
        if param_name in inline:
            parts.append(f"{param_name}={inline[param_name]}")
            continue
        if _is_keras_default(spec.keras_path, param_name, value):
            continue
        parts.append(f"{param_name}={render_value(value)}")
    if name is not None:
        parts.append(f'name={render_value(name)}')
    return ", ".join(parts)


class IdentifierPool(NamePool):
    """
    > [!AML-DOC-UNIT]
    Allocates Keras layer names for the generated module, drawing from the same
    unseeded pool the compiler uses so the two agree name for name.
    @sideEffects inherits NamePool's allocation bookkeeping
    @context Shadowing is handled separately, by `naming.python_variable`, because it
             concerns the Python variable rather than the layer's name.
    """
