"""
> [!AML-DOC-FILE]
@file       naming.py
@description The single rule for turning a user-typed layer label into a name that
             Keras and Python both accept.
@module     nnarch.naming
@exports    PYTHON_KEYWORDS, sanitize_name, NamePool
@created    2026-09-30
@context    RISK:MED [amm: E.4]. The compiler and the code generator must produce
            byte-identical layer names, or an exported model differs from the one
            the app ran. Living at the package root keeps this free of any
            TensorFlow import.
"""

from __future__ import annotations

PYTHON_KEYWORDS: frozenset[str] = frozenset({
    "False", "None", "True", "and", "as", "assert", "async", "await", "break", "class",
    "continue", "def", "del", "elif", "else", "except", "finally", "for", "from",
    "global", "if", "import", "in", "is", "lambda", "nonlocal", "not", "or", "pass",
    "raise", "return", "try", "while", "with", "yield",
})

SHADOWED_IDENTIFIERS: frozenset[str] = frozenset({
    # Names the generated `model.py` binds itself.
    "keras", "layers", "model", "inputs", "outputs", "np", "tf", "build_model",
    # Builtins a generated variable must not shadow. "input" is the one that matters:
    # it is an obvious label for an input layer.
    "input", "type", "id", "list", "dict", "set", "str", "int", "float", "bool",
    "len", "map", "filter", "sum", "min", "max", "next", "range", "print", "format",
    "object", "bytes", "all", "any", "eval", "compile", "open", "vars", "dir", "hash",
})
"""
> [!AML-DOC-UNIT]
Identifiers a generated Python variable must not take, either because the generated
module binds them itself or because they are builtins.

These constrain only the Python variable name, never the Keras layer name: a layer
labelled "input" stays named `input` in both the running and the exported model, and
only its variable becomes `input_`. Renaming the layer instead would show the user a
`input_2` they never asked for, and would make the exported and compiled models agree
on a name neither of them needed to change.
"""


def python_variable(name: str) -> str:
    """
    > [!AML-DOC-UNIT]
    Adapt a layer name for use as a variable in generated code.
    @param name an already-sanitized layer name
    @returns the same name, with a trailing underscore when it would shadow a builtin
             or a name the generated module binds
    @sideEffects none
    @context Trailing underscore is PEP 8's own remedy for exactly this collision.
    """
    return f"{name}_" if name in SHADOWED_IDENTIFIERS else name


def sanitize_name(label: str, fallback: str = "") -> str:
    """
    > [!AML-DOC-UNIT]
    Reduce a free-text label to a lowercase identifier.
    @param label    the user's label, which may contain spaces, punctuation or nothing
    @param fallback used when the label yields no usable characters
    @returns a valid Python identifier, never empty
    @sideEffects none
    @context This is the canonical rule. Both `ir.compiler` and `export.codegen`
             call it, so a layer called "conv one" is named `conv_one` in the running
             model and in the exported file alike.
    """
    cleaned = "".join(char if char.isalnum() else "_" for char in label.strip().lower())
    cleaned = "_".join(part for part in cleaned.split("_") if part)
    if not cleaned:
        cleaned = "".join(char if char.isalnum() else "_" for char in fallback.strip().lower())
        cleaned = "_".join(part for part in cleaned.split("_") if part)
    if not cleaned:
        return "layer"
    if cleaned[0].isdigit() or cleaned in PYTHON_KEYWORDS:
        cleaned = f"layer_{cleaned}"
    return cleaned


class NamePool:
    """
    > [!AML-DOC-UNIT]
    Allocates unique sanitized names, so two layers labelled the same do not collide.
    @sideEffects records every allocated name
    """

    def __init__(self, reserved: frozenset[str] = frozenset()) -> None:
        self._taken: set[str] = set(reserved)

    def allocate(self, label: str, fallback: str = "") -> str:
        """
        > [!AML-DOC-UNIT]
        Reserve a unique name derived from a label.
        @param label    the user's label
        @param fallback used when the label yields nothing, typically the node id
        @returns a unique identifier, numbered on collision
        @sideEffects records the returned name as taken
        """
        base = sanitize_name(label, fallback)
        candidate = base
        counter = 2
        while candidate in self._taken:
            candidate = f"{base}_{counter}"
            counter += 1
        self._taken.add(candidate)
        return candidate
