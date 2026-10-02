"""
> [!AML-DOC-FILE]
@file       export/inline.py
@description Extracts the source of non-Keras layer classes, with the module-level
             names they depend on, so an exported project carries its own layer
             implementations and imports nothing from this tool.
@module     nnarch.export.inline
@exports    InlineResult, collect_inline_sources
@created    2026-09-30
@context    Serves task 06 item 3. Research layers [task 04] live in
            `nnarch.layers.*`; an exported project must not import `nnarch`, so the
            classes it uses travel with it as source text.
"""

from __future__ import annotations

import ast
import inspect
from dataclasses import dataclass, field

from nnarch.catalog.introspect import resolve_keras_class
from nnarch.catalog.spec import LayerSpec

_TOOL_PACKAGE = "nnarch"


@dataclass
class InlineResult:
    """
    > [!AML-DOC-UNIT]
    Source text gathered for the custom layers an exported model needs.
    @param source   the body of the generated `custom_layers.py`, empty when the
                    model uses only stock Keras layers
    @param symbols  names the generated module defines, in emission order
    @param problems human-readable reasons a class could not be inlined
    """

    source: str = ""
    symbols: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    @property
    def needed(self) -> bool:
        """
        > [!AML-DOC-UNIT]
        Whether a `custom_layers.py` has to be written at all.
        @returns True when at least one symbol was collected
        """
        return bool(self.symbols)


def _module_index(module_source: str) -> tuple[dict[str, ast.stmt], list[ast.stmt]]:
    """
    > [!AML-DOC-UNIT]
    Index a module's top-level definitions by the names they bind.
    @param module_source full text of the module
    @returns (mapping of bound name to the statement that binds it, list of the
             module's import statements)
    @raises SyntaxError when the module does not parse
    @sideEffects none
    """
    tree = ast.parse(module_source)
    bindings: dict[str, ast.stmt] = {}
    imports: list[ast.stmt] = []

    for statement in tree.body:
        if isinstance(statement, (ast.Import, ast.ImportFrom)):
            imports.append(statement)
        elif isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bindings[statement.name] = statement
        elif isinstance(statement, ast.Assign):
            for target in statement.targets:
                if isinstance(target, ast.Name):
                    bindings[target.id] = statement
        elif isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
            bindings[statement.target.id] = statement
    return bindings, imports


def _referenced_names(statement: ast.stmt) -> set[str]:
    """
    > [!AML-DOC-UNIT]
    Collect every bare name a statement mentions, including decorators and bases.
    @param statement the AST statement to scan
    @returns set of identifier strings
    @sideEffects none
    """
    return {node.id for node in ast.walk(statement) if isinstance(node, ast.Name)}


def _render_imports(imports: list[ast.stmt]) -> tuple[list[str], list[str]]:
    """
    > [!AML-DOC-UNIT]
    Render a module's imports, dropping those that point back at this tool.
    @param imports the module's top-level import statements
    @returns (source lines to emit, names that were dropped because they come from
             this tool and so cannot travel with the export). `__future__` imports
             are skipped because the generated header supplies its own.
    @sideEffects none
    """
    lines: list[str] = []
    dropped: list[str] = []
    for statement in imports:
        if isinstance(statement, ast.ImportFrom):
            # The generated header emits its own __future__ import, and a duplicate
            # would be redundant noise in a file meant to be read and edited.
            if statement.module == "__future__":
                continue
            root = (statement.module or "").split(".")[0]
            if root == _TOOL_PACKAGE or statement.level:
                dropped.extend(alias.asname or alias.name for alias in statement.names)
                continue
        elif isinstance(statement, ast.Import):
            kept = [a for a in statement.names if a.name.split(".")[0] != _TOOL_PACKAGE]
            dropped.extend(
                a.asname or a.name for a in statement.names
                if a.name.split(".")[0] == _TOOL_PACKAGE
            )
            if not kept:
                continue
        text = ast.unparse(statement)
        if text not in lines:
            lines.append(text)
    return lines, dropped


def collect_inline_sources(specs: list[LayerSpec]) -> InlineResult:
    """
    > [!AML-DOC-UNIT]
    Gather the source needed to redefine a set of non-Keras layers standalone.
    @param specs catalog specs of the layers a model actually uses; stock Keras
                 layers are ignored
    @returns InlineResult holding the generated module source, the symbols it
             defines and any class that could not be carried over
    @raises nothing; a class that cannot be extracted becomes an entry in
            `problems` so the caller can surface it as a diagnostic
    @sideEffects imports the modules that define the requested classes
    @context Pulls each class plus the module-level helpers and constants it
             references, transitively, rather than the whole family module, so a
             project that uses one layer does not carry ten.
    """
    result = InlineResult()
    custom = [spec for spec in specs if not spec.keras_path.startswith("keras.")]
    if not custom:
        return result

    by_module: dict[str, list[str]] = {}
    for spec in custom:
        module_path, _, class_name = spec.keras_path.rpartition(".")
        by_module.setdefault(module_path, []).append(class_name)

    import_lines: list[str] = []
    blocks: list[str] = []

    for module_path, class_names in sorted(by_module.items()):
        try:
            module = __import__(module_path, fromlist=["*"])
            module_source = inspect.getsource(module)
        except (ImportError, OSError, TypeError) as exc:
            result.problems.append(f"{module_path}: source unavailable ({exc})")
            continue

        try:
            bindings, imports = _module_index(module_source)
        except SyntaxError as exc:
            result.problems.append(f"{module_path}: does not parse ({exc})")
            continue

        wanted: list[str] = []
        pending = list(class_names)
        seen: set[str] = set()
        while pending:
            name = pending.pop(0)
            if name in seen:
                continue
            seen.add(name)
            statement = bindings.get(name)
            if statement is None:
                result.problems.append(f"{module_path}.{name}: not a module-level definition")
                continue
            wanted.append(name)
            pending.extend(
                sorted(_referenced_names(statement) & set(bindings) - seen)
            )

        lines, dropped = _render_imports(imports)
        for line in lines:
            if line not in import_lines:
                import_lines.append(line)

        unusable = sorted(set(dropped) & {
            ref for name in wanted for ref in _referenced_names(bindings[name])
        })
        for name in unusable:
            result.problems.append(
                f"{module_path}: {name} comes from {_TOOL_PACKAGE} and cannot be inlined"
            )

        for name in sorted(wanted, key=lambda n: bindings[n].lineno):
            blocks.append(ast.unparse(bindings[name]))
            result.symbols.append(name)

    if not result.symbols:
        return result

    header = '"""Layer implementations this model needs, extracted so it runs standalone."""'
    result.source = "\n".join([
        header, "", "from __future__ import annotations", "",
        *import_lines, "", "", ("\n\n\n".join(blocks)), "",
    ])
    return result
