"""
> [!AML-DOC-FILE]
@file       ir/compiler.py
@description Turns a graph IR into a real `keras.Model` by building it symbolically
             through the functional API, which yields exact shapes and authentic
             Keras error messages.
@module     nnarch.ir.compiler
@exports    CompileResult, compile_graph, coerce_params, build_layer
@created    2026-09-30
@context    RISK:HIGH [amm: B.3]. Feeds shape inference, the activation engine
            [task 05] and code generation [task 06]. Shape math is NEVER
            reimplemented here: Keras is asked, so the tool and TensorFlow cannot
            disagree.
"""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass, field
from typing import Any

from nnarch.catalog import CallStyle, Diagnostic, Registry
from nnarch.naming import NamePool, sanitize_name
from nnarch.catalog.introspect import resolve_keras_class
from nnarch.catalog.spec import LayerSpec, ParamType

from .schema import GraphIR, Node, ShapeInfo
from .validate import validate_structure


class Codes:
    """
    > [!AML-DOC-UNIT]
    Diagnostic identifiers produced by compilation, as opposed to the structural
    codes in `validate.py`. Part of the public contract; never renamed.
    """

    LAYER_CONSTRUCTION_FAILED = "layer_construction_failed"
    LAYER_CALL_FAILED = "layer_call_failed"
    MODEL_ASSEMBLY_FAILED = "model_assembly_failed"
    BAD_LAMBDA_EXPRESSION = "bad_lambda_expression"
    DISABLED_NODE_ARITY = "disabled_node_arity"
    MISSING_INPUT_TENSOR = "missing_input_tensor"


_TUPLE_TYPES = frozenset({ParamType.INT_TUPLE, ParamType.FLOAT_TUPLE, ParamType.SHAPE})

_LAMBDA_NAMESPACE_DOC = """
Names a Lambda expression may use: `x` for the input tensor, `ops` for
`keras.ops`, and the `math` module. Nothing else is in scope, and builtins are
removed, so an expression cannot reach the filesystem or the network.
"""


@dataclass
class CompileResult:
    """
    > [!AML-DOC-UNIT]
    Outcome of compiling a graph.
    @param model            the assembled Keras model, or None when compilation failed
    @param shapes           per-node output shape, dtype and weight count
    @param diagnostics      problems found, structural ones included
    @param params_total     total weights in the model
    @param trainable_params weights that gradient descent will update
    @sideEffects holds a live Keras model; callers that cache it must key the cache
                 on the graph contents [task 05]
    """

    model: Any | None = None
    shapes: dict[str, ShapeInfo] = field(default_factory=dict)
    diagnostics: list[Diagnostic] = field(default_factory=list)
    params_total: int = 0
    trainable_params: int = 0

    @property
    def ok(self) -> bool:
        """
        > [!AML-DOC-UNIT]
        Whether the graph compiled into a usable model.
        @returns True when a model was produced and no diagnostic is an error
        """
        return self.model is not None and not any(
            diagnostic.severity == "error" for diagnostic in self.diagnostics
        )


#: Everything a Lambda expression is allowed to name.
_LAMBDA_NAMES = frozenset({"x", "ops", "math"})


def _free_names(tree: ast.AST) -> set[str]:
    """
    > [!AML-DOC-UNIT]
    The names an expression reads without binding them itself.
    @param tree parsed expression
    @returns the free names, which must all be in scope for it to run
    @sideEffects none
    @context A comprehension's own variable and a nested lambda's arguments are bound
             by the expression, so `ops.stack([x[i] for i in (0, 1)])` names only
             `ops` and `x`. Counting those as unknown would refuse a valid
             expression, which is worse than the error being prevented.
    """
    bound: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Lambda):
            args = node.args
            for argument in [*args.posonlyargs, *args.args, *args.kwonlyargs]:
                bound.add(argument.arg)
            if args.vararg:
                bound.add(args.vararg.arg)
            if args.kwarg:
                bound.add(args.kwarg.arg)
        elif isinstance(node, ast.comprehension):
            for target in ast.walk(node.target):
                if isinstance(target, ast.Name):
                    bound.add(target.id)

    read = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    return read - bound

def _lambda_from_expression(expression: str):
    """
    > [!AML-DOC-UNIT]
    Compile a Lambda layer's textual expression into a callable.
    @param expression Python expression over the input tensor `x`, e.g. `x * 2.0`
    @returns a single-argument callable suitable for `keras.layers.Lambda`
    @raises ValueError when the expression is not valid Python
    @sideEffects none at definition time; the expression runs when the layer is called
    @context Evaluated with builtins stripped and only `x`, `ops` and `math` in
             scope, so it behaves like a spreadsheet formula field rather than an
             arbitrary code-execution hole.
    """
    import keras

    if not expression.strip():
        raise ValueError("needs a Python expression over `x`, such as `x[:, -1, :]`")

    try:
        tree = ast.parse(expression, "<lambda-layer>", "eval")
    except SyntaxError as exc:
        raise ValueError(f"not a valid Python expression: {exc.msg}") from exc

    # Names are checked here, not left to fail when the model runs. An expression
    # that mentions something out of scope is wrong the moment it is typed, and
    # saying so then is the difference between a note under the field and a
    # `NameError` raised from inside Keras halfway through a forward pass [E-040].
    unknown = sorted(_free_names(tree) - _LAMBDA_NAMES)
    if unknown:
        raise ValueError(
            f"{', '.join(repr(name) for name in unknown)} "
            f"{'is' if len(unknown) == 1 else 'are'} not available here. An expression "
            f"may use `x` for the input tensor, `ops` for keras.ops, and `math` — "
            f"the expression itself, not the line of code that built the layer. "
            f"For example `x[:, -1, :]`."
        )

    code = compile(tree, "<lambda-layer>", "eval")
    namespace = {"__builtins__": {}, "ops": keras.ops, "math": math}

    def call(x: Any) -> Any:
        """
        > [!AML-DOC-UNIT]
        Evaluate the user's expression against one tensor.
        @param x the input tensor, bound to the name `x` in the expression
        @returns whatever the expression produces
        @raises Exception propagating any failure in the expression, which the
                compiler turns into a diagnostic on the offending node
        @sideEffects none beyond the expression's own effects, which cannot reach
                     the filesystem or network because builtins are stripped
        """
        return eval(code, namespace, {"x": x})

    call.__doc__ = f"Lambda expression: {expression}"
    return call


def coerce_params(spec: LayerSpec, params: dict[str, Any]) -> dict[str, Any]:
    """
    > [!AML-DOC-UNIT]
    Convert IR parameter values into what the Keras constructor expects.
    @param spec   catalog spec describing the parameter types
    @param params raw values from the IR, as they came out of JSON
    @returns constructor keyword arguments
    @sideEffects none
    @context JSON has no tuple type, so shapes and kernel sizes arrive as lists and
             must be converted back. Values that are None are omitted entirely so
             the Keras default stays authoritative rather than being overwritten by
             an explicit None [amm: E.4].
    """
    merged = {**spec.defaults(), **params}
    coerced: dict[str, Any] = {}

    for name, value in merged.items():
        param = spec.param(name)
        if param is None or value is None:
            continue
        if param.type in _TUPLE_TYPES and isinstance(value, list):
            coerced[name] = tuple(value)
        elif param.type is ParamType.INT and isinstance(value, float) and value.is_integer():
            coerced[name] = int(value)
        else:
            coerced[name] = value
    return coerced


def build_layer(
    node: Node,
    registry: Registry,
    graph: GraphIR,
    *,
    name: str | None = None,
    pool: NamePool | None = None,
) -> Any:
    """
    > [!AML-DOC-UNIT]
    Instantiate the Keras layer a node describes.
    @param node     the node to instantiate
    @param registry catalog used to resolve the type and coerce parameters
    @param graph    owning graph, needed to resolve a wrapped child node
    @param name     Keras layer name, pre-allocated by the caller so the naming
                    sequence is reproducible; falls back to the bare sanitized label
    @param pool     name pool used to name a wrapped child, drawn in the same order
                    the code generator draws it
    @returns an unbuilt Keras layer instance
    @raises KeyError when the node type is not in the catalog
    @raises ValueError when a Lambda expression is invalid or a wrapped child is
            missing
    @raises Exception propagating whatever the Keras constructor raises, so the
            caller can surface the real message
    @sideEffects constructs Keras objects, and recursively constructs the child of
                 a wrapper layer such as Bidirectional
    @context `trainable=True` is dropped rather than forwarded, because it is the
             Keras default and a few layers do not accept the keyword at all.
             Every layer is named explicitly: Keras would otherwise auto-generate
             names that differ from the exported file's [amm: E.4].
    """
    spec = registry.require(node.type)
    params = coerce_params(spec, node.params)
    if params.get("trainable", False) is True:
        params.pop("trainable")

    layer_param = spec.param("layer")
    if layer_param is not None and layer_param.type is ParamType.LAYER_REF:
        child_id = params.get("layer")
        child = graph.node_map().get(child_id) if isinstance(child_id, str) else None
        if child is None:
            raise ValueError(f"{spec.label} wraps node {child_id!r}, which is not in the graph")
        child_name = pool.allocate(child.name, child.id) if pool else None
        params["layer"] = build_layer(child, registry, graph, name=child_name, pool=pool)

    function_param = spec.param("function")
    if function_param is not None and isinstance(params.get("function"), str):
        params["function"] = _lambda_from_expression(params["function"])

    params.setdefault("name", name or sanitize_name(node.name, node.id))

    return resolve_keras_class(spec.keras_path)(**params)


def _gather_inputs(
    node: Node,
    spec: LayerSpec,
    graph: GraphIR,
    tensors: dict[str, Any],
) -> tuple[list[Any], dict[str, Any]]:
    """
    > [!AML-DOC-UNIT]
    Collect the tensors entering a node, split by how the layer wants to receive them.
    @param node    the consuming node
    @param spec    its catalog spec, which declares the call style and port order
    @param graph   owning graph, used to read the incoming edges
    @param tensors tensors produced so far, keyed by node id
    @returns (positional tensors in port order, keyword tensors by port name)
    @raises KeyError when a producer's tensor is absent, which cannot happen for a
            topologically ordered walk over a validated graph
    @sideEffects none
    """
    children = graph.child_node_ids()
    edges = [
        edge for edge in graph.incoming(node.id)
        if edge.source not in children and _source_tensor(edge, tensors) is not None
    ]

    if spec.call_style is CallStyle.QUERY_VALUE_KEY:
        keywords = {edge.target_port: _source_tensor(edge, tensors) for edge in edges}
        return [], keywords

    if spec.call_style is CallStyle.RECURRENT:
        # The sequence is positional; the starting state is a keyword, and a list
        # even when there is one of it, because that is what Keras accepts for both
        # the single-state layers and LSTM's pair [E-032].
        sequence = [
            _source_tensor(edge, tensors)
            for edge in edges if (edge.target_port or "input") == "input"
        ]
        state = [
            _source_tensor(edge, tensors)
            for edge in sorted(
                (edge for edge in edges if edge.target_port == "initial_state"),
                key=lambda edge: edge.order,
            )
        ]
        return sequence, ({"initial_state": state} if state else {})

    port_order = [port.name for port in spec.inputs] or [None]
    edges.sort(key=lambda edge: (
        port_order.index(edge.target_port) if edge.target_port in port_order else len(port_order),
        edge.order,
    ))
    return [_source_tensor(edge, tensors) for edge in edges], {}



def _port_key(node_id: str, index: int) -> str:
    """
    > [!AML-DOC-UNIT]
    The key one of a node's outputs is held under.
    @param node_id the producing node
    @param index   which output
    @returns the node's own id for the first output, a suffixed key for the rest
    @sideEffects none
    @context The first output keeps the bare node id, so every single-output layer
             and every existing project reads exactly as it did before.
    """
    return node_id if index == 0 else f"{node_id}::output_{index}"


def _source_tensor(edge: Any, tensors: dict[str, Any]) -> Any:
    """
    > [!AML-DOC-UNIT]
    The tensor an edge carries, which may not be its source's first output.
    @param edge    the edge
    @param tensors every tensor produced so far
    @returns the tensor, or None when the source has not produced one
    @sideEffects none
    """
    port = edge.source_port or "output"
    if port.startswith("output_"):
        return tensors.get(f"{edge.source}::{port}")
    return tensors.get(edge.source)

def _shape_of(tensor: Any, layer: Any | None) -> ShapeInfo:
    """
    > [!AML-DOC-UNIT]
    Read back the shape Keras assigned to a symbolic tensor.
    @param tensor the KerasTensor produced by calling a layer
    @param layer  the layer that produced it, or None for an Input node
    @returns ShapeInfo with the shape, dtype and the layer's weight count
    @sideEffects none
    """
    weights = 0
    if layer is not None:
        try:
            weights = int(layer.count_params())
        except (ValueError, AttributeError):
            weights = 0
    return ShapeInfo(
        shape=[None if dim is None else int(dim) for dim in tensor.shape],
        dtype=str(tensor.dtype),
        params=weights,
    )


def compile_graph(graph: GraphIR, registry: Registry) -> CompileResult:
    """
    > [!AML-DOC-UNIT]
    Validate a graph, then build it symbolically into a Keras model.
    @param graph    the architecture to compile
    @param registry the layer catalog
    @returns CompileResult holding the model, per-node shapes, diagnostics and
             parameter counts. On failure `model` is None and the diagnostics say
             which node broke and what Keras said about it.
    @raises nothing for any user-authored graph: Keras failures become diagnostics
    @sideEffects imports keras and constructs Keras layers and a model
    @context Inference is delegated to Keras by building through the functional API,
             so reported shapes are the ones training will actually see.
    """
    import keras

    diagnostics = validate_structure(graph, registry)
    if any(diagnostic.severity == "error" for diagnostic in diagnostics):
        return CompileResult(diagnostics=diagnostics)

    ordered, _ = graph.topological_order()
    node_map = graph.node_map()
    tensors: dict[str, Any] = {}
    shapes: dict[str, ShapeInfo] = {}
    pool = NamePool()

    for node_id in ordered:
        node = node_map[node_id]
        spec = registry.require(node.type)

        if node.disabled:
            positional, _ = _gather_inputs(node, spec, graph, tensors)
            if len(positional) != 1:
                diagnostics.append(Diagnostic(
                    severity="error", code=Codes.DISABLED_NODE_ARITY, node_id=node_id,
                    message=(
                        f"{spec.label} is disabled, which passes its input straight "
                        f"through, but it has {len(positional)} inputs instead of one."
                    ),
                ))
                return CompileResult(diagnostics=diagnostics, shapes=shapes)
            tensors[node_id] = positional[0]
            shapes[node_id] = _shape_of(positional[0], None)
            continue


        layer_name = pool.allocate(node.name, node_id)

        if spec.call_style is CallStyle.SOURCE:
            try:
                tensor = keras.Input(name=layer_name, **coerce_params(spec, node.params))
            except Exception as exc:
                diagnostics.append(Diagnostic(
                    severity="error", code=Codes.LAYER_CONSTRUCTION_FAILED, node_id=node_id,
                    message=f"{spec.label} could not be created: {exc}",
                ))
                return CompileResult(diagnostics=diagnostics, shapes=shapes)
            tensors[node_id] = tensor
            shapes[node_id] = _shape_of(tensor, None)
            continue

        try:
            layer = build_layer(node, registry, graph, name=layer_name, pool=pool)
        except ValueError as exc:
            code = (Codes.BAD_LAMBDA_EXPRESSION if spec.param("function") is not None
                    else Codes.LAYER_CONSTRUCTION_FAILED)
            diagnostics.append(Diagnostic(
                severity="error", code=code, node_id=node_id,
                message=f"{spec.label} could not be created: {exc}",
            ))
            return CompileResult(diagnostics=diagnostics, shapes=shapes)
        except Exception as exc:
            diagnostics.append(Diagnostic(
                severity="error", code=Codes.LAYER_CONSTRUCTION_FAILED, node_id=node_id,
                message=f"{spec.label} could not be created: {exc}",
            ))
            return CompileResult(diagnostics=diagnostics, shapes=shapes)

        positional, keywords = _gather_inputs(node, spec, graph, tensors)
        if not positional and not keywords:
            diagnostics.append(Diagnostic(
                severity="error", code=Codes.MISSING_INPUT_TENSOR, node_id=node_id,
                message=f"{spec.label} has no incoming data.",
            ))
            return CompileResult(diagnostics=diagnostics, shapes=shapes)

        try:
            if spec.call_style is CallStyle.QUERY_VALUE_KEY:
                tensor = layer(**keywords)
            elif spec.call_style is CallStyle.LIST:
                tensor = layer(positional)
            else:
                tensor = layer(*positional, **keywords)
        except Exception as exc:
            diagnostics.append(Diagnostic(
                severity="error", code=Codes.LAYER_CALL_FAILED, node_id=node_id,
                message=f"{spec.label} rejected its input: {exc}",
            ))
            return CompileResult(diagnostics=diagnostics, shapes=shapes)

        # A layer may return several tensors: a recurrent layer with
        # `return_state=True` returns its output and its state, and a free-running
        # model is built by feeding that state back [E-034]. The first is the node's
        # ordinary output; the rest are reachable through `output_1`, `output_2`, …
        if isinstance(tensor, (list, tuple)):
            for position, part in enumerate(tensor):
                tensors[_port_key(node_id, position)] = part
            tensor = tensor[0]
        tensors[node_id] = tensor
        shapes[node_id] = _shape_of(tensor, layer)

    input_ids = [node_id for node_id in graph.resolved_inputs() if node_id in tensors]
    output_ids = [node_id for node_id in graph.resolved_outputs() if node_id in tensors]

    try:
        model = keras.Model(
            inputs=[tensors[node_id] for node_id in input_ids],
            outputs=[tensors[node_id] for node_id in output_ids],
            name=sanitize_name(graph.name, "model"),
        )
    except Exception as exc:
        diagnostics.append(Diagnostic(
            severity="error", code=Codes.MODEL_ASSEMBLY_FAILED,
            message=f"The model could not be assembled: {exc}",
        ))
        return CompileResult(diagnostics=diagnostics, shapes=shapes)

    trainable = int(sum(math.prod(weight.shape) for weight in model.trainable_weights))
    total = trainable + int(
        sum(math.prod(weight.shape) for weight in model.non_trainable_weights)
    )
    return CompileResult(
        model=model,
        shapes=shapes,
        diagnostics=diagnostics,
        params_total=total,
        trainable_params=trainable,
    )
