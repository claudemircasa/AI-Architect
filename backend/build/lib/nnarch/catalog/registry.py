"""
> [!AML-DOC-FILE]
@file       catalog/registry.py
@description In-process store of every placeable LayerSpec, plus the bootstrap that
             populates it from the Keras taxonomy and the research-layer families.
@module     nnarch.catalog.registry
@exports    Registry, registry, bootstrap
@created    2026-09-30
@context    RISK:HIGH [amm: B.1]. Consumed by the palette, the property panel, the
            graph validator and the code generator. Spec ids are persisted inside
            saved projects, so they must never be renamed [amm: E.2].
"""

from __future__ import annotations

import threading
from collections import defaultdict
from typing import Any, Iterable

from .spec import Category, LayerSpec


class DuplicateSpecError(ValueError):
    """
    > [!AML-DOC-UNIT]
    Raised when two LayerSpecs claim the same id. This is a programming error in
    the catalog tables, never a user error.
    """


class Registry:
    """
    > [!AML-DOC-UNIT]
    Thread-safe collection of LayerSpecs keyed by stable id.
    @sideEffects mutates internal dicts on register; reads are lock-free once built
    """

    def __init__(self) -> None:
        self._specs: dict[str, LayerSpec] = {}
        self._lock = threading.Lock()
        self._bootstrapped = False

    def register(self, spec: LayerSpec) -> LayerSpec:
        """
        > [!AML-DOC-UNIT]
        Add one spec to the registry.
        @param spec the LayerSpec to store
        @returns the same spec, for fluent use
        @raises DuplicateSpecError when the id is already taken
        """
        with self._lock:
            if spec.id in self._specs:
                raise DuplicateSpecError(f"layer spec id already registered: {spec.id!r}")
            self._specs[spec.id] = spec
        return spec

    def register_all(self, specs: Iterable[LayerSpec]) -> None:
        """
        > [!AML-DOC-UNIT]
        Add many specs.
        @param specs iterable of LayerSpec
        @raises DuplicateSpecError on the first colliding id
        """
        for spec in specs:
            self.register(spec)

    def get(self, spec_id: str) -> LayerSpec | None:
        """
        > [!AML-DOC-UNIT]
        Look up a spec by id.
        @param spec_id stable layer identifier, e.g. "keras.Conv2D"
        @returns the LayerSpec, or None when unknown
        """
        return self._specs.get(spec_id)

    def require(self, spec_id: str) -> LayerSpec:
        """
        > [!AML-DOC-UNIT]
        Look up a spec, failing loudly when absent.
        @param spec_id stable layer identifier
        @returns the LayerSpec
        @raises KeyError when the id is not registered
        """
        spec = self._specs.get(spec_id)
        if spec is None:
            raise KeyError(f"unknown layer type: {spec_id!r}")
        return spec

    def all(self) -> list[LayerSpec]:
        """
        > [!AML-DOC-UNIT]
        Every registered spec.
        @returns list of LayerSpec sorted by category then label
        """
        return sorted(self._specs.values(), key=lambda s: (s.category.value, s.label))

    def by_category(self) -> dict[str, list[LayerSpec]]:
        """
        > [!AML-DOC-UNIT]
        Group specs for palette rendering.
        @returns mapping of category value to its specs, label-sorted
        """
        grouped: dict[str, list[LayerSpec]] = defaultdict(list)
        for spec in self.all():
            grouped[spec.category.value].append(spec)
        return dict(grouped)

    def as_json(self) -> dict[str, Any]:
        """
        > [!AML-DOC-UNIT]
        Serialize the whole catalog for the `GET /catalog` response.
        @returns dict with the category tree, flat spec list and enum vocabularies
                 the frontend needs to render select controls
        @sideEffects none
        """
        from .introspect import ACTIVATIONS, CONSTRAINTS, DTYPES, INITIALIZERS, REGULARIZERS

        return {
            "count": len(self._specs),
            "categories": [
                {
                    "id": category,
                    "label": _CATEGORY_LABELS.get(category, category),
                    "layers": [spec.model_dump(mode="json") for spec in specs],
                }
                for category, specs in sorted(self.by_category().items())
            ],
            "vocabularies": {
                "activations": ACTIVATIONS,
                "initializers": INITIALIZERS,
                "regularizers": REGULARIZERS,
                "constraints": CONSTRAINTS,
                "dtypes": DTYPES,
            },
        }

    def bootstrap(self) -> Registry:
        """
        > [!AML-DOC-UNIT]
        Populate the registry once, from the Keras taxonomy and the research
        families. Safe to call repeatedly; later calls are no-ops.
        @returns self
        @sideEffects imports TensorFlow/Keras, which is slow and therefore deferred
                     until the first catalog request [task 01]
        """
        with self._lock:
            if self._bootstrapped:
                return self
            self._bootstrapped = True

        from .keras_specs import build_keras_specs

        self.register_all(build_keras_specs())

        try:
            from .research_specs import build_research_specs
        except ImportError:
            return self
        self.register_all(build_research_specs())
        return self


_CATEGORY_LABELS: dict[str, str] = {
    Category.CORE.value: "Core",
    Category.CONVOLUTION.value: "Convolution",
    Category.POOLING.value: "Pooling",
    Category.RECURRENT.value: "Recurrent",
    Category.NORMALIZATION.value: "Normalization",
    Category.REGULARIZATION.value: "Regularization",
    Category.ATTENTION.value: "Attention",
    Category.RESHAPING.value: "Reshaping",
    Category.MERGING.value: "Merging",
    Category.ACTIVATION.value: "Activation",
    Category.PREPROCESSING.value: "Preprocessing",
    Category.RESEARCH_SSM.value: "Research / State-Space (Mamba, S4)",
    Category.RESEARCH_KAN.value: "Research / Kolmogorov-Arnold",
    Category.RESEARCH_ATTENTION.value: "Research / Attention variants",
    Category.RESEARCH_MOE.value: "Research / Mixture of Experts",
    Category.RESEARCH_FFN.value: "Research / Gated feed-forward",
    Category.RESEARCH_RECURRENT.value: "Research / Modern recurrent",
    Category.RESEARCH_CONV.value: "Research / Modern convolution",
    Category.RESEARCH_MIXER.value: "Research / Token mixing",
    Category.RESEARCH_GRAPH.value: "Research / Graph neural networks",
    Category.RESEARCH_NEURO.value: "Research / Spiking & neuromorphic",
    Category.RESEARCH_MISC.value: "Research / Other",
}

registry = Registry()


def bootstrap() -> Registry:
    """
    > [!AML-DOC-UNIT]
    Module-level convenience for populating the shared registry.
    @returns the bootstrapped singleton Registry
    """
    return registry.bootstrap()
