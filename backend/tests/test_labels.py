"""
> [!AML-DOC-FILE]
@file       tests/test_labels.py
@description Regression tests for reading a file of output names and putting each
             vocabulary on the layer it belongs to.
@module     tests.test_labels
@exports    (pytest test functions)
@created    2026-10-01
@context    A model's outputs are indices, and an index is not an answer [E-042]. The
            file that prompted these holds six vocabularies of three different sizes,
            three of them stored backwards, for a model with eight outputs — so every
            awkward part of the problem is in one real file.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from nnarch.catalog import Registry, bootstrap
from nnarch.data.labels import match_to_size, parse_label_file
from nnarch.data.spec import DatasetChoice
from nnarch.data import make_adapter
from nnarch.ir import GraphIR, compile_graph
from nnarch.ir.schema import Edge, Node
from nnarch.viz import capture_activations


@pytest.fixture(scope="module")
def registry() -> Registry:
    """
    > [!AML-DOC-UNIT]
    The catalog, built once.
    @returns the bootstrapped registry
    """
    return bootstrap()


def test_every_shape_of_labels_file_is_read() -> None:
    """
    > [!AML-DOC-UNIT]
    A list, an index-to-name mapping, a name-to-index mapping and a text file.
    @raises AssertionError when any of the four is misread
    @context All four are shapes people actually have, which is the only reason the
             reader accepts four.
    """
    cases = [
        (json.dumps(["cat", "dog"]).encode(), "a.json"),
        (json.dumps({"0": "cat", "1": "dog"}).encode(), "b.json"),
        (json.dumps({"cat": 0, "dog": 1}).encode(), "c.json"),
        (b"cat\ndog\n", "d.txt"),
        (b"0,cat\n1,dog\n", "e.csv"),
    ]
    for payload, name in cases:
        parsed = parse_label_file(payload, name)
        assert parsed.vocabularies[0].names == ["cat", "dog"], name


def test_the_direction_that_needed_no_guessing_wins() -> None:
    """
    > [!AML-DOC-UNIT]
    An index-to-name mapping beats the inverse of an alias table of the same length.
    @raises AssertionError when the inverted one is chosen
    @context This is not cosmetic. In the file that prompted this, inverting
             `chord2idx` gives `A:7(b9)/5` for the index whose real name, in that same
             file's `idx2chord`, is `A:maj`. Several names share an index in an alias
             table and picking one of them is a guess.
    """
    document = {
        "name2idx": {"A:maj": 0, "A:maj/3": 0, "A:min": 1, "A:min/b3": 1},
        "idx2name": {"0": "A:maj", "1": "A:min"},
    }
    parsed = parse_label_file(json.dumps(document).encode(), "vocab.json")
    by_size = match_to_size(parsed)

    assert by_size[2] == ["A:maj", "A:min"]
    inverted = {v.name: v.inverted for v in parsed.vocabularies}
    assert inverted["name2idx"] is True
    assert inverted["idx2name"] is False


def test_a_file_with_no_names_in_it_is_refused() -> None:
    """
    > [!AML-DOC-UNIT]
    Something that is not a vocabulary is an error, not an empty list.
    @raises AssertionError when rubbish is accepted
    """
    with pytest.raises(ValueError):
        parse_label_file(b"{}", "empty.json")
    with pytest.raises(ValueError):
        parse_label_file(b"not json at all", "x.json")


def test_each_output_gets_the_vocabulary_of_its_own_size(registry: Registry) -> None:
    """
    > [!AML-DOC-UNIT]
    Two heads of different widths get different names.
    @param registry the catalog
    @raises AssertionError when one list is put on both
    @context `class_names` was a single flat list applied to anything that looked like
             probabilities, so a model with a 4-class head and a 3-class head would
             have had the same names on both [E-042].
    """
    graph = GraphIR(
        name="two-heads",
        nodes=[
            Node(id="in", type="keras.Input", name="in", params={"shape": [8]}),
            Node(id="wide", type="keras.Dense", name="wide",
                 params={"units": 4, "activation": "softmax"}),
            Node(id="narrow", type="keras.Dense", name="narrow",
                 params={"units": 3, "activation": "softmax"}),
        ],
        edges=[
            Edge(id="e1", source="in", target="wide"),
            Edge(id="e2", source="in", target="narrow"),
        ],
    )
    assert compile_graph(graph, registry).model is not None

    result = capture_activations(
        graph, registry,
        make_adapter(DatasetChoice(kind="synthetic"), [8], 4),
        colormap="viridis",
        labels_by_size={4: ["n", "e", "s", "w"], 3: ["red", "green", "blue"]},
    )
    by_label = {a.label: a.labels for a in result.activations}
    assert by_label["wide"] == ["n", "e", "s", "w"]
    assert by_label["narrow"] == ["red", "green", "blue"]


def test_names_do_not_need_the_values_to_be_a_distribution(registry: Registry) -> None:
    """
    > [!AML-DOC-UNIT]
    A head that emits logits is labelled too.
    @param registry the catalog
    @raises AssertionError when only softmax outputs get names
    @context Names were attached only to outputs whose values summed to one, so a head
             whose softmax lives in the loss showed bare indices although its names
             were sitting right there. If a vocabulary is as long as the layer is
             wide, it belongs to it.
    """
    graph = GraphIR(
        name="logits",
        nodes=[
            Node(id="in", type="keras.Input", name="in", params={"shape": [8]}),
            Node(id="head", type="keras.Dense", name="head", params={"units": 3}),
        ],
        edges=[Edge(id="e", source="in", target="head")],
    )
    result = capture_activations(
        graph, registry,
        make_adapter(DatasetChoice(kind="synthetic"), [8], 3),
        colormap="viridis",
        labels_by_size={3: ["red", "green", "blue"]},
    )
    head = next(a for a in result.activations if a.label == "head")
    assert head.labels == ["red", "green", "blue"]
    assert abs(float(np.sum(head.series)) - 1.0) > 1e-3, "this test needs non-probabilities"


def test_a_layer_with_no_matching_vocabulary_keeps_its_indices(
    registry: Registry,
) -> None:
    """
    > [!AML-DOC-UNIT]
    Nothing is invented for a width the file does not cover.
    @param registry the catalog
    @raises AssertionError when names are stretched or truncated onto a layer
    @context Truncating a 170-name list onto a 12-wide output would label it, wrongly
             and convincingly. An unlabelled axis is the honest answer.
    """
    graph = GraphIR(
        name="unmatched",
        nodes=[
            Node(id="in", type="keras.Input", name="in", params={"shape": [8]}),
            Node(id="head", type="keras.Dense", name="head",
                 params={"units": 5, "activation": "softmax"}),
        ],
        edges=[Edge(id="e", source="in", target="head")],
    )
    result = capture_activations(
        graph, registry,
        make_adapter(DatasetChoice(kind="synthetic"), [8], 5),
        colormap="viridis",
        labels_by_size={170: [f"chord {i}" for i in range(170)]},
    )
    head = next(a for a in result.activations if a.label == "head")
    assert head.labels == [], f"names were invented for a 5-wide layer: {head.labels}"
