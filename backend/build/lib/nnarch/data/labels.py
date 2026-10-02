"""
> [!AML-DOC-FILE]
@file        src/nnarch/data/labels.py
@description Read a file of output names and match each vocabulary to the layer it
             belongs to.
@module      nnarch.data.labels
@exports     Vocabulary, parse_label_file, match_to_size
@created     2026-10-01
@context     A model's outputs are indices, and the index is not the answer: class 47
             means nothing without the list that says what 47 is. The lists live in
             whatever file the training pipeline built, so they are read from there
             rather than typed in [E-042].

             A real model has several outputs with different vocabularies, and a real
             vocabulary file holds several of them side by side. Matching them by
             *length* is what makes that work without asking anyone to wire names to
             layers by hand: a vocabulary of 170 names belongs to the output with 170
             classes, and when two outputs have 170 it belongs to both, which is
             right — they are the same vocabulary read twice.
"""

from __future__ import annotations

import csv
import io
import json
from typing import Any

from pydantic import BaseModel, Field


class Vocabulary(BaseModel):
    """
    > [!AML-DOC-UNIT]
    One list of names, and where it came from.
    @param name     the key it was found under, or the file's name for a bare list
    @param names    the labels, in index order
    @param inverted True when it was stored name-to-index and had to be turned round
    """

    name: str
    names: list[str] = Field(default_factory=list)
    inverted: bool = False


class LabelFile(BaseModel):
    """
    > [!AML-DOC-UNIT]
    Everything read out of one labels file.
    @param filename     what the user picked
    @param vocabularies the lists it held
    @param note         one sentence describing what was found
    """

    filename: str = ""
    vocabularies: list[Vocabulary] = Field(default_factory=list)
    note: str = ""


def _ordered_from_mapping(mapping: dict[Any, Any]) -> tuple[list[str], bool] | None:
    """
    > [!AML-DOC-UNIT]
    Turn an index-to-name mapping into a list in index order.
    @param mapping keys that are integers, or strings that spell integers
    @returns (the names in order, whether the mapping had to be inverted), or None
             when it is not a vocabulary at all
    @sideEffects none
    @context JSON has no integer keys, so `{"0": "C", "1": "C#"}` arrives with string
             keys and has to be read back. A mapping the other way round — name to
             index, as `chord2idx` is — is inverted here rather than refused, because
             both directions are ordinary ways to save a vocabulary.
    """
    if not mapping:
        return None

    def as_index(value: Any) -> int | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                return None
        return None

    keys = [as_index(key) for key in mapping]
    if all(key is not None for key in keys):
        pairs = sorted(zip(keys, mapping.values()), key=lambda pair: pair[0])  # type: ignore[arg-type]
        return [str(name) for _, name in pairs], False

    values = [as_index(value) for value in mapping.values()]
    if all(value is not None for value in values):
        # name -> index. Several names can share an index, which is how an alias table
        # is written, and picking one of them is a guess: this file's `chord2idx`
        # inverts to `A:7(b9)/5` where its own `idx2chord` says `A:maj`. The
        # inversion is kept, and marked, so the direction that needed no guessing
        # wins when both are present [E-042].
        ordered: dict[int, str] = {}
        for name, index in zip(mapping.keys(), values):
            ordered.setdefault(int(index), str(name))  # type: ignore[arg-type]
        if ordered:
            return [ordered.get(i, str(i)) for i in range(max(ordered) + 1)], True
    return None


def parse_label_file(payload: bytes, filename: str) -> LabelFile:
    """
    > [!AML-DOC-UNIT]
    Read whichever shape of labels file the user has.
    @param payload  the file's bytes
    @param filename its name, which decides how it is read
    @returns every vocabulary found, and a sentence describing them
    @raises ValueError when nothing in the file reads as a list of names
    @sideEffects none
    @context Four shapes are accepted because all four are what people actually have:
             a bare list, an index-to-name mapping, a name-to-index mapping, and a
             file of several of those under named keys. A plain text or CSV file is
             one name per line.
    """
    stem = filename.rsplit("/", 1)[-1] or "labels"

    if filename.lower().endswith((".txt", ".csv", ".tsv")):
        text = payload.decode("utf-8", errors="replace")
        if filename.lower().endswith((".csv", ".tsv")):
            delimiter = "\t" if filename.lower().endswith(".tsv") else ","
            rows = list(csv.reader(io.StringIO(text), delimiter=delimiter))
            # The last column of each row, so "0,C" and "C" both work.
            names = [row[-1].strip() for row in rows if row and row[-1].strip()]
        else:
            names = [line.strip() for line in text.splitlines() if line.strip()]
        if not names:
            raise ValueError(f"'{stem}' has no names in it.")
        return LabelFile(
            filename=stem,
            vocabularies=[Vocabulary(name=stem, names=names)],
            note=f"'{stem}': {len(names)} names.",
        )

    try:
        document = json.loads(payload.decode("utf-8", errors="replace"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"'{stem}' is not readable as JSON or as a list of lines: {exc}")

    found: list[Vocabulary] = []

    if isinstance(document, list):
        found.append(Vocabulary(name=stem, names=[str(item) for item in document]))
    elif isinstance(document, dict):
        direct = _ordered_from_mapping(document)
        nested = {
            key: value
            for key, value in document.items()
            if isinstance(value, (list, dict)) and value
        }
        if nested:
            for key, value in nested.items():
                if isinstance(value, list):
                    read: tuple[list[str], bool] | None = ([str(i) for i in value], False)
                else:
                    read = _ordered_from_mapping(value)
                if read:
                    found.append(
                        Vocabulary(name=str(key), names=read[0], inverted=read[1])
                    )
        if not found and direct:
            found.append(Vocabulary(name=stem, names=direct[0], inverted=direct[1]))

    if not found:
        raise ValueError(
            f"'{stem}' holds no list of names. A labels file is a list, a mapping of "
            f"index to name, a mapping of name to index, or several of those under "
            f"named keys."
        )

    # Longest first, and at equal length the one that was already index-to-name,
    # because inverting an alias table picks an arbitrary name per index [E-042].
    found.sort(key=lambda vocabulary: (-len(vocabulary.names), vocabulary.inverted))
    described = ", ".join(f"{v.name} ({len(v.names)})" for v in found[:6])
    more = f", and {len(found) - 6} more" if len(found) > 6 else ""
    return LabelFile(
        filename=stem,
        vocabularies=found,
        note=f"'{stem}': {described}{more}.",
    )


def match_to_size(label_file: LabelFile) -> dict[int, list[str]]:
    """
    > [!AML-DOC-UNIT]
    Index the vocabularies by how many names they hold.
    @param label_file the parsed file
    @returns {count: names}, so a layer with that many outputs can find its own
    @sideEffects none
    @context Length is the only thing a vocabulary and a layer share, and it is enough
             in practice: a 170-name list belongs to the 170-class output. Where two
             are the same length, the one that did not have to be inverted wins.
    """
    by_size: dict[int, list[str]] = {}
    for vocabulary in label_file.vocabularies:
        by_size.setdefault(len(vocabulary.names), vocabulary.names)
    return by_size
