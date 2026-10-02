"""
> [!AML-DOC-FILE]
@file       data/spec.py
@description Declarative description of a data source and what it produces.
@module     nnarch.data.spec
@exports    Modality, DatasetKind, DatasetChoice, DatasetSpec, BUILTIN_DATASETS
@created    2026-10-01
@context    The choice travels from the editor to the activation preview [task 05],
            the training runtime [task 07] and the exported project [task 06]. One
            description, read by all three, is what keeps in-app preprocessing and
            exported preprocessing identical [amm: E.4].
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class DatasetKind(str, Enum):
    """
    > [!AML-DOC-UNIT]
    Where a sample comes from.

    - `synthetic`: generated noise of the right shape. Needs no network and no
      files, so it always works and is the default for previewing a graph.
    - `builtin`: one of the datasets Keras can download.
    - `upload`: a single file the user dropped in, for watching one example flow
      through the layers.
    - `folder`: a directory on disk, laid out as Keras expects.
    """

    SYNTHETIC = "synthetic"
    BUILTIN = "builtin"
    UPLOAD = "upload"
    FOLDER = "folder"


class Modality(str, Enum):
    """
    > [!AML-DOC-UNIT]
    The kind of thing a sample is, which decides how it is read from disk and how
    it is drawn on screen [task 11].
    """

    IMAGE = "image"
    TEXT = "text"
    AUDIO = "audio"
    TABULAR = "tabular"
    TENSOR = "tensor"


BUILTIN_DATASETS: dict[str, dict[str, Any]] = {
    "mnist": {
        "label": "MNIST handwritten digits",
        "classes": 10, "modality": Modality.IMAGE, "shape": [28, 28, 1],
        "class_names": [str(digit) for digit in range(10)],
    },
    "fashion_mnist": {
        "label": "Fashion-MNIST clothing",
        "classes": 10, "modality": Modality.IMAGE, "shape": [28, 28, 1],
        "class_names": ["t-shirt", "trouser", "pullover", "dress", "coat",
                        "sandal", "shirt", "sneaker", "bag", "ankle boot"],
    },
    "cifar10": {
        "label": "CIFAR-10 small photographs",
        "classes": 10, "modality": Modality.IMAGE, "shape": [32, 32, 3],
        "class_names": ["airplane", "automobile", "bird", "cat", "deer",
                        "dog", "frog", "horse", "ship", "truck"],
    },
    "cifar100": {
        "label": "CIFAR-100 small photographs",
        "classes": 100, "modality": Modality.IMAGE, "shape": [32, 32, 3],
        "class_names": [],
    },
    "imdb": {
        "label": "IMDB film reviews",
        "classes": 2, "modality": Modality.TEXT, "shape": None,
        "class_names": ["negative", "positive"],
    },
    "reuters": {
        "label": "Reuters newswires",
        "classes": 46, "modality": Modality.TEXT, "shape": None,
        "class_names": [],
    },
}
"""
> [!AML-DOC-UNIT]
Datasets `tf.keras.datasets` can fetch, with everything needed to emit a loader and
to label a prediction: class count, modality, the sample shape the raw arrays must be
reshaped to, and human-readable class names where they exist.
"""


class DatasetChoice(BaseModel):
    """
    > [!AML-DOC-UNIT]
    Which data to use, as chosen in the editor.
    @param kind      where the data comes from
    @param name      dataset name when `kind` is `builtin`
    @param path      directory when `kind` is `folder`
    @param samples   how many examples to generate when `kind` is `synthetic`
    @param seed      seed for synthetic generation, so a preview is reproducible
    @param pattern   generator for synthetic data: noise, ones, zeros, ramp,
                     checkerboard or sine
    @param text      the text to feed when previewing a text model
    @param validation_split fraction of a folder dataset held back for validation
    """

    kind: DatasetKind = DatasetKind.SYNTHETIC
    name: str = "mnist"
    path: str = ""
    samples: int = 512
    seed: int = 0
    pattern: Literal["noise", "ones", "zeros", "ramp", "checkerboard", "sine"] = "noise"
    text: str = ""
    validation_split: float = 0.2


class DatasetSpec(BaseModel):
    """
    > [!AML-DOC-UNIT]
    What a source actually provides, resolved against the model it feeds.
    @param modality    kind of data, which picks the renderer [task 11]
    @param input_shape shape of one example, excluding the batch axis
    @param num_classes number of labels, or None for regression
    @param class_names human-readable labels, when the source knows them
    @param count       examples available, when known
    @param label       display name for the source
    """

    modality: Modality = Modality.TENSOR
    input_shape: list[int] = Field(default_factory=list)
    num_classes: int | None = None
    class_names: list[str] = Field(default_factory=list)
    count: int | None = None
    label: str = ""
