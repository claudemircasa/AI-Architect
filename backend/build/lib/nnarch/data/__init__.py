"""
> [!AML-DOC-FILE]
@file       data/__init__.py
@description Data sources for previewing activations, training in the app, and
             loading data in exported projects.
@module     nnarch.data
@exports    DatasetAdapter, DatasetChoice, DatasetError, DatasetKind, DatasetSpec,
            Modality, BUILTIN_DATASETS, make_adapter, decode_upload, detect_modality
@created    2026-10-01
@context    [amm: E.4] each adapter both provides data and emits the code that loads
            it, so the app and an exported project cannot preprocess differently.
"""

from __future__ import annotations

from .adapters import (
    BuiltinAdapter,
    DatasetAdapter,
    DatasetError,
    FolderAdapter,
    SyntheticAdapter,
    UploadAdapter,
    decode_upload,
    detect_modality,
    make_adapter,
)
from .spec import (
    BUILTIN_DATASETS,
    DatasetChoice,
    DatasetKind,
    DatasetSpec,
    Modality,
)

__all__ = [
    "BUILTIN_DATASETS",
    "BuiltinAdapter",
    "DatasetAdapter",
    "DatasetChoice",
    "DatasetError",
    "DatasetKind",
    "DatasetSpec",
    "FolderAdapter",
    "Modality",
    "SyntheticAdapter",
    "UploadAdapter",
    "decode_upload",
    "detect_modality",
    "make_adapter",
]
