"""
> [!AML-DOC-FILE]
@file       tests/conftest.py
@description Makes the test fixtures package importable by dotted name, which the
             source-inlining exporter needs in order to resolve a layer's module.
@module     tests.conftest
@exports    (pytest configuration side effect)
@created    2026-09-30
@context    `export.inline` resolves classes from `LayerSpec.keras_path`, so a
            fixture layer must be reachable as `fixtures.<module>.<Class>`.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
