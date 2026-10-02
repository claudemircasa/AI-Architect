"""
> [!AML-DOC-FILE]
@file        src/nnarch/__main__.py
@description Module entry point, so the engine can be started as `python -m nnarch`.
@module      nnarch.__main__
@exports     (executable module)
@created     2026-10-01
@context     The packaged desktop app runs the engine from a Python runtime copied
             inside the `.app`. A console script cannot be used there: pip writes its
             interpreter path into the script's shebang at install time, and that path
             stops existing the moment the runtime is copied somewhere else. Running
             the module instead means the only path that matters is the interpreter
             the shell invokes, which it knows [E-026].
"""

import sys

from nnarch.api.server import main

if __name__ == "__main__":
    sys.exit(main())
