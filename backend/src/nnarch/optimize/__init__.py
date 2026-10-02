"""
> [!AML-DOC-FILE]
@file        src/nnarch/optimize/__init__.py
@description Architecture rewriting: proposals that simplify or modernise a graph,
             each measured rather than asserted.
@module      nnarch.optimize
@exports     Proposal, analyse, apply_proposals
@created     2026-10-02
@context     Two kinds of change, and the difference between them is the whole point.
             An *exact* rewrite leaves a model that computes the same function — the
             identity layer that does nothing, the reshape to the shape it already
             has. A *substitution* leaves a different model that fills the same role,
             and it has to be retrained. Presenting the second as though it were the
             first would turn a tool into a trap [E-054].
"""

from .rules import Proposal, RuleKind, analyse_rules
from .report import Measurement, OptimizeReport, analyse, apply_proposals

__all__ = [
    "Measurement",
    "OptimizeReport",
    "Proposal",
    "RuleKind",
    "analyse",
    "analyse_rules",
    "apply_proposals",
]
