"""Clinician–AI dyad: judge the clinician working WITH the AI, not the model alone.

Design: every case is read three ways against the same reference standard —
the AI alone, the clinician alone (before seeing the AI), and the clinician
with the AI output in view. All proportions carry Wilson 95% intervals; paired
differences use the exact McNemar test on the same cases.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Sequence

from .metrics import PairedComparison, Proportion, accuracy, mcnemar_exact, wilson


@dataclass(frozen=True)
class DyadResult:
    """Accuracy of each arm, paired comparisons, and reliance measures."""

    ai_alone: Proportion
    clinician_alone: Proportion
    clinician_with_ai: Proportion
    dyad_vs_clinician: PairedComparison
    dyad_vs_ai: PairedComparison
    accepted_wrong_ai: Proportion       # automation bias: followed the AI when the AI was wrong
    harmful_switch: Proportion          # was right alone, became wrong after seeing a wrong AI
    beneficial_switch: Proportion       # was wrong alone, became right after seeing a right AI
    overrode_right_ai: Proportion       # rejected a correct AI answer (algorithm aversion)

    def lines(self) -> list[str]:
        return [
            f"AI alone            : {self.ai_alone}",
            f"Clinician alone     : {self.clinician_alone}",
            f"Clinician with AI   : {self.clinician_with_ai}",
            f"Dyad vs clinician   : {self.dyad_vs_clinician}",
            f"Dyad vs AI          : {self.dyad_vs_ai}",
            f"Accepted wrong AI   : {self.accepted_wrong_ai}  (automation bias, among cases the AI got wrong)",
            f"Harmful switch      : {self.harmful_switch}  (right alone -> wrong with a wrong AI)",
            f"Beneficial switch   : {self.beneficial_switch}  (wrong alone -> right with a right AI)",
            f"Overrode a right AI : {self.overrode_right_ai}  (among cases the AI got right)",
        ]

    def __str__(self) -> str:
        return "\n".join(self.lines())


def _same_length(*seqs: Sequence) -> int:
    n = {len(s) for s in seqs}
    if len(n) != 1:
        raise ValueError(f"all inputs must have the same length, got {sorted(len(s) for s in seqs)}")
    (length,) = n
    if length == 0:
        raise ValueError("empty input")
    return length


def dyad_study(reference: Sequence[Hashable], ai: Sequence[Hashable],
               clinician_alone: Sequence[Hashable],
               clinician_with_ai: Sequence[Hashable]) -> DyadResult:
    """Analyse a three-arm reader study on the same cases."""
    _same_length(reference, ai, clinician_alone, clinician_with_ai)
    rows = list(zip(reference, ai, clinician_alone, clinician_with_ai))

    ai_wrong = [(r, a, c, d) for r, a, c, d in rows if a != r]
    ai_right = [(r, a, c, d) for r, a, c, d in rows if a == r]
    accepted = sum(d == a for r, a, c, d in ai_wrong)
    harmful_base = [(r, a, c, d) for r, a, c, d in ai_wrong if c == r]
    harmful = sum(d != r for r, a, c, d in harmful_base)
    beneficial_base = [(r, a, c, d) for r, a, c, d in ai_right if c != r]
    beneficial = sum(d == r for r, a, c, d in beneficial_base)
    overrode = sum(d != r for r, a, c, d in ai_right)

    return DyadResult(
        ai_alone=accuracy(reference, ai),
        clinician_alone=accuracy(reference, clinician_alone),
        clinician_with_ai=accuracy(reference, clinician_with_ai),
        dyad_vs_clinician=mcnemar_exact(reference, clinician_with_ai, clinician_alone),
        dyad_vs_ai=mcnemar_exact(reference, clinician_with_ai, ai),
        accepted_wrong_ai=wilson(accepted, len(ai_wrong)),
        harmful_switch=wilson(harmful, len(harmful_base)),
        beneficial_switch=wilson(beneficial, len(beneficial_base)),
        overrode_right_ai=wilson(overrode, len(ai_right)),
    )
