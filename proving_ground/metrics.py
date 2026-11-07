"""Metrics for testing an AI tool (or a human rater) against a reference standard.

Every proportion comes with a Wilson score interval, which stays inside [0, 1]
even for small samples and proportions near 0 or 1.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Sequence

import numpy as np
from scipy.stats import binomtest, norm


@dataclass(frozen=True)
class Proportion:
    """A count-based proportion with its Wilson score interval."""

    k: int
    n: int
    estimate: float
    low: float
    high: float

    def __str__(self) -> str:
        if self.n == 0:
            return "n/a (n = 0)"
        return (f"{100 * self.estimate:.1f}% ({self.k}/{self.n}; "
                f"95% CI {100 * self.low:.1f}–{100 * self.high:.1f}%)")


def wilson(k: int, n: int, confidence: float = 0.95) -> Proportion:
    """Wilson score interval for k successes out of n trials."""
    if n < 0 or k < 0 or k > n:
        raise ValueError(f"need 0 <= k <= n, got k={k}, n={n}")
    if n == 0:
        return Proportion(k, n, float("nan"), float("nan"), float("nan"))
    z = norm.ppf(1 - (1 - confidence) / 2)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return Proportion(k, n, p, max(0.0, centre - half), min(1.0, centre + half))


def _check_pair(a: Sequence, b: Sequence) -> None:
    if len(a) != len(b):
        raise ValueError(f"length mismatch: {len(a)} vs {len(b)}")
    if len(a) == 0:
        raise ValueError("empty input")


def accuracy(reference: Sequence[Hashable], predicted: Sequence[Hashable]) -> Proportion:
    """Share of cases where the prediction equals the reference label."""
    _check_pair(reference, predicted)
    k = sum(r == p for r, p in zip(reference, predicted))
    return wilson(k, len(reference))


def confusion_matrix(reference: Sequence[Hashable], predicted: Sequence[Hashable],
                     labels: Sequence[Hashable] | None = None) -> tuple[list, np.ndarray]:
    """Rows are reference labels, columns are predicted labels."""
    _check_pair(reference, predicted)
    if labels is None:
        labels = sorted(set(reference) | set(predicted), key=str)
    index = {lab: i for i, lab in enumerate(labels)}
    m = np.zeros((len(labels), len(labels)), dtype=int)
    for r, p in zip(reference, predicted):
        m[index[r], index[p]] += 1
    return list(labels), m


def binary_metrics(reference: Sequence[Hashable], predicted: Sequence[Hashable],
                   positive: Hashable) -> dict[str, Proportion]:
    """Sensitivity, specificity, PPV and NPV for one label treated as 'positive'."""
    _check_pair(reference, predicted)
    tp = sum(r == positive and p == positive for r, p in zip(reference, predicted))
    fn = sum(r == positive and p != positive for r, p in zip(reference, predicted))
    tn = sum(r != positive and p != positive for r, p in zip(reference, predicted))
    fp = sum(r != positive and p == positive for r, p in zip(reference, predicted))
    return {
        "sensitivity": wilson(tp, tp + fn),
        "specificity": wilson(tn, tn + fp),
        "ppv": wilson(tp, tp + fp),
        "npv": wilson(tn, tn + fn),
    }


def cohen_kappa(a: Sequence[Hashable], b: Sequence[Hashable]) -> float:
    """Cohen's kappa: agreement between two raters beyond chance."""
    labels, m = confusion_matrix(a, b)
    n = m.sum()
    observed = np.trace(m) / n
    expected = float((m.sum(axis=1) * m.sum(axis=0)).sum()) / n**2
    if expected == 1.0:
        return 1.0
    return float((observed - expected) / (1 - expected))


@dataclass(frozen=True)
class PairedComparison:
    """Exact McNemar test on paired right/wrong outcomes of two raters."""

    only_a_correct: int
    only_b_correct: int
    p_value: float

    def __str__(self) -> str:
        return (f"discordant pairs: A right/B wrong = {self.only_a_correct}, "
                f"A wrong/B right = {self.only_b_correct}; exact McNemar p = {self.p_value:.4f}")


def mcnemar_exact(reference: Sequence[Hashable], rater_a: Sequence[Hashable],
                  rater_b: Sequence[Hashable]) -> PairedComparison:
    """Compare two raters judged on the same cases against the same reference."""
    _check_pair(reference, rater_a)
    _check_pair(reference, rater_b)
    a_ok = [r == x for r, x in zip(reference, rater_a)]
    b_ok = [r == x for r, x in zip(reference, rater_b)]
    only_a = sum(x and not y for x, y in zip(a_ok, b_ok))
    only_b = sum(y and not x for x, y in zip(a_ok, b_ok))
    discordant = only_a + only_b
    p = 1.0 if discordant == 0 else binomtest(only_a, discordant, 0.5).pvalue
    return PairedComparison(only_a, only_b, float(p))


def brier_score(outcome: Sequence[int], probability: Sequence[float]) -> float:
    """Mean squared error of predicted probabilities for a 0/1 outcome (lower is better)."""
    _check_pair(outcome, probability)
    y = np.asarray(outcome, dtype=float)
    p = np.asarray(probability, dtype=float)
    if np.any((p < 0) | (p > 1)):
        raise ValueError("probabilities must be in [0, 1]")
    return float(np.mean((p - y) ** 2))


def calibration_table(outcome: Sequence[int], probability: Sequence[float],
                      bins: int = 5) -> list[dict]:
    """Observed event rate against mean predicted probability, in equal-width bins."""
    _check_pair(outcome, probability)
    y = np.asarray(outcome, dtype=float)
    p = np.asarray(probability, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if mask.any():
            rows.append({"bin": f"{lo:.1f}–{hi:.1f}", "n": int(mask.sum()),
                         "mean_predicted": float(p[mask].mean()),
                         "observed_rate": float(y[mask].mean())})
    return rows
