"""After deployment: drift checks against a pre-registered threshold, and site-by-site transportability.

The rule is fixed BEFORE data arrive: a minimum acceptable accuracy. For each
period (or site) the toolkit reports accuracy with its Wilson 95% interval and a
status:

- "ok"          the point estimate meets the threshold;
- "watch"       the point estimate is below the threshold, but the interval still includes it;
- "requalify"   the whole interval lies below the threshold, so performance is
                credibly worse than the pre-registered minimum and the tool should
                be re-qualified (re-tested, recalibrated or withdrawn) before further use;
- "too few"     fewer cases than the pre-registered minimum sample per period.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Sequence

from .metrics import Proportion, wilson


@dataclass(frozen=True)
class GroupCheck:
    group: Hashable
    accuracy: Proportion
    status: str

    def __str__(self) -> str:
        return f"{str(self.group):>12} : {self.accuracy}  -> {self.status}"


def _status(p: Proportion, threshold: float, min_n: int) -> str:
    if p.n < min_n:
        return "too few"
    if p.high < threshold:
        return "requalify"
    if p.estimate < threshold:
        return "watch"
    return "ok"


def _grouped(reference: Sequence[Hashable], predicted: Sequence[Hashable],
             groups: Sequence[Hashable], threshold: float, min_n: int) -> list[GroupCheck]:
    if not (len(reference) == len(predicted) == len(groups)):
        raise ValueError("reference, predicted and groups must have the same length")
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be between 0 and 1")
    order: list[Hashable] = []
    tally: dict[Hashable, list[int]] = {}
    for r, p, g in zip(reference, predicted, groups):
        if g not in tally:
            order.append(g)
            tally[g] = [0, 0]
        tally[g][0] += int(r == p)
        tally[g][1] += 1
    out = []
    for g in order:
        k, n = tally[g]
        prop = wilson(k, n)
        out.append(GroupCheck(g, prop, _status(prop, threshold, min_n)))
    return out


def drift_check(reference: Sequence[Hashable], predicted: Sequence[Hashable],
                periods: Sequence[Hashable], *, threshold: float, min_n: int = 30) -> list[GroupCheck]:
    """Accuracy per period (in order of first appearance) against a pre-registered threshold."""
    return _grouped(reference, predicted, periods, threshold, min_n)


def site_check(reference: Sequence[Hashable], predicted: Sequence[Hashable],
               sites: Sequence[Hashable], *, threshold: float, min_n: int = 30) -> list[GroupCheck]:
    """Accuracy per site, to test whether a tool validated at one site transports to others."""
    return _grouped(reference, predicted, sites, threshold, min_n)
