"""Write a short Markdown evaluation report for an AI tool tested against a reference standard."""
from __future__ import annotations

from typing import Hashable, Sequence

from .metrics import accuracy, binary_metrics, cohen_kappa, confusion_matrix, mcnemar_exact


def evaluation_report(reference: Sequence[Hashable], ai: Sequence[Hashable], *,
                      tool_name: str = "AI tool", reference_name: str = "reference standard",
                      comparator: Sequence[Hashable] | None = None,
                      comparator_name: str = "human rater") -> str:
    """Return a Markdown report. Every number comes from the data passed in."""
    lines = [f"# Evaluation of {tool_name}", "",
             f"Reference standard: {reference_name}. Cases: {len(reference)}.", "",
             "## Accuracy", "",
             f"- {tool_name}: {accuracy(reference, ai)}"]
    if comparator is not None:
        lines.append(f"- {comparator_name}: {accuracy(reference, comparator)}")

    labels, m = confusion_matrix(reference, ai)
    lines += ["", f"## Confusion matrix ({tool_name})", "",
              "| reference \\ predicted | " + " | ".join(map(str, labels)) + " |",
              "|---|" + "---|" * len(labels)]
    for lab, row in zip(labels, m):
        lines.append(f"| {lab} | " + " | ".join(str(v) for v in row) + " |")

    lines += ["", "## Per-label results (each label against all others)", "",
              "| label | sensitivity | specificity |", "|---|---|---|"]
    for lab in labels:
        b = binary_metrics(reference, ai, positive=lab)
        lines.append(f"| {lab} | {b['sensitivity']} | {b['specificity']} |")

    lines += ["", "## Agreement", "",
              f"- Cohen's kappa, {tool_name} vs {reference_name}: {cohen_kappa(reference, ai):.3f}"]
    if comparator is not None:
        lines.append(f"- Paired comparison with {comparator_name}: {mcnemar_exact(reference, ai, comparator)}")

    lines += ["", "## Before trusting this result, check", "",
              "- Was the reference standard measured blind to the AI output?",
              "- Do the cases look like the patients the tool will meet in practice?",
              "- Which cases did the tool get wrong, and would a clinician have caught them?",
              "- Has the result been reproduced on data the developers never saw?"]
    return "\n".join(lines) + "\n"
