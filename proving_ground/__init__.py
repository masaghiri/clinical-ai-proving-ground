"""Clinical AI Proving Ground: test an AI tool against a reference standard."""
from .metrics import (Proportion, accuracy, binary_metrics, brier_score, calibration_table,
                      cohen_kappa, confusion_matrix, mcnemar_exact, wilson)
from .report import evaluation_report

__all__ = ["Proportion", "accuracy", "binary_metrics", "brier_score", "calibration_table",
           "cohen_kappa", "confusion_matrix", "evaluation_report", "mcnemar_exact", "wilson"]
__version__ = "0.1.1"
