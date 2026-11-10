"""Clinical AI Proving Ground: test an AI tool against a reference standard."""
from .dyad import DyadResult, dyad_study
from .metrics import (PairedComparison, Proportion, accuracy, binary_metrics, brier_score, calibration_table,
                      cohen_kappa, confusion_matrix, mcnemar_exact, wilson)
from .monitoring import GroupCheck, drift_check, site_check
from .report import evaluation_report

__all__ = ["DyadResult", "GroupCheck", "PairedComparison", "Proportion", "accuracy", "binary_metrics",
           "brier_score", "calibration_table", "cohen_kappa", "confusion_matrix", "drift_check", "dyad_study",
           "evaluation_report", "mcnemar_exact", "site_check", "wilson"]
__version__ = "0.2.0"
