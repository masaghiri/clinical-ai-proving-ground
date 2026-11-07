"""Full workflow on SYNTHETIC data: an AI tool and a human rater judged on the same cases.

No patient data are used. The labels mimic a three-class task (file tip short, exact or long
relative to the apical foramen); the error rates are invented for illustration only.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from proving_ground import brier_score, calibration_table, evaluation_report

rng = np.random.default_rng(2010)
labels = ["short", "exact", "long"]
n = 200
reference = rng.choice(labels, size=n, p=[0.35, 0.40, 0.25])


def noisy(truth, error_rate):
    out = truth.copy()
    flip = rng.random(len(truth)) < error_rate
    for i in np.where(flip)[0]:
        out[i] = rng.choice([lab for lab in labels if lab != truth[i]])
    return out


ai = noisy(reference, 0.08)
human = noisy(reference, 0.22)

print(evaluation_report(list(reference), list(ai), tool_name="Neural network (synthetic)",
                        reference_name="stereomicroscopy (synthetic)",
                        comparator=list(human), comparator_name="clinician (synthetic)"))

# Calibration for a probabilistic output: chance that the file tip is 'exact'.
outcome = (reference == "exact").astype(int)
prob = np.clip(0.75 * outcome + 0.15 + rng.normal(0, 0.12, n), 0, 1)
print(f"Brier score (synthetic 'exact' probabilities): {brier_score(outcome, prob):.3f}")
print("bin       n    mean predicted   observed rate")
for row in calibration_table(outcome, prob, bins=5):
    print(f"{row['bin']:<8} {row['n']:>3}    {row['mean_predicted']:>13.2f}   {row['observed_rate']:>13.2f}")
