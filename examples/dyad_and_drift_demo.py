"""Clinician–AI dyad study and post-deployment drift check on SYNTHETIC data.

No patient data are used. A simulated reader study: each case is read by the AI alone,
by a clinician alone, and by the same clinician with the AI output in view. The
behaviour (how often clinicians follow the AI) is invented for illustration only.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from proving_ground import drift_check, dyad_study, site_check

rng = np.random.default_rng(2026)
labels = ["normal", "abnormal"]
n = 300
reference = rng.choice(labels, size=n, p=[0.6, 0.4])


def noisy(truth, error_rate):
    out = truth.copy()
    flip = rng.random(len(truth)) < error_rate
    out[flip] = np.where(truth[flip] == "normal", "abnormal", "normal")
    return out


ai = noisy(reference, 0.12)
alone = noisy(reference, 0.20)
# With the AI in view, the simulated clinician follows the AI 70% of the time.
follow = rng.random(n) < 0.70
with_ai = np.where(follow, ai, alone)

print("== Clinician–AI dyad (synthetic reader study, 300 cases) ==")
print(dyad_study(list(reference), list(ai), list(alone), list(with_ai)))

# Post-deployment: accuracy by quarter against a pre-registered minimum of 85%.
quarters = np.repeat(["2027-Q1", "2027-Q2", "2027-Q3", "2027-Q4"], 75)
error_by_quarter = {"2027-Q1": 0.08, "2027-Q2": 0.10, "2027-Q3": 0.16, "2027-Q4": 0.30}
monitored = reference.copy()
for q, rate in error_by_quarter.items():
    idx = quarters == q
    monitored[idx] = noisy(reference[idx], rate)

print("\n== Drift check (pre-registered minimum accuracy 85%, at least 30 cases per quarter) ==")
for row in drift_check(list(reference), list(monitored), list(quarters), threshold=0.85):
    print(row)

sites = np.repeat(["Site A", "Site B", "Site C"], 100)
print("\n== Transportability across sites (same threshold) ==")
for row in site_check(list(reference), list(monitored), list(sites), threshold=0.85):
    print(row)
