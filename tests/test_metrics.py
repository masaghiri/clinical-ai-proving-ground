import math

import pytest

from proving_ground import (accuracy, binary_metrics, brier_score, calibration_table, cohen_kappa,
                            evaluation_report, mcnemar_exact, wilson)


def test_wilson_matches_closed_form():
    # 48/50 with z = 1.959964: reference values computed independently.
    p = wilson(48, 50)
    assert p.estimate == pytest.approx(0.96)
    assert p.low == pytest.approx(0.8654, abs=1e-3)
    assert p.high == pytest.approx(0.9890, abs=1e-3)


def test_wilson_stays_inside_unit_interval():
    for k, n in [(0, 5), (5, 5), (1, 1), (0, 1)]:
        p = wilson(k, n)
        assert 0.0 <= p.low <= p.estimate <= p.high <= 1.0


def test_wilson_rejects_bad_counts():
    with pytest.raises(ValueError):
        wilson(6, 5)


def test_wilson_empty_is_nan():
    assert math.isnan(wilson(0, 0).estimate)


def test_accuracy_and_binary_metrics():
    ref = ["a", "a", "b", "b"]
    pred = ["a", "b", "b", "b"]
    assert accuracy(ref, pred).k == 3
    m = binary_metrics(ref, pred, positive="a")
    assert (m["sensitivity"].k, m["sensitivity"].n) == (1, 2)
    assert (m["specificity"].k, m["specificity"].n) == (2, 2)


def test_kappa_perfect_and_chance():
    assert cohen_kappa(["x", "y", "x"], ["x", "y", "x"]) == pytest.approx(1.0)
    assert cohen_kappa(["x", "x", "y", "y"], ["x", "y", "x", "y"]) == pytest.approx(0.0)


def test_mcnemar_counts_and_symmetry():
    ref = ["a"] * 10
    a = ["a"] * 8 + ["b"] * 2
    b = ["b"] * 6 + ["a"] * 4
    r = mcnemar_exact(ref, a, b)
    assert (r.only_a_correct, r.only_b_correct) == (6, 2)
    r2 = mcnemar_exact(ref, b, a)
    assert r.p_value == pytest.approx(r2.p_value)


def test_mcnemar_no_discordance():
    assert mcnemar_exact(["a", "b"], ["a", "b"], ["a", "b"]).p_value == 1.0


def test_brier_and_calibration():
    assert brier_score([1, 0], [1.0, 0.0]) == 0.0
    assert brier_score([1, 0], [0.5, 0.5]) == pytest.approx(0.25)
    rows = calibration_table([0, 1, 1], [0.1, 0.9, 1.0], bins=5)
    assert sum(r["n"] for r in rows) == 3


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        accuracy(["a"], ["a", "b"])


def test_report_contains_sections():
    text = evaluation_report(["a", "b", "a"], ["a", "b", "b"], comparator=["a", "a", "a"])
    for heading in ["## Accuracy", "## Confusion matrix", "## Agreement", "McNemar"]:
        assert heading in text
