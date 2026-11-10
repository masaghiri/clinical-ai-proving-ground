import pytest

from proving_ground import drift_check, dyad_study, site_check

# Ten cases, hand-built so every reliance count can be checked by eye.
#            0    1    2    3    4    5    6    7    8    9
REF   = ["a", "a", "a", "a", "a", "b", "b", "b", "b", "b"]
AI    = ["a", "a", "a", "b", "b", "b", "b", "a", "a", "b"]   # wrong on 3, 4, 7, 8
ALONE = ["a", "b", "b", "a", "a", "b", "a", "b", "a", "b"]   # wrong on 1, 2, 6, 8
WITH  = ["a", "a", "b", "b", "a", "b", "b", "a", "a", "b"]   # wrong on 2, 3, 7, 8


def test_dyad_arm_accuracies():
    r = dyad_study(REF, AI, ALONE, WITH)
    assert (r.ai_alone.k, r.ai_alone.n) == (6, 10)
    assert (r.clinician_alone.k, r.clinician_alone.n) == (6, 10)
    assert (r.clinician_with_ai.k, r.clinician_with_ai.n) == (6, 10)


def test_dyad_reliance_measures():
    r = dyad_study(REF, AI, ALONE, WITH)
    # AI wrong on 3, 4, 7, 8; clinician-with-AI copied the AI on 3, 7, 8 -> 3/4
    assert (r.accepted_wrong_ai.k, r.accepted_wrong_ai.n) == (3, 4)
    # right alone and AI wrong: 3, 4, 7 -> became wrong on 3 and 7 -> 2/3
    assert (r.harmful_switch.k, r.harmful_switch.n) == (2, 3)
    # wrong alone and AI right: 1, 2, 6 -> became right on 1 and 6 -> 2/3
    assert (r.beneficial_switch.k, r.beneficial_switch.n) == (2, 3)
    # AI right on 0, 1, 2, 5, 6, 9 -> clinician-with-AI wrong on 2 -> 1/6
    assert (r.overrode_right_ai.k, r.overrode_right_ai.n) == (1, 6)


def test_dyad_paired_counts():
    r = dyad_study(REF, AI, ALONE, WITH)
    # with-AI right & alone wrong: 1, 6 ; with-AI wrong & alone right: 3, 7
    assert (r.dyad_vs_clinician.only_a_correct, r.dyad_vs_clinician.only_b_correct) == (2, 2)
    assert r.dyad_vs_clinician.p_value == pytest.approx(1.0)


def test_dyad_text_and_length_check():
    text = str(dyad_study(REF, AI, ALONE, WITH))
    assert "Accepted wrong AI" in text and "Clinician with AI" in text
    with pytest.raises(ValueError):
        dyad_study(REF, AI, ALONE, WITH[:-1])


def test_drift_statuses():
    ref = ["x"] * 150
    # period 1: 48/50 right; period 2: 42/50; period 3: 30/50; period 4: 5/5 (too few)
    pred = ["x"] * 48 + ["y"] * 2 + ["x"] * 42 + ["y"] * 8 + ["x"] * 30 + ["y"] * 20
    periods = ["Q1"] * 50 + ["Q2"] * 50 + ["Q3"] * 50
    out = drift_check(ref + ["x"] * 5, pred + ["x"] * 5, periods + ["Q4"] * 5, threshold=0.85)
    status = {c.group: c.status for c in out}
    assert status == {"Q1": "ok", "Q2": "watch", "Q3": "requalify", "Q4": "too few"}
    assert [c.group for c in out] == ["Q1", "Q2", "Q3", "Q4"]


def test_site_check_and_validation():
    out = site_check(["x"] * 4, ["x", "x", "y", "y"], ["A", "A", "B", "B"], threshold=0.5, min_n=2)
    assert {c.group: (c.accuracy.k, c.accuracy.n) for c in out} == {"A": (2, 2), "B": (0, 2)}
    with pytest.raises(ValueError):
        site_check(["x"], ["x"], ["A"], threshold=1.5)
    with pytest.raises(ValueError):
        drift_check(["x", "x"], ["x"], ["A", "A"], threshold=0.5)
