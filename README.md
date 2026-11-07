# Clinical AI Proving Ground

A small, dependable toolkit and protocol for one question: **does this AI tool actually work, compared with a reference standard?**

It grew out of a rule I have followed since my first neural network in 2010–2011: no AI result without a reference standard. That network, trained on features extracted from radiographs, located the minor apical foramen correctly in 93% of test samples ([Int Endod J 2012](https://doi.org/10.1111/j.1365-2591.2011.01970.x)); tested in human cadavers against stereomicroscopy, it was right for 96% of teeth, compared with 76% for experienced endodontists ([J Endod 2012](https://doi.org/10.1016/j.joen.2012.05.004)).

The Proving Ground is designed for teaching as much as for research: a master's student can take a real AI tool, collect a reference standard, and produce an honest written evaluation.

## What is included

| File | Purpose |
|---|---|
| [`proving_ground/metrics.py`](proving_ground/metrics.py) | Accuracy, sensitivity, specificity, PPV and NPV with **Wilson 95% intervals**; confusion matrix; Cohen's kappa; **exact McNemar test** for an AI tool and a human rater judged on the same cases; Brier score and calibration table for probabilities |
| [`proving_ground/report.py`](proving_ground/report.py) | Writes a Markdown evaluation report from your data |
| [`PROTOCOL.md`](PROTOCOL.md) | A one-page evaluation protocol to fill in **before** collecting data |
| [`RUBRIC.md`](RUBRIC.md) | A rubric for checking the output of large language models and AI assistants |
| [`examples/`](examples) | A re-analysis of published counts, and a full workflow on synthetic data |

## Quick start

```bash
pip install numpy scipy
git clone https://github.com/masaghiri/clinical-ai-proving-ground
cd clinical-ai-proving-ground
python examples/reanalysis_cadaver_2012.py
python examples/synthetic_paired_demo.py
```

```python
from proving_ground import accuracy, mcnemar_exact, evaluation_report

reference = ["exact", "short", "long", "exact"]   # e.g. stereomicroscopy after extraction
ai        = ["exact", "short", "exact", "exact"]
clinician = ["exact", "long", "long", "short"]

print(accuracy(reference, ai))                      # 75.0% (3/4; 95% CI ...)
print(mcnemar_exact(reference, ai, clinician))      # paired comparison on the same cases
print(evaluation_report(reference, ai, comparator=clinician))
```

Example output from the published counts (48/50 vs 38/50):

```
Neural network : 96.0% (48/50; 95% CI 86.5–98.9%)
Endodontists   : 76.0% (38/50; 95% CI 62.6–85.7%)
```

## Tests

```bash
pip install pytest
python -m pytest -q
```

## Principles

1. Name the reference standard before collecting data, and measure it blind to the AI output.
2. Report every proportion with its count and interval, never a bare percentage.
3. Compare against the people who do the task today, on the same cases.
4. Look at the failures one by one; report them as clearly as the successes.
5. Keep patient data out of public tools and public repositories.

## Status and use

Research and teaching software, version 0.1. **Not a medical device and not for clinical decisions.** No patient data are included in this repository; the demonstration data are synthetic.

## Citation

See [`CITATION.cff`](CITATION.cff). Author: M. Ali Saghiri, MS, PhD ([ORCID 0000-0002-5064-7828](https://orcid.org/0000-0002-5064-7828)).

## License

MIT. See [`LICENSE`](LICENSE).
