"""Re-express a published result with Wilson intervals, using only the published summary counts.

Saghiri MA, Garcia-Godoy F, Gutmann JL, Lotfi M, Asgar K. The reliability of artificial neural
network in locating minor apical foramen: a cadaver study. J Endod. 2012;38(8):1130-4.
https://doi.org/10.1016/j.joen.2012.05.004

50 teeth; stereomicroscopy after extraction was the reference standard. The paper reports the
neural network correct for 96% of teeth and endodontists for 76%, i.e. 48/50 and 38/50.

The case-level (paired) data are not published, so a paired test such as McNemar's cannot be run
from these counts; see synthetic_paired_demo.py for how it works when paired data are available.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from proving_ground import wilson

ann = wilson(48, 50)
endo = wilson(38, 50)
print("Neural network :", ann)
print("Endodontists   :", endo)
print("The intervals do not overlap:", ann.low > endo.high)
