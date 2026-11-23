# Pilot study: removal-operator ranking reversals

Pilot for the MSc thesis proposal "When Removal Flips the Ranking: Predicting
Reversals in Attribution Benchmarks on Correlated Tabular Data" (Dibora Lacha,
Addis Ababa University, 2026). It reproduces Figure 2(b–c) and the numbers in
Section 3.6 of the proposal.

Setup: 6 equicorrelated Gaussian features, 200 random linear models per
correlation level, 500 test points per model; exact interventional and
conditional SHAP compared under marginal and conditional removal.

## Run
```
pip install -r requirements.txt
python pilot.py            # writes pilot_results.json (~1 minute, fixed seed 0)
python make_figs.py pilot_results.json pilot_figure.pdf
```
Tested with Python 3.13, numpy 2.5, scipy 1.18, matplotlib 3.11.
