# Inference vs Prediction on a Complex Survey: Diabetes in NHANES 2017-2018

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/nhanes-inference-vs-prediction/blob/main/notebooks/nhanes_inference_vs_prediction.ipynb)

A worked data-science project that answers two different questions on the same survey data and shows
why they need different tools:

- **Inference:** which factors are associated with diabetes, how large are the effects, and how certain are we?
- **Prediction:** how accurately can diabetes be flagged in advance for screening outreach?

It uses public CDC data (no login), survey weights, strata and PSUs throughout, justifies every test,
and cross-checks the Python results against R's `survey` package.

## Headline results

| | Finding |
|---|---|
| Prevalence | 14.1% of US adults (design SE 0.6 points). The unweighted sample says 19.8% because NHANES oversamples older, minority and low-income adults. |
| Strongest associations | Age (OR about 2.05 per decade at age 50) and BMI (OR 1.10 per unit). Education gradient: college graduates OR 0.36 vs less than 9th grade. |
| Ethnicity | NH Asian adults OR 2.78 at the same BMI; without BMI in the model it falls to 1.69 and loses significance after Holm correction. |
| Best predictor | Plain weighted logistic regression (cross-validated AUC about 0.79). Random forest and gradient boosting do not beat it. |
| Targeting | Reaching the highest-risk 20% of the population captures roughly half of all diabetics (about 2.5x random); about 44% if BMI is unavailable. |
| Verification | Python and R agree: coefficients to four decimals, standard errors within 0.6%. |

Tables and figures from the real NHANES run are in [`results/`](results/). Targeting figures are approximate (see `results/README.md`).

## Quick start

### Option A: Google Colab (recommended)

1. Click the **Open in Colab** badge above (after you edit `YOUR_USERNAME` to your GitHub name), or go to <https://colab.research.google.com>, choose **File > Upload notebook**, and select `notebooks/nhanes_inference_vs_prediction.ipynb`.
2. Choose **Runtime > Run all**.
3. When the data cell runs, an upload box appears. Select the five NHANES files together:
   `DEMO_J.xpt`, `BMX_J.xpt`, `GHB_J.xpt`, `DIQ_J.xpt`, `SMQ_J.xpt`.
   File names must match exactly (rename copies such as `DEMO_J (1).xpt`).
4. The notebook runs the full pipeline and offers `results.zip` for download at the end.

Notes: the R cross-check cell installs R's `survey` package the first time (a few minutes) and is optional.
Colab clears uploaded files when a session resets, so you will need to upload them again after a reset.

### Option B: local script

```bash
pip install -r requirements.txt
# put the five .xpt files in data/nhanes/
python main.py nhanes                       # run from the repository root; BMI sensitivity sections are notebook-only
python main.py nhanes --synthetic           # smoke test of the code path on FAKE data, not real results
Rscript crosscheck.R && python src/crosscheck.py   # design-based R check of the logistic model
```

`shap` (optional) and `pyreadstat` (only for the NFHS-5 option) can be removed from `requirements.txt` if they fail to install.

## Getting the data

NHANES 2017-2018 is public use from the US CDC / NCHS (see the CDC Data User Agreement). Use the **2017-2018** cycle
(file suffix `_J`) from the CDC NHANES site, and download the **Data [XPT]** link for each:

| Component | Dataset | File |
|---|---|---|
| Demographics | Demographic Variables and Sample Weights | `DEMO_J` |
| Examination | Body Measures | `BMX_J` |
| Laboratory | Glycohemoglobin | `GHB_J` |
| Questionnaire | Diabetes | `DIQ_J` |
| Questionnaire | Smoking - Cigarette Use | `SMQ_J` |

Other cycles use different suffixes and some variable names and codes differ, so they will not work without changes.
Never mix files from different cycles. The recoding rule for each variable is documented in `CODEBOOK` in `src/nhanes.py`.

## Pipeline and decisions

```text
NHANES files -> merge on SEQN -> documented recoding -> weighted descriptives and tests (Holm/BH)
  -> design-based regression (OLS on HbA1c, logistic on diabetes) with diagnostics
  -> sensitivity: complete-case vs multiple imputation; model with and without BMI
  -> Ridge / Lasso / Random Forest / Gradient Boosting with PSU-grouped CV
  -> Python vs R cross-check
```

| Decision | Choice | Why |
|---|---|---|
| Weight | Exam weight `WTMEC2YR`, normalised to mean 1 | The analysis mixes interview items with exam and lab items |
| Group comparison | Welch / Mann-Whitney / ANOVA / Kruskal-Wallis chosen from Shapiro and Levene | Test must match assumptions; with large n Shapiro rejects trivially, so large groups lean on the CLT |
| Categorical association | Chi-square + Cramér's V (Fisher for small 2x2) | Effect size, not only p |
| Multiplicity | Holm and Benjamini-Hochberg | Families of tests inflate false positives |
| Regression inference | Design-based SEs (strata + PSU linearisation), t with design df (15 here) | Same estimator as R `survey::svyglm`; statsmodels' own robust SEs are unsupported with survey weights |
| Selection | Lasso | Stepwise inflates false positives |
| Cross-validation | **GroupKFold by PSU** | Random folds leak village-level information and inflate accuracy |
| Targeting metric | Top 20% of the **weighted population** | A respondent-count cutoff covers a different population share because weights vary |
| Missing data | Median impute + indicator inside CV (prediction); MICE with Rubin's rules vs deletion (inference) | Tests whether conclusions depend on missing-data handling |
| BMI | Models with and without it | BMI is partly a mediator; the two versions answer different questions |
| Interpretation | Odds ratios; permutation importance (and SHAP if installed) | Importance is predictive association, not causation |

The classical tests cannot use survey weights, so treat them as exploratory; the weighted regressions are confirmatory.

## Repository layout

```text
notebooks/nhanes_inference_vs_prediction.ipynb   full pipeline, self-contained, runnable on Colab
main.py                                          script version (no BMI sensitivity sections)
crosscheck.R                                     design-based R check (svydesign + svyglm)
requirements.txt
src/
  nhanes.py       download/read, CODEBOOK, recoding, design variables, synthetic smoke-test data
  tests.py        weighted tables, test selection, effect sizes, Holm/BH
  design.py       design-based (strata + PSU) variance for weighted linear and logistic models
  regression.py   weighted OLS and logistic, diagnostics, odds ratios
  impute.py       MICE with Rubin's rules; deletion vs imputation comparison
  prediction.py   Ridge/Lasso/RF/GB, PSU-grouped CV, targeting metric, importance
  crosscheck.py   Python vs R comparison
  data.py, features.py   NFHS-5 alternative (see below)
data/             instructions for the NHANES files; put the .xpt files in data/nhanes/
results/          tables and figures from the real run (see results/README.md)
outputs/          where new runs write results (git-ignored)
```

## Outputs

`hypothesis_tests.csv`, `py_ols.csv`, `py_logit.csv`, `r_ols.csv`, `r_logit.csv`, `logit_with_vs_without_bmi.csv`,
`deletion_vs_mi.csv`, `model_comparison.csv`, `prediction_with_vs_without_bmi.csv`, `permutation_importance.csv`,
`lasso_selected.csv`, `ols_diagnostics.png`, `prediction_summary.png`, `analysis_data.csv` (input to the R check).
Copies from the real run are kept in `results/`.

## How to read the two tracks

- **Inference** (odds ratios, HbA1c coefficients): adjusted associations with confidence intervals and multiplicity-adjusted
  p-values. Use for describing disparities. Still association, not causation.
- **Prediction** (AUC, Brier score, reach and precision of a targeted list): judged out of sample on held-out PSUs.
  Use for building a screening list.
- A factor can matter as an association and little as a predictor (education: OR 0.36, under 0.01 AUC).

## Limitations

- Cross-sectional, one US two-year cycle: no causal claims and no transfer to other populations.
- Only 30 PSUs (15 strata): design df = 15, so intervals are wide; cross-validated AUC moves by about 0.005 depending on how PSUs fall into folds.
- Adults are selected by dropping rows rather than subsetting the full design, which slightly understates SEs.
- The diabetes definition includes undiagnosed cases (HbA1c of 6.5% or more), so binary and continuous outcomes overlap.
- HbA1c is right-skewed; a log-scale model has not been fitted.
- Multiple imputation assumes missing at random and omits design variables; the Lasso retains a missing-BMI indicator, a warning sign.
- No external validation and no subgroup calibration check.

## References

NCHS, *NHANES Analytic Guidelines* and the 2017-2018 documentation (weights, strata, PSU); T. Lumley, the R `survey` package.

## License

MIT (see `LICENSE`; replace `YOUR NAME` with the copyright holder). NHANES data are public-use; see the CDC Data User Agreement.
