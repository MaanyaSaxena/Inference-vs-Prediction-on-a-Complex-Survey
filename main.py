"""Run the full pipeline.
   python main.py nhanes              # real NHANES 2017-18 (downloads from cdc.gov)
   python main.py nhanes --synthetic  # smoke test of the code path, NOT real results
   python main.py nfhs5               # NFHS-5 (needs the DHS file; else synthetic)
"""
import sys, warnings; sys.path.insert(0, "src")
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import tests as T, prediction as P

which = sys.argv[1] if len(sys.argv) > 1 else "nhanes"
if which == "nhanes":
    import nhanes as D; df = D.load(synthetic="--synthetic" in sys.argv)
else:
    import data as D; df = D.load()
spec = D.SPEC; OUT = "outputs"
X = spec["build_X"](df); yc, y, w, psu = df[spec["cont"]].values, df[spec["binary"]].values, df.w.values, df.psu.values
print(f"N={len(df):,}  weighted {spec['bin_label']} prevalence = {np.average(y, weights=w):.1%}  (unweighted {y.mean():.1%})")
print("Missing % by variable:\n", (df[spec["miss_cols"]].isna().mean() * 100).round(1).to_string())
print(f"PSUs: {df.psu.nunique()}  strata: {df.strata.nunique()}  (few clusters => cluster-robust SEs are optimistic)")

# 1. weighted descriptives -----------------------------------------------------------------
for c in spec["group_cols"]:
    print(f"\n-- weighted % {spec['bin_label']} by {c}\n", T.weighted_prop_table(df, c, outcome=spec["binary"]).to_string())

# 2. tests + multiplicity --------------------------------------------------------------------
tests = T.run_all(df, spec); tests.to_csv(f"{OUT}/hypothesis_tests.csv", index=False)
print("\n-- tests (Holm/BH across the family)\n", tests.round(4).to_string())

# 3-4. inference + imputation ---------------------------------------------------------------
try:
    import regression as R, impute as I
    ols_tab, vif, diag = R.fit_wls_haz(X, yc, w, psu, OUT, strata=df.strata.values)
    ols_tab.to_csv(f"{OUT}/py_ols.csv"); print(f"\n-- WLS on {spec['cont_label']}\n", ols_tab.to_string(), "\n", diag, "\nVIF\n", vif.to_string())
    fit, n_cc = R.fit_logit(X, y, w, psu, df.strata.values)
    orr = R.odds_ratio_table(fit); orr.to_csv(f"{OUT}/py_logit.csv")
    print(f"\n-- Weighted logistic (complete cases n={n_cc:,})\n", orr.to_string())
    mi = I.mice_logit(X, y, w, psu, df.strata.values, m=10)
    cmp_ = I.compare_deletion_vs_mi(orr, mi); cmp_.to_csv(f"{OUT}/deletion_vs_mi.csv")
    print("\n-- Complete-case vs MICE\n", cmp_.to_string())
except ImportError as e:
    print("\n[skip] inference track needs statsmodels:", e)

# 5. prediction ----------------------------------------------------------------------------
res, oof = P.cross_validate(X, y, w, psu); res.to_csv(f"{OUT}/model_comparison.csv", index=False)
print("\n-- PSU-grouped 5-fold CV, weighted metrics\n", res.to_string())
perm, lasso_sel, shap_vals = P.importance(X, y, w, psu)
perm.to_csv(f"{OUT}/permutation_importance.csv", index=False); lasso_sel.to_csv(f"{OUT}/lasso_selected.csv", index=False)
print("\n-- permutation importance (GB, held-out)\n", perm.head(10).to_string())
print("\n-- Lasso-retained features\n", lasso_sel.to_string())
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
res.set_index("model").auc.plot.barh(ax=ax[0], title="Weighted AUC (PSU-grouped CV)"); ax[0].set_xlim(0.45, None)
perm.head(10).iloc[::-1].plot.barh(x="feature", y="auc_drop", ax=ax[1], legend=False, title="Permutation importance (AUC drop)")
plt.tight_layout(); plt.savefig(f"{OUT}/prediction_summary.png", dpi=120)

# 6. export for R cross-check ------------------------------------------------------------------
exp = X.copy(); exp["y_cont"], exp["y_bin"], exp["w"], exp["psu"], exp["strata"] = yc, y, w, psu, df.strata.values
exp.to_csv(f"{OUT}/analysis_data.csv", index=False)
print("\nExported outputs/analysis_data.csv -> Rscript crosscheck.R && python src/crosscheck.py")
