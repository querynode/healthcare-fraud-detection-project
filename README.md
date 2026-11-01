# Healthcare Provider Fraud Detection

Detect potentially fraudulent Medicare providers from claims data. Claims are aggregated to
the **provider level** (the unit that is actually labelled), engineered into 30 behavioural
features, and fed to three models compared on precision-recall metrics because only ~9% of
providers are fraudulent.

## Data
Public Kaggle dataset *Healthcare Provider Fraud Detection Analysis* (beneficiary, inpatient
and outpatient claims for 5,410 labelled and 1,353 unseen providers). See `data/raw/README.md`.

## Approach
1. **Feature engineering** (`src/features.py`) - claim volume and reimbursement statistics,
   inpatient share, stay length, diagnosis/procedure counts, missing-physician rates,
   beneficiary mix (age, chronic conditions, deceased rate), plus fraud-oriented signals:
   repeat beneficiary-physician pairs, same-day duplicate claims, top-physician concentration
   and diagnosis diversity.
2. **Modelling** (`src/train.py`) - Logistic Regression, Random Forest and HistGradientBoosting,
   all with class weighting. 80/20 stratified split; 5-fold CV on the 80% for model selection.
3. **Threshold** chosen on out-of-fold predictions to maximise F1 (no test-set peeking).
4. **Explainability** - permutation importance measured on PR-AUC.

## Results (hold-out set, 1,082 providers)
| Metric | Value |
|---|---|
| ROC-AUC | 0.956 |
| PR-AUC | 0.746 (baseline 0.094) |
| Precision / Recall | 0.64 / 0.69 |
| F1 | 0.67 |

CV PR-AUC: HistGradientBoosting 0.684, Random Forest 0.667, Logistic Regression 0.665.
Figures are in `reports/figures/`; predictions for the unseen providers are in
`outputs/unseen_predictions.csv`.

## Run
```bash
pip install -r requirements.txt
python -m src.train --raw data/raw --out outputs
```

## Limitations / next steps
- Labels are provider-level; individual fraudulent claims cannot be identified.
- ~1 in 3 flagged providers is a false positive, so output should prioritise human audit, not auto-penalise.
- Next: SHAP explanations, cost-sensitive threshold using audit cost vs. recovered amount,
  a Streamlit review dashboard.
