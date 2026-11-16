# Credit Card Fraud Detection

Detecting fraudulent card transactions in a dataset where only **0.17%** of transactions are fraud, with a tuned decision threshold, SHAP explanations and an interactive Streamlit app.

**Live app:** _add your Streamlit Cloud link here_

---

## Problem

A bank wants to catch fraudulent transactions without blocking genuine customers. With 492 frauds in 284,807 transactions, a model that predicts "not fraud" for everything scores **99.8% accuracy** and catches nothing, so accuracy is not a usable metric here. This project uses precision, recall and PR-AUC instead.

## Data

[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (ULB Machine Learning Group): European cardholders, September 2013.

- `V1` to `V28` are anonymised PCA components, so they can be ranked by importance but not interpreted in business terms
- `Time` and `Amount` are raw; `Class` is the label (1 = fraud)

## Approach

1. **Stratified 80/20 split** to keep the 0.17% fraud ratio in both sets
2. **Scaled `Time` and `Amount`**, fitting the scaler on the training set only to avoid data leakage
3. **Baseline logistic regression**, then two fixes for the imbalance: class weights and SMOTE (on the training set only)
4. **Threshold tuning on a separate validation set**, keeping the test set untouched until the final evaluation
5. **PR-AUC** to compare models independently of any threshold
6. **SHAP** to explain individual predictions

## Results (test set: 56,962 transactions, 98 frauds)

| Model | Frauds caught | Frauds missed | Genuine customers blocked | Precision | Recall |
|---|---|---|---|---|---|
| Baseline (threshold 0.5) | 63 | 35 | 13 | 0.83 | 0.64 |
| Class weights (threshold 0.5) | 90 | 8 | 1,386 | 0.06 | 0.92 |
| SMOTE (threshold 0.5) | 90 | 8 | 1,461 | 0.06 | 0.92 |
| **Class weights, tuned threshold 0.99** | **84** | **14** | **67** | **0.56** | **0.86** |

The tuned model catches **21 more frauds than the baseline** and blocks **1,319 fewer genuine customers than the untuned weighted model**. Precision was 0.56 on both validation and test, so the threshold generalised.

## Key findings

- **Class weights over SMOTE.** Both reached 0.92 recall, but SMOTE blocked 75 more genuine customers, doubled the training data, and risks overfitting because every synthetic fraud is built from the same 394 real cases.
- **The real gain came from the threshold, not the weighting.** PR-AUC was almost the same for all models (baseline 0.74, weighted 0.72). Class weights mostly shifted the scores upwards; choosing the threshold properly is what improved the result.
- **SHAP's top feature was misleading.** SHAP ranked `Amount` first, but the median fraud amount (9.25) is *lower* than the median genuine amount (22.0). The ranking is driven by a few extreme transactions affecting a linear model.
- **The model catches "loud" frauds and misses "quiet" ones.** Two frauds of exactly 1.00 got opposite results: one scored 1.000, with almost every feature pointing to fraud; the other scored 0.017, where one strong signal (low `V10`) was cancelled out by several normal-looking features.

## Limitations and next steps

- Most missed frauds scored far below any workable threshold (several under 0.05), so threshold tuning cannot catch them; only 2 of the 14 were near-misses (0.984 and 0.988).
- Logistic regression adds up evidence and cannot learn that one strong signal is enough. A **tree-based model** (Random Forest, XGBoost) is the next step.
- A log-transform of `Amount` would reduce the effect of outliers.

## Streamlit app

- Pick a transaction and see its fraud score, real amount and flag/approve decision at the 0.99 threshold
- A SHAP waterfall chart explaining that one prediction
- A table of every fraud the model missed, with real amounts and scores

The app uses a demo set of all 98 test frauds plus 1,000 genuine test transactions, because the full dataset (150 MB) is too large for GitHub.

## What I learned

_Write 2–3 sentences in your own words._

## Run locally

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Files

| File | Purpose |
|---|---|
| `CreditCard_Fraud_Detection.ipynb` | full analysis: training, threshold tuning, PR-AUC, SHAP |
| `app.py` | Streamlit app |
| `model.pkl` | class-weighted logistic regression |
| `scaler.pkl` | scaler fitted on the training set |
| `demo.csv` | 98 test frauds plus 1,000 genuine test transactions (scaled) |
| `background.csv` | 1,000 training rows used as the SHAP background |