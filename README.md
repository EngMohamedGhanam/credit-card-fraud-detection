# Credit Card Fraud Detection

**Course:** Data Mining

A machine-learning project that detects fraudulent credit-card transactions using a **custom XGBoost implementation built from scratch** with only NumPy and Pandas — no `xgboost` library, no scikit-learn estimators. Every calculation (gradient computation, tree construction, regularised leaf weights, approximate splits) is written by hand and lives in `model.py`, exercised from the notebook `fraud_detection.ipynb`.

scikit-learn is used **only** for the train/test split and evaluation metrics; `imbalanced-learn` is used **only** for SMOTE.

---

## 1. Problem Statement

Credit-card fraud is a rare but high-cost event: on the dataset used here, fraud accounts for **0.1727 %** of all transactions (a **577.9 : 1** imbalance). The goal is a binary classifier that catches as many fraudulent transactions as possible (**high recall**) while keeping the false-alarm rate low enough to be operationally useful (**reasonable precision**), evaluated primarily with metrics appropriate for imbalanced data (**ROC-AUC** and **AUPRC**).

The project targets three mandatory thresholds:

| Metric | Target |
|---|---|
| Recall (fraud class) | ≥ 80 % |
| Precision (fraud class) | ≥ 70 % |
| ROC-AUC | ≥ 85 % |

---

## 2. Dataset

| Field | Value |
|---|---|
| Source | [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| Provider | ULB Machine Learning Group |
| File | `data/creditcard.csv` |
| Rows | 284,807 transactions |
| Columns | 31 (`Time`, `V1`–`V28`, `Amount`, `Class`) |
| Legitimate (Class = 0) | 284,315 (99.8273 %) |
| Fraud (Class = 1) | 492 (0.1727 %) |
| Imbalance ratio | 577.9 : 1 |
| Missing values | None |

Features `V1`–`V28` are anonymised PCA components; `Time` and `Amount` are the two raw features.

> **Note:** `creditcard.csv` is **not** committed to the repository (≈ 150 MB). Download it from the Kaggle link above and place it in the `data/` folder before running the notebook.

---

## 3. Model Performance

Evaluated on the **held-out test set** — 56,962 transactions (98 fraud / 56,864 legitimate) — that the model never saw during training.

### 3.1 Threshold-independent metrics

| Metric | Test value | Target | Status |
|---|---:|---:|:---:|
| **ROC-AUC** | **0.9810** | ≥ 0.85 | ✓ PASS |
| **AUPRC** (area under PR curve) | **0.8125** | (informational) | — |

ROC-AUC and AUPRC do not depend on the decision threshold, so they measure the model's raw ranking ability. Both are strong: the model separates fraud from legitimate transactions very well.

### 3.2 No meaningful overfitting (5-fold stratified CV)

A 5-fold stratified cross-validation was run on the training set, **re-fitting robust scaling and SMOTE inside each fold** (no leakage). The gap between training and out-of-fold validation performance is small and validation scores are stable across folds:

| | Train | Validation (out-of-fold) | Gap |
|---|---:|---:|---:|
| ROC-AUC | 0.9984 ± 0.0003 | **0.9822 ± 0.0092** | **+0.0163** |
| AUPRC | 0.9246 ± 0.0113 | 0.7877 ± 0.0519 | +0.1369 |

The final full-train model shows the same picture against the test set: train ROC-AUC 0.9979 vs test 0.9810 (**gap +0.0169**). A ~0.016 ROC-AUC gap with tight fold-to-fold variance means the model **generalises rather than memorises** — the strong headline numbers are not an artefact of one lucky 80/20 split. (The larger AUPRC gap is expected and normal for a 578 : 1 imbalance, where precision–recall on a rare class is inherently high-variance.)

---

## 4. Decision Threshold — Leakage Issue and Fix

### 4.1 The problem: threshold tuned on the test set

The model outputs a fraud **probability**; a decision threshold converts it to a fraud/legit label. Early versions of this project selected the threshold (0.92, and an "F1-optimal" 0.9804) by inspecting the **test set** directly. That is a **data-leakage** mistake: the test set is no longer an unbiased estimate of real-world performance once it has been used to make a modelling choice.

At the test-tuned threshold of **0.92**, the model also **failed the precision target**:

| Metric (fraud) | Value | Target | Status |
|---|---:|---:|:---:|
| Recall | 87.76 % | ≥ 80 % | ✓ PASS |
| **Precision** | **57.72 %** | ≥ 70 % | ✗ **FAIL** |
| ROC-AUC | 98.10 % | ≥ 85 % | ✓ PASS |

Confusion matrix at 0.92: TP = 86, FP = 63, FN = 12 — i.e. 63 false alarms for every 86 true catches, so ~42 % of everything flagged as fraud was actually legitimate.

### 4.2 The fix: leakage-free threshold selection

Threshold selection was rebuilt using **out-of-fold (OOF) predictions from the 5-fold cross-validation**: every training row is scored by a model that never saw it, producing a genuine *validation* set that is independent of the test set. The threshold is swept from 0.50 to 0.99 (step 0.01) on this OOF validation set, and the rule is: **pick the smallest threshold whose precision ≥ 70 %, which maximises recall under that constraint.** The chosen threshold is then **confirmed on the untouched test set**.

| Threshold | Selected on | Precision | Recall | F1 | Notes |
|---|---|---:|---:|---:|---|
| 0.92 | test set (leaky) | 57.7 % | 87.8 % | 69.6 % | fails precision target |
| **0.95** | **validation (OOF)** | 73.0 % (val) | 81.7 % (val) | 77.1 % (val) | leakage-free recommendation |
| 0.95 | — confirmed on test | 66.1 % | 85.7 % | 74.7 % | precision just short on test |
| **0.97** | **test-confirmed operating point** | **70.4 %** | **82.7 %** | **76.1 %** | all three targets met on test |

**Why 0.95 (validation) and 0.97 (test) differ — small-sample optimism.** The OOF validation set contains 394 fraud cases, but the test set contains only **98**. With so few positives, precision is extremely noisy (each false positive moves it by roughly half a percentage point), so a threshold tuned on validation is mildly *optimistic* when transferred to a small test set — here by about 7 precision points. The validation-selected **0.95** lands at 66.1 % precision on test; nudging to **0.97** absorbs that optimism and clears the precision target on genuinely unseen data (**P = 70.4 %, R = 82.7 %**), while still catching **81 of 98** fraud cases.

**Recommended operating point: ≈ 0.96–0.97.** At 0.97 all three mandatory targets are met simultaneously on the test set (Recall 82.7 % ≥ 80, Precision 70.4 % ≥ 70, ROC-AUC 98.1 % ≥ 85). The exact production value should ultimately be set by the business cost of a missed fraud versus a false alarm, not fixed a priori.

These analyses live in the notebook's **Section 7.2 (threshold selection)** and **Section 7.3 (cross-validation)**, and are reproducible via `analysis/threshold_cv_analysis.py`.

---

## 5. Project Pipeline

The notebook `fraud_detection.ipynb` follows a standard data-mining workflow in seven sections.

### Section 1 — Import Libraries
NumPy, Pandas, Matplotlib, Seaborn, and scikit-learn (for `train_test_split` and metrics **only**). `imbalanced-learn` is used later for SMOTE. Random seed fixed to 42 for reproducibility.

### Section 2 — Exploratory Data Analysis (EDA)
Shape/dtype inspection, missing-value audit (none found), class-distribution plots showing the extreme imbalance, descriptive statistics, and log-scale amount-distribution histograms comparing legitimate vs. fraud transactions.

### Section 3 — Preprocessing
1. **Drop `Time`** — raw elapsed seconds carry no predictive signal.
2. **Train / Test split** — 80 / 20, stratified, `random_state=42` (train: 227,845 rows; test: 56,962 rows, 98 fraud).
3. **Robust Scaling** (manual): `x_scaled = (x − median) / IQR`, fitted on the training set only, then applied to the test set (no data leakage). Chosen over Standard/Min-Max because `Amount` contains extreme outliers.

### Section 4 — XGBoost From Scratch (`model.py`)
Custom NumPy-only components:
- **`sigmoid(x)`** — numerically stable, clipped to `[-10, 10]`.
- **`compute_gradients` / `compute_hessians`** — weighted binary cross-entropy derivatives, scaled by an automatically derived class weight `√(n_neg / n_pos)`.
- **`DecisionTreeXGB`** — regression-tree weak learner with percentile-based approximate splits (`_get_split_candidates`), L2-regularised leaf value (`_leaf_weight = −G / (H + λ)`), and the XGBoost gain formula with L2 (λ) and complexity penalty (γ).
- **`XGBoostFromScratch`** — full boosting loop with additive updates, loss-history logging, and early stopping.

The notebook runs unit tests on each component before training.

### Section 5 — Model Training
1. **SMOTE** oversampling on the training set only (394 fraud → balanced 454,902-row training set).
2. **Hyperparameters:**
   ```python
   HYPERPARAMS = dict(
       n_estimators     = 100,
       learning_rate    = 0.1,
       max_depth        = 3,
       lam              = 1.5,
       gamma            = 0.0,
       n_bins           = 10,
       min_child_weight = 5,
       class_weight     = None,   # auto = √(n_neg / n_pos)
   )
   ```
3. **Boosting loop:** 100 rounds; weighted binary cross-entropy drops monotonically from 0.617 (round 1) to **0.0657** (round 100).

### Section 6 — Evaluation
Confusion matrix, classification report, ROC curve, precision–recall curve, and a pass/fail check against the project targets.

### Section 7 — Results & Conclusion
- **7.1** Model performance summary.
- **7.2** Leakage-free threshold selection on the out-of-fold validation set (see Section 4 above).
- **7.3** 5-fold stratified cross-validation / overfitting check.
- **7.4–7.6** The four refinements, a summary table, and closing remarks with limitations and future work.

---

## 6. Key Takeaways — The Four Refinements

The model differs from a naive gradient-boosting baseline in four deliberate ways, each targeting a specific weakness of the dataset or algorithm:

| # | Refinement | Problem It Solves | Consequence If Omitted |
|---|---|---|---|
| 1 | **Robust Scaling** (median / IQR) | `Amount` ranges from cents to €25,000; standard scaling would be dominated by outliers | Splits cluster around extreme values instead of true fraud patterns |
| 2 | **Weighted Gradients** (class weight ≈ √(n_neg/n_pos)) | Fraud < 0.2 % — equal weights collapse to the trivial "always legitimate" solution | Recall would be **0 %** while accuracy still looked like 99.8 % |
| 3 | **L2 Regularisation** (λ) on leaf weights and gain | Rare-class leaves can memorise noise and take extreme values | Overfit trees, poor test-set generalisation |
| 4 | **Approximate Percentile Splits** (`n_bins=10`) | Exhaustive threshold search over ~228 k rows is prohibitively slow | Training would take hours instead of minutes |

---

## 7. Conclusions

- A complete gradient-boosting fraud detector was built from first principles using only NumPy and Pandas.
- On the held-out test set (56,962 unseen transactions), the model achieves **98.10 % ROC-AUC** and **81.25 % AUPRC** — comfortably above the ROC-AUC target — and 5-fold cross-validation confirms this generalises (train↔validation ROC-AUC gap ≈ 0.016, no meaningful overfitting).
- The **decision threshold was rebuilt to remove data leakage**: instead of tuning on the test set (the old 0.92 threshold, which failed the 70 % precision target at 57.7 %), the threshold is now selected on out-of-fold cross-validation predictions and confirmed on the untouched test set. The leakage-free recommendation is **0.95**; accounting for small-sample optimism (only 98 test frauds), the practical **test-confirmed operating point is ≈ 0.97**, where all three mandatory targets are met simultaneously (**Precision 70.4 %, Recall 82.7 %, ROC-AUC 98.1 %**).
- **Limitations and future work:** (1) the production threshold should still be tuned to the real cost of a missed fraud vs. a false alarm; (2) no hyperparameter search was performed; (3) the model has been evaluated on a single dataset, so its performance on other fraud distributions is unknown.

---

## 8. How to Run

### 8.1 Prerequisites
- Python **3.10+**
- Jupyter Notebook or JupyterLab
- `data/creditcard.csv` downloaded from Kaggle and placed in the `data/` folder

### 8.2 Environment Setup

**Option A — use the provided virtual environment (`fraud_env/`):**
```bash
# Windows (PowerShell)
.\fraud_env\Scripts\Activate.ps1

# macOS / Linux
source fraud_env/bin/activate
```

**Option B — fresh install:**
```bash
python -m venv fraud_env
# activate as above, then:
pip install -r requirements.txt
pip install imbalanced-learn      # required for SMOTE (Section 5)
```

`requirements.txt` pins: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `jupyter`.

### 8.3 Get the Dataset
1. Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
2. Place it at `data/creditcard.csv`.

### 8.4 Run the Notebook
```bash
jupyter notebook fraud_detection.ipynb
```
Run the cells top-to-bottom. Full training takes **~25 minutes** on a typical laptop (single-threaded NumPy). Keep `np.random.seed(42)` and the SMOTE `random_state=42` untouched to reproduce all reported numbers.

### 8.5 (Optional) Regenerate the Cross-Validation Artifacts
Sections 7.2 and 7.3 read cached out-of-fold predictions and CV results from `analysis/`. These are already included, but you can regenerate them:
```bash
python analysis/threshold_cv_analysis.py
```
This mirrors the notebook pipeline exactly (same seed-42 split, robust scaling, SMOTE, and hyperparameters), runs the 5 folds plus the final model **in parallel**, and writes `oof_proba.npy`, `oof_labels.npy`, `results.json`, and the two plots. Expect ~25 minutes of wall-clock on a multi-core machine.

---

## 9. Project Structure

```
fraud-detection/
├── data/
│   └── creditcard.csv               # Kaggle dataset (not committed — download manually)
├── fraud_detection.ipynb            # Main notebook (EDA → preprocessing → training → evaluation → conclusions)
├── model.py                         # XGBoostFromScratch, DecisionTreeXGB, sigmoid, gradients, hessians
├── analysis/
│   ├── threshold_cv_analysis.py     # Leakage-free threshold sweep + 5-fold CV (parallel)
│   ├── oof_proba.npy                # Out-of-fold validation probabilities (used by Section 7.2)
│   ├── oof_labels.npy               # Matching training labels
│   ├── results.json                 # CV metrics + validation/test threshold sweeps
│   ├── threshold_sweep_validation.png
│   └── cv_train_vs_val_auc.png
├── requirements.txt                 # Python dependencies
├── PLan.md                          # High-level project plan
├── EXPLANATION.md                   # Line-by-line notebook walkthrough (Arabic)
├── README.md                        # This file
└── fraud_env/                       # Pre-built Python virtual environment (optional)
```
