"""
Threshold selection + 5-fold cross-validation (overfitting) analysis.

This script does NOT change the model architecture. It imports the existing
`XGBoostFromScratch` from model.py and mirrors the notebook pipeline exactly:

    drop Time -> stratified 80/20 split (seed 42) -> robust scale (fit on train)
    -> SMOTE on train only -> same HYPERPARAMS.

Two deliverables:

1. THRESHOLD SWEEP on a *validation* set built from out-of-fold (OOF) CV
   predictions (every training row is predicted while it is held out of
   training). Sweep threshold 0.50..0.99 step 0.01, plot precision & recall,
   recommend the threshold that satisfies precision >= 0.70 while maximizing
   recall. The recommended threshold is then CONFIRMED on the untouched test
   set -- avoiding the threshold-on-test leakage in the original notebook.

2. OVERFITTING CHECK via 5-fold stratified CV: per-fold train vs validation
   ROC-AUC / AUPRC gap, plus a final full-train vs held-out-test comparison.

Folds and the final model are trained in parallel (independent processes).
"""

import os
import sys
import json
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score, confusion_matrix,
)
from imblearn.over_sampling import SMOTE
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE      = os.path.dirname(os.path.abspath(__file__))
PROJECT   = os.path.dirname(HERE)
DATA_PATH = os.path.join(PROJECT, "data", "creditcard.csv")
CACHE_DIR = os.environ.get("CV_CACHE", os.path.join(HERE, "_cache"))
os.makedirs(CACHE_DIR, exist_ok=True)

sys.path.insert(0, PROJECT)  # so workers can import model.py

# Exact notebook hyperparameters (Section 5.1).
# n_estimators / n_splits overridable via env only for fast smoke-testing;
# defaults reproduce the notebook exactly.
HYPERPARAMS = dict(
    n_estimators=int(os.environ.get("CV_N_EST", 100)),
    learning_rate=0.1, max_depth=3,
    lam=1.5, gamma=0.0, n_bins=10, min_child_weight=5, class_weight=None,
)
N_SPLITS   = int(os.environ.get("CV_N_SPLITS", 5))
SEED       = 42
TRAIN_EVAL_SUBSAMPLE = 40000  # rows used to estimate train-side AUC (speed)


# ----------------------------------------------------------------------------
# Data preparation (run once in main, cached to .npy for workers)
# ----------------------------------------------------------------------------
def prepare_data():
    import pandas as pd
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=["Time"])
    feature_cols = [c for c in df.columns if c != "Class"]
    X = df[feature_cols].values.astype(np.float64)
    y = df["Class"].values.astype(np.int64)
    Xtr_raw, Xte_raw, ytr, yte = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )
    np.save(os.path.join(CACHE_DIR, "Xtr_raw.npy"), Xtr_raw)
    np.save(os.path.join(CACHE_DIR, "Xte_raw.npy"), Xte_raw)
    np.save(os.path.join(CACHE_DIR, "ytr.npy"), ytr)
    np.save(os.path.join(CACHE_DIR, "yte.npy"), yte)
    with open(os.path.join(CACHE_DIR, "features.json"), "w") as f:
        json.dump(feature_cols, f)
    return Xtr_raw.shape, Xte_raw.shape, int(ytr.sum()), int(yte.sum())


def _load(name):
    return np.load(os.path.join(CACHE_DIR, name + ".npy"))


def robust_fit(X):
    med = np.median(X, axis=0)
    q25 = np.percentile(X, 25, axis=0)
    q75 = np.percentile(X, 75, axis=0)
    iqr = q75 - q25
    iqr[iqr == 0] = 1.0
    return med, iqr


def robust_apply(X, med, iqr):
    return (X - med) / iqr


# ----------------------------------------------------------------------------
# Workers (module-level for Windows spawn)
# ----------------------------------------------------------------------------
def fold_worker(args):
    fold_id, train_idx, val_idx = args
    from model import XGBoostFromScratch
    rng = np.random.default_rng(SEED + fold_id)

    Xtr_raw = _load("Xtr_raw")
    ytr     = _load("ytr")

    X_ft_raw, y_ft = Xtr_raw[train_idx], ytr[train_idx]      # fold-train (natural)
    X_fv_raw, y_fv = Xtr_raw[val_idx],   ytr[val_idx]        # fold-val   (natural)

    med, iqr = robust_fit(X_ft_raw)                          # scaler on fold-train only
    X_ft = robust_apply(X_ft_raw, med, iqr)
    X_fv = robust_apply(X_fv_raw, med, iqr)

    sm = SMOTE(random_state=SEED, k_neighbors=5)             # SMOTE on fold-train only
    X_ft_bal, y_ft_bal = sm.fit_resample(X_ft, y_ft)

    t0 = time.time()
    model = XGBoostFromScratch(**HYPERPARAMS)
    model.fit(X_ft_bal, y_ft_bal, verbose=False)
    dt = time.time() - t0

    # Validation (out-of-fold) probabilities on natural distribution
    val_proba = model.predict_proba(X_fv)

    # Train-side estimate on a natural (pre-SMOTE) subsample incl. all fraud
    fraud_i = np.where(y_ft == 1)[0]
    legit_i = np.where(y_ft == 0)[0]
    keep_legit = rng.choice(legit_i, size=min(len(legit_i),
                            max(0, TRAIN_EVAL_SUBSAMPLE - len(fraud_i))), replace=False)
    sub = np.concatenate([fraud_i, keep_legit])
    tr_proba = model.predict_proba(X_ft[sub])
    y_tr_sub = y_ft[sub]

    res = dict(
        fold=fold_id,
        seconds=dt,
        class_weight=float(model.class_weight_),
        n_train_bal=int(len(y_ft_bal)),
        train_auc=float(roc_auc_score(y_tr_sub, tr_proba)),
        train_auprc=float(average_precision_score(y_tr_sub, tr_proba)),
        val_auc=float(roc_auc_score(y_fv, val_proba)),
        val_auprc=float(average_precision_score(y_fv, val_proba)),
        val_idx=val_idx.tolist(),
        val_proba=val_proba.tolist(),
    )
    print(f"[fold {fold_id}] done in {dt:.0f}s  "
          f"train_auc={res['train_auc']:.4f}  val_auc={res['val_auc']:.4f}",
          flush=True)
    return res


def final_worker(_):
    from model import XGBoostFromScratch
    rng = np.random.default_rng(SEED)

    Xtr_raw, Xte_raw = _load("Xtr_raw"), _load("Xte_raw")
    ytr, yte         = _load("ytr"), _load("yte")

    med, iqr = robust_fit(Xtr_raw)
    X_tr = robust_apply(Xtr_raw, med, iqr)
    X_te = robust_apply(Xte_raw, med, iqr)

    sm = SMOTE(random_state=SEED, k_neighbors=5)
    X_tr_bal, y_tr_bal = sm.fit_resample(X_tr, ytr)

    t0 = time.time()
    model = XGBoostFromScratch(**HYPERPARAMS)
    model.fit(X_tr_bal, y_tr_bal, verbose=False)
    dt = time.time() - t0

    test_proba = model.predict_proba(X_te)

    fraud_i = np.where(ytr == 1)[0]
    legit_i = np.where(ytr == 0)[0]
    keep_legit = rng.choice(legit_i, size=min(len(legit_i),
                            max(0, TRAIN_EVAL_SUBSAMPLE - len(fraud_i))), replace=False)
    sub = np.concatenate([fraud_i, keep_legit])
    tr_proba = model.predict_proba(X_tr[sub])
    y_tr_sub = ytr[sub]

    res = dict(
        seconds=dt,
        final_loss=float(model.loss_history_[-1]),
        class_weight=float(model.class_weight_),
        train_auc=float(roc_auc_score(y_tr_sub, tr_proba)),
        train_auprc=float(average_precision_score(y_tr_sub, tr_proba)),
        test_auc=float(roc_auc_score(yte, test_proba)),
        test_auprc=float(average_precision_score(yte, test_proba)),
        test_proba=test_proba.tolist(),
    )
    print(f"[final] done in {dt:.0f}s  final_loss={res['final_loss']:.5f}  "
          f"train_auc={res['train_auc']:.4f}  test_auc={res['test_auc']:.4f}",
          flush=True)
    return res


# ----------------------------------------------------------------------------
# Threshold sweep helpers
# ----------------------------------------------------------------------------
def sweep(y_true, proba, lo=0.50, hi=0.99, step=0.01):
    ths = np.round(np.arange(lo, hi + 1e-9, step), 4)
    rows = []
    for th in ths:
        pred = (proba >= th).astype(int)
        rows.append(dict(
            threshold=float(th),
            precision=float(precision_score(y_true, pred, zero_division=0)),
            recall=float(recall_score(y_true, pred, zero_division=0)),
            f1=float(f1_score(y_true, pred, zero_division=0)),
            tp=int(((pred == 1) & (y_true == 1)).sum()),
            fp=int(((pred == 1) & (y_true == 0)).sum()),
            fn=int(((pred == 0) & (y_true == 1)).sum()),
        ))
    return rows


def recommend(rows, min_precision=0.70):
    ok = [r for r in rows if r["precision"] >= min_precision]
    if ok:
        best_recall = max(r["recall"] for r in ok)
        cands = [r for r in ok if r["recall"] == best_recall]
        return min(cands, key=lambda r: r["threshold"]), True
    # fallback: closest to target precision with best F1
    return max(rows, key=lambda r: (r["precision"], r["f1"])), False


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    t_start = time.time()
    print("Preparing data ...", flush=True)
    shp_tr, shp_te, fr_tr, fr_te = prepare_data()
    print(f"  train {shp_tr} ({fr_tr} fraud) | test {shp_te} ({fr_te} fraud)", flush=True)

    ytr = _load("ytr")
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    fold_args = [(i, tr.astype(np.int64), va.astype(np.int64))
                 for i, (tr, va) in enumerate(skf.split(np.zeros(len(ytr)), ytr))]

    print(f"Launching {N_SPLITS} folds + final model in parallel ...", flush=True)
    fold_results, final_result = [], None
    with ProcessPoolExecutor(max_workers=N_SPLITS + 1) as ex:
        futs = {ex.submit(fold_worker, a): ("fold", a[0]) for a in fold_args}
        futs[ex.submit(final_worker, None)] = ("final", -1)
        for fut in as_completed(futs):
            kind, fid = futs[fut]
            r = fut.result()
            if kind == "fold":
                fold_results.append(r)
            else:
                final_result = r

    fold_results.sort(key=lambda r: r["fold"])

    # -- Assemble OOF validation predictions over the full training set --
    oof = np.full(len(ytr), np.nan)
    for r in fold_results:
        oof[np.array(r["val_idx"])] = np.array(r["val_proba"])
    assert not np.isnan(oof).any(), "some training rows have no OOF prediction"

    oof_auc   = roc_auc_score(ytr, oof)
    oof_auprc = average_precision_score(ytr, oof)

    # -- Threshold sweep on the OOF validation set --
    val_rows = sweep(ytr, oof)
    rec, met = recommend(val_rows, 0.70)

    # -- Confirm the recommended threshold on the held-out TEST set --
    yte = _load("yte")
    test_proba = np.array(final_result["test_proba"])
    test_rows  = sweep(yte, test_proba)

    def metrics_at(y_true, proba, th):
        pred = (proba >= th).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        return dict(
            threshold=float(th),
            precision=float(precision_score(y_true, pred, zero_division=0)),
            recall=float(recall_score(y_true, pred, zero_division=0)),
            f1=float(f1_score(y_true, pred, zero_division=0)),
            tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
        )

    th_rec = rec["threshold"]
    test_at_rec = metrics_at(yte, test_proba, th_rec)
    test_at_050 = metrics_at(yte, test_proba, 0.50)
    test_at_092 = metrics_at(yte, test_proba, 0.92)  # notebook's fixed threshold

    # -- CV overfitting summary --
    def ms(key):
        vals = np.array([r[key] for r in fold_results])
        return float(vals.mean()), float(vals.std())
    tr_auc_m, tr_auc_s   = ms("train_auc")
    va_auc_m, va_auc_s   = ms("val_auc")
    tr_apr_m, tr_apr_s   = ms("train_auprc")
    va_apr_m, va_apr_s   = ms("val_auprc")

    summary = dict(
        hyperparams=HYPERPARAMS, n_splits=N_SPLITS, seed=SEED,
        oof_auc=oof_auc, oof_auprc=oof_auprc,
        cv=dict(
            train_auc_mean=tr_auc_m, train_auc_std=tr_auc_s,
            val_auc_mean=va_auc_m, val_auc_std=va_auc_s,
            train_auprc_mean=tr_apr_m, train_auprc_std=tr_apr_s,
            val_auprc_mean=va_apr_m, val_auprc_std=va_apr_s,
            auc_gap=tr_auc_m - va_auc_m,
        ),
        folds=[{k: r[k] for k in
                ("fold", "seconds", "class_weight", "n_train_bal",
                 "train_auc", "val_auc", "train_auprc", "val_auprc")}
               for r in fold_results],
        final=dict(
            seconds=final_result["seconds"],
            final_loss=final_result["final_loss"],
            train_auc=final_result["train_auc"],
            test_auc=final_result["test_auc"],
            train_auprc=final_result["train_auprc"],
            test_auprc=final_result["test_auprc"],
            train_test_auc_gap=final_result["train_auc"] - final_result["test_auc"],
        ),
        recommendation=dict(
            met_precision_target=met,
            validation=rec,
            test_at_recommended=test_at_rec,
            test_at_0_50=test_at_050,
            test_at_0_92=test_at_092,
        ),
        wall_clock_seconds=time.time() - t_start,
    )

    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(dict(summary=summary, val_sweep=val_rows, test_sweep=test_rows),
                  f, indent=2)
    np.save(os.path.join(HERE, "oof_proba.npy"), oof)

    # ---------------- Plot 1: validation threshold sweep ----------------
    ths = [r["threshold"] for r in val_rows]
    prec = [r["precision"] for r in val_rows]
    rcl = [r["recall"] for r in val_rows]
    f1s = [r["f1"] for r in val_rows]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(ths, prec, color="#1565C0", lw=2, label="Precision")
    ax.plot(ths, rcl, color="#E53935", lw=2, label="Recall")
    ax.plot(ths, f1s, color="#2E7D32", lw=1.4, ls="--", alpha=0.8, label="F1")
    ax.axhline(0.70, color="#1565C0", lw=1, ls=":", alpha=0.7, label="Precision target 0.70")
    ax.axvline(th_rec, color="black", lw=1.5, ls="--", alpha=0.8,
               label=f"Recommended = {th_rec:.2f}")
    ax.scatter([th_rec], [rec["precision"]], color="#1565C0", s=70, zorder=5)
    ax.scatter([th_rec], [rec["recall"]], color="#E53935", s=70, zorder=5)
    ax.annotate(f"P={rec['precision']:.3f}\nR={rec['recall']:.3f}",
                xy=(th_rec, rec["recall"]),
                xytext=(th_rec + 0.02, rec["recall"] - 0.12),
                fontsize=10,
                arrowprops=dict(arrowstyle="->", lw=1))
    ax.set_xlabel("Decision threshold", fontsize=12)
    ax.set_ylabel("Score (fraud class)", fontsize=12)
    ax.set_title("Threshold Sweep on Out-of-Fold Validation Set (5-fold CV)\n"
                 "Precision & Recall vs threshold (0.50 -> 0.99, step 0.01)",
                 fontsize=13, fontweight="bold")
    ax.set_xlim(0.50, 0.99)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=10, loc="center left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(HERE, "threshold_sweep_validation.png"), dpi=130)
    plt.close(fig)

    # ---------------- Plot 2: CV train-vs-val bars ----------------
    fig, ax = plt.subplots(figsize=(10, 5))
    folds = [r["fold"] for r in fold_results]
    x = np.arange(len(folds))
    ax.bar(x - 0.18, [r["train_auc"] for r in fold_results], width=0.36,
           color="#90CAF9", label="Train ROC-AUC")
    ax.bar(x + 0.18, [r["val_auc"] for r in fold_results], width=0.36,
           color="#1565C0", label="Validation ROC-AUC")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {i}" for i in folds])
    ax.set_ylim(0.90, 1.005)
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Train vs Out-of-Fold Validation ROC-AUC per Fold\n"
                 "(small gap => no meaningful overfitting)",
                 fontsize=13, fontweight="bold")
    for i, r in enumerate(fold_results):
        ax.text(i, 0.905, f"gap {r['train_auc']-r['val_auc']:+.3f}",
                ha="center", fontsize=9)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(HERE, "cv_train_vs_val_auc.png"), dpi=130)
    plt.close(fig)

    # ---------------- Console report ----------------
    print("\n" + "=" * 68)
    print("  RESULTS")
    print("=" * 68)
    print(f"  Wall-clock: {summary['wall_clock_seconds']:.0f}s")
    print(f"\n  5-FOLD CV (overfitting check)")
    print(f"    Train ROC-AUC : {tr_auc_m:.4f} +/- {tr_auc_s:.4f}")
    print(f"    Val   ROC-AUC : {va_auc_m:.4f} +/- {va_auc_s:.4f}   (gap {tr_auc_m-va_auc_m:+.4f})")
    print(f"    Train AUPRC   : {tr_apr_m:.4f} +/- {tr_apr_s:.4f}")
    print(f"    Val   AUPRC   : {va_apr_m:.4f} +/- {va_apr_s:.4f}   (gap {tr_apr_m-va_apr_m:+.4f})")
    print(f"    Final full-train vs test ROC-AUC gap: "
          f"{final_result['train_auc']-final_result['test_auc']:+.4f} "
          f"(train {final_result['train_auc']:.4f} / test {final_result['test_auc']:.4f})")
    print(f"    OOF pooled ROC-AUC/AUPRC: {oof_auc:.4f} / {oof_auprc:.4f}")
    print(f"\n  RECOMMENDED THRESHOLD (validation): {th_rec:.2f} "
          f"(met precision>=0.70: {met})")
    print(f"    On validation : P={rec['precision']:.4f}  R={rec['recall']:.4f}  F1={rec['f1']:.4f}")
    print(f"    On TEST set   : P={test_at_rec['precision']:.4f}  R={test_at_rec['recall']:.4f}  "
          f"F1={test_at_rec['f1']:.4f}  (TP={test_at_rec['tp']} FP={test_at_rec['fp']} FN={test_at_rec['fn']})")
    print(f"    TEST @0.50    : P={test_at_050['precision']:.4f}  R={test_at_050['recall']:.4f}")
    print(f"    TEST @0.92    : P={test_at_092['precision']:.4f}  R={test_at_092['recall']:.4f}")
    print("=" * 68)
    print("  Wrote: results.json, oof_proba.npy, "
          "threshold_sweep_validation.png, cv_train_vs_val_auc.png")


if __name__ == "__main__":
    main()
