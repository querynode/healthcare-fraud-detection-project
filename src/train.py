"""Train / evaluate models and score the unseen providers.

Usage: python -m src.train --raw data/raw --out outputs
"""
import argparse, json
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score, precision_recall_curve,
                             roc_auc_score, confusion_matrix, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import build_provider_features, load_split

SEED = 42


def models():
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
        "Random Forest": RandomForestClassifier(
            n_estimators=400, min_samples_leaf=2, class_weight="balanced_subsample",
            n_jobs=-1, random_state=SEED),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            learning_rate=0.05, max_depth=4, max_iter=300, l2_regularization=1.0,
            class_weight="balanced", random_state=SEED),
    }


def best_threshold(y, p):
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-9, None)
    return float(thr[f1.argmax()])


def main(raw, out):
    out = Path(out); fig = Path("reports/figures")
    fig.mkdir(parents=True, exist_ok=True); out.mkdir(exist_ok=True)

    labels, bene, ip, op = load_split(raw, "train")
    X = build_provider_features(ip, op, bene).merge(labels, on="Provider")
    y = (X.pop("PotentialFraud") == "Yes").astype(int)
    X.pop("Provider")
    print(f"Providers: {len(X)} | fraud rate: {y.mean():.1%} | features: {X.shape[1]}")

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

    rows, oof = [], {}
    for name, m in models().items():
        p = cross_val_predict(m, Xtr, ytr, cv=cv, method="predict_proba")[:, 1]
        oof[name] = p
        rows.append({"model": name, "cv_roc_auc": roc_auc_score(ytr, p),
                     "cv_pr_auc": average_precision_score(ytr, p)})
    cv_table = pd.DataFrame(rows).sort_values("cv_pr_auc", ascending=False)
    print(cv_table.round(4).to_string(index=False))
    best = cv_table.iloc[0]["model"]

    thr = best_threshold(ytr, oof[best])
    model = models()[best].fit(Xtr, ytr)
    pte = model.predict_proba(Xte)[:, 1]
    yhat = (pte >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(yte, yhat).ravel()
    metrics = {
        "best_model": best, "threshold": thr,
        "test_roc_auc": roc_auc_score(yte, pte), "test_pr_auc": average_precision_score(yte, pte),
        "test_f1": f1_score(yte, yhat), "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n_providers": int(len(X)), "fraud_rate": float(y.mean()), "n_features": int(X.shape[1]),
    }
    print(json.dumps(metrics, indent=2))
    cv_table.to_csv(out / "cv_results.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

    plt.figure(figsize=(6, 4.5))
    for n, p in oof.items():
        fpr, tpr, _ = roc_curve(ytr, p)
        plt.plot(fpr, tpr, label=f"{n} ({roc_auc_score(ytr, p):.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=.7)
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title("Cross-validated ROC"); plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(fig / "roc_cv.png", dpi=130); plt.close()

    plt.figure(figsize=(6, 4.5))
    prec, rec, _ = precision_recall_curve(yte, pte)
    plt.plot(rec, prec); plt.axhline(y.mean(), ls="--", c="grey", label="baseline (fraud rate)")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"Hold-out PR curve - {best}")
    plt.legend(); plt.tight_layout(); plt.savefig(fig / "pr_curve.png", dpi=130); plt.close()

    imp = permutation_importance(model, Xte, yte, scoring="average_precision",
                                 n_repeats=10, random_state=SEED, n_jobs=-1)
    imp_s = pd.Series(imp.importances_mean, index=X.columns).sort_values()
    imp_s.tail(15).plot.barh(figsize=(6.5, 5))
    plt.xlabel("Drop in PR-AUC when shuffled"); plt.title("Top 15 features (permutation importance)")
    plt.tight_layout(); plt.savefig(fig / "feature_importance.png", dpi=130); plt.close()
    imp_s.sort_values(ascending=False).to_csv(out / "feature_importance.csv", header=["importance"])

    top = imp_s.sort_values(ascending=False).index[:4]
    fig2, ax = plt.subplots(1, 4, figsize=(14, 3.4))
    for a, c in zip(ax, top):
        d = pd.DataFrame({"v": X[c].clip(upper=X[c].quantile(.99)), "Fraud": y.map({0: "No", 1: "Yes"})})
        d.boxplot(by="Fraud", column="v", ax=a, grid=False); a.set_title(c); a.set_xlabel("")
    fig2.suptitle(""); plt.tight_layout(); plt.savefig(fig / "top_features_by_class.png", dpi=130); plt.close()

    final = models()[best].fit(X, y)
    joblib.dump({"model": final, "features": list(X.columns), "threshold": thr}, out / "model.joblib")
    _, ub, uip, uop = load_split(raw, "unseen")
    UX = build_provider_features(uip, uop, ub).set_index("Provider").reindex(columns=X.columns, fill_value=0)
    pred = pd.DataFrame({"Provider": UX.index, "fraud_probability": final.predict_proba(UX)[:, 1]})
    pred["PotentialFraud"] = np.where(pred["fraud_probability"] >= thr, "Yes", "No")
    pred.sort_values("fraud_probability", ascending=False).to_csv(out / "unseen_predictions.csv", index=False)
    print("Unseen flagged:", (pred.PotentialFraud == "Yes").sum(), "of", len(pred))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw"); ap.add_argument("--out", default="outputs")
    a = ap.parse_args(); main(a.raw, a.out)
