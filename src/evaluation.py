"""src/evaluation.py - Métricas, resúmenes multi-semilla y figuras."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

METRIC_COLS = ["Recall_%", "Precision_%", "F1_%", "FPR_%", "ROC_AUC", "AP",
               "fit_s", "predict_s", "flows_per_s"]


def compute_metrics(y_true, y_pred, scores=None):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    pre = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    auc = ap = np.nan
    if scores is not None and len(np.unique(y_true)) == 2:
        auc = float(roc_auc_score(y_true, scores))
        ap = float(average_precision_score(y_true, scores))
    return {"TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
            "Recall_%": 100 * rec, "Precision_%": 100 * pre, "F1_%": 100 * f1,
            "FPR_%": 100 * fpr, "ROC_AUC": auc, "AP": ap}


def per_class_recall(class_names, y_pred):
    """Recall por clase de ataque original (todas las no benignas)."""
    out = []
    y_pred = np.asarray(y_pred)
    for c in np.unique(class_names):
        m = class_names == c
        out.append({"class": c, "n": int(m.sum()), "pred_ataque_%": 100 * float(y_pred[m].mean())})
    return out


def summarize(runs):
    """Media ± desviación estándar entre semillas, por modelo (orden de aparición)."""
    g = runs.groupby("model", sort=False)[METRIC_COLS]
    mean, std = g.mean(), g.std(ddof=1).fillna(0.0)
    text = pd.DataFrame(index=mean.index)
    for c in METRIC_COLS:
        d = 4 if c in ("ROC_AUC", "AP") else (4 if c.endswith("_s") and c != "flows_per_s" else 2)
        if c == "flows_per_s":
            d = 0
        text[c] = [f"{m:.{d}f} ± {s:.{d}f}" for m, s in zip(mean[c], std[c])]
    return mean, std, text


def plot_confusions(cms, path):
    names = list(cms)
    ncols = min(4, len(names))
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.4 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, n in zip(axes.ravel(), names):
        cm = np.asarray(cms[n])
        ax.axis("on")
        ax.imshow(np.log10(cm + 1), cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=9, fontweight="bold")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Benigno", "Ataque"]); ax.set_yticklabels(["Benigno", "Ataque"])
        ax.set_xlabel("Predicho"); ax.set_ylabel("Real"); ax.set_title(n, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_comparison(mean, std, path):
    metrics = ["Recall_%", "Precision_%", "F1_%", "FPR_%"]
    models = list(mean.index)
    x = np.arange(len(metrics))
    w = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for i, m in enumerate(models):
        ax.bar(x + (i - (len(models) - 1) / 2) * w, mean.loc[m, metrics], w,
               yerr=std.loc[m, metrics], capsize=2, label=m)
    ax.set_xticks(x); ax.set_xticklabels(["Recall", "Precisión", "F1", "FPR"], fontsize=11)
    ax.set_ylabel("Porcentaje (%) — conjunto de PRUEBA")
    ax.set_title("Comparación de modelos (media ± desv. est. entre semillas)")
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
