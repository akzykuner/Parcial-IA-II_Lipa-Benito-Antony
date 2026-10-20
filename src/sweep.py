"""src/sweep.py - Barrido de (d, k) y varianza acumulada de PCA. Usa SOLO el conjunto de validación."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from src.data_loader import make_splits
from src.evaluation import compute_metrics
from src.intelligent_clustering import BalancedClusterVoteClassifier


def run_sweep(data, seed, out_dir):
    tr, va, _ = make_splits(data.y, data.class_names, seed)
    Xtr, ytr, Xva, yva = data.X.iloc[tr], data.y[tr], data.X.iloc[va], data.y[va]

    max_d = min(max(config.SWEEP_D), Xtr.shape[1])
    ref = BalancedClusterVoteClassifier(n_components=max_d, n_clusters=config.SWEEP_K[0], seed=seed).fit(Xtr, ytr)
    cum = np.cumsum(ref.pca_.explained_variance_ratio_)

    rows = []
    for d in config.SWEEP_D:
        for k in config.SWEEP_K:
            clf = BalancedClusterVoteClassifier(n_components=d, n_clusters=k, seed=seed).fit(Xtr, ytr)
            m = compute_metrics(yva, clf.predict(Xva), clf.decision_scores(Xva))
            rows.append({"d": d, "k": k, "F1_val_%": m["F1_%"], "Recall_val_%": m["Recall_%"],
                         "FPR_val_%": m["FPR_%"], "AP_val": m["AP"]})
            print(f"[sweep] d={d:>2} k={k:>2}  F1={m['F1_%']:.2f}  Recall={m['Recall_%']:.2f}  FPR={m['FPR_%']:.2f}")
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "sweep_results.csv", index=False)

    piv = df.pivot(index="d", columns="k", values="F1_val_%")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    im = ax.imshow(piv.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns)
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.values[i, j]:.1f}", ha="center", va="center", color="w", fontsize=8)
    ax.set_xlabel("k (clústeres)"); ax.set_ylabel("d (componentes PCA)")
    ax.set_title("F1 en VALIDACIÓN (%)")
    fig.colorbar(im); fig.tight_layout(); fig.savefig(out_dir / "sweep_heatmap.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, len(cum) + 1), 100 * cum, marker="o")
    ax.axvline(config.PCA_N_COMPONENTS, color="r", linestyle="--",
               label=f"d = {config.PCA_N_COMPONENTS}")
    ax.set_xlabel("Número de componentes"); ax.set_ylabel("Varianza acumulada (%)")
    ax.set_title("Varianza explicada acumulada (muestra balanceada de entrenamiento)")
    ax.grid(alpha=0.5); ax.legend(); fig.tight_layout()
    fig.savefig(out_dir / "pca_variance.png", dpi=200); plt.close(fig)

    d_cfg = config.PCA_N_COMPONENTS
    if d_cfg <= len(cum):
        print(f"[sweep] Varianza acumulada con d={d_cfg}: {100 * cum[d_cfg - 1]:.2f} %")
    best = df.sort_values("F1_val_%", ascending=False).iloc[0]
    print(f"[sweep] Mejor (d,k) por F1 en validación: d={int(best['d'])}, k={int(best['k'])} "
          f"(F1={best['F1_val_%']:.2f}). Configurado: d={d_cfg}, k={config.N_CLUSTERS}")
    return df
