"""
main.py - Pipeline experimental: LDDoS (SlowLoris) con agrupamiento inteligente
------------------------------------------------------------------------------
Uso:
    python main.py --list-classes      # verifica nombres de clase en los CSV
    python main.py                     # experimento completo (5 semillas)
    python main.py --sweep             # además, barrido de d y k (solo validación)
"""
import argparse
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import sklearn

import config
from src.baseline_model import EWMADetector, StaticVolumetricThreshold
from src.data_loader import list_classes, load_dataset, make_splits
from src.evaluation import compute_metrics, per_class_recall, plot_comparison, plot_confusions, summarize
from src.intelligent_clustering import BalancedClusterVoteClassifier, PCAKMeansAnomalyDetector
from src.reference_models import IsolationForestDetector, RandomForestSupervised
from src.sweep import run_sweep


def build_models(seed):
    models = [StaticVolumetricThreshold()]
    if config.RUN_EWMA:
        models.append(EWMADetector())
    models += [
        PCAKMeansAnomalyDetector(seed=seed),
        BalancedClusterVoteClassifier(seed=seed, balanced=True),
        BalancedClusterVoteClassifier(seed=seed, balanced=False),
        IsolationForestDetector(seed=seed),
        RandomForestSupervised(seed=seed),
    ]
    return models


def run_experiment(data, seeds, out_dir):
    rows, pc_rows, cms = [], [], {}
    for i, seed in enumerate(seeds):
        tr, va, te = make_splits(data.y, data.class_names, seed)
        Xtr, ytr = data.X.iloc[tr], data.y[tr]
        Xva, yva = data.X.iloc[va], data.y[va]
        Xte, yte, cte = data.X.iloc[te], data.y[te], data.class_names[te]
        print(f"\n=== Semilla {seed} | train={len(tr):,} val={len(va):,} test={len(te):,} "
              f"(ataques en test: {int(yte.sum()):,}) ===")
        for model in build_models(seed):
            t0 = time.perf_counter()
            model.fit(Xtr, ytr)
            model.calibrate(Xva, yva)
            fit_s = time.perf_counter() - t0

            t0 = time.perf_counter()
            y_pred = model.predict(Xte)
            pred_s = time.perf_counter() - t0
            scores = model.decision_scores(Xte)

            m = compute_metrics(yte, y_pred, scores)
            m.update(model=model.name, seed=seed, fit_s=fit_s, predict_s=pred_s,
                     flows_per_s=len(te) / max(pred_s, 1e-9))
            rows.append(m)
            for r in per_class_recall(cte, y_pred):
                pc_rows.append({"model": model.name, "seed": seed, **r})
            if i == 0:
                cms[model.name] = np.array([[m["TN"], m["FP"]], [m["FN"], m["TP"]]])
            print(f"  {model.name:<32} Recall={m['Recall_%']:6.2f}  Prec={m['Precision_%']:6.2f}  "
                  f"F1={m['F1_%']:6.2f}  FPR={m['FPR_%']:5.2f}  AP={m['AP']:.4f}  "
                  f"fit={fit_s:6.1f}s  pred={pred_s:5.2f}s")

    runs = pd.DataFrame(rows)
    runs.to_csv(out_dir / "results_per_seed.csv", index=False)
    pd.DataFrame(pc_rows).groupby(["model", "class"], sort=False)[["n", "pred_ataque_%"]] \
        .mean().to_csv(out_dir / "per_class_detection.csv")

    mean, std, text = summarize(runs)
    text.to_csv(out_dir / "results_summary.csv")
    mean.join(std, rsuffix="_std").to_csv(out_dir / "results_summary_numeric.csv")
    plot_comparison(mean, std, out_dir / "comparison_models.png")
    plot_confusions(cms, out_dir / "confusion_matrices.png")

    print("\n" + "=" * 100)
    print(f" RESULTADOS EN PRUEBA (media ± desv. est., {len(seeds)} semillas) ")
    print("=" * 100)
    print(text[["Recall_%", "Precision_%", "F1_%", "FPR_%", "ROC_AUC", "AP"]].to_string())
    print("-" * 100)
    print(text[["fit_s", "predict_s", "flows_per_s"]].to_string())
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list-classes", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--seeds", type=int, default=len(config.SEEDS))
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--data-dir", type=str, default=None)
    ap.add_argument("--output-dir", type=str, default=None)
    args = ap.parse_args()

    if args.data_dir:
        config.RAW_DATA_DIR = Path(args.data_dir)
    if args.output_dir:
        config.OUTPUT_DIR = Path(args.output_dir)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.list_classes:
        print(list_classes().to_string())
        return

    data = load_dataset(use_cache=not args.no_cache)
    print("\n[data] Resumen de limpieza:")
    print(json.dumps(data.stats, indent=2, ensure_ascii=False, default=int))
    n, n_att = len(data.y), int(data.y.sum())
    print(f"[data] Total final: {n:,} | Benigno: {n - n_att:,} | Ataque: {n_att:,} ({100 * n_att / n:.2f} %)")

    seeds = config.SEEDS[:args.seeds]
    runs = run_experiment(data, seeds, config.OUTPUT_DIR)

    if args.sweep:
        print("\n>>> Barrido de d y k (solo validación)")
        run_sweep(data, seeds[0], config.OUTPUT_DIR)

    info = {
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "python": platform.python_version(), "numpy": np.__version__,
        "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
        "seeds": seeds, "dataset_stats": data.stats,
        "config": {k: (str(v) if isinstance(v, Path) else v)
                   for k, v in vars(config).items() if k.isupper()},
    }
    (config.OUTPUT_DIR / "run_info.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False, default=int), encoding="utf-8")
    print(f"\nSalidas guardadas en: {config.OUTPUT_DIR}")


if __name__ == "__main__":
    main()

