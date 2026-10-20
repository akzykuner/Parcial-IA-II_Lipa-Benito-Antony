"""
main.py - Pipeline Ejecutable Principal del Proyecto LDDoS (Tema 10)
---------------------------------------------------------------------
Coordina la ingesta de tráfico crudo CICIoT2023, ejecuta la evaluación de la
línea de base (Pre-Test EWMA) y el Agrupamiento Inteligente (Post-Test PCA + DBSCAN),
mostrando la tabla comparativa y guardando las gráficas.
"""

import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np

import config
from src.data_loader import load_and_preprocess_data
from src.baseline_model import run_baseline_evaluation
from src.intelligent_clustering import run_intelligent_clustering_evaluation
from src.evaluation import calculate_metrics, plot_comparison

def main():
    print("=" * 81)
    print(" EXPERIMENTO DE DETECCIÓN DE LDDOS MEDIANTE AGRUPAMIENTO INTELIGENTE (TEMA 10) ")
    print("=" * 81)
    print("\nCargando estructura de flujos de tráfico crudo CICIoT2023...")

    X_scaled, y_true, df_raw = load_and_preprocess_data(sample_limit=None)
    n_total = len(y_true)
    n_lddos = np.sum(y_true == config.LDDOS_LABEL)
    n_benign = np.sum(y_true == config.BENIGN_LABEL)

    print(f"Dataset Muestreado Total: {n_total:,} flujos | Benigno: {n_benign:,} ({n_benign/n_total*100:.1f}%) | LDDoS: {n_lddos:,} ({n_lddos/n_total*100:.1f}%)")

    # 1. EVALUACIÓN PRE-TEST (LÍNEA DE BASE EWMA TRADICIONAL)
    print("\n>>> Ejecutando Evaluación Pre-Test (Sistema Tradicional EWMA / Umbrales)...")
    y_pred_baseline = run_baseline_evaluation(df_raw, y_true)
    metrics_baseline = calculate_metrics(y_true, y_pred_baseline, model_name="Pre-Test (Tradicional EWMA)")
    print(f"    - Pre-Test Recall (TPR): {metrics_baseline['TPR (Recall %)']}% | F1-Score: {metrics_baseline['F1-Score (%)']}% | FPR: {metrics_baseline['FPR (%)']}%")

    # 2. EVALUACIÓN POST-TEST (AGRUPAMIENTO INTELIGENTE)
    print("\n>>> Ejecutando Evaluación Post-Test (Agrupamiento Inteligente PCA + DBSCAN/Two-Step)...")
    y_pred_clustering, X_pca = run_intelligent_clustering_evaluation(X_scaled, y_true=y_true)
    metrics_clustering = calculate_metrics(y_true, y_pred_clustering, model_name="Post-Test (Agrupamiento Inteligente)")
    print(f"    - Post-Test Recall (TPR): {metrics_clustering['TPR (Recall %)']}% | F1-Score: {metrics_clustering['F1-Score (%)']}% | FPR: {metrics_clustering['FPR (%)']}%")

    # 3. TABLA COMPARATIVA
    df_results = pd.DataFrame([metrics_baseline, metrics_clustering])
    print("\n" + "=" * 81)
    print(" TABLA COMPARATIVA EXPERIMENTAL (PRE-TEST VS POST-TEST) ")
    print("=" * 81)
    print(df_results.to_string(index=False))
    print("=" * 81)

    # 4. GENERACIÓN DE GRÁFICO
    chart_path = config.OUTPUT_DIR / "comparativa_pre_post_test.png"
    plot_comparison(df_results, chart_path)
    print(f"\nGráfico comparativo guardado en: {chart_path}")
    print("Experimento completado exitosamente.\n")

if __name__ == "__main__":
    main()