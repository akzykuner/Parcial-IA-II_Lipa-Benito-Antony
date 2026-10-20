"""
src/evaluation.py - Módulo de Cálculo de Métricas y Generación de Gráficos
-------------------------------------------------------------------------
Calcula la matriz de confusión, métricas de rendimiento y genera la gráfica
comparativa Pre-Test vs. Post-Test.
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

def calculate_metrics(y_true, y_pred, model_name="Model"):
    """
    Calcula la matriz de confusión y las métricas clave de rendimiento:
    Recall (TPR), Precision, F1-Score, y False Positive Rate (FPR).
    """
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0

    tpr = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0.0
    precision = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0.0
    f1 = (2 * precision * tpr / (precision + tpr)) if (precision + tpr) > 0 else 0.0
    fpr = (fp / (fp + tn) * 100) if (fp + tn) > 0 else 0.0

    metrics = {
        'Model': model_name,
        'TP': int(tp),
        'FP': int(fp),
        'TN': int(tn),
        'FN': int(fn),
        'TPR (Recall %)': round(float(tpr), 2),
        'Precision (%)': round(float(precision), 2),
        'F1-Score (%)': round(float(f1), 2),
        'FPR (%)': round(float(fpr), 2)
    }
    return metrics

def plot_comparison(df_results, output_path):
    """
    Genera un gráfico comparativo de barras (Pre-Test vs Post-Test)
    para las tres métricas clave (Recall, F1-Score, FPR).
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    metrics_to_plot = ['TPR (Recall %)', 'F1-Score (%)', 'FPR (%)']
    x = np.arange(len(metrics_to_plot))
    width = 0.35

    # Indexación correcta usando .loc e índice de fila explícito
    pre_test_vals = df_results.loc[0, metrics_to_plot].values.astype(float)
    post_test_vals = df_results.loc[1, metrics_to_plot].values.astype(float)

    label_pre = str(df_results.loc[0, 'Model'])
    label_post = str(df_results.loc[1, 'Model'])

    rects1 = ax.bar(x - width/2, pre_test_vals, width, label=label_pre, color='#e74c3c')
    rects2 = ax.bar(x + width/2, post_test_vals, width, label=label_post, color='#2ecc71')

    ax.set_ylabel('Porcentaje (%)', fontsize=12, fontweight='bold')
    ax.set_title('Evaluación Experimental Cuasi-Experimental: Pre-Test vs. Post-Test\n(Detección de LDDoS mediante Agrupamiento Inteligente)', fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_to_plot, fontsize=11, fontweight='bold')
    ax.legend(fontsize=11)
    ax.set_ylim(0, 115)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold', fontsize=10)

    autolabel(rects1)
    autolabel(rects2)

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()