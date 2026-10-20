"""
src/baseline_model.py - Modelo Pre-Test (Sistema Tradicional EWMA / Umbrales Estáticos)
--------------------------------------------------------------------------------------
Este módulo implementa el enfoque de detección tradicional basado en Promedio Móvil
Exponencialmente Ponderado (EWMA) y umbrales estáticos de velocidad/volumen de tráfico.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import config

class EWMABaselineDetector:
    """
    Detector tradicional de anomalías basado en EWMA sobre métricas de velocidad de flujo.
    """
    def __init__(self, alpha=config.EWMA_ALPHA, threshold_std=config.EWMA_THRESHOLD_STD):
        self.alpha = alpha
        self.threshold_std = threshold_std

    def fit_predict(self, df_raw):
        rates = df_raw['Rate']
        # EWMA vectorizado en C/Pandas
        ewma_mean = rates.ewm(alpha=self.alpha).mean()
        ewma_std = rates.ewm(alpha=self.alpha).std().fillna(0)

        threshold = ewma_mean + self.threshold_std * ewma_std
        predictions = ((rates > threshold) & (rates > 15.0)).astype(int).values
        return predictions

def run_baseline_evaluation(df_raw, y_true=None):
    """
    Ejecuta el modelo tradicional EWMA sobre el dataframe crudo de tráfico y retorna predicciones.
    Acepta opcionalmente y_true para mantener compatibilidad con main.py.
    """
    detector = EWMABaselineDetector()
    y_pred_baseline = detector.fit_predict(df_raw)
    return y_pred_baseline