"""
src/baseline_model.py - Modelos Pre-Test (sistema tradicional volumétrico)
--------------------------------------------------------------------------
1) StaticVolumetricThreshold: alarma si Rate supera un umbral estático. El umbral
   se fija con el percentil 99 del tráfico benigno de validación (sin ver ataques).
   Es la forma estacionaria de un detector EWMA con umbral fijo.
2) EWMADetector (referencia, opcional): EWMA recursivo sobre las filas en ORDEN DE
   ARCHIVO. Los CSV de CICIoT2023 no son una serie temporal ordenada, por lo que este
   detector es solo una referencia y NO debe presentarse como detección temporal.
   La media/desviación usan solo el pasado (shift(1)); no hay umbrales ocultos.
"""
import numpy as np
import pandas as pd

import config
from src.base import Detector, ScoreThresholdDetector


class StaticVolumetricThreshold(ScoreThresholdDetector):
    name = "Pre: Umbral Rate"

    def fit(self, X_train, y_train):
        return self  # sin parámetros: el umbral se fija en calibrate()

    def decision_scores(self, X):
        return X[config.RATE_COLUMN].to_numpy(dtype=np.float64)


class EWMADetector(Detector):
    name = "Pre: EWMA (ref.)"

    def __init__(self, alpha=None, k=None):
        self.alpha = config.EWMA_ALPHA if alpha is None else alpha
        self.k = config.EWMA_K if k is None else k
        self.threshold_ = self.k

    def fit(self, X_train, y_train):
        return self

    def decision_scores(self, X):
        r = pd.Series(X[config.RATE_COLUMN].to_numpy(dtype=np.float64))
        mean = r.ewm(alpha=self.alpha, adjust=False).mean().shift(1)
        std = r.ewm(alpha=self.alpha, adjust=False).std().shift(1)
        z = (r - mean) / (std + 1e-9)
        return z.fillna(0.0).to_numpy()

    def predict(self, X):
        return (self.decision_scores(X) > self.k).astype(np.int8)
