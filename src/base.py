"""src/base.py - Interfaz común de los detectores."""
import numpy as np
import config


class Detector:
    """Interfaz: fit -> calibrate -> predict / decision_scores."""
    name = "Detector"
    needs_labels = False
    threshold_ = None

    def fit(self, X_train, y_train):
        raise NotImplementedError

    def calibrate(self, X_val, y_val):
        return self

    def decision_scores(self, X):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError


class ScoreThresholdDetector(Detector):
    """
    Detector con puntaje de anomalía y umbral fijado SOLO con tráfico benigno de
    validación (percentil TARGET_BENIGN_QUANTILE). Todos los detectores de este
    tipo se comparan así al mismo FPR objetivo.
    """

    def calibrate(self, X_val, y_val, q=None):
        q = config.TARGET_BENIGN_QUANTILE if q is None else q
        benign = X_val[np.asarray(y_val) == 0]
        self.threshold_ = float(np.quantile(self.decision_scores(benign), q))
        return self

    def predict(self, X):
        return (self.decision_scores(X) > self.threshold_).astype(np.int8)
