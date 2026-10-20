"""src/reference_models.py - Modelos de referencia: Isolation Forest (no supervisado) y Random Forest (cota supervisada)."""
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier

import config
from src.base import Detector, ScoreThresholdDetector
from src.data_loader import Preprocessor


class IsolationForestDetector(ScoreThresholdDetector):
    name = "Ref: IsolationForest"

    def __init__(self, seed=42):
        self.seed = seed

    def fit(self, X_train, y_train):
        rng = np.random.default_rng(self.seed)
        idx = np.flatnonzero(np.asarray(y_train) == 0)
        if len(idx) > config.MAX_FIT_ROWS:
            idx = rng.choice(idx, config.MAX_FIT_ROWS, replace=False)
        Xb = X_train.iloc[idx]
        self.prep_ = Preprocessor().fit(Xb)
        self.model_ = IsolationForest(n_estimators=config.IF_N_ESTIMATORS,
                                      random_state=self.seed, n_jobs=-1).fit(self.prep_.transform(Xb))
        return self

    def decision_scores(self, X):
        return -self.model_.score_samples(self.prep_.transform(X))


class RandomForestSupervised(Detector):
    name = "Ref: RandomForest (sup.)"
    needs_labels = True

    def __init__(self, seed=42):
        self.seed = seed

    def fit(self, X_train, y_train):
        y = np.asarray(y_train)
        rng = np.random.default_rng(self.seed)
        idx = np.arange(len(y))
        if len(idx) > config.MAX_FIT_ROWS:
            idx = rng.choice(idx, config.MAX_FIT_ROWS, replace=False)
        self.model_ = RandomForestClassifier(
            n_estimators=config.RF_N_ESTIMATORS, class_weight="balanced_subsample",
            n_jobs=-1, random_state=self.seed).fit(X_train.iloc[idx].to_numpy(), y[idx])
        return self

    def decision_scores(self, X):
        return self.model_.predict_proba(np.asarray(X))[:, 1]

    def predict(self, X):
        return (self.decision_scores(X) >= 0.5).astype(np.int8)
