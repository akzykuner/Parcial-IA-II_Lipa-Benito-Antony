"""
src/intelligent_clustering.py - Modelos Post-Test (PCA + MiniBatchKMeans)
------------------------------------------------------------------------
Dos variantes, claramente diferenciadas por el uso de etiquetas:

A) PCAKMeansAnomalyDetector  -> ONE-CLASS. Ajusta scaler, PCA y centroides SOLO con
   tráfico benigno de entrenamiento. Puntaje = distancia al centroide benigno más
   cercano (en el espacio original estandarizado = distancia latente + residuo de
   reconstrucción). Umbral = percentil 99 del benigno de validación. No usa etiquetas
   de ataque en ningún paso.

B) BalancedClusterVoteClassifier -> SEMISUPERVISADO (propuesta original del artículo).
   Ajusta scaler/PCA/centroides con una muestra 1:1 del conjunto de ENTRENAMIENTO y
   etiqueta cada clúster por voto mayoritario con las etiquetas de entrenamiento.
   Con balanced=False es la ablación sin balancear (muestra con prevalencia natural).
Ninguna variante ve datos de validación/prueba al ajustarse.
"""
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances_argmin_min

import config
from src.base import Detector, ScoreThresholdDetector
from src.data_loader import Preprocessor


def _make_kmeans(k, seed):
    return MiniBatchKMeans(n_clusters=k, batch_size=config.KMEANS_BATCH_SIZE,
                           n_init=config.KMEANS_N_INIT, random_state=seed)


class PCAKMeansAnomalyDetector(ScoreThresholdDetector):
    name = "Post-A: PCA+KM benigno"
    needs_labels = False  # solo requiere saber qué flujos de entrenamiento son benignos

    def __init__(self, n_components=None, n_clusters=None, seed=42, include_residual=True):
        self.n_components = config.PCA_N_COMPONENTS if n_components is None else n_components
        self.n_clusters = config.N_CLUSTERS if n_clusters is None else n_clusters
        self.seed = seed
        self.include_residual = include_residual

    def fit(self, X_train, y_train):
        rng = np.random.default_rng(self.seed)
        idx = np.flatnonzero(np.asarray(y_train) == 0)
        if len(idx) > config.MAX_FIT_ROWS:
            idx = rng.choice(idx, config.MAX_FIT_ROWS, replace=False)
        Xb = X_train.iloc[idx]
        self.prep_ = Preprocessor().fit(Xb)
        Z0 = self.prep_.transform(Xb)
        self.pca_ = PCA(n_components=self.n_components, random_state=self.seed).fit(Z0)
        self.km_ = _make_kmeans(self.n_clusters, self.seed).fit(self.pca_.transform(Z0))
        self.centers_orig_ = self.pca_.inverse_transform(self.km_.cluster_centers_)
        return self

    def decision_scores(self, X):
        Xs = self.prep_.transform(X)
        if self.include_residual:
            return pairwise_distances_argmin_min(Xs, self.centers_orig_)[1]
        return pairwise_distances_argmin_min(self.pca_.transform(Xs),
                                             self.km_.cluster_centers_)[1]


class BalancedClusterVoteClassifier(Detector):
    needs_labels = True

    def __init__(self, n_components=None, n_clusters=None, seed=42, balanced=True):
        self.n_components = config.PCA_N_COMPONENTS if n_components is None else n_components
        self.n_clusters = config.N_CLUSTERS if n_clusters is None else n_clusters
        self.seed = seed
        self.balanced = balanced
        self.name = "Post-B: PCA+KM voto 1:1" if balanced else "Abl: PCA+KM voto sin balancear"

    def fit(self, X_train, y_train):
        y = np.asarray(y_train)
        rng = np.random.default_rng(self.seed)
        if self.balanced:
            b, a = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
            n = min(len(b), len(a), config.MAX_FIT_ROWS // 2)
            idx = np.concatenate([rng.choice(b, n, replace=False), rng.choice(a, n, replace=False)])
        else:
            idx = rng.choice(len(y), min(len(y), config.MAX_FIT_ROWS), replace=False)
        Xs = X_train.iloc[idx]
        ys = y[idx]
        self.prep_ = Preprocessor().fit(Xs)
        self.pca_ = PCA(n_components=self.n_components, random_state=self.seed).fit(self.prep_.transform(Xs))
        Z = self.pca_.transform(self.prep_.transform(Xs))
        self.km_ = _make_kmeans(self.n_clusters, self.seed).fit(Z)
        lab = self.km_.predict(Z)
        ratio = np.zeros(self.n_clusters)
        for c in range(self.n_clusters):
            m = lab == c
            ratio[c] = ys[m].mean() if m.any() else 0.0
        self.attack_ratio_ = ratio                       # fracción de ataque por clúster
        self.lut_ = (ratio >= 0.5).astype(np.int8)       # voto mayoritario
        return self

    def _clusters(self, X):
        return self.km_.predict(self.pca_.transform(self.prep_.transform(X)))

    def decision_scores(self, X):
        return self.attack_ratio_[self._clusters(X)]

    def predict(self, X):
        return self.lut_[self._clusters(X)]
