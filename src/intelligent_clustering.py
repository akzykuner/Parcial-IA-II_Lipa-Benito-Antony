"""
src/intelligent_clustering.py - Modelo Post-Test (Opción 5: Sub-muestreo Balanceado)
------------------------------------------------------------------------------------
Entrena la proyección PCA y los centroides de MiniBatchKMeans sobre una muestra
estratificada balanceada para evitar que la masa del 97.6% de ataques colapse el
espacio latente. Luego asigna e infiere para la totalidad de los datos (7.8M).
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
import config

class IntelligentLDDoSClusterer:
    """
    Agrupamiento Inteligente con Entrenamiento de Centroides en Muestra Balanceada (Opción 5).
    """
    def __init__(self, n_components=config.PCA_N_COMPONENTS, n_clusters=12):
        self.n_components = n_components
        self.n_clusters = n_clusters
        self.pca = PCA(n_components=self.n_components, random_state=config.RANDOM_SEED)

    def fit_predict(self, X_scaled, y_true=None):
        n_samples, n_features = X_scaled.shape

        if y_true is not None and len(np.unique(y_true)) > 1:
            benign_indices = np.flatnonzero(y_true == config.BENIGN_LABEL)
            lddos_indices = np.flatnonzero(y_true == config.LDDOS_LABEL)

            # Muestra equilibrada por clase para el ajuste de centroides
            n_samples_per_class = min(len(benign_indices), len(lddos_indices))

            np.random.seed(config.RANDOM_SEED)
            sampled_benign = np.random.choice(benign_indices, size=n_samples_per_class, replace=False)
            sampled_lddos = np.random.choice(lddos_indices, size=n_samples_per_class, replace=False)

            balanced_train_idx = np.concatenate([sampled_benign, sampled_lddos])

            # 1. Ajustar PCA en espacio balanceado
            X_pca_balanced = self.pca.fit_transform(X_scaled[balanced_train_idx])

            # Transformar espacio completo (7.8M)
            X_pca_full = self.pca.transform(X_scaled)

            # 2. Entrenar MiniBatchKMeans en espacio balanceado
            mbkm = MiniBatchKMeans(
                n_clusters=self.n_clusters,
                batch_size=config.KMEANS_BATCH_SIZE,
                random_state=config.RANDOM_SEED,
                n_init=3
            )
            mbkm.fit(X_pca_balanced)

            # Predecir clústeres para la totalidad de los datos (7.8M)
            cluster_labels_full = mbkm.predict(X_pca_full)
            cluster_labels_balanced = mbkm.labels_

            # 3. Perfilado de clústeres basado en la proporción balanceada
            cluster_is_attack = {}
            for c in range(self.n_clusters):
                mask = (cluster_labels_balanced == c)
                if np.sum(mask) > 0:
                    ratio_lddos = np.mean(y_true[balanced_train_idx][mask])
                    cluster_is_attack[c] = 1 if ratio_lddos >= 0.5 else 0
                else:
                    cluster_is_attack[c] = 0

            # Asignación de predicciones finales
            y_pred = np.array([cluster_is_attack[c] for c in cluster_labels_full], dtype=int)

        else:
            X_pca_full = self.pca.fit_transform(X_scaled)
            mbkm = MiniBatchKMeans(
                n_clusters=self.n_clusters,
                batch_size=config.KMEANS_BATCH_SIZE,
                random_state=config.RANDOM_SEED,
                n_init=3
            )
            cluster_labels_full = mbkm.fit_predict(X_pca_full)
            centroids = mbkm.cluster_centers_
            distances = np.linalg.norm(X_pca_full - centroids[cluster_labels_full], axis=1)
            threshold = np.percentile(distances, 85)
            y_pred = np.where(distances >= threshold, 1, 0)

        return y_pred, X_pca_full

def run_intelligent_clustering_evaluation(X_scaled, y_true=None):
    clusterer = IntelligentLDDoSClusterer()
    y_pred_clustering, X_pca = clusterer.fit_predict(X_scaled, y_true=y_true)
    return y_pred_clustering, X_pca