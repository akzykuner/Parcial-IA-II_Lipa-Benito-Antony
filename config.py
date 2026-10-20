"""
config.py - Configuración Global e Hiperparámetros del Proyecto LDDoS
----------------------------------------------------------------------
Tema 10: Detección de DDoS de baja tasa (LDDoS) mediante agrupamiento inteligente
Dataset: CICIoT2023 (Flujos crudos de tráfico de red)
"""

import os
from pathlib import Path

# ------------------------------------------------------------------------------
# 1. RUTAS DEL PROYECTO
# ------------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
OUTPUT_DIR = BASE_DIR / "output"

# Crear directorios si no existen
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------------------------
# 2. CONFIGURACIÓN DEL DATASET Y MUESTREO EXPERIMENTAL (CICIoT2023)
# ------------------------------------------------------------------------------
RANDOM_SEED = 42
TOTAL_SAMPLES = 100000          # Muestra total para evaluación cuasi-experimental
BENIGN_RATIO = 0.80             # 80% Tráfico benigno (80,000 flujos)
LDDOS_RATIO = 0.20              # 20% Tráfico de ataque LDDoS sutil (20,000 flujos)

# Características clave de tráfico de red en CICIoT2023 para detección de LDDoS
FEATURE_COLUMNS = [
    'flow_duration', 'Header_Length', 'Protocol_Type', 'Duration',
    'Rate', 'Srate', 'Drate', 'fin_flag_number', 'syn_flag_number',
    'rst_flag_number', 'psh_flag_number', 'ack_flag_number',
    'ece_flag_number', 'cwr_flag_number', 'ack_count', 'syn_count',
    'fin_count', 'urg_count', 'rst_count', 'HTTP', 'HTTPS', 'TCP',
    'UDP', 'ICMP', 'Tot_sum', 'Min', 'Max', 'AVG', 'Std', 'Tot_size',
    'IAT', 'Number', 'Magnitue', 'Radius', 'Covariance', 'Variance', 'Weight'
]

TARGET_COLUMN = 'label'
BENIGN_LABEL = 0
LDDOS_LABEL = 1

# ------------------------------------------------------------------------------
# 3. HIPERPARÁMETROS DEL MODELO PRE-TEST (TRADICIONAL EWMA / UMBRALES)
# ------------------------------------------------------------------------------
EWMA_ALPHA = 0.10               # Factor de suavizado exponencial
EWMA_THRESHOLD_STD = 2.5        # Umbral en desviaciones estándar sobre el promedio móvil

# ------------------------------------------------------------------------------
# 4. HIPERPARÁMETROS DEL MODELO POST-TEST (AGRUPAMIENTO INTELIGENTE)
# ------------------------------------------------------------------------------
# Reducción de Dimensionalidad (PCA)
PCA_N_COMPONENTS = 10           # Conserva >90% de la varianza latente

# Agrupamiento basado en Densidad Adaptativo (DBSCAN / Two-Step Clustering)
DBSCAN_EPS = 0.50               # Radio de vecindad épsilon en espacio escalado
DBSCAN_MIN_SAMPLES = 15         # Número mínimo de puntos para formar un núcleo de clúster
KMEANS_BATCH_SIZE = 1000        # Tamaño de lote para Mini-Batch K-Means en pre-agrupamiento