"""
config.py - Configuración global del experimento (UNICA fuente de verdad)
=========================================================================
Tema 10: Detección de DDoS de baja tasa (LDDoS) mediante agrupamiento inteligente.
Dataset: CICIoT2023 (CSV de flujos). Ruta B: BenignTraffic vs DDoS-SlowLoris.

Todo valor que aparezca en el artículo debe salir de este archivo.
"""
from pathlib import Path

# ----------------------------------------------------------------------------
# 1. RUTAS
# ----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"            # aquí van train.csv, test.csv, validation.csv
PROCESSED_DIR = DATA_DIR / "processed"     # caché del subconjunto filtrado
OUTPUT_DIR = BASE_DIR / "output"

# ----------------------------------------------------------------------------
# 2. DEFINICIÓN DE CLASES (verificar con:  python main.py --list-classes)
# ----------------------------------------------------------------------------
LABEL_COLUMN = "label"
BENIGN_CLASSES = ["BenignTraffic"]
# Clase(s) positiva(s) = ataque de BAJA TASA. Puede ampliarse con otras DoS lentas
# si aparecen en --list-classes. NO incluir floods volumétricos aquí.
POSITIVE_CLASSES = ["DDoS-SlowLoris"]

# ----------------------------------------------------------------------------
# 3. CARACTERÍSTICAS (37 de las 46 del CSV procesado de CICIoT2023)
#    Se excluyen DNS, Telnet, SMTP, SSH, IRC, DHCP, ARP, IPv, LLC.
# ----------------------------------------------------------------------------
FEATURE_COLUMNS = [
    "flow_duration", "Header_Length", "Protocol_Type", "Duration",
    "Rate", "Srate", "Drate", "fin_flag_number", "syn_flag_number",
    "rst_flag_number", "psh_flag_number", "ack_flag_number",
    "ece_flag_number", "cwr_flag_number", "ack_count", "syn_count",
    "fin_count", "urg_count", "rst_count", "HTTP", "HTTPS", "TCP",
    "UDP", "ICMP", "Tot_sum", "Min", "Max", "AVG", "Std", "Tot_size",
    "IAT", "Number", "Magnitue", "Radius", "Covariance", "Variance", "Weight",
]
RATE_COLUMN = "Rate"

# ----------------------------------------------------------------------------
# 4. CARGA Y LIMPIEZA
# ----------------------------------------------------------------------------
CHUNK_SIZE = 500_000          # filas por bloque al leer los CSV (control de RAM)
DROP_DUPLICATES = True        # evita que filas idénticas caigan en train y test
LOG1P_FEATURES = False        # True = log1p con signo antes del Z-Score (ablación opcional)

# ----------------------------------------------------------------------------
# 5. PROTOCOLO EXPERIMENTAL
# ----------------------------------------------------------------------------
SEEDS = [42, 43, 44, 45, 46]  # una partición y un muestreo distintos por semilla
TEST_SIZE = 0.30              # fracción del total
VAL_SIZE = 0.10               # fracción del total (calibración de umbrales / barrido d,k)
TARGET_BENIGN_QUANTILE = 0.99 # umbral = percentil 99 del benigno de validación (FPR objetivo ~1 %)

# ----------------------------------------------------------------------------
# 6. HIPERPARÁMETROS DEL MODELO PROPUESTO
# ----------------------------------------------------------------------------
PCA_N_COMPONENTS = 5
N_CLUSTERS = 12
KMEANS_BATCH_SIZE = 10_000
KMEANS_N_INIT = 3
MAX_FIT_ROWS = 400_000        # tope de filas para ajustar cualquier modelo

# Barrido para justificar d y k (se evalúa SOLO en validación)
SWEEP_D = [2, 3, 5, 8, 10, 15]
SWEEP_K = [4, 8, 12, 16, 24]

# ----------------------------------------------------------------------------
# 7. LÍNEAS BASE Y MODELOS DE REFERENCIA
# ----------------------------------------------------------------------------
RUN_EWMA = False              # EWMA sobre orden de archivo (referencia; el CSV no es una serie temporal)
EWMA_ALPHA = 0.05
EWMA_K = 3.0
IF_N_ESTIMATORS = 100
RF_N_ESTIMATORS = 100
