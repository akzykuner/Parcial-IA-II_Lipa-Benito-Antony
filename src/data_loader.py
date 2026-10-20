"""
src/data_loader.py - Módulo de Carga, Limpieza, Mapeo y Preprocesamiento
-------------------------------------------------------------------------
Añade soporte para etiquetas 'BenignTraffic' de CICIoT2023 y muestreo eficiente.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import os
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

import config

def standardize_column_names(df):
    df.columns = df.columns.str.strip()
    rename_dict = {
        'Protocol Type': 'Protocol_Type',
        'Tot sum': 'Tot_sum',
        'Tot size': 'Tot_size'
    }
    df.rename(columns=rename_dict, inplace=True)
    return df

def map_labels(y_raw):
    s = pd.Series(y_raw)
    if pd.api.types.is_numeric_dtype(s.dtype):
        return s.fillna(0).astype(int).values

    y_str = s.astype(str).str.strip().str.lower()
    # Soporta 'benign', 'benigntraffic', 'benign_traffic', 'normal', '0'
    benign_keywords = ['benign', 'benigntraffic', 'benign_traffic', 'normal', '0', '0.0']

    return np.where(y_str.isin(benign_keywords), config.BENIGN_LABEL, config.LDDOS_LABEL)

def load_and_preprocess_data(sample_limit=config.TOTAL_SAMPLES):
    raw_dir = config.RAW_DATA_DIR
    csv_files = glob.glob(os.path.join(raw_dir, "*.csv"))
    unified_csv = raw_dir / "CICIoT2023_raw.csv"

    if unified_csv.exists():
        print(f"Cargando archivo único: {unified_csv.name}")
        df = pd.read_csv(unified_csv)
    elif len(csv_files) > 0:
        print(f"Detectados {len(csv_files)} archivos CSV en data/raw/. Concatenando...")
        df_list = [pd.read_csv(f) for f in sorted(csv_files)]
        df = pd.concat(df_list, ignore_index=True)
    else:
        print("No se encontraron archivos CSV en data/raw/.")
        raise FileNotFoundError("Coloque los archivos CSV en data/raw/")

    df = standardize_column_names(df)

    # Mapeo de etiqueta objetivo
    target_col = config.TARGET_COLUMN if config.TARGET_COLUMN in df.columns else df.columns[-1]
    y_full = map_labels(df[target_col].values)
    df[config.TARGET_COLUMN] = y_full

    # Muestreo representativo para evitar saturar la memoria RAM con 7.8M de filas
    if sample_limit is not None and len(df) > sample_limit:
        print(f"Muestreando {sample_limit:,} filas de un total de {len(df):,} para procesamiento eficiente...")
        df = shuffle(df, random_state=config.RANDOM_SEED).reset_index(drop=True)
        df = df.iloc[:sample_limit].copy()
        y_full = df[config.TARGET_COLUMN].values

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.median(numeric_only=True), inplace=True)

    available_features = [col for col in config.FEATURE_COLUMNS if col in df.columns]
    X = df[available_features].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y_full, df
