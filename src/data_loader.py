"""
src/data_loader.py - Carga por bloques, filtrado de clases, limpieza y particiones
-------------------------------------------------------------------------------
* Lee los CSV originales de CICIoT2023 por bloques y CONSERVA SOLO las clases
  configuradas (benigno + ataque de baja tasa). Evita cargar 7.8M de filas en RAM.
* Elimina NaN/Inf (se eliminan, no se imputan) y, opcionalmente, duplicados exactos.
* Provee particiones estratificadas train/val/test. El escalado NO se hace aquí:
  cada modelo ajusta su propio Preprocessor SOLO con datos de entrenamiento.
"""
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import config


def _norm(c):
    return str(c).strip().replace(" ", "_")


@dataclass
class Dataset:
    X: pd.DataFrame          # características (float32), orden original de archivo
    y: np.ndarray            # 0 = benigno, 1 = ataque de baja tasa
    class_names: np.ndarray  # etiqueta original (str)
    stats: dict              # conteos antes/después de cada paso de limpieza


# ----------------------------------------------------------------------------
def _csv_files(raw_dir):
    raw_dir = Path(raw_dir or config.RAW_DATA_DIR)
    files = sorted(raw_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No hay archivos .csv en {raw_dir}. Descargue CICIoT2023 (train/test/validation) "
            f"y colóquelos allí (ver README)."
        )
    return files


def list_classes(raw_dir=None):
    """Conteo de filas por etiqueta en TODOS los CSV (para verificar nombres de clase)."""
    lab = config.LABEL_COLUMN.lower()
    total = pd.Series(dtype="int64")
    for f in _csv_files(raw_dir):
        reader = pd.read_csv(f, usecols=lambda c: _norm(c).lower() == lab,
                             chunksize=config.CHUNK_SIZE)
        for chunk in reader:
            vc = chunk.iloc[:, 0].astype(str).str.strip().value_counts()
            total = total.add(vc, fill_value=0)
    return total.astype("int64").sort_values(ascending=False)


def _cache_path(files):
    key = json.dumps({
        "b": sorted(c.lower() for c in config.BENIGN_CLASSES),
        "p": sorted(c.lower() for c in config.POSITIVE_CLASSES),
        "f": config.FEATURE_COLUMNS,
        "dd": config.DROP_DUPLICATES,
        "files": [(f.name, f.stat().st_size) for f in files],
    }, sort_keys=True)
    return config.PROCESSED_DIR / f"filtered_{hashlib.md5(key.encode()).hexdigest()[:10]}.pkl"


def load_dataset(raw_dir=None, use_cache=True, verbose=True):
    files = _csv_files(raw_dir)
    cache = _cache_path(files)
    if use_cache and cache.exists():
        if verbose:
            print(f"[data] Cargando caché: {cache.name}")
        return pd.read_pickle(cache)

    lab = config.LABEL_COLUMN
    feats = list(config.FEATURE_COLUMNS)
    benign_lc = {c.lower() for c in config.BENIGN_CLASSES}
    wanted = benign_lc | {c.lower() for c in config.POSITIVE_CLASSES}
    keep_cols = set(feats)

    def usecols(c):
        n = _norm(c)
        return n in keep_cols or n.lower() == lab.lower()

    parts, offset, total_rows = [], 0, 0
    for f in files:
        if verbose:
            print(f"[data] Leyendo {f.name} ...")
        for chunk in pd.read_csv(f, usecols=usecols, chunksize=config.CHUNK_SIZE):
            chunk.columns = [lab if _norm(c).lower() == lab.lower() else _norm(c)
                             for c in chunk.columns]
            missing = [c for c in feats if c not in chunk.columns]
            if missing:
                raise ValueError(f"Faltan columnas en {f.name}: {missing}")
            lab_s = chunk[lab].astype(str).str.strip()
            mask = lab_s.str.lower().isin(wanted).to_numpy()
            kept = chunk.loc[mask, feats].copy()
            kept[lab] = lab_s[mask].to_numpy()
            parts.append(kept)
            total_rows += len(chunk)
            offset += len(chunk)

    df = pd.concat(parts, ignore_index=True)
    found = {c.lower() for c in df[lab].unique()}
    absent = wanted - found
    if absent:
        raise ValueError(
            f"Clases configuradas que no existen en los datos: {sorted(absent)}. "
            f"Ejecute `python main.py --list-classes` y corrija config.py."
        )

    stats = {"filas_totales_csv": int(total_rows),
             "tras_filtro_de_clases": df[lab].value_counts().to_dict()}

    df[feats] = df[feats].replace([np.inf, -np.inf], np.nan)
    n_bad = int(df[feats].isna().any(axis=1).sum())
    df = df.dropna(subset=feats)
    stats["filas_eliminadas_NaN_Inf"] = n_bad
    stats["tras_limpieza_NaN_Inf"] = df[lab].value_counts().to_dict()

    if config.DROP_DUPLICATES:
        n0 = len(df)
        df = df.drop_duplicates(subset=feats + [lab])
        stats["duplicados_eliminados"] = int(n0 - len(df))
    stats["final_por_clase"] = df[lab].value_counts().to_dict()

    df = df.reset_index(drop=True)   # el índice refleja el orden relativo de archivo
    class_names = df[lab].to_numpy()
    y = (~pd.Series(class_names).str.lower().isin(benign_lc)).astype(np.int8).to_numpy()
    X = df[feats].astype(np.float32)

    if len(np.unique(y)) < 2:
        raise ValueError("El conjunto filtrado tiene una sola clase.")
    data = Dataset(X=X, y=y, class_names=class_names, stats=stats)

    if use_cache:
        config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        pd.to_pickle(data, cache)
        if verbose:
            print(f"[data] Caché guardada: {cache.name}")
    return data


# ----------------------------------------------------------------------------
def make_splits(y, class_names, seed):
    """Partición estratificada train/val/test. Devuelve índices ordenados (orden de archivo)."""
    idx = np.arange(len(y))
    try:
        strat = np.asarray(class_names)
        tv, te = train_test_split(idx, test_size=config.TEST_SIZE, stratify=strat, random_state=seed)
        val_rel = config.VAL_SIZE / (1.0 - config.TEST_SIZE)
        tr, va = train_test_split(tv, test_size=val_rel, stratify=strat[tv], random_state=seed)
    except ValueError:  # clase con muy pocas filas: estratificar solo por 0/1
        tv, te = train_test_split(idx, test_size=config.TEST_SIZE, stratify=y, random_state=seed)
        val_rel = config.VAL_SIZE / (1.0 - config.TEST_SIZE)
        tr, va = train_test_split(tv, test_size=val_rel, stratify=y[tv], random_state=seed)
    return np.sort(tr), np.sort(va), np.sort(te)


class Preprocessor:
    """Z-Score (opcionalmente con log1p con signo). Se ajusta SOLO con datos de entrenamiento."""

    def __init__(self, log1p=None):
        self.log1p = config.LOG1P_FEATURES if log1p is None else log1p
        self.scaler = StandardScaler()

    def _t(self, X):
        X = np.asarray(X, dtype=np.float64)
        return np.sign(X) * np.log1p(np.abs(X)) if self.log1p else X

    def fit(self, X):
        self.scaler.fit(self._t(X))
        return self

    def transform(self, X):
        return self.scaler.transform(self._t(X))
