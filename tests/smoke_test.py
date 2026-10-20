"""
tests/smoke_test.py - SOLO verifica que el código corre de punta a punta.
Genera un CSV SINTÉTICO diminuto con las columnas de CICIoT2023. Sus números NO
tienen ningún valor científico y NO deben usarse en el artículo.
Uso:  python tests/smoke_test.py
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config

ORIGINAL_NAMES = {"Protocol_Type": "Protocol Type", "Tot_sum": "Tot sum", "Tot_size": "Tot size"}
EXTRA = ["DNS", "Telnet", "SMTP", "SSH", "IRC", "DHCP", "ARP", "IPv", "LLC"]


def make(n, kind, rng):
    d = {c: rng.normal(0, 1, n) for c in config.FEATURE_COLUMNS}
    if kind == "BenignTraffic":
        d["Rate"] = rng.lognormal(3, 1, n)
    elif kind == "DDoS-SlowLoris":
        d["Rate"] = rng.lognormal(3.2, 1, n)       # tasa casi igual al benigno
        d["syn_count"] = rng.normal(2, 1, n)
        d["IAT"] = rng.normal(3, 1, n)
    else:                                           # flood volumétrico
        d["Rate"] = rng.lognormal(7, 1, n)
        d["Number"] = rng.normal(4, 1, n)
    for c in EXTRA:
        d[c] = np.zeros(n)
    df = pd.DataFrame(d).rename(columns=ORIGINAL_NAMES)
    df["label"] = kind
    return df


def main():
    rng = np.random.default_rng(0)
    tmp = Path(tempfile.mkdtemp())
    raw = tmp / "raw"
    raw.mkdir()
    for i in range(2):
        df = pd.concat([make(4000, "BenignTraffic", rng), make(600, "DDoS-SlowLoris", rng),
                        make(1000, "DoS-HTTP_Flood", rng)], ignore_index=True).sample(frac=1, random_state=i)
        df.iloc[5, 4] = np.inf            # fila con Inf
        df.iloc[7, 6] = np.nan            # fila con NaN
        df = pd.concat([df, df.iloc[:20]])  # duplicados exactos
        df.to_csv(raw / f"part{i}.csv", index=False)
    out = tmp / "out"
    base = [sys.executable, str(ROOT / "main.py"), "--data-dir", str(raw),
            "--output-dir", str(out), "--no-cache"]
    subprocess.run(base + ["--list-classes"], check=True)
    subprocess.run(base + ["--seeds", "2", "--sweep"], check=True)
    print("\nSMOKE TEST OK. Archivos:", sorted(p.name for p in out.iterdir()))


if __name__ == "__main__":
    main()
