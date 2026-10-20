# Detección de LDDoS (SlowLoris) con agrupamiento inteligente — CICIoT2023

Examen Parcial de Inteligencia Artificial II (UNI-FIEE, 2026-2). Tema 10.

## Diseño (Ruta B)
- **Positivo (ataque de baja tasa):** `DDoS-SlowLoris`. **Negativo:** `BenignTraffic`. Los demás ataques se descartan.
- **Sin fuga de datos:** partición estratificada 60/10/30 (train/val/test); el escalado, PCA, centroides y umbrales se ajustan solo con *train*; los umbrales se calibran con benigno de *val*; todas las métricas se reportan en *test*.
- **5 semillas** (partición y muestreo distintos); se reporta media ± desviación.
- Se eliminan NaN/Inf y duplicados exactos *antes* de particionar.

| Modelo | Uso de etiquetas | Rol |
|---|---|---|
| Pre: Umbral Rate | ninguna (umbral = p99 del benigno de val) | línea base volumétrica |
| Pre: EWMA (ref.) | ninguna | opcional (`RUN_EWMA`); el CSV no es serie temporal |
| Post-A: PCA+KM benigno | solo identifica el benigno de train | detector *one-class* |
| Post-B: PCA+KM voto 1:1 | etiquetas de train | **semisupervisado** (propuesta original) |
| Abl: PCA+KM voto sin balancear | etiquetas de train | ablación del balanceo |
| Ref: IsolationForest | solo benigno de train | referencia no supervisada |
| Ref: RandomForest | etiquetas de train | cota supervisada |

## Datos
1. Descargue CICIoT2023 (CSV `train`, `test`, `validation`) desde el sitio oficial del Canadian Institute for Cybersecurity (UNB) y colóquelos en `data/raw/`.
2. Verifique los nombres de clase: `python main.py --list-classes`.
   Si difieren de `BenignTraffic` / `DDoS-SlowLoris`, corrija `config.py`.

## Ejecución
```bash
pip install -r requirements.txt
python main.py --list-classes
python main.py --sweep          # experimento completo + barrido (d, k) en validación
python tests/smoke_test.py      # verificación con datos SINTÉTICOS (no usar sus números)
```
Salidas en `output/`: `results_summary.csv`, `results_per_seed.csv`, `per_class_detection.csv`,
`sweep_results.csv`, `comparison_models.png`, `confusion_matrices.png`, `sweep_heatmap.png`,
`pca_variance.png`, `run_info.json` (versiones y configuración).

## Reproducibilidad
Todos los hiperparámetros están en `config.py`. Los valores del artículo deben copiarse de allí
y de `output/` (no escribir cifras a mano).
