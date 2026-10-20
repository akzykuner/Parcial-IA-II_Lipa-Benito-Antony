# Detección de LDDoS (SlowLoris) con agrupamiento inteligente — CICIoT2023

Examen Parcial de Inteligencia Artificial II.

## Datos
1. Descargue CICIoT2023 (CSV `train`, `test`, `validation`) desde el sitio oficial del Canadian Institute for Cybersecurity (UNB) o Kaggle y colóquelos en `data/raw/`.
2. Verifique los nombres de clase: `python main.py --list-classes`.
   Si difieren de `BenignTraffic` / `DDoS-SlowLoris`, corrija `config.py`.

## Ejecución
```bash
pip install -r requirements.txt
python main.py --list-classes
python main.py --sweep          # experimento completo
python tests/smoke_test.py      # verificación con datos SINTÉTICOS
```
Salidas en `output/`: `results_summary.csv`, `results_per_seed.csv`, `per_class_detection.csv`,
`sweep_results.csv`, `comparison_models.png`, `confusion_matrices.png`, `sweep_heatmap.png`,
`pca_variance.png`, `run_info.json` (versiones y configuración).

## Reproducibilidad
Todos los hiperparámetros están en `config.py`.
