# TC5061 · Experimentación con MLflow: AI vs Human

Clasificador binario de texto (humano = 0, IA = 1) con **TF-IDF + Regresión Logística**, entrenado sobre el dataset de Kaggle
[`datascikhan/ai-vs-human-text-classification-benchmark-50k`](https://www.kaggle.com/datasets/datascikhan/ai-vs-human-text-classification-benchmark-50k).
Cada ejecución queda registrada en MLflow Tracking (parámetros, métricas, artefactos y modelo).

## Estructura

```
train.py                     # carga, limpieza, entrenamiento, evaluación y registro en MLflow
run_experiments.sh           # 8 corridas que varían los hiperparámetros
requirements.txt             # versiones fijas de las dependencias
notebooks/01_eda_ai_vs_human.ipynb   # solo exploración de datos
data/                        # aquí va el CSV (no se versiona)
```

## Instalación

Requiere Python 3.12 o superior.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Descarga el CSV desde Kaggle y colócalo en `data/AI_vs_Human_Text_Classification_Benchmark.csv`.
Si omites `--data-path`, el script intenta descargarlo con `kagglehub`, y para eso necesitas credenciales de Kaggle.

## Servidor de MLflow

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db \
              --artifacts-destination ./mlartifacts --serve-artifacts \
              --host 127.0.0.1 --port 5000
```

Interfaz: http://localhost:5000

## Ejecución

Con valores por defecto:

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
python train.py --data-path data/AI_vs_Human_Text_Classification_Benchmark.csv
```

Con hiperparámetros:

```bash
python train.py --data-path data/AI_vs_Human_Text_Classification_Benchmark.csv \
                --C 0.001 --max-features 10 --ngram-max 1 --min-df 2 --seed 42
```

Todas las corridas del experimento:

```bash
bash run_experiments.sh
```

| Argumento | Default | Descripción |
|---|---|---|
| `--C` | 1.0 | Inverso de la regularización de la regresión logística |
| `--max-features` | 50000 | Tamaño máximo del vocabulario TF-IDF |
| `--ngram-max` | 2 | N-grama máximo (rango `(1, ngram_max)`) |
| `--min-df` | 2 | Frecuencia mínima de documento de un término |
| `--max-iter` | 1000 | Iteraciones máximas del solver |
| `--seed` | 42 | Semilla global (partición, `random`, `numpy`, solver) |
| `--test-size` | 0.20 | Proporción de prueba (estratificada) |
| `--data-path` | (kagglehub) | Ruta al CSV |
| `--tracking-uri` | `$MLFLOW_TRACKING_URI` o `sqlite:///mlflow.db` | Servidor de tracking |
| `--experiment-name` | `tc5061_ai_vs_human_tfidf_logreg` | Experimento de MLflow |
| `--run-name` | automático | Nombre de la corrida |

## Qué se registra en MLflow

- **Parámetros** (`log_param`): hiperparámetros, semilla, partición y columnas excluidas.
- **Métricas** (`log_metric`): accuracy, precision, recall, F1, ROC-AUC, log loss, accuracy y F1 de entrenamiento, tamaño del vocabulario y tiempo de entrenamiento.
- **Artefactos** (`log_artifact`): `graficas/evaluacion.png` (matriz de confusión y curva ROC), `reportes/metrics.json`, `reportes/classification_report.txt` y `reportes/log_limpieza.json`.
- **Modelo** (`mlflow.sklearn.log_model`): el pipeline completo, con firma y ejemplo de entrada.
- **Tag** `pred_fingerprint`: hash de las probabilidades predichas en prueba.

## Reproducibilidad

`set_seed()` fija `PYTHONHASHSEED`, `random` y `numpy`, y la misma semilla se pasa a `train_test_split` y a `LogisticRegression`.
Dos ejecuciones con los mismos argumentos producen las mismas métricas y el mismo `pred_fingerprint`
(en `run_experiments.sh`, las corridas `r4_mf_10` y `r8_mf_10_repeat` lo comprueban).
Las versiones de todas las dependencias están fijadas en `requirements.txt`.
