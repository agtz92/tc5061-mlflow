"""
Entrenamiento del clasificador AI vs Human (TF-IDF + Regresion Logistica) con MLflow Tracking.

Migrado desde el notebook de la actividad diagnostica (TC5061). El notebook en notebooks/
queda solo para la exploracion de datos; la carga, limpieza, entrenamiento y evaluacion
viven aqui.

Uso basico (valores por defecto):
    python train.py --data-path data/AI_vs_Human_Text_Classification_Benchmark.csv

Ver todos los argumentos:
    python train.py --help
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import tempfile
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # sin ventana: el script corre en terminal / servidor
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

TEXT_COL = "Raw_Text"
LABEL_COL = "Label"
KAGGLE_DATASET = "datascikhan/ai-vs-human-text-classification-benchmark-50k"
CSV_NAME = "AI_vs_Human_Text_Classification_Benchmark.csv"
# Columnas con fuga de la etiqueta (ver EDA en el notebook); el modelo solo usa Raw_Text.
COLS_EXCLUIDAS = [
    "Source_Model", "Model_Size", "Generation_Prompt_Type", "RAG_Context_Provided",
    "Model_Temperature", "Detection_Score_Benchmark_Model_A", "Text_ID",
]


# --------------------------------------------------------------------------------------
# Argumentos
# --------------------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Entrena TF-IDF + LogisticRegression y registra la corrida en MLflow.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Hiperparametros del modelo
    p.add_argument("--max-features", type=int, default=50000,
                   help="Tamano maximo del vocabulario TF-IDF.")
    p.add_argument("--ngram-max", type=int, default=2,
                   help="N-grama maximo (el minimo siempre es 1).")
    p.add_argument("--min-df", type=int, default=2,
                   help="Documentos minimos en los que debe aparecer un termino.")
    p.add_argument("--C", type=float, default=1.0,
                   help="Inverso de la fuerza de regularizacion de la regresion logistica.")
    p.add_argument("--max-iter", type=int, default=1000,
                   help="Iteraciones maximas del solver.")
    # Reproducibilidad y particion
    p.add_argument("--seed", type=int, default=42, help="Semilla aleatoria global.")
    p.add_argument("--test-size", type=float, default=0.20, help="Proporcion del conjunto de prueba.")
    # Datos y MLflow
    p.add_argument("--data-path", type=str, default=None,
                   help="Ruta al CSV. Si se omite, se descarga con kagglehub.")
    p.add_argument("--tracking-uri", type=str, default=os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"),
                   help="URI del servidor de tracking (o variable MLFLOW_TRACKING_URI).")
    p.add_argument("--experiment-name", type=str, default="tc5061_ai_vs_human_tfidf_logreg",
                   help="Nombre del experimento en MLflow.")
    p.add_argument("--run-name", type=str, default=None, help="Nombre de la corrida (opcional).")
    return p.parse_args(argv)


# --------------------------------------------------------------------------------------
# Reproducibilidad
# --------------------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """Fija todas las fuentes de aleatoriedad usadas por el script."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


# --------------------------------------------------------------------------------------
# Datos
# --------------------------------------------------------------------------------------
def resolve_data_path(data_path: str | None) -> Path:
    """Devuelve la ruta al CSV; si no se pasa, lo descarga de Kaggle con kagglehub."""
    if data_path:
        path = Path(data_path)
        if not path.exists():
            raise FileNotFoundError(f"No existe el archivo de datos: {path}")
        return path
    import kagglehub  # import diferido: solo se necesita si no hay CSV local

    folder = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    return folder / CSV_NAME


def load_data(path: Path) -> pd.DataFrame:
    """Lee el CSV y valida que existan las columnas necesarias."""
    df = pd.read_csv(path)
    missing = {TEXT_COL, LABEL_COL} - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas en el CSV: {missing}")
    return df


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Aplica la limpieza definida en el notebook y devuelve (texto, etiqueta) + log de pasos."""
    log: list[dict] = []

    def registrar(paso: str, antes: int, despues: int, motivo: str) -> None:
        log.append({"paso": paso, "filas_antes": antes, "filas_despues": despues,
                    "eliminadas": antes - despues, "motivo": motivo})

    d = df[[TEXT_COL, LABEL_COL]].copy()
    d[TEXT_COL] = d[TEXT_COL].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()

    antes = len(d)
    d = d[(d[TEXT_COL] != "") & (d[TEXT_COL] != "nan") & d[LABEL_COL].notna()]
    d[LABEL_COL] = d[LABEL_COL].astype(int)
    registrar("faltantes", antes, len(d), "Texto vacio o etiqueta nula")

    norm = d[TEXT_COL].str.lower()
    etiquetas = d.groupby(norm)[LABEL_COL].nunique()
    contradictorios = set(etiquetas[etiquetas > 1].index)
    antes = len(d)
    d = d[~norm.isin(contradictorios)]
    registrar("etiquetas_contradictorias", antes, len(d), "Mismo texto con etiqueta 0 y 1")

    antes = len(d)
    d = d.loc[~d[TEXT_COL].str.lower().duplicated(keep="first")]
    registrar("duplicados", antes, len(d), "Texto repetido, se conserva una copia")

    # Outliers cortos: IQR sobre log(numero de palabras), con minimo de 5 palabras
    n_pal = d[TEXT_COL].str.split().str.len()
    q1, q3 = np.log1p(n_pal).quantile([0.25, 0.75])
    lim_inf = max(5, int(np.expm1(q1 - 1.5 * (q3 - q1))))
    antes = len(d)
    d = d[n_pal >= lim_inf]
    registrar("outliers_cortos", antes, len(d), f"Menos de {lim_inf} palabras")

    return d.reset_index(drop=True), log


def split_data(df: pd.DataFrame, test_size: float, seed: int):
    """Particion estratificada train/test."""
    return train_test_split(df[TEXT_COL], df[LABEL_COL], test_size=test_size,
                            stratify=df[LABEL_COL], random_state=seed)


# --------------------------------------------------------------------------------------
# Modelo
# --------------------------------------------------------------------------------------
def build_model(args: argparse.Namespace) -> Pipeline:
    """Pipeline TF-IDF + LogisticRegression; el vectorizador se ajusta solo con train."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=args.max_features,
            ngram_range=(1, args.ngram_max),
            min_df=args.min_df,
            max_df=0.95,
            sublinear_tf=True,
            strip_accents="unicode",
            lowercase=True,
        )),
        ("clf", LogisticRegression(
            C=args.C, solver="liblinear", max_iter=args.max_iter, random_state=args.seed,
        )),
    ])


def train_model(model: Pipeline, X_train, y_train) -> float:
    t0 = time.time()
    model.fit(X_train, y_train)
    return time.time() - t0


# --------------------------------------------------------------------------------------
# Evaluacion
# --------------------------------------------------------------------------------------
def evaluate(model: Pipeline, X_train, y_train, X_test, y_test) -> tuple[dict, np.ndarray, np.ndarray]:
    """Calcula metricas en test (y train para detectar sobreajuste)."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred_tr = model.predict(X_train)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "log_loss": log_loss(y_test, y_proba),
        "train_accuracy": accuracy_score(y_train, y_pred_tr),
        "train_f1": f1_score(y_train, y_pred_tr, zero_division=0),
        "vocab_size": float(len(model.named_steps["tfidf"].vocabulary_)),
    }
    return metrics, y_pred, y_proba


def plot_evaluation(y_test, y_pred, y_proba, out_path: Path) -> None:
    """Matriz de confusion + curva ROC en una sola figura."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=["Humano", "IA"]).plot(
        ax=axes[0], cmap="Blues", values_format=",", colorbar=False)
    axes[0].set_title("Matriz de confusion (test)")
    RocCurveDisplay.from_predictions(y_test, y_proba, ax=axes[1], name="TF-IDF + LogReg")
    axes[1].plot([0, 1], [0, 1], "k--", alpha=0.5)
    axes[1].set_title("Curva ROC (test)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def predictions_fingerprint(y_proba: np.ndarray) -> str:
    """Huella de las probabilidades predichas; igual huella = misma corrida reproducida."""
    return hashlib.sha256(np.round(y_proba, 10).tobytes()).hexdigest()[:16]


# --------------------------------------------------------------------------------------
# MLflow
# --------------------------------------------------------------------------------------
def log_run(args, model, metrics, train_time, y_test, y_pred, y_proba, X_test,
            clean_log, n_rows) -> str:
    """Registra parametros, metricas, artefactos y modelo dentro de mlflow.start_run()."""
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment_name)
    run_name = args.run_name or f"C{args.C}_mf{args.max_features}_ng{args.ngram_max}_md{args.min_df}_s{args.seed}"

    with mlflow.start_run(run_name=run_name) as run:
        # Hiperparametros y configuracion
        for k, v in {
            "max_features": args.max_features, "ngram_max": args.ngram_max,
            "min_df": args.min_df, "C": args.C, "max_iter": args.max_iter,
            "seed": args.seed, "test_size": args.test_size,
            "solver": "liblinear", "max_df": 0.95, "sublinear_tf": True,
            "n_rows": n_rows, "n_test": len(y_test),
            "columnas_excluidas": ",".join(COLS_EXCLUIDAS),
        }.items():
            mlflow.log_param(k, v)

        # Metricas
        for k, v in metrics.items():
            mlflow.log_metric(k, float(v))
        mlflow.log_metric("train_time_s", train_time)

        # Artefactos
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            plot_evaluation(y_test, y_pred, y_proba, tmp / "evaluacion.png")
            (tmp / "metrics.json").write_text(json.dumps(
                {**metrics, "train_time_s": train_time}, indent=2))
            (tmp / "classification_report.txt").write_text(classification_report(
                y_test, y_pred, target_names=["Humano", "IA"], digits=4))
            (tmp / "log_limpieza.json").write_text(json.dumps(clean_log, indent=2, ensure_ascii=False))
            mlflow.log_artifact(str(tmp / "evaluacion.png"), artifact_path="graficas")
            mlflow.log_artifact(str(tmp / "metrics.json"), artifact_path="reportes")
            mlflow.log_artifact(str(tmp / "classification_report.txt"), artifact_path="reportes")
            mlflow.log_artifact(str(tmp / "log_limpieza.json"), artifact_path="reportes")

        # Modelo
        ejemplo = X_test.iloc[:3].to_frame()
        firma = infer_signature(ejemplo, model.predict(ejemplo[TEXT_COL]))
        mlflow.sklearn.log_model(model, name="modelo", signature=firma, input_example=ejemplo)

        mlflow.set_tags({
            "algoritmo": "TF-IDF + LogisticRegression",
            "problema": "clasificacion_binaria",
            "autor": "Alfredo",
            "pred_fingerprint": predictions_fingerprint(y_proba),
        })
        return run.info.run_id


# --------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    set_seed(args.seed)

    data_path = resolve_data_path(args.data_path)
    df_raw = load_data(data_path)
    df, clean_log = clean_data(df_raw)
    X_train, X_test, y_train, y_test = split_data(df, args.test_size, args.seed)

    model = build_model(args)
    train_time = train_model(model, X_train, y_train)
    metrics, y_pred, y_proba = evaluate(model, X_train, y_train, X_test, y_test)

    run_id = log_run(args, model, metrics, train_time, y_test, y_pred, y_proba, X_test,
                     clean_log, len(df))

    print(f"Run ID: {run_id}")
    for k, v in metrics.items():
        print(f"  {k:16s} {v:.6f}")
    print(f"  {'train_time_s':16s} {train_time:.2f}")
    print(f"  pred_fingerprint {predictions_fingerprint(y_proba)}")


if __name__ == "__main__":
    main()
