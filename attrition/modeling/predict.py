"""Predicciones con un modelo ya entrenado.

Uso:
    python -m attrition.modeling.predict                    # usa models/modelo_attrition.joblib
    python -m attrition.modeling.predict --run-id <RUN_ID>  # usa el modelo guardado en MLflow
"""

import argparse
from pathlib import Path

import joblib
from loguru import logger
import mlflow.sklearn
import pandas as pd

from attrition.config import ARCHIVO_PROCESADO, COLUMNA_OBJETIVO, MODELS_DIR, PROCESSED_DATA_DIR


class Predictor:
    """Carga un pipeline entrenado (local o desde MLflow) y genera predicciones."""

    def __init__(self, ruta_modelo=MODELS_DIR / "modelo_attrition.joblib", run_id=None):
        self.ruta_modelo = Path(ruta_modelo)
        self.run_id = run_id
        self.pipeline = None

    def cargar_modelo(self):
        if self.run_id:
            self.pipeline = mlflow.sklearn.load_model(f"runs:/{self.run_id}/model")
            logger.info(f"Modelo cargado desde MLflow (run {self.run_id})")
        else:
            self.pipeline = joblib.load(self.ruta_modelo)
            logger.info(f"Modelo cargado desde {self.ruta_modelo}")
        return self

    def predecir(self, df):
        X = df.drop(columns=COLUMNA_OBJETIVO, errors="ignore")
        resultado = df.copy()
        resultado["prob_attrition"] = self.pipeline.predict_proba(X)[:, 1]
        resultado["prediccion"] = ["Yes" if p == 1 else "No" for p in self.pipeline.predict(X)]
        return resultado


def main():
    parser = argparse.ArgumentParser(description="Genera predicciones de attrition.")
    parser.add_argument("--data", type=Path, default=PROCESSED_DATA_DIR / ARCHIVO_PROCESADO)
    parser.add_argument("--run-id", default=None, help="Run de MLflow del que se toma el modelo")
    parser.add_argument("--salida", type=Path, default=PROCESSED_DATA_DIR / "predicciones.csv")
    args = parser.parse_args()

    predictor = Predictor(run_id=args.run_id).cargar_modelo()
    resultado = predictor.predecir(pd.read_csv(args.data))
    resultado.to_csv(args.salida, index=False)
    logger.success(f"Predicciones guardadas en {args.salida} ({len(resultado)} filas)")


if __name__ == "__main__":
    main()
