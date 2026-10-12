"""Entrenamiento y evaluación del modelo de attrition, con registro en MLflow.

Forma de uso (todos los argumentos tienen valor por defecto):
    python -m attrition.modeling.train
    python -m attrition.modeling.train --C 0.1 --l1-ratio 0.8 --class-weight balanced

O a través del MLproject:
    mlflow run . --env-manager local -P C=0.1
"""

import argparse
import json
import os
import random
import warnings

import joblib
from loguru import logger
import mlflow
from mlflow.models import infer_signature
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from attrition.config import (
    ARCHIVO_PROCESADO,
    MODELS_DIR,
    NOMBRE_EXPERIMENTO,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
    SEMILLA,
)
from attrition.features import PreparadorDatos
from attrition.plots import Graficador

# Ocultamos el aviso repetido que sklearn muestra en cada fold.
warnings.filterwarnings("ignore", category=ConvergenceWarning)
# Aviso informativo de MLflow: el dataset no tiene nulos, así que las columnas enteras no afectan.
warnings.filterwarnings("ignore", message="Hint: Inferred schema contains integer")


class ModeloAttrition:
    """Pipeline de preprocesamiento + regresión logística Elastic Net."""

    def __init__(
        self, C=1.0, l1_ratio=0.5, max_iter=4000, class_weight="none", semilla=SEMILLA
    ):
        self.C = C
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.semilla = semilla

        # Con solver="saga", l1_ratio controla la mezcla: 0 = L2, 1 = L1, entre ambos = Elastic Net.
        # (En scikit-learn >= 1.8 ya no se usa penalty="elasticnet", basta con l1_ratio).
        regresion = LogisticRegression(
            solver="saga",
            C=C,
            l1_ratio=l1_ratio,
            max_iter=max_iter,
            class_weight=None if class_weight == "none" else class_weight,
            random_state=semilla,
        )
        preprocesador = PreparadorDatos().construir_preprocesador()
        self.pipeline = Pipeline(steps=[("ct", preprocesador), ("modelo", regresion)])

    def parametros(self):
        return {
            "C": self.C,
            "l1_ratio": self.l1_ratio,
            "max_iter": self.max_iter,
            "class_weight": self.class_weight,
            "seed": self.semilla,
            "solver": "saga",
        }

    def validacion_cruzada(self, X, y, n_splits=5, n_repeats=3):
        """Validación cruzada estratificada repetida. Devuelve media y desviación de cada métrica."""
        cv = RepeatedStratifiedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=self.semilla
        )
        metricas = ["accuracy", "f1", "recall", "roc_auc"]
        resultados = cross_validate(self.pipeline, X, y, scoring=metricas, cv=cv)

        resumen = {}
        for metrica in metricas:
            valores = resultados[f"test_{metrica}"]
            resumen[f"cv_{metrica}_mean"] = float(np.mean(valores))
            resumen[f"cv_{metrica}_std"] = float(np.std(valores))
        return resumen

    def entrenar(self, X, y):
        self.pipeline.fit(X, y)
        return self

    def predecir(self, X):
        return self.pipeline.predict(X)

    def predecir_probabilidad(self, X):
        # Columna 1 = probabilidad de la clase positiva (Attrition = Yes)
        return self.pipeline.predict_proba(X)[:, 1]

    def iteraciones(self):
        return int(self.pipeline.named_steps["modelo"].n_iter_.max())

    def convergio(self):
        # Si usó todas las iteraciones disponibles, el solver se detuvo antes de converger
        return self.iteraciones() < self.max_iter

    def coeficientes(self):
        nombres = self.pipeline.named_steps["ct"].get_feature_names_out()
        valores = self.pipeline.named_steps["modelo"].coef_[0]
        tabla = pd.DataFrame({"variable": nombres, "coeficiente": valores})
        tabla["abs"] = tabla["coeficiente"].abs()
        return tabla.sort_values("abs", ascending=False).drop(columns="abs")


class Evaluador:
    """Calcula las métricas de prueba. La clase positiva (1) es Attrition = Yes."""

    def __init__(self, y_real, y_pred, y_probabilidad, nombres_clases):
        self.y_real = y_real
        self.y_pred = y_pred
        self.y_probabilidad = y_probabilidad
        self.nombres_clases = nombres_clases

    def metricas(self):
        return {
            "test_accuracy": accuracy_score(self.y_real, self.y_pred),
            "test_balanced_accuracy": balanced_accuracy_score(self.y_real, self.y_pred),
            "test_precision": precision_score(
                self.y_real, self.y_pred, zero_division=0
            ),
            "test_recall": recall_score(self.y_real, self.y_pred, zero_division=0),
            "test_f1": f1_score(self.y_real, self.y_pred, zero_division=0),
            "test_roc_auc": roc_auc_score(self.y_real, self.y_probabilidad),
        }

    def reporte(self):
        return classification_report(
            self.y_real,
            self.y_pred,
            target_names=self.nombres_clases,
            digits=3,
            zero_division=0,
        )


class Experimento:
    """Une datos, modelo, evaluación y MLflow en una sola corrida."""

    def __init__(self, args):
        self.args = args
        self.modelo = ModeloAttrition(
            C=args.C,
            l1_ratio=args.l1_ratio,
            max_iter=args.max_iter,
            class_weight=args.class_weight,
            semilla=args.seed,
        )
        self.graficador = Graficador()

    def preparar_datos(self):
        """Partición 70/15/15 estratificada, igual que en el notebook.

        Train y Validation se juntan (85 %) porque se usa validación cruzada;
        Test (15 %) se guarda para la evaluación final.
        """
        df = pd.read_csv(self.args.data)
        X, y = PreparadorDatos().separar_x_y(df)
        semilla = self.args.seed

        X_train, X_resto, y_train, y_resto = train_test_split(
            X, y, train_size=0.70, stratify=y, random_state=semilla
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_resto, y_resto, test_size=0.5, stratify=y_resto, random_state=semilla
        )

        # El LabelEncoder se ajusta solo con train para no filtrar información
        codificador = LabelEncoder().fit(y_train)
        self.nombres_clases = list(codificador.classes_)  # ['No', 'Yes']

        self.X_train_val = pd.concat([X_train, X_val])
        self.y_train_val = codificador.transform(pd.concat([y_train, y_val]))
        self.X_test = X_test
        self.y_test = codificador.transform(y_test)

        logger.info(
            f"Train+Val: {len(self.y_train_val)} filas | Test: {len(self.y_test)} filas"
        )

    def registrar_parametros(self):
        parametros = self.modelo.parametros()
        parametros["cv_splits"] = self.args.cv_splits
        parametros["cv_repeats"] = self.args.cv_repeats
        parametros["data"] = self.args.data

        # Si la corrida se lanzó con `mlflow run`, MLflow ya registró los parámetros del
        # MLproject. Un parámetro no se puede registrar dos veces, así que esos se saltan.
        ya_registrados = mlflow.active_run().data.params
        for nombre, valor in parametros.items():
            if nombre not in ya_registrados:
                mlflow.log_param(nombre, valor)

    def ejecutar(self):
        self.preparar_datos()

        # Con `mlflow run` la corrida ya existe y el experimento se elige con --experiment-name,
        # así que solo lo fijamos aquí cuando el script se ejecuta directo con python.
        if "MLFLOW_RUN_ID" not in os.environ:
            mlflow.set_experiment(self.args.experiment_name)
        with mlflow.start_run(run_name=self.args.run_name):
            self.registrar_parametros()

            # 1. Validación cruzada sobre Train+Val
            metricas_cv = self.modelo.validacion_cruzada(
                self.X_train_val,
                self.y_train_val,
                self.args.cv_splits,
                self.args.cv_repeats,
            )

            # 2. Entrenamiento final y evaluación en Test
            self.modelo.entrenar(self.X_train_val, self.y_train_val)
            y_pred = self.modelo.predecir(self.X_test)
            y_prob = self.modelo.predecir_probabilidad(self.X_test)
            evaluador = Evaluador(self.y_test, y_pred, y_prob, self.nombres_clases)

            metricas = {**metricas_cv, **evaluador.metricas()}
            metricas["n_iter"] = self.modelo.iteraciones()
            for nombre, valor in metricas.items():
                mlflow.log_metric(nombre, valor)
            mlflow.set_tag("convergio", self.modelo.convergio())

            # 3. Artefactos
            self.guardar_artefactos(metricas, evaluador, y_pred, y_prob)

            # 4. Modelo
            self.registrar_modelo()

            self.mostrar_resumen(metricas, evaluador)

    def guardar_artefactos(self, metricas, evaluador, y_pred, y_prob):
        titulo = mlflow.active_run().info.run_name

        archivos = [
            self.graficador.matriz_confusion(
                self.y_test, y_pred, self.nombres_clases, titulo
            ),
            self.graficador.curva_roc(self.y_test, y_prob, titulo),
        ]

        ruta_metricas = REPORTS_DIR / "metricas.json"
        ruta_metricas.write_text(json.dumps(metricas, indent=2), encoding="utf-8")
        ruta_reporte = REPORTS_DIR / "reporte_clasificacion.txt"
        ruta_reporte.write_text(evaluador.reporte(), encoding="utf-8")
        ruta_coeficientes = REPORTS_DIR / "coeficientes.csv"
        self.modelo.coeficientes().to_csv(ruta_coeficientes, index=False)
        archivos += [ruta_metricas, ruta_reporte, ruta_coeficientes]

        for archivo in archivos:
            mlflow.log_artifact(str(archivo), artifact_path="evaluacion")

    def registrar_modelo(self):
        ejemplo = self.X_train_val.head(5)
        firma = infer_signature(ejemplo, self.modelo.predecir(ejemplo))
        mlflow.sklearn.log_model(
            sk_model=self.modelo.pipeline,
            name="model",
            signature=firma,
            input_example=ejemplo,
            skops_trusted_types=["numpy.dtype"],
        )
        # Copia local del último modelo entrenado (la usa predict.py por defecto)
        MODELS_DIR.mkdir(exist_ok=True)
        joblib.dump(self.modelo.pipeline, MODELS_DIR / "modelo_attrition.joblib")

    def mostrar_resumen(self, metricas, evaluador):
        run = mlflow.active_run()
        logger.success(
            f"Corrida '{run.info.run_name}' registrada (run_id={run.info.run_id})"
        )
        logger.info(
            f"Convergió: {self.modelo.convergio()} en {metricas['n_iter']} iteraciones"
        )
        logger.info(
            f"CV  accuracy={metricas['cv_accuracy_mean']:.4f}  f1={metricas['cv_f1_mean']:.4f}  "
            f"roc_auc={metricas['cv_roc_auc_mean']:.4f}"
        )
        logger.info(
            f"Test accuracy={metricas['test_accuracy']:.4f}  recall={metricas['test_recall']:.4f}  "
            f"f1={metricas['test_f1']:.4f}  roc_auc={metricas['test_roc_auc']:.4f}"
        )
        print(evaluador.reporte())


def fijar_semilla(semilla):
    """Fija las semillas globales. Además, cada componente de sklearn recibe random_state."""
    random.seed(semilla)
    np.random.seed(semilla)


def leer_argumentos():
    parser = argparse.ArgumentParser(
        description="Entrena una regresión logística Elastic Net y registra la corrida en MLflow.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        default=str(PROCESSED_DATA_DIR / ARCHIVO_PROCESADO),
        help="CSV limpio generado por attrition.features",
    )
    parser.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="Inverso de la regularización (más pequeño = más regularización)",
    )
    parser.add_argument(
        "--l1-ratio",
        type=float,
        default=0.5,
        help="Mezcla Elastic Net: 0 = L2 (Ridge), 1 = L1 (Lasso)",
    )
    parser.add_argument(
        "--max-iter", type=int, default=4000, help="Iteraciones máximas del solver"
    )
    parser.add_argument(
        "--class-weight",
        choices=["none", "balanced"],
        default="none",
        help="'balanced' compensa el desbalance de clases",
    )
    parser.add_argument("--seed", type=int, default=SEMILLA, help="Semilla aleatoria")
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--cv-repeats", type=int, default=3)
    parser.add_argument("--experiment-name", default=NOMBRE_EXPERIMENTO)
    parser.add_argument(
        "--run-name", default=None, help="Nombre de la corrida en MLflow"
    )
    return parser.parse_args()


def main():
    args = leer_argumentos()
    fijar_semilla(args.seed)
    Experimento(args).ejecutar()


if __name__ == "__main__":
    main()
