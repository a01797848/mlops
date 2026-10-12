"""Limpieza de datos y preprocesamiento (feature engineering).

Uso:
    python -m attrition.features    # lee data/raw y guarda data/processed/attrition_limpio.csv
"""

import argparse
from pathlib import Path

from loguru import logger
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from attrition.config import (
    ARCHIVO_PROCESADO,
    COLUMNA_OBJETIVO,
    COLUMNAS_A_ELIMINAR,
    COLUMNAS_BINARIAS,
    COLUMNAS_NOMINALES,
    COLUMNAS_NUMERICAS,
    COLUMNAS_ORDINALES,
    PROCESSED_DATA_DIR,
)
from attrition.dataset import DatasetAttrition


class PreparadorDatos:
    """Agrupa los pasos de preparación decididos en el EDA."""

    def __init__(self, columnas_a_eliminar=COLUMNAS_A_ELIMINAR):
        self.columnas_a_eliminar = list(columnas_a_eliminar)

    def limpiar(self, df):
        """Quita las columnas constantes o identificadoras."""
        columnas_presentes = [c for c in self.columnas_a_eliminar if c in df.columns]
        return df.drop(columns=columnas_presentes)

    def separar_x_y(self, df):
        X = df.drop(columns=COLUMNA_OBJETIVO)
        y = df[COLUMNA_OBJETIVO]
        return X, y

    def construir_preprocesador(self):
        """Devuelve el ColumnTransformer con una transformación por tipo de variable.

        No se ajusta aquí: se ajusta dentro del pipeline del modelo, solo con los
        datos de entrenamiento, para evitar fuga de información.
        """
        numericas = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        ordinales = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ordinal", OrdinalEncoder()),
            ]
        )
        binarias = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(drop="if_binary", handle_unknown="ignore")),
            ]
        )
        nominales = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        return ColumnTransformer(
            transformers=[
                ("num", numericas, COLUMNAS_NUMERICAS),
                ("catOrd", ordinales, COLUMNAS_ORDINALES),
                ("catBin", binarias, COLUMNAS_BINARIAS),
                ("catNom", nominales, COLUMNAS_NOMINALES),
            ],
            remainder="drop",
        )


def main():
    parser = argparse.ArgumentParser(
        description="Limpia el dataset raw y lo guarda en data/processed."
    )
    parser.add_argument(
        "--salida", type=Path, default=PROCESSED_DATA_DIR / ARCHIVO_PROCESADO
    )
    args = parser.parse_args()

    df = DatasetAttrition().cargar()
    df_limpio = PreparadorDatos().limpiar(df)

    args.salida.parent.mkdir(parents=True, exist_ok=True)
    df_limpio.to_csv(args.salida, index=False)
    logger.success(f"Datos limpios guardados en {args.salida} {df_limpio.shape}")


if __name__ == "__main__":
    main()
