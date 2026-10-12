"""Descarga y lectura del dataset IBM HR Analytics Employee Attrition.

Uso:
    python -m attrition.dataset            # descarga el CSV a data/raw
    python -m attrition.dataset --forzar   # lo vuelve a descargar aunque ya exista
"""

import argparse
from pathlib import Path
import shutil

from loguru import logger
import pandas as pd

from attrition.config import ARCHIVO_RAW, KAGGLE_DATASET, RAW_DATA_DIR


class DatasetAttrition:
    """Obtiene el CSV original de Kaggle y lo lee como DataFrame.

    El archivo de data/raw no se modifica nunca; la limpieza se guarda
    aparte en data/processed (ver features.py).
    """

    def __init__(self, ruta_csv=RAW_DATA_DIR / ARCHIVO_RAW):
        self.ruta_csv = Path(ruta_csv)

    def existe(self):
        return self.ruta_csv.exists()

    def descargar(self):
        # kagglehub guarda el archivo en su propia caché, por eso lo copiamos a data/raw
        import kagglehub

        carpeta_kaggle = kagglehub.dataset_download(KAGGLE_DATASET)
        archivo_origen = Path(carpeta_kaggle) / ARCHIVO_RAW

        self.ruta_csv.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(archivo_origen, self.ruta_csv)
        logger.info(f"Dataset guardado en {self.ruta_csv}")

    def cargar(self):
        if not self.existe():
            raise FileNotFoundError(
                f"No se encontró {self.ruta_csv}. Primero ejecuta: python -m attrition.dataset"
            )
        df = pd.read_csv(self.ruta_csv)
        logger.info(f"Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
        return df


def main():
    parser = argparse.ArgumentParser(description="Descarga el dataset de attrition a data/raw.")
    parser.add_argument(
        "--forzar", action="store_true", help="Descargar aunque el archivo ya exista."
    )
    args = parser.parse_args()

    dataset = DatasetAttrition()
    if dataset.existe() and not args.forzar:
        logger.info(f"El archivo ya existe en {dataset.ruta_csv}, no se descarga de nuevo.")
    else:
        dataset.descargar()

    df = dataset.cargar()
    logger.success(f"Listo. Distribución de Attrition: {df['Attrition'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
