"""Gráficas de evaluación del modelo."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # para guardar imágenes sin abrir ventanas
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

from attrition.config import FIGURES_DIR


class Graficador:
    """Genera y guarda las gráficas en reports/figures."""

    def __init__(self, carpeta=FIGURES_DIR):
        self.carpeta = Path(carpeta)
        self.carpeta.mkdir(parents=True, exist_ok=True)

    def matriz_confusion(self, y_real, y_pred, etiquetas, titulo):
        fig, ax = plt.subplots(figsize=(5, 4.5))
        ConfusionMatrixDisplay.from_predictions(
            y_real, y_pred, display_labels=etiquetas, cmap="Blues", colorbar=False, ax=ax
        )
        ax.set_title(titulo)
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Valor real")
        return self._guardar(fig, "matriz_confusion.png")

    def curva_roc(self, y_real, y_probabilidad, titulo):
        fig, ax = plt.subplots(figsize=(5, 4.5))
        RocCurveDisplay.from_predictions(y_real, y_probabilidad, name="Regresión logística", ax=ax)
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray")  # referencia de un modelo al azar
        ax.set_title(titulo)
        ax.set_xlabel("Tasa de falsos positivos")
        ax.set_ylabel("Tasa de verdaderos positivos")
        return self._guardar(fig, "curva_roc.png")

    def _guardar(self, fig, nombre_archivo):
        ruta = self.carpeta / nombre_archivo
        fig.tight_layout()
        fig.savefig(ruta, dpi=120)
        plt.close(fig)
        return ruta
