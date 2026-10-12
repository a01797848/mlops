# attrition-mlflow

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Prediccion de attrition (IBM HR) con experimentos registrados en MLflow

**TC5061.10 – MLOps** · Mariely Toribio Rodríguez (A01797848)

Modelo base de regresión logística Elastic Net sobre el dataset *IBM HR Analytics Employee Attrition & Performance*, migrado del notebook a un paquete de Python con la plantilla Cookiecutter Data Science y con cada corrida registrada en MLflow.

## Cómo ejecutarlo

```bash
# 1. Entorno e instalación (instala también el paquete attrition en modo editable)
python -m venv .venv
.venv\Scripts\activate            # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt

# 2. Datos
python -m attrition.dataset        # descarga el CSV a data/raw
python -m attrition.features       # limpia y guarda data/processed/attrition_limpio.csv

# 3. Entrenamiento con valores por defecto
python -m attrition.modeling.train

# 4. O a través del MLproject, cambiando hiperparámetros
mlflow run . --env-manager local --experiment-name Attrition-LogReg-ElasticNet --run-name 03_C_0.1 -P C=0.1

# 5. Interfaz de MLflow (desde esta misma carpeta)
mlflow ui                          # http://127.0.0.1:5000
```

`python -m attrition.modeling.train --help` muestra todos los argumentos.

### Hiperparámetros

| Argumento (script) | Parámetro (MLproject) | Defecto | Descripción |
|---|---|---|---|
| `--C` | `C` | 1.0 | Inverso de la regularización |
| `--l1-ratio` | `l1_ratio` | 0.5 | 0 = L2 (Ridge), 1 = L1 (Lasso) |
| `--max-iter` | `max_iter` | 4000 | Iteraciones máximas del solver `saga` |
| `--class-weight` | `class_weight` | none | `none` o `balanced` |
| `--seed` | `seed` | 17 | Semilla para partición, validación cruzada y modelo |
| `--data` | `data` | data/processed/attrition_limpio.csv | CSV limpio |

### Qué se registra en MLflow

- **Parámetros:** los hiperparámetros anteriores, el solver y la configuración de la validación cruzada.
- **Métricas:** media y desviación de la validación cruzada 5×3 (`cv_accuracy`, `cv_f1`, `cv_recall`, `cv_roc_auc`) y métricas en Test (`test_accuracy`, `test_balanced_accuracy`, `test_precision`, `test_recall`, `test_f1`, `test_roc_auc`).
- **Artefactos** (carpeta `evaluacion`): matriz de confusión, curva ROC, `metricas.json`, reporte de clasificación y coeficientes.
- **Modelo:** el pipeline completo (preprocesamiento + regresión) con `mlflow.sklearn.log_model`.

### Reproducibilidad

La semilla (`--seed`) se pasa como `random_state` a las dos particiones `train_test_split`, a `RepeatedStratifiedKFold` y a `LogisticRegression` (el solver `saga` baraja los datos). Además se fijan `random` y `numpy`. Dos corridas con los mismos argumentos (`repro_1` y `repro_2`) dieron exactamente las mismas métricas y coeficientes. Las versiones usadas están en `requirements.txt`.

El análisis de resultados está en `reports/analisis_resultados.md`.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         attrition and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── attrition   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes attrition a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

