from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# ---------------------------------------------------------------------------
# Constantes del proyecto de attrition
# ---------------------------------------------------------------------------

# Datos de origen (Kaggle) y nombres de archivos
KAGGLE_DATASET = "pavansubhasht/ibm-hr-analytics-attrition-dataset"
ARCHIVO_RAW = "WA_Fn-UseC_-HR-Employee-Attrition.csv"
ARCHIVO_PROCESADO = "attrition_limpio.csv"

COLUMNA_OBJETIVO = "Attrition"

# Columnas sin valor predictivo (constantes o identificadores), según el EDA
COLUMNAS_A_ELIMINAR = ["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"]

# Tipos de variables definidos en el notebook de EDA
COLUMNAS_NUMERICAS = [
    "Age",
    "DailyRate",
    "DistanceFromHome",
    "HourlyRate",
    "MonthlyIncome",
    "MonthlyRate",
    "NumCompaniesWorked",
    "PercentSalaryHike",
    "TotalWorkingYears",
    "TrainingTimesLastYear",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager",
]
COLUMNAS_ORDINALES = [
    "Education",
    "EnvironmentSatisfaction",
    "JobInvolvement",
    "JobSatisfaction",
    "PerformanceRating",
    "RelationshipSatisfaction",
    "WorkLifeBalance",
    "StockOptionLevel",
    "JobLevel",
]
COLUMNAS_BINARIAS = ["OverTime", "Gender"]
COLUMNAS_NOMINALES = ["BusinessTravel", "Department", "EducationField", "JobRole", "MaritalStatus"]

# Valores por defecto del experimento
NOMBRE_EXPERIMENTO = "Attrition-LogReg-ElasticNet"
SEMILLA = 17

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
