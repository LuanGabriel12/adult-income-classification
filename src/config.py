"""Configuracoes centrais, caminhos portaveis, schema e constantes do laboratorio."""

from pathlib import Path

# Diretorios portaveis resolvidos a partir da raiz do repositorio
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"
DOCS_DIR = PROJECT_ROOT / "docs"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Dataset canonico local gerado diretamente do OpenML ID 1590 (versao 2)
CANONICAL_DATASET_FILENAME = "adult_openml_1590_v2.csv"
CANONICAL_DATASET_PATH = DATA_DIR / CANONICAL_DATASET_FILENAME
CANONICAL_DATASET_SHA256 = (
    "8f970e20c9991979ec0a9ccb65ce97543d678709a2c6f8ef23f7fb32f40adddc"
)
CANONICAL_DATASET_BYTES = 5271056

# Metadados oficiais de proveniencia (OpenML 1590 v2 / UCI Adult ID 2)
DATASET_PROVENANCE = {
    "nome": "Adult / Census Income",
    "openml_id": 1590,
    "openml_nome": "adult",
    "openml_versao": 2,
    "openml_file_id": 1595261,
    "openml_arff_md5": "bb6510925e5d4b23d136715febb2cdf5",
    "openml_arff_sha256": (
        "77aa1703717a29f0b5642e94c3ba1defd2486f0b34d4d8eccc1b37a5f7d226b0"
    ),
    "openml_url": "https://www.openml.org/d/1590",
    "openml_licenca": "Public",
    "uci_id": 2,
    "uci_url": "https://archive.ics.uci.edu/dataset/2/adult",
    "uci_doi": "https://doi.org/10.24432/C5XW20",
    "uci_licenca": "CC BY 4.0 (Creative Commons Attribution 4.0 International)",
    "autoria": "Barry Becker e Ronny Kohavi (Silicon Graphics, 1996)",
    "periodo_historico": "1994 (US Census Bureau — Current Population Survey, CPS)",
    "registros_esperados": 48842,
    "features_esperadas": 14,
}

# Schema esperado das colunas do Adult / Census Income
NUMERIC_COLUMNS = [
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]

CATEGORICAL_COLUMNS = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

FEATURE_COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
]

TARGET_COLUMN = "class"
EXPECTED_ALL_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]

NEGATIVE_CLASS = "<=50K"
POSITIVE_CLASS = ">50K"
TARGET_CLASSES = (NEGATIVE_CLASS, POSITIVE_CLASS)

SENSITIVE_COLUMNS = ["sex", "race"]

# Categorias historicas oficiais da fonte (OpenML 1590 v2)
EXPECTED_CATEGORIES = {
    "workclass": [
        "Private",
        "Self-emp-not-inc",
        "Self-emp-inc",
        "Federal-gov",
        "Local-gov",
        "State-gov",
        "Without-pay",
        "Never-worked",
    ],
    "education": [
        "Bachelors",
        "Some-college",
        "11th",
        "HS-grad",
        "Prof-school",
        "Assoc-acdm",
        "Assoc-voc",
        "9th",
        "7th-8th",
        "12th",
        "Masters",
        "1st-4th",
        "10th",
        "Doctorate",
        "5th-6th",
        "Preschool",
    ],
    "marital-status": [
        "Married-civ-spouse",
        "Divorced",
        "Never-married",
        "Separated",
        "Widowed",
        "Married-spouse-absent",
        "Married-AF-spouse",
    ],
    "occupation": [
        "Tech-support",
        "Craft-repair",
        "Other-service",
        "Sales",
        "Exec-managerial",
        "Prof-specialty",
        "Handlers-cleaners",
        "Machine-op-inspct",
        "Adm-clerical",
        "Farming-fishing",
        "Transport-moving",
        "Priv-house-serv",
        "Protective-serv",
        "Armed-Forces",
    ],
    "relationship": [
        "Wife",
        "Own-child",
        "Husband",
        "Not-in-family",
        "Other-relative",
        "Unmarried",
    ],
    "race": [
        "White",
        "Asian-Pac-Islander",
        "Amer-Indian-Eskimo",
        "Other",
        "Black",
    ],
    "sex": ["Female", "Male"],
    "native-country": [
        "United-States",
        "Cambodia",
        "England",
        "Puerto-Rico",
        "Canada",
        "Germany",
        "Outlying-US(Guam-USVI-etc)",
        "India",
        "Japan",
        "Greece",
        "South",
        "China",
        "Cuba",
        "Iran",
        "Honduras",
        "Philippines",
        "Italy",
        "Poland",
        "Jamaica",
        "Vietnam",
        "Mexico",
        "Portugal",
        "Ireland",
        "France",
        "Dominican-Republic",
        "Laos",
        "Ecuador",
        "Taiwan",
        "Haiti",
        "Columbia",
        "Hungary",
        "Guatemala",
        "Nicaragua",
        "Scotland",
        "Thailand",
        "Yugoslavia",
        "El-Salvador",
        "Trinadad&Tobago",
        "Peru",
        "Hong",
        "Holand-Netherlands",
    ],
}

# Configuracao metodologica preservada da V1
TEST_SIZE = 0.20
RANDOM_STATE = 42
CV_SPLITS = 5
LOGISTIC_MAX_ITER = 1000
DECISION_TREE_CONTROLLED_MAX_DEPTH = 7

# Nomes padronizados dos modelos comparados
MODEL_DUMMY = "DummyClassifier (most_frequent)"
MODEL_LOGISTIC = "Logistic Regression (max_iter=1000)"
MODEL_DECISION_TREE_UNBOUNDED = "Decision Tree (sem limite)"
MODEL_DECISION_TREE_CONTROLLED = "Decision Tree (max_depth=7)"

# Criterio pre-registrado de selecao (aplicado somente sobre CV no treino)
MAX_OVERFITTING_GAP_ACCURACY = 0.05

# Limiares de alerta para auditoria de dados e fairness
RARE_CATEGORY_THRESHOLD_RATIO = 0.005  # < 0,5% da base
SMALL_GROUP_MIN_SAMPLES = 100
SMALL_GROUP_MIN_POSITIVES = 30

# Referencias historicas documentadas na V1 (uso estritamente comparativo)
V1_HISTORICAL_BENCHMARK = {
    "modelo": MODEL_LOGISTIC,
    "train_rows": 39073,
    "test_rows": 9769,
    "accuracy": 0.8523902139420616,
    "recall": 0.5885372112917023,
    "precision": 0.7413793103448276,
    "f1": 0.6561754887935145,
    "tn": 6951,
    "fp": 480,
    "fn": 962,
    "tp": 1376,
    "equal_opportunity_diff_race": 0.32195121951219513,
    "equalized_odds_diff_race": 0.32195121951219513,
}
