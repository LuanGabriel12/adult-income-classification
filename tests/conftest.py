"""Fixtures sinteticas 100% offline para testes unitarios e de integracao."""

from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def dataset_sintetico_adult() -> pd.DataFrame:
    """Dataset sintetico pequeno (80 linhas) com todas as 14 features do Adult + target.

    Projetado para rodar offline em milissegundos, mantendo todas as classes,
    grupos de sex e race, valores ausentes controlados e duplicatas conhecidas.
    """
    linhas = []
    races = ["White", "Black", "Asian-Pac-Islander", "Other", "Amer-Indian-Eskimo"]
    educations = [
        ("Bachelors", 13),
        ("HS-grad", 9),
        ("Masters", 14),
        ("Some-college", 10),
    ]
    occupations = [
        "Exec-managerial",
        "Prof-specialty",
        "Craft-repair",
        "Adm-clerical",
        "Sales",
    ]
    workclasses = ["Private", "Self-emp-not-inc", "Local-gov", "State-gov"]

    for i in range(80):
        eh_positivo = i % 2 == 0
        sex = "Male" if i % 2 == 0 else "Female"
        race = races[i % len(races)]
        edu_nome, edu_num = educations[i % len(educations)]
        occ = occupations[i % len(occupations)]
        wc = workclasses[i % len(workclasses)]

        if eh_positivo:
            age = 42 + (i % 20)
            horas = 45 + (i % 15)
            cap_gain = 5000 if i % 4 == 0 else 0
            marital = "Married-civ-spouse"
            rel = "Husband" if sex == "Male" else "Wife"
            classe = ">50K"
        else:
            age = 20 + (i % 18)
            horas = 25 + (i % 16)
            cap_gain = 0
            marital = "Never-married"
            rel = "Not-in-family"
            classe = "<=50K"

        # Injeta ausencias controladas em workclass, occupation e native-country
        if i in (3, 19):
            wc = None
            occ = None
        if i == 7:
            native_country = None
        else:
            native_country = "United-States" if i % 10 != 0 else "Mexico"

        linhas.append(
            {
                "age": age,
                "workclass": wc,
                "fnlwgt": 100000 + i * 1500,
                "education": edu_nome,
                "education-num": edu_num,
                "marital-status": marital,
                "occupation": occ,
                "relationship": rel,
                "race": race,
                "sex": sex,
                "capital-gain": cap_gain,
                "capital-loss": 0,
                "hours-per-week": horas,
                "native-country": native_country,
                "class": classe,
            }
        )

    # Cria 1 duplicata exata controlada (linha 79 igual a linha 0)
    linhas[-1] = dict(linhas[0])
    return pd.DataFrame(linhas)


@pytest.fixture
def csv_sintetico_adult(tmp_path: Path, dataset_sintetico_adult: pd.DataFrame) -> Path:
    """Grava o dataset sintetico em CSV UTF-8/LF dentro de tmp_path."""
    caminho = tmp_path / "adult_sintetico.csv"
    dataset_sintetico_adult.to_csv(caminho, index=False, encoding="utf-8", lineterminator="\n")
    return caminho


@pytest.fixture
def cenario_fairness_conhecido() -> dict[str, pd.Series]:
    """Cenario de 16 observacoes com metricas de fairness calculaveis manualmente.

    Grupo A (n=8, 4 positivos reais, 4 negativos reais):
      TP=4, FN=0 -> Recall = 4/4 = 1.00
      FP=1, TN=3 -> FPR    = 1/4 = 0.25
      Acc = 7/8 = 0.875 | Selection Rate = 5/8 = 0.625

    Grupo B (n=8, 4 positivos reais, 4 negativos reais):
      TP=2, FN=2 -> Recall = 2/4 = 0.50
      FP=0, TN=4 -> FPR    = 0/4 = 0.00
      Acc = 6/8 = 0.750 | Selection Rate = 2/8 = 0.250

    Disparidades exatas esperadas:
      Equal Opportunity Difference (diff Recall) = 1.00 - 0.50 = 0.50
      Diff FPR                                   = 0.25 - 0.00 = 0.25
      Equalized Odds Difference                  = max(0.50, 0.25) = 0.50
      Recall Ratio                               = 0.50 / 1.00 = 0.50
      Selection Rate Diff                        = 0.625 - 0.250 = 0.375
      Selection Rate Ratio                       = 0.250 / 0.625 = 0.40
    """
    indices = list(range(100, 116))
    y_true = pd.Series(
        [
            ">50K", ">50K", ">50K", ">50K", "<=50K", "<=50K", "<=50K", "<=50K",
            ">50K", ">50K", ">50K", ">50K", "<=50K", "<=50K", "<=50K", "<=50K",
        ],
        index=indices,
        name="class",
    )
    y_pred = pd.Series(
        [
            ">50K", ">50K", ">50K", ">50K", ">50K", "<=50K", "<=50K", "<=50K",
            ">50K", ">50K", "<=50K", "<=50K", "<=50K", "<=50K", "<=50K", "<=50K",
        ],
        index=indices,
        name="y_pred",
    )
    grupo = pd.Series(
        ["GrupoA"] * 8 + ["GrupoB"] * 8,
        index=indices,
        name="grupo_sensivel",
    )
    return {"y_true": y_true, "y_pred": y_pred, "grupo": grupo}
