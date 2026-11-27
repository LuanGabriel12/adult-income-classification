"""Testes de pre-processamento, anti-leakage, metricas, matriz de confusao e selecao."""

import pandas as pd
import pytest

from src.config import MODEL_DUMMY, MODEL_LOGISTIC
from src.data.loader import separar_treino_teste, validar_estrutura_dataset
from src.modeling.evaluation import (
    calcular_estatisticas_folds,
    calcular_metricas_classificacao,
    selecionar_melhor_modelo,
)
from src.modeling.training import criar_pipeline_modelo, gerar_previsoes, treinar_pipeline


def test_preprocessamento_trata_ausentes_numericos_categoricos_e_categoria_desconhecida() -> None:
    """Pipeline deve imputar NaNs numericos/categoricos e ignorar categorias ineditas no teste."""
    X_train = pd.DataFrame(
        {
            "age": [20.0, 40.0, None, 60.0, 22.0, 55.0],
            "workclass": ["Private", "Local-gov", None, "Private", "Private", "Local-gov"],
        }
    )
    y_train = pd.Series(["<=50K", ">50K", "<=50K", ">50K", "<=50K", ">50K"], name="class")

    X_test = pd.DataFrame(
        {
            "age": [None, 50.0],
            "workclass": ["CategoriaInedita", None],
        },
        index=[10, 11],
    )

    pipe = criar_pipeline_modelo(
        nome_modelo=MODEL_LOGISTIC,
        colunas_numericas=["age"],
        colunas_categoricas=["workclass"],
    )
    pipe = treinar_pipeline(pipe, X_train, y_train)
    preds = gerar_previsoes(pipe, X_test)

    assert list(preds.index) == [10, 11]
    assert set(preds.tolist()).issubset({"<=50K", ">50K"})


def test_preprocessamento_sem_data_leakage_aprende_apenas_no_treino(
    dataset_sintetico_adult: pd.DataFrame,
) -> None:
    """Comprova que a media do StandardScaler corresponde a X_train e nao a base total X."""
    X, y = validar_estrutura_dataset(dataset_sintetico_adult)
    X_train, X_test, y_train, _y_test = separar_treino_teste(X, y)

    # Forca um valor extremo apenas em X_test para comprovar que o scaler do treino nao o enxerga
    X_completo_com_extremo = X.copy()
    X_completo_com_extremo.loc[X_test.index[0], "age"] = 9999

    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

    pipe = criar_pipeline_modelo(MODEL_LOGISTIC, num_cols, cat_cols)
    pipe = treinar_pipeline(pipe, X_train, y_train)

    scaler = pipe.named_steps["preprocessador"].named_transformers_["numericas"].named_steps["scaler"]
    media_aprendida_age = float(scaler.mean_[0])
    assert media_aprendida_age == pytest.approx(float(X_train["age"].mean()), rel=1e-12)
    assert media_aprendida_age != pytest.approx(float(X_completo_com_extremo["age"].mean()), rel=1e-3)


def test_matriz_confusao_ordem_das_classes_e_metricas_manuais() -> None:
    """Verifica ordem [[TN, FP], [FN, TP]] e confere Accuracy, Precision, Recall e F1 manualmente.

    Dados construidos com valores distintos para cada celula:
      TN = 5, FP = 2, FN = 1, TP = 3 (total = 11)
      Accuracy  = (5 + 3) / 11 = 8/11
      Precision = 3 / (3 + 2)  = 3/5 = 0.6
      Recall    = 3 / (3 + 1)  = 3/4 = 0.75
      F1        = 2 * 0.6 * 0.75 / (0.6 + 0.75) = 0.9 / 1.35 = 2/3
      FPR       = 2 / (2 + 5)  = 2/7
      Sel Rate  = (3 + 2) / 11 = 5/11
    """
    y_true = pd.Series(
        ["<=50K"] * 5 + ["<=50K"] * 2 + [">50K"] * 1 + [">50K"] * 3,
        name="class",
    )
    y_pred = pd.Series(
        ["<=50K"] * 5 + [">50K"] * 2 + ["<=50K"] * 1 + [">50K"] * 3,
        name="y_pred",
    )

    res = calcular_metricas_classificacao(y_true, y_pred)
    assert res["matriz_confusao"] == {"tn": 5, "fp": 2, "fn": 1, "tp": 3}
    assert res["accuracy"] == pytest.approx(8 / 11, abs=1e-12)
    assert res["precision"] == pytest.approx(3 / 5, abs=1e-12)
    assert res["recall"] == pytest.approx(3 / 4, abs=1e-12)
    assert res["f1"] == pytest.approx(2 / 3, abs=1e-12)
    assert res["fpr"] == pytest.approx(2 / 7, abs=1e-12)
    assert res["selection_rate"] == pytest.approx(5 / 11, abs=1e-12)


def test_metricas_denominador_zero_sinalizam_indefinicao() -> None:
    """Quando nao ha previsoes positivas (como DummyClassifier), Precision e F1 ficam sinalizadas."""
    y_true = pd.Series(["<=50K", "<=50K", ">50K", ">50K"])
    y_pred = pd.Series(["<=50K", "<=50K", "<=50K", "<=50K"])

    res = calcular_metricas_classificacao(y_true, y_pred)
    assert res["accuracy"] == pytest.approx(0.5)
    assert res["recall"] == pytest.approx(0.0)
    assert res["precision"] is None
    assert res["f1"] is None
    assert res["denominadores_zero"]["precision_indefinida"] is True
    assert res["denominadores_zero"]["f1_indefinido"] is True


def test_estatisticas_folds_calculo_manual_exato() -> None:
    """Confere media, desvio-padrao populacional, min, max e amplitude em valores conhecidos."""
    # Valores [2, 4, 4, 4, 5, 5, 7, 9]: media=5.0, variancia pop=4.0, dp=2.0
    stats = calcular_estatisticas_folds([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
    assert stats["media"] == pytest.approx(5.0)
    assert stats["desvio_padrao"] == pytest.approx(2.0)
    assert stats["minimo"] == pytest.approx(2.0)
    assert stats["maximo"] == pytest.approx(9.0)
    assert stats["amplitude"] == pytest.approx(7.0)


def test_selecao_modelo_descarta_overfitting_severo() -> None:
    """Mesmo com F1 maior na validacao, modelo com gap treino-validacao > 0.05 deve ser descartado."""
    tabela = pd.DataFrame(
        [
            {
                "modelo": MODEL_DUMMY,
                "eh_baseline": True,
                "val_f1_mean": 0.0,
                "val_recall_mean": 0.0,
                "gap_train_val_accuracy": 0.0,
            },
            {
                "modelo": "Modelo Overfitting",
                "eh_baseline": False,
                "val_f1_mean": 0.75,
                "val_recall_mean": 0.70,
                "gap_train_val_accuracy": 0.18,
            },
            {
                "modelo": MODEL_LOGISTIC,
                "eh_baseline": False,
                "val_f1_mean": 0.66,
                "val_recall_mean": 0.60,
                "gap_train_val_accuracy": 0.002,
            },
        ]
    )

    escolhido, df_decisao, _resumo = selecionar_melhor_modelo(tabela, max_gap_overfitting=0.05)
    assert escolhido == MODEL_LOGISTIC
    assert bool(df_decisao.loc[df_decisao["modelo"] == "Modelo Overfitting", "elegivel_selecao"].iloc[0]) is False
