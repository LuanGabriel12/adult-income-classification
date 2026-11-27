"""Fabricas procedurais de pipelines e treinamento dos modelos avaliados."""

from typing import Any

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.config import (
    DECISION_TREE_CONTROLLED_MAX_DEPTH,
    LOGISTIC_MAX_ITER,
    MODEL_DECISION_TREE_CONTROLLED,
    MODEL_DECISION_TREE_UNBOUNDED,
    MODEL_DUMMY,
    MODEL_LOGISTIC,
    RANDOM_STATE,
)
from src.modeling.preprocessing import criar_preprocessador


def criar_pipeline_modelo(
    nome_modelo: str,
    colunas_numericas: list[str],
    colunas_categoricas: list[str],
    random_state: int = RANDOM_STATE,
) -> Pipeline:
    """Cria uma nova instancia nao ajustada da Pipeline para o modelo solicitado.

    Criar uma nova instancia a cada experimento ou divisao evita compartilhamento
    de estado mutavel entre folds e cenarios, preservando a separacao treino/validacao.
    """
    preprocessador = criar_preprocessador(
        colunas_numericas=colunas_numericas,
        colunas_categoricas=colunas_categoricas,
    )

    if nome_modelo == MODEL_DUMMY:
        estimador: Any = DummyClassifier(strategy="most_frequent")
    elif nome_modelo == MODEL_LOGISTIC:
        estimador = LogisticRegression(max_iter=LOGISTIC_MAX_ITER)
    elif nome_modelo == MODEL_DECISION_TREE_UNBOUNDED:
        estimador = DecisionTreeClassifier(random_state=random_state)
    elif nome_modelo == MODEL_DECISION_TREE_CONTROLLED:
        estimador = DecisionTreeClassifier(
            max_depth=DECISION_TREE_CONTROLLED_MAX_DEPTH,
            random_state=random_state,
        )
    else:
        raise ValueError(f"Modelo nao suportado: '{nome_modelo}'.")

    return Pipeline(
        [
            ("preprocessador", preprocessador),
            ("modelo", estimador),
        ]
    )


def criar_catalogo_modelos(
    colunas_numericas: list[str],
    colunas_categoricas: list[str],
    random_state: int = RANDOM_STATE,
) -> dict[str, Pipeline]:
    """Retorna os 4 modelos previstos na metodologia na ordem padrao de avaliacao."""
    nomes = [
        MODEL_DUMMY,
        MODEL_LOGISTIC,
        MODEL_DECISION_TREE_UNBOUNDED,
        MODEL_DECISION_TREE_CONTROLLED,
    ]
    return {
        nome: criar_pipeline_modelo(
            nome_modelo=nome,
            colunas_numericas=colunas_numericas,
            colunas_categoricas=colunas_categoricas,
            random_state=random_state,
        )
        for nome in nomes
    }


def treinar_pipeline(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    """Ajusta uma Pipeline exclusivamente sobre o conjunto de treinamento."""
    if len(X_train) == 0 or len(X_train) != len(y_train):
        raise ValueError("X_train e y_train devem ser nao vazios e ter o mesmo numero de linhas.")
    if list(X_train.index) != list(y_train.index):
        raise ValueError("Indices de X_train e y_train nao estao alinhados.")
    return pipeline.fit(X_train, y_train)


def gerar_previsoes(
    pipeline_treinada: Pipeline,
    X_eval: pd.DataFrame,
) -> pd.Series:
    """Gera previsoes de classe preservando exatamente o indice de X_eval."""
    if len(X_eval) == 0:
        raise ValueError("X_eval nao pode ser vazio para geracao de previsoes.")
    previsoes = pipeline_treinada.predict(X_eval)
    return pd.Series(list(previsoes), index=X_eval.index, name="y_pred")
