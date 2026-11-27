"""Experimento controlado comparando Cenario A (com sex e race) vs Cenario B (sem sex e race)."""

from typing import Any

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.analysis.fairness import executar_auditoria_fairness
from src.config import (
    CV_SPLITS,
    NEGATIVE_CLASS,
    POSITIVE_CLASS,
    RANDOM_STATE,
    SENSITIVE_COLUMNS,
)
from src.modeling.evaluation import calcular_metricas_classificacao
from src.modeling.training import criar_pipeline_modelo, gerar_previsoes, treinar_pipeline


def _gerar_previsoes_oof_cenario(
    X_train_modelo: pd.DataFrame,
    y_train: pd.Series,
    modelo_nome: str,
    colunas_numericas: list[str],
    colunas_categoricas: list[str],
    cv_splits: int = CV_SPLITS,
    random_state: int = RANDOM_STATE,
) -> pd.Series:
    """Gera previsoes fora da amostra (out-of-fold) nas mesmas divisoes estratificadas do treino."""
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=False)
    partes_pred: list[pd.Series] = []

    for indices_treino_pos, indices_val_pos in skf.split(X_train_modelo, y_train):
        X_fold_train = X_train_modelo.iloc[indices_treino_pos]
        y_fold_train = y_train.iloc[indices_treino_pos]
        X_fold_val = X_train_modelo.iloc[indices_val_pos]

        pipeline_fold = criar_pipeline_modelo(
            nome_modelo=modelo_nome,
            colunas_numericas=colunas_numericas,
            colunas_categoricas=colunas_categoricas,
            random_state=random_state,
        )
        pipeline_fold = treinar_pipeline(pipeline_fold, X_fold_train, y_fold_train)
        pred_fold = gerar_previsoes(pipeline_fold, X_fold_val)
        partes_pred.append(pred_fold)

    y_pred_oof = pd.concat(partes_pred).loc[y_train.index]
    if list(y_pred_oof.index) != list(y_train.index):
        raise RuntimeError("Desalinhamento de indices ao recompor previsoes out-of-fold.")
    return y_pred_oof.rename("y_pred_oof")


def _subtrair_opcional(valor_b: float | None, valor_a: float | None) -> float | None:
    """Calcula delta (Cenario B - Cenario A) preservando None quando indefinido."""
    if valor_a is None or valor_b is None:
        return None
    return float(valor_b - valor_a)


def _montar_linhas_comparacao(
    escopo: str,
    metricas_a: dict[str, Any],
    metricas_b: dict[str, Any],
    fairness_a: dict[str, Any],
    fairness_b: dict[str, Any],
    atributos_sensiveis: list[str],
) -> list[dict[str, Any]]:
    """Constroi linhas tabulares comparando Cenario A vs Cenario B em um escopo."""
    linhas: list[dict[str, Any]] = []

    for metrica in ["accuracy", "recall", "precision", "f1", "fpr", "selection_rate"]:
        val_a = metricas_a[metrica]
        val_b = metricas_b[metrica]
        linhas.append(
            {
                "escopo": escopo,
                "categoria_metrica": "desempenho_geral",
                "atributo_sensivel": "global",
                "metrica": metrica,
                "cenario_a_com_sensiveis": val_a,
                "cenario_b_sem_sensiveis": val_b,
                "delta_b_menos_a": _subtrair_opcional(val_b, val_a),
            }
        )

    for attr in atributos_sensiveis:
        ra = fairness_a["por_atributo"][attr]["resumo"]
        rb = fairness_b["por_atributo"][attr]["resumo"]
        for chave in [
            "equal_opportunity_difference",
            "equalized_odds_difference",
            "recall_diff",
            "fpr_diff",
            "selection_rate_diff",
            "recall_ratio",
            "fpr_ratio",
            "selection_rate_ratio",
        ]:
            val_a = ra[chave]
            val_b = rb[chave]
            linhas.append(
                {
                    "escopo": escopo,
                    "categoria_metrica": "fairness_disparidade",
                    "atributo_sensivel": attr,
                    "metrica": chave,
                    "cenario_a_com_sensiveis": val_a,
                    "cenario_b_sem_sensiveis": val_b,
                    "delta_b_menos_a": _subtrair_opcional(val_b, val_a),
                }
            )

    return linhas


def executar_comparacao_atributos_sensiveis(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    modelo_nome: str,
    colunas_numericas: list[str],
    colunas_categoricas: list[str],
    atributos_sensiveis: list[str] = SENSITIVE_COLUMNS,
    cv_splits: int = CV_SPLITS,
    pos_label: str = POSITIVE_CLASS,
    neg_label: str = NEGATIVE_CLASS,
    random_state: int = RANDOM_STATE,
) -> dict[str, Any]:
    """Compara Cenario A (com sex e race) e Cenario B (sem sex e race) sob protocolo controlado."""
    for attr in atributos_sensiveis:
        if attr not in X_train.columns or attr not in X_test.columns:
            raise ValueError(f"Atributo sensivel '{attr}' ausente para comparacao de cenarios.")

    sensiveis_train = X_train[atributos_sensiveis].copy()
    sensiveis_test = X_test[atributos_sensiveis].copy()

    X_train_a = X_train.copy()
    X_test_a = X_test.copy()
    cat_a = list(colunas_categoricas)

    X_train_b = X_train.drop(columns=atributos_sensiveis).copy()
    X_test_b = X_test.drop(columns=atributos_sensiveis).copy()
    cat_b = [col for col in colunas_categoricas if col not in atributos_sensiveis]
    num_b = [col for col in colunas_numericas if col not in atributos_sensiveis]

    y_pred_oof_a = _gerar_previsoes_oof_cenario(
        X_train_modelo=X_train_a,
        y_train=y_train,
        modelo_nome=modelo_nome,
        colunas_numericas=list(colunas_numericas),
        colunas_categoricas=cat_a,
        cv_splits=cv_splits,
        random_state=random_state,
    )
    y_pred_oof_b = _gerar_previsoes_oof_cenario(
        X_train_modelo=X_train_b,
        y_train=y_train,
        modelo_nome=modelo_nome,
        colunas_numericas=num_b,
        colunas_categoricas=cat_b,
        cv_splits=cv_splits,
        random_state=random_state,
    )

    metricas_oof_a = calcular_metricas_classificacao(y_train, y_pred_oof_a, pos_label, neg_label)
    metricas_oof_b = calcular_metricas_classificacao(y_train, y_pred_oof_b, pos_label, neg_label)

    fairness_oof_a = executar_auditoria_fairness(
        sensiveis_train, y_train, y_pred_oof_a, atributos_sensiveis, pos_label, neg_label
    )
    fairness_oof_b = executar_auditoria_fairness(
        sensiveis_train, y_train, y_pred_oof_b, atributos_sensiveis, pos_label, neg_label
    )

    pipe_a = treinar_pipeline(
        criar_pipeline_modelo(modelo_nome, list(colunas_numericas), cat_a, random_state),
        X_train_a,
        y_train,
    )
    y_pred_test_a = gerar_previsoes(pipe_a, X_test_a)

    pipe_b = treinar_pipeline(
        criar_pipeline_modelo(modelo_nome, num_b, cat_b, random_state),
        X_train_b,
        y_train,
    )
    y_pred_test_b = gerar_previsoes(pipe_b, X_test_b)

    metricas_test_a = calcular_metricas_classificacao(y_test, y_pred_test_a, pos_label, neg_label)
    metricas_test_b = calcular_metricas_classificacao(y_test, y_pred_test_b, pos_label, neg_label)

    fairness_test_a = executar_auditoria_fairness(
        sensiveis_test, y_test, y_pred_test_a, atributos_sensiveis, pos_label, neg_label
    )
    fairness_test_b = executar_auditoria_fairness(
        sensiveis_test, y_test, y_pred_test_b, atributos_sensiveis, pos_label, neg_label
    )

    verificacao_alinhamento = {
        "train_b_alinhado_sensiveis": list(X_train_b.index) == list(sensiveis_train.index),
        "test_b_alinhado_sensiveis": list(X_test_b.index) == list(sensiveis_test.index),
        "oof_a_alinhado_y_train": list(y_pred_oof_a.index) == list(y_train.index),
        "oof_b_alinhado_y_train": list(y_pred_oof_b.index) == list(y_train.index),
        "test_a_alinhado_y_test": list(y_pred_test_a.index) == list(y_test.index),
        "test_b_alinhado_y_test": list(y_pred_test_b.index) == list(y_test.index),
    }

    linhas_comp = _montar_linhas_comparacao(
        "validacao_cv_treino_oof",
        metricas_oof_a,
        metricas_oof_b,
        fairness_oof_a,
        fairness_oof_b,
        atributos_sensiveis,
    ) + _montar_linhas_comparacao(
        "teste_historico_descritivo",
        metricas_test_a,
        metricas_test_b,
        fairness_test_a,
        fairness_test_b,
        atributos_sensiveis,
    )

    return {
        "protocolo": {
            "modelo": modelo_nome,
            "cv_splits": cv_splits,
            "atributos_removidos_cenario_b": list(atributos_sensiveis),
            "features_cenario_a": list(X_train_a.columns),
            "features_cenario_b": list(X_train_b.columns),
            "politica_experimental": (
                "Comparacao primaria executada via validacao cruzada out-of-fold no treino "
                "(StratifiedKFold 5-fold); avaliacao no teste historico reportada apenas "
                "como diagnostico descritivo de ambos os cenarios, sem uso para selecao."
            ),
        },
        "verificacao_alinhamento": verificacao_alinhamento,
        "cv_oof": {
            "cenario_a": {"metricas": metricas_oof_a, "fairness": fairness_oof_a},
            "cenario_b": {"metricas": metricas_oof_b, "fairness": fairness_oof_b},
        },
        "teste_historico": {
            "cenario_a": {"metricas": metricas_test_a, "fairness": fairness_test_a},
            "cenario_b": {"metricas": metricas_test_b, "fairness": fairness_test_b},
        },
        "tabela_comparativa": pd.DataFrame(linhas_comp),
    }
