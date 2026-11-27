"""Validacao cruzada, selecao pre-registrada de modelo e avaliacao final."""

from typing import Any

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_validate

from src.config import (
    CV_SPLITS,
    MAX_OVERFITTING_GAP_ACCURACY,
    MODEL_DUMMY,
    NEGATIVE_CLASS,
    POSITIVE_CLASS,
    TARGET_CLASSES,
)
from src.modeling.training import criar_catalogo_modelos


def calcular_estatisticas_folds(valores: list[float]) -> dict[str, float]:
    """Calcula media, desvio-padrao populacional, minimo, maximo e amplitude.

    Implementado com aritmetica pura em Python (sem NumPy direto) para manter
    total transparencia didatica e compatibilidade com a V1 (`scores.std()`).
    """
    if not valores:
        raise ValueError("A lista de valores dos folds nao pode ser vazia.")
    n = len(valores)
    media = sum(valores) / n
    variancia = sum((v - media) ** 2 for v in valores) / n
    desvio_padrao = variancia ** 0.5
    minimo = min(valores)
    maximo = max(valores)
    return {
        "media": float(media),
        "desvio_padrao": float(desvio_padrao),
        "minimo": float(minimo),
        "maximo": float(maximo),
        "amplitude": float(maximo - minimo),
    }


def _criar_scorers_validacao(pos_label: str = POSITIVE_CLASS) -> dict[str, Any]:
    """Constroi dicionario de scorers para validacao cruzada.

    Usa zero_division=0 para permitir avaliar o DummyClassifier (que nunca preve
    a classe positiva) na mesma tabela comparativa sem UndefinedMetricWarning.
    """
    return {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, pos_label=pos_label, zero_division=0),
        "recall": make_scorer(recall_score, pos_label=pos_label, zero_division=0),
        "f1": make_scorer(f1_score, pos_label=pos_label, zero_division=0),
    }


def executar_validacao_cruzada_modelos(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    colunas_numericas: list[str],
    colunas_categoricas: list[str],
    cv_splits: int = CV_SPLITS,
    pos_label: str = POSITIVE_CLASS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Executa validacao cruzada estratificada (somente no treino) para os 4 modelos.

    Retorna:
        - tabela_resumo: 1 linha por modelo com medias de treino/validacao,
          gaps treino - validacao e dispersao entre folds.
        - tabela_folds: detalhamento de todas as metricas em cada fold.
    """
    catalogo = criar_catalogo_modelos(
        colunas_numericas=colunas_numericas,
        colunas_categoricas=colunas_categoricas,
    )
    scorers = _criar_scorers_validacao(pos_label=pos_label)

    linhas_resumo: list[dict[str, Any]] = []
    linhas_folds: list[dict[str, Any]] = []

    for nome_modelo, pipeline in catalogo.items():
        resultado = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv_splits,
            scoring=scorers,
            return_train_score=True,
        )

        metricas = ["accuracy", "precision", "recall", "f1"]
        linha_resumo: dict[str, Any] = {
            "modelo": nome_modelo,
            "eh_baseline": nome_modelo == MODEL_DUMMY,
            "cv_splits": cv_splits,
        }

        for metrica in metricas:
            valores_treino = [float(v) for v in resultado[f"train_{metrica}"]]
            valores_val = [float(v) for v in resultado[f"test_{metrica}"]]
            stats_treino = calcular_estatisticas_folds(valores_treino)
            stats_val = calcular_estatisticas_folds(valores_val)

            linha_resumo[f"train_{metrica}_mean"] = stats_treino["media"]
            linha_resumo[f"train_{metrica}_std"] = stats_treino["desvio_padrao"]
            linha_resumo[f"val_{metrica}_mean"] = stats_val["media"]
            linha_resumo[f"val_{metrica}_std"] = stats_val["desvio_padrao"]
            linha_resumo[f"val_{metrica}_min"] = stats_val["minimo"]
            linha_resumo[f"val_{metrica}_max"] = stats_val["maximo"]
            linha_resumo[f"val_{metrica}_amplitude"] = stats_val["amplitude"]
            linha_resumo[f"gap_train_val_{metrica}"] = (
                stats_treino["media"] - stats_val["media"]
            )

        linhas_resumo.append(linha_resumo)

        for indice_fold in range(cv_splits):
            linhas_folds.append(
                {
                    "modelo": nome_modelo,
                    "fold": indice_fold + 1,
                    "train_accuracy": float(resultado["train_accuracy"][indice_fold]),
                    "val_accuracy": float(resultado["test_accuracy"][indice_fold]),
                    "train_precision": float(resultado["train_precision"][indice_fold]),
                    "val_precision": float(resultado["test_precision"][indice_fold]),
                    "train_recall": float(resultado["train_recall"][indice_fold]),
                    "val_recall": float(resultado["test_recall"][indice_fold]),
                    "train_f1": float(resultado["train_f1"][indice_fold]),
                    "val_f1": float(resultado["test_f1"][indice_fold]),
                    "gap_accuracy": float(
                        resultado["train_accuracy"][indice_fold]
                        - resultado["test_accuracy"][indice_fold]
                    ),
                    "gap_f1": float(
                        resultado["train_f1"][indice_fold]
                        - resultado["test_f1"][indice_fold]
                    ),
                }
            )

    tabela_resumo = pd.DataFrame(linhas_resumo)
    tabela_folds = pd.DataFrame(linhas_folds)
    return tabela_resumo, tabela_folds


def selecionar_melhor_modelo(
    tabela_comparacao: pd.DataFrame,
    max_gap_overfitting: float = MAX_OVERFITTING_GAP_ACCURACY,
) -> tuple[str, pd.DataFrame, dict[str, Any]]:
    """Aplica o criterio pre-registrado de selecao usando exclusivamente metricas de CV.

    Regras definidas ANTES da avaliacao final no conjunto de teste:
    1. Exclui a baseline (DummyClassifier) da selecao final, usando-a como piso minimo.
    2. Exige superacao da baseline em val_f1_mean e val_recall_mean.
    3. Rejeita modelos com overfitting severo (gap_train_val_accuracy > max_gap_overfitting).
    4. Entre os elegiveis, escolhe o maior val_f1_mean (desempate: maior val_recall_mean,
       depois menor gap_train_val_accuracy).
    """
    df = tabela_comparacao.copy()

    linhas_baseline = df.loc[df["eh_baseline"]]
    if len(linhas_baseline) == 1:
        baseline_f1 = float(linhas_baseline["val_f1_mean"].iloc[0])
        baseline_recall = float(linhas_baseline["val_recall_mean"].iloc[0])
    else:
        baseline_f1 = 0.0
        baseline_recall = 0.0

    elegiveis: list[bool] = []
    motivos: list[str] = []

    for indice in df.index:
        eh_baseline = bool(df.loc[indice, "eh_baseline"])
        val_f1 = float(df.loc[indice, "val_f1_mean"])
        val_recall = float(df.loc[indice, "val_recall_mean"])
        gap_acc = float(df.loc[indice, "gap_train_val_accuracy"])

        if eh_baseline:
            elegiveis.append(False)
            motivos.append("Baseline de referencia minima (nao candidata a modelo final)")
        elif val_f1 <= baseline_f1 or val_recall <= baseline_recall:
            elegiveis.append(False)
            motivos.append("Nao superou a baseline em F1 e Recall na validacao cruzada")
        elif gap_acc > max_gap_overfitting:
            elegiveis.append(False)
            motivos.append(
                f"Rejeitado por overfitting severo (gap treino-validacao de Accuracy = {gap_acc:.4f} > {max_gap_overfitting:.2f})"
            )
        else:
            elegiveis.append(True)
            motivos.append("Elegivel pelos criterios de generalizacao e desempenho")

    df["elegivel_selecao"] = elegiveis
    df["diagnostico_selecao"] = motivos

    candidatos = df.loc[df["elegivel_selecao"]].copy()
    if len(candidatos) == 0:
        # Fallback explicito se nenhum modelo cumprir o limite estrito (ex.: base sintetica extrema)
        candidatos = df.loc[~df["eh_baseline"]].copy()
        if len(candidatos) == 0:
            raise ValueError("Nao ha modelos candidatos disponiveis para selecao.")

    candidatos = candidatos.sort_values(
        by=["val_f1_mean", "val_recall_mean", "gap_train_val_accuracy"],
        ascending=[False, False, True],
    )
    modelo_escolhido = str(candidatos["modelo"].iloc[0])

    df["selecionado"] = df["modelo"] == modelo_escolhido
    for indice in df.index:
        if bool(df.loc[indice, "selecionado"]):
            df.loc[indice, "diagnostico_selecao"] = (
                "Selecionado: maior F1 de validacao entre modelos sem overfitting severo"
            )

    resumo_selecao = {
        "modelo_selecionado": modelo_escolhido,
        "criterio_primario": "Maior val_f1_mean na validacao cruzada (classe positiva >50K)",
        "criterio_desempate": "Maior val_recall_mean, seguido por menor gap_train_val_accuracy",
        "limite_overfitting_accuracy": max_gap_overfitting,
        "baseline_val_f1": baseline_f1,
        "baseline_val_recall": baseline_recall,
        "conjunto_usado_na_selecao": "Apenas X_train/y_train via validacao cruzada 5-fold (X_test nao consultado)",
    }
    return modelo_escolhido, df, resumo_selecao


def calcular_metricas_classificacao(
    y_true: pd.Series,
    y_pred: pd.Series,
    pos_label: str = POSITIVE_CLASS,
    neg_label: str = NEGATIVE_CLASS,
) -> dict[str, Any]:
    """Calcula metricas de classificacao binaria, matriz de confusao e classification report.

    Sinaliza explicitamente casos de denominador zero para evitar interpretar
    metricas indefinidas como zero sem explicacao.
    """
    if len(y_true) == 0 or len(y_true) != len(y_pred):
        raise ValueError("y_true e y_pred devem ser nao vazios e ter o mesmo comprimento.")

    labels = [neg_label, pos_label]
    matriz = confusion_matrix(y_true, y_pred, labels=labels)
    tn = int(matriz[0][0])
    fp = int(matriz[0][1])
    fn = int(matriz[1][0])
    tp = int(matriz[1][1])
    total = tn + fp + fn + tp

    positivos_reais = tp + fn
    negativos_reais = tn + fp
    positivos_previstos = tp + fp
    negativos_previstos = tn + fn

    accuracy = float(accuracy_score(y_true, y_pred))

    recall_indefinido = positivos_reais == 0
    precision_indefinida = positivos_previstos == 0
    fpr_indefinido = negativos_reais == 0

    recall = (
        None
        if recall_indefinido
        else float(recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0))
    )
    precision = (
        None
        if precision_indefinida
        else float(precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0))
    )

    if recall is None or precision is None or (precision + recall) == 0.0:
        f1 = 0.0 if (positivos_reais > 0 and positivos_previstos > 0) else None
        f1_indefinido = f1 is None
    else:
        f1 = float(f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0))
        f1_indefinido = False

    fpr = None if fpr_indefinido else float(fp / negativos_reais)
    selection_rate = float(positivos_previstos / total) if total > 0 else None

    relatorio_dict = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    relatorio_texto = classification_report(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )

    df_matriz = pd.DataFrame(
        [
            {"classe_real": neg_label, "previsto_negativo": tn, "previsto_positivo": fp},
            {"classe_real": pos_label, "previsto_negativo": fn, "previsto_positivo": tp},
        ]
    )

    return {
        "total_registros": total,
        "labels_ordem": list(TARGET_CLASSES) if labels == list(TARGET_CLASSES) else labels,
        "positivos_reais": positivos_reais,
        "negativos_reais": negativos_reais,
        "positivos_previstos": positivos_previstos,
        "negativos_previstos": negativos_previstos,
        "matriz_confusao": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "matriz_confusao_df": df_matriz,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "selection_rate": selection_rate,
        "denominadores_zero": {
            "precision_indefinida": precision_indefinida,
            "recall_indefinido": recall_indefinido,
            "f1_indefinido": f1_indefinido,
            "fpr_indefinido": fpr_indefinido,
        },
        "classification_report_dict": relatorio_dict,
        "classification_report_text": relatorio_texto,
    }
