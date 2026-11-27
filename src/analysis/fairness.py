"""Auditoria de fairness por grupo com MetricFrame, Fairlearn e protecao de denominadores."""

from functools import partial
from typing import Any

import pandas as pd
from fairlearn.metrics import (
    MetricFrame,
    count,
    equal_opportunity_difference,
    equalized_odds_difference,
    false_positive_rate,
    selection_rate,
)
from sklearn.metrics import accuracy_score, recall_score

from src.config import (
    NEGATIVE_CLASS,
    POSITIVE_CLASS,
    SENSITIVE_COLUMNS,
    SMALL_GROUP_MIN_POSITIVES,
    SMALL_GROUP_MIN_SAMPLES,
)
from src.data.validation import normalizar_serie_texto


def recall_seguro(
    y_true: Any,
    y_pred: Any,
    pos_label: str = POSITIVE_CLASS,
) -> float:
    """Calcula Recall (TPR) retornando NaN quando nao ha exemplos positivos reais no grupo."""
    serie_real = pd.Series(y_true)
    positivos_reais = int((serie_real == pos_label).sum())
    if positivos_reais == 0:
        return float("nan")
    return float(recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0))


def fpr_seguro(
    y_true: Any,
    y_pred: Any,
    pos_label: str = POSITIVE_CLASS,
) -> float:
    """Calcula False Positive Rate retornando NaN quando nao ha exemplos negativos reais."""
    serie_real = pd.Series(y_true)
    negativos_reais = int((serie_real != pos_label).sum())
    if negativos_reais == 0:
        return float("nan")
    return float(false_positive_rate(y_true, y_pred, pos_label=pos_label))


def _valor_ou_none(valor: Any) -> float | None:
    """Converte float valido ou retorna None para valores NaN/ausentes."""
    if pd.isna(valor):
        return None
    return float(valor)


def auditar_fairness_atributo(
    y_true: pd.Series,
    y_pred: pd.Series,
    sensitive_feature: pd.Series,
    nome_atributo: str,
    pos_label: str = POSITIVE_CLASS,
    neg_label: str = NEGATIVE_CLASS,
    min_amostra_alerta: int = SMALL_GROUP_MIN_SAMPLES,
    min_positivos_alerta: int = SMALL_GROUP_MIN_POSITIVES,
) -> dict[str, Any]:
    """Audita metricas por grupo, diferencas, razoes, EOD e EOdds para um atributo sensivel.

    Preserva rigorosamente o alinhamento de indices entre y_true, y_pred e
    sensitive_feature e impede que metricas indefinidas (denominador zero)
    sejam apresentadas silenciosamente como zero.
    """
    if len(y_true) == 0:
        raise ValueError("Nao e possivel executar auditoria de fairness com amostra vazia.")
    if len(y_true) != len(y_pred) or len(y_true) != len(sensitive_feature):
        raise ValueError("y_true, y_pred e sensitive_feature devem ter o mesmo comprimento.")
    if (
        list(y_true.index) != list(y_pred.index)
        or list(y_true.index) != list(sensitive_feature.index)
    ):
        raise ValueError(
            f"Desalinhamento de indices detectado na auditoria de fairness de '{nome_atributo}'."
        )

    serie_sens = normalizar_serie_texto(sensitive_feature, "Ausente")
    y_true_str = y_true.astype(str)
    y_pred_str = y_pred.astype(str)

    metricas_mf = {
        "accuracy": accuracy_score,
        "recall": partial(recall_seguro, pos_label=pos_label),
        "fpr": partial(fpr_seguro, pos_label=pos_label),
        "selection_rate": partial(selection_rate, pos_label=pos_label),
        "count": count,
    }

    quadro = MetricFrame(
        metrics=metricas_mf,
        y_true=y_true_str,
        y_pred=y_pred_str,
        sensitive_features=serie_sens,
    )

    by_group = quadro.by_group.copy()
    total_geral = int(len(y_true_str))

    trabalho = pd.DataFrame(
        {
            "grupo": serie_sens,
            "y_true": y_true_str,
            "y_pred": y_pred_str,
        }
    )

    linhas_grupos: list[dict[str, Any]] = []
    grupos_recall_indef: list[str] = []
    grupos_fpr_indef: list[str] = []
    grupos_amostra_pequena: list[str] = []

    for grupo in by_group.index.tolist():
        sub = trabalho.loc[trabalho["grupo"] == grupo]
        n_grupo = int(len(sub))
        tp = int(((sub["y_true"] == pos_label) & (sub["y_pred"] == pos_label)).sum())
        tn = int(((sub["y_true"] == neg_label) & (sub["y_pred"] == neg_label)).sum())
        fp = int(((sub["y_true"] == neg_label) & (sub["y_pred"] == pos_label)).sum())
        fn = int(((sub["y_true"] == pos_label) & (sub["y_pred"] == neg_label)).sum())

        positivos_reais = tp + fn
        negativos_reais = tn + fp
        positivos_previstos = tp + fp

        recall_indef = positivos_reais == 0
        fpr_indef = negativos_reais == 0
        amostra_pequena = (
            n_grupo < min_amostra_alerta or positivos_reais < min_positivos_alerta
        )

        if recall_indef:
            grupos_recall_indef.append(str(grupo))
        if fpr_indef:
            grupos_fpr_indef.append(str(grupo))
        if amostra_pequena:
            grupos_amostra_pequena.append(str(grupo))

        observacoes: list[str] = []
        if recall_indef:
            observacoes.append("Recall indefinido (0 positivos reais)")
        if fpr_indef:
            observacoes.append("FPR indefinido (0 negativos reais)")
        if amostra_pequena:
            observacoes.append(
                f"Amostra pequena (n={n_grupo}, positivos_reais={positivos_reais})"
            )

        linhas_grupos.append(
            {
                "atributo": nome_atributo,
                "grupo": str(grupo),
                "count": n_grupo,
                "proporcao_amostra": float(n_grupo / total_geral) if total_geral > 0 else 0.0,
                "positivos_reais": positivos_reais,
                "negativos_reais": negativos_reais,
                "positivos_previstos": positivos_previstos,
                "taxa_real_positiva": float(positivos_reais / n_grupo) if n_grupo > 0 else 0.0,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "accuracy": _valor_ou_none(by_group.loc[grupo, "accuracy"]),
                "recall": _valor_ou_none(by_group.loc[grupo, "recall"]),
                "fpr": _valor_ou_none(by_group.loc[grupo, "fpr"]),
                "selection_rate": _valor_ou_none(by_group.loc[grupo, "selection_rate"]),
                "recall_indefinido": recall_indef,
                "fpr_indefinido": fpr_indef,
                "amostra_pequena": amostra_pequena,
                "observacao": "; ".join(observacoes) if observacoes else "Denominadores validos",
            }
        )

    tabela_por_grupo = pd.DataFrame(linhas_grupos)

    metricas_desempenho = ["accuracy", "recall", "fpr", "selection_rate"]
    diff_series = quadro.difference()[metricas_desempenho]
    ratio_series = quadro.ratio()[metricas_desempenho]

    diferencas: dict[str, float | None] = {}
    razoes: dict[str, float | None] = {}
    for metrica in metricas_desempenho:
        valores_validos = [float(v) for v in tabela_por_grupo[metrica].dropna().tolist()]
        if len(valores_validos) < 2:
            diferencas[metrica] = None
            razoes[metrica] = None
        else:
            diferencas[metrica] = _valor_ou_none(diff_series[metrica])
            if diferencas[metrica] is None:
                diferencas[metrica] = float(max(valores_validos) - min(valores_validos))
            max_val = max(valores_validos)
            min_val = min(valores_validos)
            if max_val == 0.0:
                razoes[metrica] = None
            else:
                ratio_mf = _valor_ou_none(ratio_series[metrica])
                razoes[metrica] = ratio_mf if ratio_mf is not None else float(min_val / max_val)

    y_true_bin = (y_true_str == pos_label).astype(int)
    y_pred_bin = (y_pred_str == pos_label).astype(int)

    possui_denom_zero = bool(grupos_recall_indef or grupos_fpr_indef)
    grupos_excluidos_eodds = sorted(set(grupos_recall_indef).union(set(grupos_fpr_indef)))
    recalls_validos = [float(v) for v in tabela_por_grupo["recall"].dropna().tolist()]
    fprs_validos = [float(v) for v in tabela_por_grupo["fpr"].dropna().tolist()]
    qtd_grupos = int(len(tabela_por_grupo))
    notas: list[str] = []

    if not grupos_recall_indef and qtd_grupos >= 2:
        eod: float | None = float(
            equal_opportunity_difference(
                y_true_bin,
                y_pred_bin,
                sensitive_features=serie_sens,
            )
        )
        equal_opportunity_escopo = "integral_todos_os_grupos"
    elif len(recalls_validos) >= 2:
        eod = float(max(recalls_validos) - min(recalls_validos))
        equal_opportunity_escopo = "parcial_exclui_grupos_denominador_zero"
        notas.append(
            f"EOD parcial (comparacao incompleta): calculado apenas sobre "
            f"{len(recalls_validos)}/{qtd_grupos} grupos com positivos_reais > 0 "
            f"(excluidos por denominador zero: {grupos_recall_indef})."
        )
    else:
        eod = None
        equal_opportunity_escopo = "indefinido_menos_de_dois_grupos_validos"
        notas.append(
            "EOD indefinido: menos de 2 grupos possuem positivos_reais > 0 "
            f"(grupos sem positivos reais: {grupos_recall_indef})."
        )

    if not possui_denom_zero and qtd_grupos >= 2:
        eodds: float | None = float(
            equalized_odds_difference(
                y_true_bin,
                y_pred_bin,
                sensitive_features=serie_sens,
            )
        )
        equalized_odds_escopo = "integral_todos_os_grupos"
    elif diferencas["recall"] is not None and diferencas["fpr"] is not None:
        eodds = float(max(diferencas["recall"], diferencas["fpr"]))
        equalized_odds_escopo = "parcial_exclui_grupos_denominador_zero"
        notas.append(
            "EOdds parcial (comparacao incompleta, nao integral): "
            f"TPR valido em {len(recalls_validos)}/{qtd_grupos} grupos e "
            f"FPR valido em {len(fprs_validos)}/{qtd_grupos} grupos; "
            f"grupos excluidos em ao menos um eixo por denominador zero: {grupos_excluidos_eodds}."
        )
    else:
        eodds = None
        if diferencas["recall"] is None and diferencas["fpr"] is None:
            equalized_odds_escopo = "indefinido_ambos_eixos_indisponiveis"
            notas.append(
                "EOdds indefinido: ambos os eixos (TPR e FPR) possuem menos de 2 grupos "
                "com denominador valido."
            )
        else:
            eixo_ausente = "TPR (Recall)" if diferencas["recall"] is None else "FPR"
            equalized_odds_escopo = "indefinido_eixo_tpr_ou_fpr_ausente"
            notas.append(
                f"EOdds indefinido (avaliacao incompleta): eixo {eixo_ausente} possui menos de 2 "
                "grupos com denominador valido; nao reportado como avaliacao integral de Equalized Odds."
            )

    if grupos_fpr_indef:
        notas.append(
            "FPR indefinido nos grupos com negativos_reais == 0 "
            f"({grupos_fpr_indef}); nao substituido por zero."
        )

    avaliacao_fairness_parcial = equal_opportunity_escopo.startswith(
        "parcial"
    ) or equalized_odds_escopo.startswith("parcial")

    overall = quadro.overall
    metricas_globais = {
        "accuracy": _valor_ou_none(overall["accuracy"]),
        "recall": _valor_ou_none(overall["recall"]),
        "fpr": _valor_ou_none(overall["fpr"]),
        "selection_rate": _valor_ou_none(overall["selection_rate"]),
    }

    medias_simples: dict[str, float | None] = {}
    for metrica in metricas_desempenho:
        vals = [float(v) for v in tabela_por_grupo[metrica].dropna().tolist()]
        medias_simples[metrica] = float(sum(vals) / len(vals)) if vals else None

    resumo = {
        "atributo": nome_atributo,
        "total_registros": total_geral,
        "qtd_grupos": qtd_grupos,
        "equal_opportunity_difference": eod,
        "equal_opportunity_escopo": equal_opportunity_escopo,
        "equalized_odds_difference": eodds,
        "equalized_odds_escopo": equalized_odds_escopo,
        "avaliacao_fairness_parcial": avaliacao_fairness_parcial,
        "accuracy_diff": diferencas["accuracy"],
        "recall_diff": diferencas["recall"],
        "fpr_diff": diferencas["fpr"],
        "selection_rate_diff": diferencas["selection_rate"],
        "accuracy_ratio": razoes["accuracy"],
        "recall_ratio": razoes["recall"],
        "fpr_ratio": razoes["fpr"],
        "selection_rate_ratio": razoes["selection_rate"],
        "possui_denominadores_zero": possui_denom_zero,
        "grupos_recall_indefinido": grupos_recall_indef,
        "grupos_fpr_indefinido": grupos_fpr_indef,
        "grupos_excluidos_equalized_odds": grupos_excluidos_eodds,
        "grupos_amostra_pequena": grupos_amostra_pequena,
        "nota_denominadores_zero": (
            " ".join(notas)
            if notas
            else "Todos os grupos possuem positivos e negativos reais > 0 (comparacao integral)."
        ),
        "metricas_globais": metricas_globais,
        "medias_simples_entre_grupos": medias_simples,
    }

    return {
        "atributo": nome_atributo,
        "tabela_por_grupo": tabela_por_grupo,
        "diferencas": diferencas,
        "razoes": razoes,
        "resumo": resumo,
    }


def executar_auditoria_fairness(
    X_eval: pd.DataFrame,
    y_true: pd.Series,
    y_pred: pd.Series,
    atributos_sensiveis: list[str] = SENSITIVE_COLUMNS,
    pos_label: str = POSITIVE_CLASS,
    neg_label: str = NEGATIVE_CLASS,
) -> dict[str, Any]:
    """Executa auditoria de fairness para todos os atributos sensiveis solicitados."""
    por_atributo: dict[str, dict[str, Any]] = {}
    linhas_resumo: list[dict[str, Any]] = []

    for atributo in atributos_sensiveis:
        if atributo not in X_eval.columns:
            raise ValueError(f"Atributo sensivel '{atributo}' ausente em X_eval.")
        resultado_attr = auditar_fairness_atributo(
            y_true=y_true,
            y_pred=y_pred,
            sensitive_feature=X_eval[atributo],
            nome_atributo=atributo,
            pos_label=pos_label,
            neg_label=neg_label,
        )
        por_atributo[atributo] = resultado_attr
        r = resultado_attr["resumo"]
        linhas_resumo.append(
            {
                "atributo": r["atributo"],
                "total_registros": r["total_registros"],
                "qtd_grupos": r["qtd_grupos"],
                "equal_opportunity_difference": r["equal_opportunity_difference"],
                "equal_opportunity_escopo": r["equal_opportunity_escopo"],
                "equalized_odds_difference": r["equalized_odds_difference"],
                "equalized_odds_escopo": r["equalized_odds_escopo"],
                "avaliacao_fairness_parcial": r["avaliacao_fairness_parcial"],
                "accuracy_diff": r["accuracy_diff"],
                "recall_diff": r["recall_diff"],
                "fpr_diff": r["fpr_diff"],
                "selection_rate_diff": r["selection_rate_diff"],
                "accuracy_ratio": r["accuracy_ratio"],
                "recall_ratio": r["recall_ratio"],
                "fpr_ratio": r["fpr_ratio"],
                "selection_rate_ratio": r["selection_rate_ratio"],
                "possui_denominadores_zero": r["possui_denominadores_zero"],
                "grupos_excluidos_equalized_odds": ", ".join(r["grupos_excluidos_equalized_odds"]),
                "qtd_grupos_amostra_pequena": len(r["grupos_amostra_pequena"]),
                "grupos_amostra_pequena": ", ".join(r["grupos_amostra_pequena"]),
            }
        )

    tabela_resumo = pd.DataFrame(linhas_resumo)
    return {
        "por_atributo": por_atributo,
        "tabela_resumo": tabela_resumo,
    }
