"""Camada de diagnostico descritivo de erros de classificacao (TP, TN, FP, FN)."""

from typing import Any

import pandas as pd

from src.config import NEGATIVE_CLASS, POSITIVE_CLASS
from src.data.validation import normalizar_serie_texto


def classificar_faixa_etaria(idade: Any) -> str:
    """Agrupa a idade em faixas interpretaveis via regras explicitas (sem pd.cut)."""
    if pd.isna(idade):
        return "Ausente"
    valor = float(idade)
    if valor < 17:
        return "<17"
    if valor <= 24:
        return "17-24"
    if valor <= 34:
        return "25-34"
    if valor <= 44:
        return "35-44"
    if valor <= 54:
        return "45-54"
    if valor <= 64:
        return "55-64"
    return "65+"


def classificar_faixa_horas(horas: Any) -> str:
    """Agrupa horas trabalhadas por semana em faixas descritivas via regras explicitas."""
    if pd.isna(horas):
        return "Ausente"
    valor = float(horas)
    if valor < 35:
        return "<35h (parcial)"
    if valor <= 40:
        return "35-40h (padrao)"
    if valor <= 50:
        return "41-50h (estendida)"
    return ">50h (intensa)"


def classificar_tipo_erro(
    y_true: pd.Series,
    y_pred: pd.Series,
    pos_label: str = POSITIVE_CLASS,
    neg_label: str = NEGATIVE_CLASS,
) -> pd.Series:
    """Rotula cada registro de forma mutuamente exclusiva como TP, TN, FP ou FN."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true e y_pred devem possuir o mesmo numero de linhas.")
    if list(y_true.index) != list(y_pred.index):
        raise ValueError("Indices de y_true e y_pred devem estar alinhados.")

    rotulos: list[str] = []
    for real, previsto in zip(y_true.tolist(), y_pred.tolist()):
        if real == pos_label and previsto == pos_label:
            rotulos.append("TP")
        elif real == neg_label and previsto == neg_label:
            rotulos.append("TN")
        elif real == neg_label and previsto == pos_label:
            rotulos.append("FP")
        elif real == pos_label and previsto == neg_label:
            rotulos.append("FN")
        else:
            raise ValueError(
                f"Combinacao inesperada de classes: real='{real}', previsto='{previsto}'."
            )

    return pd.Series(rotulos, index=y_true.index, name="tipo_resultado")


def _agrupar_erros_por_dimensao(
    df_analise: pd.DataFrame,
    coluna_dimensao: str,
    nome_dimensao: str,
) -> list[dict[str, Any]]:
    """Agrega TP, TN, FP, FN e taxas de erro para cada categoria de uma dimensao."""
    linhas: list[dict[str, Any]] = []
    serie_dim = normalizar_serie_texto(df_analise[coluna_dimensao], "Ausente")
    trabalho = pd.DataFrame(
        {
            "categoria": serie_dim,
            "tipo_resultado": df_analise["tipo_resultado"],
        }
    )

    for categoria, grupo in trabalho.groupby("categoria"):
        total = int(len(grupo))
        tp = int((grupo["tipo_resultado"] == "TP").sum())
        tn = int((grupo["tipo_resultado"] == "TN").sum())
        fp = int((grupo["tipo_resultado"] == "FP").sum())
        fn = int((grupo["tipo_resultado"] == "FN").sum())

        positivos_reais = tp + fn
        negativos_reais = tn + fp
        positivos_previstos = tp + fp
        erros_totais = fp + fn

        taxa_erro = float(erros_totais / total) if total > 0 else None
        taxa_fp = float(fp / negativos_reais) if negativos_reais > 0 else None
        taxa_fn = float(fn / positivos_reais) if positivos_reais > 0 else None
        recall_seg = float(tp / positivos_reais) if positivos_reais > 0 else None
        precision_seg = float(tp / positivos_previstos) if positivos_previstos > 0 else None
        taxa_real_pos = float(positivos_reais / total) if total > 0 else None

        linhas.append(
            {
                "dimensao": nome_dimensao,
                "categoria": str(categoria),
                "total": total,
                "positivos_reais": positivos_reais,
                "negativos_reais": negativos_reais,
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "erros_totais": erros_totais,
                "taxa_erro": taxa_erro,
                "taxa_fp": taxa_fp,
                "taxa_fn": taxa_fn,
                "recall_segmento": recall_seg,
                "precision_segmento": precision_seg,
                "taxa_real_positiva": taxa_real_pos,
            }
        )

    linhas.sort(key=lambda item: (-item["erros_totais"], -item["total"], item["categoria"]))
    return linhas


def executar_analise_erros(
    X_eval: pd.DataFrame,
    y_true: pd.Series,
    y_pred: pd.Series,
    pos_label: str = POSITIVE_CLASS,
    neg_label: str = NEGATIVE_CLASS,
) -> dict[str, Any]:
    """Executa diagnostico descritivo de erros sem alterar modelo ou configuracoes."""
    if len(X_eval) != len(y_true) or len(X_eval) != len(y_pred):
        raise ValueError("X_eval, y_true e y_pred devem possuir o mesmo numero de linhas.")
    if list(X_eval.index) != list(y_true.index) or list(X_eval.index) != list(y_pred.index):
        raise ValueError("Indices de X_eval, y_true e y_pred devem estar alinhados.")

    tipos = classificar_tipo_erro(
        y_true=y_true,
        y_pred=y_pred,
        pos_label=pos_label,
        neg_label=neg_label,
    )

    df_analise = X_eval.copy()
    df_analise["y_true"] = y_true.astype(str)
    df_analise["y_pred"] = y_pred.astype(str)
    df_analise["tipo_resultado"] = tipos

    if "age" in df_analise.columns:
        df_analise["faixa_etaria"] = [
            classificar_faixa_etaria(valor) for valor in df_analise["age"].tolist()
        ]
    if "hours-per-week" in df_analise.columns:
        df_analise["faixa_horas"] = [
            classificar_faixa_horas(valor)
            for valor in df_analise["hours-per-week"].tolist()
        ]

    dimensoes_planejadas = [
        ("faixa_etaria", "faixa_etaria"),
        ("education", "education"),
        ("faixa_horas", "faixa_horas"),
        ("occupation", "occupation"),
        ("workclass", "workclass"),
        ("marital-status", "marital-status"),
        ("relationship", "relationship"),
    ]

    todas_linhas: list[dict[str, Any]] = []
    dimensoes_analisadas: list[str] = []
    for coluna, nome_dim in dimensoes_planejadas:
        if coluna in df_analise.columns:
            dimensoes_analisadas.append(nome_dim)
            todas_linhas.extend(
                _agrupar_erros_por_dimensao(
                    df_analise=df_analise,
                    coluna_dimensao=coluna,
                    nome_dimensao=nome_dim,
                )
            )

    tabela_erros = pd.DataFrame(todas_linhas)

    total = int(len(df_analise))
    tp = int((tipos == "TP").sum())
    tn = int((tipos == "TN").sum())
    fp = int((tipos == "FP").sum())
    fn = int((tipos == "FN").sum())
    erros_totais = fp + fn
    positivos_reais = tp + fn
    negativos_reais = tn + fp

    resumo_global = {
        "total_registros": total,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "acertos_totais": tp + tn,
        "erros_totais": erros_totais,
        "taxa_erro_global": float(erros_totais / total) if total > 0 else 0.0,
        "participacao_fp_nos_erros": float(fp / erros_totais) if erros_totais > 0 else 0.0,
        "participacao_fn_nos_erros": float(fn / erros_totais) if erros_totais > 0 else 0.0,
        "taxa_fp_sobre_negativos": float(fp / negativos_reais) if negativos_reais > 0 else None,
        "taxa_fn_sobre_positivos": float(fn / positivos_reais) if positivos_reais > 0 else None,
    }

    return {
        "resumo_global": resumo_global,
        "dimensoes_analisadas": dimensoes_analisadas,
        "tabela_erros": tabela_erros,
        "registros_classificados": df_analise[["y_true", "y_pred", "tipo_resultado"]].copy(),
    }
