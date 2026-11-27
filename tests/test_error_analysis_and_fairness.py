"""Testes de analise de erros, auditoria de fairness, grupos pequenos e denominadores zero."""

import pandas as pd
import pytest

from src.analysis.error_analysis import (
    classificar_faixa_etaria,
    classificar_faixa_horas,
    classificar_tipo_erro,
    executar_analise_erros,
)
from src.analysis.fairness import auditar_fairness_atributo


def test_classificacao_tp_tn_fp_fn_e_reconciliacao_por_dimensao() -> None:
    """Confirma rotulacao exata de erros e soma integral em cada dimensao analisada."""
    X_eval = pd.DataFrame(
        {
            "age": [22, 35, 50, 68],
            "education": ["HS-grad", "Bachelors", "Masters", "HS-grad"],
            "hours-per-week": [30, 40, 50, 60],
            "occupation": ["Sales", None, "Exec-managerial", "Sales"],
            "workclass": ["Private", None, "Local-gov", "Private"],
            "marital-status": ["Never-married", "Married-civ-spouse", "Married-civ-spouse", "Divorced"],
            "relationship": ["Own-child", "Husband", "Wife", "Not-in-family"],
        },
        index=[10, 20, 30, 40],
    )
    y_true = pd.Series(["<=50K", "<=50K", ">50K", ">50K"], index=[10, 20, 30, 40])
    y_pred = pd.Series(["<=50K", ">50K", ">50K", "<=50K"], index=[10, 20, 30, 40])

    tipos = classificar_tipo_erro(y_true, y_pred)
    assert tipos.tolist() == ["TN", "FP", "TP", "FN"]

    resultado = executar_analise_erros(X_eval, y_true, y_pred)
    assert resultado["resumo_global"]["tn"] == 1
    assert resultado["resumo_global"]["fp"] == 1
    assert resultado["resumo_global"]["tp"] == 1
    assert resultado["resumo_global"]["fn"] == 1

    tabela = resultado["tabela_erros"]
    for _dim, grupo in tabela.groupby("dimensao"):
        assert int(grupo["total"].sum()) == 4
        assert int(grupo["tp"].sum()) == 1
        assert int(grupo["tn"].sum()) == 1
        assert int(grupo["fp"].sum()) == 1
        assert int(grupo["fn"].sum()) == 1

    assert classificar_faixa_etaria(22) == "17-24"
    assert classificar_faixa_etaria(68) == "65+"
    assert classificar_faixa_horas(30) == "<35h (parcial)"
    assert classificar_faixa_horas(60) == ">50h (intensa)"


def test_fairness_cenario_conhecido_verificacao_manual_independente(
    cenario_fairness_conhecido: dict[str, pd.Series],
) -> None:
    """Verifica metricas por grupo, EOD, EOdds, diferencas e razoes contra calculo manual."""
    aud = auditar_fairness_atributo(
        y_true=cenario_fairness_conhecido["y_true"],
        y_pred=cenario_fairness_conhecido["y_pred"],
        sensitive_feature=cenario_fairness_conhecido["grupo"],
        nome_atributo="grupo_teste",
    )

    df_g = aud["tabela_por_grupo"].set_index("grupo")
    assert df_g.loc["GrupoA", "recall"] == pytest.approx(1.0, abs=1e-12)
    assert df_g.loc["GrupoA", "fpr"] == pytest.approx(0.25, abs=1e-12)
    assert df_g.loc["GrupoA", "accuracy"] == pytest.approx(0.875, abs=1e-12)
    assert df_g.loc["GrupoA", "selection_rate"] == pytest.approx(0.625, abs=1e-12)

    assert df_g.loc["GrupoB", "recall"] == pytest.approx(0.5, abs=1e-12)
    assert df_g.loc["GrupoB", "fpr"] == pytest.approx(0.0, abs=1e-12)
    assert df_g.loc["GrupoB", "accuracy"] == pytest.approx(0.75, abs=1e-12)
    assert df_g.loc["GrupoB", "selection_rate"] == pytest.approx(0.25, abs=1e-12)

    res = aud["resumo"]
    assert res["equal_opportunity_difference"] == pytest.approx(0.5, abs=1e-12)
    assert res["equal_opportunity_escopo"] == "integral_todos_os_grupos"
    assert res["equalized_odds_difference"] == pytest.approx(0.5, abs=1e-12)
    assert res["equalized_odds_escopo"] == "integral_todos_os_grupos"
    assert res["avaliacao_fairness_parcial"] is False
    assert res["grupos_excluidos_equalized_odds"] == []
    assert res["fpr_diff"] == pytest.approx(0.25, abs=1e-12)
    assert res["selection_rate_diff"] == pytest.approx(0.375, abs=1e-12)
    assert res["recall_ratio"] == pytest.approx(0.5, abs=1e-12)
    assert res["selection_rate_ratio"] == pytest.approx(0.4, abs=1e-12)
    # Ambos os grupos tem n=8 (<100), logo devem receber alerta de amostra pequena
    assert set(res["grupos_amostra_pequena"]) == {"GrupoA", "GrupoB"}


def test_fairness_denominador_zero_nao_mascara_como_zero() -> None:
    """Grupo sem positivos reais deve gerar NaN e escopo explicitamente PARCIAL (nunca integral)."""
    idx = list(range(10))
    y_true = pd.Series(
        [">50K", ">50K", "<=50K", "<=50K", ">50K", "<=50K", "<=50K", "<=50K", "<=50K", "<=50K"],
        index=idx,
    )
    y_pred = pd.Series(
        [">50K", ">50K", "<=50K", "<=50K", "<=50K", "<=50K", "<=50K", "<=50K", "<=50K", "<=50K"],
        index=idx,
    )
    grupos = pd.Series(
        ["G1", "G1", "G1", "G1", "G2", "G2", "G2", "G2", "G3_sem_pos", "G3_sem_pos"],
        index=idx,
    )

    aud = auditar_fairness_atributo(y_true, y_pred, grupos, nome_atributo="race")
    df_g = aud["tabela_por_grupo"].set_index("grupo")
    res = aud["resumo"]

    assert bool(df_g.loc["G3_sem_pos", "recall_indefinido"]) is True
    assert pd.isna(df_g.loc["G3_sem_pos", "recall"])
    assert res["possui_denominadores_zero"] is True
    assert "G3_sem_pos" in res["grupos_recall_indefinido"]
    assert res["grupos_excluidos_equalized_odds"] == ["G3_sem_pos"]
    # EOD calculado apenas entre G1 (Recall=1.0) e G2 (Recall=0.0) -> 1.0, mas PARCIAL
    assert res["equal_opportunity_difference"] == pytest.approx(1.0)
    assert res["equal_opportunity_escopo"] == "parcial_exclui_grupos_denominador_zero"
    # Ambos os eixos (TPR e FPR) tem >= 2 grupos validos, mas G3 foi excluido de TPR -> PARCIAL
    assert res["equalized_odds_difference"] == pytest.approx(1.0)
    assert res["equalized_odds_escopo"] == "parcial_exclui_grupos_denominador_zero"
    assert res["avaliacao_fairness_parcial"] is True
    assert "nao integral" in res["nota_denominadores_zero"]


def test_fairness_equalized_odds_indefinido_quando_um_eixo_nao_comparavel() -> None:
    """Se FPR nao possui >= 2 grupos validos, EOdds deve ser None (nao apresentar eixo unico)."""
    idx = list(range(6))
    # G1 possui 2 positivos e 2 negativos; G2 possui 2 positivos e 0 negativos (FPR indefinido)
    y_true = pd.Series([">50K", ">50K", "<=50K", "<=50K", ">50K", ">50K"], index=idx)
    y_pred = pd.Series([">50K", ">50K", "<=50K", ">50K", ">50K", "<=50K"], index=idx)
    grupos = pd.Series(["G1", "G1", "G1", "G1", "G2", "G2"], index=idx)

    aud = auditar_fairness_atributo(y_true, y_pred, grupos, nome_atributo="sex")
    res = aud["resumo"]

    # TPR e comparavel em ambos os grupos (G1=1.0, G2=0.5 -> EOD=0.5, integral)
    assert res["equal_opportunity_difference"] == pytest.approx(0.5)
    assert res["equal_opportunity_escopo"] == "integral_todos_os_grupos"
    # FPR so e definido em G1 -> eixo FPR nao comparavel -> EOdds DEVE ser None
    assert res["fpr_diff"] is None
    assert res["equalized_odds_difference"] is None
    assert res["equalized_odds_escopo"] == "indefinido_eixo_tpr_ou_fpr_ausente"
    assert res["grupos_fpr_indefinido"] == ["G2"]
    assert res["grupos_excluidos_equalized_odds"] == ["G2"]
    assert "nao reportado como avaliacao integral" in res["nota_denominadores_zero"]


def test_fairness_relatorio_e_summary_explicitam_escopo_parcial() -> None:
    """Tabela resumo e rotulo de relatorio devem explicitar quando EOdds e parcial ou indefinido."""
    from src.analysis.fairness import executar_auditoria_fairness
    from src.reporting.exports import _rotulo_escopo_fairness

    idx = list(range(10))
    X_eval = pd.DataFrame(
        {
            "sex": ["Female"] * 4 + ["Male"] * 6,
            "race": ["G1"] * 4 + ["G2"] * 4 + ["G3_sem_pos"] * 2,
        },
        index=idx,
    )
    y_true = pd.Series(
        [">50K", ">50K", "<=50K", "<=50K", ">50K", "<=50K", "<=50K", "<=50K", "<=50K", "<=50K"],
        index=idx,
    )
    y_pred = pd.Series(
        [">50K", ">50K", "<=50K", "<=50K", "<=50K", "<=50K", "<=50K", "<=50K", "<=50K", "<=50K"],
        index=idx,
    )

    aud_geral = executar_auditoria_fairness(X_eval, y_true, y_pred, atributos_sensiveis=["sex", "race"])
    df_res = aud_geral["tabela_resumo"].set_index("atributo")
    assert df_res.loc["sex", "equalized_odds_escopo"] == "integral_todos_os_grupos"
    assert bool(df_res.loc["sex", "avaliacao_fairness_parcial"]) is False
    assert df_res.loc["race", "equalized_odds_escopo"] == "parcial_exclui_grupos_denominador_zero"
    assert bool(df_res.loc["race", "avaliacao_fairness_parcial"]) is True
    assert df_res.loc["race", "grupos_excluidos_equalized_odds"] == "G3_sem_pos"
    assert "PARCIAL" in _rotulo_escopo_fairness(str(df_res.loc["race", "equalized_odds_escopo"]))
    assert "incompleta" in _rotulo_escopo_fairness("indefinido_eixo_tpr_ou_fpr_ausente")


def test_fairness_rejeita_desalinhamento_de_indices(
    cenario_fairness_conhecido: dict[str, pd.Series],
) -> None:
    """Se os indices dos atributos sensiveis estiverem desalinhados, deve levantar ValueError."""
    grupo_desalinhado = cenario_fairness_conhecido["grupo"].reset_index(drop=True)
    with pytest.raises(ValueError, match="Desalinhamento de indices"):
        auditar_fairness_atributo(
            y_true=cenario_fairness_conhecido["y_true"],
            y_pred=cenario_fairness_conhecido["y_pred"],
            sensitive_feature=grupo_desalinhado,
            nome_atributo="sex",
        )


def test_media_simples_entre_grupos_difere_da_metrica_global_em_grupos_desiguais() -> None:
    """Comprova que media simples de Recall por grupo nao substitui o Recall global ponderado."""
    # Grupo Grande: 9 positivos, TP=9 -> Recall=1.0
    # Grupo Pequeno: 1 positivo, TP=0 -> Recall=0.0
    # Recall global ponderado = 9/10 = 0.90; media simples dos grupos = (1.0 + 0.0)/2 = 0.50
    idx = list(range(12))
    y_true = pd.Series([">50K"] * 9 + ["<=50K"] + [">50K", "<=50K"], index=idx)
    y_pred = pd.Series([">50K"] * 9 + ["<=50K"] + ["<=50K", "<=50K"], index=idx)
    grupos = pd.Series(["Grande"] * 10 + ["Pequeno"] * 2, index=idx)

    aud = auditar_fairness_atributo(y_true, y_pred, grupos, nome_atributo="sex")
    assert aud["resumo"]["metricas_globais"]["recall"] == pytest.approx(0.90)
    assert aud["resumo"]["medias_simples_entre_grupos"]["recall"] == pytest.approx(0.50)
