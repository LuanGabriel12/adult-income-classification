"""Camada de validacao matematica e consistencia metodologica independente."""

from typing import Any

import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import (
    NEGATIVE_CLASS,
    POSITIVE_CLASS,
    SENSITIVE_COLUMNS,
    V1_HISTORICAL_BENCHMARK,
)


def _registrar_checagem(
    codigo: str,
    descricao: str,
    status: str,
    valor_esperado: str,
    valor_obtido: str,
    detalhes: str,
) -> dict[str, Any]:
    """Cria registro padronizado para o relatorio de reconciliacao."""
    return {
        "codigo": codigo,
        "descricao": descricao,
        "status": status,
        "valor_esperado": valor_esperado,
        "valor_obtido": valor_obtido,
        "detalhes": detalhes,
    }


def _igual_com_tolerancia(a: float | None, b: float | None, tol: float = 1e-12) -> bool:
    """Compara dois numeros decimais com tolerancia explicita."""
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return abs(float(a) - float(b)) <= tol


def executar_validacao_matematica(
    X: pd.DataFrame,
    y: pd.Series,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    y_pred_test: pd.Series,
    pipeline_final: Pipeline,
    metricas_teste: dict[str, Any],
    analise_erros: dict[str, Any],
    auditoria_fairness: dict[str, Any],
    comparacao_sensiveis: dict[str, Any],
    metadados_carga: dict[str, Any],
) -> dict[str, Any]:
    """Executa 16 verificacoes matematicas e metodologicas independentes (VAL-01..VAL-16)."""
    checagens: list[dict[str, Any]] = []

    # VAL-01: Quantidade de registros de treino e teste
    soma_split = len(X_train) + len(X_test)
    ok_val01 = (
        soma_split == len(X)
        and len(X_train) == len(y_train)
        and len(X_test) == len(y_test)
    )
    checagens.append(
        _registrar_checagem(
            "VAL-01",
            "Conservacao de registros na divisao treino/teste",
            "APROVADO" if ok_val01 else "REPROVADO",
            f"total={len(X)}",
            f"treino={len(X_train)}, teste={len(X_test)}, soma={soma_split}",
            "Verifica soma exata e correspondencia de tamanhos entre X e y.",
        )
    )

    # VAL-02: Ausencia de sobreposicao de indices entre treino e teste
    set_train = set(X_train.index.tolist())
    set_test = set(X_test.index.tolist())
    set_total = set(X.index.tolist())
    intersecao = set_train.intersection(set_test)
    uniao = set_train.union(set_test)
    ok_val02 = len(intersecao) == 0 and uniao == set_total
    checagens.append(
        _registrar_checagem(
            "VAL-02",
            "Ausencia de sobreposicao de indices entre treino e teste",
            "APROVADO" if ok_val02 else "REPROVADO",
            "intersecao=0 e uniao=X.index",
            f"intersecao={len(intersecao)}, uniao={len(uniao)}/{len(set_total)}",
            "Confirma particionamento disjunto e completo dos indices originais.",
        )
    )

    # VAL-03: Distribuicao das classes e estratificacao
    total_pos = int((y == POSITIVE_CLASS).sum())
    total_neg = int((y == NEGATIVE_CLASS).sum())
    soma_pos = int((y_train == POSITIVE_CLASS).sum()) + int((y_test == POSITIVE_CLASS).sum())
    soma_neg = int((y_train == NEGATIVE_CLASS).sum()) + int((y_test == NEGATIVE_CLASS).sum())
    prop_train = float((y_train == POSITIVE_CLASS).mean())
    prop_test = float((y_test == POSITIVE_CLASS).mean())
    ok_val03 = soma_pos == total_pos and soma_neg == total_neg and abs(prop_train - prop_test) < 0.02
    checagens.append(
        _registrar_checagem(
            "VAL-03",
            "Conservacao da distribuicao das classes e estratificacao",
            "APROVADO" if ok_val03 else "REPROVADO",
            f"<=50K={total_neg}, >50K={total_pos}",
            f"<=50K={soma_neg}, >50K={soma_pos} (pos_train={prop_train:.4f}, pos_test={prop_test:.4f})",
            "Garante soma exata das classes e preservacao da proporcao positiva.",
        )
    )

    # VAL-04: Correspondencia entre y_test, y_pred_test e atributos sensiveis
    idx_ok = (
        list(X_test.index) == list(y_test.index)
        and list(y_test.index) == list(y_pred_test.index)
        and all(
            list(X_test[attr].index) == list(y_pred_test.index)
            for attr in SENSITIVE_COLUMNS
            if attr in X_test.columns
        )
    )
    checagens.append(
        _registrar_checagem(
            "VAL-04",
            "Alinhamento exato entre X_test, y_test, y_pred e atributos sensiveis",
            "APROVADO" if idx_ok else "REPROVADO",
            f"indices identicos ({len(y_test)} linhas)",
            "alinhados" if idx_ok else "desalinhados",
            "Impede troca silenciosa de posicoes entre previsoes e grupos sensiveis.",
        )
    )

    # VAL-05: Soma dos elementos e marginais da matriz de confusao
    cm = metricas_teste["matriz_confusao"]
    tn, fp, fn, tp = cm["tn"], cm["fp"], cm["fn"], cm["tp"]
    soma_cm = tn + fp + fn + tp
    pos_reais = int((y_test == POSITIVE_CLASS).sum())
    neg_reais = int((y_test == NEGATIVE_CLASS).sum())
    pos_prev = int((y_pred_test == POSITIVE_CLASS).sum())
    neg_prev = int((y_pred_test == NEGATIVE_CLASS).sum())
    ok_val05 = (
        soma_cm == len(y_test)
        and (tp + fn) == pos_reais
        and (tn + fp) == neg_reais
        and (tp + fp) == pos_prev
        and (tn + fn) == neg_prev
    )
    checagens.append(
        _registrar_checagem(
            "VAL-05",
            "Soma e marginais da matriz de confusao",
            "APROVADO" if ok_val05 else "REPROVADO",
            f"N={len(y_test)}, pos={pos_reais}, neg={neg_reais}",
            f"TN={tn}, FP={fp}, FN={fn}, TP={tp} (soma={soma_cm})",
            "Valida TN+FP+FN+TP=N e correspondencia com reais e previstos.",
        )
    )

    # VAL-06: Reproducao independente de Accuracy, Precision, Recall e F1 a partir da matriz
    acc_manual = (tp + tn) / soma_cm if soma_cm > 0 else None
    prec_manual = tp / (tp + fp) if (tp + fp) > 0 else None
    rec_manual = tp / (tp + fn) if (tp + fn) > 0 else None
    if prec_manual is not None and rec_manual is not None and (prec_manual + rec_manual) > 0:
        f1_manual: float | None = 2.0 * prec_manual * rec_manual / (prec_manual + rec_manual)
    elif (tp + fp) > 0 and (tp + fn) > 0:
        f1_manual = 0.0
    else:
        f1_manual = None

    ok_val06 = (
        _igual_com_tolerancia(acc_manual, metricas_teste["accuracy"])
        and _igual_com_tolerancia(prec_manual, metricas_teste["precision"])
        and _igual_com_tolerancia(rec_manual, metricas_teste["recall"])
        and _igual_com_tolerancia(f1_manual, metricas_teste["f1"])
    )
    checagens.append(
        _registrar_checagem(
            "VAL-06",
            "Reproducao independente de Accuracy, Precision, Recall e F1 pela matriz",
            "APROVADO" if ok_val06 else "REPROVADO",
            f"Acc={acc_manual:.6f}, Rec={rec_manual:.6f}, Prec={prec_manual:.6f}, F1={f1_manual:.6f}"
            if None not in (acc_manual, rec_manual, prec_manual, f1_manual)
            else "Metricas com denominador zero tratadas",
            f"Acc={metricas_teste['accuracy']:.6f}, Rec={metricas_teste['recall']:.6f}, "
            f"Prec={metricas_teste['precision']:.6f}, F1={metricas_teste['f1']:.6f}"
            if None
            not in (
                metricas_teste["accuracy"],
                metricas_teste["recall"],
                metricas_teste["precision"],
                metricas_teste["f1"],
            )
            else "Metricas com denominador zero sinalizadas",
            "Confirma concordancia absoluta (tolerancia <= 1e-12) com o scikit-learn.",
        )
    )

    # VAL-07: Conferencia independente do classification_report (macro avg e weighted avg)
    cr = metricas_teste["classification_report_dict"]
    rec_neg_manual = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    rec_pos_manual = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    macro_rec_manual = (rec_neg_manual + rec_pos_manual) / 2.0
    weighted_rec_manual = (rec_neg_manual * neg_reais + rec_pos_manual * pos_reais) / soma_cm
    ok_val07 = (
        int(cr[NEGATIVE_CLASS]["support"]) == neg_reais
        and int(cr[POSITIVE_CLASS]["support"]) == pos_reais
        and _igual_com_tolerancia(macro_rec_manual, float(cr["macro avg"]["recall"]))
        and _igual_com_tolerancia(weighted_rec_manual, float(cr["weighted avg"]["recall"]))
    )
    checagens.append(
        _registrar_checagem(
            "VAL-07",
            "Reconciliacao do classification_report (suportes, macro avg e weighted avg)",
            "APROVADO" if ok_val07 else "REPROVADO",
            f"macro_rec={macro_rec_manual:.6f}, weighted_rec={weighted_rec_manual:.6f}",
            f"macro_rec={float(cr['macro avg']['recall']):.6f}, weighted_rec={float(cr['weighted avg']['recall']):.6f}",
            "Confirma diferenca matematica entre media igualitaria (macro) e ponderada (weighted).",
        )
    )

    # VAL-08: Reconciliacao da analise de erros em todas as dimensoes
    tabela_erros = analise_erros["tabela_erros"]
    dimensoes_ok = True
    dims_verificadas = 0
    for _dimensao, grupo_dim in tabela_erros.groupby("dimensao"):
        dims_verificadas += 1
        if (
            int(grupo_dim["total"].sum()) != len(y_test)
            or int(grupo_dim["tp"].sum()) != tp
            or int(grupo_dim["tn"].sum()) != tn
            or int(grupo_dim["fp"].sum()) != fp
            or int(grupo_dim["fn"].sum()) != fn
        ):
            dimensoes_ok = False
    ok_val08 = dims_verificadas > 0 and dimensoes_ok
    checagens.append(
        _registrar_checagem(
            "VAL-08",
            "Conservacao de TP/TN/FP/FN em todas as dimensoes da analise de erros",
            "APROVADO" if ok_val08 else "REPROVADO",
            f"{dims_verificadas} dimensoes somando N={len(y_test)} e matriz global",
            "todas reconciliadas" if ok_val08 else "divergencia detectada",
            "Garante que categorias ausentes e faixas cobrem 100% dos registros sem duplicacao.",
        )
    )

    # VAL-09 e VAL-10: Conservacao de registros, positivos e negativos por sex e race
    for codigo, atributo in [("VAL-09", "sex"), ("VAL-10", "race")]:
        df_g = auditoria_fairness["por_atributo"][atributo]["tabela_por_grupo"]
        soma_n = int(df_g["count"].sum())
        soma_p = int(df_g["positivos_reais"].sum())
        soma_neg_g = int(df_g["negativos_reais"].sum())
        ok_attr = soma_n == len(y_test) and soma_p == pos_reais and soma_neg_g == neg_reais
        checagens.append(
            _registrar_checagem(
                codigo,
                f"Conservacao de registros, positivos e negativos por grupo ({atributo})",
                "APROVADO" if ok_attr else "REPROVADO",
                f"N={len(y_test)}, pos={pos_reais}, neg={neg_reais}",
                f"N={soma_n}, pos={soma_p}, neg={soma_neg_g} ({len(df_g)} grupos)",
                f"Confirma cobertura total da amostra de teste pelos grupos de {atributo}.",
            )
        )

    # VAL-11: Correspondencia entre metricas globais e metricas por grupo com ponderacao correta
    linhas_ponderacao: list[dict[str, Any]] = []
    ponderacao_ok = True
    for atributo in SENSITIVE_COLUMNS:
        aud = auditoria_fairness["por_atributo"][atributo]
        df_g = aud["tabela_por_grupo"]
        glob = aud["resumo"]["metricas_globais"]
        simples = aud["resumo"]["medias_simples_entre_grupos"]

        acc_pond = float(
            sum(float(df_g.loc[i, "accuracy"]) * int(df_g.loc[i, "count"]) for i in df_g.index)
            / int(df_g["count"].sum())
        )
        sel_pond = float(
            sum(
                float(df_g.loc[i, "selection_rate"]) * int(df_g.loc[i, "count"])
                for i in df_g.index
            )
            / int(df_g["count"].sum())
        )

        pos_validos_soma = int(df_g.loc[~df_g["recall_indefinido"], "positivos_reais"].sum())
        rec_pond = (
            float(
                sum(
                    float(df_g.loc[i, "recall"]) * int(df_g.loc[i, "positivos_reais"])
                    for i in df_g.index
                    if not bool(df_g.loc[i, "recall_indefinido"])
                )
                / pos_validos_soma
            )
            if pos_validos_soma > 0
            else None
        )

        neg_validos_soma = int(df_g.loc[~df_g["fpr_indefinido"], "negativos_reais"].sum())
        fpr_pond = (
            float(
                sum(
                    float(df_g.loc[i, "fpr"]) * int(df_g.loc[i, "negativos_reais"])
                    for i in df_g.index
                    if not bool(df_g.loc[i, "fpr_indefinido"])
                )
                / neg_validos_soma
            )
            if neg_validos_soma > 0
            else None
        )

        pares_check = [
            ("accuracy", glob["accuracy"], acc_pond, simples["accuracy"], "count (N_g)"),
            ("recall", glob["recall"], rec_pond, simples["recall"], "positivos_reais (TP_g + FN_g)"),
            ("fpr", glob["fpr"], fpr_pond, simples["fpr"], "negativos_reais (TN_g + FP_g)"),
            ("selection_rate", glob["selection_rate"], sel_pond, simples["selection_rate"], "count (N_g)"),
        ]
        for metrica, val_global, val_pond, val_simples, peso_desc in pares_check:
            bate = _igual_com_tolerancia(val_global, val_pond)
            if not bate:
                ponderacao_ok = False
            linhas_ponderacao.append(
                {
                    "atributo": atributo,
                    "metrica": metrica,
                    "metrica_global": val_global,
                    "recalculo_ponderado": val_pond,
                    "media_simples_grupos": val_simples,
                    "desvio_media_simples": (
                        float(val_simples - val_global)
                        if (val_simples is not None and val_global is not None)
                        else None
                    ),
                    "peso_correto": peso_desc,
                    "reconciliado": bate,
                }
            )

    checagens.append(
        _registrar_checagem(
            "VAL-11",
            "Reconciliacao ponderada entre metricas por grupo e metricas globais",
            "APROVADO" if ponderacao_ok else "REPROVADO",
            "soma ponderada == metrica global (tol <= 1e-12)",
            "todas as 8 combinacoes reconciliadas" if ponderacao_ok else "divergencia na ponderacao",
            "Comprova que Recall exige ponderacao por positivos reais e FPR por negativos reais (nao media simples).",
        )
    )

    # VAL-12: Reproducao independente de Equal Opportunity Difference e Equalized Odds Difference
    eod_eodds_ok = True
    detalhes_fair: list[str] = []
    for atributo in SENSITIVE_COLUMNS:
        aud = auditoria_fairness["por_atributo"][atributo]
        df_g = aud["tabela_por_grupo"]
        res_attr = aud["resumo"]
        recalls_validos = [
            float(v) for v in df_g.loc[~df_g["recall_indefinido"], "recall"].tolist()
        ]
        fprs_validos = [
            float(v) for v in df_g.loc[~df_g["fpr_indefinido"], "fpr"].tolist()
        ]
        eod_manual = (
            float(max(recalls_validos) - min(recalls_validos))
            if len(recalls_validos) >= 2
            else None
        )
        diff_fpr_manual = (
            float(max(fprs_validos) - min(fprs_validos)) if len(fprs_validos) >= 2 else None
        )
        eodds_manual = (
            float(max(eod_manual, diff_fpr_manual))
            if (eod_manual is not None and diff_fpr_manual is not None)
            else None
        )

        possui_zero_attr = bool(res_attr["possui_denominadores_zero"])
        escopo_eodds_esperado = (
            "integral_todos_os_grupos"
            if (not possui_zero_attr and len(df_g) >= 2)
            else (
                "parcial_exclui_grupos_denominador_zero"
                if eodds_manual is not None
                else (
                    "indefinido_ambos_eixos_indisponiveis"
                    if (eod_manual is None and diff_fpr_manual is None)
                    else "indefinido_eixo_tpr_ou_fpr_ausente"
                )
            )
        )

        eod_fair = res_attr["equal_opportunity_difference"]
        eodds_fair = res_attr["equalized_odds_difference"]
        escopo_eodds_obtido = res_attr["equalized_odds_escopo"]
        if not (
            _igual_com_tolerancia(eod_manual, eod_fair)
            and _igual_com_tolerancia(eodds_manual, eodds_fair)
            and escopo_eodds_obtido == escopo_eodds_esperado
        ):
            eod_eodds_ok = False
        eod_txt = f"{eod_fair:.6f}" if eod_fair is not None else "Indefinido"
        eodds_txt = f"{eodds_fair:.6f}" if eodds_fair is not None else "Indefinido"
        detalhes_fair.append(
            f"{atributo}: EOD={eod_txt}, EOdds={eodds_txt} ({escopo_eodds_obtido})"
        )

    checagens.append(
        _registrar_checagem(
            "VAL-12",
            "Reproducao independente de Equal Opportunity e Equalized Odds vs Fairlearn",
            "APROVADO" if eod_eodds_ok else "REPROVADO",
            "max-min manual == Fairlearn (tol <= 1e-12) e escopo integral/parcial/indefinido consistente",
            "; ".join(detalhes_fair),
            "Valida EOD = max(TPR_g)-min(TPR_g), EOdds = max(diff_TPR, diff_FPR) apenas com ambos os eixos validos, e sinalizacao explicita de comparacao parcial.",
        )
    )

    # VAL-13: Consistencia do experimento controlado com atributos sensiveis
    proto = comparacao_sensiveis["protocolo"]
    feats_a = proto["features_cenario_a"]
    feats_b = proto["features_cenario_b"]
    removidos = proto["atributos_removidos_cenario_b"]
    alinhamentos = comparacao_sensiveis["verificacao_alinhamento"]
    alinhamento_ok = all(bool(v) for v in alinhamentos.values())
    ok_val13 = (
        len(feats_a) - len(removidos) == len(feats_b)
        and all(attr not in feats_b for attr in removidos)
        and alinhamento_ok
    )
    checagens.append(
        _registrar_checagem(
            "VAL-13",
            "Integridade do experimento Cenario A vs Cenario B (atributos sensiveis)",
            "APROVADO" if ok_val13 else "REPROVADO",
            f"Cenario A={len(feats_a)} features, Cenario B={len(feats_a) - len(removidos)} sem {removidos}",
            f"Cenario A={len(feats_a)}, Cenario B={len(feats_b)}, alinhamento={alinhamento_ok}",
            "Confirma remocao efetiva de sex/race no Cenario B sem desalinhar indices da auditoria.",
        )
    )

    # VAL-14: Evidencia anti-data-leakage no pre-processamento
    preprocessador_ajustado = pipeline_final.named_steps["preprocessador"]
    scaler = preprocessador_ajustado.named_transformers_["numericas"].named_steps["scaler"]
    colunas_num_treinadas = list(preprocessador_ajustado.transformers_[0][2])
    primeira_num = colunas_num_treinadas[0]
    media_scaler = float(scaler.mean_[0])
    media_treino_manual = float(X_train[primeira_num].mean())
    media_base_total = float(X[primeira_num].mean())
    ok_val14 = _igual_com_tolerancia(media_scaler, media_treino_manual, tol=1e-9)
    checagens.append(
        _registrar_checagem(
            "VAL-14",
            f"Ajuste exclusivo do pre-processamento no treino (anti-leakage em '{primeira_num}')",
            "APROVADO" if ok_val14 else "REPROVADO",
            f"media_treino={media_treino_manual:.6f}",
            f"scaler.mean_={media_scaler:.6f} (media_total_base={media_base_total:.6f})",
            "Confirma que StandardScaler aprendeu estatisticas apenas de X_train, nunca de X completo.",
        )
    )

    # VAL-15: Auditoria de denominadores iguais a zero e escopo parcial explicito
    grupos_indef_total = sum(
        len(auditoria_fairness["por_atributo"][attr]["resumo"]["grupos_recall_indefinido"])
        + len(auditoria_fairness["por_atributo"][attr]["resumo"]["grupos_fpr_indefinido"])
        for attr in SENSITIVE_COLUMNS
    )
    mascarado_como_zero = False
    parcial_nao_sinalizado = False
    for attr in SENSITIVE_COLUMNS:
        aud_attr = auditoria_fairness["por_atributo"][attr]
        df_g = aud_attr["tabela_por_grupo"]
        res_attr = aud_attr["resumo"]
        for i in df_g.index:
            if bool(df_g.loc[i, "recall_indefinido"]) and not pd.isna(df_g.loc[i, "recall"]):
                mascarado_como_zero = True
            if bool(df_g.loc[i, "fpr_indefinido"]) and not pd.isna(df_g.loc[i, "fpr"]):
                mascarado_como_zero = True
        if bool(res_attr["possui_denominadores_zero"]) and (
            res_attr["equalized_odds_escopo"] == "integral_todos_os_grupos"
        ):
            parcial_nao_sinalizado = True

    ok_val15 = (not mascarado_como_zero) and (not parcial_nao_sinalizado)
    checagens.append(
        _registrar_checagem(
            "VAL-15",
            "Tratamento transparente de denominadores iguais a zero e escopo parcial",
            "APROVADO" if ok_val15 else "REPROVADO",
            "0 metricas indefinidas mascaradas como zero e nenhuma avaliacao parcial rotulada como integral",
            f"grupos_com_denominador_zero={grupos_indef_total}, mascarados=0, parciais_nao_sinalizados=0"
            if ok_val15
            else "metrica indefinida mascarada ou avaliacao parcial apresentada como integral",
            "Garante que divisoes 0/0 retornam NaN/indefinido e que Equalized Odds parcial/incompleto nunca e apresentado como integral.",
        )
    )

    # VAL-16: Reproducao dos resultados historicos da V1 (apenas para o dataset oficial OpenML 1590 v2)
    if bool(metadados_carga.get("dataset_oficial_openml_1590")):
        ref = V1_HISTORICAL_BENCHMARK
        eod_race = auditoria_fairness["por_atributo"]["race"]["resumo"][
            "equal_opportunity_difference"
        ]
        eodds_race = auditoria_fairness["por_atributo"]["race"]["resumo"][
            "equalized_odds_difference"
        ]
        ok_val16 = (
            tn == ref["tn"]
            and fp == ref["fp"]
            and fn == ref["fn"]
            and tp == ref["tp"]
            and _igual_com_tolerancia(metricas_teste["accuracy"], ref["accuracy"], tol=1e-6)
            and _igual_com_tolerancia(metricas_teste["recall"], ref["recall"], tol=1e-6)
            and _igual_com_tolerancia(metricas_teste["precision"], ref["precision"], tol=1e-6)
            and _igual_com_tolerancia(metricas_teste["f1"], ref["f1"], tol=1e-6)
            and _igual_com_tolerancia(eod_race, ref["equal_opportunity_diff_race"], tol=1e-6)
            and _igual_com_tolerancia(eodds_race, ref["equalized_odds_diff_race"], tol=1e-6)
        )
        checagens.append(
            _registrar_checagem(
                "VAL-16",
                "Reproducao fiel da referencia historica V1 (OpenML 1590 v2)",
                "APROVADO" if ok_val16 else "REPROVADO",
                "Acc=0.85239, Rec=0.58854, Prec=0.74138, F1=0.65618, CM=6951/480/962/1376, EOD_race=0.321951",
                f"Acc={metricas_teste['accuracy']:.5f}, Rec={metricas_teste['recall']:.5f}, "
                f"Prec={metricas_teste['precision']:.5f}, F1={metricas_teste['f1']:.5f}, "
                f"CM={tn}/{fp}/{fn}/{tp}, EOD_race={eod_race:.6f}",
                "Compara a execucao atual com os resultados historicos reproduzidos da V1.",
            )
        )
    else:
        checagens.append(
            _registrar_checagem(
                "VAL-16",
                "Reproducao fiel da referencia historica V1 (OpenML 1590 v2)",
                "NAO_APLICAVEL",
                "Aplicavel apenas ao dataset oficial OpenML 1590 v2 verificado por SHA-256",
                "Dataset alternativo/sintetico",
                "Evita falsas comparacoes historicas quando executado sobre outro conjunto de dados.",
            )
        )

    tabela_checagens = pd.DataFrame(checagens)
    aprovadas = int((tabela_checagens["status"] == "APROVADO").sum())
    reprovadas = int((tabela_checagens["status"] == "REPROVADO").sum())
    nao_aplicaveis = int((tabela_checagens["status"] == "NAO_APLICAVEL").sum())

    return {
        "total_checagens": int(len(tabela_checagens)),
        "total_aprovadas": aprovadas,
        "total_reprovadas": reprovadas,
        "total_nao_aplicaveis": nao_aplicaveis,
        "aprovado_geral": reprovadas == 0,
        "tabela_checagens": tabela_checagens,
        "comparacao_ponderada_vs_simples": pd.DataFrame(linhas_ponderacao),
    }
