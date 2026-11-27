"""Auditoria reproduzivel de qualidade dos dados do Adult / Census Income."""

from typing import Any

import pandas as pd

from src.config import (
    EXPECTED_CATEGORIES,
    NEGATIVE_CLASS,
    POSITIVE_CLASS,
    RARE_CATEGORY_THRESHOLD_RATIO,
    SENSITIVE_COLUMNS,
)


def normalizar_serie_texto(serie: pd.Series, marcador_ausente: str) -> pd.Series:
    """Converte uma Series para texto sem alterar a original e substituindo ausentes.

    Evita ``.astype("object").fillna(...)`` para nao acionar downcasting implicito
    (FutureWarning no Pandas 2.2) e preserva integralmente os valores ausentes na
    Series de entrada.
    """
    mascara_ausente = serie.isna()
    if not bool(mascara_ausente.any()):
        return serie.astype(str).copy()
    texto = serie.astype(str).copy()
    texto.loc[mascara_ausente] = marcador_ausente
    return texto


def _contar_sobreposicao_exata_treino_teste(
    df_treino: pd.DataFrame,
    df_teste: pd.DataFrame,
) -> int:
    """Conta quantas linhas do teste possuem combinacao identica no treino.

    Substitui valores ausentes por um marcador temporario em copias isoladas
    para que o merge trate ausencias de forma deterministica sem alterar os dados.
    """
    colunas = list(df_treino.columns)
    treino_norm = pd.DataFrame(
        {col: normalizar_serie_texto(df_treino[col], "<AUSENTE>") for col in colunas},
        index=df_treino.index,
    )
    teste_norm = pd.DataFrame(
        {col: normalizar_serie_texto(df_teste[col], "<AUSENTE>") for col in colunas},
        index=df_teste.index,
    )

    treino_unicos = treino_norm.drop_duplicates()
    correspondencias = teste_norm.merge(treino_unicos, on=colunas, how="inner")
    return int(len(correspondencias))


def auditar_qualidade_dados(
    X: pd.DataFrame,
    y: pd.Series,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    metadados_carga: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Executa auditoria completa de qualidade sem remover registros silenciosamente."""
    total_linhas = int(len(X))
    total_features = int(X.shape[1])
    colunas_numericas = X.select_dtypes(include=["number"]).columns.tolist()
    colunas_categoricas = X.select_dtypes(exclude=["number"]).columns.tolist()

    # 1. Tipos e valores ausentes por coluna
    resumo_colunas: list[dict[str, Any]] = []
    for col in X.columns:
        ausentes = int(X[col].isna().sum())
        resumo_colunas.append(
            {
                "coluna": col,
                "tipo_logico": "numerica" if col in colunas_numericas else "categorica",
                "dtype": str(X[col].dtype),
                "valores_unicos": int(X[col].nunique(dropna=True)),
                "ausentes": ausentes,
                "percentual_ausentes": float(ausentes / total_linhas) if total_linhas > 0 else 0.0,
            }
        )

    # Analise conjunta de ausencias entre workclass e occupation
    analise_ausentes_trabalho: dict[str, Any] = {}
    if "workclass" in X.columns and "occupation" in X.columns:
        ambos_na = int((X["workclass"].isna() & X["occupation"].isna()).sum())
        so_workclass_na = int((X["workclass"].isna() & X["occupation"].notna()).sum())
        so_occupation_na = int((X["workclass"].notna() & X["occupation"].isna()).sum())
        never_worked_occ_na = int(
            ((X["workclass"] == "Never-worked") & X["occupation"].isna()).sum()
        )
        analise_ausentes_trabalho = {
            "ambos_ausentes": ambos_na,
            "apenas_workclass_ausente": so_workclass_na,
            "apenas_occupation_ausente": so_occupation_na,
            "never_worked_com_occupation_ausente": never_worked_occ_na,
        }

    # 2. Inconsistencias de codificacao e categorias inesperadas
    inconsistencias_codificacao: list[dict[str, Any]] = []
    categorias_inesperadas: list[dict[str, Any]] = []
    for col in colunas_categoricas:
        serie_nao_nula = X[col].dropna().astype(str)
        qtd_interrogacao = int((serie_nao_nula == "?").sum())
        qtd_espacos = int((serie_nao_nula != serie_nao_nula.str.strip()).sum())
        if qtd_interrogacao > 0 or qtd_espacos > 0:
            inconsistencias_codificacao.append(
                {
                    "coluna": col,
                    "marcadores_interrogacao": qtd_interrogacao,
                    "espacos_laterais": qtd_espacos,
                }
            )
        if col in EXPECTED_CATEGORIES:
            esperadas = set(EXPECTED_CATEGORIES[col])
            mascara_inesperada = ~serie_nao_nula.isin(esperadas)
            if bool(mascara_inesperada.any()):
                valores_estranhos = sorted(set(serie_nao_nula.loc[mascara_inesperada].tolist()))
                categorias_inesperadas.append(
                    {
                        "coluna": col,
                        "categorias_inesperadas": valores_estranhos,
                        "quantidade_registros": int(mascara_inesperada.sum()),
                    }
                )

    y_str = y.astype(str)
    alvo_com_ponto_final = int(y_str.str.endswith(".").sum())
    alvo_com_espacos = int((y_str != y_str.str.strip()).sum())

    # Redundancia education vs education-num
    redundancia_educacao: dict[str, Any] = {}
    if "education" in X.columns and "education-num" in X.columns:
        max_num_por_edu = int(
            X.groupby("education", observed=True)["education-num"].nunique().max()
        )
        max_edu_por_num = int(
            X.groupby("education-num", observed=True)["education"].nunique().max()
        )
        redundancia_educacao = {
            "mapeamento_1_para_1": max_num_por_edu == 1 and max_edu_por_num == 1,
            "niveis_education": int(X["education"].nunique(dropna=True)),
            "niveis_education_num": int(X["education-num"].nunique(dropna=True)),
        }

    # Cruzamento sex vs relationship
    cruzamentos_sex_relationship: dict[str, int] = {}
    if "sex" in X.columns and "relationship" in X.columns:
        cruzamentos_sex_relationship = {
            "female_com_husband": int(
                ((X["sex"] == "Female") & (X["relationship"] == "Husband")).sum()
            ),
            "male_com_wife": int(
                ((X["sex"] == "Male") & (X["relationship"] == "Wife")).sum()
            ),
        }

    # 3. Duplicatas e sobreposicao treino x teste (sem remocao automatica)
    df_completo = pd.concat([X, y], axis=1)
    duplicatas_xy_excedentes = int(df_completo.duplicated(keep="first").sum())
    duplicatas_xy_envolvidas = int(df_completo.duplicated(keep=False).sum())
    duplicatas_x_excedentes = int(X.duplicated(keep="first").sum())
    duplicatas_x_envolvidas = int(X.duplicated(keep=False).sum())

    df_norm_dup = pd.DataFrame(
        {col: normalizar_serie_texto(df_completo[col], "<AUSENTE>") for col in X.columns},
        index=df_completo.index,
    )
    df_norm_dup[y.name] = df_completo[y.name]
    grupos_x_conflito = df_norm_dup.groupby(list(X.columns))[y.name].nunique()
    grupos_x_com_target_conflitante = int((grupos_x_conflito > 1).sum())

    duplicatas_sem_fnlwgt_xy = 0
    if "fnlwgt" in X.columns:
        colunas_sem_fnl = [col for col in X.columns if col != "fnlwgt"]
        duplicatas_sem_fnlwgt_xy = int(
            df_completo.duplicated(subset=colunas_sem_fnl + [y.name], keep="first").sum()
        )

    treino_completo = pd.concat([X_train, y_train], axis=1)
    teste_completo = pd.concat([X_test, y_test], axis=1)
    sobreposicao_indices = len(set(X_train.index).intersection(set(X_test.index)))
    sobreposicao_teste_xy_no_treino = _contar_sobreposicao_exata_treino_teste(
        treino_completo, teste_completo
    )
    sobreposicao_teste_x_no_treino = _contar_sobreposicao_exata_treino_teste(X_train, X_test)

    # 4. Distribuicao das classes e desbalanceamento
    qtd_neg = int((y_str == NEGATIVE_CLASS).sum())
    qtd_pos = int((y_str == POSITIVE_CLASS).sum())
    distribuicao_classes = {
        NEGATIVE_CLASS: {
            "quantidade": qtd_neg,
            "proporcao": float(qtd_neg / total_linhas) if total_linhas > 0 else 0.0,
        },
        POSITIVE_CLASS: {
            "quantidade": qtd_pos,
            "proporcao": float(qtd_pos / total_linhas) if total_linhas > 0 else 0.0,
        },
        "razao_majoritaria_sobre_minoritaria": (
            float(qtd_neg / qtd_pos) if qtd_pos > 0 else None
        ),
    }

    # 5. Distribuicao dos atributos sensiveis
    distribuicao_sensiveis: dict[str, list[dict[str, Any]]] = {}
    for attr in SENSITIVE_COLUMNS:
        if attr in X.columns:
            serie_attr = normalizar_serie_texto(X[attr], "Ausente")
            linhas_attr: list[dict[str, Any]] = []
            for grupo, qtd in serie_attr.value_counts().items():
                mascara = serie_attr == str(grupo)
                pos_grupo = int((y_str.loc[mascara] == POSITIVE_CLASS).sum())
                neg_grupo = int((y_str.loc[mascara] == NEGATIVE_CLASS).sum())
                linhas_attr.append(
                    {
                        "grupo": str(grupo),
                        "quantidade": int(qtd),
                        "proporcao_base": float(qtd / total_linhas) if total_linhas > 0 else 0.0,
                        "positivos_reais": pos_grupo,
                        "negativos_reais": neg_grupo,
                        "taxa_positiva": float(pos_grupo / qtd) if qtd > 0 else 0.0,
                    }
                )
            distribuicao_sensiveis[attr] = linhas_attr

    # 6. Estatisticas numericas e valores atipicos (IQR)
    estatisticas_numericas: list[dict[str, Any]] = []
    for col in colunas_numericas:
        serie = X[col].dropna()
        if len(serie) == 0:
            continue
        q25 = float(serie.quantile(0.25))
        mediana = float(serie.median())
        q75 = float(serie.quantile(0.75))
        iqr = q75 - q25
        lim_inf = q25 - 1.5 * iqr
        lim_sup = q75 + 1.5 * iqr
        atipicos_iqr = int(((serie < lim_inf) | (serie > lim_sup)).sum())
        estatisticas_numericas.append(
            {
                "coluna": col,
                "min": float(serie.min()),
                "q25": q25,
                "mediana": mediana,
                "q75": q75,
                "max": float(serie.max()),
                "media": float(serie.mean()),
                "atipicos_iqr": atipicos_iqr,
                "percentual_atipicos_iqr": float(atipicos_iqr / len(serie)),
            }
        )

    extremos_censo: dict[str, Any] = {}
    if "capital-gain" in X.columns:
        cg = X["capital-gain"].dropna()
        cg_menor_99999 = cg.loc[cg < 99999]
        extremos_censo["capital_gain_zero_qtd"] = int((cg == 0).sum())
        extremos_censo["capital_gain_zero_pct"] = float((cg == 0).mean()) if len(cg) > 0 else 0.0
        extremos_censo["capital_gain_99999_qtd"] = int((cg == 99999).sum())
        extremos_censo["capital_gain_max_abaixo_99999"] = (
            float(cg_menor_99999.max()) if len(cg_menor_99999) > 0 else None
        )
    if "capital-loss" in X.columns:
        cl = X["capital-loss"].dropna()
        extremos_censo["capital_loss_zero_qtd"] = int((cl == 0).sum())
        extremos_censo["capital_loss_zero_pct"] = float((cl == 0).mean()) if len(cl) > 0 else 0.0
        extremos_censo["capital_loss_max"] = float(cl.max()) if len(cl) > 0 else None
    if "hours-per-week" in X.columns:
        hpw = X["hours-per-week"].dropna()
        extremos_censo["hours_per_week_99_qtd"] = int((hpw == 99).sum())
        extremos_censo["hours_per_week_40_qtd"] = int((hpw == 40).sum())
        extremos_censo["hours_per_week_1_qtd"] = int((hpw == 1).sum())
    if "age" in X.columns:
        age = X["age"].dropna()
        extremos_censo["age_90_qtd"] = int((age == 90).sum())
        extremos_censo["age_17_qtd"] = int((age == 17).sum())

    # 7. Investigacao especifica de fnlwgt (peso amostral do CPS)
    analise_fnlwgt: dict[str, Any] = {}
    if "fnlwgt" in X.columns:
        fnl = X["fnlwgt"].dropna()
        if len(fnl) > 0:
            min_fnl = float(fnl.min())
            max_fnl = float(fnl.max())
            analise_fnlwgt = {
                "min": min_fnl,
                "mediana": float(fnl.median()),
                "media": float(fnl.mean()),
                "max": max_fnl,
                "valores_unicos": int(fnl.nunique()),
                "razao_max_sobre_min": float(max_fnl / min_fnl) if min_fnl > 0 else None,
            }

    # 8. Categorias raras (< 0,5% da base)
    categorias_raras: list[dict[str, Any]] = []
    for col in colunas_categoricas:
        contagens = X[col].dropna().astype(str).value_counts()
        for categoria, qtd in contagens.items():
            proporcao = float(qtd / total_linhas) if total_linhas > 0 else 0.0
            if proporcao < RARE_CATEGORY_THRESHOLD_RATIO:
                categorias_raras.append(
                    {
                        "coluna": col,
                        "categoria": str(categoria),
                        "quantidade": int(qtd),
                        "percentual": proporcao,
                    }
                )
    categorias_raras.sort(key=lambda item: (item["quantidade"], item["coluna"], item["categoria"]))

    return {
        "metadados_carga": metadados_carga or {},
        "dimensoes": {
            "total_linhas": total_linhas,
            "total_features": total_features,
            "treino_linhas": int(len(X_train)),
            "teste_linhas": int(len(X_test)),
            "colunas_numericas": colunas_numericas,
            "colunas_categoricas": colunas_categoricas,
        },
        "resumo_colunas": resumo_colunas,
        "analise_ausentes_trabalho": analise_ausentes_trabalho,
        "inconsistencias_codificacao": inconsistencias_codificacao,
        "categorias_inesperadas": categorias_inesperadas,
        "alvo_com_ponto_final": alvo_com_ponto_final,
        "alvo_com_espacos": alvo_com_espacos,
        "redundancia_educacao": redundancia_educacao,
        "cruzamentos_sex_relationship": cruzamentos_sex_relationship,
        "duplicatas": {
            "possui_identificador_unico": False,
            "duplicatas_xy_excedentes": duplicatas_xy_excedentes,
            "duplicatas_xy_linhas_envolvidas": duplicatas_xy_envolvidas,
            "duplicatas_x_excedentes": duplicatas_x_excedentes,
            "duplicatas_x_linhas_envolvidas": duplicatas_x_envolvidas,
            "grupos_x_com_target_conflitante": grupos_x_com_target_conflitante,
            "duplicatas_xy_ignorando_fnlwgt": duplicatas_sem_fnlwgt_xy,
            "sobreposicao_indices_treino_teste": sobreposicao_indices,
            "sobreposicao_teste_xy_no_treino": sobreposicao_teste_xy_no_treino,
            "sobreposicao_teste_x_no_treino": sobreposicao_teste_x_no_treino,
            "politica_adotada": "Preservacao integral (sem remocao automatica de duplicatas)",
        },
        "distribuicao_classes": distribuicao_classes,
        "distribuicao_sensiveis": distribuicao_sensiveis,
        "estatisticas_numericas": estatisticas_numericas,
        "extremos_censo": extremos_censo,
        "analise_fnlwgt": analise_fnlwgt,
        "categorias_raras": categorias_raras,
    }
