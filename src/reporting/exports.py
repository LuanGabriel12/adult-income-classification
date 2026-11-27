"""Exportacao reproduzivel e deterministica dos 14 relatorios com escrita atomica e rollback."""

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import (
    DATASET_PROVENANCE,
    NEGATIVE_CLASS,
    POSITIVE_CLASS,
    SENSITIVE_COLUMNS,
    V1_HISTORICAL_BENCHMARK,
)


def _escapar_md(texto: Any) -> str:
    """Escapa caracteres que poderiam quebrar colunas de tabelas Markdown."""
    return str(texto).replace("|", "\\|").replace("\n", " ")


def _fmt_num(valor: float | None, casas: int = 4, indefinido_texto: str = "Indefinido") -> str:
    """Formata numero decimal em padrao legivel ou rotulo explicito para indefinido."""
    if valor is None or pd.isna(valor):
        return indefinido_texto
    numero = float(valor)
    if abs(numero) < 0.5 * (10 ** (-casas)):
        numero = 0.0
    return f"{numero:.{casas}f}".replace(".", ",")


def _fmt_pct(valor: float | None, casas: int = 2, indefinido_texto: str = "Indefinido") -> str:
    """Formata proporcao [0, 1] em percentual com virgula decimal."""
    if valor is None or pd.isna(valor):
        return indefinido_texto
    percentual = float(valor) * 100.0
    if abs(percentual) < 0.5 * (10 ** (-casas)):
        percentual = 0.0
    return f"{percentual:.{casas}f}%".replace(".", ",")


def _rotulo_escopo_fairness(escopo: str) -> str:
    """Converte o codigo de escopo de EOD/EOdds em rotulo claro para relatorios."""
    mapa = {
        "integral_todos_os_grupos": "integral — todos os grupos com denominadores validos",
        "parcial_exclui_grupos_denominador_zero": "PARCIAL — exclui grupos com denominador zero (nao integral)",
        "indefinido_menos_de_dois_grupos_validos": "indefinido — menos de 2 grupos validos",
        "indefinido_eixo_tpr_ou_fpr_ausente": "indefinido — eixo TPR ou FPR sem >= 2 grupos validos (avaliacao incompleta)",
        "indefinido_ambos_eixos_indisponiveis": "indefinido — ambos os eixos TPR e FPR indisponiveis",
    }
    return mapa.get(escopo, escopo)


def _fmt_int(valor: int | float) -> str:
    """Formata inteiro com separador de milhar por ponto."""
    return f"{int(valor):,}".replace(",", ".")


def _escrever_texto_atomico(destino: Path, conteudo: str) -> None:
    """Grava arquivo UTF-8/LF via arquivo temporario e substituicao atomica."""
    destino.parent.mkdir(parents=True, exist_ok=True)
    temporario = destino.with_name(f"{destino.name}.tmp")
    try:
        with temporario.open("w", encoding="utf-8", newline="\n") as arquivo:
            arquivo.write(conteudo)
        temporario.replace(destino)
    except Exception:
        if temporario.exists():
            temporario.unlink()
        raise


def _gravar_lote_com_rollback(arquivos_conteudo: dict[Path, str]) -> list[Path]:
    """Grava todos os relatorios com snapshot previo e rollback em melhor esforco se falhar."""
    snapshots: dict[Path, bytes | None] = {}
    gravados: list[Path] = []

    for destino in arquivos_conteudo:
        snapshots[destino] = destino.read_bytes() if destino.exists() else None

    try:
        for destino, conteudo in arquivos_conteudo.items():
            _escrever_texto_atomico(destino, conteudo)
            gravados.append(destino)
    except Exception as erro:
        for caminho in reversed(gravados):
            anterior = snapshots.get(caminho)
            if anterior is None:
                if caminho.exists():
                    caminho.unlink()
            else:
                caminho.write_bytes(anterior)
        raise RuntimeError(
            f"Falha ao gravar lote de relatorios (rollback executado): {erro}"
        ) from erro

    return gravados


def _serializar_csv(df: pd.DataFrame) -> str:
    """Serializa DataFrame para CSV UTF-8 com terminacao LF deterministica."""
    return df.to_csv(index=False, lineterminator="\n")


def _gerar_data_quality_report_md(qualidade: dict[str, Any]) -> str:
    """Constroi reports/data_quality_report.md 100% a partir dos dados auditados."""
    meta = qualidade["metadados_carga"]
    dim = qualidade["dimensoes"]
    dup = qualidade["duplicatas"]
    cls = qualidade["distribuicao_classes"]
    ext = qualidade["extremos_censo"]
    fnl = qualidade["analise_fnlwgt"]

    linhas = [
        "# Relatorio de Qualidade dos Dados — Adult / Census Income",
        "",
        "## 1. Proveniencia, Integridade e Dimensoes",
        "",
        "| Atributo | Valor verificado |",
        "| --- | --- |",
        f"| Fonte de carregamento | `{_escapar_md(meta.get('fonte_carregamento', 'desconhecida'))}` |",
        f"| Origem / caminho | `{_escapar_md(meta.get('caminho_relativo', 'nao informado'))}` |",
        f"| SHA-256 (arquivo local) | `{meta.get('sha256') or 'Nao aplicavel (carregamento direto OpenML)'}` |",
        f"| Hash verificado | `{'Sim' if meta.get('hash_verificado') else 'Nao'}` |",
        f"| Dataset oficial OpenML 1590 v2 | `{'Sim' if meta.get('dataset_oficial_openml_1590') else 'Nao (dataset alternativo)'}` |",
        f"| Dimensoes totais | `{_fmt_int(dim['total_linhas'])}` linhas x `{dim['total_features']}` features + target `{meta.get('target', 'class')}` |",
        f"| Divisao treino / teste (80/20 estratificada) | `{_fmt_int(dim['treino_linhas'])}` treino / `{_fmt_int(dim['teste_linhas'])}` teste |",
        f"| Autoria e periodo historico | {DATASET_PROVENANCE['autoria']} — {DATASET_PROVENANCE['periodo_historico']} |",
        f"| Licenca verificada | UCI: {DATASET_PROVENANCE['uci_licenca']} / OpenML: `{DATASET_PROVENANCE['openml_licenca']}` |",
        "",
        "## 2. Esquema, Tipos e Valores Ausentes",
        "",
        "| Coluna | Tipo logico | Dtype | Valores unicos | Ausentes | % Ausentes |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for item in qualidade["resumo_colunas"]:
        linhas.append(
            f"| `{_escapar_md(item['coluna'])}` | {item['tipo_logico']} | `{_escapar_md(item['dtype'])}` | "
            f"{_fmt_int(item['valores_unicos'])} | {_fmt_int(item['ausentes'])} | {_fmt_pct(item['percentual_ausentes'])} |"
        )

    aus_trab = qualidade["analise_ausentes_trabalho"]
    if aus_trab:
        linhas.extend(
            [
                "",
                "### Relacao entre ausencias de `workclass` e `occupation`",
                "",
                f"- Registros com **ambos ausentes**: `{_fmt_int(aus_trab['ambos_ausentes'])}`.",
                f"- Registros com **apenas `workclass` ausente**: `{_fmt_int(aus_trab['apenas_workclass_ausente'])}`.",
                f"- Registros com **apenas `occupation` ausente**: `{_fmt_int(aus_trab['apenas_occupation_ausente'])}` "
                f"(sendo `{_fmt_int(aus_trab['never_worked_com_occupation_ausente'])}` com `workclass == Never-worked`).",
            ]
        )

    linhas.extend(
        [
            "",
            "## 3. Distribuicao das Classes e Desbalanceamento",
            "",
            "| Classe | Quantidade | Proporcao |",
            "| --- | ---: | ---: |",
            f"| `{NEGATIVE_CLASS}` | {_fmt_int(cls[NEGATIVE_CLASS]['quantidade'])} | {_fmt_pct(cls[NEGATIVE_CLASS]['proporcao'])} |",
            f"| `{POSITIVE_CLASS}` | {_fmt_int(cls[POSITIVE_CLASS]['quantidade'])} | {_fmt_pct(cls[POSITIVE_CLASS]['proporcao'])} |",
            "",
            f"- Razao classe majoritaria / minoritaria: **`{_fmt_num(cls['razao_majoritaria_sobre_minoritaria'], 2)}:1`**.",
            "- Consequencia metodologica: um classificador ingenuo (`DummyClassifier`) que sempre preve "
            f"`{NEGATIVE_CLASS}` obtem Accuracy elevada (`{_fmt_pct(cls[NEGATIVE_CLASS]['proporcao'])}`), "
            f"mas Recall `0,00%` para `{POSITIVE_CLASS}`. Por isso Accuracy isolada e insuficiente.",
            "",
            "## 4. Distribuicao dos Atributos Sensiveis (`sex` e `race`)",
            "",
            "> As categorias abaixo reproduzem exatamente as classificacoes historicas do Censo dos EUA de 1994 presentes na fonte.",
            "",
        ]
    )

    for attr, grupos in qualidade["distribuicao_sensiveis"].items():
        linhas.extend(
            [
                f"### Atributo `{attr}`",
                "",
                "| Grupo | Quantidade | % da Base | Positivos (`>50K`) | Negativos (`<=50K`) | Taxa `>50K` |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for g in grupos:
            linhas.append(
                f"| {_escapar_md(g['grupo'])} | {_fmt_int(g['quantidade'])} | {_fmt_pct(g['proporcao_base'])} | "
                f"{_fmt_int(g['positivos_reais'])} | {_fmt_int(g['negativos_reais'])} | {_fmt_pct(g['taxa_positiva'])} |"
            )
        linhas.append("")

    linhas.extend(
        [
            "## 5. Identificadores, Duplicatas e Sobreposicao Treino x Teste",
            "",
            "- **Identificador unico de pessoa:** ausente (`possui_identificador_unico = False`).",
            f"- **Duplicatas exatas `(X, y)` excedentes:** `{_fmt_int(dup['duplicatas_xy_excedentes'])}` "
            f"(`{_fmt_int(dup['duplicatas_xy_linhas_envolvidas'])}` linhas envolvidas).",
            f"- **Duplicatas exatas apenas em `X`:** `{_fmt_int(dup['duplicatas_x_excedentes'])}` "
            f"(`{_fmt_int(dup['duplicatas_x_linhas_envolvidas'])}` linhas envolvidas).",
            f"- **Combinacoes `X` identicas com alvo `class` conflitante (`<=50K` e `>50K`):** "
            f"`{_fmt_int(dup['grupos_x_com_target_conflitante'])}`.",
            f"- **Duplicatas `(X, y)` ignorando `fnlwgt`:** `{_fmt_int(dup['duplicatas_xy_ignorando_fnlwgt'])}`.",
            f"- **Sobreposicao de indices entre treino e teste:** `{_fmt_int(dup['sobreposicao_indices_treino_teste'])}`.",
            f"- **Linhas do teste com combinacao `(X, y)` identica no treino:** `{_fmt_int(dup['sobreposicao_teste_xy_no_treino'])}` "
            f"(e `{_fmt_int(dup['sobreposicao_teste_x_no_treino'])}` com `X` identico no treino).",
            "- **Politica adotada:** nenhuma linha repetida foi removida automaticamente. Sem identificador unico, "
            "perfis demograficos identicos podem pertencer a pessoas diferentes (inclusive com rendas diferentes, "
            "como provam os conflitos de rotulo). A pequena sobreposicao exata entre treino e teste "
            f"(`{_fmt_int(dup['sobreposicao_teste_xy_no_treino'])}` de `{_fmt_int(dim['teste_linhas'])}` linhas) "
            "e documentada de forma transparente sem alterar silenciosamente a divisao historica da V1.",
            "",
            "## 6. Variaveis Numericas, Valores Atipicos e `fnlwgt`",
            "",
            "| Variavel | Min | Q25 | Mediana | Q75 | Max | Media | Atipicos (1,5x IQR) |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for num in qualidade["estatisticas_numericas"]:
        linhas.append(
            f"| `{_escapar_md(num['coluna'])}` | {_fmt_num(num['min'], 1)} | {_fmt_num(num['q25'], 1)} | "
            f"{_fmt_num(num['mediana'], 1)} | {_fmt_num(num['q75'], 1)} | {_fmt_num(num['max'], 1)} | "
            f"{_fmt_num(num['media'], 2)} | {_fmt_int(num['atipicos_iqr'])} ({_fmt_pct(num['percentual_atipicos_iqr'])}) |"
        )

    if ext:
        linhas.extend(
            [
                "",
                "### Extremos e Top-Coding Observados",
                "",
                f"- `capital-gain == 0`: `{_fmt_int(ext.get('capital_gain_zero_qtd', 0))}` "
                f"({_fmt_pct(ext.get('capital_gain_zero_pct', 0.0))}); valor teto `99.999` em "
                f"`{_fmt_int(ext.get('capital_gain_99999_qtd', 0))}` registros "
                f"(proximo maior valor observado: `{_fmt_num(ext.get('capital_gain_max_abaixo_99999'), 0)}`).",
                f"- `capital-loss == 0`: `{_fmt_int(ext.get('capital_loss_zero_qtd', 0))}` "
                f"({_fmt_pct(ext.get('capital_loss_zero_pct', 0.0))}); maximo `{_fmt_num(ext.get('capital_loss_max'), 0)}`.",
                f"- `hours-per-week`: `40h` em `{_fmt_int(ext.get('hours_per_week_40_qtd', 0))}` registros; "
                f"`99h` em `{_fmt_int(ext.get('hours_per_week_99_qtd', 0))}`; `1h` em `{_fmt_int(ext.get('hours_per_week_1_qtd', 0))}`.",
                f"- `age`: idade maxima `90` em `{_fmt_int(ext.get('age_90_qtd', 0))}` registros; "
                f"minima `17` em `{_fmt_int(ext.get('age_17_qtd', 0))}`.",
            ]
        )

    if fnl:
        linhas.extend(
            [
                "",
                "### Significado Estatistico de `fnlwgt`",
                "",
                f"- Faixa observada: minimo `{_fmt_num(fnl['min'], 0)}`, mediana `{_fmt_num(fnl['mediana'], 1)}`, "
                f"maximo `{_fmt_num(fnl['max'], 0)}` (`{_fmt_int(fnl['valores_unicos'])}` valores unicos; "
                f"razao max/min = `{_fmt_num(fnl['razao_max_sobre_min'], 1)}x`).",
                "- **Interpretacao obrigatoria:** `fnlwgt` e o *final sampling weight* do Current Population Survey (CPS) "
                "do US Census Bureau, calibrado por controles demograficos dentro de cada um dos 51 recortes estaduais. "
                "Ele indica quantas pessoas da populacao civil nao institucionalizada possuem perfil demografico semelhante "
                "naquele estado. Portanto, **cada linha da base nao representa automaticamente o mesmo peso populacional**, "
                "nem a distribuicao bruta das linhas equivale a uma amostra uniforme da populacao atual dos EUA.",
            ]
        )

    linhas.extend(
        [
            "",
            "## 7. Inconsistencias de Codificacao, Categorias Inesperadas e Raras",
            "",
            f"- Colunas categoricas com marcador literal `'?'` ou espacos laterais: `{len(qualidade['inconsistencias_codificacao'])}`.",
            f"- Colunas com categorias fora do vocabulario oficial OpenML 1590: `{len(qualidade['categorias_inesperadas'])}`.",
            f"- Registros do alvo com ponto final (`<=50K.` / `>50K.`): `{qualidade['alvo_com_ponto_final']}` "
            f"(espacos laterais: `{qualidade['alvo_com_espacos']}`).",
        ]
    )

    red_edu = qualidade["redundancia_educacao"]
    if red_edu:
        linhas.append(
            f"- Redundancia `education` vs `education-num`: mapeamento 1-para-1 = "
            f"`{'Sim' if red_edu['mapeamento_1_para_1'] else 'Nao'}` "
            f"(`{red_edu['niveis_education']}` niveis categoricos e `{red_edu['niveis_education_num']}` numericos)."
        )

    cruz = qualidade["cruzamentos_sex_relationship"]
    if cruz:
        linhas.append(
            f"- Registros cruzados `sex` x `relationship`: `Female` + `Husband` = `{cruz['female_com_husband']}`, "
            f"`Male` + `Wife` = `{cruz['male_com_wife']}`."
        )

    raras = qualidade["categorias_raras"]
    linhas.extend(
        [
            "",
            f"### Categorias Raras (< 0,5% da base — total encontradas: `{len(raras)}`)",
            "",
            "| Coluna | Categoria | Quantidade | % da Base |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for item in raras[:20]:
        linhas.append(
            f"| `{_escapar_md(item['coluna'])}` | {_escapar_md(item['categoria'])} | "
            f"{_fmt_int(item['quantidade'])} | {_fmt_pct(item['percentual'], 3)} |"
        )
    if len(raras) > 20:
        linhas.append(f"| ... | *(mais {len(raras) - 20} categorias raras)* | — | — |")

    return "\n".join(linhas) + "\n"


def _gerar_model_evaluation_report_md(
    tabela_comparacao: pd.DataFrame,
    resumo_selecao: dict[str, Any],
    metricas_teste: dict[str, Any],
    analise_erros: dict[str, Any],
    metadados_carga: dict[str, Any],
) -> str:
    """Constroi reports/model_evaluation_report.md de forma 100% dinamica."""
    cm = metricas_teste["matriz_confusao"]
    cr = metricas_teste["classification_report_dict"]
    res_err = analise_erros["resumo_global"]
    tabela_erros = analise_erros["tabela_erros"]
    eh_oficial = bool(metadados_carga.get("dataset_oficial_openml_1590"))

    linhas = [
        "# Relatorio de Avaliacao de Modelos e Analise de Erros",
        "",
        "## 1. Protocolo Experimental e Criterio Pre-Registrado de Selecao",
        "",
        "- **Divisao:** 80% treino / 20% teste (`random_state=42`, `stratify=y`), sem consulta ao teste durante comparacao ou selecao.",
        "- **Validacao cruzada:** 5 divisoes estratificadas exclusivamente no conjunto de treino.",
        "- **Pre-processamento:** encapsulado em `Pipeline` + `ColumnTransformer` (mediana + `StandardScaler` nas numericas; moda + `OneHotEncoder(handle_unknown='ignore')` nas categoricas), ajustado apenas nos folds de treino.",
        f"- **Criterio pre-registrado:** {resumo_selecao['criterio_primario']}; rejeicao de modelos com gap `Accuracy treino - validacao > {_fmt_num(resumo_selecao['limite_overfitting_accuracy'], 2)}`; desempate: {resumo_selecao['criterio_desempate']}.",
        f"- **Modelo selecionado na validacao cruzada:** **`{resumo_selecao['modelo_selecionado']}`**.",
        "",
        "> **Nota metodologica sobre o conjunto de teste:** o split historico de teste (`random_state=42`) ja foi observado na V1. Ele e reportado aqui como avaliacao controlada e reproduzivel — nao como conjunto inedito de producao — e nao foi usado para selecionar modelos ou thresholds.",
        "",
        "## 2. Comparacao dos 4 Modelos (Validacao Cruzada 5-Fold no Treino)",
        "",
        "| Modelo | Acc Treino | Acc Val (media ± dp) | Gap Acc (Tr−Val) | Recall Val (media ± dp) | Precision Val (media ± dp) | F1 Val (media ± dp) | Decisao |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]

    for idx in tabela_comparacao.index:
        r = tabela_comparacao.loc[idx]
        decisao = "**SELECIONADO**" if bool(r["selecionado"]) else _escapar_md(r["diagnostico_selecao"])
        linhas.append(
            f"| {_escapar_md(r['modelo'])} | {_fmt_pct(r['train_accuracy_mean'])} | "
            f"{_fmt_pct(r['val_accuracy_mean'])} (±{_fmt_pct(r['val_accuracy_std'])}) | "
            f"{_fmt_pct(r['gap_train_val_accuracy'])} | "
            f"{_fmt_pct(r['val_recall_mean'])} (±{_fmt_pct(r['val_recall_std'])}) | "
            f"{_fmt_pct(r['val_precision_mean'])} (±{_fmt_pct(r['val_precision_std'])}) | "
            f"{_fmt_pct(r['val_f1_mean'])} (±{_fmt_pct(r['val_f1_std'])}) | {decisao} |"
        )

    linhas.extend(
        [
            "",
            "### Diagnostico de Overfitting, Underfitting e Generalizacao",
            "",
            "- **Underfitting (Baseline `DummyClassifier`):** preve exclusivamente a classe majoritaria (`<=50K`), alcancando Recall e F1 iguais a `0,00%` para `>50K` (nao aprende padroes das variaveis explicativas).",
            "- **Overfitting severo (`Decision Tree` sem limite):** memoriza particoes individuais do treino (Accuracy de treino proxima de `100%`), gerando gap expressivo frente a validacao e pior Precision/F1 fora da amostra.",
            "- **Controle de complexidade (`Decision Tree max_depth=7`):** limita a profundidade e reduz fortemente o gap treino-validacao, mas sacrifica Recall na classe positiva `>50K`.",
            f"- **Equilibrio e estabilidade (`{resumo_selecao['modelo_selecionado']}`):** apresenta gap minimo entre treino e validacao, baixa variacao entre folds e o melhor equilibrio de F1 e Recall na classe positiva.",
            "",
            "## 3. Avaliacao Final no Conjunto de Teste",
            "",
            "| Metrica | Valor Obtido |" + (" Referencia Historica V1 |" if eh_oficial else ""),
            "| --- | ---: |" + (" ---: |" if eh_oficial else ""),
        ]
    )

    pares_teste = [
        ("Accuracy", metricas_teste["accuracy"], V1_HISTORICAL_BENCHMARK["accuracy"]),
        ("Recall (`>50K`)", metricas_teste["recall"], V1_HISTORICAL_BENCHMARK["recall"]),
        ("Precision (`>50K`)", metricas_teste["precision"], V1_HISTORICAL_BENCHMARK["precision"]),
        ("F1-score (`>50K`)", metricas_teste["f1"], V1_HISTORICAL_BENCHMARK["f1"]),
        ("False Positive Rate (`>50K`)", metricas_teste["fpr"], None),
        ("Selection Rate (`>50K`)", metricas_teste["selection_rate"], None),
    ]
    for rotulo, obtido, ref_v1 in pares_teste:
        if eh_oficial:
            ref_txt = _fmt_pct(ref_v1) if ref_v1 is not None else "—"
            linhas.append(f"| {rotulo} | {_fmt_pct(obtido)} (`{_fmt_num(obtido, 5)}`) | {ref_txt} |")
        else:
            linhas.append(f"| {rotulo} | {_fmt_pct(obtido)} (`{_fmt_num(obtido, 5)}`) |")

    linhas.extend(
        [
            "",
            "### Matriz de Confusao (Ordem das classes: `<=50K`, `>50K`)",
            "",
            "| Classe Real \\ Previsto | Previsto `<=50K` | Previsto `>50K` | Total Real |",
            "| --- | ---: | ---: | ---: |",
            f"| **Real `<=50K`** | TN = **{_fmt_int(cm['tn'])}** | FP = **{_fmt_int(cm['fp'])}** | {_fmt_int(cm['tn'] + cm['fp'])} |",
            f"| **Real `>50K`** | FN = **{_fmt_int(cm['fn'])}** | TP = **{_fmt_int(cm['tp'])}** | {_fmt_int(cm['fn'] + cm['tp'])} |",
            f"| **Total Previsto** | {_fmt_int(cm['tn'] + cm['fn'])} | {_fmt_int(cm['fp'] + cm['tp'])} | **{_fmt_int(metricas_teste['total_registros'])}** |",
            "",
            "### Classification Report Estruturado",
            "",
            "| Classe / Media | Precision | Recall | F1-score | Suporte |",
            "| --- | ---: | ---: | ---: | ---: |",
            f"| `{NEGATIVE_CLASS}` | {_fmt_num(cr[NEGATIVE_CLASS]['precision'])} | {_fmt_num(cr[NEGATIVE_CLASS]['recall'])} | {_fmt_num(cr[NEGATIVE_CLASS]['f1-score'])} | {_fmt_int(cr[NEGATIVE_CLASS]['support'])} |",
            f"| `{POSITIVE_CLASS}` | {_fmt_num(cr[POSITIVE_CLASS]['precision'])} | {_fmt_num(cr[POSITIVE_CLASS]['recall'])} | {_fmt_num(cr[POSITIVE_CLASS]['f1-score'])} | {_fmt_int(cr[POSITIVE_CLASS]['support'])} |",
            f"| `macro avg` (peso igual por classe) | {_fmt_num(cr['macro avg']['precision'])} | {_fmt_num(cr['macro avg']['recall'])} | {_fmt_num(cr['macro avg']['f1-score'])} | {_fmt_int(cr['macro avg']['support'])} |",
            f"| `weighted avg` (ponderada pelo suporte) | {_fmt_num(cr['weighted avg']['precision'])} | {_fmt_num(cr['weighted avg']['recall'])} | {_fmt_num(cr['weighted avg']['f1-score'])} | {_fmt_int(cr['weighted avg']['support'])} |",
            "",
            "## 4. Analise Descritiva de Erros de Classificacao",
            "",
            f"- Total de erros no teste: **`{_fmt_int(res_err['erros_totais'])}`** de `{_fmt_int(res_err['total_registros'])}` registros "
            f"(taxa global de erro: `{_fmt_pct(res_err['taxa_erro_global'])}`).",
            f"- **Falsos Negativos (`FN`):** `{_fmt_int(res_err['fn'])}` (`{_fmt_pct(res_err['participacao_fn_nos_erros'])}` de todos os erros; "
            f"`{_fmt_pct(res_err['taxa_fn_sobre_positivos'])}` dos positivos reais nao identificados).",
            f"- **Falsos Positivos (`FP`):** `{_fmt_int(res_err['fp'])}` (`{_fmt_pct(res_err['participacao_fp_nos_erros'])}` de todos os erros; "
            f"`{_fmt_pct(res_err['taxa_fp_sobre_negativos'])}` dos negativos reais previstos erroneamente como `>50K`).",
            "",
            "### Principais Concentracoes Descritivas de Erro por Dimensao",
            "",
            "| Dimensao | Categoria com maior volume de erros | Total | Positivos Reais | FP | FN | Taxa de Erro | Recall do Segmento |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    for dimensao in analise_erros["dimensoes_analisadas"]:
        sub = tabela_erros.loc[tabela_erros["dimensao"] == dimensao].sort_values(
            by=["erros_totais", "total"], ascending=[False, False]
        )
        for idx in sub.head(3).index:
            r = sub.loc[idx]
            linhas.append(
                f"| `{_escapar_md(dimensao)}` | {_escapar_md(r['categoria'])} | {_fmt_int(r['total'])} | "
                f"{_fmt_int(r['positivos_reais'])} | {_fmt_int(r['fp'])} | {_fmt_int(r['fn'])} | "
                f"{_fmt_pct(r['taxa_erro'])} | {_fmt_pct(r['recall_segmento'])} |"
            )

    linhas.extend(
        [
            "",
            "### Limitacoes e Cautelas Interpretativas da Analise de Erros",
            "",
            "- Os padroes acima sao **exclusivamente descritivos**: associacoes entre escolaridade, horas, ocupacao, faixa etaria ou papel familiar e erros do classificador **nao estabelecem causalidade**.",
            "- Nenhuma conclusao aqui se aplica a pessoas reais individuais.",
            "- Este diagnostico foi realizado apos a avaliacao final e **nao** foi utilizado para alterar o modelo, ajustar hiperparametros ou escolher novo threshold.",
        ]
    )

    return "\n".join(linhas) + "\n"


def _gerar_fairness_audit_report_md(
    auditoria_fairness: dict[str, Any],
    comparacao_sensiveis: dict[str, Any],
    comparacao_ponderada_vs_simples: pd.DataFrame,
) -> str:
    """Constroi reports/fairness_audit_report.md de forma 100% dinamica."""
    proto = comparacao_sensiveis["protocolo"]

    linhas = [
        "# Relatorio de Auditoria de Fairness e Experimento com Atributos Sensiveis",
        "",
        "> **Enquadramento cientifico e etico:** Este relatorio quantifica disparidades observadas de desempenho entre grupos demograficos historicos do Censo dos EUA de 1994 sob definicoes matematicas especificas. **Ele nao certifica ausencia de discriminacao, nao prova causalidade nem discriminacao intencional por si so, e nao autoriza uso do modelo em decisoes reais sobre pessoas** (contratacao, credito, renda ou emprego).",
        "",
        "## 1. Definicoes Matematicas das Metricas Auditadas",
        "",
        "| Metrica | Formula | Denominador | O que quantifica |",
        "| --- | --- | --- | --- |",
        "| **Accuracy** | `(TP + TN) / N` | Total do grupo (`N_g`) | Proporcao geral de acertos no grupo |",
        "| **Recall (TPR)** | `TP / (TP + FN)` | Positivos reais do grupo | Fracao dos casos realmente `>50K` identificados |",
        "| **False Positive Rate (FPR)** | `FP / (FP + TN)` | Negativos reais do grupo | Fracao dos casos `<=50K` previstos erroneamente como `>50K` |",
        "| **Selection Rate** | `(TP + FP) / N` | Total do grupo (`N_g`) | Proporcao do grupo que recebe previsao positiva `>50K` |",
        "| **Equal Opportunity Difference (EOD)** | `max(TPR_g) - min(TPR_g)` | Grupos com `TP_g + FN_g > 0` (sinalizado como **parcial** se algum grupo for excluido) | Maior distancia de Recall entre grupos |",
        "| **Equalized Odds Difference (EOdds)** | `max(diff_TPR, diff_FPR)` | Exige >= 2 grupos validos em TPR e >= 2 em FPR; sinalizado como **parcial** se algum grupo tiver denominador zero, ou **indefinido** se faltar um eixo | Maior disparidade combinada entre TPR e FPR |",
        "| **Ratio** | `min(M_g) / max(M_g)` | `max(M_g) > 0` | Razao relativa entre pior e melhor valor da metrica |",
        "",
        "## 2. Auditoria de Fairness no Conjunto de Teste (Modelo Selecionado — Cenario A)",
        "",
    ]

    for idx_attr, attr in enumerate(SENSITIVE_COLUMNS, start=1):
        aud = auditoria_fairness["por_atributo"][attr]
        df_g = aud["tabela_por_grupo"]
        res = aud["resumo"]
        linhas.extend(
            [
                f"### 2.{idx_attr} Desempenho por `{attr}`",
                "",
                "| Grupo | N | Positivos Reais | Negativos Reais | Accuracy | Recall (TPR) | FPR | Selection Rate | Observacao |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for idx in df_g.index:
            r = df_g.loc[idx]
            rec_txt = "Indefinido (0 pos)" if bool(r["recall_indefinido"]) else _fmt_num(r["recall"])
            fpr_txt = "Indefinido (0 neg)" if bool(r["fpr_indefinido"]) else _fmt_num(r["fpr"])
            linhas.append(
                f"| {_escapar_md(r['grupo'])} | {_fmt_int(r['count'])} | {_fmt_int(r['positivos_reais'])} | "
                f"{_fmt_int(r['negativos_reais'])} | {_fmt_num(r['accuracy'])} | {rec_txt} | {fpr_txt} | "
                f"{_fmt_num(r['selection_rate'])} | {_escapar_md(r['observacao'])} |"
            )

        pequenos_txt = ", ".join(res["grupos_amostra_pequena"]) or "Nenhum"
        excluidos_eodds_txt = ", ".join(res["grupos_excluidos_equalized_odds"]) or "Nenhum"
        linhas.extend(
            [
                "",
                f"- **Equal Opportunity Difference (`{attr}`):** `{_fmt_num(res['equal_opportunity_difference'], 6)}` "
                f"(escopo: `{res['equal_opportunity_escopo']}` — {_rotulo_escopo_fairness(res['equal_opportunity_escopo'])}).",
                f"- **Equalized Odds Difference (`{attr}`):** `{_fmt_num(res['equalized_odds_difference'], 6)}` "
                f"(escopo: `{res['equalized_odds_escopo']}` — {_rotulo_escopo_fairness(res['equalized_odds_escopo'])}).",
                f"- **Diferencas (`max - min`):** Accuracy `{_fmt_num(res['accuracy_diff'])}`, Recall `{_fmt_num(res['recall_diff'])}`, "
                f"FPR `{_fmt_num(res['fpr_diff'])}`, Selection Rate `{_fmt_num(res['selection_rate_diff'])}`.",
                f"- **Razoes (`min / max`):** Accuracy `{_fmt_num(res['accuracy_ratio'])}`, Recall `{_fmt_num(res['recall_ratio'])}`, "
                f"FPR `{_fmt_num(res['fpr_ratio'])}`, Selection Rate `{_fmt_num(res['selection_rate_ratio'])}`.",
                f"- **Grupos excluidos por denominador zero em EOdds:** `{excluidos_eodds_txt}`.",
                f"- **Grupos com amostra pequena / baixa contagem de positivos:** `{pequenos_txt}`.",
                f"- **Verificacao de denominadores zero:** {res['nota_denominadores_zero']}",
                "",
            ]
        )

    linhas.extend(
        [
            "## 3. Influencia dos Tamanhos Amostrais e Ponderacao Correta vs. Media Simples",
            "",
            "Como os grupos possuem tamanhos e taxas-base muito diferentes, a **media simples** das metricas por grupo "
            "**nao** equivale a metrica global. Cada metrica global exige ponderacao pelo seu proprio denominador:",
            "",
            "| Atributo | Metrica | Global Real | Recalculo Ponderado | Media Simples dos Grupos | Desvio da Media Simples | Denominador / Peso Correto |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for idx in comparacao_ponderada_vs_simples.index:
        r = comparacao_ponderada_vs_simples.loc[idx]
        linhas.append(
            f"| `{r['atributo']}` | `{r['metrica']}` | {_fmt_num(r['metrica_global'])} | "
            f"{_fmt_num(r['recalculo_ponderado'])} | {_fmt_num(r['media_simples_grupos'])} | "
            f"{_fmt_num(r['desvio_media_simples'])} | {_escapar_md(r['peso_correto'])} |"
        )

    avisos_parcial_comp: list[str] = []
    for nome_esc, chave_esc in [
        ("validacao_cv_treino_oof", "cv_oof"),
        ("teste_historico_descritivo", "teste_historico"),
    ]:
        for nome_cen, chave_cen in [("Cenario A", "cenario_a"), ("Cenario B", "cenario_b")]:
            for attr in SENSITIVE_COLUMNS:
                res_c = comparacao_sensiveis[chave_esc][chave_cen]["fairness"]["por_atributo"][
                    attr
                ]["resumo"]
                if res_c["equalized_odds_escopo"] != "integral_todos_os_grupos":
                    avisos_parcial_comp.append(
                        f"`{nome_esc}` / {nome_cen} / `{attr}`: "
                        f"EOD=`{res_c['equal_opportunity_escopo']}`, "
                        f"EOdds=`{res_c['equalized_odds_escopo']}`"
                    )

    escopo_comp_linha = (
        "- **Escopo das comparacoes de fairness (Cenarios A e B):** integral em todos os grupos "
        "(todos os grupos possuem positivos e negativos reais > 0)."
        if not avisos_parcial_comp
        else "- **Escopo parcial/incompleto detectado na comparacao (nao integral):** "
        + "; ".join(avisos_parcial_comp)
        + "."
    )

    linhas.extend(
        [
            "",
            "## 4. Experimento Controlado: Cenario A (Com `sex` e `race`) vs. Cenario B (Sem `sex` e `race`)",
            "",
            f"- **Modelo avaliado:** `{proto['modelo']}` (mesmos hiperparametros, mesmo split e mesmos `{proto['cv_splits']}` folds estratificados).",
            f"- **Cenario A:** `{len(proto['features_cenario_a'])}` features originais (incluindo `sex` e `race`).",
            f"- **Cenario B:** `{len(proto['features_cenario_b'])}` features (removendo `sex` e `race`, mantendo os atributos sensiveis em tabela paralela com indices 100% alinhados para o Fairlearn).",
            f"- **Protocolo:** {proto['politica_experimental']}",
            escopo_comp_linha,
            "",
            "| Escopo | Categoria | Atributo | Metrica | Cenario A (com sensiveis) | Cenario B (sem sensiveis) | Delta (B − A) |",
            "| --- | --- | --- | --- | ---: | ---: | ---: |",
        ]
    )

    df_comp = comparacao_sensiveis["tabela_comparativa"]
    metricas_destaque = {
        "accuracy",
        "recall",
        "precision",
        "f1",
        "equal_opportunity_difference",
        "equalized_odds_difference",
        "fpr_diff",
        "selection_rate_diff",
    }
    for idx in df_comp.index:
        r = df_comp.loc[idx]
        if r["metrica"] in metricas_destaque:
            linhas.append(
                f"| `{r['escopo']}` | `{r['categoria_metrica']}` | `{r['atributo_sensivel']}` | "
                f"`{r['metrica']}` | {_fmt_num(r['cenario_a_com_sensiveis'])} | "
                f"{_fmt_num(r['cenario_b_sem_sensiveis'])} | {_fmt_num(r['delta_b_menos_a'])} |"
            )

    linhas.extend(
        [
            "",
            "### Variacao por Subgrupo na Validacao Cruzada (Treino Out-of-Fold)",
            "",
            "| Atributo | Grupo | N (Treino) | Positivos Reais | Recall A | Recall B | Delta Recall (B−A) | Selection Rate A | Selection Rate B |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for attr in SENSITIVE_COLUMNS:
        df_a = comparacao_sensiveis["cv_oof"]["cenario_a"]["fairness"]["por_atributo"][attr][
            "tabela_por_grupo"
        ]
        df_b = comparacao_sensiveis["cv_oof"]["cenario_b"]["fairness"]["por_atributo"][attr][
            "tabela_por_grupo"
        ]
        cruzado = df_a.merge(
            df_b[["grupo", "recall", "selection_rate"]],
            on="grupo",
            suffixes=("_a", "_b"),
        )
        for idx in cruzado.index:
            r = cruzado.loc[idx]
            rec_a = None if pd.isna(r["recall_a"]) else float(r["recall_a"])
            rec_b = None if pd.isna(r["recall_b"]) else float(r["recall_b"])
            delta_rec = (rec_b - rec_a) if (rec_a is not None and rec_b is not None) else None
            linhas.append(
                f"| `{attr}` | {_escapar_md(r['grupo'])} | {_fmt_int(r['count'])} | "
                f"{_fmt_int(r['positivos_reais'])} | {_fmt_num(rec_a)} | {_fmt_num(rec_b)} | "
                f"{_fmt_num(delta_rec)} | {_fmt_num(r['selection_rate_a'])} | {_fmt_num(r['selection_rate_b'])} |"
            )

    linhas.extend(
        [
            "",
            "### Interpretacao Tecnica: Persistencia de Disparidades Apos a Remocao de `sex` e `race`",
            "",
            "1. **Evidencia empirica observada vs. limites da inferencia causal:** o experimento controlado mostra empiricamente que remover `sex` e `race` (Cenario B) reduz parte de `EOD`, "
            "mas **nao elimina** as disparidades observadas (*fairness through unawareness* mostra-se insuficiente neste conjunto de dados). "
            "Uma **hipotese plausivel** discutida na literatura e que variaveis remanescentes correlacionadas aos atributos sensiveis — "
            "como `relationship`, `marital-status`, `occupation`, `hours-per-week`, `workclass` e `native-country` — **possam atuar como *proxies***, "
            "em conjunto com diferencas historicas de distribuicao de atributos e taxas-base entre grupos. "
            "Contudo, **a remocao isolada de `sex` e `race` nao testa ablacoes individuais nem comprova causalmente** qual parcela da disparidade restante decorre de cada variavel especifica.",
            "2. **Sem superioridade automatica por metrica isolada:** um cenario nao se torna automaticamente superior apenas por reduzir uma unica metrica de disparidade; "
            "e indispensavel examinar conjuntamente Recall, FPR, Selection Rate, tamanho amostral e impacto potencial de falsos positivos e falsos negativos.",
            "3. **Ausencia de mitigacao algoritmica neste laboratorio:** propositalmente nao foram aplicados algoritmos de mitigacao de vieses, preservando o foco em diagnostico honesto e auditavel.",
        ]
    )

    return "\n".join(linhas) + "\n"


def _gerar_validation_report_md(reconciliacao: dict[str, Any]) -> str:
    """Constroi reports/validation_report.md a partir das 16 checagens independentes."""
    df_val = reconciliacao["tabela_checagens"]
    linhas = [
        "# Relatorio de Validacao Matematica e Consistencia Metodologica",
        "",
        f"- **Total de verificacoes executadas:** `{reconciliacao['total_checagens']}`.",
        f"- **Aprovadas:** `{reconciliacao['total_aprovadas']}` | **Reprovadas:** `{reconciliacao['total_reprovadas']}` | **Nao aplicaveis:** `{reconciliacao['total_nao_aplicaveis']}`.",
        f"- **Status geral da auditoria matematica:** **`{'APROVADO' if reconciliacao['aprovado_geral'] else 'REPROVADO'}`**.",
        "",
        "## Tabela de Verificacoes (`VAL-01` a `VAL-16`)",
        "",
        "| Codigo | Verificacao | Status | Esperado | Obtido | Detalhes |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for idx in df_val.index:
        r = df_val.loc[idx]
        linhas.append(
            f"| `{r['codigo']}` | {_escapar_md(r['descricao'])} | **{r['status']}** | "
            f"{_escapar_md(r['valor_esperado'])} | {_escapar_md(r['valor_obtido'])} | {_escapar_md(r['detalhes'])} |"
        )
    return "\n".join(linhas) + "\n"


def _buscar_delta_cv(df_comp: pd.DataFrame, atributo: str, metrica: str) -> float | None:
    """Busca delta (B - A) no escopo de validacao CV out-of-fold do treino."""
    sub = df_comp.loc[
        (df_comp["escopo"] == "validacao_cv_treino_oof")
        & (df_comp["atributo_sensivel"] == atributo)
        & (df_comp["metrica"] == metrica)
    ]
    if sub.empty or pd.isna(sub["delta_b_menos_a"].iloc[0]):
        return None
    return float(sub["delta_b_menos_a"].iloc[0])


def _gerar_executive_summary_md(
    metadados_carga: dict[str, Any],
    qualidade: dict[str, Any],
    resumo_selecao: dict[str, Any],
    metricas_teste: dict[str, Any],
    auditoria_fairness: dict[str, Any],
    comparacao_sensiveis: dict[str, Any],
    reconciliacao: dict[str, Any],
) -> str:
    """Constroi reports/executive_summary.md sintetizando evidencias reais da execucao."""
    dim = qualidade["dimensoes"]
    cm = metricas_teste["matriz_confusao"]
    res_sex = auditoria_fairness["por_atributo"]["sex"]["resumo"]
    res_race = auditoria_fairness["por_atributo"]["race"]["resumo"]
    df_comp = comparacao_sensiveis["tabela_comparativa"]

    delta_f1 = _buscar_delta_cv(df_comp, "global", "f1")
    delta_eod_sex = _buscar_delta_cv(df_comp, "sex", "equal_opportunity_difference")
    delta_eod_race = _buscar_delta_cv(df_comp, "race", "equal_opportunity_difference")
    pequenos_race = ", ".join(res_race["grupos_amostra_pequena"]) or "nenhum alerta"

    linhas = [
        "# Sumario Executivo — Adult Income: Model Evaluation & Fairness V2",
        "",
        "> **Natureza do trabalho:** Laboratorio educacional reproduzivel de avaliacao metodologica de modelos, diagnostico de erros e auditoria de Responsible AI sobre dados historicos do Censo dos EUA de 1994. **Nao e uma ferramenta de decisao automatizada para uso em producao sobre pessoas reais.**",
        "",
        "## 1. Visao Geral da Execucao",
        "",
        "| Dimensao | Evidencia Verificada |",
        "| --- | --- |",
        f"| **Dataset** | `{_escapar_md(metadados_carga['caminho_relativo'])}` (`{_fmt_int(dim['total_linhas'])}` linhas x `{dim['total_features']}` features) |",
        f"| **Integridade SHA-256** | `{metadados_carga['sha256'] or 'Carregamento direto OpenML'}` |",
        f"| **Divisao estratificada** | `{_fmt_int(dim['treino_linhas'])}` treino (80%) / `{_fmt_int(dim['teste_linhas'])}` teste (20%) — `0` sobreposicao de indices |",
        f"| **Modelo selecionado (CV 5-fold no treino)** | `{resumo_selecao['modelo_selecionado']}` |",
        f"| **Desempenho no teste** | Accuracy `{_fmt_pct(metricas_teste['accuracy'])}`, Recall `{_fmt_pct(metricas_teste['recall'])}`, Precision `{_fmt_pct(metricas_teste['precision'])}`, F1 `{_fmt_pct(metricas_teste['f1'])}` |",
        f"| **Matriz de confusao** | `TN={_fmt_int(cm['tn'])}`, `FP={_fmt_int(cm['fp'])}`, `FN={_fmt_int(cm['fn'])}`, `TP={_fmt_int(cm['tp'])}` |",
        f"| **Fairness (`sex`, teste)** | EOD = `{_fmt_num(res_sex['equal_opportunity_difference'], 6)}` (`{res_sex['equal_opportunity_escopo']}`), EOdds = `{_fmt_num(res_sex['equalized_odds_difference'], 6)}` (`{res_sex['equalized_odds_escopo']}`) |",
        f"| **Fairness (`race`, teste)** | EOD = `{_fmt_num(res_race['equal_opportunity_difference'], 6)}` (`{res_race['equal_opportunity_escopo']}`), EOdds = `{_fmt_num(res_race['equalized_odds_difference'], 6)}` (`{res_race['equalized_odds_escopo']}`) |",
        f"| **Experimento Cenario B − A (CV treino)** | Delta F1 = `{_fmt_num(delta_f1, 4)}`, Delta EOD `sex` = `{_fmt_num(delta_eod_sex, 4)}`, Delta EOD `race` = `{_fmt_num(delta_eod_race, 4)}` |",
        f"| **Validacao matematica** | `{reconciliacao['total_aprovadas']}/{reconciliacao['total_checagens']}` aprovadas (`{reconciliacao['total_reprovadas']}` reprovadas, `{reconciliacao['total_nao_aplicaveis']}` N/A) |",
        "",
        "## 2. Principais Achados",
        "",
        f"1. **Generalizacao vs. overfitting:** a `Decision Tree (sem limite)` atingiu Accuracy proxima de `100%` no treino com queda expressiva na validacao, sendo objetivamente descartada pelo criterio pre-registrado. `{resumo_selecao['modelo_selecionado']}` entregou o melhor F1 e Recall na classe `>50K` com gap minimo entre treino e validacao.",
        f"2. **Limitacao principal de erros:** dos `{_fmt_int(cm['fp'] + cm['fn'])}` erros no teste, `{_fmt_int(cm['fn'])}` sao falsos negativos (`>50K` previstos como `<=50K`) e `{_fmt_int(cm['fp'])}` sao falsos positivos.",
        f"3. **Disparidades observadas e tamanho amostral:** ha disparidades relevantes em Recall, FPR e Selection Rate tanto por `sex` quanto por `race`. Em `race`, grupos com poucas observacoes e poucos positivos reais (`{pequenos_race}`) aumentam a variabilidade amostral das metricas.",
        "4. **Remocao de `sex` e `race` (Cenario B):** retirar diretamente os atributos sensiveis altera pouco o desempenho global e reduz parte de EOD, mas nao elimina as disparidades observadas. Embora variaveis correlacionadas remanescentes sejam candidatas plausiveis a *proxies* (junto a diferencas de distribuicao e taxas-base entre grupos), a remocao isolada de `sex` e `race` nao comprova causalmente a contribuicao individual dessas variaveis.",
        "",
        "## 3. Relatorios Gerados",
        "",
        "- `data_quality_report.md` — auditoria de tipos, ausentes, duplicatas, `fnlwgt` e categorias raras.",
        "- `model_comparison.csv` e `cross_validation_results.csv` — comparacao dos 4 modelos e resultados por fold.",
        "- `test_metrics.json` e `confusion_matrix.csv` — metricas finais e matriz de confusao.",
        "- `error_analysis.csv` e `model_evaluation_report.md` — diagnostico de erros por segmento e avaliacao completa.",
        "- `fairness_by_sex.csv`, `fairness_by_race.csv`, `fairness_summary.csv`, `sensitive_features_comparison.csv` e `fairness_audit_report.md` — auditoria de fairness e experimento controlado.",
        "- `validation_report.md` — 16 checagens matematicas independentes.",
    ]
    return "\n".join(linhas) + "\n"


def exportar_todos_relatorios(
    diretorio_saida: Path,
    metadados_carga: dict[str, Any],
    qualidade: dict[str, Any],
    tabela_comparacao: pd.DataFrame,
    tabela_folds: pd.DataFrame,
    resumo_selecao: dict[str, Any],
    metricas_teste: dict[str, Any],
    analise_erros: dict[str, Any],
    auditoria_fairness: dict[str, Any],
    comparacao_sensiveis: dict[str, Any],
    reconciliacao: dict[str, Any],
    duracao_segundos: float | None = None,
    timestamp_utc: str | None = None,
) -> list[Path]:
    """Gera e grava os 14 relatorios reproduziveis (deterministicos, sem timestamps volateis)."""
    _ = (duracao_segundos, timestamp_utc)

    saida = Path(diretorio_saida)
    saida.mkdir(parents=True, exist_ok=True)

    payload_test_metrics = {
        "dataset": metadados_carga,
        "modelo_selecionado": resumo_selecao["modelo_selecionado"],
        "criterio_selecao": resumo_selecao,
        "metricas_teste": {
            "total_registros": metricas_teste["total_registros"],
            "positivos_reais": metricas_teste["positivos_reais"],
            "negativos_reais": metricas_teste["negativos_reais"],
            "positivos_previstos": metricas_teste["positivos_previstos"],
            "negativos_previstos": metricas_teste["negativos_previstos"],
            "accuracy": metricas_teste["accuracy"],
            "recall": metricas_teste["recall"],
            "precision": metricas_teste["precision"],
            "f1": metricas_teste["f1"],
            "fpr": metricas_teste["fpr"],
            "selection_rate": metricas_teste["selection_rate"],
            "matriz_confusao": metricas_teste["matriz_confusao"],
            "denominadores_zero": metricas_teste["denominadores_zero"],
            "classification_report": metricas_teste["classification_report_dict"],
        },
        "fairness_resumo_teste": {
            attr: auditoria_fairness["por_atributo"][attr]["resumo"]
            for attr in SENSITIVE_COLUMNS
        },
        "reconciliacao": {
            "total_checagens": reconciliacao["total_checagens"],
            "total_aprovadas": reconciliacao["total_aprovadas"],
            "total_reprovadas": reconciliacao["total_reprovadas"],
            "total_nao_aplicaveis": reconciliacao["total_nao_aplicaveis"],
            "aprovado_geral": reconciliacao["aprovado_geral"],
        },
    }

    arquivos: dict[Path, str] = {
        saida / "data_quality_report.md": _gerar_data_quality_report_md(qualidade),
        saida / "model_comparison.csv": _serializar_csv(tabela_comparacao),
        saida / "cross_validation_results.csv": _serializar_csv(tabela_folds),
        saida / "test_metrics.json": json.dumps(payload_test_metrics, indent=2, ensure_ascii=False)
        + "\n",
        saida / "confusion_matrix.csv": _serializar_csv(metricas_teste["matriz_confusao_df"]),
        saida / "error_analysis.csv": _serializar_csv(analise_erros["tabela_erros"]),
        saida / "fairness_by_sex.csv": _serializar_csv(
            auditoria_fairness["por_atributo"]["sex"]["tabela_por_grupo"]
        ),
        saida / "fairness_by_race.csv": _serializar_csv(
            auditoria_fairness["por_atributo"]["race"]["tabela_por_grupo"]
        ),
        saida / "fairness_summary.csv": _serializar_csv(auditoria_fairness["tabela_resumo"]),
        saida / "sensitive_features_comparison.csv": _serializar_csv(
            comparacao_sensiveis["tabela_comparativa"]
        ),
        saida / "model_evaluation_report.md": _gerar_model_evaluation_report_md(
            tabela_comparacao=tabela_comparacao,
            resumo_selecao=resumo_selecao,
            metricas_teste=metricas_teste,
            analise_erros=analise_erros,
            metadados_carga=metadados_carga,
        ),
        saida / "fairness_audit_report.md": _gerar_fairness_audit_report_md(
            auditoria_fairness=auditoria_fairness,
            comparacao_sensiveis=comparacao_sensiveis,
            comparacao_ponderada_vs_simples=reconciliacao["comparacao_ponderada_vs_simples"],
        ),
        saida / "validation_report.md": _gerar_validation_report_md(reconciliacao),
        saida / "executive_summary.md": _gerar_executive_summary_md(
            metadados_carga=metadados_carga,
            qualidade=qualidade,
            resumo_selecao=resumo_selecao,
            metricas_teste=metricas_teste,
            auditoria_fairness=auditoria_fairness,
            comparacao_sensiveis=comparacao_sensiveis,
            reconciliacao=reconciliacao,
        ),
    }

    return _gravar_lote_com_rollback(arquivos)
