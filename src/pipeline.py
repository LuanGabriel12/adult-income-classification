"""Orquestrador ponta a ponta do laboratorio Adult Income: Model Evaluation & Fairness V2."""

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.analysis.error_analysis import executar_analise_erros
from src.analysis.fairness import executar_auditoria_fairness
from src.analysis.feature_comparison import executar_comparacao_atributos_sensiveis
from src.config import (
    CANONICAL_DATASET_PATH,
    CANONICAL_DATASET_SHA256,
    CV_SPLITS,
    RANDOM_STATE,
    REPORTS_DIR,
    SENSITIVE_COLUMNS,
    TEST_SIZE,
)
from src.data.loader import (
    carregar_dataset_local,
    carregar_dataset_openml,
    separar_treino_teste,
)
from src.data.validation import auditar_qualidade_dados
from src.modeling.evaluation import (
    calcular_metricas_classificacao,
    executar_validacao_cruzada_modelos,
    selecionar_melhor_modelo,
)
from src.modeling.training import (
    criar_pipeline_modelo,
    gerar_previsoes,
    treinar_pipeline,
)
from src.reporting.exports import exportar_todos_relatorios
from src.reporting.reconciliation import executar_validacao_matematica


def executar_pipeline(
    fonte: str = "local",
    caminho_dataset: Path | None = None,
    diretorio_saida: Path = REPORTS_DIR,
    verificar_hash: bool = True,
    exportar_relatorios: bool = True,
    cv_splits: int = CV_SPLITS,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> dict[str, Any]:
    """Executa todas as etapas do laboratorio com validacao matematica antes da gravacao."""
    inicio = time.perf_counter()
    timestamp_utc = datetime.now(timezone.utc).isoformat()

    # 1. Carregamento e validacao estrutural
    if fonte == "local":
        caminho = Path(caminho_dataset) if caminho_dataset is not None else CANONICAL_DATASET_PATH
        sha_esperado = CANONICAL_DATASET_SHA256 if verificar_hash else None
        X, y, metadados_carga = carregar_dataset_local(
            caminho=caminho,
            verificar_hash=verificar_hash,
            sha256_esperado=sha_esperado,
            exigir_colunas_adult=True,
        )
    elif fonte == "openml":
        X, y, metadados_carga = carregar_dataset_openml()
    else:
        raise ValueError(f"Fonte de dados nao suportada: '{fonte}'. Use 'local' ou 'openml'.")

    # 2. Divisao estratificada 80/20
    X_train, X_test, y_train, y_test = separar_treino_teste(
        X=X,
        y=y,
        test_size=test_size,
        random_state=random_state,
    )

    # 3. Auditoria completa de qualidade dos dados
    qualidade = auditar_qualidade_dados(
        X=X,
        y=y,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        metadados_carga=metadados_carga,
    )

    colunas_numericas = qualidade["dimensoes"]["colunas_numericas"]
    colunas_categoricas = qualidade["dimensoes"]["colunas_categoricas"]

    # 4. Validacao cruzada 5-fold (somente no treino) e selecao pre-registrada
    tabela_cv_bruta, tabela_folds = executar_validacao_cruzada_modelos(
        X_train=X_train,
        y_train=y_train,
        colunas_numericas=colunas_numericas,
        colunas_categoricas=colunas_categoricas,
        cv_splits=cv_splits,
    )
    modelo_selecionado, tabela_comparacao, resumo_selecao = selecionar_melhor_modelo(
        tabela_comparacao=tabela_cv_bruta,
    )

    # 5. Treinamento final do modelo selecionado em X_train e avaliacao em X_test
    pipeline_final = criar_pipeline_modelo(
        nome_modelo=modelo_selecionado,
        colunas_numericas=colunas_numericas,
        colunas_categoricas=colunas_categoricas,
        random_state=random_state,
    )
    pipeline_final = treinar_pipeline(pipeline_final, X_train, y_train)
    y_pred_test = gerar_previsoes(pipeline_final, X_test)

    metricas_teste = calcular_metricas_classificacao(
        y_true=y_test,
        y_pred=y_pred_test,
    )

    # 6. Analise descritiva de erros de classificacao
    analise_erros = executar_analise_erros(
        X_eval=X_test,
        y_true=y_test,
        y_pred=y_pred_test,
    )

    # 7. Auditoria de fairness por sex e race
    auditoria_fairness = executar_auditoria_fairness(
        X_eval=X_test,
        y_true=y_test,
        y_pred=y_pred_test,
        atributos_sensiveis=SENSITIVE_COLUMNS,
    )

    # 8. Experimento controlado: Cenario A (com sex/race) vs Cenario B (sem sex/race)
    comparacao_sensiveis = executar_comparacao_atributos_sensiveis(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        modelo_nome=modelo_selecionado,
        colunas_numericas=colunas_numericas,
        colunas_categoricas=colunas_categoricas,
        atributos_sensiveis=SENSITIVE_COLUMNS,
        cv_splits=cv_splits,
        random_state=random_state,
    )

    # 9. Validacao matematica independente (16 checagens)
    reconciliacao = executar_validacao_matematica(
        X=X,
        y=y,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        y_pred_test=y_pred_test,
        pipeline_final=pipeline_final,
        metricas_teste=metricas_teste,
        analise_erros=analise_erros,
        auditoria_fairness=auditoria_fairness,
        comparacao_sensiveis=comparacao_sensiveis,
        metadados_carga=metadados_carga,
    )

    if not reconciliacao["aprovado_geral"]:
        falhas = reconciliacao["tabela_checagens"].loc[
            reconciliacao["tabela_checagens"]["status"] == "REPROVADO",
            "codigo",
        ].tolist()
        raise RuntimeError(
            f"Falha na validacao matematica independente; checagens reprovadas: {falhas}"
        )

    duracao_segundos = float(time.perf_counter() - inicio)

    # 10. Exportacao atomica com rollback dos 14 relatorios
    arquivos_gerados: list[Path] = []
    if exportar_relatorios:
        arquivos_gerados = exportar_todos_relatorios(
            diretorio_saida=Path(diretorio_saida),
            metadados_carga=metadados_carga,
            qualidade=qualidade,
            tabela_comparacao=tabela_comparacao,
            tabela_folds=tabela_folds,
            resumo_selecao=resumo_selecao,
            metricas_teste=metricas_teste,
            analise_erros=analise_erros,
            auditoria_fairness=auditoria_fairness,
            comparacao_sensiveis=comparacao_sensiveis,
            reconciliacao=reconciliacao,
            duracao_segundos=duracao_segundos,
            timestamp_utc=timestamp_utc,
        )

    return {
        "metadados_carga": metadados_carga,
        "qualidade": qualidade,
        "tabela_comparacao": tabela_comparacao,
        "tabela_folds": tabela_folds,
        "resumo_selecao": resumo_selecao,
        "metricas_teste": metricas_teste,
        "analise_erros": analise_erros,
        "auditoria_fairness": auditoria_fairness,
        "comparacao_sensiveis": comparacao_sensiveis,
        "reconciliacao": reconciliacao,
        "arquivos_gerados": arquivos_gerados,
        "duracao_segundos": duracao_segundos,
        "timestamp_utc": timestamp_utc,
    }
