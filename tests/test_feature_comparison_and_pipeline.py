"""Testes do experimento com atributos sensiveis, integracao offline, rollback e AST."""

import ast
from pathlib import Path

import pandas as pd
import pytest

from src.analysis.feature_comparison import executar_comparacao_atributos_sensiveis
from src.config import MODEL_LOGISTIC, PROJECT_ROOT
from src.data.loader import separar_treino_teste, validar_estrutura_dataset
from src.pipeline import executar_pipeline
from src.reporting import exports as exports_mod

RELATORIOS_ESPERADOS = {
    "data_quality_report.md",
    "model_comparison.csv",
    "cross_validation_results.csv",
    "test_metrics.json",
    "confusion_matrix.csv",
    "error_analysis.csv",
    "fairness_by_sex.csv",
    "fairness_by_race.csv",
    "fairness_summary.csv",
    "sensitive_features_comparison.csv",
    "model_evaluation_report.md",
    "fairness_audit_report.md",
    "validation_report.md",
    "executive_summary.md",
}

BIBLIOTECAS_PROIBIDAS = {
    "numpy",
    "matplotlib",
    "seaborn",
    "tensorflow",
    "torch",
    "xgboost",
    "lightgbm",
    "shap",
    "polars",
    "pyspark",
    "scipy",
    "joblib",
    "streamlit",
    "fastapi",
    "flask",
    "django",
}


def test_experimento_atributos_sensiveis_remove_colunas_mantendo_alinhamento(
    dataset_sintetico_adult: pd.DataFrame,
) -> None:
    """Cenario B deve remover sex e race mantendo indices perfeitamente alinhados ao Fairlearn."""
    X, y = validar_estrutura_dataset(dataset_sintetico_adult)
    X_train, X_test, y_train, y_test = separar_treino_teste(X, y)
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

    comp = executar_comparacao_atributos_sensiveis(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        modelo_nome=MODEL_LOGISTIC,
        colunas_numericas=num_cols,
        colunas_categoricas=cat_cols,
        cv_splits=4,
    )

    proto = comp["protocolo"]
    assert len(proto["features_cenario_a"]) == 14
    assert len(proto["features_cenario_b"]) == 12
    assert "sex" not in proto["features_cenario_b"]
    assert "race" not in proto["features_cenario_b"]
    assert all(comp["verificacao_alinhamento"].values())

    escopos = set(comp["tabela_comparativa"]["escopo"].tolist())
    assert escopos == {"validacao_cv_treino_oof", "teste_historico_descritivo"}


def test_pipeline_offline_completo_consistencia_e_reexecucao_deterministica(
    csv_sintetico_adult: Path,
    tmp_path: Path,
) -> None:
    """Executa pipeline completo offline duas vezes e valida os 14 relatorios byte a byte."""
    dir_1 = tmp_path / "exec_1"
    dir_2 = tmp_path / "exec_2"

    res_1 = executar_pipeline(
        fonte="local",
        caminho_dataset=csv_sintetico_adult,
        diretorio_saida=dir_1,
        verificar_hash=False,
        exportar_relatorios=True,
        cv_splits=4,
    )
    res_2 = executar_pipeline(
        fonte="local",
        caminho_dataset=csv_sintetico_adult,
        diretorio_saida=dir_2,
        verificar_hash=False,
        exportar_relatorios=True,
        cv_splits=4,
    )

    assert res_1["reconciliacao"]["aprovado_geral"] is True
    assert res_2["reconciliacao"]["aprovado_geral"] is True

    # Em dataset sintetico, VAL-16 deve ser marcada como NAO_APLICAVEL (sem falsos numeros da V1)
    df_val = res_1["reconciliacao"]["tabela_checagens"]
    status_val16 = str(df_val.loc[df_val["codigo"] == "VAL-16", "status"].iloc[0])
    assert status_val16 == "NAO_APLICAVEL"

    gerados_1 = {p.name for p in dir_1.iterdir() if p.is_file()}
    gerados_2 = {p.name for p in dir_2.iterdir() if p.is_file()}
    assert gerados_1 == RELATORIOS_ESPERADOS
    assert gerados_2 == RELATORIOS_ESPERADOS

    df_fair_sum = pd.read_csv(dir_1 / "fairness_summary.csv")
    for coluna_obrigatoria in (
        "equal_opportunity_escopo",
        "equalized_odds_escopo",
        "avaliacao_fairness_parcial",
        "grupos_excluidos_equalized_odds",
    ):
        assert coluna_obrigatoria in df_fair_sum.columns

    for nome in sorted(RELATORIOS_ESPERADOS):
        bytes_1 = (dir_1 / nome).read_bytes()
        bytes_2 = (dir_2 / nome).read_bytes()
        assert bytes_1 == bytes_2, f"Nao determinismo detectado em {nome}"
        assert b"\r\n" not in bytes_1, f"Terminacao CRLF indevida em {nome}"
        texto = bytes_1.decode("utf-8")
        assert "C:\\Users\\" not in texto, f"Caminho local absoluto vazou em {nome}"


def test_rollback_lote_exportacao_preserva_estado_anterior(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regressao: falha no meio da gravacao do lote deve restaurar o arquivo anterior."""
    arq_1 = tmp_path / "relatorio_1.txt"
    arq_2 = tmp_path / "relatorio_2.txt"
    arq_1.write_text("conteudo_original_1\n", encoding="utf-8", newline="\n")

    escritor_real = exports_mod._escrever_texto_atomico
    chamadas = {"n": 0}

    def escritor_com_falha(destino: Path, conteudo: str) -> None:
        chamadas["n"] += 1
        if chamadas["n"] == 2:
            raise OSError("Erro simulado de disco na segunda escrita")
        escritor_real(destino, conteudo)

    monkeypatch.setattr(exports_mod, "_escrever_texto_atomico", escritor_com_falha)

    with pytest.raises(RuntimeError, match="rollback"):
        exports_mod._gravar_lote_com_rollback(
            {
                arq_1: "novo_conteudo_1\n",
                arq_2: "novo_conteudo_2\n",
            }
        )

    assert arq_1.read_text(encoding="utf-8") == "conteudo_original_1\n"
    assert not arq_2.exists()
    assert list(tmp_path.glob("*.tmp")) == []


def test_ausencia_de_importacoes_proibidas_em_todo_codigo() -> None:
    """Auditoria estatica via AST garantindo 0 importacoes diretas de bibliotecas proibidas."""
    arquivos_py = [PROJECT_ROOT / "main.py"]
    arquivos_py.extend(sorted((PROJECT_ROOT / "src").rglob("*.py")))
    arquivos_py.extend(sorted((PROJECT_ROOT / "tests").rglob("*.py")))

    violacoes: list[str] = []
    for arquivo in arquivos_py:
        arvore = ast.parse(arquivo.read_text(encoding="utf-8"))
        for no in ast.walk(arvore):
            if isinstance(no, ast.Import):
                for alias in no.names:
                    raiz = alias.name.split(".")[0]
                    if raiz in BIBLIOTECAS_PROIBIDAS:
                        violacoes.append(f"{arquivo.name}: import {alias.name}")
            elif isinstance(no, ast.ImportFrom) and no.module:
                raiz = no.module.split(".")[0]
                if raiz in BIBLIOTECAS_PROIBIDAS:
                    violacoes.append(f"{arquivo.name}: from {no.module} import ...")

    assert violacoes == [], f"Importacoes proibidas encontradas: {violacoes}"


def test_cli_main_executa_offline_e_trata_erros(
    csv_sintetico_adult: Path,
    tmp_path: Path,
) -> None:
    """Valida codigos de saida da CLI (0 em sucesso offline, 2 em entrada inexistente)."""
    from main import main as cli_main

    saida_cli = tmp_path / "cli_reports"
    codigo_ok = cli_main(
        [
            "--source",
            "local",
            "--dataset-path",
            str(csv_sintetico_adult),
            "--output-dir",
            str(saida_cli),
            "--no-verify-hash",
        ]
    )
    assert codigo_ok == 0
    assert {p.name for p in saida_cli.iterdir() if p.is_file()} == RELATORIOS_ESPERADOS

    codigo_erro = cli_main(
        [
            "--source",
            "local",
            "--dataset-path",
            str(tmp_path / "inexistente.csv"),
            "--no-verify-hash",
        ]
    )
    assert codigo_erro == 2

