"""Testes de carregamento, integridade SHA-256, validacao de schema e auditoria de dados."""

from pathlib import Path

import pandas as pd
import pytest

from src.config import (
    CANONICAL_DATASET_BYTES,
    CANONICAL_DATASET_PATH,
    CANONICAL_DATASET_SHA256,
    FEATURE_COLUMNS,
)
from src.data.loader import (
    calcular_sha256_arquivo,
    carregar_dataset_local,
    separar_treino_teste,
    validar_estrutura_dataset,
)
from src.data.validation import auditar_qualidade_dados, normalizar_serie_texto


def test_dataset_canonico_integridade_sha256_e_lf() -> None:
    """Valida que o CSV canonico local existe, usa apenas LF e bate o SHA-256 oficial."""
    assert CANONICAL_DATASET_PATH.exists()
    conteudo_bruto = CANONICAL_DATASET_PATH.read_bytes()
    assert b"\r\n" not in conteudo_bruto, "O CSV canonico deve preservar terminacao LF."
    assert len(conteudo_bruto) == CANONICAL_DATASET_BYTES
    assert calcular_sha256_arquivo(CANONICAL_DATASET_PATH) == CANONICAL_DATASET_SHA256


def test_carregar_dataset_local_rejeita_hash_divergente(csv_sintetico_adult: Path) -> None:
    """Deve levantar ValueError quando o SHA-256 esperado nao corresponde ao arquivo."""
    with pytest.raises(ValueError, match="SHA-256"):
        carregar_dataset_local(
            caminho=csv_sintetico_adult,
            verificar_hash=True,
            sha256_esperado="0" * 64,
        )


def test_carregar_dataset_local_arquivo_ausente(tmp_path: Path) -> None:
    """Deve levantar FileNotFoundError para caminho inexistente."""
    with pytest.raises(FileNotFoundError):
        carregar_dataset_local(caminho=tmp_path / "nao_existe.csv", verificar_hash=False)


def test_carregar_dataset_local_rejeita_cabecalho_duplicado(tmp_path: Path) -> None:
    """Regressao: rejeita CSV com duas colunas de mesmo nome antes do Pandas renomear."""
    caminho = tmp_path / "cabecalho_duplicado.csv"
    caminho.write_text("age,age,class\n30,31,<=50K\n40,41,>50K\n", encoding="utf-8", newline="\n")
    with pytest.raises(ValueError, match="duplicados"):
        carregar_dataset_local(
            caminho=caminho,
            verificar_hash=False,
            exigir_colunas_adult=False,
        )


def test_carregar_dataset_local_rejeita_deslocamento_de_colunas(tmp_path: Path) -> None:
    """Regressao: rejeita CSV malformado onde mais virgulas criam indice diferente de RangeIndex."""
    caminho = tmp_path / "colunas_deslocadas.csv"
    caminho.write_text(
        "age,class\n30,<=50K,extra_invalido\n45,>50K,extra_invalido\n",
        encoding="utf-8",
        newline="\n",
    )
    with pytest.raises(ValueError, match="RangeIndex"):
        carregar_dataset_local(
            caminho=caminho,
            verificar_hash=False,
            exigir_colunas_adult=False,
        )


def test_validar_estrutura_rejeita_classes_inesperadas(
    dataset_sintetico_adult: pd.DataFrame,
) -> None:
    """Deve rejeitar rotulos fora de ('<=50K', '>50K'), como ponto final residual."""
    df_invalido = dataset_sintetico_adult.copy()
    df_invalido.loc[0, "class"] = "<=50K."
    with pytest.raises(ValueError, match="Classes inesperadas"):
        validar_estrutura_dataset(df_invalido)


def test_validar_estrutura_rejeita_target_ausente(
    dataset_sintetico_adult: pd.DataFrame,
) -> None:
    """Deve rejeitar valores ausentes na coluna alvo."""
    df_nulo = dataset_sintetico_adult.copy()
    df_nulo.loc[1, "class"] = None
    with pytest.raises(ValueError, match="ausente"):
        validar_estrutura_dataset(df_nulo)


def test_validar_estrutura_rejeita_classe_unica(
    dataset_sintetico_adult: pd.DataFrame,
) -> None:
    """Deve rejeitar dataset contendo apenas uma classe no target."""
    df_unica = dataset_sintetico_adult.copy()
    df_unica["class"] = "<=50K"
    with pytest.raises(ValueError, match="ambas as classes"):
        validar_estrutura_dataset(df_unica)


def test_separar_treino_teste_estratificacao_e_indices_disjuntos(
    dataset_sintetico_adult: pd.DataFrame,
) -> None:
    """Verifica 0 sobreposicao de indices, conservacao de linhas e ordem de features."""
    X, y = validar_estrutura_dataset(dataset_sintetico_adult)
    assert list(X.columns) == FEATURE_COLUMNS

    X_train, X_test, y_train, y_test = separar_treino_teste(X, y, test_size=0.20, random_state=42)
    assert len(X_train) + len(X_test) == len(X)
    assert len(X_train) == 64
    assert len(X_test) == 16
    assert list(X_train.index) == list(y_train.index)
    assert list(X_test.index) == list(y_test.index)
    assert set(X_train.index).isdisjoint(set(X_test.index))
    assert set(X_train.index).union(set(X_test.index)) == set(X.index)


def test_auditoria_qualidade_detecta_ausentes_e_duplicatas_sem_remover(
    csv_sintetico_adult: Path,
) -> None:
    """Auditoria deve medir ausentes e duplicatas mantendo todos os 80 registros intactos."""
    X, y, meta = carregar_dataset_local(caminho=csv_sintetico_adult, verificar_hash=False)
    X_train, X_test, y_train, y_test = separar_treino_teste(X, y)
    auditoria = auditar_qualidade_dados(X, y, X_train, X_test, y_train, y_test, meta)

    assert auditoria["dimensoes"]["total_linhas"] == 80
    assert auditoria["duplicatas"]["duplicatas_xy_excedentes"] == 1
    assert auditoria["duplicatas"]["possui_identificador_unico"] is False
    assert auditoria["analise_ausentes_trabalho"]["ambos_ausentes"] == 2
    assert auditoria["analise_fnlwgt"]["min"] == pytest.approx(100000.0)


def test_normalizar_serie_texto_preserva_ausentes_e_tipos_sem_mutar() -> None:
    """Conversao de texto deve evitar downcasting, nao mutar a Series e preservar ausentes reais."""
    serie_int = pd.Series([10, 20, 30], index=[5, 6, 7])
    serie_float = pd.Series([1.5, float("nan"), 3.0], index=[5, 6, 7])
    serie_obj = pd.Series(["Private", None, "nan"], index=[5, 6, 7], dtype="object")
    serie_cat = pd.Series(["A", None, "B"], index=[5, 6, 7], dtype="category")

    norm_int = normalizar_serie_texto(serie_int, "<AUSENTE>")
    norm_float = normalizar_serie_texto(serie_float, "<AUSENTE>")
    norm_obj = normalizar_serie_texto(serie_obj, "<AUSENTE>")
    norm_cat = normalizar_serie_texto(serie_cat, "<AUSENTE>")

    assert norm_int.tolist() == ["10", "20", "30"]
    assert norm_float.tolist() == ["1.5", "<AUSENTE>", "3.0"]
    # Literal "nan" deve permanecer "nan", enquanto None real vira marcador
    assert norm_obj.tolist() == ["Private", "<AUSENTE>", "nan"]
    assert norm_cat.tolist() == ["A", "<AUSENTE>", "B"]

    # Series originais preservam exatamente seus valores ausentes
    assert int(serie_float.isna().sum()) == 1
    assert int(serie_obj.isna().sum()) == 1
    assert int(serie_cat.isna().sum()) == 1
    assert list(norm_cat.index) == [5, 6, 7]
