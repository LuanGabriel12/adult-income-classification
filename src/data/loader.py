"""Carregamento reproduzivel (CSV canonico local e OpenML), verificacao SHA-256 e split."""

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from src.config import (
    CANONICAL_DATASET_PATH,
    CANONICAL_DATASET_SHA256,
    DATASET_PROVENANCE,
    FEATURE_COLUMNS,
    PROJECT_ROOT,
    RANDOM_STATE,
    TARGET_CLASSES,
    TARGET_COLUMN,
    TEST_SIZE,
)


def calcular_sha256_arquivo(caminho: Path) -> str:
    """Calcula o hash SHA-256 de um arquivo lendo-o em blocos binarios de 1 MB."""
    caminho_path = Path(caminho)
    if not caminho_path.exists():
        raise FileNotFoundError(f"Arquivo nao encontrado para calculo de SHA-256: {caminho_path}")
    hasher = hashlib.sha256()
    with caminho_path.open("rb") as arquivo:
        bloco = arquivo.read(1024 * 1024)
        while bloco:
            hasher.update(bloco)
            bloco = arquivo.read(1024 * 1024)
    return hasher.hexdigest()


def formatar_caminho_relativo(caminho: Path) -> str:
    """Retorna caminho portavel relativo a raiz do projeto quando possivel."""
    resolvido = Path(caminho).resolve()
    try:
        return resolvido.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return resolvido.name


def validar_cabecalho_csv_sem_duplicatas(caminho: Path) -> list[str]:
    """Inspeciona a primeira linha do CSV bruto para rejeitar cabecalhos vazios ou duplicados."""
    primeira_linha = pd.read_csv(caminho, header=None, nrows=1, dtype=str, keep_default_na=False)
    if primeira_linha.empty:
        raise ValueError("Arquivo CSV vazio: cabecalho ausente.")
    nomes_brutos = [str(valor) for valor in primeira_linha.iloc[0].tolist()]
    nomes_limpos = [nome.strip() for nome in nomes_brutos]

    if any(nome == "" for nome in nomes_limpos):
        raise ValueError("Arquivo CSV contem coluna com cabecalho vazio.")

    vistos: set[str] = set()
    duplicados: list[str] = []
    for nome in nomes_limpos:
        if nome in vistos and nome not in duplicados:
            duplicados.append(nome)
        vistos.add(nome)
    if duplicados:
        raise ValueError(f"Arquivo CSV contem cabecalhos duplicados: {duplicados}")

    return nomes_limpos


def validar_estrutura_dataset(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    classes_esperadas: tuple[str, str] = TARGET_CLASSES,
    exigir_colunas_adult: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """Valida integridade estrutural, colunas obrigatorias e classes validas do target."""
    if df.empty:
        raise ValueError("O dataset fornecido esta vazio.")
    if not isinstance(df.index, pd.RangeIndex) and df.index.duplicated().any():
        raise ValueError("O dataset contem indices duplicados.")

    if target_column not in df.columns:
        raise ValueError(f"Coluna alvo obrigatoria ausente: '{target_column}'.")

    if exigir_colunas_adult:
        faltantes = [col for col in FEATURE_COLUMNS if col not in df.columns]
        if faltantes:
            raise ValueError(f"Colunas obrigatorias do Adult ausentes: {faltantes}")

    y = df[target_column].copy()
    if y.isna().any():
        qtd_nulos = int(y.isna().sum())
        raise ValueError(
            f"A coluna alvo '{target_column}' contem {qtd_nulos} valor(es) ausente(s)."
        )

    y_texto = y.astype(str)
    classes_validas_set = set(classes_esperadas)
    mascara_valida = y_texto.isin(classes_validas_set)
    if not bool(mascara_valida.all()):
        inesperadas = sorted(set(y_texto.loc[~mascara_valida].tolist()))
        raise ValueError(
            f"Classes inesperadas encontradas em '{target_column}': {inesperadas}. "
            f"Esperado: {list(classes_esperadas)}."
        )

    classes_presentes = set(y_texto.tolist())
    if len(classes_presentes) < 2:
        raise ValueError(
            f"A coluna alvo '{target_column}' deve conter ambas as classes "
            f"{list(classes_esperadas)}; encontrado apenas: {sorted(classes_presentes)}."
        )

    X = df.drop(columns=[target_column]).copy()
    if X.shape[1] == 0:
        raise ValueError("O dataset nao possui colunas de features.")

    if exigir_colunas_adult:
        X = X[FEATURE_COLUMNS].copy()

    return X, y_texto.rename(target_column)


def carregar_dataset_local(
    caminho: Path = CANONICAL_DATASET_PATH,
    verificar_hash: bool = True,
    sha256_esperado: str | None = CANONICAL_DATASET_SHA256,
    exigir_colunas_adult: bool = True,
) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    """Carrega dataset CSV local com verificacao opcional de SHA-256 e validacao de schema."""
    caminho_path = Path(caminho)
    if not caminho_path.exists():
        raise FileNotFoundError(
            f"Dataset local nao encontrado em '{caminho_path}'. "
            "Execute com '--source openml' ou restaure 'data/adult_openml_1590_v2.csv'."
        )

    validar_cabecalho_csv_sem_duplicatas(caminho_path)

    sha256_calculado = calcular_sha256_arquivo(caminho_path)
    tamanho_bytes = int(caminho_path.stat().st_size)

    if verificar_hash and sha256_esperado is not None and sha256_calculado != sha256_esperado:
        raise ValueError(
            "Divergencia de integridade SHA-256 no dataset local. "
            f"Esperado: {sha256_esperado}; Obtido: {sha256_calculado}."
        )

    df = pd.read_csv(caminho_path, na_values=[""], keep_default_na=False)
    if (
        not isinstance(df.index, pd.RangeIndex)
        or df.index.start != 0
        or df.index.step != 1
    ):
        raise ValueError(
            "Deslocamento estrutural detectado na leitura do CSV "
            "(indice diferente de RangeIndex padrao 0..N-1)."
        )

    X, y = validar_estrutura_dataset(df, exigir_colunas_adult=exigir_colunas_adult)

    metadados = {
        "fonte_carregamento": "csv_local_canonico",
        "caminho_relativo": formatar_caminho_relativo(caminho_path),
        "tamanho_bytes": tamanho_bytes,
        "sha256": sha256_calculado,
        "hash_verificado": bool(verificar_hash and sha256_esperado is not None),
        "dataset_oficial_openml_1590": sha256_calculado == CANONICAL_DATASET_SHA256,
        "linhas": int(len(X)),
        "features": int(X.shape[1]),
        "target": TARGET_COLUMN,
    }
    return X, y, metadados


def carregar_dataset_openml() -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    """Carrega o dataset diretamente do OpenML (ID 1590, versao 2), preservando o metodo da V1."""
    adult = fetch_openml(
        name=DATASET_PROVENANCE["openml_nome"],
        version=DATASET_PROVENANCE["openml_versao"],
        as_frame=True,
    )
    df = pd.concat([adult.data, adult.target], axis=1)
    X, y = validar_estrutura_dataset(df, exigir_colunas_adult=True)

    metadados = {
        "fonte_carregamento": "openml_fetch",
        "caminho_relativo": "openml://adult/version=2 (id=1590)",
        "tamanho_bytes": None,
        "sha256": None,
        "hash_verificado": False,
        "dataset_oficial_openml_1590": (
            int(len(X)) == DATASET_PROVENANCE["registros_esperados"]
            and int(X.shape[1]) == DATASET_PROVENANCE["features_esperadas"]
        ),
        "linhas": int(len(X)),
        "features": int(X.shape[1]),
        "target": TARGET_COLUMN,
    }
    return X, y, metadados


def exportar_dataset_canonico_de_openml(
    caminho_destino: Path = CANONICAL_DATASET_PATH,
) -> dict[str, Any]:
    """Exporta copia local canonica (UTF-8, LF) a partir do OpenML ID 1590 v2."""
    X, y, _ = carregar_dataset_openml()
    caminho_destino.parent.mkdir(parents=True, exist_ok=True)
    df_canonico = pd.concat([X, y], axis=1)
    df_canonico.to_csv(caminho_destino, index=False, encoding="utf-8", lineterminator="\n")
    sha256 = calcular_sha256_arquivo(caminho_destino)
    return {
        "caminho_relativo": formatar_caminho_relativo(caminho_destino),
        "tamanho_bytes": int(caminho_destino.stat().st_size),
        "sha256": sha256,
        "linhas": int(len(df_canonico)),
    }


def separar_treino_teste(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Realiza divisao estratificada 80/20 preservando indices originais sem sobreposicao."""
    if len(X) != len(y):
        raise ValueError("X e y devem possuir o mesmo numero de registros.")
    if list(X.index) != list(y.index):
        raise ValueError("Indices de X e y nao estao alinhados antes da divisao treino/teste.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    sobreposicao = set(X_train.index).intersection(set(X_test.index))
    if sobreposicao:
        raise RuntimeError(
            f"Sobreposicao indevida de indices entre treino e teste: {len(sobreposicao)}"
        )

    return X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy()
