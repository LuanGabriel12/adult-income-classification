"""Pre-processamento encapsulado em ColumnTransformer para evitar data leakage."""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def criar_preprocessador(
    colunas_numericas: list[str],
    colunas_categoricas: list[str],
) -> ColumnTransformer:
    """Constroi um ColumnTransformer separado para colunas numericas e categoricas.

    Todas as estatisticas (mediana, media, desvio-padrao, moda e categorias do
    One-Hot Encoding) sao aprendidas exclusivamente no conjunto de treino de
    cada divisao quando encapsuladas dentro de uma Pipeline.

    Na base oficial Adult (OpenML 1590 v2), nao existem valores ausentes nas
    colunas numericas; portanto, ``SimpleImputer(strategy="median")`` preserva
    100% de paridade numerica com a V1 enquanto protege a pipeline contra
    ausencias numericas em entradas alternativas ou testes unitarios.
    """
    if not colunas_numericas and not colunas_categoricas:
        raise ValueError("Informe ao menos uma coluna numerica ou categorica.")

    transformadores: list[tuple[str, Pipeline, list[str]]] = []

    if colunas_numericas:
        tratamento_numerico = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformadores.append(("numericas", tratamento_numerico, list(colunas_numericas)))

    if colunas_categoricas:
        tratamento_categorico = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        transformadores.append(("categoricas", tratamento_categorico, list(colunas_categoricas)))

    return ColumnTransformer(transformers=transformadores, remainder="drop")
