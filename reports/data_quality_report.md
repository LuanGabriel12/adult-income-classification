# Relatorio de Qualidade dos Dados — Adult / Census Income

## 1. Proveniencia, Integridade e Dimensoes

| Atributo | Valor verificado |
| --- | --- |
| Fonte de carregamento | `csv_local_canonico` |
| Origem / caminho | `data/adult_openml_1590_v2.csv` |
| SHA-256 (arquivo local) | `8f970e20c9991979ec0a9ccb65ce97543d678709a2c6f8ef23f7fb32f40adddc` |
| Hash verificado | `Sim` |
| Dataset oficial OpenML 1590 v2 | `Sim` |
| Dimensoes totais | `48.842` linhas x `14` features + target `class` |
| Divisao treino / teste (80/20 estratificada) | `39.073` treino / `9.769` teste |
| Autoria e periodo historico | Barry Becker e Ronny Kohavi (Silicon Graphics, 1996) — 1994 (US Census Bureau — Current Population Survey, CPS) |
| Licenca verificada | UCI: CC BY 4.0 (Creative Commons Attribution 4.0 International) / OpenML: `Public` |

## 2. Esquema, Tipos e Valores Ausentes

| Coluna | Tipo logico | Dtype | Valores unicos | Ausentes | % Ausentes |
| --- | --- | --- | ---: | ---: | ---: |
| `age` | numerica | `int64` | 74 | 0 | 0,00% |
| `workclass` | categorica | `str` | 8 | 2.799 | 5,73% |
| `fnlwgt` | numerica | `int64` | 28.523 | 0 | 0,00% |
| `education` | categorica | `str` | 16 | 0 | 0,00% |
| `education-num` | numerica | `int64` | 16 | 0 | 0,00% |
| `marital-status` | categorica | `str` | 7 | 0 | 0,00% |
| `occupation` | categorica | `str` | 14 | 2.809 | 5,75% |
| `relationship` | categorica | `str` | 6 | 0 | 0,00% |
| `race` | categorica | `str` | 5 | 0 | 0,00% |
| `sex` | categorica | `str` | 2 | 0 | 0,00% |
| `capital-gain` | numerica | `int64` | 123 | 0 | 0,00% |
| `capital-loss` | numerica | `int64` | 99 | 0 | 0,00% |
| `hours-per-week` | numerica | `int64` | 96 | 0 | 0,00% |
| `native-country` | categorica | `str` | 41 | 857 | 1,75% |

### Relacao entre ausencias de `workclass` e `occupation`

- Registros com **ambos ausentes**: `2.799`.
- Registros com **apenas `workclass` ausente**: `0`.
- Registros com **apenas `occupation` ausente**: `10` (sendo `10` com `workclass == Never-worked`).

## 3. Distribuicao das Classes e Desbalanceamento

| Classe | Quantidade | Proporcao |
| --- | ---: | ---: |
| `<=50K` | 37.155 | 76,07% |
| `>50K` | 11.687 | 23,93% |

- Razao classe majoritaria / minoritaria: **`3,18:1`**.
- Consequencia metodologica: um classificador ingenuo (`DummyClassifier`) que sempre preve `<=50K` obtem Accuracy elevada (`76,07%`), mas Recall `0,00%` para `>50K`. Por isso Accuracy isolada e insuficiente.

## 4. Distribuicao dos Atributos Sensiveis (`sex` e `race`)

> As categorias abaixo reproduzem exatamente as classificacoes historicas do Censo dos EUA de 1994 presentes na fonte.

### Atributo `sex`

| Grupo | Quantidade | % da Base | Positivos (`>50K`) | Negativos (`<=50K`) | Taxa `>50K` |
| --- | ---: | ---: | ---: | ---: | ---: |
| Male | 32.650 | 66,85% | 9.918 | 22.732 | 30,38% |
| Female | 16.192 | 33,15% | 1.769 | 14.423 | 10,93% |

### Atributo `race`

| Grupo | Quantidade | % da Base | Positivos (`>50K`) | Negativos (`<=50K`) | Taxa `>50K` |
| --- | ---: | ---: | ---: | ---: | ---: |
| White | 41.762 | 85,50% | 10.607 | 31.155 | 25,40% |
| Black | 4.685 | 9,59% | 566 | 4.119 | 12,08% |
| Asian-Pac-Islander | 1.519 | 3,11% | 409 | 1.110 | 26,93% |
| Amer-Indian-Eskimo | 470 | 0,96% | 55 | 415 | 11,70% |
| Other | 406 | 0,83% | 50 | 356 | 12,32% |

## 5. Identificadores, Duplicatas e Sobreposicao Treino x Teste

- **Identificador unico de pessoa:** ausente (`possui_identificador_unico = False`).
- **Duplicatas exatas `(X, y)` excedentes:** `52` (`101` linhas envolvidas).
- **Duplicatas exatas apenas em `X`:** `57` (`111` linhas envolvidas).
- **Combinacoes `X` identicas com alvo `class` conflitante (`<=50K` e `>50K`):** `5`.
- **Duplicatas `(X, y)` ignorando `fnlwgt`:** `6.374`.
- **Sobreposicao de indices entre treino e teste:** `0`.
- **Linhas do teste com combinacao `(X, y)` identica no treino:** `17` (e `20` com `X` identico no treino).
- **Politica adotada:** nenhuma linha repetida foi removida automaticamente. Sem identificador unico, perfis demograficos identicos podem pertencer a pessoas diferentes (inclusive com rendas diferentes, como provam os conflitos de rotulo). A pequena sobreposicao exata entre treino e teste (`17` de `9.769` linhas) e documentada de forma transparente sem alterar silenciosamente a divisao historica da V1.

## 6. Variaveis Numericas, Valores Atipicos e `fnlwgt`

| Variavel | Min | Q25 | Mediana | Q75 | Max | Media | Atipicos (1,5x IQR) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `age` | 17,0 | 28,0 | 37,0 | 48,0 | 90,0 | 38,64 | 216 (0,44%) |
| `fnlwgt` | 12285,0 | 117550,5 | 178144,5 | 237642,0 | 1490400,0 | 189664,13 | 1.453 (2,97%) |
| `education-num` | 1,0 | 9,0 | 10,0 | 12,0 | 16,0 | 10,08 | 1.794 (3,67%) |
| `capital-gain` | 0,0 | 0,0 | 0,0 | 0,0 | 99999,0 | 1079,07 | 4.035 (8,26%) |
| `capital-loss` | 0,0 | 0,0 | 0,0 | 0,0 | 4356,0 | 87,50 | 2.282 (4,67%) |
| `hours-per-week` | 1,0 | 40,0 | 40,0 | 45,0 | 99,0 | 40,42 | 13.496 (27,63%) |

### Extremos e Top-Coding Observados

- `capital-gain == 0`: `44.807` (91,74%); valor teto `99.999` em `244` registros (proximo maior valor observado: `41310`).
- `capital-loss == 0`: `46.560` (95,33%); maximo `4356`.
- `hours-per-week`: `40h` em `22.803` registros; `99h` em `137`; `1h` em `27`.
- `age`: idade maxima `90` em `55` registros; minima `17` em `595`.

### Significado Estatistico de `fnlwgt`

- Faixa observada: minimo `12285`, mediana `178144,5`, maximo `1490400` (`28.523` valores unicos; razao max/min = `121,3x`).
- **Interpretacao obrigatoria:** `fnlwgt` e o *final sampling weight* do Current Population Survey (CPS) do US Census Bureau, calibrado por controles demograficos dentro de cada um dos 51 recortes estaduais. Ele indica quantas pessoas da populacao civil nao institucionalizada possuem perfil demografico semelhante naquele estado. Portanto, **cada linha da base nao representa automaticamente o mesmo peso populacional**, nem a distribuicao bruta das linhas equivale a uma amostra uniforme da populacao atual dos EUA.

## 7. Inconsistencias de Codificacao, Categorias Inesperadas e Raras

- Colunas categoricas com marcador literal `'?'` ou espacos laterais: `0`.
- Colunas com categorias fora do vocabulario oficial OpenML 1590: `0`.
- Registros do alvo com ponto final (`<=50K.` / `>50K.`): `0` (espacos laterais: `0`).
- Redundancia `education` vs `education-num`: mapeamento 1-para-1 = `Sim` (`16` niveis categoricos e `16` numericos).
- Registros cruzados `sex` x `relationship`: `Female` + `Husband` = `1`, `Male` + `Wife` = `3`.

### Categorias Raras (< 0,5% da base — total encontradas: `44`)

| Coluna | Categoria | Quantidade | % da Base |
| --- | --- | ---: | ---: |
| `native-country` | Holand-Netherlands | 1 | 0,002% |
| `workclass` | Never-worked | 10 | 0,020% |
| `occupation` | Armed-Forces | 15 | 0,031% |
| `native-country` | Hungary | 19 | 0,039% |
| `native-country` | Honduras | 20 | 0,041% |
| `native-country` | Scotland | 21 | 0,043% |
| `workclass` | Without-pay | 21 | 0,043% |
| `native-country` | Laos | 23 | 0,047% |
| `native-country` | Outlying-US(Guam-USVI-etc) | 23 | 0,047% |
| `native-country` | Yugoslavia | 23 | 0,047% |
| `native-country` | Trinadad&Tobago | 27 | 0,055% |
| `native-country` | Cambodia | 28 | 0,057% |
| `native-country` | Hong | 30 | 0,061% |
| `native-country` | Thailand | 30 | 0,061% |
| `marital-status` | Married-AF-spouse | 37 | 0,076% |
| `native-country` | Ireland | 37 | 0,076% |
| `native-country` | France | 38 | 0,078% |
| `native-country` | Ecuador | 45 | 0,092% |
| `native-country` | Peru | 46 | 0,094% |
| `native-country` | Greece | 49 | 0,100% |
| ... | *(mais 24 categorias raras)* | — | — |
