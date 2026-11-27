# Dicionário de Dados — Adult / Census Income (OpenML ID 1590, v2)

## 1. Proveniência, Autoria e Licença

- **Dataset:** Adult / Census Income.
- **Autoria original:** Barry Becker e Ronny Kohavi (Silicon Graphics, 1996), extraído do banco de dados do Censo dos EUA de 1994 (*Current Population Survey — CPS*).
- **Condições históricas de extração (documentadas na fonte):** `((AAGE > 16) && (AGI > 100) && (AFNLWGT > 1) && (HRSWK > 0))`.
- **UCI Machine Learning Repository (ID 2):** [https://archive.ics.uci.edu/dataset/2/adult](https://archive.ics.uci.edu/dataset/2/adult) · DOI `https://doi.org/10.24432/C5XW20` · Licença **CC BY 4.0**.
- **OpenML (ID 1590, versão 2):** [https://www.openml.org/d/1590](https://www.openml.org/d/1590) · Campo `licence: Public` · `file_id=1595261` · ARFF `MD5: bb6510925e5d4b23d136715febb2cdf5`.
- **Cópia local canônica verificável:** `data/adult_openml_1590_v2.csv` (`48.842` linhas × `15` colunas, `5.271.056` bytes, UTF-8, `LF`, `SHA-256 = 8f970e20c9991979ec0a9ccb65ce97543d678709a2c6f8ef23f7fb32f40adddc`).

---

## 2. Diferenças Entre UCI Bruto e OpenML 1590 v2

| Aspecto | Arquivos Brutos UCI (`adult.data` + `adult.test`) | OpenML ID 1590 v2 / Cópia Local Canônica |
| --- | --- | --- |
| **Particionamento** | Separado em `adult.data` (`32.561`) e `adult.test` (`16.281`) | Unificado em `48.842` linhas (registros do antigo teste aparecem primeiro) |
| **Rótulos de `class`** | `adult.test` traz ponto final (`<=50K.` / `>50K.`) | Padronizado em `<=50K` (`37.155`) e `>50K` (`11.687`) |
| **Valores ausentes** | Codificados como `"?"` (com espaço inicial no CSV bruto) | Convertidos em ausentes reais (`NaN` / campo vazio `""`) |

---

## 3. Catálogo de Variáveis

| Coluna | Papel | Tipo Lógico | Valores Únicos | Ausentes | Descrição e Observações Auditadas |
| --- | --- | --- | ---: | ---: | --- |
| `age` | Feature | Numérica (`int64`) | `74` | `0` | Idade do respondente em anos (`17` a `90`; `55` registros no teto `90`, `595` com `17`). |
| `workclass` | Feature | Categórica (`str`) | `8` | `2.799` (`5,73%`) | Classe de trabalho (`Private` = `33.906`; raras: `Without-pay` = `21`, `Never-worked` = `10`). |
| `fnlwgt` | Feature | Numérica (`int64`) | `28.523` | `0` | *Final sampling weight* do CPS (`12.285` a `1.490.400`, mediana `178.144,5`). Peso populacional calibrado por estado — ver Seção 4. |
| `education` | Feature | Categórica (`str`) | `16` | `0` | Escolaridade nominal (`HS-grad` = `15.784`, `Some-college` = `10.878`, `Bachelors` = `8.025`, `Preschool` = `83`). Redundância 1-para-1 com `education-num`. |
| `education-num` | Feature | Numérica (`int64`) | `16` | `0` | Versão ordinal numérica (`1` a `16`) estritamente 1-para-1 com `education`. |
| `marital-status` | Feature | Categórica (`str`) | `7` | `0` | Estado civil (`Married-civ-spouse` = `22.379`, `Never-married` = `16.117`; rara: `Married-AF-spouse` = `37`). |
| `occupation` | Feature | Categórica (`str`) | `14` | `2.809` (`5,75%`) | Ocupação (`Prof-specialty` = `6.172`, `Craft-repair` = `6.112`, `Exec-managerial` = `6.086`; rara: `Armed-Forces` = `15`). Todos os `10` ausentes extras sobre `workclass` são `Never-worked`. |
| `relationship` | Feature | Categórica (`str`) | `6` | `0` | Papel no domicílio (`Husband` = `19.716`, `Not-in-family` = `12.583`, `Wife` = `2.331`). Fortemente associada a `sex` e à renda na amostra (candidata a variável *proxy*). Há `4` cruzamentos atípicos (`Female`+`Husband`: `1`; `Male`+`Wife`: `3`). |
| `race` | Feature / **Sensível** | Categórica (`str`) | `5` | `0` | Classificação histórica do Censo 1994 (`White` `85,50%`, `Black` `9,59%`, `Asian-Pac-Islander` `3,11%`, `Amer-Indian-Eskimo` `0,96%`, `Other` `0,83%`). |
| `sex` | Feature / **Sensível** | Categórica (`str`) | `2` | `0` | Classificação binária histórica do Censo 1994 (`Male` `66,85%`, `Female` `33,15%`). Taxa `>50K`: `30,38%` (`Male`) vs. `10,93%` (`Female`). |
| `capital-gain` | Feature | Numérica (`int64`) | `123` | `0` | Ganho de capital anual (`91,74%` igual a `0`; `244` registros com valor teto/censurado `99.999`; próximo maior valor é `41.310`). |
| `capital-loss` | Feature | Numérica (`int64`) | `99` | `0` | Perda de capital anual (`95,33%` igual a `0`; máximo `4.356`). |
| `hours-per-week` | Feature | Numérica (`int64`) | `96` | `0` | Horas trabalhadas por semana (`1` a `99`; moda `40h` com `22.803` registros; `137` no teto `99h`). |
| `native-country` | Feature | Categórica (`str`) | `41` | `857` (`1,75%`) | País/território de origem (`United-States` = `43.832` [89,74%]; `Holand-Netherlands` = `1` único registro; grafias históricas como `Columbia`, `Trinadad&Tobago`, `Yugoslavia`). |
| `class` | **Target** | Binária (`str`) | `2` | `0` | Faixa de renda anual observada em 1994: `<=50K` (`37.155` — `76,07%`, classe negativa) e `>50K` (`11.687` — `23,93%`, classe positiva). Razão `3,18:1`. |

---

## 4. Interpretação Crítica de `fnlwgt` e Representatividade

O campo `fnlwgt` é o peso final de amostragem calculado pelo US Census Bureau para alinhar a amostra do CPS a três conjuntos de controles demográficos (população 16+ por estado, origem hispânica por idade/sexo e raça por idade/sexo). Como o CPS combina **51 amostras estaduais com probabilidades de seleção distintas**, pessoas com perfil parecido só recebem pesos parecidos *dentro do mesmo estado*.

**Consequências analíticas obrigatórias:**
1. Não é válido interpretar cada linha como representante de peso idêntico da população dos EUA.
2. A amostra tem forte sub-representação de `Female` (`33,15%`), `Amer-Indian-Eskimo` (`0,96%`) e `Other` (`0,83%`), o que impacta diretamente a estabilidade estatística das métricas de fairness.
3. Ausência de identificador único impede distinguir se as `52` linhas exatamente repetidas (`X + y`) são duplicações de coleta ou pessoas distintas com mesmo perfil demográfico — motivo pelo qual a V2 documenta e preserva todas as linhas sem remoção silenciosa.
