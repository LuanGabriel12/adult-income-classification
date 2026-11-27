# Relatorio de Avaliacao de Modelos e Analise de Erros

## 1. Protocolo Experimental e Criterio Pre-Registrado de Selecao

- **Divisao:** 80% treino / 20% teste (`random_state=42`, `stratify=y`), sem consulta ao teste durante comparacao ou selecao.
- **Validacao cruzada:** 5 divisoes estratificadas exclusivamente no conjunto de treino.
- **Pre-processamento:** encapsulado em `Pipeline` + `ColumnTransformer` (mediana + `StandardScaler` nas numericas; moda + `OneHotEncoder(handle_unknown='ignore')` nas categoricas), ajustado apenas nos folds de treino.
- **Criterio pre-registrado:** Maior val_f1_mean na validacao cruzada (classe positiva >50K); rejeicao de modelos com gap `Accuracy treino - validacao > 0,05`; desempate: Maior val_recall_mean, seguido por menor gap_train_val_accuracy.
- **Modelo selecionado na validacao cruzada:** **`Logistic Regression (max_iter=1000)`**.

> **Nota metodologica sobre o conjunto de teste:** o split historico de teste (`random_state=42`) ja foi observado na V1. Ele e reportado aqui como avaliacao controlada e reproduzivel — nao como conjunto inedito de producao — e nao foi usado para selecionar modelos ou thresholds.

## 2. Comparacao dos 4 Modelos (Validacao Cruzada 5-Fold no Treino)

| Modelo | Acc Treino | Acc Val (media ± dp) | Gap Acc (Tr−Val) | Recall Val (media ± dp) | Precision Val (media ± dp) | F1 Val (media ± dp) | Decisao |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| DummyClassifier (most_frequent) | 76,07% | 76,07% (±0,00%) | 0,00% | 0,00% (±0,00%) | 0,00% (±0,00%) | 0,00% (±0,00%) | Baseline de referencia minima (nao candidata a modelo final) |
| Logistic Regression (max_iter=1000) | 85,22% | 85,11% (±0,36%) | 0,11% | 59,61% (±1,02%) | 73,20% (±1,14%) | 65,70% (±0,83%) | **SELECIONADO** |
| Decision Tree (sem limite) | 99,99% | 81,10% (±0,38%) | 18,90% | 61,92% (±0,74%) | 60,21% (±0,80%) | 61,05% (±0,75%) | Rejeitado por overfitting severo (gap treino-validacao de Accuracy = 0.1890 > 0.05) |
| Decision Tree (max_depth=7) | 85,84% | 85,48% (±0,36%) | 0,36% | 56,22% (±2,07%) | 76,99% (±1,94%) | 64,94% (±1,13%) | Elegivel pelos criterios de generalizacao e desempenho |

### Diagnostico de Overfitting, Underfitting e Generalizacao

- **Underfitting (Baseline `DummyClassifier`):** preve exclusivamente a classe majoritaria (`<=50K`), alcancando Recall e F1 iguais a `0,00%` para `>50K` (nao aprende padroes das variaveis explicativas).
- **Overfitting severo (`Decision Tree` sem limite):** memoriza particoes individuais do treino (Accuracy de treino proxima de `100%`), gerando gap expressivo frente a validacao e pior Precision/F1 fora da amostra.
- **Controle de complexidade (`Decision Tree max_depth=7`):** limita a profundidade e reduz fortemente o gap treino-validacao, mas sacrifica Recall na classe positiva `>50K`.
- **Equilibrio e estabilidade (`Logistic Regression (max_iter=1000)`):** apresenta gap minimo entre treino e validacao, baixa variacao entre folds e o melhor equilibrio de F1 e Recall na classe positiva.

## 3. Avaliacao Final no Conjunto de Teste

| Metrica | Valor Obtido | Referencia Historica V1 |
| --- | ---: | ---: |
| Accuracy | 85,24% (`0,85239`) | 85,24% |
| Recall (`>50K`) | 58,85% (`0,58854`) | 58,85% |
| Precision (`>50K`) | 74,14% (`0,74138`) | 74,14% |
| F1-score (`>50K`) | 65,62% (`0,65618`) | 65,62% |
| False Positive Rate (`>50K`) | 6,46% (`0,06459`) | — |
| Selection Rate (`>50K`) | 19,00% (`0,18999`) | — |

### Matriz de Confusao (Ordem das classes: `<=50K`, `>50K`)

| Classe Real \ Previsto | Previsto `<=50K` | Previsto `>50K` | Total Real |
| --- | ---: | ---: | ---: |
| **Real `<=50K`** | TN = **6.951** | FP = **480** | 7.431 |
| **Real `>50K`** | FN = **962** | TP = **1.376** | 2.338 |
| **Total Previsto** | 7.913 | 1.856 | **9.769** |

### Classification Report Estruturado

| Classe / Media | Precision | Recall | F1-score | Suporte |
| --- | ---: | ---: | ---: | ---: |
| `<=50K` | 0,8784 | 0,9354 | 0,9060 | 7.431 |
| `>50K` | 0,7414 | 0,5885 | 0,6562 | 2.338 |
| `macro avg` (peso igual por classe) | 0,8099 | 0,7620 | 0,7811 | 9.769 |
| `weighted avg` (ponderada pelo suporte) | 0,8456 | 0,8524 | 0,8462 | 9.769 |

## 4. Analise Descritiva de Erros de Classificacao

- Total de erros no teste: **`1.442`** de `9.769` registros (taxa global de erro: `14,76%`).
- **Falsos Negativos (`FN`):** `962` (`66,71%` de todos os erros; `41,15%` dos positivos reais nao identificados).
- **Falsos Positivos (`FP`):** `480` (`33,29%` de todos os erros; `6,46%` dos negativos reais previstos erroneamente como `>50K`).

### Principais Concentracoes Descritivas de Erro por Dimensao

| Dimensao | Categoria com maior volume de erros | Total | Positivos Reais | FP | FN | Taxa de Erro | Recall do Segmento |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `faixa_etaria` | 35-44 | 2.460 | 826 | 147 | 332 | 19,47% | 59,81% |
| `faixa_etaria` | 45-54 | 1.708 | 669 | 104 | 247 | 20,55% | 63,08% |
| `faixa_etaria` | 25-34 | 2.595 | 444 | 104 | 235 | 13,06% | 47,07% |
| `education` | HS-grad | 3.095 | 481 | 81 | 362 | 14,31% | 24,74% |
| `education` | Some-college | 2.246 | 436 | 88 | 260 | 15,49% | 40,37% |
| `education` | Bachelors | 1.607 | 652 | 177 | 122 | 18,61% | 81,29% |
| `faixa_horas` | 35-40h (padrao) | 5.307 | 1.127 | 200 | 546 | 14,06% | 51,55% |
| `faixa_horas` | 41-50h (estendida) | 1.747 | 680 | 114 | 237 | 20,09% | 65,15% |
| `faixa_horas` | >50h (intensa) | 1.060 | 441 | 138 | 134 | 25,66% | 69,61% |
| `occupation` | Exec-managerial | 1.191 | 550 | 155 | 109 | 22,17% | 80,18% |
| `occupation` | Craft-repair | 1.205 | 283 | 33 | 204 | 19,67% | 27,92% |
| `occupation` | Prof-specialty | 1.227 | 581 | 85 | 146 | 18,83% | 74,87% |
| `workclass` | Private | 6.800 | 1.489 | 271 | 663 | 13,74% | 55,47% |
| `workclass` | Self-emp-not-inc | 799 | 224 | 45 | 106 | 18,90% | 52,68% |
| `workclass` | Local-gov | 584 | 176 | 45 | 63 | 18,49% | 64,20% |
| `marital-status` | Married-civ-spouse | 4.537 | 1.985 | 466 | 713 | 25,99% | 64,08% |
| `marital-status` | Never-married | 3.186 | 161 | 5 | 121 | 3,95% | 24,84% |
| `marital-status` | Divorced | 1.326 | 138 | 5 | 90 | 7,16% | 34,78% |
| `relationship` | Husband | 3.986 | 1.763 | 403 | 643 | 26,24% | 63,53% |
| `relationship` | Not-in-family | 2.478 | 260 | 14 | 175 | 7,63% | 32,69% |
| `relationship` | Wife | 482 | 215 | 59 | 67 | 26,14% | 68,84% |

### Limitacoes e Cautelas Interpretativas da Analise de Erros

- Os padroes acima sao **exclusivamente descritivos**: associacoes entre escolaridade, horas, ocupacao, faixa etaria ou papel familiar e erros do classificador **nao estabelecem causalidade**.
- Nenhuma conclusao aqui se aplica a pessoas reais individuais.
- Este diagnostico foi realizado apos a avaliacao final e **nao** foi utilizado para alterar o modelo, ajustar hiperparametros ou escolher novo threshold.
