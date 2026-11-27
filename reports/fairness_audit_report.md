# Relatorio de Auditoria de Fairness e Experimento com Atributos Sensiveis

> **Enquadramento cientifico e etico:** Este relatorio quantifica disparidades observadas de desempenho entre grupos demograficos historicos do Censo dos EUA de 1994 sob definicoes matematicas especificas. **Ele nao certifica ausencia de discriminacao, nao prova causalidade nem discriminacao intencional por si so, e nao autoriza uso do modelo em decisoes reais sobre pessoas** (contratacao, credito, renda ou emprego).

## 1. Definicoes Matematicas das Metricas Auditadas

| Metrica | Formula | Denominador | O que quantifica |
| --- | --- | --- | --- |
| **Accuracy** | `(TP + TN) / N` | Total do grupo (`N_g`) | Proporcao geral de acertos no grupo |
| **Recall (TPR)** | `TP / (TP + FN)` | Positivos reais do grupo | Fracao dos casos realmente `>50K` identificados |
| **False Positive Rate (FPR)** | `FP / (FP + TN)` | Negativos reais do grupo | Fracao dos casos `<=50K` previstos erroneamente como `>50K` |
| **Selection Rate** | `(TP + FP) / N` | Total do grupo (`N_g`) | Proporcao do grupo que recebe previsao positiva `>50K` |
| **Equal Opportunity Difference (EOD)** | `max(TPR_g) - min(TPR_g)` | Grupos com `TP_g + FN_g > 0` (sinalizado como **parcial** se algum grupo for excluido) | Maior distancia de Recall entre grupos |
| **Equalized Odds Difference (EOdds)** | `max(diff_TPR, diff_FPR)` | Exige >= 2 grupos validos em TPR e >= 2 em FPR; sinalizado como **parcial** se algum grupo tiver denominador zero, ou **indefinido** se faltar um eixo | Maior disparidade combinada entre TPR e FPR |
| **Ratio** | `min(M_g) / max(M_g)` | `max(M_g) > 0` | Razao relativa entre pior e melhor valor da metrica |

## 2. Auditoria de Fairness no Conjunto de Teste (Modelo Selecionado — Cenario A)

### 2.1 Desempenho por `sex`

| Grupo | N | Positivos Reais | Negativos Reais | Accuracy | Recall (TPR) | FPR | Selection Rate | Observacao |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Female | 3.259 | 361 | 2.898 | 0,9257 | 0,5069 | 0,0221 | 0,0758 | Denominadores validos |
| Male | 6.510 | 1.977 | 4.533 | 0,8157 | 0,6034 | 0,0918 | 0,2472 | Denominadores validos |

- **Equal Opportunity Difference (`sex`):** `0,096514` (escopo: `integral_todos_os_grupos` — integral — todos os grupos com denominadores validos).
- **Equalized Odds Difference (`sex`):** `0,096514` (escopo: `integral_todos_os_grupos` — integral — todos os grupos com denominadores validos).
- **Diferencas (`max - min`):** Accuracy `0,1101`, Recall `0,0965`, FPR `0,0697`, Selection Rate `0,1714`.
- **Razoes (`min / max`):** Accuracy `0,8811`, Recall `0,8401`, FPR `0,2406`, Selection Rate `0,3066`.
- **Grupos excluidos por denominador zero em EOdds:** `Nenhum`.
- **Grupos com amostra pequena / baixa contagem de positivos:** `Nenhum`.
- **Verificacao de denominadores zero:** Todos os grupos possuem positivos e negativos reais > 0 (comparacao integral).

### 2.2 Desempenho por `race`

| Grupo | N | Positivos Reais | Negativos Reais | Accuracy | Recall (TPR) | FPR | Selection Rate | Observacao |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Amer-Indian-Eskimo | 96 | 13 | 83 | 0,8958 | 0,3077 | 0,0120 | 0,0521 | Amostra pequena (n=96, positivos_reais=13) |
| Asian-Pac-Islander | 311 | 82 | 229 | 0,8264 | 0,6220 | 0,1004 | 0,2379 | Denominadores validos |
| Black | 968 | 129 | 839 | 0,9081 | 0,4574 | 0,0226 | 0,0806 | Denominadores validos |
| Other | 82 | 10 | 72 | 0,8902 | 0,3000 | 0,0278 | 0,0610 | Amostra pequena (n=82, positivos_reais=10) |
| White | 8.312 | 2.104 | 6.208 | 0,8460 | 0,5984 | 0,0701 | 0,2038 | Denominadores validos |

- **Equal Opportunity Difference (`race`):** `0,321951` (escopo: `integral_todos_os_grupos` — integral — todos os grupos com denominadores validos).
- **Equalized Odds Difference (`race`):** `0,321951` (escopo: `integral_todos_os_grupos` — integral — todos os grupos com denominadores validos).
- **Diferencas (`max - min`):** Accuracy `0,0817`, Recall `0,3220`, FPR `0,0884`, Selection Rate `0,1859`.
- **Razoes (`min / max`):** Accuracy `0,9100`, Recall `0,4824`, FPR `0,1200`, Selection Rate `0,2189`.
- **Grupos excluidos por denominador zero em EOdds:** `Nenhum`.
- **Grupos com amostra pequena / baixa contagem de positivos:** `Amer-Indian-Eskimo, Other`.
- **Verificacao de denominadores zero:** Todos os grupos possuem positivos e negativos reais > 0 (comparacao integral).

## 3. Influencia dos Tamanhos Amostrais e Ponderacao Correta vs. Media Simples

Como os grupos possuem tamanhos e taxas-base muito diferentes, a **media simples** das metricas por grupo **nao** equivale a metrica global. Cada metrica global exige ponderacao pelo seu proprio denominador:

| Atributo | Metrica | Global Real | Recalculo Ponderado | Media Simples dos Grupos | Desvio da Media Simples | Denominador / Peso Correto |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `sex` | `accuracy` | 0,8524 | 0,8524 | 0,8707 | 0,0183 | count (N_g) |
| `sex` | `recall` | 0,5885 | 0,5885 | 0,5552 | -0,0334 | positivos_reais (TP_g + FN_g) |
| `sex` | `fpr` | 0,0646 | 0,0646 | 0,0569 | -0,0077 | negativos_reais (TN_g + FP_g) |
| `sex` | `selection_rate` | 0,1900 | 0,1900 | 0,1615 | -0,0285 | count (N_g) |
| `race` | `accuracy` | 0,8524 | 0,8524 | 0,8733 | 0,0209 | count (N_g) |
| `race` | `recall` | 0,5885 | 0,5885 | 0,4571 | -0,1315 | positivos_reais (TP_g + FN_g) |
| `race` | `fpr` | 0,0646 | 0,0646 | 0,0466 | -0,0180 | negativos_reais (TN_g + FP_g) |
| `race` | `selection_rate` | 0,1900 | 0,1900 | 0,1271 | -0,0629 | count (N_g) |

## 4. Experimento Controlado: Cenario A (Com `sex` e `race`) vs. Cenario B (Sem `sex` e `race`)

- **Modelo avaliado:** `Logistic Regression (max_iter=1000)` (mesmos hiperparametros, mesmo split e mesmos `5` folds estratificados).
- **Cenario A:** `14` features originais (incluindo `sex` e `race`).
- **Cenario B:** `12` features (removendo `sex` e `race`, mantendo os atributos sensiveis em tabela paralela com indices 100% alinhados para o Fairlearn).
- **Protocolo:** Comparacao primaria executada via validacao cruzada out-of-fold no treino (StratifiedKFold 5-fold); avaliacao no teste historico reportada apenas como diagnostico descritivo de ambos os cenarios, sem uso para selecao.
- **Escopo das comparacoes de fairness (Cenarios A e B):** integral em todos os grupos (todos os grupos possuem positivos e negativos reais > 0).

| Escopo | Categoria | Atributo | Metrica | Cenario A (com sensiveis) | Cenario B (sem sensiveis) | Delta (B − A) |
| --- | --- | --- | --- | ---: | ---: | ---: |
| `validacao_cv_treino_oof` | `desempenho_geral` | `global` | `accuracy` | 0,8511 | 0,8515 | 0,0004 |
| `validacao_cv_treino_oof` | `desempenho_geral` | `global` | `recall` | 0,5961 | 0,5946 | -0,0015 |
| `validacao_cv_treino_oof` | `desempenho_geral` | `global` | `precision` | 0,7318 | 0,7341 | 0,0022 |
| `validacao_cv_treino_oof` | `desempenho_geral` | `global` | `f1` | 0,6570 | 0,6570 | 0,0000 |
| `validacao_cv_treino_oof` | `fairness_disparidade` | `sex` | `equal_opportunity_difference` | 0,1198 | 0,0972 | -0,0227 |
| `validacao_cv_treino_oof` | `fairness_disparidade` | `sex` | `equalized_odds_difference` | 0,1198 | 0,0972 | -0,0227 |
| `validacao_cv_treino_oof` | `fairness_disparidade` | `sex` | `fpr_diff` | 0,0762 | 0,0730 | -0,0032 |
| `validacao_cv_treino_oof` | `fairness_disparidade` | `sex` | `selection_rate_diff` | 0,1815 | 0,1756 | -0,0059 |
| `validacao_cv_treino_oof` | `fairness_disparidade` | `race` | `equal_opportunity_difference` | 0,2979 | 0,1752 | -0,1228 |
| `validacao_cv_treino_oof` | `fairness_disparidade` | `race` | `equalized_odds_difference` | 0,2979 | 0,1752 | -0,1228 |
| `validacao_cv_treino_oof` | `fairness_disparidade` | `race` | `fpr_diff` | 0,0878 | 0,0742 | -0,0136 |
| `validacao_cv_treino_oof` | `fairness_disparidade` | `race` | `selection_rate_diff` | 0,2031 | 0,1683 | -0,0348 |
| `teste_historico_descritivo` | `desempenho_geral` | `global` | `accuracy` | 0,8524 | 0,8517 | -0,0007 |
| `teste_historico_descritivo` | `desempenho_geral` | `global` | `recall` | 0,5885 | 0,5860 | -0,0026 |
| `teste_historico_descritivo` | `desempenho_geral` | `global` | `precision` | 0,7414 | 0,7401 | -0,0012 |
| `teste_historico_descritivo` | `desempenho_geral` | `global` | `f1` | 0,6562 | 0,6541 | -0,0021 |
| `teste_historico_descritivo` | `fairness_disparidade` | `sex` | `equal_opportunity_difference` | 0,0965 | 0,0738 | -0,0227 |
| `teste_historico_descritivo` | `fairness_disparidade` | `sex` | `equalized_odds_difference` | 0,0965 | 0,0738 | -0,0227 |
| `teste_historico_descritivo` | `fairness_disparidade` | `sex` | `fpr_diff` | 0,0697 | 0,0688 | -0,0009 |
| `teste_historico_descritivo` | `fairness_disparidade` | `sex` | `selection_rate_diff` | 0,1714 | 0,1669 | -0,0045 |
| `teste_historico_descritivo` | `fairness_disparidade` | `race` | `equal_opportunity_difference` | 0,3220 | 0,1307 | -0,1913 |
| `teste_historico_descritivo` | `fairness_disparidade` | `race` | `equalized_odds_difference` | 0,3220 | 0,1307 | -0,1913 |
| `teste_historico_descritivo` | `fairness_disparidade` | `race` | `fpr_diff` | 0,0884 | 0,0763 | -0,0120 |
| `teste_historico_descritivo` | `fairness_disparidade` | `race` | `selection_rate_diff` | 0,1859 | 0,1417 | -0,0441 |

### Variacao por Subgrupo na Validacao Cruzada (Treino Out-of-Fold)

| Atributo | Grupo | N (Treino) | Positivos Reais | Recall A | Recall B | Delta Recall (B−A) | Selection Rate A | Selection Rate B |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `sex` | Female | 12.933 | 1.408 | 0,4943 | 0,5121 | 0,0178 | 0,0735 | 0,0763 |
| `sex` | Male | 26.140 | 7.941 | 0,6142 | 0,6092 | -0,0049 | 0,2550 | 0,2520 |
| `race` | Amer-Indian-Eskimo | 374 | 42 | 0,3810 | 0,4762 | 0,0952 | 0,0535 | 0,0775 |
| `race` | Asian-Pac-Islander | 1.208 | 327 | 0,6789 | 0,6514 | -0,0275 | 0,2566 | 0,2459 |
| `race` | Black | 3.717 | 437 | 0,4279 | 0,5240 | 0,0961 | 0,0716 | 0,0917 |
| `race` | Other | 324 | 40 | 0,5000 | 0,5250 | 0,0250 | 0,0772 | 0,0833 |
| `race` | White | 33.450 | 8.503 | 0,6031 | 0,5970 | -0,0061 | 0,2091 | 0,2057 |

### Interpretacao Tecnica: Persistencia de Disparidades Apos a Remocao de `sex` e `race`

1. **Evidencia empirica observada vs. limites da inferencia causal:** o experimento controlado mostra empiricamente que remover `sex` e `race` (Cenario B) reduz parte de `EOD`, mas **nao elimina** as disparidades observadas (*fairness through unawareness* mostra-se insuficiente neste conjunto de dados). Uma **hipotese plausivel** discutida na literatura e que variaveis remanescentes correlacionadas aos atributos sensiveis — como `relationship`, `marital-status`, `occupation`, `hours-per-week`, `workclass` e `native-country` — **possam atuar como *proxies***, em conjunto com diferencas historicas de distribuicao de atributos e taxas-base entre grupos. Contudo, **a remocao isolada de `sex` e `race` nao testa ablacoes individuais nem comprova causalmente** qual parcela da disparidade restante decorre de cada variavel especifica.
2. **Sem superioridade automatica por metrica isolada:** um cenario nao se torna automaticamente superior apenas por reduzir uma unica metrica de disparidade; e indispensavel examinar conjuntamente Recall, FPR, Selection Rate, tamanho amostral e impacto potencial de falsos positivos e falsos negativos.
3. **Ausencia de mitigacao algoritmica neste laboratorio:** propositalmente nao foram aplicados algoritmos de mitigacao de vieses, preservando o foco em diagnostico honesto e auditavel.
