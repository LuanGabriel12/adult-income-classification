# Metodologia de Auditoria de Fairness e Experimento com Atributos Sensíveis

## 1. Formulação Matemática das Métricas de Fairness

Para cada grupo $g$ definido por um atributo sensível ($A \in \{\text{sex}, \text{race}\}$), definimos a matriz de confusão restrita ao grupo ($TP_g, FP_g, FN_g, TN_g$), o total do grupo $N_g = TP_g + FP_g + FN_g + TN_g$, o número de positivos reais $P_g = TP_g + FN_g$ e o número de negativos reais $Q_g = TN_g + FP_g$:

1. **Accuracy por grupo:**
   $$\text{Accuracy}_g = \frac{TP_g + TN_g}{N_g} \quad (N_g > 0)$$
2. **Recall / True Positive Rate (TPR) por grupo:**
   $$\text{TPR}_g = \frac{TP_g}{TP_g + FN_g} = \frac{TP_g}{P_g} \quad (P_g > 0)$$
   Mede, entre os indivíduos do grupo que realmente pertencem à classe `>50K`, qual proporção o modelo conseguiu identificar.
3. **False Positive Rate (FPR) por grupo:**
   $$\text{FPR}_g = \frac{FP_g}{FP_g + TN_g} = \frac{FP_g}{Q_g} \quad (Q_g > 0)$$
   Mede, entre os indivíduos do grupo que realmente pertencem à classe `<=50K`, qual proporção foi incorretamente classificada como `>50K`.
4. **Selection Rate por grupo:**
   $$\text{SR}_g = \frac{TP_g + FP_g}{N_g} \quad (N_g > 0)$$
   Mede a proporção de indivíduos do grupo que recebeu a previsão positiva `>50K` (base de critérios de *Demographic Parity*).
5. **Difference (`MetricFrame.difference()`):**
   $$\Delta(M) = \max_{g} M_g - \min_{g} M_g$$
6. **Ratio (`MetricFrame.ratio()`):**
   $$\text{Ratio}(M) = \frac{\min_{g} M_g}{\max_{g} M_g} \quad (\max_g M_g > 0)$$
7. **Equal Opportunity Difference (EOD):**
   $$\text{EOD} = \max_{g} \text{TPR}_g - \min_{g} \text{TPR}_g$$
   Avalia a maior disparidade de oportunidade (taxa de verdadeiros positivos / Recall) entre os grupos. Quanto mais próximo de `0`, menor a diferença de Recall entre o grupo melhor e o pior atendido.
8. **Equalized Odds Difference (EOdds):**
   $$\text{EOdds} = \max\left(\max_g \text{TPR}_g - \min_g \text{TPR}_g,\; \max_g \text{FPR}_g - \min_g \text{FPR}_g\right)$$
   Considera simultaneamente a disparidade de verdadeiros positivos (TPR) e de falsos positivos (FPR), retornando a maior das duas diferenças.

---

## 2. Por Que a Média Simples Entre Grupos Não Substitui a Métrica Global

Uma armadilha matemática frequente é tirar a média aritmética simples das métricas por grupo esperando recuperar o valor global. Isso falha sempre que os grupos têm tamanhos ou taxas-base diferentes:

- **Accuracy global** pondera por $N_g$: $\sum_g \text{Accuracy}_g \cdot \frac{N_g}{N}$.
- **Recall global** pondera por **positivos reais** $P_g$: $\sum_g \text{TPR}_g \cdot \frac{P_g}{P}$.
- **FPR global** pondera por **negativos reais** $Q_g$: $\sum_g \text{FPR}_g \cdot \frac{Q_g}{Q}$.
- **Selection Rate global** pondera por $N_g$: $\sum_g \text{SR}_g \cdot \frac{N_g}{N}$.

**Evidência real no teste do Adult (`race`, `Logistic Regression`):**
- O **Recall global real** é `0,5885` (reconciliado com tolerância $\le 10^{-12}$ pela soma ponderada por $P_g$ em `VAL-11`).
- A **média simples** dos 5 grupos de `race` é `0,4571` — um desvio de `-0,1315` (`13,15 p.p.`), porque `White` concentra `2.104` dos `2.338` positivos reais (`90,0%`), enquanto `Other` (`10` positivos) e `Amer-Indian-Eskimo` (`13` positivos) pesam `20%` cada na média simples apesar de somarem menos de `1%` dos positivos reais!

---

## 3. Proteção Contra Denominadores Iguais a Zero e Sensibilidade Amostral

1. **Denominador zero ($P_g = 0$ ou $Q_g = 0$) e escopo explícito (integral vs. parcial vs. indefinido):** em `src/analysis/fairness.py`, as funções `recall_seguro` e `fpr_seguro` retornam `float("nan")` quando o denominador é zero e marcam `recall_indefinido=True` ou `fpr_indefinido=True`, **nunca** mascarando um grupo sem positivos reais como se tivesse Recall `0,0`. Além disso, `resumo`, `fairness_summary.csv`, `test_metrics.json`, os relatórios Markdown e a CLI registram explicitamente o escopo (`equal_opportunity_escopo` e `equalized_odds_escopo`):
   - `integral_todos_os_grupos`: todos os grupos possuem $P_g > 0$ e $Q_g > 0$ (caso do Adult oficial).
   - `parcial_exclui_grupos_denominador_zero`: ambos os eixos (TPR e FPR) têm $\ge 2$ grupos válidos, mas ao menos um grupo foi excluído por denominador zero — sinalizado explicitamente como **comparação parcial, não integral**, listando `grupos_excluidos_equalized_odds`.
   - `indefinido_eixo_tpr_ou_fpr_ausente` / `indefinido_ambos_eixos_indisponiveis`: se faltar um dos eixos (menos de 2 grupos válidos em TPR ou em FPR), `equalized_odds_difference` retorna `None` para nunca apresentar uma avaliação incompleta de eixo único como Equalized Odds integral (validado em `VAL-12`, `VAL-15` e na suíte Pytest).
2. **Amostras pequenas no conjunto de teste:**
   - `Other`: `n = 82`, apenas **`10` positivos reais** (`TP=3, FN=7`, Recall = `0,3000`). Cada única observação positiva alterada muda o Recall em **`10 p.p.`**.
   - `Amer-Indian-Eskimo`: `n = 96`, apenas **`13` positivos reais** (`TP=4, FN=9`, Recall = `0,3077`). Cada única observação positiva alterada muda o Recall em **`7,7 p.p.`**.
   - Por isso, `EOD (race) = 0,321951` no teste (`0,6220` em `Asian-Pac-Islander` menos `0,3000` em `Other`) deve sempre ser interpretado junto ao suporte real de cada grupo — razão pela qual `fairness_by_race.csv` exibe `count`, `positivos_reais`, `negativos_reais` e a flag `amostra_pequena`.

---

## 4. Experimento Controlado: Cenário A vs. Cenário B (Com vs. Sem `sex` e `race`)

Em `src/analysis/feature_comparison.py`, comparamos sob exatamente o mesmo split, mesmos 5 folds estratificados (`StratifiedKFold(n_splits=5, shuffle=False)`) e mesmo modelo (`Logistic Regression`):
- **Cenário A:** `14` features originais (incluindo `sex` e `race`).
- **Cenário B:** `12` features (`sex` e `race` removidos), preservando `X[["sex", "race"]]` com índices rigorosamente alinhados para auditoria no Fairlearn.

### Resultados na Validação Cruzada Out-of-Fold no Treino (`39.073` registros — protocolo primário)
- **Desempenho global:** Accuracy `0,8511 -> 0,8515` (`+0,0004`), Recall `0,5961 -> 0,5946` (`-0,0015`), Precision `0,7318 -> 0,7341` (`+0,0022`), F1 `0,6570 -> 0,6570` (`0,0000`).
- **`sex`:** EOD `0,1198 -> 0,0972` (`-0,0227`), Selection Rate Diff `0,1815 -> 0,1756` (`-0,0059`), FPR Diff `0,0762 -> 0,0730` (`-0,0032`).
- **`race`:** EOD `0,2979 -> 0,1752` (`-0,1228`), Selection Rate Diff `0,2031 -> 0,1683` (`-0,0348`), FPR Diff `0,0878 -> 0,0742` (`-0,0136`).
- **Por subgrupo (CV OOF):** no Cenário B, o Recall subiu para `Female` (`0,4943 -> 0,5121`), `Black` (`0,4279 -> 0,5240`), `Amer-Indian-Eskimo` (`0,3810 -> 0,4762`) e `Other` (`0,5000 -> 0,5250`), enquanto caiu ligeiramente para `Asian-Pac-Islander` (`0,6789 -> 0,6514`), `White` (`0,6031 -> 0,5970`) e `Male` (`0,6142 -> 0,6092`).

### Interpretação Crítica: Persistência de Disparidades e Limites da Inferência Causal
O experimento demonstra empiricamente que remover `sex` e `race` reduz parte de `EOD`, mas **não elimina** as disparidades observadas (*fairness through unawareness* é insuficiente neste conjunto de dados). Uma **hipótese plausível** amplamente discutida na literatura é que variáveis remanescentes fortemente correlacionadas aos atributos sensíveis — como **`relationship`** (`Husband` quase exclusivamente `Male`; `Wife` quase exclusivamente `Female`), **`marital-status`**, **`occupation`**, **`hours-per-week`**, **`workclass`** e **`native-country`** — **possam atuar como *proxies***, em conjunto com diferenças históricas nas distribuições de atributos e nas taxas-base entre grupos. Entretanto, **a remoção isolada apenas de `sex` e `race` não isola nem comprova causalmente** qual fração da disparidade restante decorre de cada variável específica; validar causalmente essa hipótese exigiria ablações controladas adicionais ou análise causal dedicada. Além disso, reduzir uma única métrica de disparidade não torna um cenário automaticamente superior sem avaliar o conjunto de métricas, trade-offs e contexto de aplicação.
