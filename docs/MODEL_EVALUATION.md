# Avaliação de Modelos, Generalização e Análise de Erros

## 1. Fundamentos Matemáticos dos Modelos Avaliados

### 1.1 `DummyClassifier(strategy="most_frequent")` (Baseline)
Ignora todas as features $X$ e prevê sempre a classe majoritária observada no treino (`<=50K`).
- **Função metodológica:** define o piso mínimo. Como `76,07%` dos registros são `<=50K`, qualquer modelo precisa superar com folga esse patamar — especialmente em Recall e F1 da classe positiva `>50K`, onde a baseline obtém `0,00%`.

### 1.2 `LogisticRegression(max_iter=1000)`
Estima a probabilidade condicional da classe positiva $P(y = 1 \mid x)$ aplicando a função sigmoide a uma combinação linear das features padronizadas e codificadas:

$$P(y = 1 \mid x) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + \dots + \beta_p x_p)}}$$

- Prevê `>50K` quando $P(y = 1 \mid x) \ge 0{,}50$.
- Por aplicar regularização L2 padrão (`C=1.0`) e aprender uma fronteira linear no espaço das variáveis transformadas, exige padronização numérica (`StandardScaler`) para convergência estável (garantida por `max_iter=1000`) e tende a apresentar baixa variância entre treino e validação.

### 1.3 `DecisionTreeClassifier` (Sem Limite vs. `max_depth=7`)
Particiona recursivamente o espaço das variáveis buscando divisões que minimizem a impureza de Gini nos nós filhos:

$$\text{Gini}(S) = 1 - \sum_{k} p_k^2$$

- **Sem controle de profundidade:** a árvore continua dividindo até que quase todas as folhas fiquem puras (contendo 1 ou pouquíssimos exemplos do treino). Isso gera **Accuracy de treino de `99,99%`**, mas aprende ruído amostral e combinações idiossincráticas (ex.: valores específicos de `fnlwgt`), derrubando a Accuracy de validação para `81,10%` (**overfitting severo**, gap de `18,90 p.p.`).
- **Com `max_depth=7`:** limita a árvore a no máximo 7 níveis de decisão (até $2^7 = 128$ folhas). Reduz drasticamente o gap treino − validação para `0,36 p.p.` (`85,84%` vs. `85,48%`), mas, com menos partições disponíveis, prioriza alta Precision (`76,99%`) ao custo de menor Recall (`56,22%`) e maior variação de Recall entre folds (±`2,07%`).

---

## 2. Treino, Validação, Teste, Generalização, Overfitting e Underfitting

| Conceito | Significado Técnico | Evidência Concreta Neste Projeto |
| --- | --- | --- |
| **Desempenho de treinamento** | Métrica medida nos mesmos dados usados no `fit()`. Mede ajuste, não capacidade preditiva real. | `Decision Tree` sem limite atingiu `99,99%` no treino. |
| **Desempenho de validação** | Métrica em partições separadas do treino (CV 5-fold), usada para comparar e selecionar modelos. | `Logistic Regression`: Acc `85,11%`, Recall `59,61%`, F1 `65,70%`. |
| **Desempenho de teste** | Estimativa final do modelo já escolhido em dados separados do treinamento. | `Logistic Regression`: Acc `85,24%`, Recall `58,85%`, F1 `65,62%`. |
| **Generalização** | Capacidade de manter desempenho consistente em dados não vistos no treinamento. | `Logistic Regression` manteve Acc de `85,22%` (treino) → `85,11%` (val) → `85,24%` (teste). |
| **Overfitting** | Alta performance no treino acompanhada de queda relevante na validação (alta variância). | `Decision Tree` sem limite: `99,99%` treino vs. `81,10%` validação (`gap = 18,90 p.p.`). |
| **Underfitting** | Modelo simples demais para capturar os padrões dos dados (alto viés). | `DummyClassifier`: `76,07%` Acc, `0,00%` Recall e `0,00%` F1 em treino e validação. |

---

## 3. Matriz de Confusão e Métricas de Classificação

Com a ordem explícita `labels=["<=50K", ">50K"]`:

- **TN = 6.951:** reais `<=50K` previstos corretamente como `<=50K`.
- **FP = 480:** reais `<=50K` previstos erroneamente como `>50K`.
- **FN = 962:** reais `>50K` previstos erroneamente como `<=50K`.
- **TP = 1.376:** reais `>50K` previstos corretamente como `>50K`.

Fórmulas reconciliadas independentemente em `VAL-06`:

$$\text{Accuracy} = \frac{TP + TN}{TN + FP + FN + TP} = \frac{1376 + 6951}{9769} = 0{,}85239$$

$$\text{Precision} = \frac{TP}{TP + FP} = \frac{1376}{1376 + 480} = 0{,}74138$$

$$\text{Recall (TPR)} = \frac{TP}{TP + FN} = \frac{1376}{1376 + 962} = 0{,}58854$$

$$\text{F1-score} = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = 0{,}65618$$

### `macro avg` vs. `weighted avg`
- **`macro avg`** dá peso `50%` para `<=50K` e `50%` para `>50K`. Como o Recall da classe minoritária é `0,5885` e o da majoritária é `0,9354`, o Recall macro cai para `(0,9354 + 0,5885) / 2 = 0,7620`, expondo o desbalanceamento de desempenho.
- **`weighted avg`** pondera pelo suporte (`7.431` vs. `2.338`). Por isso o Recall ponderado (`0,8524`) coincide com a Accuracy e é fortemente dominado pela classe majoritária `<=50K`.

---

## 4. Leitura Técnica da Análise de Erros

No teste, o modelo errou `1.442` vezes (`962` FN e `480` FP). Dois terços de todos os erros são **falsos negativos** — casos reais de `>50K` classificados como `<=50K`:

1. **Onde ocorrem mais FN:** escolaridade `HS-grad` (`362` FN, Recall `24,74%`) e `Some-college` (`260` FN, Recall `40,37%`), ocupações operacionais/técnicas como `Craft-repair` (`204` FN, Recall `27,92%`), estado civil `Never-married` (`121` FN, Recall `24,84%`) e vínculo `Private` (`663` FN). O modelo linear associa fortemente `>50K` a ensino superior + estado civil `Married-civ-spouse` + cargos executivos/especialistas, subestimando perfis de alta renda fora desse estereótipo estatístico.
2. **Onde ocorrem mais FP:** `Married-civ-spouse` (`466` dos `480` FP totais!), `Husband` (`403` FP), `Bachelors` (`177` FP), `Exec-managerial` (`155` FP) e jornada `>50h` (`138` FP). O mesmo padrão linear que subestima perfis não tradicionais tende a superestimar a chance de `>50K` para indivíduos casados com ensino superior e alta carga horária cuja renda real ficou em `<=50K`.

Essas observações são puramente descritivas do comportamento estatístico da fronteira linear sobre dados de 1994 e não foram usadas para modificar o modelo avaliado.
