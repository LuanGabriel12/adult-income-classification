# Metodologia Experimental — Adult Income: Model Evaluation & Fairness V2

## 1. Princípios Metodológicos

Este laboratório segue seis regras rígidas de rigor científico:

1. **Separação estrita entre treino, validação e teste:** todas as decisões de comparação e seleção ocorrem exclusivamente em `X_train`/`y_train` via validação cruzada estratificada de 5 divisões.
2. **Prevenção estrutural de *data leakage*:** imputação (`SimpleImputer`), padronização (`StandardScaler`) e codificação (`OneHotEncoder`) permanecem encapsuladas dentro de `Pipeline` + `ColumnTransformer`, sendo ajustadas somente no conjunto de treino de cada fold.
3. **Critério de seleção pré-registrado:** a regra de escolha do modelo é definida em código (`src/config.py` e `src/modeling/evaluation.py`) antes de observar os resultados de teste.
4. **Honestidade sobre o teste histórico:** o split `test_size=0.20, random_state=42, stratify=y` já havia sido avaliado na V1; portanto, na V2 ele é tratado como referência histórica controlada, nunca como avaliação inédita de produção, e jamais usado para tuning.
5. **Reprodutibilidade verificável:** cópia local canônica verificada por `SHA-256`, seeds determinísticos (`random_state=42`), relatórios determinísticos byte a byte e suporte preservado a `--source openml`.
6. **Reconciliação matemática independente:** nenhuma execução grava relatórios se qualquer uma das 16 verificações (`VAL-01` a `VAL-16`) falhar.

---

## 2. Desenho Experimental Passo a Passo

### Etapa 1 — Carregamento e Validação Estrutural
- Carrega `data/adult_openml_1590_v2.csv` (validando `SHA-256`, ausência de cabeçalhos duplicados, ausência de deslocamento de colunas `RangeIndex(start=0, step=1)`, presença das 14 features e classes estritamente em `("<=50K", ">50K")`) ou diretamente via `fetch_openml(name="adult", version=2, as_frame=True)`.

### Etapa 2 — Divisão Estratificada 80/20
- `train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)`:
  - Treino: `39.073` linhas (`29.724` `<=50K` / `9.349` `>50K`, `23,93%` positivos).
  - Teste: `9.769` linhas (`7.431` `<=50K` / `2.338` `>50K`, `23,93%` positivos).
- Índices originais preservados, `0` sobreposição de índices (`VAL-02`).

### Etapa 3 — Auditoria de Qualidade (Sem Remoção Silenciosa)
- Audita tipos, ausentes, duplicatas (`X + y` e `X`), conflitos de rótulo, sobreposição exata entre treino e teste (`17` linhas em `(X, y)`, `20` em `X`), `fnlwgt`, valores extremos e categorias raras.
- Nenhuma linha é descartada automaticamente, preservando a integridade histórica e evitando suposições não comprováveis na ausência de chave primária.

### Etapa 4 — Validação Cruzada 5-Fold (Somente no Treino)
- Compara os 4 modelos obrigatórios:
  1. `DummyClassifier(strategy="most_frequent")`
  2. `LogisticRegression(max_iter=1000)`
  3. `DecisionTreeClassifier(random_state=42)` (sem limite de profundidade)
  4. `DecisionTreeClassifier(max_depth=7, random_state=42)`
- Calcula, para cada fold e na média agregada, **Accuracy, Precision, Recall e F1 (`pos_label=">50K"`)** tanto no treino quanto na validação, o **gap treino − validação** e a dispersão entre folds (desvio-padrão, mínimo, máximo e amplitude em Python puro).

### Etapa 5 — Seleção Pré-Registrada e Avaliação Final
- Exclui `DummyClassifier` (piso mínimo), exige superação da baseline em Recall e F1, rejeita qualquer modelo com `gap_train_val_accuracy > 0,05` (descartando a árvore sem limite com gap de `18,90 p.p.`) e escolhe o maior F1 de validação (`Logistic Regression`, `65,70%` vs. `64,94%` da árvore `max_depth=7`).
- Ajusta uma nova instância limpa do modelo selecionado em `X_train` e avalia em `X_test`.

### Etapa 6 — Diagnóstico de Erros e Auditoria de Fairness
- Classifica cada previsão de `X_test` em `TP`, `TN`, `FP` ou `FN` e agrega padrões descritivos em 7 dimensões.
- Audita `sex` e `race` com `MetricFrame`, `equal_opportunity_difference` e `equalized_odds_difference`, com proteção explícita contra denominadores zero e sinalização de grupos pequenos (`n < 100` ou `positivos_reais < 30`).

### Etapa 7 — Experimento Controlado com Atributos Sensíveis (Cenário A vs. Cenário B)
- **Cenário A:** todas as `14` features originais (incluindo `sex` e `race`).
- **Cenário B:** `12` features (`X.drop(columns=["sex", "race"])`), preservando `X[["sex", "race"]]` com índices idênticos para o Fairlearn.
- **Comparação primária:** previsões *out-of-fold* nas mesmas 5 divisões estratificadas de `X_train` (`StratifiedKFold(n_splits=5, shuffle=False)`), sem consultar `X_test`.
- **Análise descritiva complementar:** ambos os cenários também são avaliados em `X_test` para transparência descritiva, sem usar o teste para alterar escolhas.

### Etapa 8 — Validação Matemática Independente (`VAL-01` a `VAL-16`)
- Reconcilia contagens de split, índices disjuntos, estratificação, matriz de confusão, recálculo manual de todas as métricas, `classification_report`, somas de erros nas 7 dimensões, conservação por grupo, ponderação correta vs. média simples, recálculo manual de EOD/EOdds, alinhamento do Cenário B, evidência anti-leakage no `StandardScaler`, ausência de mascaramento de denominadores zero e paridade histórica V1 (apenas quando `SHA-256` oficial é confirmado).
