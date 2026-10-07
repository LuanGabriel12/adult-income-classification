# Interview Notes — Adult Income Classification

## Por que Logistic Regression?

Porque teve Recall e F1 maiores que a árvore controlada, apresentou estabilidade na validação cruzada e manteve uma solução simples de explicar.

## O que é Accuracy?

É a proporção total de previsões corretas. Pode esconder desempenho ruim na classe minoritária quando as classes são desbalanceadas.

## O que é Recall?

Entre os casos realmente `>50K`, mede quantos o modelo identificou. Fórmula: `TP / (TP + FN)`.

## O que é Precision?

Entre as previsões `>50K`, mede quantas estavam corretas. Fórmula: `TP / (TP + FP)`.

## O que é F1?

É a média harmônica entre Precision e Recall. É útil quando quero equilibrar essas duas métricas.

## O que mostra a matriz de confusão?

Ela separa acertos e erros em TN, FP, FN e TP. Neste projeto: `[[6951, 480], [962, 1376]]`.

## O que é falso positivo?

É prever `>50K` para uma pessoa cuja classe real é `<=50K`. O projeto teve 480 falsos positivos.

## O que é falso negativo?

É prever `<=50K` para uma pessoa cuja classe real é `>50K`. O projeto teve 962 falsos negativos.

## Por que usar DummyClassifier?

Para criar uma referência mínima. Ele mostra se o modelo treinado supera uma estratégia ingênua baseada na classe mais frequente.

## O que é Pipeline?

É uma sequência que aplica pré-processamento e modelo de forma consistente. Também ajuda a evitar data leakage na validação.

## O que é ColumnTransformer?

Permite aplicar transformações diferentes a grupos de colunas, como padronização nas numéricas e One-Hot Encoding nas categóricas.

## O que é OneHotEncoder?

Converte categorias em colunas binárias. `handle_unknown="ignore"` evita erro ao aparecer uma categoria nova no teste.

## O que é SimpleImputer?

Preenche valores ausentes. Aqui, categorias ausentes são substituídas pela categoria mais frequente aprendida no treino.

## O que é StandardScaler?

Centraliza e padroniza variáveis numéricas. Isso ajuda a Regressão Logística quando as features estão em escalas diferentes.

## O que é cross-validation?

É dividir o treino em combinações de treino e validação para avaliar desempenho e estabilidade sem consultar o teste final.

## O que é overfitting?

É quando o modelo aprende demais o treino e generaliza mal. A árvore sem controle mostrou Accuracy de treino quase perfeita e validação inferior.

## Para que serve classification_report?

Reúne Precision, Recall, F1 e suporte de cada classe, facilitando a comparação entre a classe majoritária e a minoritária.

## O que é macro avg?

É a média das métricas das classes dando o mesmo peso a cada uma, independentemente do tamanho.

## O que é weighted avg?

É a média ponderada pelo suporte de cada classe. A classe mais frequente influencia mais o resultado.

## O que significa fairness neste projeto?

É uma auditoria de métricas por grupos sensíveis. Ela identifica disparidades, mas não prova sozinha discriminação nem garante justiça.

## O que é FPR?

False Positive Rate mede, entre os casos realmente negativos, quantos foram classificados como positivos: `FP / (FP + TN)`.

## O que é Selection Rate?

É a proporção de exemplos de um grupo que recebeu a previsão positiva `>50K`.

## O que é Equal Opportunity Difference?

É a maior diferença de taxa de verdadeiros positivos, equivalente ao Recall, entre grupos. Quanto mais perto de zero, menor a disparidade nessa métrica.

## O que é Equalized Odds Difference?

Resume disparidades de verdadeiros positivos e falsos positivos entre grupos. Quanto mais perto de zero, menores essas diferenças, mas a interpretação depende do contexto e da amostra.