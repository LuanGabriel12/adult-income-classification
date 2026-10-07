# Adult Income Classification

Projeto educacional de Machine Learning para prever se a renda anual de uma pessoa pertence à classe `>50K` ou `<=50K`. O trabalho prioriza um fluxo reproduzível, comparação honesta de modelos e análise cuidadosa de métricas de fairness.

## Objetivo

Construir e avaliar um classificador binário para a classe `>50K`, mantendo o conjunto de teste reservado para a avaliação final. Neste estudo, o Recall e o F1 da classe positiva tiveram mais peso na escolha do modelo do que pequenos ganhos de Accuracy.

## Dataset

- **Nome:** Adult / Census Income
- **Origem:** [OpenML — Adult, versão 2](https://www.openml.org/search?type=data&sort=runs&id=1590) e [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/2/adult)
- **Tamanho:** aproximadamente 48.842 registros e 14 features
- **Target:** `class`, com as categorias `<=50K` e `>50K`
- **Carregamento:** `fetch_openml`, sem arquivo de dados local

O dataset reúne atributos demográficos e profissionais. Por ser histórico e possuir variáveis sensíveis, é adequado para estudo, mas não justifica uso direto em decisões reais.

## Problema

Trata-se de uma classificação binária. A classe positiva é `>50K`. Além do desempenho geral, o projeto observa falsos negativos e diferenças de desempenho entre grupos.

## Estrutura do projeto

```text
adult-income-classification/
├── notebooks/
│   └── adult_income_analysis.ipynb
├── .gitignore
├── INTERVIEW_NOTES.md
├── README.md
└── requirements.txt
```

## Preparação dos dados

- Separação de 80% para treino e 20% para teste com `random_state=42`.
- Uso de `stratify=y` para preservar a proporção das classes.
- Imputação categórica pela categoria mais frequente com `SimpleImputer`.
- Padronização numérica com `StandardScaler`.
- Codificação categórica com `OneHotEncoder(handle_unknown="ignore")`.
- Aplicação dos fluxos com `ColumnTransformer`.
- Pré-processamento e modelo unidos em `Pipeline`, reduzindo risco de data leakage.

## Baseline

O `DummyClassifier(strategy="most_frequent")` foi usado como referência mínima. Ele sempre prevê a classe mais frequente e permite verificar se os modelos aprenderam padrões além da distribuição das classes. A baseline foi calculada por validação cruzada somente no treino.

## Modelos avaliados

- Logistic Regression
- Decision Tree sem controle de profundidade
- Decision Tree controlada com `max_depth=7`

## Validação cruzada

A comparação usou validação cruzada com cinco divisões apenas em `X_train` e `y_train`. A árvore sem controle obteve Accuracy de treino próxima de 100%, mas desempenho de validação inferior, sinal de overfitting. A árvore controlada reduziu essa diferença.

## Modelo escolhido

A **Logistic Regression** foi mantida como modelo final porque apresentou Recall e F1 maiores que a Decision Tree controlada, além de boa estabilidade. Para este estudo, identificar a classe `>50K` foi considerado mais importante do que ganhar poucos décimos de Accuracy.

## Resultados finais

| Métrica | Resultado |
|---|---:|
| Accuracy | 0,85239 |
| Recall (`>50K`) | 0,58854 |
| Precision (`>50K`) | 0,74138 |
| F1 (`>50K`) | 0,65618 |

Matriz de confusão, com linhas representando a classe real e colunas a classe prevista:

```text
                 Previsto <=50K   Previsto >50K
Real <=50K             6951             480
Real >50K               962            1376
```

- TN: 6.951
- FP: 480
- FN: 962
- TP: 1.376

## Classification Report

A classe `<=50K` teve Recall de aproximadamente 0,94 e F1 de 0,91. Para `>50K`, o Recall ficou em cerca de 0,59 e o F1 em 0,66. Portanto, o modelo reconhece melhor a classe majoritária e ainda deixa de identificar uma parcela relevante dos casos positivos.

O `macro avg` atribui o mesmo peso a cada classe. O `weighted avg` considera o número de exemplos de cada classe e, por isso, é mais influenciado pela classe `<=50K`, que é majoritária.

## Fairness

A auditoria foi realizada com `MetricFrame`, separando resultados por `sex` e `race`. Foram observadas Accuracy, Recall, False Positive Rate (FPR), Selection Rate e quantidade de exemplos (`count`), além de `difference()` e `ratio()`.

Neste experimento, `sex` e `race` permanecem entre as features utilizadas pelo modelo e também são usados como atributos sensíveis na auditoria de fairness. A análise mede disparidades de desempenho entre grupos; remover diretamente esses atributos não garantiria, por si só, a eliminação de possíveis vieses.

Principais observações:

- Por `sex`, o Recall foi aproximadamente 0,603 para Male e 0,507 para Female; o FPR e a Selection Rate também diferiram.
- Por `race`, o Recall variou aproximadamente de 0,300 a 0,622 entre os grupos presentes no teste.
- A diferença de Recall por `race` foi aproximadamente 0,322.
- Equal Opportunity Difference por `race`: aproximadamente 0,322.
- Equalized Odds Difference por `race`: aproximadamente 0,322.
- Alguns grupos possuem poucas observações; por isso, suas métricas podem ser menos estáveis.

Foram observadas disparidades de desempenho entre grupos. Esses resultados não provam automaticamente discriminação e também não permitem afirmar que o modelo é justo. Seria necessário investigar dados, contexto, representatividade, incerteza das métricas e impactos antes de qualquer aplicação real.

## Limitações

- Recall ainda limitado para a classe positiva.
- Quantidade relevante de falsos negativos: 962.
- A auditoria de fairness indica disparidades entre grupos.
- Alguns grupos têm amostras pequenas e muito diferentes entre si.
- O dataset é histórico e usado aqui como exemplo educacional.
- Os resultados não justificam o uso do modelo em decisões reais sobre pessoas.
- Este projeto foi desenvolvido para estudo e portfólio.

## Próximos passos

Possibilidades futuras, não implementadas neste projeto:

- avaliar outros modelos;
- realizar tuning moderado;
- estudar ajuste de threshold;
- aprofundar a análise de fairness;
- avaliar calibração de probabilidades;
- testar feature engineering.

## Como executar

Requer Python 3.12 ou compatível.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
jupyter notebook
```

Abra `notebooks/adult_income_analysis.ipynb` e execute todas as células em ordem. Na primeira execução, o scikit-learn baixa o dataset do OpenML, portanto é necessário acesso à internet.
