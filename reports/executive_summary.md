# Sumario Executivo — Adult Income: Model Evaluation & Fairness V2

> **Natureza do trabalho:** Laboratorio educacional reproduzivel de avaliacao metodologica de modelos, diagnostico de erros e auditoria de Responsible AI sobre dados historicos do Censo dos EUA de 1994. **Nao e uma ferramenta de decisao automatizada para uso em producao sobre pessoas reais.**

## 1. Visao Geral da Execucao

| Dimensao | Evidencia Verificada |
| --- | --- |
| **Dataset** | `data/adult_openml_1590_v2.csv` (`48.842` linhas x `14` features) |
| **Integridade SHA-256** | `8f970e20c9991979ec0a9ccb65ce97543d678709a2c6f8ef23f7fb32f40adddc` |
| **Divisao estratificada** | `39.073` treino (80%) / `9.769` teste (20%) — `0` sobreposicao de indices |
| **Modelo selecionado (CV 5-fold no treino)** | `Logistic Regression (max_iter=1000)` |
| **Desempenho no teste** | Accuracy `85,24%`, Recall `58,85%`, Precision `74,14%`, F1 `65,62%` |
| **Matriz de confusao** | `TN=6.951`, `FP=480`, `FN=962`, `TP=1.376` |
| **Fairness (`sex`, teste)** | EOD = `0,096514` (`integral_todos_os_grupos`), EOdds = `0,096514` (`integral_todos_os_grupos`) |
| **Fairness (`race`, teste)** | EOD = `0,321951` (`integral_todos_os_grupos`), EOdds = `0,321951` (`integral_todos_os_grupos`) |
| **Experimento Cenario B − A (CV treino)** | Delta F1 = `0,0000`, Delta EOD `sex` = `-0,0227`, Delta EOD `race` = `-0,1228` |
| **Validacao matematica** | `16/16` aprovadas (`0` reprovadas, `0` N/A) |

## 2. Principais Achados

1. **Generalizacao vs. overfitting:** a `Decision Tree (sem limite)` atingiu Accuracy proxima de `100%` no treino com queda expressiva na validacao, sendo objetivamente descartada pelo criterio pre-registrado. `Logistic Regression (max_iter=1000)` entregou o melhor F1 e Recall na classe `>50K` com gap minimo entre treino e validacao.
2. **Limitacao principal de erros:** dos `1.442` erros no teste, `962` sao falsos negativos (`>50K` previstos como `<=50K`) e `480` sao falsos positivos.
3. **Disparidades observadas e tamanho amostral:** ha disparidades relevantes em Recall, FPR e Selection Rate tanto por `sex` quanto por `race`. Em `race`, grupos com poucas observacoes e poucos positivos reais (`Amer-Indian-Eskimo, Other`) aumentam a variabilidade amostral das metricas.
4. **Remocao de `sex` e `race` (Cenario B):** retirar diretamente os atributos sensiveis altera pouco o desempenho global e reduz parte de EOD, mas nao elimina as disparidades observadas. Embora variaveis correlacionadas remanescentes sejam candidatas plausiveis a *proxies* (junto a diferencas de distribuicao e taxas-base entre grupos), a remocao isolada de `sex` e `race` nao comprova causalmente a contribuicao individual dessas variaveis.

## 3. Relatorios Gerados

- `data_quality_report.md` — auditoria de tipos, ausentes, duplicatas, `fnlwgt` e categorias raras.
- `model_comparison.csv` e `cross_validation_results.csv` — comparacao dos 4 modelos e resultados por fold.
- `test_metrics.json` e `confusion_matrix.csv` — metricas finais e matriz de confusao.
- `error_analysis.csv` e `model_evaluation_report.md` — diagnostico de erros por segmento e avaliacao completa.
- `fairness_by_sex.csv`, `fairness_by_race.csv`, `fairness_summary.csv`, `sensitive_features_comparison.csv` e `fairness_audit_report.md` — auditoria de fairness e experimento controlado.
- `validation_report.md` — 16 checagens matematicas independentes.
