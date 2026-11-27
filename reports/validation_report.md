# Relatorio de Validacao Matematica e Consistencia Metodologica

- **Total de verificacoes executadas:** `16`.
- **Aprovadas:** `16` | **Reprovadas:** `0` | **Nao aplicaveis:** `0`.
- **Status geral da auditoria matematica:** **`APROVADO`**.

## Tabela de Verificacoes (`VAL-01` a `VAL-16`)

| Codigo | Verificacao | Status | Esperado | Obtido | Detalhes |
| --- | --- | --- | --- | --- | --- |
| `VAL-01` | Conservacao de registros na divisao treino/teste | **APROVADO** | total=48842 | treino=39073, teste=9769, soma=48842 | Verifica soma exata e correspondencia de tamanhos entre X e y. |
| `VAL-02` | Ausencia de sobreposicao de indices entre treino e teste | **APROVADO** | intersecao=0 e uniao=X.index | intersecao=0, uniao=48842/48842 | Confirma particionamento disjunto e completo dos indices originais. |
| `VAL-03` | Conservacao da distribuicao das classes e estratificacao | **APROVADO** | <=50K=37155, >50K=11687 | <=50K=37155, >50K=11687 (pos_train=0.2393, pos_test=0.2393) | Garante soma exata das classes e preservacao da proporcao positiva. |
| `VAL-04` | Alinhamento exato entre X_test, y_test, y_pred e atributos sensiveis | **APROVADO** | indices identicos (9769 linhas) | alinhados | Impede troca silenciosa de posicoes entre previsoes e grupos sensiveis. |
| `VAL-05` | Soma e marginais da matriz de confusao | **APROVADO** | N=9769, pos=2338, neg=7431 | TN=6951, FP=480, FN=962, TP=1376 (soma=9769) | Valida TN+FP+FN+TP=N e correspondencia com reais e previstos. |
| `VAL-06` | Reproducao independente de Accuracy, Precision, Recall e F1 pela matriz | **APROVADO** | Acc=0.852390, Rec=0.588537, Prec=0.741379, F1=0.656175 | Acc=0.852390, Rec=0.588537, Prec=0.741379, F1=0.656175 | Confirma concordancia absoluta (tolerancia <= 1e-12) com o scikit-learn. |
| `VAL-07` | Reconciliacao do classification_report (suportes, macro avg e weighted avg) | **APROVADO** | macro_rec=0.761971, weighted_rec=0.852390 | macro_rec=0.761971, weighted_rec=0.852390 | Confirma diferenca matematica entre media igualitaria (macro) e ponderada (weighted). |
| `VAL-08` | Conservacao de TP/TN/FP/FN em todas as dimensoes da analise de erros | **APROVADO** | 7 dimensoes somando N=9769 e matriz global | todas reconciliadas | Garante que categorias ausentes e faixas cobrem 100% dos registros sem duplicacao. |
| `VAL-09` | Conservacao de registros, positivos e negativos por grupo (sex) | **APROVADO** | N=9769, pos=2338, neg=7431 | N=9769, pos=2338, neg=7431 (2 grupos) | Confirma cobertura total da amostra de teste pelos grupos de sex. |
| `VAL-10` | Conservacao de registros, positivos e negativos por grupo (race) | **APROVADO** | N=9769, pos=2338, neg=7431 | N=9769, pos=2338, neg=7431 (5 grupos) | Confirma cobertura total da amostra de teste pelos grupos de race. |
| `VAL-11` | Reconciliacao ponderada entre metricas por grupo e metricas globais | **APROVADO** | soma ponderada == metrica global (tol <= 1e-12) | todas as 8 combinacoes reconciliadas | Comprova que Recall exige ponderacao por positivos reais e FPR por negativos reais (nao media simples). |
| `VAL-12` | Reproducao independente de Equal Opportunity e Equalized Odds vs Fairlearn | **APROVADO** | max-min manual == Fairlearn (tol <= 1e-12) e escopo integral/parcial/indefinido consistente | sex: EOD=0.096514, EOdds=0.096514 (integral_todos_os_grupos); race: EOD=0.321951, EOdds=0.321951 (integral_todos_os_grupos) | Valida EOD = max(TPR_g)-min(TPR_g), EOdds = max(diff_TPR, diff_FPR) apenas com ambos os eixos validos, e sinalizacao explicita de comparacao parcial. |
| `VAL-13` | Integridade do experimento Cenario A vs Cenario B (atributos sensiveis) | **APROVADO** | Cenario A=14 features, Cenario B=12 sem ['sex', 'race'] | Cenario A=14, Cenario B=12, alinhamento=True | Confirma remocao efetiva de sex/race no Cenario B sem desalinhar indices da auditoria. |
| `VAL-14` | Ajuste exclusivo do pre-processamento no treino (anti-leakage em 'age') | **APROVADO** | media_treino=38.697182 | scaler.mean_=38.697182 (media_total_base=38.643585) | Confirma que StandardScaler aprendeu estatisticas apenas de X_train, nunca de X completo. |
| `VAL-15` | Tratamento transparente de denominadores iguais a zero e escopo parcial | **APROVADO** | 0 metricas indefinidas mascaradas como zero e nenhuma avaliacao parcial rotulada como integral | grupos_com_denominador_zero=0, mascarados=0, parciais_nao_sinalizados=0 | Garante que divisoes 0/0 retornam NaN/indefinido e que Equalized Odds parcial/incompleto nunca e apresentado como integral. |
| `VAL-16` | Reproducao fiel da referencia historica V1 (OpenML 1590 v2) | **APROVADO** | Acc=0.85239, Rec=0.58854, Prec=0.74138, F1=0.65618, CM=6951/480/962/1376, EOD_race=0.321951 | Acc=0.85239, Rec=0.58854, Prec=0.74138, F1=0.65618, CM=6951/480/962/1376, EOD_race=0.321951 | Compara a execucao atual com os resultados historicos reproduzidos da V1. |
