# Responsible AI, Limitações Científicas e Éticas

## 1. Posicionamento do Projeto

Este repositório foi construído exclusivamente como um **laboratório educacional de avaliação metodológica de modelos de Machine Learning e auditoria de Responsible AI**. Ele **não** é um produto pronto para produção e **não** apresenta a classificação de faixa de renda como ferramenta de decisão automatizada.

---

## 2. Limitações Estruturais da Base Histórica (Censo dos EUA de 1994)

1. **Defasagem histórica de mais de três décadas:** os dados provêm do *Current Population Survey (CPS)* de 1994. O limiar de US\$ 50.000/ano, a estrutura ocupacional, o perfil educacional e a composição demográfica dos EUA em 1994 não representam a realidade econômica e social contemporânea.
2. **A variável alvo é apenas uma faixa de renda observada:** `<=50K` vs. `>50K` reflete renda bruta reportada naquele recorte, e **não** mede competência profissional, merecimento, idoneidade ou capacidade de pagamento.
3. **Representatividade e `fnlwgt`:** os filtros originais de extração (`age > 16`, `AGI > 100`, `fnlwgt > 1`, `hours-per-week > 0`) e o desenho amostral ponderado por estado (`fnlwgt`) fazem com que as contagens brutas da tabela não correspondam automaticamente à população geral dos EUA (nem mesmo de 1994).
4. **Desigualdades históricas embutidas nos dados:** as taxas-base observadas na amostra (`30,38%` de `>50K` em `Male` vs. `10,93%` em `Female`; `25,40%` em `White` e `26,93%` em `Asian-Pac-Islander` vs. `12,08%` em `Black`, `11,70%` em `Amer-Indian-Eskimo` e `12,32%` em `Other`) carregam desigualdades históricas estruturais do mercado de trabalho da década de 1990. Qualquer modelo treinado nesses dados tende a reproduzir ou amplificar essas assimetrias.
5. **Classificações históricas preservadas com transparência:** rótulos como `Amer-Indian-Eskimo` e o binário `Male`/`Female` são categorias históricas da fonte de 1994 preservadas intactas para fidelidade científica e auditoria — não representam convenções modernas de identidade ou classificação demográfica.

---

## 3. O Que a Auditoria de Fairness Pode — e NÃO Pode — Concluir

### O que a auditoria demonstra objetivamente
- Quantifica **disparidades estatísticas observadas** de Accuracy, Recall (TPR), FPR e Selection Rate entre grupos sob definições matemáticas explícitas (`EOD`, `EOdds`, `difference`, `ratio`).
- Evidencia empiricamente a instabilidade em grupos com baixo suporte (`Other` com `10` positivos reais e `Amer-Indian-Eskimo` com `13` no teste).
- Mostra empiricamente que a simples exclusão de `sex` e `race` (Cenário B) reduz parte de `EOD`, mas não elimina as disparidades observadas — evidenciando a insuficiência prática de *fairness through unawareness* neste dataset, sem isolar causalmente qual parcela decorre de variáveis correlacionadas (candidatas a *proxies*) ou de diferenças de distribuição e taxas-base entre grupos.

### Conclusões que NÃO são permitidas
- **Não certifica ausência de discriminação:** valores baixos de EOD ou EOdds não provam que um sistema é "justo".
- **Não prova discriminação intencional nem causalidade:** diferenças entre grupos observacionais não estabelecem mecanismo causal sem análise de contexto, covariáveis, coleta e impacto social.
- **Nenhuma métrica isolada captura todas as dimensões de fairness:** critérios como *Demographic Parity* (Selection Rate), *Equal Opportunity* (TPR) e *Equalized Odds* (TPR + FPR) frequentemente entram em conflito matemático quando as taxas-base diferem entre grupos (teoremas de impossibilidade de Chouldechova / Kleinberg).
- **Ausência proposital de mitigação automática:** nenhum algoritmo de pós-processamento ou mitigação de viés foi aplicado, pois mitigar artificialmente métricas em dados históricos sem contexto decisório real pode gerar falsa sensação de segurança.

---

## 4. Usos Explicitamente Proibidos

O modelo, os pesos e as regras analisados neste projeto **NÃO devem ser utilizados** para:
- Recrutamento, triagem de currículos, contratação ou promoção profissional.
- Concessão de crédito, precificação de empréstimos, seguros ou análise de risco financeiro individual.
- Alocação de benefícios públicos, habitação, educação ou qualquer decisão com impacto jurídico, econômico ou pessoal sobre seres humanos reais.
