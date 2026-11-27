# Dataset — Adult / Census Income (OpenML ID 1590, v2)

## 1. Proveniencia e Licenca Verificadas

| Atributo | Especificacao verificada |
| --- | --- |
| **Nome** | Adult / Census Income |
| **Autoria original** | Barry Becker e Ronny Kohavi (Silicon Graphics, 1996) |
| **Base historica de origem** | US Census Bureau — Current Population Survey (CPS) de 1994 |
| **Fonte UCI** | [UCI Machine Learning Repository — Adult (ID 2)](https://archive.ics.uci.edu/dataset/2/adult) |
| **DOI oficial** | [https://doi.org/10.24432/C5XW20](https://doi.org/10.24432/C5XW20) |
| **Licenca UCI** | [CC BY 4.0 (Creative Commons Attribution 4.0 International)](https://creativecommons.org/licenses/by/4.0/) |
| **Fonte OpenML (usada na V1)** | [OpenML Dataset ID 1590 (`adult`, versao 2)](https://www.openml.org/d/1590) |
| **Campo `licence` no OpenML** | `Public` |
| **Arquivo ARFF OpenML (`file_id=1595261`)** | `5.962.724` bytes · `MD5 = bb6510925e5d4b23d136715febb2cdf5` · `SHA-256 = 77aa1703717a29f0b5642e94c3ba1defd2486f0b34d4d8eccc1b37a5f7d226b0` |

> **Atribuicao obrigatoria (CC BY 4.0):** Becker, B., & Kohavi, R. (1996). *Adult* [Dataset]. UCI Machine Learning Repository. DOI: `https://doi.org/10.24432/C5XW20`. Distribuido tambem via OpenML ID 1590 (v2). A licenca CC BY 4.0 permite copia e redistribuicao com atribuicao adequada, possibilitando incluir a copia local canonica verificavel para reproducao offline.

---

## 2. Representacao Canonica Local (`adult_openml_1590_v2.csv`)

Para garantir execucao offline reproduzivel e verificacao criptografica sem perder paridade com a V1 (`fetch_openml`), o arquivo `data/adult_openml_1590_v2.csv` foi exportado diretamente de `fetch_openml(name="adult", version=2, as_frame=True)` sob a seguinte especificacao canonica:

- **Codificacao:** `UTF-8` (sem BOM).
- **Quebra de linha:** `LF` (`\n`), protegida contra conversao automatica no Git por `.gitattributes` (`data/adult_openml_1590_v2.csv -text`).
- **Delimitador:** virgula (`,`), sem coluna de indice.
- **Valores ausentes:** representados como campo vazio (`""`), lidos explicitamente com `pd.read_csv(..., na_values=[""], keep_default_na=False)`.
- **Dimensoes:** `48.842` linhas x `15` colunas (`14` features + target `class`).
- **Tamanho exato:** `5.271.056` bytes.
- **SHA-256 verificado:** `8f970e20c9991979ec0a9ccb65ce97543d678709a2c6f8ef23f7fb32f40adddc`.

---

## 3. Diferencas Entre os Arquivos Brutos da UCI e a Versao OpenML 1590 v2

1. **Unificacao de treino e teste originais:** na UCI, a base vem separada em `adult.data` (`32.561` linhas) e `adult.test` (`16.281` linhas). A versao 2 do OpenML une os dois arquivos em `48.842` registros (com os registros do antigo `adult.test` aparecendo primeiro no ARFF).
2. **Normalizacao do rotulo alvo:** no arquivo bruto `adult.test` da UCI, as classes possuem um ponto final (`<=50K.` e `>50K.`). No OpenML 1590 v2, todas as classes estao padronizadas como `<=50K` (`37.155`) e `>50K` (`11.687`).
3. **Valores ausentes:** marcadores `?` do ARFF sao interpretados pelo `fetch_openml` como `NaN` (`workclass`: `2.799`, `occupation`: `2.809`, `native-country`: `857`) e preservados como campos vazios no CSV canonico.

---

## 4. Como Verificar ou Regenerar a Copia Local

```powershell
# Executar o pipeline usando a copia local canonica verificada por SHA-256 (padrao)
python main.py

# Executar carregando diretamente do OpenML (preservando o metodo historico da V1)
python main.py --source openml

# Regenerar a copia local canonica a partir do OpenML ID 1590 v2
python main.py --export-canonical
```
