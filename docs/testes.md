# ✅ Testes e solução de problemas

[⬅️ README](../README.md) · [Dados](dados-sinteticos.md) · [German Credit](german-credit.md) · [Modelo](modelo.md) · [Métricas](metricas.md) · [Validação cruzada](validacao-cruzada.md) · [Mitigação](mitigacao.md) · [Inferência](inferencia.md) · [API](api.md) · **Testes**

---

## Cobertura da suíte

São **295 testes** no runner nativo do Node (`npm test`), sem dependência de desenvolvimento, escritos no formato **Given / When / Then**:

```javascript
it('dada a probabilidade exatamente no limiar, quando classificada, então retorna ALTO RISCO', () => {
  // Given — o corte usa >=, então o limiar pertence à classe positiva
  const probability = DECISION_THRESHOLD;

  // When
  const label = classify(probability);

  // Then
  assert.equal(label, 'ALTO RISCO');
});
```

A suíte cobre o que é determinístico e verificável sem treinar a rede — mais alguns casos que treinam redes minúsculas em memória, para verificar o que só existe depois do treino:

| Alvo                   | O que é verificado                                                          |
| ---------------------- | --------------------------------------------------------------------------- |
| `normalizeIncome`      | Piso, teto e meio da faixa mapeiam para `0`, `1` e `0.5`                      |
| `normalizeLatePayments`| `0` atrasos → `0`; máximo de atrasos → `1`                                    |
| `toFeatureVector`      | Vetor esperado, largura 4 e determinismo (proteção contra *skew*)             |
| `classify`             | Comportamento no limiar `0.5`, inclusive logo abaixo dele                     |
| `createDataset`        | Tamanho, features dentro de `[0, 1]`, rótulos binários, ambas as classes presentes |
| `createCustomers` / `toDataset` | Unidades brutas dentro das faixas, rótulos binários, equivalência com `createDataset` e **mesma semente → mesmos clientes** |
| `createGaussian`       | Média `≈ 0` e desvio `≈ 1` em 20 mil sorteios, simetria em torno de zero, reprodutibilidade e `log(0)` que não vira infinito |
| `clamp` / `quantile`   | Limites, ordenação interna, **entrada não modificada** e o corte no percentil 85 produzindo 15% acima |
| `riskScore`            | Monotonicidade em dívida e renda (o único coeficiente negativo) e determinismo |
| `measureCustomer`      | Ruído zero → identidade, nada fora dos limites válidos nem com ruído enorme, atrasos ainda inteiros e nenhuma coluna extra vazando |
| Desbalanceamento       | A taxa pedida é entregue em `0.05`, `0.25` e `0.5`, e o baseline do dataset padrão passa de `0.8` |
| Ruído de rótulo        | Sem ruído a regra reproduz o rótulo exatamente, `0.05` troca ~5% e `1.0` inverte **todos** |
| Teto imposto pelo ruído | Com ruído, a **própria regra geradora** erra ao ver só a medida (`< 100%`); sem ruído acerta tudo |
| `data/customers.csv`   | O arquivo versionado é **idêntico** ao que o gerador produz, e a minoria fica entre 10% e 20% |
| `toCsv`                | Cabeçalho, contagem de linhas e precisão por coluna (inteiros saem inteiros)   |
| `writeCustomersCsv` / `readCustomersCsv` | Criação da pasta, ida e volta dos valores, parse numérico, colunas fora de ordem e arquivo inexistente |
| `ensureCsv`            | Cria quando falta, **não altera** quando já existe, e sobrescreve só na chamada explícita |
| `loadDatasetCsv`       | Formato igual ao de `createDataset`, features em `[0, 1]`, mesma `toFeatureVector` e treino rodando a partir do arquivo |
| `splitDataset`         | Proporções, nada perdido e **ausência de sobreposição entre treino e teste**  |
| `parseDelimited`       | Cabeçalho vira chave, CRLF do Windows e quebra de linha final sem gerar linha vazia |
| `toOrdinal`            | Código conhecido vira posição, `A41` não é confundido com `A410` e código desconhecido **lança** em vez de virar `0` |
| `toGermanCustomer`     | Numéricos → `Number`, códigos → índices, `class 2` → `risk 1`, cobertura de todas as colunas e o atributo 9 indo para a coluna de auditoria |
| `oneHotEncode` / `ordinalEncode` | Posição acesa, soma sempre `1`, largura correta e categoria única sem divisão por zero |
| `germanFeatureNames` / `toGermanVector` | **57** entradas em one-hot e **19** em ordinal, nomes `campo=código`, vetor do mesmo tamanho dos nomes, numéricas primeiro e bloco categórico só com `0` e `1` |
| Atributo protegido | **Nenhuma feature vem do sexo**, a coluna existe no CSV mas não no modelo, e o scaler não a mede |
| `isFemale` / `summarizeGroup` / `auditByGroup` | Identificação do grupo, taxa real separada da marcada, FNR sem `NaN`, paridade quando o tratamento é igual e razão infinita quando um grupo não tem aprovação |
| `rateThreshold` | Fração pedida vira fração marcada, corte no ponto médio, listas vazias e frações `0`/`1` nos extremos, e **empate na fronteira que transforma a fração em piso** |
| `fitGroupThresholds` | O grupo marcado demais recebe o limiar mais alto, grupo vazio vira limiar inalcançável, **os scores não são tocados** e calibrar no conjunto auditado devolve paridade por construção |
| `thresholdFor` / auditoria por grupo | Número vale para os dois, par vale um para cada, a razão sobe com limiares por grupo e os dois limiares ficam registrados na auditoria |
| `summarizeDecisions` | Acurácia e custo a partir dos erros, o falso negativo pesando 5×, corte por grupo acertando os quatro casos e custos próprios substituindo os do laboratório |
| `formatMitigation` | Limiares, razão e custo das duas políticas na mesma tabela |
| `resolveMitigation` | Ausência é desligado, `--mitigar` liga e **`--mitigar=false` lança em vez de ligar a política que o usuário quis desligar** |
| `createGermanSource` | As duas variantes diferem só na codificação e a ordinal é alcançável por `--source` |
| `parseGermanCsv`       | Texto bruto da UCI → clientes prontos, ponta a ponta |
| `data/german-credit.csv` | O arquivo versionado tem as 1.000 linhas, **300 maus pagadores**, as 21 colunas declaradas e nenhum valor não-finito |
| `fitMinMaxScaler` / `applyMinMaxScaler` | Mínimo e amplitude, coluna constante sem divisão por zero, valor fora da faixa **não** cortado, ordem do vetor e **ausência de vazamento do teste para a escala** |
| `createRandom` / `shuffle` | Mesma semente → mesma sequência, faixa `[0, 1)`, permutação, entrada não modificada e embaralhamento reproduzível |
| `splitCustomers`       | Proporções e ausência de sobreposição, agora sobre clientes brutos |
| `stratifiedSplitCustomers` | Proporção de classes **idêntica** nos dois lados, tamanhos, ausência de sobreposição, **ordem preservada dentro de cada parte** e um caso em que o corte cru entrega um teste 100% inadimplente |
| `stratifiedFolds` | Mesma proporção em cada dobra, cada cliente em exatamente uma, e diferença de no máximo uma linha quando o k não divide a classe |
| `summarize` | Média, erro padrão e amplitude; uma amostra só dá erro `0` e não `NaN`; valores idênticos dão erro exatamente `0` |
| `resolveFolds` | Ausência é `null`, `--cv` usa o padrão, `--cv=k` usa o pedido, e `k = 1`, `k = 2.5`, `k = 999` ou `abc` **lançam** |
| `rocFromScores` | Curva sem modelo nenhum, AUC `1` e `0` nos extremos, classe única sem divisão por zero e `computeRocCurve` delegando para ela |
| `crossValidate` | Cada cliente testado exatamente uma vez, **baseline idêntico em todas as dobras**, resumo com média e erro padrão, e fonte sem atributo protegido não produzindo auditoria |
| `TRAINING` | A configuração de treino é **uma só** para o fluxo principal e a validação cruzada |
| `majorityBaseline`     | Piso da classe majoritária com maioria negativa, positiva e empate |
| `SOURCES` / `resolveSourceId` | Padrão, seleção por flag, fonte inválida, **contrato cumprido por todas as fontes**, tamanho do vetor, escala medida e mensagem acionável quando o CSV real falta |
| Regularização por fonte | Sintético com os freios **desligados**, German com as constantes do laboratório, as duas variantes diferindo **só** na codificação e a linha de comando vencendo a fonte na mesclagem |
| `toCsv` com schema     | Cabeçalho do German Credit e compatibilidade da chamada sem opções |
| `buildModel`           | 3 densas + 2 de dropout, entrada `[null, 4]` ou `[null, 57]`, saída `[null, 1]`, **225** e **1.073 parâmetros**, ativações e loss |
| `createRegularizer`    | Lambda chega intacto na camada e `λ = 0` devolve **`null`**, não uma penalidade inerte |
| `buildModel` — regularização | L2 em **todas** as densas, dropout **só** depois das ocultas, `dropout: 0` restaura a topologia de 3 camadas e o total de parâmetros **não muda** |
| `parseNumericFlag` / `resolveRegularization` | Padrões, leitura, limites inclusivos, valor negativo, dropout acima de `0.9` e **`--l2=` vazio que não vira zero em silêncio** |
| Comportamento de L2 e dropout | Dropout **desligado na inferência** (duas predições idênticas com taxa `0.9`), `evaluate` **não** cobra a penalidade e o modelo regularizado sobrevive ao salvar/recarregar |
| `computeConfusionMatrix` | TP/TN/FP/FN contra predições conhecidas, layout da matriz, efeito do limiar, coerência com a `accuracy` do `evaluate` e ausência de vazamento de tensores |
| `formatConfusionMatrix` | Estrutura da tabela, as quatro contagens presentes e colunas alinhadas |
| `computeMetrics` | Fórmulas contra cálculo manual, F1 conferido pelas duas formas, casos degenerados sem `NaN` e a média harmônica abaixo da aritmética |
| `formatMetrics` | Uma linha por métrica, valores com 4 casas decimais |
| `computeRocCurve` | AUC de classificadores perfeito/invertido/aleatório, coerência com Mann-Whitney, invariância a reescala monotônica, monotonicidade dos pontos, empates agrupados e classe única sem divisão por zero |
| `formatRocCurve` | Estrutura do gráfico, largura da área, curva, diagonal e marcador do limiar |
| `chooseThresholdByYouden` | Maximização de `TPR - FPR` e corte sem erros no classificador perfeito |
| `chooseThresholdByCost` | Custo mínimo global, corte que desce com FN caro, divergência em relação a Youden, recusa total quando FP é proibitivo e confirmação na matriz recomputada |
| `formatTable` / `formatThresholdComparison` | Alinhamento com larguras variadas, estrutura da tabela e limiar infinito legível |
| `saveModel` / `loadModel` | Arquivos gerados, `trainingConfig` gravado, arquitetura preservada, **predição idêntica** após recarregar, modelo já compilado e treino retomável |

---

## 🩺 Solução de problemas

**`TypeError: (0 , util_1.isNullOrUndefined) is not a function`**

Você está em um Node novo demais. O **Node 23 removeu** os type-checkers legados do módulo `util` (depreciados desde o Node 4), e o `@tensorflow/tfjs-node@4.22.0` ainda os utiliza em 5 pontos do backend nativo.

| Node | `util.isNullOrUndefined` | Projeto |
| ---- | ------------------------ | ------- |
| 20 (LTS)  | presente  | ✅ funciona |
| 22 (LTS)  | presente  | ✅ funciona |
| 23        | removido  | ❌ quebra   |
| 24 (LTS)  | removido  | ❌ quebra   |

O `tfjs-node` não recebe release desde **janeiro de 2025**, e o `4.23.0-rc.0` mantém as mesmas chamadas — não há correção upstream. Por isso o projeto fixa **Node 22** no `.nvmrc` e declara `"engines": { "node": ">=20.0.0 <23.0.0" }`.

Solução:

```bash
nvm install 22 && nvm use
rm -rf node_modules package-lock.json && npm install
```

---

[⬅️ Voltar ao README](../README.md)
