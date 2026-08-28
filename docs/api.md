# 🧩 API do módulo e exemplo de saída

[⬅️ README](../README.md) · [Dados](dados-sinteticos.md) · [German Credit](german-credit.md) · [Modelo](modelo.md) · [Métricas](metricas.md) · [Validação cruzada](validacao-cruzada.md) · [Mitigação](mitigacao.md) · [Inferência](inferencia.md) · [Serviço](servico.md) · **API** · [Testes](testes.md)

---

## 🧩 API do módulo

O `index.js` só executa o treino quando chamado direto (`node index.js`); ao ser importado, apenas expõe suas funções:

```javascript
if (require.main === module) {
  const argv = process.argv.slice(2);
  const run = async () => main(resolveSourceId(argv), {
    ...resolveRegularization(argv),
    mitigate: resolveMitigation(argv),
  });   // simplificado: o arquivo também trata --cv, --units e --arquiteturas

  run().catch((error) => {
    console.error(`\n${error.message}\n`);
    process.exitCode = 1;
  });
}
```

Envolver em uma função `async` faz o erro **síncrono** de `resolveSourceId` virar rejeição e cair no mesmo `.catch` dos erros assíncronos — sem isso, `--source=xpto` imprimiria um *stack trace* no lugar da lista de fontes válidas.

É isso que permite testar as partes sem treinar a rede. O que ele exporta:

| Export | Tipo | Papel |
| ------ | ---- | ----- |
| `INCOME_MIN`, `INCOME_RANGE`, `MAX_LATE_PAYMENTS` | constantes | Faixas usadas na normalização |
| `DECISION_THRESHOLD` | constante | Corte `0.5` do classificador |
| `SYNTHETIC_SEED`, `SYNTHETIC_TOTAL` | constantes | Semente e tamanho do dataset sintético |
| `SYNTHETIC_POSITIVE_RATE` | constante | Fração alvo de inadimplentes — `0.15` |
| `SYNTHETIC_FEATURE_NOISE`, `SYNTHETIC_LABEL_NOISE` | constantes | Ruído de medição (`0.05`) e de rótulo (`0.02`) |
| `SYNTHETIC_BOUNDS` | constante | Faixa válida de cada coluna, usada para escalar e limitar o ruído |
| `normalizeIncome`, `normalizeLatePayments` | função | Normalizações min-max individuais |
| `toFeatureVector` | função | Cliente bruto → vetor de 4 features (sintético) |
| `classify` | função | Probabilidade → `'ALTO RISCO'` / `'BAIXO RISCO'` |
| `MODEL_DIR` | constante | Pasta `./model` onde o modelo é persistido |
| `CSV_PATH`, `CSV_COLUMNS`, `CSV_LABEL_COLUMN`, `CSV_PRECISION` | constantes | Caminho, esquema e precisão do arquivo |
| `createGaussian` | função | Sorteio uniforme → normal padrão (Box-Muller) |
| `clamp` | função | Prende um valor entre mínimo e máximo |
| `quantile` | função | Valor abaixo do qual está uma fração dos dados |
| `riskScore` | função | Regra que gera os rótulos, aplicada ao cliente **verdadeiro** |
| `measureCustomer` | função | Cliente verdadeiro → cliente **medido**, com ruído |
| `createCustomers` | função | Gera clientes em unidades brutas, com `risk` |
| `toDataset` | função | Clientes brutos → `{ features, labels }` |
| `createDataset` | função | Gera `{ features, labels }` sintéticos |
| `toCsv` | função | Clientes → texto CSV |
| `writeCustomersCsv` | função | Grava o CSV, criando a pasta se preciso |
| `ensureCsv` | função | Cria o CSV só se ele não existir → `{ path, created }` |
| `readCustomersCsv` | função async | Lê o CSV via `tf.data.csv` → clientes brutos |
| `loadDatasetCsv` | função async | CSV → `{ features, labels }` normalizados |
| `splitDataset` | função | Divide features já normalizadas em treino e teste |
| `GERMAN_CSV_PATH`, `GERMAN_SOURCE_URL` | constantes | Arquivo local e endereço na UCI do dataset real |
| `GERMAN_NUMERIC`, `GERMAN_CATEGORICAL` | constantes | As 7 colunas com magnitude e as 12 qualitativas com seus códigos |
| `GERMAN_AUDIT_COLUMN`, `GERMAN_AUDIT_CODES`, `FEMALE_CODE` | constantes | Atributo 9: fora do modelo, dentro da auditoria |
| `GERMAN_SOURCE_ATTRIBUTES` | constante | De que `AttributeN` do arquivo vem cada coluna |
| `GERMAN_COLUMNS`, `GERMAN_PRECISION` | constantes | As 21 colunas do CSV e sua precisão |
| `oneHotEncode`, `ordinalEncode` | funções | Índice de categoria → vetor de features |
| `germanFeatureNames` | função | Nome de cada posição do vetor, por codificação |
| `toGermanVector` | função | Cliente → numéricas escaladas + qualitativas codificadas |
| `isFemale`, `toAuditRows`, `summarizeGroup`, `approvalRatio`, `auditByGroup` | funções | Auditoria de disparidade por grupo |
| `formatAudit` | função | Auditoria → tabela para o terminal |
| `thresholdFor` | função | Um número vale para os dois grupos; um par `{ women, men }` vale um para cada |
| `rateThreshold` | função | Scores + fração → limiar que marca essa fração, no ponto médio da fronteira |
| `fitGroupThresholds` | função | Calibra no **treino** um limiar por grupo que iguala as taxas de aprovação |
| `summarizeDecisions` | função | Política de limiar → falsos positivos, falsos negativos, acurácia e custo |
| `formatMitigation` | função | Políticas → tabela comparando razão de aprovação e preço |
| `createGermanSource` | função | Fábrica das duas variantes do dataset real |
| `parseDelimited` | função | Texto CSV → lista de objetos (parser mínimo) |
| `toOrdinal` | função | Código `'A11'` → posição na lista documentada; lança se desconhecido |
| `toGermanCustomer` | função | Linha da UCI → cliente no vocabulário do projeto |
| `parseGermanCsv` | função | Texto bruto da UCI → clientes prontos |
| `fitMinMaxScaler` | função | Clientes de treino → `{ featureNames, min, range }` |
| `applyMinMaxScaler` | função | Scaler + cliente → vetor normalizado |
| `SYNTHETIC_SOURCE`, `GERMAN_SOURCE`, `GERMAN_ORDINAL_SOURCE`, `SOURCES` | objetos | As três fontes de dados e o registro delas |
| `DEFAULT_SOURCE_ID`, `resolveSourceId` | constante / função | Fonte padrão e leitura de `--source=` |
| `L2_LAMBDA`, `DROPOUT_RATE` | constantes | Intensidade padrão dos dois freios contra overfitting |
| `parseNumericFlag`, `resolveRegularization` | funções | Leitura validada de `--l2=` e `--dropout=`; devolve só o que foi pedido, para a fonte manter o resto |
| `resolveMitigation` | função | Leitura de `--mitigar`: interruptor sem valor, e recusa `--mitigar=false` em vez de adivinhar |
| `resolveUnits` | função | Leitura de `--units=64,32`; `--units=0` pede a regressão logística e vazio é erro |
| `resolveArchitectureRun` | função | Leitura de `--arquiteturas` e `--repeticoes=k` |
| `SHUFFLE_SEED`, `createRandom`, `shuffle` | constante / funções | Semente e embaralhamento reproduzível (*mulberry32*) |
| `splitCustomers` | função | Divide clientes **brutos** em treino e teste, por fatiamento simples |
| `stratifiedSplitCustomers` | função | O mesmo, preservando a proporção de classes — é o que `main` usa |
| `stratifiedFolds` | função | Atribui uma dobra a cada cliente, mantendo a proporção de classes |
| `majorityBaseline` | função | Piso da acurácia: sempre chutar a classe majoritária |
| `compileModel` | função | Aplica optimizer, loss e métricas a um modelo |
| `createRegularizer` | função | Devolve a penalidade L2 da camada, ou `null` quando `λ = 0` |
| `HIDDEN_UNITS` | constante | A topologia padrão das camadas ocultas (`[16, 8]`) |
| `buildModel` | função | Monta e compila a rede: entradas da fonte, camadas ocultas e os dois freios. Com `units: []` devolve uma **regressão logística** |
| `saveModel` | função async | Salva o modelo em `file://<dir>` com o otimizador |
| `loadModel` | função async | Carrega de `model.json` e garante que vem compilado |
| `saveArtifacts` | função async | Salva o **pacote**: pesos + `metadata.json` com scaler, ordem das features e limiar. É o que `main` usa |
| `readMetadata` | função | Lê o `metadata.json` do pacote; instrui a rodar `npm start` se faltar |
| `assertServable` | função | Recusa gravar um pacote sem contrato ou com limiar fora de `[0, 1]` — inclusive o `Infinity` da ponta da curva |
| `assertConsistent` | função | Recusa carregar um pacote cuja versão, número de entradas ou **ordem de features** não bata com o código |
| `loadArtifacts` | função async | Pacote → `{ model, metadata, source, toVector, predict, dispose }`, com a normalização já embutida |
| `ARTIFACTS_VERSION`, `METADATA_FILE`, `metadataPath`, `PROBABILITY_DECIMALS`, `round` | constantes / funções | Formato do pacote e o arredondamento comum ao limiar gravado e à resposta servida |
| `predictRisk` | função | Cliente bruto → probabilidade, já liberando os tensores |
| `computeConfusionMatrix` | função | `{ truePositives, trueNegatives, falsePositives, falseNegatives, matrix }` |
| `formatConfusionMatrix` | função | Matriz → tabela alinhada para o terminal |
| `computeMetrics` | função | Matriz → `{ precision, recall, f1Score }` |
| `formatMetrics` | função | Métricas → três linhas comentadas para o terminal |
| `rocFromScores` | função | Scores + rótulos → curva, sem precisar de modelo (é o que a validação cruzada usa) |
| `computeRocCurve` | função | `{ points: [{ fpr, tpr, threshold }], auc }` varrendo todos os limiares |
| `formatRocCurve` | função | Pontos → gráfico ASCII, com marcador opcional do limiar |
| `FALSE_POSITIVE_COST`, `FALSE_NEGATIVE_COST` | constantes | Custos relativos dos dois erros (`1` e `5`) |
| `scorePoint` | função | Ponto da curva → contagens absolutas e custo |
| `chooseThresholdByYouden` | função | Corte que maximiza `TPR - FPR` |
| `chooseThresholdByCost` | função | Corte de menor custo esperado |
| `formatTable` | função | Cabeçalho + linhas → tabela alinhada |
| `formatThresholdComparison` | função | Candidatos → tabela comparativa de limiares |
| `evaluateModel` | função | Roda `evaluate` e devolve `{ loss, accuracy }` |
| `TRAINING`, `fitModel` | constante / função | A configuração de treino em um lugar só, usada pelos dois caminhos |
| `CV_FOLDS`, `resolveFolds` | constante / função | Padrão de dobras e leitura de `--cv` / `--cv=k` |
| `summarize` | função | Valores → `{ mean, standardError, lowest, highest }` |
| `crossValidate` | função async | k dobras estratificadas → métricas por dobra, resumo, scores fora da amostra e auditoria dos 1.000 |
| `formatCrossValidation` | função | Resultado → tabela com uma linha por dobra, média e erro padrão |
| `reportCrossValidation` | função async | Caminho de execução de `--cv`: roda e imprime |
| `ARCHITECTURES` | constante | As oito topologias comparadas, começando na regressão logística |
| `compareArchitectures` | função async | Roda a validação cruzada de cada arquitetura e resume as métricas de todas as dobras |
| `formatArchitectureComparison` | função | Comparação → tabela com parâmetros, épocas e métricas com erro padrão |
| `reportArchitectures` | função async | Caminho de execução de `--arquiteturas`: roda e imprime |
| `main` | função async | Pipeline completo para uma fonte: treina, avalia, salva, recarrega e prevê |
| `validateCustomer` | função | Payload + schema → `{ errors, customer }`; devolve **todos** os erros de uma vez |
| `validateCategorical`, `isNumber` | funções | As duas checagens elementares: índice de código válido, e número que não é `NaN` nem `Infinity` |
| `describeSchema` | função | Schema interno → contrato publicável, com faixa e códigos por campo (é o `GET /schema`) |
| `scoreCustomer` | função | `{ predict, threshold, schema, metadata }` → função que transforma um payload em `{ status, body }`. **Não** conhece tensor, scaler nem codificação: é o que torna a API testável sem treinar |
| `createRoutes`, `createApi`, `createRequestListener` | funções | As três rotas, o servidor `node:http` e o roteador com `404`/`405`/`500` |
| `readJsonBody`, `isJsonRequest`, `sendJson` | funções | Corpo com teto de 16 KB, checagem de `content-type` e resposta JSON |
| `listen` | função async | `server.listen` promisificado; devolve a porta **real** (importa com `--port=0`) |
| `API_PORT`, `API_BODY_LIMIT`, `resolvePort` | constantes / função | Porta padrão `3000`, teto do corpo e leitura de `--port=` |

---

## 📤 Exemplo de saída

```text
Fonte: German Credit — UCI/Statlog (Hofmann, 1994), one-hot
Arquivo: /caminho/do/projeto/data/german-credit.csv
Clientes lidos: 1000
Features: durationMonths, creditAmount, installmentRate, ..., foreignWorker=A201, foreignWorker=A202
Regularização: L2 = 0.003, dropout = 0.2

dense_Dense1 (Dense)        [[null,57]]               [null,16]                 928
dropout_Dropout1 (Dropout)  [[null,16]]               [null,16]                 0
dense_Dense2 (Dense)        [[null,16]]               [null,8]                  136
dropout_Dropout2 (Dropout)  [[null,8]]                [null,8]                  0
dense_Dense3 (Dense)        [[null,8]]                [null,1]                  9
Total params: 1073

Test loss: 0.5289
Train accuracy: 0.8200
Test accuracy: 0.7150
Baseline (classe majoritária): 0.7000
Diferença treino − teste: 0.1050

Matriz de confusão (limiar 0.5):
           | Predito BAIXO | Predito ALTO
-----------+---------------+-------------
Real BAIXO |      119 (TN) |      21 (FP)
Real ALTO  |       36 (FN) |      24 (TP)

Precision: 0.5333 - dos marcados como ALTO RISCO, quantos eram
Recall:    0.4000 - dos que eram ALTO RISCO, quantos foram pegos
F1-score:  0.4571 - média harmônica entre precision e recall

Curva ROC (O = limiar 0.5, . = aleatório):
    TPR
1.0 |                          **************|
    |                **********       ...    |
    |              **              ...       |
    |             *             ...          |
    |         ****          ....             |
    |                    ...                 |
0.5 |      ***        ...                    |
    |     *O       ...                       |
    |  ***     ....                          |
    |       ...                              |
    | *  ...                                 |
    |*...                                    |
0.0 +----------------------------------------+
    0.0                               FPR 1.0
AUC: 0.7477

Ajuste do limiar (FP custa 1, FN custa 5):
Estratégia     | Limiar |    FPR |    TPR | FP | FN | Custo
---------------+--------+--------+--------+----+----+------
Padrão (0.5)   | 0.5000 | 0.1500 | 0.4000 | 21 | 36 |   201
Youden (max J) | 0.2591 | 0.4071 | 0.8500 | 57 |  9 |   102
Menor custo    | 0.2591 | 0.4071 | 0.8500 | 57 |  9 |   102

Matriz no limiar escolhido (0.2591):
           | Predito BAIXO | Predito ALTO
-----------+---------------+-------------
Real BAIXO |       83 (TN) |      57 (FP)
Real ALTO  |        9 (FN) |      51 (TP)

Auditoria por sexo — limiar único (o modelo nunca recebeu esta coluna):
Grupo    |   N | Inadimp. real | Marcados ALTO | FN não pegos
---------+-----+---------------+---------------+-------------
Mulheres |  66 |         30.3% |         57.6% |        10.0%
Homens   | 134 |         29.9% |         52.2% |        17.5%

Razão de aprovação (regra dos 4/5): 0.888

Mitigação (limiares calibrados no TREINO, auditados no teste):
Política         | Limiar M | Limiar H | Razão aprov. | Acurácia | Custo
-----------------+----------+----------+--------------+----------+------
Limiar único     |   0.2591 |   0.2591 |        0.888 |   0.6700 |   102
Limiar por grupo |   0.2806 |   0.2396 |        1.032 |   0.6600 |   108
Política ativa: limiar único. Use --mitigar para decidir por grupo.

Probabilidade de alto risco: 0.7835
Classificação: ALTO RISCO
Modelo salvo em: /caminho/do/projeto/model
Modelo recarregado — test loss: 0.5289
Modelo recarregado — test accuracy: 0.7150
Modelo recarregado — probabilidade: 0.7835
Mesma predição do modelo original? sim
```

Os valores exatos não importam. O que se observa é o **comportamento**:

- ✅ `loss` de treino diminui ao longo das épocas;
- ✅ `val_loss` acompanha (se subir enquanto a de treino cai → overfitting);
- ✅ `accuracy` sobe;
- ✅ o baseline sai `0.7000` **sempre**, porque a divisão é [estratificada](validacao-cruzada.md#-validação-cruzada-e-split-estratificado) — a régua não depende do sorteio;
- ✅ as métricas antes e depois do `save`/`load` são **idênticas** → a persistência não perdeu nada;
- ✅ `(TP + TN) / total` reproduz a acurácia do `evaluate`;
- ✅ a curva ROC se descola da diagonal e a AUC fica bem acima de `0.5` → o score **ordena** os clientes;
- ✅ o limiar sugerido custa menos que o `0.5` herdado → havia margem na régua, não no modelo.

E três coisas que **só aparecem no dataset real**, e que são o motivo de ele estar aqui:

- ⚠️ a acurácia (`0.7150`) fica a 1,5 ponto do baseline (`0.7000`) → acurácia sozinha não diz se o modelo presta, e a AUC de `0.7477` ao lado mostra que o problema é o corte. Esta divisão é das piores do dataset: [`npm run cv`](validacao-cruzada.md#-validação-cruzada-e-split-estratificado) mede quase cinco pontos de ganho;
- ⚠️ precision (`0.5333`) e recall (`0.4000`) ficam longe uma da outra → no corte `0.5` o modelo deixa passar seis em cada dez maus pagadores, e é por isso que [ajustar o limiar](metricas.md#-ajuste-do-limiar-de-decisão) deixa de ser refinamento e vira necessidade;
- ⚠️ a razão de aprovação entre os grupos fica em `0.888` → o modelo trata mulheres e homens de forma diferente [sem nunca ter recebido a coluna de sexo](german-credit.md#a-coluna-que-o-modelo-não-recebeu). A tabela seguinte mostra o que um [limiar por grupo](mitigacao.md#-mitigação-da-disparidade) faz com esse número — `1.032` — e o que ele cobra: seis pontos de custo.

No dataset sintético, os mesmos números ficam em `0.9416` de acurácia contra `0.8423` de baseline, com AUC `0.9733` — o [lado a lado completo](german-credit.md#comparação-lado-a-lado) está na seção do dataset real. Note que **o alerta da acurácia vale para os dois**: com 15,8% de inadimplentes, `0.94` também são só 10 pontos acima de chutar sempre "bom pagador".

---

[⬅️ Voltar ao README](../README.md)
