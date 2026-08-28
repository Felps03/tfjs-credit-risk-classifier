# 🔮 Inferência, persistência e memória

[⬅️ README](../README.md) · [Dados](dados-sinteticos.md) · [German Credit](german-credit.md) · [Modelo](modelo.md) · [Métricas](metricas.md) · [Validação cruzada](validacao-cruzada.md) · [Mitigação](mitigacao.md) · **Inferência** · [Serviço](servico.md) · [API](api.md) · [Testes](testes.md)

---

## 🔮 Inferência

Um cliente novo passa pela **mesma normalização** usada no treino:

```javascript
const newCustomer = {
  income: 3500,
  debtRatio: 0.72,
  latePayments: 3,
  creditUtilization: 0.88,
};

const input = tf.tensor2d([toFeatureVector(newCustomer)]);

const prediction  = model.predict(input);
const probability = prediction.dataSync()[0];

console.log('Classificação:', classify(probability));
```

Repare que é **a mesma `toFeatureVector`** usada na geração do dataset — não há uma segunda cópia da fórmula de normalização no caminho da inferência.

Renda baixa, endividamento alto, atrasos e crédito quase estourado — o modelo deve devolver uma probabilidade próxima de `1`.

---

## 💾 Persistência do modelo

Treinar a rede toda vez que o processo sobe é inviável fora de um laboratório. Depois de avaliar, o `main` grava o modelo em disco e o recarrega para provar que nada se perdeu no caminho.

### Salvar

O `@tensorflow/tfjs-node` registra o esquema `file://`, então `model.save` escreve direto no sistema de arquivos:

```javascript
const MODEL_DIR = path.join(__dirname, 'model');

await model.save(`file://${MODEL_DIR}`, { includeOptimizer: true });
```

Dois arquivos são gerados na pasta:

| Arquivo | Conteúdo |
| ------- | -------- |
| `model.json` | Topologia das camadas, metadados e o `trainingConfig` (loss, métricas e otimizador) |
| `weights.bin` | Os pesos em binário — e, com `includeOptimizer`, também os momentos do Adam |

O `includeOptimizer: true` é o detalhe que costuma escapar. **Sem ele**, o modelo recarregado volta *sem compilação*: `predict` funciona, mas `evaluate` e `fit` quebram até você chamar `compile` de novo. **Com ele**, o estado do otimizador (`iter`, `m`, `v`) viaja junto e o treino pode ser retomado exatamente de onde parou.

### Recarregar

```javascript
const model = await tf.loadLayersModel(`file://${MODEL_DIR}/model.json`);
```

Repare no caminho: para **salvar** aponta-se a **pasta**; para **carregar**, o arquivo **`model.json`**.

O `loadModel` do projeto ainda mantém uma rede de segurança — se o artefato tiver vindo sem otimizador, ele recompila com a mesma configuração do treino:

```javascript
const loadModel = async (dir = MODEL_DIR) => {
  const model = await tf.loadLayersModel(`${toFileUrl(dir)}/model.json`);

  if (!model.optimizer) {
    compileModel(model);
  }

  return model;
};
```

É por isso que `compileModel` foi extraída de `buildModel`: a configuração de compilação existe em um só lugar e serve tanto para o modelo novo quanto para o restaurado — o mesmo princípio de fonte única que vale para a normalização.

### Como o projeto verifica que funcionou

O `main` executa o ciclo completo — salva, dá `dispose` no modelo original, recarrega e mede de novo:

```javascript
await saveArtifacts(model, { source, scaler, featureNames, threshold, ... });
model.dispose();

const loadedModel = await loadModel();

evaluateModel(loadedModel, xTest, yTest); // mesmas loss e accuracy
predictRisk(loadedModel, newCustomer);    // mesma probabilidade, bit a bit
```

A comparação é por igualdade exata, não por tolerância: os pesos lidos do disco são os mesmos bytes que estavam na memória, então a probabilidade tem que bater até o último dígito. Se divergir, algo se perdeu na serialização.

> ✅ O pré-processamento **é** salvo junto. Esta seção já avisou, por várias versões, que carregar pesos novos com normalização antiga é uma das formas mais silenciosas de *training-serving skew* — e enquanto quem recarregava era o mesmo processo que tinha acabado de treinar, o aviso podia ficar como aviso. A [API REST](servico.md#-a-correção-o-modelo-como-pacote) acabou com esse luxo: `saveArtifacts` grava um `metadata.json` com o `scaler` medido naquele treino, a ordem exata das features e o limiar escolhido, e `loadArtifacts` **recusa** o pacote se qualquer um dos três não bater com o que o código gera hoje.

> ⚠️ O que continua no código: as constantes do dataset sintético (`INCOME_MIN`, `INCOME_RANGE`, `MAX_LATE_PAYMENTS`). Ali é legítimo — nós geramos os dados, então a escala nunca foi medida, e o `scaler` daquela fonte é `null` de propósito.

A pasta `/model/` está no `.gitignore`: artefato de build, não código-fonte.

---

## 🧹 Gerenciamento de memória

Tensores do TensorFlow.js vivem fora do garbage collector do JavaScript e precisam ser liberados explicitamente:

```javascript
tf.dispose([xTrain, yTrain, xTest, yTest]);

loadedModel.dispose();
```

O `dispose` do modelo é separado: libera os pesos, que não estão na lista de tensores avulsos. E são **dois** modelos ao longo do `main` — o treinado é descartado logo após o `save`, e o recarregado no fim.

Os tensores de curta duração (entrada, predição, métricas do `evaluate`) não aparecem nessa lista porque `predictRisk` e `evaluateModel` já os liberam internamente — quem chama recebe apenas números:

```javascript
const predictRisk = (model, customer, toVector = toFeatureVector) => {
  const input = tf.tensor2d([toVector(customer)]);
  const prediction = model.predict(input);
  const probability = prediction.dataSync()[0];

  tf.dispose([input, prediction]);

  return probability;
};
```

Em um script curto isso é apenas boa prática; em uma API de longa duração, esquecer o `dispose` vaza memória a cada requisição. Para blocos intermediários, `tf.tidy()` faz a limpeza automaticamente.

> 📌 Ao final do `main` ainda restam ~29 tensores vivos. Eles **não** são dos objetos acima: vêm do estado interno do `fit` e do otimizador Adam, que o tfjs mantém e não expõe para descarte. O ponto do `dispose` é liberar o que o código possui — o que ele não possui, só some com o processo.

---

[⬅️ Voltar ao README](../README.md)
