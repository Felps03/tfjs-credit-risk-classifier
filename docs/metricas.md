# 📊 Métricas de avaliação e ajuste do limiar

[⬅️ README](../README.md) · [Dados](dados-sinteticos.md) · [German Credit](german-credit.md) · [Modelo](modelo.md) · **Métricas** · [Validação cruzada](validacao-cruzada.md) · [Mitigação](mitigacao.md) · [Inferência](inferencia.md) · [Serviço](servico.md) · [API](api.md) · [Testes](testes.md)

---

## 🧮 Matriz de confusão

A `accuracy` diz **quanto** o modelo acerta. A matriz de confusão diz **como ele erra** — e em risco de crédito os dois erros custam coisas muito diferentes:

| Erro | O que aconteceu | Custo real |
| ---- | --------------- | ---------- |
| **Falso positivo (FP)** | Cliente bom classificado como alto risco | Crédito negado a quem pagaria → receita perdida |
| **Falso negativo (FN)** | Cliente ruim classificado como baixo risco | Crédito concedido a quem não paga → prejuízo direto |

Uma acurácia de 96% pode esconder qualquer distribuição entre esses dois. Só a matriz separa.

### Como é calculada

```javascript
const computeConfusionMatrix = (model, xTest, yTest, threshold = DECISION_THRESHOLD) =>
  tf.tidy(() => {
    const probabilities = model.predict(xTest);

    const predicted = probabilities.greaterEqual(threshold).cast('int32').reshape([-1]);
    const actual = yTest.cast('int32').reshape([-1]);

    const [[trueNegatives, falsePositives], [falseNegatives, truePositives]] =
      tf.math.confusionMatrix(actual, predicted, 2).arraySync();

    return { truePositives, trueNegatives, falsePositives, falseNegatives, /* ... */ };
  });
```

O `tf.math.confusionMatrix` do core (não confundir com o `tfvis.render.confusionMatrix`, que é de navegador) devolve um tensor `2×2` em que **a linha é a classe real e a coluna é a predita**:

```text
[[TN, FP],
 [FN, TP]]
```

Todo o cálculo roda dentro de `tf.tidy()`, então nenhum tensor intermediário sobrevive à chamada — a função devolve apenas números.

### O papel do limiar

Repare no parâmetro `threshold`. **A matriz depende do limiar de decisão; a `loss` não.**

O modelo devolve sempre a mesma probabilidade — o que muda é onde você corta. Subir o corte de `0.5` para `0.7` torna o classificador mais exigente: menos falsos positivos, mais falsos negativos. É exatamente esse trade-off que a matriz torna visível, e ele se ajusta **sem retreinar nada**:

```javascript
computeConfusionMatrix(model, xTest, yTest, 0.7); // mais conservador na aprovação
```

Por isso o limiar é uma decisão de negócio: quem escolhe é o custo relativo de FP e FN, não o modelo.

### Saída

```text
Test accuracy: 0.9419
Baseline (classe majoritária): 0.8423

Matriz de confusão (limiar 0.5):
           | Predito BAIXO | Predito ALTO
-----------+---------------+-------------
Real BAIXO |      202 (TN) |       1 (FP)
Real ALTO  |       13 (FN) |      25 (TP)
```

A diagonal principal (TN e TP) são os acertos; fora dela, os erros. Uma checagem útil: `(TP + TN) / total` tem que reproduzir a `accuracy` do `evaluate` — acima, `(202 + 25) / 241 = 0.9419`. A suíte de testes verifica justamente isso.

E a matriz já conta uma história que a acurácia esconde. **Um único falso positivo** parece excelente até você olhar a linha de baixo: 13 dos 38 inadimplentes passaram como bons. O modelo não está calibrado — está **encolhido**, marcando ALTO RISCO só quando tem certeza absoluta, porque [positivos são raros no dataset](dados-sinteticos.md#-geração-dos-dados-sintéticos). Uma coluna de falsos positivos quase vazia é o retrato de um limiar no lugar errado, não de um classificador impecável — e o [ajuste do limiar](#-ajuste-do-limiar-de-decisão) derruba os 13 falsos negativos para 2, ao preço de 27 falsos positivos e de um custo que cai de `66` para `37`.

---

## 📏 Precision, recall e F1-score

A matriz mostra os erros; estas três métricas os **resumem em números comparáveis**. Todas derivam direto das mesmas contagens — nenhuma predição nova é feita:

| Métrica | Fórmula | Pergunta que responde |
| ------- | ------- | --------------------- |
| **Precision** | `TP / (TP + FP)` | Dos que o modelo marcou como ALTO RISCO, quantos eram de fato? |
| **Recall** | `TP / (TP + FN)` | Dos clientes que eram ALTO RISCO, quantos o modelo pegou? |
| **F1-score** | `2 · (P · R) / (P + R)` | Média harmônica das duas |

```javascript
const computeMetrics = ({ truePositives, falsePositives, falseNegatives }) => {
  const precision = safeDivide(truePositives, truePositives + falsePositives);
  const recall = safeDivide(truePositives, truePositives + falseNegatives);

  return {
    precision,
    recall,
    f1Score: safeDivide(2 * precision * recall, precision + recall),
  };
};
```

### Por que as três juntas

Cada uma isolada é **trivialmente manipulável**:

| Estratégia degenerada | Precision | Recall | F1 |
| --------------------- | --------- | ------ | -- |
| Marcar como ALTO RISCO só o caso mais óbvio | ~`1.00` | ~`0.01` | ~`0.02` |
| Marcar **todo mundo** como ALTO RISCO | baixa | `1.00` | baixa |

O F1 é **média harmônica**, não aritmética, exatamente por isso: ela é puxada pelo menor dos dois valores. Com precision `0.25` e recall `1.00`, a média aritmética daria `0.62` — o F1 dá `0.40`. Um modelo só consegue F1 alto quando acerta **nas duas pontas**.

### Divisão por zero

Um lote sem nenhum positivo previsto zeraria o denominador da precision. O projeto segue a convenção do scikit-learn (`zero_division=0`) — denominador zero vira `0`, nunca `NaN`:

```javascript
const safeDivide = (numerator, denominator) =>
  (denominator === 0 ? 0 : numerator / denominator);
```

Sem isso, um único lote degenerado contamina o relatório inteiro com `NaN`.

### Saída

```text
Precision: 0.9615 - dos marcados como ALTO RISCO, quantos eram
Recall:    0.6579 - dos que eram ALTO RISCO, quantos foram pegos
F1-score:  0.7813 - média harmônica entre precision e recall
```

Conferindo contra a matriz da seção anterior: `25 / (25 + 1) = 0.9615` e `25 / (25 + 13) = 0.6579`.

Precision `0.9615` é o caso extremo da tabela acima — o modelo quase só marca o que é óbvio. Repare que o **F1 desce para `0.78`** mesmo com precision quase perfeita: a média harmônica se recusa a premiar uma ponta às custas da outra. É exatamente para isso que ela existe.

### E o limiar, de novo

Como precision e recall vêm da matriz, **elas herdam a dependência do limiar**. Subir o corte tipicamente sobe a precision e derruba o recall; baixá-lo faz o contrário. Escolher o corte é escolher onde parar nessa gangorra — e é isso que a **curva ROC**, logo a seguir, permite fazer visualmente em vez de no chute.

---

## 📈 Curva ROC e AUC

Matriz, precision e recall descrevem **um** limiar. A curva ROC descreve **todos**.

Cada ponto da curva é um corte possível, com o que ele entrega e o que ele custa:

| Eixo | Nome | Fórmula | Leitura |
| ---- | ---- | ------- | ------- |
| Y | **TPR** (recall) | `TP / (TP + FN)` | Quanto do risco real o modelo captura |
| X | **FPR** | `FP / (FP + TN)` | Quanto de cliente bom ele queima no caminho |

A **AUC** — área sob a curva — resume tudo em um número e, por ser calculada sobre a curva inteira, **não depende do limiar**. Ela mede a capacidade de **ordenar**: é a probabilidade de um cliente de alto risco receber score maior que um de baixo risco.

| AUC | Significado |
| --- | ----------- |
| `1.00` | Separação perfeita — todo positivo pontua acima de todo negativo |
| `0.50` | Moeda: o score não ordena nada |
| `0.00` | Ordenação invertida (o modelo sabe separar, só trocou os rótulos) |

### Como é calculada

Não existe AUC no `@tensorflow/tfjs-node` — `tf.math` só expõe `confusionMatrix`. O cálculo é feito na mão, em três passos:

```javascript
// 1. Ordena do score mais alto para o mais baixo. Descer nessa lista
//    é ir afrouxando o limiar.
const ranked = scores
  .map((score, index) => ({ score, label: actuals[index] }))
  .sort((a, b) => b.score - a.score);

// 2. Acumula TP e FP a cada passo, registrando um ponto (FPR, TPR).
//    Scores empatados viram um único ponto: nenhum limiar os separa.
if (isLast || score !== ranked[index + 1].score) {
  points.push({ fpr: falsePositives / negatives, tpr: truePositives / positives, threshold: score });
}

// 3. Integra por trapézios — a curva é linear entre pontos consecutivos.
const auc = points.slice(1).reduce((total, point, index) => {
  const previous = points[index];

  return total + ((point.fpr - previous.fpr) * (point.tpr + previous.tpr)) / 2;
}, 0);
```

O tratamento de empates não é detalhe: sem ele a AUC fica dependente da ordem em que os dados chegaram. Com ele, o resultado coincide com a definição de **Mann-Whitney** (empate vale meio ponto) — e a suíte de testes compara as duas formas justamente para travar isso.

### Saída

```text
Curva ROC (O = limiar 0.5, . = aleatório):
    TPR
1.0 |     ***********************************|
    |   **                            ...    |
    | **                           ...       |
    |                           ...          |
    |O                      ....             |
    |                    ...                 |
0.5 |                 ...                    |
    |              ...                       |
    |          ....                          |
    |       ...                              |
    |    ...                                 |
    |*...                                    |
0.0 +----------------------------------------+
    0.0                               FPR 1.0
AUC: 0.9599
```

- `*` é a curva, `.` é a diagonal do classificador aleatório (`AUC = 0.5`);
- `O` marca **onde o limiar `0.5` colocou o modelo** na curva — o elo entre esta seção e a matriz de confusão;
- quanto mais a curva se descola da diagonal rumo ao canto superior esquerdo, maior a AUC.

### Por que a AUC não substitui as outras métricas

AUC alta significa que o modelo **ordena bem** — não que o limiar em uso é o certo. Um modelo com AUC `0.99` ainda pode estar operando num corte péssimo, com recall alto e precision no chão. A AUC diz que existe um bom ponto na curva; **escolher** esse ponto continua sendo decisão de negócio, guiada pelo custo relativo de FP e FN.

É por isso que as quatro medidas convivem no relatório:

| Medida | Depende do limiar? | Responde |
| ------ | ------------------ | -------- |
| `loss` | ❌ | Quão calibradas estão as probabilidades |
| Matriz / precision / recall / F1 | ✅ | Como o modelo se comporta **no corte escolhido** |
| AUC | ❌ | Quão bem o modelo **ordena**, em qualquer corte |

---

## 🧭 Ajuste do limiar de decisão

Até aqui o corte era **herdado**: `0.5`, porque é o meio do intervalo. Com a curva na mão dá para **escolhê-lo** — e há dois critérios, que respondem a perguntas diferentes.

### Youden: o melhor ponto quando os erros custam igual

O índice de Youden é `J = TPR - FPR` — geometricamente, o ponto da curva **mais distante da diagonal**:

```javascript
const chooseThresholdByYouden = (roc) => roc.points
  .map((point) => ({ ...point, youdenJ: point.tpr - point.fpr }))
  .reduce((best, point) => (point.youdenJ > best.youdenJ ? point : best));
```

É a referência neutra. E é o critério **errado** para crédito, porque parte de uma hipótese que não se sustenta: que recusar um bom pagador dói tanto quanto aprovar um inadimplente.

### Menor custo esperado: o critério que leva o negócio em conta

```javascript
const FALSE_POSITIVE_COST = 1;  // cliente bom recusado → receita perdida
const FALSE_NEGATIVE_COST = 5;  // cliente ruim aprovado → prejuízo direto
```

Números do laboratório — o que importa é a **razão** entre eles. Com FN valendo 5× FP, o corte ótimo desce: vale aceitar mais alarmes falsos para não deixar passar inadimplente.

Para pôr preço nos erros é preciso converter as **taxas** da curva de volta em **contagens absolutas** — por isso `computeRocCurve` devolve também `positives` e `negatives`:

```javascript
const scorePoint = (point, { positives, negatives }, costs) => {
  const falsePositives = Math.round(point.fpr * negatives);
  const falseNegatives = Math.round((1 - point.tpr) * positives);

  return {
    ...point,
    falsePositives,
    falseNegatives,
    cost: falsePositives * costs.falsePositive + falseNegatives * costs.falseNegative,
  };
};
```

### Saída

```text
Ajuste do limiar (FP custa 1, FN custa 5) — dataset sintético:
Estratégia     | Limiar |    FPR |    TPR | FP | FN | Custo
---------------+--------+--------+--------+----+----+------
Padrão (0.5)   | 0.5000 | 0.0049 | 0.6579 |  1 | 13 |    66
Youden (max J) | 0.2322 | 0.1330 | 0.9474 | 27 |  2 |    37
Menor custo    | 0.2322 | 0.1330 | 0.9474 | 27 |  2 |    37

Matriz no limiar escolhido (0.2322):
           | Predito BAIXO | Predito ALTO
-----------+---------------+-------------
Real BAIXO |      176 (TN) |      27 (FP)
Real ALTO  |        2 (FN) |      36 (TP)
```

Duas leituras dessa tabela:

1. **O corte herdado era caro.** Sair de `0.5` derrubou o custo de `66` para `37` — 44% mais barato, sem retreinar nada. A rede é a mesma; só a régua mudou. Compare as duas matrizes: os falsos negativos caem de **13 para 2**, ao preço de 26 falsos positivos. Como FN vale 5× FP, é uma troca boa por larga margem.
2. **Aqui Youden e custo coincidiram — nem sempre coincidem.** Nesta execução os dois critérios apontaram `0.2322`. Isso acontece quando o ponto de maior `TPR - FPR` já é o de menor custo. No dataset real eles frequentemente divergem: o critério de custo continua descendo depois de Youden, aceitando mais falsos positivos para eliminar falsos negativos — troca que só compensa porque FN vale 5×.

O efeito da razão de custos, isolado numa curva difícil:

| Critério | Limiar | FPR | TPR | FP | FN | Custo |
| -------- | ------ | --- | --- | -- | -- | ----- |
| Youden | `0.5000` | `0.4286` | `1.0000` | 3 | 0 | 3 |
| Menor custo, `FN = 1` | `0.9000` | `0.0000` | `0.3333` | 0 | 2 | 2 |
| Menor custo, `FN = 20` | `0.5000` | `0.4286` | `1.0000` | 3 | 0 | 3 |

Com FN barato o corte **sobe** (aprova pouco, erra pouco por excesso); com FN caro **desce** até capturar todos os positivos. Youden fica no mesmo lugar nos dois casos, porque não enxerga custo nenhum.

### Onde o corte é escolhido

Escolher o corte é fácil; escolher **onde** escolhê-lo é o que separa um número publicável de um número bonito.

Durante boa parte da vida deste projeto a resposta estava errada, e o erro tinha a assinatura de todos os erros caros que ele documenta: **nada quebrava**. A curva ROC era montada sobre o conjunto de TESTE, o corte de menor custo era escolhido nela, e a matriz publicada era recalculada no mesmo teste. O resultado é um corte ajustado às 200 linhas em que ele seria medido — o melhor corte **possível** naquele sorteio, que nenhum modelo alcança em dado novo.

A correção é uma fatia:

```text
1.000 clientes
├── 800 treino
│   ├── 640  ajustam os pesos
│   └── 160  CALIBRAÇÃO — validam o early stopping e escolhem o corte
└── 200 teste — intocado até a hora de reportar
```

A fatia não é nova. `model.fit` já reservava exatamente estes 20% para o early stopping, invisíveis dentro do `validationSplit`. O que mudou é que agora ela é separada antes, tem nome, e ganha uma segunda função legítima:

```javascript
const { fitCustomers, calibrationCustomers } = splitCalibration(trainCustomers);

await fitModel(model, xFit, yFit, { validationData: [xCal, yCal] });

const calibrationRoc = computeRocCurve(model, xCal, yCal);   // ESCOLHE o corte
const roc = computeRocCurve(model, xTest, yTest);            // reporta a AUC
```

Repare na segunda linha: a **AUC continua saindo do teste**, e isso é correto — ela não depende de limiar nenhum, então medi-la ali não otimiza nada. Só o CORTE precisa nascer fora.

### O tamanho do otimismo

A separação não é rigor abstrato: ela tem um número, e ele é impresso em toda execução.

```text
Ajuste do limiar em 160 clientes de CALIBRAÇÃO (FP custa 1, FN custa 5):
Estratégia     | Limiar |    FPR |    TPR | FP | FN | Custo
---------------+--------+--------+--------+----+----+------
Padrão (0.5)   | 0.5000 | 0.0893 | 0.3542 | 10 | 31 |   165
Youden (max J) | 0.2613 | 0.2857 | 0.7917 | 32 | 10 |    82
Menor custo    | 0.1587 | 0.5357 | 0.9792 | 60 |  1 |    65

Matriz no limiar escolhido (0.1587) — 200 clientes de TESTE:
           | Predito BAIXO | Predito ALTO
-----------+---------------+-------------
Real BAIXO |       61 (TN) |      79 (FP)
Real ALTO  |        5 (FN) |      55 (TP)
Custo por cliente: 0.406 na calibração (onde o corte foi escolhido) · 0.520 no teste (onde ele vale).
```

**28% mais caro por cliente** fora do conjunto que o escolheu. Essa distância é o que a versão anterior deste projeto publicava como se fosse desempenho, e é ela que a fatia de calibração devolve à realidade.

> Os dois blocos citam FP e FN diferentes de propósito: o de cima é a calibração, o de baixo é o teste. **Se eles batessem seria porque o corte foi escolhido onde é medido.** A tela do serviço diz a mesma coisa, na faixa dos três cortes.

A mesma correção vale para cada dobra da [validação cruzada](validacao-cruzada.md#-validação-cruzada-e-split-estratificado): a dobra de treino se parte de novo, o corte sai dessa fatia, e o custo reportado é o desse corte aplicado à dobra de teste. Foi por isso que os custos das tabelas de comparação subiram ~10% de uma medição para a outra, sem que nenhum modelo tivesse piorado.

### O que ainda não é ideal

A fatia de calibração faz **duas** coisas: valida a parada do early stopping e escolhe o corte. O ideal seriam dois conjuntos separados.

Com 1.000 linhas, não são. Gastar mais 200 clientes para separar as duas funções custa em capacidade — e a [comparação de arquiteturas](modelo.md#-comparando-arquiteturas) mostra que o gargalo deste projeto é exatamente o dado. A reutilização é declarada em vez de escondida, que é a diferença entre uma limitação e um erro.

### O caso extremo

Se o falso positivo for proibitivo, o ponto `(0, 0)` da curva — **não aprovar ninguém** — é um candidato legítimo, e o `threshold` sai como `Infinity`. A tabela imprime `(nenhum)` em vez de `Infinity`, e o comportamento está coberto por teste. Um classificador que se recusa a classificar é uma resposta válida quando o erro custa caro o suficiente.

### O que isso muda no projeto

O `DECISION_THRESHOLD = 0.5` continua sendo o corte da inferência: a escolha do limiar é **exibida como recomendação**, não aplicada automaticamente. Trocar o corte em produção é decisão de quem responde pelo custo — o código entrega a evidência, não a decisão.

---

[⬅️ Voltar ao README](../README.md)
