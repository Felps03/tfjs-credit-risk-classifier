const tf = require('@tensorflow/tfjs-node');

const { DECISION_THRESHOLD } = require('./00-constants');
const { formatTable } = require('./13a-format');

// --------------------------------------------------
// 13. Matriz de confusão
// --------------------------------------------------
// A `accuracy` diz quanto o modelo acerta; a matriz diz COMO ele erra.
// Em risco de crédito os dois erros custam coisas diferentes:
//   falso positivo (FP) → cliente bom recusado  → receita perdida
//   falso negativo (FN) → cliente ruim aprovado → prejuízo direto
//
// `tf.math.confusionMatrix` devolve linha = real, coluna = predito:
//   [[TN, FP],
//    [FN, TP]]
const computeConfusionMatrix = (
  model,
  xTest,
  yTest,
  threshold = DECISION_THRESHOLD,
) => tf.tidy(() => {
  const probabilities = model.predict(xTest);

  // Diferente da loss, a matriz depende do limiar de decisão:
  // mexer no corte redistribui FP e FN sem retreinar nada.
  //
  // Usa o mesmo `>=` de `classify`: o limiar pertence à classe positiva.
  // Atenção: a `accuracy` do `evaluate` vem de `binaryAccuracy`, que
  // arredonda — e `tf.round(0.5)` é 0 (round-half-to-even). Nos dois
  // critérios só discordam em probabilidade EXATAMENTE 0.5, o que
  // acontece, por exemplo, num modelo recém-inicializado cujas ReLUs
  // zeraram: sigmoid(0) = 0.5 na bica.
  const predicted = probabilities.greaterEqual(threshold).cast('int32').reshape([-1]);
  const actual = yTest.cast('int32').reshape([-1]);

  const [[trueNegatives, falsePositives], [falseNegatives, truePositives]] =
    tf.math.confusionMatrix(actual, predicted, 2).arraySync();

  return {
    truePositives,
    trueNegatives,
    falsePositives,
    falseNegatives,
    matrix: [[trueNegatives, falsePositives], [falseNegatives, truePositives]],
  };
});

const formatConfusionMatrix = ({
  trueNegatives,
  falsePositives,
  falseNegatives,
  truePositives,
}) => formatTable(
  ['', 'Predito BAIXO', 'Predito ALTO'],
  [
    ['Real BAIXO', `${trueNegatives} (TN)`, `${falsePositives} (FP)`],
    ['Real ALTO', `${falseNegatives} (FN)`, `${truePositives} (TP)`],
  ],
);

module.exports = {
  computeConfusionMatrix,
  formatConfusionMatrix,
};
