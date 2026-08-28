// --------------------------------------------------
// 14. Precision, recall e F1-score
// --------------------------------------------------
// Derivados direto da matriz — nenhuma predição nova é feita.
// As três respondem a perguntas diferentes sobre a MESMA classe positiva:
//   precision → dos que o modelo marcou como ALTO RISCO, quantos eram
//   recall    → dos que eram ALTO RISCO, quantos o modelo pegou
//   f1        → média harmônica das duas, que pune o desequilíbrio
//
// Otimizar uma sozinha é trivial e inútil: aprovar ninguém dá precision
// alta; recusar todo mundo dá recall 1. O F1 só sobe quando as duas sobem.

// Convenção do scikit-learn (`zero_division=0`): denominador zero vira 0,
// não NaN. Sem isso um lote sem positivos contamina o relatório inteiro.
const safeDivide = (numerator, denominator) =>
  (denominator === 0 ? 0 : numerator / denominator);

const computeMetrics = ({
  truePositives,
  falsePositives,
  falseNegatives,
}) => {
  const precision = safeDivide(truePositives, truePositives + falsePositives);
  const recall = safeDivide(truePositives, truePositives + falseNegatives);

  return {
    precision,
    recall,
    f1Score: safeDivide(2 * precision * recall, precision + recall),
  };
};

const METRIC_DESCRIPTIONS = [
  ['Precision', 'precision', 'dos marcados como ALTO RISCO, quantos eram'],
  ['Recall', 'recall', 'dos que eram ALTO RISCO, quantos foram pegos'],
  ['F1-score', 'f1Score', 'média harmônica entre precision e recall'],
];

const formatMetrics = (metrics) => {
  const labelWidth = Math.max(
    ...METRIC_DESCRIPTIONS.map(([label]) => label.length),
  );

  return METRIC_DESCRIPTIONS
    .map(([label, key, description]) => [
      `${label}:`.padEnd(labelWidth + 1),
      metrics[key].toFixed(4),
      `- ${description}`,
    ].join(' '))
    .join('\n');
};

module.exports = {
  safeDivide,
  computeMetrics,
  METRIC_DESCRIPTIONS,
  formatMetrics,
};
