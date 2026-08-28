const { formatTable } = require('./13a-format');

// --------------------------------------------------
// 16. Escolher o limiar a partir da curva
// --------------------------------------------------
// Até aqui o corte era herdado (`0.5`). Com a curva na mão dá para
// escolhê-lo — e há duas maneiras, que respondem a perguntas diferentes.

// Custos relativos dos dois erros. Números do laboratório, não do mercado:
// o que importa é a RAZÃO entre eles. Aqui, aprovar um inadimplente
// custa 5x mais do que recusar um bom pagador.
const FALSE_POSITIVE_COST = 1;
const FALSE_NEGATIVE_COST = 5;

// Converte as taxas do ponto de volta em contagens absolutas e põe preço.
const scorePoint = (point, { positives, negatives }, costs) => {
  const falsePositives = Math.round(point.fpr * negatives);
  const falseNegatives = Math.round((1 - point.tpr) * positives);

  return {
    ...point,
    falsePositives,
    falseNegatives,
    cost: falsePositives * costs.falsePositive
      + falseNegatives * costs.falseNegative,
  };
};

// Índice de Youden: J = TPR - FPR. Geometricamente, o ponto da curva mais
// distante da diagonal. É o melhor corte quando os dois erros custam IGUAL
// — hipótese que quase nunca vale em crédito, mas é a referência neutra.
const chooseThresholdByYouden = (roc) => roc.points
  .map((point) => ({ ...point, youdenJ: point.tpr - point.fpr }))
  // Empate resolvido pelo primeiro: os pontos vêm do limiar mais alto
  // para o mais baixo, então o desempate é sempre o corte mais exigente.
  .reduce((best, point) => (point.youdenJ > best.youdenJ ? point : best));

// Menor custo esperado. Aqui a assimetria entra em cena: com FN valendo
// 5x FP, o corte ótimo desce, aceitando mais alarmes falsos para não
// deixar passar inadimplente.
const chooseThresholdByCost = (roc, costs = {
  falsePositive: FALSE_POSITIVE_COST,
  falseNegative: FALSE_NEGATIVE_COST,
}) => roc.points
  .map((point) => scorePoint(point, roc, costs))
  .reduce((best, point) => (point.cost < best.cost ? point : best));

const formatThreshold = (threshold) =>
  (Number.isFinite(threshold) ? threshold.toFixed(4) : '  (nenhum)');

const formatThresholdComparison = (candidates) => formatTable(
  ['Estratégia', 'Limiar', 'FPR', 'TPR', 'FP', 'FN', 'Custo'],
  candidates.map(({ label, point }) => [
    label,
    formatThreshold(point.threshold),
    point.fpr.toFixed(4),
    point.tpr.toFixed(4),
    String(point.falsePositives),
    String(point.falseNegatives),
    String(point.cost),
  ]),
);

module.exports = {
  FALSE_POSITIVE_COST,
  FALSE_NEGATIVE_COST,
  scorePoint,
  chooseThresholdByYouden,
  chooseThresholdByCost,
  formatThreshold,
  formatThresholdComparison,
};
