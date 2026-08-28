const tf = require('@tensorflow/tfjs-node');

// --------------------------------------------------
// 15. Curva ROC e AUC
// --------------------------------------------------
// Matriz, precision e recall descrevem UM limiar. A ROC descreve TODOS:
// cada ponto é um corte possível, com sua taxa de acerto (TPR) e seu
// custo em alarmes falsos (FPR).
//
// A AUC — área sob essa curva — resume a curva inteira em um número e,
// por isso, NÃO depende do limiar. Ela mede a capacidade de ORDENAR:
// é a probabilidade de um cliente de alto risco receber score maior que
// um de baixo risco. 0.5 = moeda; 1.0 = separação perfeita.
// A curva não precisa de um modelo: precisa de scores e rótulos. Separar
// as duas coisas é o que permite traçar a ROC sobre predições que vieram
// de VÁRIOS modelos — é exatamente o caso da validação cruzada, em que
// cada cliente foi pontuado pela dobra que não o viu treinar.
const rocFromScores = (scores, actuals) => {
  // Do score mais alto para o mais baixo: descer nessa lista é ir
  // afrouxando o limiar, aprovando cada vez mais casos como positivos.
  const ranked = scores
    .map((score, index) => ({ score, label: actuals[index] }))
    .sort((a, b) => b.score - a.score);

  const positives = ranked.filter(({ label }) => label === 1).length;
  const negatives = ranked.length - positives;

  // Sem as duas classes não há TPR nem FPR: a curva é indefinida.
  if (positives === 0 || negatives === 0) {
    return {
      points: [{ fpr: 0, tpr: 0 }, { fpr: 1, tpr: 1 }],
      auc: 0,
      positives,
      negatives,
    };
  }

  const points = [{ fpr: 0, tpr: 0, threshold: Infinity }];
  let truePositives = 0;
  let falsePositives = 0;

  ranked.forEach(({ score, label }, index) => {
    truePositives += label;
    falsePositives += 1 - label;

    // Scores empatados não podem ser separados por limiar nenhum,
    // então viram um único ponto da curva.
    const isLast = index === ranked.length - 1;

    if (isLast || score !== ranked[index + 1].score) {
      points.push({
        fpr: falsePositives / negatives,
        tpr: truePositives / positives,
        threshold: score,
      });
    }
  });

  // Área por trapézios: a curva é linear entre pontos consecutivos.
  const auc = points.slice(1).reduce((total, point, index) => {
    const previous = points[index];

    return total + ((point.fpr - previous.fpr) * (point.tpr + previous.tpr)) / 2;
  }, 0);

  // positives e negatives saem junto porque converter as taxas de volta
  // em contagens absolutas é o que permite pôr preço nos erros.
  return { points, auc, positives, negatives };
};

const computeRocCurve = (model, xTest, yTest) => {
  const { scores, actuals } = tf.tidy(() => ({
    scores: Array.from(model.predict(xTest).reshape([-1]).dataSync()),
    actuals: Array.from(yTest.reshape([-1]).dataSync()),
  }));

  return rocFromScores(scores, actuals);
};

// TPR da curva em um FPR qualquer, interpolando entre os pontos vizinhos.
// Serve só para desenhar: os pontos reais continuam sendo os da varredura.
const interpolateTpr = (points, fpr) => {
  const next = points.findIndex((point) => point.fpr >= fpr);

  if (next <= 0) {
    return next === 0 ? points[0].tpr : 1;
  }

  const previous = points[next - 1];
  const span = points[next].fpr - previous.fpr;

  return span === 0
    ? points[next].tpr
    : previous.tpr + ((fpr - previous.fpr) / span) * (points[next].tpr - previous.tpr);
};

const ROC_PLOT_WIDTH = 40;
const ROC_PLOT_HEIGHT = 12;

// Gráfico em texto: '*' é a curva, '.' é a diagonal do classificador
// aleatório e 'O' marca o limiar em uso. Quanto mais a curva se afasta
// da diagonal em direção ao canto superior esquerdo, maior a AUC.
const formatRocCurve = (points, options = {}) => {
  const {
    width = ROC_PLOT_WIDTH,
    height = ROC_PLOT_HEIGHT,
    mark = null,
  } = options;

  const columnFpr = (column) => column / (width - 1);

  const rows = Array.from({ length: height }, (unused, row) => {
    const upper = 1 - (row / height);
    const lower = 1 - ((row + 1) / height);
    const isLastRow = row === height - 1;
    const inBand = (value) =>
      value <= upper && (isLastRow ? value >= lower : value > lower);

    const cells = Array.from({ length: width }, (ignored, column) => {
      const fpr = columnFpr(column);

      if (mark
        && inBand(mark.tpr)
        && Math.abs(fpr - mark.fpr) <= 0.5 / (width - 1)) {
        return 'O';
      }

      if (inBand(interpolateTpr(points, fpr))) {
        return '*';
      }

      return inBand(fpr) ? '.' : ' ';
    });

    const label = { 0: '1.0', [Math.floor(height / 2)]: '0.5' }[row] ?? '   ';

    return `${label} |${cells.join('')}|`;
  });

  return [
    '    TPR',
    ...rows,
    `0.0 +${'-'.repeat(width)}+`,
    `    0.0${' '.repeat(Math.max(width - 9, 1))}FPR 1.0`,
  ].join('\n');
};

module.exports = {
  rocFromScores,
  computeRocCurve,
  interpolateTpr,
  ROC_PLOT_WIDTH,
  ROC_PLOT_HEIGHT,
  formatRocCurve,
};
