const tf = require('@tensorflow/tfjs-node');

const { HIDDEN_UNITS, SHUFFLE_SEED } = require('./00-constants');
const {
  fitGroupThresholds,
  formatAudit,
  isFemale,
  summarizeDecisions,
} = require('./07-audit');
const { evaluateModel, majorityBaseline } = require('./16a-evaluate');
const { formatTable } = require('./13a-format');
const { balancedClassWeight, buildModel, fitModel } = require('./10-model');
const { computeRocCurve, rocFromScores } = require('./15-roc');
const { shuffle, splitCalibration, stratifiedFolds } = require('./09-split');
const { DEFAULT_SOURCE_ID, SOURCES } = require('./08-sources');
const { createRandom } = require('./02-synthetic');
const { computeConfusionMatrix } = require('./13-confusion');
const {
  chooseThresholdByCost,
  FALSE_NEGATIVE_COST,
  FALSE_POSITIVE_COST,
  formatThreshold,
} = require('./16-threshold');

// --------------------------------------------------
// 17. Validação cruzada
// --------------------------------------------------
// Um hold-out de 200 linhas responde "quanto o modelo acerta?" com uma
// incerteza que ninguém vê. A validação cruzada troca essa resposta por
// outra, honesta: k estimativas independentes, cada cliente pontuado
// exatamente uma vez por um modelo que NÃO o viu treinar.
//
// Custa k treinos em vez de um. Em troca, a barra de erro deixa de ser
// suposição — e a auditoria de disparidade passa a ter os 1.000 clientes
// para medir em vez de 64 mulheres.
const CV_FOLDS = 5;

// Média com erro padrão. Com uma amostra só, a variância é indefinida e
// o erro padrão sai 0 — não porque a medida seja perfeita, e sim porque
// uma medição não tem do que discordar.
const summarize = (values) => {
  const mean = values.reduce((total, value) => total + value, 0) / values.length;
  const variance = values.length < 2
    ? 0
    : values.reduce((total, value) => total + (value - mean) ** 2, 0)
      / (values.length - 1);

  return {
    mean,
    standardError: Math.sqrt(variance / values.length),
    lowest: Math.min(...values),
    highest: Math.max(...values),
  };
};

const crossValidate = async (source, options = {}) => {
  const {
    folds = CV_FOLDS,
    seed = SHUFFLE_SEED,
    units = HIDDEN_UNITS,
    l2 = source.regularization.l2,
    dropout = source.regularization.dropout,
    verbose = 0,
    // Sobrescreve a configuração de treino compartilhada. Existe por um
    // motivo específico: comparar arquiteturas com um orçamento FIXO é
    // justo com as redes e injusto com os modelos pequenos, que precisam
    // de mais passos para sair do lugar. A documentação mede as duas
    // coisas, e este argumento é o que torna a segunda reproduzível.
    training = {},

    // Peso por classe durante o treino, igual ao fluxo principal. Aqui
    // ele importa duas vezes: `npm run cv -- --balancear` é a única
    // medida honesta do efeito, porque uma execução só não distingue o
    // efeito do sorteio.
    balance = false,
    costs = {
      falsePositive: FALSE_POSITIVE_COST,
      falseNegative: FALSE_NEGATIVE_COST,
    },
  } = options;

  source.ensure();

  const customers = shuffle(await source.read(), createRandom(seed));
  const assignment = stratifiedFolds(customers, folds);

  // O score de cada cliente vem da dobra que o deixou de fora. Ao final,
  // este vetor tem uma predição fora da amostra para o dataset INTEIRO —
  // é ele que permite auditar 1.000 pessoas de uma vez.
  const scores = new Array(customers.length);
  const groupThresholds = new Array(customers.length);
  const calibration = [];
  const results = [];

  for (let fold = 0; fold < folds; fold += 1) {
    const inFold = (ignored, index) => assignment[index] === fold;
    const trainCustomers = customers.filter((...args) => !inFold(...args));
    const testCustomers = customers.filter(inFold);

    // O scaler é remedido em CADA dobra. Reaproveitar um scaler ajustado
    // no dataset inteiro vazaria o teste da dobra para dentro do treino,
    // e o erro seria invisível: o número sairia bom.
    const scaler = source.fitScaler(trainCustomers);
    const toVector = (customer) => source.toVector(customer, scaler);

    // A dobra de treino se parte de novo, pelo mesmo motivo do fluxo
    // principal: o corte de menor custo não pode ser escolhido na dobra
    // em que ele será medido. Sem isto, o `custo` de cada dobra é o custo
    // do melhor corte POSSÍVEL naquela dobra — um número que nenhum
    // modelo alcança em dado novo.
    const { fitCustomers, calibrationCustomers } = splitCalibration(trainCustomers);

    const fitLabels = fitCustomers.map(({ risk }) => [risk]);
    const xFit = tf.tensor2d(fitCustomers.map(toVector));
    const yFit = tf.tensor2d(fitLabels);
    const xCal = tf.tensor2d(calibrationCustomers.map(toVector));
    const yCal = tf.tensor2d(calibrationCustomers.map(({ risk }) => [risk]));
    const xTrain = tf.tensor2d(trainCustomers.map(toVector));
    const yTrain = tf.tensor2d(trainCustomers.map(({ risk }) => [risk]));
    const xTest = tf.tensor2d(testCustomers.map(toVector));
    const yTest = tf.tensor2d(testCustomers.map(({ risk }) => [risk]));

    const model = buildModel(source.featureNames.length, { units, l2, dropout });
    const classWeight = balance ? balancedClassWeight(fitLabels) : null;

    // eslint-disable-next-line no-await-in-loop
    const history = await fitModel(model, xFit, yFit, {
      verbose,
      validationData: [xCal, yCal],
      ...(classWeight === null ? {} : { classWeight }),
      ...training,
    });

    const { loss, accuracy } = evaluateModel(model, xTest, yTest);

    // A AUC continua saindo do teste — ela não depende de limiar, então
    // medi-la ali é legítimo. O CORTE sai da calibração, e o custo
    // reportado é o desse corte aplicado ao teste.
    const roc = computeRocCurve(model, xTest, yTest);
    const calibrationRoc = computeRocCurve(model, xCal, yCal);
    const threshold = chooseThresholdByCost(calibrationRoc, costs).threshold;

    // O custo é o do corte escolhido APLICADO ao teste, contado na
    // matriz — sem interpolar ponto de curva nenhum. É a diferença entre
    // "quanto custaria o melhor corte desta dobra" (a pergunta errada,
    // que não se responde antes de ver a dobra) e "quanto custou o corte
    // que eu tinha como escolher".
    const testConfusion = computeConfusionMatrix(model, xTest, yTest, threshold);
    const cost = testConfusion.falsePositives * costs.falsePositive
      + testConfusion.falseNegatives * costs.falseNegative;

    const foldScores = tf.tidy(() =>
      Array.from(model.predict(xTest).reshape([-1]).dataSync()));
    const trainScores = tf.tidy(() =>
      Array.from(model.predict(xTrain).reshape([-1]).dataSync()));

    let position = 0;

    customers.forEach((ignored, index) => {
      if (assignment[index] === fold) {
        scores[index] = foldScores[position];
        position += 1;
      }
    });

    calibration.push({ fold, trainCustomers, trainScores });

    results.push({
      fold,
      trainSize: trainCustomers.length,
      testSize: testCustomers.length,
      parameters: model.countParams(),
      // Quantas épocas o early stopping deixou correr. Arquiteturas
      // maiores costumam parar antes: elas decoram mais rápido.
      epochs: history.epoch.length,
      baseline: majorityBaseline(testCustomers.map(({ risk }) => [risk])),
      loss,
      accuracy,
      auc: roc.auc,
      threshold,
      cost,
    });

    tf.dispose([xFit, yFit, xCal, yCal, xTrain, yTrain, xTest, yTest]);
    model.dispose();
  }

  // O limiar da auditoria sai da curva FORA DA AMOSTRA, montada com as k
  // dobras juntas — a mesma regra de menor custo do fluxo principal.
  const labels = customers.map(({ risk }) => risk);
  const roc = rocFromScores(scores, labels);
  const threshold = chooseThresholdByCost(roc, costs).threshold;

  const audit = source.audit
    ? source.audit(customers, scores, threshold)
    : null;

  // A mitigação da validação cruzada precisa de um limiar POR CLIENTE:
  // cada um é cortado pelos limiares que suas quatro dobras de treino
  // calibraram. Como `auditByGroup` recebe um número ou um par, e não uma
  // lista, a auditoria roda sobre a DECISÃO já tomada — 1 para marcado,
  // 0 para não — com o corte trivial em 0.5. É a mesma contagem.
  calibration.forEach(({ fold, trainCustomers, trainScores }) => {
    const pair = fitGroupThresholds(trainCustomers, trainScores, threshold);

    customers.forEach((customer, index) => {
      if (assignment[index] === fold) {
        groupThresholds[index] = pair[isFemale(customer) ? 'women' : 'men'];
      }
    });
  });

  const decided = scores.map((score, index) => (score >= groupThresholds[index] ? 1 : 0));

  const mitigated = source.audit
    ? source.audit(customers, decided, 0.5)
    : null;

  // O que cada política cobra sobre o dataset inteiro. Sem esta linha a
  // mitigação pareceria de graça: a razão de aprovação melhora sempre,
  // porque é para ela que o limiar foi ajustado.
  const decisions = {
    pooled: summarizeDecisions(customers, scores, threshold, costs),
    mitigated: summarizeDecisions(customers, decided, 0.5, costs),
  };

  return {
    folds: results,
    parameters: results[0].parameters,
    summary: {
      baseline: summarize(results.map((result) => result.baseline)),
      accuracy: summarize(results.map((result) => result.accuracy)),
      auc: summarize(results.map((result) => result.auc)),
      cost: summarize(results.map((result) => result.cost)),
      epochs: summarize(results.map((result) => result.epochs)),
    },
    outOfSample: { scores, threshold, auc: roc.auc },
    audit,
    mitigated,
    decisions,
  };
};

const formatCrossValidation = ({ folds, summary, outOfSample }) => [
  formatTable(
    ['Dobra', 'Treino', 'Teste', 'Baseline', 'Acurácia', 'AUC', 'Limiar', 'Custo'],
    [
      ...folds.map((result) => [
        String(result.fold + 1),
        String(result.trainSize),
        String(result.testSize),
        result.baseline.toFixed(4),
        result.accuracy.toFixed(4),
        result.auc.toFixed(4),
        formatThreshold(result.threshold),
        String(result.cost),
      ]),
      [
        'Média',
        '',
        '',
        summary.baseline.mean.toFixed(4),
        summary.accuracy.mean.toFixed(4),
        summary.auc.mean.toFixed(4),
        '',
        summary.cost.mean.toFixed(1),
      ],
      [
        'Erro',
        '',
        '',
        `± ${summary.baseline.standardError.toFixed(4)}`,
        `± ${summary.accuracy.standardError.toFixed(4)}`,
        `± ${summary.auc.standardError.toFixed(4)}`,
        '',
        `± ${summary.cost.standardError.toFixed(1)}`,
      ],
    ],
  ),
  '',
  // A AUC fora da amostra NÃO é a média das AUCs por dobra: ela vem de
  // uma curva só, montada com todos os clientes juntos. As duas
  // respondem perguntas diferentes e não têm por que coincidir.
  `AUC sobre o dataset inteiro (curva única, score fora da amostra): ${outOfSample.auc.toFixed(4)}`,
].join('\n');

// Caminho de execução alternativo: em vez de UM hold-out com relatório
// completo, k treinos com a estimativa e sua incerteza.
const reportCrossValidation = async (sourceId = DEFAULT_SOURCE_ID, options = {}) => {
  const source = SOURCES[sourceId];
  const { folds = CV_FOLDS } = options;

  console.log('Fonte:', source.label);
  console.log('Arquivo:', source.csvPath);
  console.log(`Validação cruzada estratificada: ${folds} dobras\n`);

  const result = await crossValidate(source, options);

  console.log(formatCrossValidation(result));

  if (result.audit) {
    const total = result.folds.reduce((sum, { testSize }) => sum + testSize, 0);

    console.log('');
    console.log(`Auditoria sobre os ${total} clientes, cada um pontuado pela dobra que não o viu:`);
    console.log(formatAudit(result.audit));
    console.log('');
    console.log('A mesma auditoria com limiar por grupo, calibrado nas dobras de treino:');
    console.log(formatAudit(result.mitigated));
    console.log('');
    console.log(formatTable(
      ['Política', 'Razão aprov.', 'Acurácia', 'Custo'],
      [
        ['Limiar único', result.audit.approvalRatio.toFixed(3),
          result.decisions.pooled.accuracy.toFixed(4),
          String(result.decisions.pooled.cost)],
        ['Limiar por grupo', result.mitigated.approvalRatio.toFixed(3),
          result.decisions.mitigated.accuracy.toFixed(4),
          String(result.decisions.mitigated.cost)],
      ],
    ));
  }

  return result;
};

module.exports = {
  CV_FOLDS,
  summarize,
  crossValidate,
  formatCrossValidation,
  reportCrossValidation,
};
