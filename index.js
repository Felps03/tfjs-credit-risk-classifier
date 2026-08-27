const tf = require('@tensorflow/tfjs-node');

// --------------------------------------------------
// 0. Constantes do laboratório
// --------------------------------------------------
// Faixas conhecidas de antemão porque nós geramos os dados.
// Em um projeto real, estas estatísticas seriam calculadas
// APENAS sobre o conjunto de treino.
const INCOME_MIN = 2000;
const INCOME_RANGE = 13000;
const MAX_LATE_PAYMENTS = 5;

// Limiar da regra que gera os rótulos sintéticos.
const RISK_RULE_THRESHOLD = 1.35;

// Limiar de decisão do classificador: escolha de negócio, não do modelo.
const DECISION_THRESHOLD = 0.5;

// --------------------------------------------------
// 1. Pré-processamento
// --------------------------------------------------
// Estas funções são a ÚNICA fonte de verdade da normalização.
// Treino e inferência precisam usar exatamente as mesmas,
// caso contrário surge training-serving skew.
const normalizeIncome = (income) => (income - INCOME_MIN) / INCOME_RANGE;

const normalizeLatePayments = (latePayments) => latePayments / MAX_LATE_PAYMENTS;

const toFeatureVector = ({
  income,
  debtRatio,
  latePayments,
  creditUtilization,
}) => [
  normalizeIncome(income),
  debtRatio,
  normalizeLatePayments(latePayments),
  creditUtilization,
];

const classify = (probability) =>
  (probability >= DECISION_THRESHOLD ? 'ALTO RISCO' : 'BAIXO RISCO');

// --------------------------------------------------
// 2. Gerar dados sintéticos
// --------------------------------------------------
const createDataset = (total = 1200) => {
  const features = [];
  const labels = [];

  for (let i = 0; i < total; i += 1) {
    const customer = {
      income: INCOME_MIN + Math.random() * INCOME_RANGE,
      debtRatio: Math.random(),
      latePayments: Math.floor(Math.random() * (MAX_LATE_PAYMENTS + 1)),
      creditUtilization: Math.random(),
    };

    const featureVector = toFeatureVector(customer);
    const [incomeNormalized, debtRatio, latePaymentsNormalized, creditUtilization] =
      featureVector;

    // Regra usada somente para criar rótulos sintéticos.
    // O modelo NÃO recebe esta fórmula: ele precisa aprender o padrão.
    const riskScore =
      1.4 * debtRatio +
      1.2 * latePaymentsNormalized +
      1.0 * creditUtilization -
      0.8 * incomeNormalized;

    features.push(featureVector);
    labels.push([riskScore > RISK_RULE_THRESHOLD ? 1 : 0]);
  }

  return { features, labels };
};

// --------------------------------------------------
// 3. Separar treino e teste
// --------------------------------------------------
const splitDataset = ({ features, labels }, trainRatio = 0.8) => {
  const trainSize = Math.floor(features.length * trainRatio);

  return {
    trainFeatures: features.slice(0, trainSize),
    trainLabels: labels.slice(0, trainSize),
    testFeatures: features.slice(trainSize),
    testLabels: labels.slice(trainSize),
  };
};

// --------------------------------------------------
// 4. Criar a MLP
// --------------------------------------------------
const buildModel = () => {
  const model = tf.sequential();

  model.add(tf.layers.dense({
    inputShape: [4],
    units: 16,
    activation: 'relu',
  }));

  model.add(tf.layers.dense({
    units: 8,
    activation: 'relu',
  }));

  model.add(tf.layers.dense({
    units: 1,
    activation: 'sigmoid',
  }));

  model.compile({
    optimizer: tf.train.adam(0.001),
    loss: 'binaryCrossentropy',
    metrics: ['accuracy'],
  });

  return model;
};

// --------------------------------------------------
// 5. Treinar, avaliar e prever
// --------------------------------------------------
const main = async () => {
  const dataset = createDataset();
  const { trainFeatures, trainLabels, testFeatures, testLabels } =
    splitDataset(dataset);

  const xTrain = tf.tensor2d(trainFeatures);
  const yTrain = tf.tensor2d(trainLabels);
  const xTest = tf.tensor2d(testFeatures);
  const yTest = tf.tensor2d(testLabels);

  const model = buildModel();
  model.summary();

  await model.fit(xTrain, yTrain, {
    epochs: 40,
    batchSize: 32,
    validationSplit: 0.2,
    shuffle: true,
    callbacks: [
      tf.callbacks.earlyStopping({
        monitor: 'val_loss',
        patience: 5,
      }),
    ],
  });

  const [lossTensor, accuracyTensor] = model.evaluate(xTest, yTest);
  const testLoss = lossTensor.dataSync()[0];
  const testAccuracy = accuracyTensor.dataSync()[0];

  console.log('Test loss:', testLoss.toFixed(4));
  console.log('Test accuracy:', testAccuracy.toFixed(4));

  // ------------------------------------------------
  // 6. Novo cliente
  // ------------------------------------------------
  const newCustomer = {
    income: 3500,
    debtRatio: 0.72,
    latePayments: 3,
    creditUtilization: 0.88,
  };

  const input = tf.tensor2d([toFeatureVector(newCustomer)]);
  const prediction = model.predict(input);
  const probability = prediction.dataSync()[0];

  console.log('Probabilidade de alto risco:', probability.toFixed(4));
  console.log('Classificação:', classify(probability));

  // ------------------------------------------------
  // 7. Limpeza de memória
  // ------------------------------------------------
  tf.dispose([
    xTrain,
    yTrain,
    xTest,
    yTest,
    input,
    prediction,
    lossTensor,
    accuracyTensor,
  ]);
  model.dispose();
};

// Só executa quando chamado direto (node index.js).
// Ao ser importado pelos testes, apenas expõe as funções.
if (require.main === module) {
  main();
}

module.exports = {
  INCOME_MIN,
  INCOME_RANGE,
  MAX_LATE_PAYMENTS,
  RISK_RULE_THRESHOLD,
  DECISION_THRESHOLD,
  normalizeIncome,
  normalizeLatePayments,
  toFeatureVector,
  classify,
  createDataset,
  splitDataset,
  buildModel,
  main,
};
