const { describe, it, after } = require('node:test');
const assert = require('node:assert/strict');

const {
  INCOME_MIN,
  INCOME_RANGE,
  MAX_LATE_PAYMENTS,
  DECISION_THRESHOLD,
  normalizeIncome,
  normalizeLatePayments,
  toFeatureVector,
  classify,
  createDataset,
  splitDataset,
  buildModel,
} = require('../index');

// ==================================================
// Pré-processamento
// ==================================================
describe('normalizeIncome', () => {
  it('dado o piso da faixa de renda, quando normalizado, então resulta em 0', () => {
    // Given
    const income = INCOME_MIN;

    // When
    const result = normalizeIncome(income);

    // Then
    assert.equal(result, 0);
  });

  it('dado o teto da faixa de renda, quando normalizado, então resulta em 1', () => {
    // Given
    const income = INCOME_MIN + INCOME_RANGE;

    // When
    const result = normalizeIncome(income);

    // Then
    assert.equal(result, 1);
  });

  it('dado o meio da faixa, quando normalizado, então resulta em 0.5', () => {
    // Given
    const income = INCOME_MIN + INCOME_RANGE / 2;

    // When
    const result = normalizeIncome(income);

    // Then
    assert.equal(result, 0.5);
  });
});

describe('normalizeLatePayments', () => {
  it('dado nenhum atraso, quando normalizado, então resulta em 0', () => {
    // Given
    const latePayments = 0;

    // When
    const result = normalizeLatePayments(latePayments);

    // Then
    assert.equal(result, 0);
  });

  it('dado o máximo de atrasos, quando normalizado, então resulta em 1', () => {
    // Given
    const latePayments = MAX_LATE_PAYMENTS;

    // When
    const result = normalizeLatePayments(latePayments);

    // Then
    assert.equal(result, 1);
  });
});

describe('toFeatureVector', () => {
  it('dado o cliente de exemplo do README, quando convertido, então produz o vetor normalizado esperado', () => {
    // Given
    const customer = {
      income: 3500,
      debtRatio: 0.72,
      latePayments: 3,
      creditUtilization: 0.88,
    };

    // When
    const vector = toFeatureVector(customer);

    // Then
    assert.deepEqual(vector, [1500 / 13000, 0.72, 3 / 5, 0.88]);
  });

  it('dado qualquer cliente, quando convertido, então devolve exatamente 4 features', () => {
    // Given
    const customer = {
      income: 9000,
      debtRatio: 0.3,
      latePayments: 1,
      creditUtilization: 0.45,
    };

    // When
    const vector = toFeatureVector(customer);

    // Then
    assert.equal(vector.length, 4, 'a camada de entrada declara inputShape [4]');
  });

  it('dado o mesmo cliente convertido duas vezes, quando comparados, então os vetores são idênticos', () => {
    // Given — normalização determinística evita training-serving skew
    const customer = {
      income: 7777,
      debtRatio: 0.51,
      latePayments: 4,
      creditUtilization: 0.19,
    };

    // When
    const first = toFeatureVector(customer);
    const second = toFeatureVector(customer);

    // Then
    assert.deepEqual(first, second);
  });
});

// ==================================================
// Regra de decisão
// ==================================================
describe('classify', () => {
  it('dada uma probabilidade alta, quando classificada, então retorna ALTO RISCO', () => {
    // Given
    const probability = 0.9142;

    // When
    const label = classify(probability);

    // Then
    assert.equal(label, 'ALTO RISCO');
  });

  it('dada uma probabilidade baixa, quando classificada, então retorna BAIXO RISCO', () => {
    // Given
    const probability = 0.1234;

    // When
    const label = classify(probability);

    // Then
    assert.equal(label, 'BAIXO RISCO');
  });

  it('dada a probabilidade exatamente no limiar, quando classificada, então retorna ALTO RISCO', () => {
    // Given — o corte usa >=, então o limiar pertence à classe positiva
    const probability = DECISION_THRESHOLD;

    // When
    const label = classify(probability);

    // Then
    assert.equal(label, 'ALTO RISCO');
  });

  it('dada uma probabilidade logo abaixo do limiar, quando classificada, então retorna BAIXO RISCO', () => {
    // Given
    const probability = DECISION_THRESHOLD - Number.EPSILON;

    // When
    const label = classify(probability);

    // Then
    assert.equal(label, 'BAIXO RISCO');
  });
});

// ==================================================
// Geração do dataset
// ==================================================
describe('createDataset', () => {
  it('dado um total explícito, quando o dataset é gerado, então devolve esse número de exemplos', () => {
    // Given
    const total = 50;

    // When
    const { features, labels } = createDataset(total);

    // Then
    assert.equal(features.length, total);
    assert.equal(labels.length, total);
  });

  it('dado nenhum argumento, quando o dataset é gerado, então usa o padrão de 1200 clientes', () => {
    // Given / When
    const { features } = createDataset();

    // Then
    assert.equal(features.length, 1200);
  });

  it('dado um dataset gerado, quando as features são inspecionadas, então todas estão normalizadas entre 0 e 1', () => {
    // Given
    const { features } = createDataset(200);

    // When
    const foraDaFaixa = features
      .flat()
      .filter((value) => value < 0 || value > 1);

    // Then
    assert.deepEqual(foraDaFaixa, [], 'nenhuma feature pode sair de [0, 1]');
  });

  it('dado um dataset gerado, quando os rótulos são inspecionados, então cada um é 0 ou 1', () => {
    // Given
    const { labels } = createDataset(200);

    // When
    const valoresInvalidos = labels
      .flat()
      .filter((value) => value !== 0 && value !== 1);

    // Then
    assert.deepEqual(valoresInvalidos, []);
  });

  it('dado um dataset grande, quando os rótulos são contados, então as duas classes aparecem', () => {
    // Given — com 1200 amostras a chance de uma classe sumir é desprezível
    const { labels } = createDataset(1200);

    // When
    const classes = new Set(labels.flat());

    // Then
    assert.deepEqual([...classes].sort(), [0, 1], 'dataset degenerado treina nada');
  });

  it('dado um dataset gerado, quando cada exemplo é medido, então tem 4 features e 1 rótulo', () => {
    // Given
    const { features, labels } = createDataset(30);

    // When
    const larguraFeatures = new Set(features.map((f) => f.length));
    const larguraLabels = new Set(labels.map((l) => l.length));

    // Then
    assert.deepEqual([...larguraFeatures], [4]);
    assert.deepEqual([...larguraLabels], [1]);
  });
});

// ==================================================
// Divisão treino / teste
// ==================================================
describe('splitDataset', () => {
  it('dado o dataset padrão, quando dividido em 80/20, então treino e teste têm os tamanhos esperados', () => {
    // Given
    const dataset = createDataset(1000);

    // When
    const { trainFeatures, testFeatures } = splitDataset(dataset);

    // Then
    assert.equal(trainFeatures.length, 800);
    assert.equal(testFeatures.length, 200);
  });

  it('dado um dataset dividido, quando as partes são somadas, então nada é perdido', () => {
    // Given
    const dataset = createDataset(137);

    // When
    const { trainFeatures, testFeatures } = splitDataset(dataset);

    // Then
    assert.equal(trainFeatures.length + testFeatures.length, 137);
  });

  it('dado um dataset dividido, quando treino e teste são comparados, então não há sobreposição', () => {
    // Given — o conjunto de teste não pode vazar para o treino
    const dataset = createDataset(100);

    // When
    const { trainFeatures, testFeatures } = splitDataset(dataset);
    const treino = new Set(trainFeatures.map((f) => f.join(',')));
    const vazamento = testFeatures.filter((f) => treino.has(f.join(',')));

    // Then
    assert.deepEqual(vazamento, []);
  });

  it('dada uma proporção customizada, quando dividido, então respeita a proporção', () => {
    // Given
    const dataset = createDataset(100);

    // When
    const { trainFeatures, testFeatures } = splitDataset(dataset, 0.6);

    // Then
    assert.equal(trainFeatures.length, 60);
    assert.equal(testFeatures.length, 40);
  });

  it('dado um dataset dividido, quando features e labels são comparados, então ficam alinhados', () => {
    // Given
    const dataset = createDataset(100);

    // When
    const { trainFeatures, trainLabels, testFeatures, testLabels } =
      splitDataset(dataset);

    // Then
    assert.equal(trainFeatures.length, trainLabels.length);
    assert.equal(testFeatures.length, testLabels.length);
  });
});

// ==================================================
// Arquitetura do modelo
// ==================================================
describe('buildModel', () => {
  const model = buildModel();

  after(() => model.dispose());

  it('dado o modelo construído, quando as camadas são contadas, então há 3 camadas densas', () => {
    // Given / When
    const total = model.layers.length;

    // Then
    assert.equal(total, 3);
  });

  it('dado o modelo construído, quando a entrada é inspecionada, então aceita 4 features', () => {
    // Given / When
    const inputShape = model.inputs[0].shape;

    // Then
    assert.deepEqual(inputShape, [null, 4]);
  });

  it('dado o modelo construído, quando a saída é inspecionada, então produz um único valor', () => {
    // Given / When
    const outputShape = model.outputs[0].shape;

    // Then
    assert.deepEqual(outputShape, [null, 1]);
  });

  it('dado o modelo construído, quando os parâmetros são contados, então totalizam 225', () => {
    // Given — 80 (4x16+16) + 136 (16x8+8) + 9 (8x1+1)
    const esperado = 80 + 136 + 9;

    // When
    const total = model.countParams();

    // Then
    assert.equal(total, esperado);
  });

  it('dado o modelo construído, quando as ativações são lidas, então são relu, relu e sigmoid', () => {
    // Given / When
    const ativacoes = model.layers.map(
      (layer) => layer.getConfig().activation,
    );

    // Then
    assert.deepEqual(ativacoes, ['relu', 'relu', 'sigmoid']);
  });

  it('dado o modelo construído, quando a compilação é verificada, então usa binaryCrossentropy', () => {
    // Given / When
    const { loss } = model;

    // Then
    assert.equal(loss, 'binaryCrossentropy');
  });

  it('dado o modelo compilado, quando uma predição é feita, então devolve probabilidade entre 0 e 1', () => {
    // Given
    const tf = require('@tensorflow/tfjs-node');
    const customer = {
      income: 3500,
      debtRatio: 0.72,
      latePayments: 3,
      creditUtilization: 0.88,
    };
    const input = tf.tensor2d([toFeatureVector(customer)]);

    // When
    const prediction = model.predict(input);
    const probability = prediction.dataSync()[0];
    tf.dispose([input, prediction]);

    // Then
    assert.ok(
      probability >= 0 && probability <= 1,
      `sigmoid deve saturar em [0, 1], recebido ${probability}`,
    );
  });
});
