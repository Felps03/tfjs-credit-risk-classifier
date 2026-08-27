const { describe, it, after, before } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');

const {
  INCOME_MIN,
  INCOME_RANGE,
  MAX_LATE_PAYMENTS,
  DECISION_THRESHOLD,
  normalizeIncome,
  normalizeLatePayments,
  toFeatureVector,
  classify,
  createCustomers,
  toDataset,
  createDataset,
  toCsv,
  writeCustomersCsv,
  ensureCsv,
  readCustomersCsv,
  loadDatasetCsv,
  CSV_COLUMNS,
  CSV_LABEL_COLUMN,
  splitDataset,
  buildModel,
  saveModel,
  loadModel,
  predictRisk,
  computeConfusionMatrix,
  formatConfusionMatrix,
  computeMetrics,
  formatMetrics,
  computeRocCurve,
  formatRocCurve,
  FALSE_POSITIVE_COST,
  FALSE_NEGATIVE_COST,
  scorePoint,
  chooseThresholdByYouden,
  chooseThresholdByCost,
  formatTable,
  formatThresholdComparison,
  evaluateModel,
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
// CSV: escrever e ler
// ==================================================
describe('createCustomers', () => {
  it('dado um total, quando os clientes são gerados, então vêm em unidades brutas', () => {
    // Given / When — é o formato que vai para o arquivo
    const [customer] = createCustomers(1);

    // Then
    assert.ok(customer.income >= INCOME_MIN);
    assert.ok(customer.income <= INCOME_MIN + INCOME_RANGE);
    assert.ok(Number.isInteger(customer.latePayments));
    assert.ok(customer.latePayments <= MAX_LATE_PAYMENTS);
  });

  it('dados clientes gerados, quando o rótulo é lido, então é 0 ou 1', () => {
    // Given / When
    const riscos = new Set(createCustomers(200).map(({ risk }) => risk));

    // Then
    assert.deepEqual([...riscos].sort(), [0, 1]);
  });

  it('dados clientes brutos, quando convertidos em dataset, então batem com createDataset', () => {
    // Given — os dois caminhos precisam produzir a mesma estrutura
    const customers = createCustomers(10);

    // When
    const { features, labels } = toDataset(customers);

    // Then
    assert.deepEqual(features[0], toFeatureVector(customers[0]));
    assert.deepEqual(labels[0], [customers[0].risk]);
  });
});

describe('toCsv', () => {
  it('dados clientes, quando serializados, então a primeira linha é o cabeçalho', () => {
    // Given / When
    const [cabecalho] = toCsv(createCustomers(3)).split('\n');

    // Then
    assert.equal(cabecalho, CSV_COLUMNS.join(','));
  });

  it('dados N clientes, quando serializados, então há N linhas além do cabeçalho', () => {
    // Given / When
    const linhas = toCsv(createCustomers(7)).split('\n');

    // Then
    assert.equal(linhas.length, 8);
  });

  it('dada uma linha, quando as colunas são contadas, então há uma por campo', () => {
    // Given / When
    const [, primeira] = toCsv(createCustomers(1)).split('\n');

    // Then
    assert.equal(primeira.split(',').length, CSV_COLUMNS.length);
  });

  it('dados clientes, quando serializados, então latePayments e risk saem inteiros', () => {
    // Given — a precisão por coluna evita "3.000000" no arquivo
    const linhas = toCsv(createCustomers(20)).split('\n').slice(1);

    // When
    const indiceAtrasos = CSV_COLUMNS.indexOf('latePayments');
    const indiceRisco = CSV_COLUMNS.indexOf(CSV_LABEL_COLUMN);

    // Then
    linhas.forEach((linha) => {
      const campos = linha.split(',');

      assert.match(campos[indiceAtrasos], /^\d$/);
      assert.match(campos[indiceRisco], /^[01]$/);
    });
  });
});

describe('writeCustomersCsv / readCustomersCsv', () => {
  let dir;

  before(() => {
    dir = fs.mkdtempSync(path.join(os.tmpdir(), 'tfjs-csv-'));
  });

  after(() => {
    fs.rmSync(dir, { recursive: true, force: true });
  });

  const csvPath = (nome) => path.join(dir, nome);

  it('dado um caminho em pasta inexistente, quando o CSV é escrito, então a pasta é criada', () => {
    // Given
    const destino = path.join(dir, 'nova', 'sub', 'customers.csv');

    // When
    writeCustomersCsv(createCustomers(3), destino);

    // Then
    assert.ok(fs.existsSync(destino));
  });

  it('dado um CSV escrito, quando lido de volta, então devolve a mesma quantidade de clientes', async () => {
    // Given
    const destino = csvPath('contagem.csv');
    writeCustomersCsv(createCustomers(25), destino);

    // When
    const lidos = await readCustomersCsv(destino);

    // Then
    assert.equal(lidos.length, 25);
  });

  it('dado um CSV escrito, quando lido de volta, então os valores sobrevivem à ida e volta', async () => {
    // Given — CSV é texto: a comparação respeita a precisão gravada
    const originais = createCustomers(30);
    const destino = csvPath('roundtrip.csv');
    writeCustomersCsv(originais, destino);

    // When
    const lidos = await readCustomersCsv(destino);

    // Then
    lidos.forEach((lido, i) => {
      assert.ok(Math.abs(lido.income - originais[i].income) < 0.005);
      assert.ok(Math.abs(lido.debtRatio - originais[i].debtRatio) < 1e-6);
      assert.equal(lido.latePayments, originais[i].latePayments);
      assert.equal(lido.risk, originais[i].risk);
    });
  });

  it('dado um CSV lido, quando os tipos são inspecionados, então tudo já vem numérico', async () => {
    // Given — tf.data.csv faz o parse; não sobra string para converter
    const destino = csvPath('tipos.csv');
    writeCustomersCsv(createCustomers(5), destino);

    // When
    const [cliente] = await readCustomersCsv(destino);

    // Then
    CSV_COLUMNS.forEach((coluna) => {
      assert.equal(typeof cliente[coluna], 'number', `${coluna} deveria ser número`);
    });
  });

  it('dado um CSV com as colunas fora de ordem, quando lido, então os campos continuam corretos', async () => {
    // Given — tf.data.csv casa pelo NOME da coluna, não pela posição
    const destino = csvPath('fora-de-ordem.csv');
    fs.writeFileSync(
      destino,
      'risk,creditUtilization,latePayments,debtRatio,income\n1,0.880000,3,0.720000,3500.00\n',
    );

    // When
    const [cliente] = await readCustomersCsv(destino);

    // Then
    assert.equal(cliente.income, 3500);
    assert.equal(cliente.latePayments, 3);
    assert.equal(cliente.risk, 1);
  });

  it('dado um CSV ausente, quando ensureCsv roda, então o arquivo é criado', () => {
    // Given
    const destino = csvPath('novo.csv');

    // When
    const resultado = ensureCsv(destino, 5);

    // Then
    assert.equal(resultado.created, true);
    assert.ok(fs.existsSync(destino));
  });

  it('dado um CSV já existente, quando ensureCsv roda de novo, então o conteúdo não muda', () => {
    // Given — é isso que permite versionar o arquivo sem diff a cada run
    const destino = csvPath('estavel.csv');
    ensureCsv(destino, 5);
    const antes = fs.readFileSync(destino, 'utf8');

    // When
    const resultado = ensureCsv(destino, 5);

    // Then
    assert.equal(resultado.created, false);
    assert.equal(fs.readFileSync(destino, 'utf8'), antes);
  });

  it('dado um CSV existente, quando writeCustomersCsv é chamado direto, então sobrescreve', () => {
    // Given — o caminho explícito do `npm run seed`
    const destino = csvPath('regerado.csv');
    ensureCsv(destino, 5);
    const antes = fs.readFileSync(destino, 'utf8');

    // When
    writeCustomersCsv(createCustomers(9), destino);

    // Then
    const depois = fs.readFileSync(destino, 'utf8');

    assert.notEqual(depois, antes);
    assert.equal(depois.trim().split('\n').length, 10);
  });

  it('dado um arquivo inexistente, quando a leitura é tentada, então falha', async () => {
    // Given
    const destino = csvPath('nao-existe.csv');

    // When / Then
    await assert.rejects(() => readCustomersCsv(destino));
  });
});

describe('loadDatasetCsv', () => {
  let dir;
  let destino;
  let originais;

  before(() => {
    dir = fs.mkdtempSync(path.join(os.tmpdir(), 'tfjs-csv-ds-'));
    destino = path.join(dir, 'customers.csv');
    originais = createCustomers(40);
    writeCustomersCsv(originais, destino);
  });

  after(() => {
    fs.rmSync(dir, { recursive: true, force: true });
  });

  it('dado um CSV, quando carregado como dataset, então tem features e labels alinhados', async () => {
    // Given / When
    const { features, labels } = await loadDatasetCsv(destino);

    // Then
    assert.equal(features.length, 40);
    assert.equal(labels.length, 40);
  });

  it('dado um CSV, quando carregado, então as features saem normalizadas em [0, 1]', async () => {
    // Given — o arquivo guarda renda em reais; a normalização é do código
    const { features } = await loadDatasetCsv(destino);

    // When
    const foraDaFaixa = features.flat().filter((v) => v < 0 || v > 1);

    // Then
    assert.deepEqual(foraDaFaixa, []);
  });

  it('dado um CSV, quando carregado, então cada exemplo tem 4 features e 1 rótulo', async () => {
    // Given / When
    const { features, labels } = await loadDatasetCsv(destino);

    // Then
    assert.deepEqual([...new Set(features.map((f) => f.length))], [4]);
    assert.deepEqual([...new Set(labels.map((l) => l.length))], [1]);
  });

  it('dado um cliente do CSV, quando vetorizado, então usa a mesma toFeatureVector do treino', async () => {
    // Given — é isso que impede o CSV de virar uma segunda fonte de verdade
    const [cliente] = await readCustomersCsv(destino);

    // When
    const { features } = await loadDatasetCsv(destino);

    // Then
    assert.deepEqual(features[0], toFeatureVector(cliente));
  });

  it('dado o dataset do CSV, quando comparado ao gerado em memória, então a estrutura é idêntica', async () => {
    // Given
    const emMemoria = toDataset(originais);

    // When
    const doArquivo = await loadDatasetCsv(destino);

    // Then — mesma forma; os valores diferem apenas pela precisão gravada
    assert.equal(doArquivo.features.length, emMemoria.features.length);
    assert.deepEqual(doArquivo.labels, emMemoria.labels);
  });

  it('dado o dataset do CSV, quando dividido, então funciona com splitDataset sem adaptação', async () => {
    // Given — o formato de saída é o mesmo de createDataset
    const dataset = await loadDatasetCsv(destino);

    // When
    const { trainFeatures, testFeatures } = splitDataset(dataset);

    // Then
    assert.equal(trainFeatures.length, 32);
    assert.equal(testFeatures.length, 8);
  });

  it('dado o dataset do CSV, quando um modelo é treinado com ele, então o fit roda', async () => {
    // Given — prova que o caminho do arquivo alimenta o treino de verdade
    const tf = require('@tensorflow/tfjs-node');
    const { features, labels } = await loadDatasetCsv(destino);
    const model = buildModel();
    const x = tf.tensor2d(features);
    const y = tf.tensor2d(labels);

    // When
    const history = await model.fit(x, y, { epochs: 1, batchSize: 8, verbose: 0 });

    // Then
    assert.equal(history.history.loss.length, 1);

    tf.dispose([x, y]);
    model.dispose();
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

// ==================================================
// Matriz de confusão
// ==================================================
describe('computeConfusionMatrix', () => {
  const tf = require('@tensorflow/tfjs-node');

  // Stub: um "modelo" é qualquer coisa com `predict`. Assim a matriz é
  // testada contra probabilidades conhecidas, sem treinar rede nenhuma.
  const fakeModel = (probabilities) => ({
    predict: () => tf.tensor2d(probabilities.map((p) => [p])),
  });

  const withTensors = (probabilities, actuals, fn) => {
    const x = tf.tensor2d(probabilities.map((p) => [p]));
    const y = tf.tensor2d(actuals.map((a) => [a]));

    try {
      return fn(x, y);
    } finally {
      tf.dispose([x, y]);
    }
  };

  it('dadas predições conhecidas, quando a matriz é computada, então TP, TN, FP e FN batem', () => {
    // Given — 2 acertos positivos, 2 acertos negativos, 1 FP e 1 FN
    const probabilities = [0.9, 0.8, 0.1, 0.2, 0.7, 0.3];
    const actuals = [1, 1, 0, 0, 0, 1];

    // When
    const result = withTensors(probabilities, actuals, (x, y) =>
      computeConfusionMatrix(fakeModel(probabilities), x, y));

    // Then
    assert.equal(result.truePositives, 2);
    assert.equal(result.trueNegatives, 2);
    assert.equal(result.falsePositives, 1);
    assert.equal(result.falseNegatives, 1);
  });

  it('dada uma matriz computada, quando as células são somadas, então cobrem todos os exemplos', () => {
    // Given
    const probabilities = [0.9, 0.8, 0.1, 0.2, 0.7, 0.3];
    const actuals = [1, 1, 0, 0, 0, 1];

    // When
    const { truePositives, trueNegatives, falsePositives, falseNegatives } =
      withTensors(probabilities, actuals, (x, y) =>
        computeConfusionMatrix(fakeModel(probabilities), x, y));

    // Then
    assert.equal(
      truePositives + trueNegatives + falsePositives + falseNegatives,
      probabilities.length,
    );
  });

  it('dada uma matriz computada, quando o campo matrix é lido, então segue o layout [[TN, FP], [FN, TP]]', () => {
    // Given
    const probabilities = [0.9, 0.8, 0.1, 0.2, 0.7, 0.3];
    const actuals = [1, 1, 0, 0, 0, 1];

    // When
    const result = withTensors(probabilities, actuals, (x, y) =>
      computeConfusionMatrix(fakeModel(probabilities), x, y));

    // Then — linha = classe real, coluna = classe predita
    assert.deepEqual(result.matrix, [
      [result.trueNegatives, result.falsePositives],
      [result.falseNegatives, result.truePositives],
    ]);
  });

  it('dado um classificador perfeito, quando a matriz é computada, então não há FP nem FN', () => {
    // Given
    const probabilities = [0.99, 0.97, 0.01, 0.02];
    const actuals = [1, 1, 0, 0];

    // When
    const result = withTensors(probabilities, actuals, (x, y) =>
      computeConfusionMatrix(fakeModel(probabilities), x, y));

    // Then
    assert.equal(result.falsePositives, 0);
    assert.equal(result.falseNegatives, 0);
    assert.equal(result.truePositives, 2);
    assert.equal(result.trueNegatives, 2);
  });

  it('dada uma probabilidade exatamente no limiar, quando a matriz é computada, então conta como positiva', () => {
    // Given — mesmo corte >= usado por classify, sem uma segunda regra
    const probabilities = [DECISION_THRESHOLD];
    const actuals = [1];

    // When
    const result = withTensors(probabilities, actuals, (x, y) =>
      computeConfusionMatrix(fakeModel(probabilities), x, y));

    // Then
    assert.equal(result.truePositives, 1);
    assert.equal(result.falseNegatives, 0);
  });

  it('dado um limiar mais alto, quando a matriz é recomputada, então FP viram TN', () => {
    // Given — as mesmas probabilidades, só o corte muda
    const probabilities = [0.6, 0.6, 0.4, 0.4];
    const actuals = [0, 0, 0, 0];

    // When
    const padrao = withTensors(probabilities, actuals, (x, y) =>
      computeConfusionMatrix(fakeModel(probabilities), x, y));
    const exigente = withTensors(probabilities, actuals, (x, y) =>
      computeConfusionMatrix(fakeModel(probabilities), x, y, 0.7));

    // Then
    assert.equal(padrao.falsePositives, 2);
    assert.equal(exigente.falsePositives, 0);
    assert.equal(exigente.trueNegatives, 4);
  });

  it('dado um modelo real, quando a matriz e o evaluate são comparados, então a acurácia é a mesma', async () => {
    // Given — 1 época só para afastar as saídas de 0.5 exato, onde os
    // dois critérios divergem de propósito (veja o teste seguinte)
    const model = buildModel();
    const { features, labels } = createDataset(300);
    const x = tf.tensor2d(features);
    const y = tf.tensor2d(labels);

    await model.fit(x, y, { epochs: 1, batchSize: 32, verbose: 0 });

    // When
    const { accuracy } = evaluateModel(model, x, y);
    const { truePositives, trueNegatives } = computeConfusionMatrix(model, x, y);
    const derivada = (truePositives + trueNegatives) / features.length;

    // Then
    assert.ok(
      Math.abs(derivada - accuracy) < 1e-6,
      `esperado ~${accuracy}, derivado ${derivada}`,
    );

    tf.dispose([x, y]);
    model.dispose();
  });

  it('dada uma probabilidade de exatamente 0.5, quando comparada ao binaryAccuracy, então os critérios divergem', () => {
    // Given — `classify` e a matriz usam >=, mas o binaryAccuracy do Keras
    // arredonda, e tf.round(0.5) é 0. Só empate exato separa os dois.
    const y = tf.tensor2d([[1], [1], [0]]);
    const probabilities = tf.tensor2d([[0.5], [0.5], [0.5]]);

    // When
    const doKeras = Array.from(tf.metrics.binaryAccuracy(y, probabilities).dataSync());
    const nosso = Array.from(
      probabilities.greaterEqual(DECISION_THRESHOLD).cast('int32').dataSync(),
    );

    // Then
    assert.deepEqual(doKeras, [0, 0, 1], 'o Keras trata 0.5 como negativo');
    assert.deepEqual(nosso, [1, 1, 1], 'o projeto trata 0.5 como positivo');

    tf.dispose([y, probabilities]);
  });

  it('dada a matriz computada, quando os tensores são contados, então nenhum vazou', () => {
    // Given — computeConfusionMatrix roda dentro de tf.tidy
    const probabilities = [0.9, 0.1];
    const actuals = [1, 0];
    const antes = tf.memory().numTensors;

    // When
    withTensors(probabilities, actuals, (x, y) =>
      computeConfusionMatrix(fakeModel(probabilities), x, y));

    // Then
    assert.equal(tf.memory().numTensors, antes);
  });
});

describe('formatConfusionMatrix', () => {
  const confusion = {
    trueNegatives: 112,
    falsePositives: 2,
    falseNegatives: 1,
    truePositives: 125,
  };

  it('dada uma matriz, quando formatada, então tem cabeçalho, separador e duas linhas', () => {
    // Given / When
    const linhas = formatConfusionMatrix(confusion).split('\n');

    // Then
    assert.equal(linhas.length, 4);
  });

  it('dada uma matriz, quando formatada, então mostra as quatro contagens', () => {
    // Given / When
    const texto = formatConfusionMatrix(confusion);

    // Then
    assert.match(texto, /112 \(TN\)/);
    assert.match(texto, /2 \(FP\)/);
    assert.match(texto, /1 \(FN\)/);
    assert.match(texto, /125 \(TP\)/);
  });

  it('dada uma matriz, quando formatada, então todas as linhas ficam alinhadas', () => {
    // Given / When
    const linhas = formatConfusionMatrix(confusion).split('\n');
    const larguras = new Set(linhas.map((linha) => linha.length));

    // Then
    assert.equal(larguras.size, 1, 'colunas desalinhadas ficam ilegíveis no terminal');
  });
});

// ==================================================
// Precision, recall e F1-score
// ==================================================
describe('computeMetrics', () => {
  it('dada uma matriz conhecida, quando as métricas são derivadas, então batem com o cálculo manual', () => {
    // Given — 108 TP, 4 FP, 2 FN
    const confusion = {
      truePositives: 108,
      trueNegatives: 126,
      falsePositives: 4,
      falseNegatives: 2,
    };

    // When
    const { precision, recall } = computeMetrics(confusion);

    // Then
    assert.equal(precision, 108 / 112);
    assert.equal(recall, 108 / 110);
  });

  it('dadas precision e recall, quando o F1 é calculado, então é a média harmônica delas', () => {
    // Given
    const confusion = {
      truePositives: 108,
      trueNegatives: 126,
      falsePositives: 4,
      falseNegatives: 2,
    };

    // When
    const { precision, recall, f1Score } = computeMetrics(confusion);
    const esperado = (2 * precision * recall) / (precision + recall);

    // Then
    assert.equal(f1Score, esperado);
  });

  it('dada uma matriz, quando o F1 é comparado à forma direta, então coincidem', () => {
    // Given — F1 também é 2TP / (2TP + FP + FN); as duas fórmulas têm que fechar
    const confusion = {
      truePositives: 108,
      trueNegatives: 126,
      falsePositives: 4,
      falseNegatives: 2,
    };

    // When
    const { f1Score } = computeMetrics(confusion);
    const direto = (2 * 108) / (2 * 108 + 4 + 2);

    // Then
    assert.ok(
      Math.abs(f1Score - direto) < 1e-12,
      `esperado ~${direto}, recebido ${f1Score}`,
    );
  });

  it('dado um classificador perfeito, quando as métricas são derivadas, então as três valem 1', () => {
    // Given
    const confusion = {
      truePositives: 50,
      trueNegatives: 50,
      falsePositives: 0,
      falseNegatives: 0,
    };

    // When
    const { precision, recall, f1Score } = computeMetrics(confusion);

    // Then
    assert.equal(precision, 1);
    assert.equal(recall, 1);
    assert.equal(f1Score, 1);
  });

  it('dado um modelo que nunca prevê a classe positiva, quando as métricas são derivadas, então não há NaN', () => {
    // Given — TP + FP = 0 zeraria o denominador da precision
    const confusion = {
      truePositives: 0,
      trueNegatives: 80,
      falsePositives: 0,
      falseNegatives: 20,
    };

    // When
    const { precision, recall, f1Score } = computeMetrics(confusion);

    // Then
    assert.equal(precision, 0);
    assert.equal(recall, 0);
    assert.equal(f1Score, 0);
  });

  it('dado um lote sem exemplos positivos, quando as métricas são derivadas, então o recall é 0 e não NaN', () => {
    // Given — TP + FN = 0
    const confusion = {
      truePositives: 0,
      trueNegatives: 60,
      falsePositives: 5,
      falseNegatives: 0,
    };

    // When
    const { recall, f1Score } = computeMetrics(confusion);

    // Then
    assert.equal(recall, 0);
    assert.equal(f1Score, 0);
  });

  it('dado um modelo que aprova todo mundo, quando as métricas são derivadas, então o recall é 1 e a precision baixa', () => {
    // Given — recall 1 sozinho não significa nada; é o F1 que denuncia
    const confusion = {
      truePositives: 30,
      trueNegatives: 0,
      falsePositives: 170,
      falseNegatives: 0,
    };

    // When
    const { precision, recall, f1Score } = computeMetrics(confusion);

    // Then
    assert.equal(recall, 1);
    assert.equal(precision, 30 / 200);
    assert.ok(f1Score < recall, 'o F1 tem que puxar o recall para baixo');
  });

  it('dadas precision e recall desiguais, quando o F1 é calculado, então fica abaixo da média aritmética', () => {
    // Given — é essa a diferença entre média harmônica e aritmética
    const confusion = {
      truePositives: 20,
      trueNegatives: 100,
      falsePositives: 60,
      falseNegatives: 5,
    };

    // When
    const { precision, recall, f1Score } = computeMetrics(confusion);
    const aritmetica = (precision + recall) / 2;

    // Then
    assert.ok(f1Score < aritmetica, `${f1Score} deveria ser menor que ${aritmetica}`);
    assert.ok(f1Score > Math.min(precision, recall));
  });

  it('dado um modelo real, quando a matriz e as métricas são encadeadas, então tudo fica em [0, 1]', () => {
    // Given
    const tf = require('@tensorflow/tfjs-node');
    const model = buildModel();
    const { features, labels } = createDataset(200);
    const x = tf.tensor2d(features);
    const y = tf.tensor2d(labels);

    // When
    const metrics = computeMetrics(computeConfusionMatrix(model, x, y));

    // Then
    Object.entries(metrics).forEach(([nome, valor]) => {
      assert.ok(
        Number.isFinite(valor) && valor >= 0 && valor <= 1,
        `${nome} fora de [0, 1]: ${valor}`,
      );
    });

    tf.dispose([x, y]);
    model.dispose();
  });
});

describe('formatMetrics', () => {
  const metrics = { precision: 0.9643, recall: 0.9818, f1Score: 0.973 };

  it('dadas as métricas, quando formatadas, então há uma linha por métrica', () => {
    // Given / When
    const linhas = formatMetrics(metrics).split('\n');

    // Then
    assert.equal(linhas.length, 3);
  });

  it('dadas as métricas, quando formatadas, então os valores aparecem com 4 casas decimais', () => {
    // Given / When
    const texto = formatMetrics(metrics);

    // Then
    assert.match(texto, /Precision: +0\.9643/);
    assert.match(texto, /Recall: +0\.9818/);
    assert.match(texto, /F1-score: +0\.9730/);
  });
});

// ==================================================
// Curva ROC e AUC
// ==================================================
describe('computeRocCurve', () => {
  const tf = require('@tensorflow/tfjs-node');

  // Mesmo stub da matriz de confusão: qualquer coisa com `predict` serve.
  const rocOf = (scores, actuals) => {
    const model = { predict: () => tf.tensor2d(scores.map((s) => [s])) };
    const x = tf.tensor2d(scores.map((s) => [s]));
    const y = tf.tensor2d(actuals.map((a) => [a]));

    try {
      return computeRocCurve(model, x, y);
    } finally {
      tf.dispose([x, y]);
    }
  };

  // AUC pela definição de Mann-Whitney: a probabilidade de um positivo
  // receber score maior que um negativo, com empate valendo meio ponto.
  const aucPorPares = (scores, actuals) => {
    const positivos = scores.filter((unused, i) => actuals[i] === 1);
    const negativos = scores.filter((unused, i) => actuals[i] === 0);

    const soma = positivos.reduce((total, p) => total + negativos.reduce(
      (parcial, n) => {
        if (p > n) return parcial + 1;

        return p === n ? parcial + 0.5 : parcial;
      },
      0,
    ), 0);

    return soma / (positivos.length * negativos.length);
  };

  it('dado um classificador perfeito, quando a AUC é calculada, então vale 1', () => {
    // Given — todo positivo com score acima de todo negativo
    const scores = [0.9, 0.8, 0.7, 0.2, 0.1, 0.05];
    const actuals = [1, 1, 1, 0, 0, 0];

    // When
    const { auc } = rocOf(scores, actuals);

    // Then
    assert.equal(auc, 1);
  });

  it('dado um classificador invertido, quando a AUC é calculada, então vale 0', () => {
    // Given — a ordenação está exatamente ao contrário
    const scores = [0.1, 0.2, 0.3, 0.8, 0.9, 0.95];
    const actuals = [1, 1, 1, 0, 0, 0];

    // When
    const { auc } = rocOf(scores, actuals);

    // Then
    assert.equal(auc, 0);
  });

  it('dados scores todos iguais, quando a AUC é calculada, então vale 0.5', () => {
    // Given — sem poder de ordenação, o modelo é uma moeda
    const scores = [0.5, 0.5, 0.5, 0.5];
    const actuals = [1, 0, 1, 0];

    // When
    const { auc } = rocOf(scores, actuals);

    // Then
    assert.equal(auc, 0.5);
  });

  it('dada uma curva qualquer, quando a AUC do trapézio é comparada à contagem de pares, então coincidem', () => {
    // Given — trapézio e Mann-Whitney têm que dar o mesmo número
    const scores = [0.92, 0.71, 0.71, 0.5, 0.44, 0.3, 0.3, 0.12];
    const actuals = [1, 1, 0, 1, 0, 1, 0, 0];

    // When
    const { auc } = rocOf(scores, actuals);

    // Then
    assert.ok(
      Math.abs(auc - aucPorPares(scores, actuals)) < 1e-12,
      `trapézio ${auc} vs pares ${aucPorPares(scores, actuals)}`,
    );
  });

  it('dados scores reescalados de forma monotônica, quando a AUC é recalculada, então não muda', () => {
    // Given — a AUC mede ORDENAÇÃO, não a calibração das probabilidades
    const scores = [0.9, 0.6, 0.55, 0.4, 0.2, 0.1];
    const actuals = [1, 0, 1, 1, 0, 0];

    // When
    const original = rocOf(scores, actuals).auc;
    const reescalado = rocOf(scores.map((s) => s ** 3), actuals).auc;

    // Then
    assert.equal(reescalado, original);
  });

  it('dada uma curva, quando os extremos são inspecionados, então vai de (0,0) a (1,1)', () => {
    // Given
    const scores = [0.9, 0.6, 0.4, 0.1];
    const actuals = [1, 0, 1, 0];

    // When
    const { points } = rocOf(scores, actuals);
    const primeiro = points[0];
    const ultimo = points[points.length - 1];

    // Then
    assert.deepEqual([primeiro.fpr, primeiro.tpr], [0, 0]);
    assert.deepEqual([ultimo.fpr, ultimo.tpr], [1, 1]);
  });

  it('dada uma curva, quando os pontos são percorridos, então FPR e TPR nunca diminuem', () => {
    // Given — afrouxar o limiar só pode aumentar as duas taxas
    const scores = [0.92, 0.71, 0.71, 0.5, 0.44, 0.3, 0.12];
    const actuals = [1, 1, 0, 1, 0, 0, 0];

    // When
    const { points } = rocOf(scores, actuals);
    const regressoes = points.filter((point, index) => index > 0 && (
      point.fpr < points[index - 1].fpr || point.tpr < points[index - 1].tpr
    ));

    // Then
    assert.deepEqual(regressoes, []);
  });

  it('dados scores empatados, quando a curva é montada, então eles viram um único ponto', () => {
    // Given — nenhum limiar consegue separar scores iguais
    const scores = [0.8, 0.5, 0.5, 0.5, 0.2];
    const actuals = [1, 1, 0, 1, 0];

    // When
    const { points } = rocOf(scores, actuals);
    const thresholds = points.slice(1).map((point) => point.threshold);

    // Then — 3 scores distintos → 3 pontos além da origem
    assert.equal(new Set(thresholds).size, thresholds.length);
    assert.equal(points.length, 4);
  });

  it('dado um lote com uma classe só, quando a curva é calculada, então não há divisão por zero', () => {
    // Given — sem negativos o FPR seria 0/0
    const scores = [0.9, 0.8, 0.7];
    const actuals = [1, 1, 1];

    // When
    const { auc, points } = rocOf(scores, actuals);

    // Then
    assert.equal(auc, 0);
    assert.ok(points.every(({ fpr, tpr }) => Number.isFinite(fpr) && Number.isFinite(tpr)));
  });

  it('dado um modelo real, quando a curva é calculada, então a AUC fica em [0, 1]', () => {
    // Given
    const model = buildModel();
    const { features, labels } = createDataset(200);
    const x = tf.tensor2d(features);
    const y = tf.tensor2d(labels);

    // When
    const { auc } = computeRocCurve(model, x, y);

    // Then
    assert.ok(auc >= 0 && auc <= 1, `AUC fora da faixa: ${auc}`);

    tf.dispose([x, y]);
    model.dispose();
  });

  it('dada a curva calculada, quando os tensores são contados, então nenhum vazou', () => {
    // Given
    const scores = [0.9, 0.1];
    const actuals = [1, 0];
    const antes = tf.memory().numTensors;

    // When
    rocOf(scores, actuals);

    // Then
    assert.equal(tf.memory().numTensors, antes);
  });

  it('dado o ponto de operação do limiar padrão, quando comparado à matriz, então o TPR é o recall', () => {
    // Given — a matriz é um ponto da curva, não outra medida
    const scores = [0.9, 0.7, 0.4, 0.3, 0.6, 0.2];
    const actuals = [1, 1, 1, 0, 0, 0];
    const model = { predict: () => tf.tensor2d(scores.map((s) => [s])) };
    const x = tf.tensor2d(scores.map((s) => [s]));
    const y = tf.tensor2d(actuals.map((a) => [a]));

    // When
    const confusion = computeConfusionMatrix(model, x, y);
    const { recall } = computeMetrics(confusion);
    const { points } = computeRocCurve(model, x, y);

    // Then — o último ponto com threshold >= 0.5 é o corte padrão
    const doLimiar = points
      .filter((point) => point.threshold >= DECISION_THRESHOLD)
      .pop();

    assert.ok(doLimiar, 'a curva precisa conter o limiar padrão');
    assert.equal(doLimiar.tpr, recall);

    tf.dispose([x, y]);
  });
});

describe('formatRocCurve', () => {
  const tf = require('@tensorflow/tfjs-node');
  const scores = [0.9, 0.8, 0.7, 0.2, 0.1, 0.05];
  const actuals = [1, 1, 1, 0, 0, 0];

  const pointsOf = () => {
    const model = { predict: () => tf.tensor2d(scores.map((s) => [s])) };
    const x = tf.tensor2d(scores.map((s) => [s]));
    const y = tf.tensor2d(actuals.map((a) => [a]));

    try {
      return computeRocCurve(model, x, y).points;
    } finally {
      tf.dispose([x, y]);
    }
  };

  it('dada uma curva, quando desenhada, então tem cabeçalho, área, eixo e legenda', () => {
    // Given / When
    const linhas = formatRocCurve(pointsOf(), { width: 20, height: 8 }).split('\n');

    // Then — 1 título + 8 linhas de área + eixo + rótulos
    assert.equal(linhas.length, 8 + 3);
  });

  it('dada uma curva, quando desenhada, então a área tem a largura pedida', () => {
    // Given / When
    const linhas = formatRocCurve(pointsOf(), { width: 20, height: 8 }).split('\n');
    const area = linhas.slice(1, 9);

    // Then — rótulo (3) + ' |' + largura + '|'
    assert.deepEqual(new Set(area.map((linha) => linha.length)), new Set([3 + 2 + 20 + 1]));
  });

  it('dada uma curva, quando desenhada, então mostra a curva e a diagonal de referência', () => {
    // Given / When
    const desenho = formatRocCurve(pointsOf(), { width: 20, height: 8 });

    // Then
    assert.ok(desenho.includes('*'), 'a curva precisa aparecer');
    assert.ok(desenho.includes('.'), 'a diagonal do classificador aleatório precisa aparecer');
  });

  it('dado um ponto de operação, quando desenhado, então ele é marcado com O', () => {
    // Given
    const mark = { fpr: 0, tpr: 1 };

    // When
    const desenho = formatRocCurve(pointsOf(), { width: 20, height: 8, mark });

    // Then
    assert.ok(desenho.includes('O'), 'o limiar em uso precisa ser marcado');
  });

  it('dada uma curva sem ponto de operação, quando desenhada, então não há marcador', () => {
    // Given / When
    const desenho = formatRocCurve(pointsOf(), { width: 20, height: 8 });

    // Then
    assert.ok(!desenho.includes('O'));
  });
});

// ==================================================
// Ajuste do limiar a partir da curva
// ==================================================
describe('chooseThresholdByYouden / chooseThresholdByCost', () => {
  const tf = require('@tensorflow/tfjs-node');

  const rocOf = (scores, actuals) => {
    const model = { predict: () => tf.tensor2d(scores.map((s) => [s])) };
    const x = tf.tensor2d(scores.map((s) => [s]));
    const y = tf.tensor2d(actuals.map((a) => [a]));

    try {
      return computeRocCurve(model, x, y);
    } finally {
      tf.dispose([x, y]);
    }
  };

  // Curva em que subir o recall custa vários falsos positivos:
  // é aí que as duas estratégias discordam.
  const scores = [0.9, 0.8, 0.7, 0.6, 0.55, 0.5, 0.45, 0.4, 0.3, 0.2];
  const actuals = [1, 0, 1, 0, 0, 1, 0, 0, 0, 0];

  it('dada uma curva, quando a ROC é calculada, então traz as contagens por classe', () => {
    // Given / When — sem elas não dá para converter taxa em custo
    const roc = rocOf(scores, actuals);

    // Then
    assert.equal(roc.positives, 3);
    assert.equal(roc.negatives, 7);
  });

  it('dada uma curva, quando o limiar de Youden é escolhido, então maximiza TPR - FPR', () => {
    // Given
    const roc = rocOf(scores, actuals);

    // When
    const escolhido = chooseThresholdByYouden(roc);
    const melhorJ = Math.max(...roc.points.map((p) => p.tpr - p.fpr));

    // Then
    assert.equal(escolhido.youdenJ, melhorJ);
  });

  it('dado um classificador perfeito, quando Youden escolhe, então acha o corte sem erro nenhum', () => {
    // Given
    const roc = rocOf([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]);

    // When
    const escolhido = chooseThresholdByYouden(roc);

    // Then
    assert.equal(escolhido.tpr, 1);
    assert.equal(escolhido.fpr, 0);
    // os scores passam por float32, então o limiar volta aproximado
    assert.ok(
      Math.abs(escolhido.threshold - 0.8) < 1e-6,
      `esperado ~0.8, recebido ${escolhido.threshold}`,
    );
  });

  it('dado um custo, quando o limiar é escolhido, então nenhum outro ponto da curva custa menos', () => {
    // Given
    const roc = rocOf(scores, actuals);
    const costs = { falsePositive: 1, falseNegative: 5 };

    // When
    const escolhido = chooseThresholdByCost(roc, costs);
    const custos = roc.points.map((point) => scorePoint(point, roc, costs).cost);

    // Then
    assert.equal(escolhido.cost, Math.min(...custos));
  });

  it('dado um falso negativo caro, quando comparado a um barato, então o limiar escolhido é menor ou igual', () => {
    // Given — FN caro empurra o corte para baixo: aprovar menos duvidoso
    const roc = rocOf(scores, actuals);

    // When
    const barato = chooseThresholdByCost(roc, { falsePositive: 1, falseNegative: 1 });
    const caro = chooseThresholdByCost(roc, { falsePositive: 1, falseNegative: 20 });

    // Then
    assert.ok(
      caro.threshold <= barato.threshold,
      `FN caro deveria baixar o corte: ${caro.threshold} vs ${barato.threshold}`,
    );
    assert.ok(caro.tpr >= barato.tpr, 'FN caro precisa capturar mais positivos');
  });

  it('dados custos assimétricos, quando comparado a Youden, então as escolhas divergem', () => {
    // Given — Youden ignora custo; é essa a diferença entre os dois critérios
    const roc = rocOf(scores, actuals);

    // When
    const youden = chooseThresholdByYouden(roc);
    const porCusto = chooseThresholdByCost(roc, { falsePositive: 1, falseNegative: 1 });

    // Then
    assert.notEqual(porCusto.threshold, youden.threshold);
  });

  it('dado um falso positivo proibitivo, quando o limiar é escolhido, então o modelo não aprova ninguém', () => {
    // Given — o score mais alto é de um negativo, então QUALQUER aprovação
    // gera FP. O ponto (0, 0) da curva é um candidato legítimo.
    const roc = rocOf([0.9, 0.8, 0.7], [0, 1, 0]);

    // When
    const escolhido = chooseThresholdByCost(roc, {
      falsePositive: 1000,
      falseNegative: 1,
    });

    // Then
    assert.equal(escolhido.falsePositives, 0);
    assert.equal(escolhido.threshold, Infinity);
  });

  it('dado um falso positivo caro mas o topo da curva confiável, quando o limiar é escolhido, então ainda vale aprovar o mais óbvio', () => {
    // Given — aqui o score mais alto É positivo: dá para lucrar sem gerar FP
    const roc = rocOf(scores, actuals);

    // When
    const escolhido = chooseThresholdByCost(roc, {
      falsePositive: 1000,
      falseNegative: 1,
    });

    // Then
    assert.equal(escolhido.falsePositives, 0);
    assert.ok(Number.isFinite(escolhido.threshold), 'não precisa recusar todo mundo');
    assert.ok(escolhido.tpr > 0, 'o positivo mais óbvio deve ser capturado');
  });

  it('dado um ponto da curva, quando pontuado, então as taxas viram contagens absolutas', () => {
    // Given
    const roc = rocOf(scores, actuals);
    const point = { fpr: 2 / 7, tpr: 2 / 3, threshold: 0.6 };

    // When
    const pontuado = scorePoint(point, roc, { falsePositive: 1, falseNegative: 5 });

    // Then
    assert.equal(pontuado.falsePositives, 2);
    assert.equal(pontuado.falseNegatives, 1);
    assert.equal(pontuado.cost, 2 * 1 + 1 * 5);
  });

  it('dado o limiar escolhido, quando a matriz é recomputada nele, então os erros previstos se confirmam', () => {
    // Given — a promessa da curva tem que valer na matriz de verdade
    const model = { predict: () => tf.tensor2d(scores.map((s) => [s])) };
    const x = tf.tensor2d(scores.map((s) => [s]));
    const y = tf.tensor2d(actuals.map((a) => [a]));
    const roc = computeRocCurve(model, x, y);

    // When
    const escolhido = chooseThresholdByCost(roc, { falsePositive: 1, falseNegative: 5 });
    const confusion = computeConfusionMatrix(model, x, y, escolhido.threshold);

    // Then
    assert.equal(confusion.falsePositives, escolhido.falsePositives);
    assert.equal(confusion.falseNegatives, escolhido.falseNegatives);

    tf.dispose([x, y]);
  });

  it('dado o custo do projeto, quando as constantes são lidas, então o falso negativo é o erro caro', () => {
    // Given / When — em crédito, aprovar inadimplente dói mais que recusar bom pagador
    // Then
    assert.ok(
      FALSE_NEGATIVE_COST > FALSE_POSITIVE_COST,
      'a assimetria é o que justifica mexer no limiar',
    );
  });
});

describe('formatTable / formatThresholdComparison', () => {
  it('dadas linhas de tamanhos diferentes, quando a tabela é montada, então todas ficam alinhadas', () => {
    // Given
    const headers = ['Estratégia', 'Valor'];
    const rows = [['Curta', '1'], ['Uma bem mais longa', '1000']];

    // When
    const linhas = formatTable(headers, rows).split('\n');

    // Then
    assert.equal(new Set(linhas.map((linha) => linha.length)).size, 1);
  });

  it('dada uma tabela, quando montada, então tem cabeçalho, divisor e uma linha por item', () => {
    // Given / When
    const linhas = formatTable(['A', 'B'], [['1', '2'], ['3', '4']]).split('\n');

    // Then
    assert.equal(linhas.length, 4);
    assert.match(linhas[1], /^-+-\+--+$/);
  });

  it('dados candidatos, quando comparados, então a tabela mostra limiar, taxas, erros e custo', () => {
    // Given
    const candidates = [{
      label: 'Menor custo',
      point: {
        threshold: 0.423,
        fpr: 0.0167,
        tpr: 1,
        falsePositives: 2,
        falseNegatives: 0,
        cost: 2,
      },
    }];

    // When
    const texto = formatThresholdComparison(candidates);

    // Then
    assert.match(texto, /Menor custo/);
    assert.match(texto, /0\.4230/);
    assert.match(texto, /1\.0000/);
  });

  it('dado um candidato que não aprova ninguém, quando formatado, então o limiar infinito é legível', () => {
    // Given — Infinity.toFixed(4) sairia como "Infinity"
    const candidates = [{
      label: 'Nunca aprova',
      point: {
        threshold: Infinity,
        fpr: 0,
        tpr: 0,
        falsePositives: 0,
        falseNegatives: 3,
        cost: 15,
      },
    }];

    // When
    const texto = formatThresholdComparison(candidates);

    // Then
    assert.match(texto, /\(nenhum\)/);
    assert.ok(!texto.includes('Infinity'));
  });
});

// ==================================================
// Persistência: salvar e recarregar
// ==================================================
describe('saveModel / loadModel', () => {
  const customer = {
    income: 3500,
    debtRatio: 0.72,
    latePayments: 3,
    creditUtilization: 0.88,
  };

  let modelDir;
  let original;
  let originalProbability;

  before(async () => {
    // Given — um modelo treinado por 1 época só para os pesos saírem do
    // estado inicial, salvo em uma pasta temporária descartável.
    modelDir = fs.mkdtempSync(path.join(os.tmpdir(), 'tfjs-model-'));
    original = buildModel();

    const { features, labels } = createDataset(200);
    const tf = require('@tensorflow/tfjs-node');
    const x = tf.tensor2d(features);
    const y = tf.tensor2d(labels);

    await original.fit(x, y, { epochs: 1, batchSize: 32, verbose: 0 });
    tf.dispose([x, y]);

    originalProbability = predictRisk(original, customer);
    await saveModel(original, modelDir);
  });

  after(() => {
    original.dispose();
    fs.rmSync(modelDir, { recursive: true, force: true });
  });

  it('dado um modelo salvo, quando a pasta é inspecionada, então contém model.json e weights.bin', () => {
    // Given / When
    const arquivos = fs.readdirSync(modelDir).sort();

    // Then
    assert.deepEqual(arquivos, ['model.json', 'weights.bin']);
  });

  it('dado um modelo salvo, quando o model.json é lido, então traz a configuração de treino', () => {
    // Given — includeOptimizer: true grava loss, métricas e otimizador
    const artefato = JSON.parse(
      fs.readFileSync(path.join(modelDir, 'model.json'), 'utf8'),
    );

    // When
    const { trainingConfig } = artefato;

    // Then
    assert.equal(trainingConfig.loss, 'binary_crossentropy');
    assert.equal(trainingConfig.optimizer_config.class_name, 'Adam');
  });

  it('dado um modelo recarregado, quando a arquitetura é comparada, então é idêntica à original', async () => {
    // Given / When
    const loaded = await loadModel(modelDir);

    // Then
    assert.equal(loaded.layers.length, original.layers.length);
    assert.equal(loaded.countParams(), original.countParams());
    assert.deepEqual(loaded.inputs[0].shape, [null, 4]);
    assert.deepEqual(loaded.outputs[0].shape, [null, 1]);

    loaded.dispose();
  });

  it('dado um modelo recarregado, quando prevê o mesmo cliente, então devolve exatamente a mesma probabilidade', async () => {
    // Given — os pesos vieram do disco, não de uma nova inicialização
    const loaded = await loadModel(modelDir);

    // When
    const probability = predictRisk(loaded, customer);

    // Then
    assert.equal(probability, originalProbability);
    assert.equal(classify(probability), classify(originalProbability));

    loaded.dispose();
  });

  it('dado um modelo recarregado, quando o otimizador é inspecionado, então já vem compilado', async () => {
    // Given / When — sem compilação não daria para avaliar nem continuar o treino
    const loaded = await loadModel(modelDir);

    // Then
    assert.ok(loaded.optimizer, 'o modelo carregado precisa estar compilado');

    loaded.dispose();
  });

  it('dado um modelo recarregado, quando avaliado, então produz as mesmas métricas do original', async () => {
    // Given
    const tf = require('@tensorflow/tfjs-node');
    const { features, labels } = createDataset(100);
    const x = tf.tensor2d(features);
    const y = tf.tensor2d(labels);
    const loaded = await loadModel(modelDir);

    // When
    const antes = evaluateModel(original, x, y);
    const depois = evaluateModel(loaded, x, y);

    // Then
    assert.equal(depois.loss, antes.loss);
    assert.equal(depois.accuracy, antes.accuracy);

    tf.dispose([x, y]);
    loaded.dispose();
  });

  it('dado um modelo recarregado, quando o treino continua, então não lança', async () => {
    // Given — o ganho de persistir é retomar o treino, não só prever
    const tf = require('@tensorflow/tfjs-node');
    const { features, labels } = createDataset(64);
    const x = tf.tensor2d(features);
    const y = tf.tensor2d(labels);
    const loaded = await loadModel(modelDir);

    // When
    const history = await loaded.fit(x, y, {
      epochs: 1,
      batchSize: 32,
      verbose: 0,
    });

    // Then
    assert.equal(history.history.loss.length, 1);

    tf.dispose([x, y]);
    loaded.dispose();
  });

  it('dado um diretório sem modelo, quando o carregamento é tentado, então falha', async () => {
    // Given
    const inexistente = path.join(modelDir, 'nao-existe');

    // When / Then
    await assert.rejects(() => loadModel(inexistente));
  });
});
