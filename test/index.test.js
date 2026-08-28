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
  SYNTHETIC_SEED,
  SYNTHETIC_TOTAL,
  SYNTHETIC_POSITIVE_RATE,
  SYNTHETIC_FEATURE_NOISE,
  SYNTHETIC_LABEL_NOISE,
  SYNTHETIC_BOUNDS,
  createGaussian,
  clamp,
  quantile,
  riskScore,
  measureCustomer,
  createCustomers,
  toDataset,
  createDataset,
  toCsv,
  writeCustomersCsv,
  ensureCsv,
  readCustomersCsv,
  loadDatasetCsv,
  CSV_PATH,
  CSV_COLUMNS,
  CSV_LABEL_COLUMN,
  splitDataset,
  L2_LAMBDA,
  DROPOUT_RATE,
  createRegularizer,
  HIDDEN_UNITS,
  ARCHITECTURES,
  compareArchitectures,
  formatArchitectureComparison,
  resolveUnits,
  resolveArchitectureRun,
  parseNumericFlag,
  resolveRegularization,
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
  GERMAN_CSV_PATH,
  GERMAN_NUMERIC,
  GERMAN_CATEGORICAL,
  GERMAN_AUDIT_COLUMN,
  GERMAN_AUDIT_CODES,
  FEMALE_CODE,
  oneHotEncode,
  ordinalEncode,
  germanFeatureNames,
  toGermanVector,
  isFemale,
  toAuditRows,
  summarizeGroup,
  thresholdFor,
  auditByGroup,
  rateThreshold,
  fitGroupThresholds,
  summarizeDecisions,
  formatMitigation,
  resolveMitigation,
  approvalRatio,
  formatAudit,
  GERMAN_ORDINAL_SOURCE,
  GERMAN_COLUMNS,
  GERMAN_PRECISION,
  SHUFFLE_SEED,
  parseDelimited,
  toOrdinal,
  toGermanCustomer,
  parseGermanCsv,
  fitMinMaxScaler,
  applyMinMaxScaler,
  SYNTHETIC_SOURCE,
  GERMAN_SOURCE,
  SOURCES,
  DEFAULT_SOURCE_ID,
  resolveSourceId,
  createRandom,
  shuffle,
  splitCustomers,
  stratifiedSplitCustomers,
  stratifiedFolds,
  summarize,
  resolveFolds,
  CV_FOLDS,
  crossValidate,
  formatCrossValidation,
  rocFromScores,
  TRAINING,
  majorityBaseline,
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
// Ruído e desbalanceamento
// ==================================================
const desvioPadrao = (valores) => {
  const media = valores.reduce((a, b) => a + b, 0) / valores.length;
  const variancia = valores
    .reduce((soma, valor) => soma + (valor - media) ** 2, 0) / valores.length;

  return { media, desvio: Math.sqrt(variancia) };
};

const taxaPositivos = (customers) =>
  customers.filter(({ risk }) => risk === 1).length / customers.length;

describe('createGaussian', () => {
  it('dados muitos sorteios, quando medidos, então a média fica perto de 0 e o desvio de 1', () => {
    // Given — Box-Muller precisa devolver a normal padrão, não qualquer curva
    const gaussian = createGaussian(createRandom(1));

    // When
    const amostras = Array.from({ length: 20000 }, () => gaussian());
    const { media, desvio } = desvioPadrao(amostras);

    // Then
    assert.ok(Math.abs(media) < 0.03, `media ${media}`);
    assert.ok(Math.abs(desvio - 1) < 0.03, `desvio ${desvio}`);
  });

  it('dados sorteios normais, quando comparados, então há valores dos dois lados de 0', () => {
    // Given / When — uma normal simétrica não pode ser só positiva
    const gaussian = createGaussian(createRandom(2));
    const amostras = Array.from({ length: 200 }, () => gaussian());

    // Then
    assert.ok(amostras.some((valor) => valor > 0));
    assert.ok(amostras.some((valor) => valor < 0));
  });

  it('dada a mesma semente, quando o gerador roda de novo, então repete a sequência', () => {
    // Given / When
    const primeira = Array.from({ length: 5 }, createGaussian(createRandom(9)));
    const segunda = Array.from({ length: 5 }, createGaussian(createRandom(9)));

    // Then
    assert.deepEqual(segunda, primeira);
  });

  it('dado um sorteio uniforme de 0, quando transformado, então não vira infinito', () => {
    // Given — log(0) = -Infinity; o gerador usa 1 - random() para evitar
    const gaussian = createGaussian(() => 0);

    // Then
    assert.ok(Number.isFinite(gaussian()));
  });
});

describe('clamp', () => {
  it('dado um valor dentro da faixa, quando limitado, então não muda', () => {
    assert.equal(clamp(0.5, 0, 1), 0.5);
  });

  it('dado um valor abaixo do mínimo, quando limitado, então vira o mínimo', () => {
    assert.equal(clamp(-3, 0, 1), 0);
  });

  it('dado um valor acima do máximo, quando limitado, então vira o máximo', () => {
    assert.equal(clamp(7, 0, 1), 1);
  });
});

describe('quantile', () => {
  it('dada uma lista, quando o quantil 0 é pedido, então devolve o menor valor', () => {
    assert.equal(quantile([5, 1, 9, 3], 0), 1);
  });

  it('dada uma lista, quando o quantil 1 é pedido, então devolve o maior valor', () => {
    assert.equal(quantile([5, 1, 9, 3], 1), 9);
  });

  it('dada uma lista fora de ordem, quando o quantil é calculado, então a ordem não importa', () => {
    // Given — a função ordena por conta própria
    const embaralhada = [9, 3, 1, 5];
    const ordenada = [1, 3, 5, 9];

    // Then
    assert.equal(quantile(embaralhada, 0.5), quantile(ordenada, 0.5));
  });

  it('dada uma lista, quando o quantil é calculado, então a lista original não muda', () => {
    // Given
    const valores = [5, 1, 9, 3];

    // When
    quantile(valores, 0.5);

    // Then
    assert.deepEqual(valores, [5, 1, 9, 3], 'sort() sem cópia mutaria a entrada');
  });

  it('dado um corte no quantil 0.85, quando os valores acima são contados, então são ~15%', () => {
    // Given — é assim que o desbalanceamento é produzido
    const valores = Array.from({ length: 1000 }, (naoUsado, i) => i);

    // When
    const corte = quantile(valores, 0.85);
    const acima = valores.filter((valor) => valor > corte).length;

    // Then
    assert.ok(Math.abs(acima / 1000 - 0.15) < 0.01, `taxa ${acima / 1000}`);
  });
});

describe('riskScore', () => {
  it('dado um cliente, quando a dívida sobe, então o escore de risco sobe', () => {
    // Given
    const base = {
      income: 8000, debtRatio: 0.2, latePayments: 1, creditUtilization: 0.3,
    };

    // When
    const endividado = riskScore({ ...base, debtRatio: 0.9 });

    // Then
    assert.ok(endividado > riskScore(base));
  });

  it('dado um cliente, quando a renda sobe, então o escore de risco cai', () => {
    // Given
    const base = {
      income: 3000, debtRatio: 0.5, latePayments: 2, creditUtilization: 0.5,
    };

    // When
    const rico = riskScore({ ...base, income: 14000 });

    // Then
    assert.ok(rico < riskScore(base), 'renda é o único coeficiente negativo');
  });

  it('dados clientes idênticos, quando pontuados, então o escore é determinístico', () => {
    // Given
    const customer = {
      income: 5000, debtRatio: 0.4, latePayments: 3, creditUtilization: 0.6,
    };

    // Then
    assert.equal(riskScore(customer), riskScore({ ...customer }));
  });
});

describe('measureCustomer', () => {
  const verdadeiro = {
    income: 8000, debtRatio: 0.5, latePayments: 2, creditUtilization: 0.5,
  };

  it('dado ruído zero, quando o cliente é medido, então a medida é a verdade', () => {
    // Given / When — o gerador limpo precisa continuar disponível
    const medido = measureCustomer(verdadeiro, 0, createGaussian(createRandom(1)));

    // Then
    assert.deepEqual(medido, verdadeiro);
  });

  it('dado ruído, quando o cliente é medido, então a medida difere da verdade', () => {
    // Given / When
    const medido = measureCustomer(verdadeiro, 0.05, createGaussian(createRandom(1)));

    // Then
    assert.notEqual(medido.income, verdadeiro.income);
    assert.notEqual(medido.debtRatio, verdadeiro.debtRatio);
  });

  it('dado ruído enorme, quando o cliente é medido, então nada sai dos limites válidos', () => {
    // Given — sem clamp apareceria renda negativa e utilização de 300%
    const gaussian = createGaussian(createRandom(3));

    // When
    const medidos = Array.from({ length: 300 }, () => measureCustomer(verdadeiro, 5, gaussian));

    // Then
    Object.entries(SYNTHETIC_BOUNDS).forEach(([coluna, [minimo, maximo]]) => {
      const fora = medidos.filter((medido) => medido[coluna] < minimo || medido[coluna] > maximo);

      assert.deepEqual(fora, [], `${coluna} saiu de [${minimo}, ${maximo}]`);
    });
  });

  it('dado ruído, quando os atrasos são medidos, então continuam inteiros', () => {
    // Given — "2,3 atrasos" não existe no sistema do banco
    const gaussian = createGaussian(createRandom(4));

    // When
    const medidos = Array.from({ length: 100 }, () => measureCustomer(verdadeiro, 0.2, gaussian));

    // Then
    assert.ok(medidos.every((medido) => Number.isInteger(medido.latePayments)));
  });

  it('dado um cliente medido, quando as colunas são listadas, então são as mesmas do CSV', () => {
    // Given / When
    const medido = measureCustomer(verdadeiro, 0.05, createGaussian(createRandom(5)));

    // Then — nada de campo extra vazando para o arquivo
    assert.deepEqual(
      Object.keys(medido).sort(),
      CSV_COLUMNS.filter((coluna) => coluna !== CSV_LABEL_COLUMN).sort(),
    );
  });
});

describe('desbalanceamento', () => {
  it('dada a taxa padrão, quando os clientes são gerados, então a minoria fica perto de 15%', () => {
    // Given / When
    const taxa = taxaPositivos(createCustomers(1200));

    // Then
    assert.ok(Math.abs(taxa - SYNTHETIC_POSITIVE_RATE) < 0.03, `taxa ${taxa}`);
  });

  it('dada uma taxa pedida, quando os clientes são gerados, então ela é respeitada', () => {
    // Given — o limiar é um quantil, então a taxa é um parâmetro de verdade
    const limpo = { featureNoise: 0, labelNoise: 0 };

    // When / Then
    [0.05, 0.25, 0.5].forEach((positiveRate) => {
      const taxa = taxaPositivos(createCustomers(1000, { ...limpo, positiveRate }));

      assert.ok(Math.abs(taxa - positiveRate) < 0.02, `pedi ${positiveRate}, veio ${taxa}`);
    });
  });

  it('dado o dataset desbalanceado, quando o baseline é medido, então passa de 0.8', () => {
    // Given — é o número que torna a acurácia sozinha inútil
    const { labels } = createDataset(1200);

    // When
    const baseline = majorityBaseline(labels);

    // Then
    assert.ok(baseline > 0.8, `baseline ${baseline}`);
  });
});

describe('ruído de rótulo', () => {
  const limpo = { featureNoise: 0, labelNoise: 0 };

  it('dado ruído de rótulo zero, quando os rótulos saem, então seguem a regra', () => {
    // Given / When
    const customers = createCustomers(500, limpo);
    const escores = customers.map(riskScore);
    const corte = quantile(escores, 1 - SYNTHETIC_POSITIVE_RATE);

    // Then — sem ruído, a regra reproduz o rótulo exatamente
    const divergentes = customers
      .filter((customer, i) => (escores[i] > corte ? 1 : 0) !== customer.risk);

    assert.deepEqual(divergentes, []);
  });

  it('dado ruído de rótulo, quando comparado ao limpo, então alguns rótulos divergem', () => {
    // Given — mesma semente, mesmos clientes; só o rótulo muda
    const semRuido = createCustomers(1000, { featureNoise: 0, labelNoise: 0 });
    const comRuido = createCustomers(1000, { featureNoise: 0, labelNoise: 0.05 });

    // When
    const trocados = semRuido.filter((customer, i) => customer.risk !== comRuido[i].risk).length;

    // Then — ~5% de 1000, com folga para a variação do sorteio
    assert.ok(trocados > 20 && trocados < 90, `trocados ${trocados}`);
  });

  it('dado ruído de rótulo total, quando os rótulos saem, então todos estão invertidos', () => {
    // Given — labelNoise 1 troca sempre: prova que a troca é o que diz ser
    const original = createCustomers(100, { featureNoise: 0, labelNoise: 0 });
    const invertido = createCustomers(100, { featureNoise: 0, labelNoise: 1 });

    // Then
    assert.ok(original.every((customer, i) => customer.risk === 1 - invertido[i].risk));
  });
});

describe('ruído de medição cria um teto', () => {
  it('dado um dataset ruidoso, quando a própria regra é aplicada à medida, então ela erra', () => {
    // Given — o rótulo vem do cliente VERDADEIRO, e o arquivo guarda a
    // MEDIDA. Nem a fórmula que gerou os rótulos os recupera a partir do
    // que o modelo vê: é um teto que nenhuma arquitetura ultrapassa.
    const customers = createCustomers(1200, { labelNoise: 0 });
    const escores = customers.map(riskScore);
    const corte = quantile(escores, 1 - SYNTHETIC_POSITIVE_RATE);

    // When
    const acertos = customers
      .filter((customer, i) => (escores[i] > corte ? 1 : 0) === customer.risk).length;
    const teto = acertos / customers.length;

    // Then
    assert.ok(teto < 1, `com ruído o teto precisa ficar abaixo de 100%, veio ${teto}`);
    assert.ok(teto > 0.9, `mas ainda bem acima do acaso, veio ${teto}`);
  });

  it('dado um dataset limpo, quando a mesma conta é feita, então o teto é 100%', () => {
    // Given — o contraste que mostra de onde vem o teto acima
    const customers = createCustomers(1200, { featureNoise: 0, labelNoise: 0 });
    const escores = customers.map(riskScore);
    const corte = quantile(escores, 1 - SYNTHETIC_POSITIVE_RATE);

    // When
    const acertos = customers
      .filter((customer, i) => (escores[i] > corte ? 1 : 0) === customer.risk).length;

    // Then
    assert.equal(acertos, customers.length);
  });
});

describe('data/customers.csv versionado', () => {
  it('dado o CSV do repositório, quando regerado pelo código, então sai idêntico', () => {
    // Given — o gerador tem semente, então o arquivo pode ser reconstruído
    // e conferido. Se este teste falhar, ou o gerador mudou sem
    // `npm run seed`, ou alguém editou o CSV à mão.
    const versionado = fs.readFileSync(CSV_PATH, 'utf8');

    // When
    const regerado = `${toCsv(createCustomers())}\n`;

    // Then
    assert.equal(regerado, versionado, 'rode `npm run seed` para atualizar o CSV');
  });

  it('dado o CSV do repositório, quando as classes são contadas, então a minoria é ~15%', () => {
    // Given / When
    const taxa = taxaPositivos(createCustomers());

    // Then
    assert.ok(taxa > 0.1 && taxa < 0.2, `taxa ${taxa}`);
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
    assert.equal(features.length, SYNTHETIC_TOTAL);
    assert.equal(SYNTHETIC_TOTAL, 1200);
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
    // Given — mesmo desbalanceado, 15% de 1200 são ~180 positivos
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

  it('dada a mesma semente, quando os clientes são gerados duas vezes, então saem idênticos', () => {
    // Given — reprodutibilidade é o que torna o CSV versionado auditável
    const primeira = createCustomers(20, { seed: 123 });

    // When
    const segunda = createCustomers(20, { seed: 123 });

    // Then
    assert.deepEqual(segunda, primeira);
  });

  it('dadas sementes diferentes, quando os clientes são gerados, então os dados mudam', () => {
    // Given / When
    const sete = createCustomers(20, { seed: 7 });
    const oito = createCustomers(20, { seed: 8 });

    // Then
    assert.notDeepEqual(oito, sete, 'a semente precisa ter efeito');
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
  it('dado o tamanho de entrada da fonte real, quando o modelo é criado, então aceita 57 features', () => {
    // Given — 7 numéricas + 50 níveis one-hot
    const model = buildModel(GERMAN_SOURCE.featureNames.length);

    // When / Then
    assert.deepEqual(model.inputs[0].shape, [null, 57]);
    assert.equal(model.countParams(), 1073);
    model.dispose();
  });

  it('dado nenhum tamanho, quando o modelo é criado, então mantém as 4 entradas do sintético', () => {
    // Given — compatibilidade com as chamadas anteriores
    const model = buildModel();

    // When / Then
    assert.deepEqual(model.inputs[0].shape, [null, 4]);
    model.dispose();
  });

  const model = buildModel();

  after(() => model.dispose());

  it('dado o modelo construído, quando as camadas são contadas, então há 3 densas e 2 de dropout', () => {
    // Given — dropout entra depois de cada camada OCULTA, nunca da saída
    const densas = model.layers.filter((layer) => layer.getClassName() === 'Dense');
    const dropouts = model.layers.filter((layer) => layer.getClassName() === 'Dropout');

    // When / Then
    assert.equal(densas.length, 3);
    assert.equal(dropouts.length, 2);
    assert.equal(model.layers.length, 5);
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
    const ativacoes = model.layers
      .filter((layer) => layer.getClassName() === 'Dense')
      .map((layer) => layer.getConfig().activation);

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
// Regularização: L2 e dropout
// ==================================================
describe('createRegularizer', () => {
  it('dado um lambda positivo, quando o regularizador é criado, então devolve um L2 com esse lambda', () => {
    // Given / When
    const regularizer = createRegularizer(0.01);

    // Then
    assert.notEqual(regularizer, null);
    // O tfjs implementa L1 e L2 na mesma classe: `regularizers.l2()` é um
    // L1L2 com l1 = 0. O que importa é onde o lambda foi parar.
    assert.equal(regularizer.getClassName(), 'L1L2');
    assert.equal(regularizer.getConfig().l1, 0);
    assert.ok(
      Math.abs(regularizer.getConfig().l2 - 0.01) < 1e-9,
      `lambda deveria chegar intacto, recebido ${regularizer.getConfig().l2}`,
    );
  });

  it('dado lambda zero, quando o regularizador é criado, então devolve null em vez de uma penalidade de peso zero', () => {
    // Given — `null` é o que o tfjs entende como "sem regularizador";
    // um L2 com lambda 0 seria matematicamente inerte, mas apareceria
    // no model.json sugerindo um freio que não existe.
    // When / Then
    assert.equal(createRegularizer(0), null);
  });

  it('dado nenhum argumento, quando o regularizador é criado, então usa a constante do laboratório', () => {
    // Given / When
    const regularizer = createRegularizer();

    // Then
    assert.equal(regularizer.getConfig().l2, L2_LAMBDA);
  });
});

describe('buildModel — regularização', () => {
  it('dados os padrões, quando o modelo é criado, então toda camada densa carrega o regularizador L2', () => {
    // Given
    const model = buildModel();

    // When
    const densas = model.layers.filter((layer) => layer.getClassName() === 'Dense');

    // Then
    assert.equal(densas.length, 3);
    densas.forEach((layer) => {
      assert.notEqual(layer.kernelRegularizer, null, `${layer.name} ficou sem L2`);
      assert.equal(layer.kernelRegularizer.getConfig().l2, L2_LAMBDA);
    });

    model.dispose();
  });

  it('dado l2 igual a zero, quando o modelo é criado, então nenhuma camada tem regularizador', () => {
    // Given
    const model = buildModel(4, { l2: 0 });

    // When
    const comPenalidade = model.layers.filter((layer) => layer.kernelRegularizer);

    // Then
    assert.deepEqual(comPenalidade, []);
    model.dispose();
  });

  it('dado dropout igual a zero, quando o modelo é criado, então a topologia volta a ser a de três camadas', () => {
    // Given — é a rede de antes deste item, byte a byte na topologia
    const model = buildModel(4, { l2: 0, dropout: 0 });

    // When / Then
    assert.equal(model.layers.length, 3);
    assert.deepEqual(
      model.layers.map((layer) => layer.getClassName()),
      ['Dense', 'Dense', 'Dense'],
    );

    model.dispose();
  });

  it('dada uma taxa de dropout, quando o modelo é criado, então ela chega nas duas camadas', () => {
    // Given
    const model = buildModel(4, { dropout: 0.35 });

    // When
    const taxas = model.layers
      .filter((layer) => layer.getClassName() === 'Dropout')
      .map((layer) => layer.getConfig().rate);

    // Then
    assert.deepEqual(taxas, [0.35, 0.35]);
    model.dispose();
  });

  it('dado o modelo com dropout, quando a última camada é lida, então é a densa de saída — a saída não leva dropout', () => {
    // Given
    const model = buildModel();

    // When
    const ultima = model.layers.at(-1);

    // Then — descartar a única unidade de saída apagaria a predição
    assert.equal(ultima.getClassName(), 'Dense');
    assert.equal(ultima.getConfig().units, 1);
    model.dispose();
  });

  it('dado dropout ligado e desligado, quando os parâmetros são contados, então o total não muda', () => {
    // Given — dropout não tem peso nenhum: ele apenas zera ativações
    const comDropout = buildModel(57);
    const semDropout = buildModel(57, { dropout: 0 });

    // When / Then
    assert.equal(comDropout.countParams(), semDropout.countParams());
    assert.equal(comDropout.countParams(), 1073);

    comDropout.dispose();
    semDropout.dispose();
  });
});

describe('resolveRegularization', () => {
  it('dado nenhum argumento, quando as flags são resolvidas, então devolve vazio — quem decide é a fonte', () => {
    // Given — o objeto vazio é o que faz `{ ...source.regularization }`
    // sobreviver intacto ao espalhamento no `main`
    const resolvido = resolveRegularization([]);

    // When / Then
    assert.deepEqual(resolvido, {});
  });

  it('dadas as duas flags zeradas, quando são resolvidas, então desligam os dois freios', () => {
    // Given / When
    const resolvido = resolveRegularization(['--l2=0', '--dropout=0']);

    // Then
    assert.deepEqual(resolvido, { l2: 0, dropout: 0 });
  });

  it('dada só uma das flags, quando é resolvida, então a outra não aparece e a fonte mantém a dela', () => {
    // Given / When
    const resolvido = resolveRegularization(['--source=synthetic', '--l2=0.05']);

    // Then
    assert.deepEqual(resolvido, { l2: 0.05 });
    assert.ok(!('dropout' in resolvido), 'dropout não pedido não deve virar chave');
  });

  it('dado um valor não numérico, quando é resolvido, então explica o que é aceito', () => {
    // Given / When / Then
    assert.throws(
      () => resolveRegularization(['--l2=muito']),
      /Valor inválido para --l2: muito.*entre 0 e 1/s,
    );
  });

  it('dado um valor negativo, quando é resolvido, então falha — penalidade negativa premiaria peso grande', () => {
    // Given / When / Then
    assert.throws(() => resolveRegularization(['--l2=-1']), /Valor inválido/);
  });

  it('dado dropout acima do teto, quando é resolvido, então falha — taxa 1 zeraria a camada inteira', () => {
    // Given / When / Then
    assert.throws(
      () => resolveRegularization(['--dropout=1']),
      /Valor inválido para --dropout: 1.*entre 0 e 0\.9/s,
    );
  });
});

describe('regularização por fonte', () => {
  it('dada a fonte sintética, quando a regularização é lida, então os dois freios estão desligados', () => {
    // Given — 225 parâmetros para 768 linhas: não há capacidade sobrando,
    // e a medição mostra que frear aqui só cobra
    assert.deepEqual(SYNTHETIC_SOURCE.regularization, { l2: 0, dropout: 0 });
  });

  it('dadas as fontes do German Credit, quando a regularização é lida, então usam as constantes do laboratório', () => {
    // Given — é o dataset com 1.073 (ou 465) parâmetros para 640 linhas
    [GERMAN_SOURCE, GERMAN_ORDINAL_SOURCE].forEach((source) => {
      assert.deepEqual(
        source.regularization,
        { l2: L2_LAMBDA, dropout: DROPOUT_RATE },
        `fonte ${source.id}`,
      );
    });
  });

  it('dadas as duas variantes do German, quando comparadas, então diferem apenas na codificação', () => {
    // Given — a regularização não pode virar uma segunda diferença
    assert.notEqual(GERMAN_SOURCE.encoding, GERMAN_ORDINAL_SOURCE.encoding);
    assert.deepEqual(
      GERMAN_SOURCE.regularization,
      GERMAN_ORDINAL_SOURCE.regularization,
    );
  });

  it('dada a fonte e uma flag, quando são mescladas, então a linha de comando vence', () => {
    // Given — é a mesma mesclagem que o `main` faz
    const daFonte = SYNTHETIC_SOURCE.regularization;
    const daLinha = resolveRegularization(['--dropout=0.4']);

    // When
    const efetivo = { ...daFonte, ...daLinha };

    // Then
    assert.deepEqual(efetivo, { l2: 0, dropout: 0.4 });
  });

  it('dada a fonte sem flag nenhuma, quando são mescladas, então nada da fonte é sobrescrito', () => {
    // Given / When
    const efetivo = { ...GERMAN_SOURCE.regularization, ...resolveRegularization([]) };

    // Then
    assert.deepEqual(efetivo, { l2: L2_LAMBDA, dropout: DROPOUT_RATE });
  });
});

describe('parseNumericFlag', () => {
  it('dado que a flag não aparece, quando é lida, então devolve o padrão', () => {
    // Given / When / Then
    assert.equal(parseNumericFlag(['--outra=3'], 'l2', 0.007, 1), 0.007);
  });

  it('dada a flag presente, quando é lida, então o texto vira número', () => {
    // Given / When
    const valor = parseNumericFlag(['--l2=0.25'], 'l2', 0.007, 1);

    // Then
    assert.equal(valor, 0.25);
    assert.equal(typeof valor, 'number');
  });

  it('dado o valor exatamente no teto, quando é lido, então é aceito', () => {
    // Given — o limite é inclusivo dos dois lados
    assert.equal(parseNumericFlag(['--r=0.9'], 'r', 0.1, 0.9), 0.9);
    assert.equal(parseNumericFlag(['--r=0'], 'r', 0.1, 0.9), 0);
  });

  it('dado um valor vazio, quando é lido, então falha em vez de virar zero silenciosamente', () => {
    // Given — `Number('')` é 0, e aceitar isso desligaria o freio sem avisar
    assert.throws(() => parseNumericFlag(['--l2='], 'l2', 0.007, 1), /Valor inválido/);
  });
});

describe('dropout e L2 no comportamento do modelo', () => {
  const tf = require('@tensorflow/tfjs-node');

  it('dado um modelo com dropout, quando prevê duas vezes, então devolve o mesmo valor — inferência desliga o dropout', () => {
    // Given — taxa altíssima: se o dropout agisse aqui, os dois valores
    // seriam diferentes com probabilidade praticamente 1
    const model = buildModel(4, { l2: 0, dropout: 0.9 });
    const input = tf.tensor2d([[0.5, 0.5, 0.5, 0.5]]);

    // When
    const primeira = model.predict(input);
    const segunda = model.predict(input);
    const [a, b] = [primeira.dataSync()[0], segunda.dataSync()[0]];

    // Then
    assert.equal(a, b);

    tf.dispose([input, primeira, segunda]);
    model.dispose();
  });

  it('dados os mesmos pesos com e sem L2, quando são avaliados, então a loss é idêntica — evaluate não cobra a penalidade', () => {
    // Given — no tfjs a penalidade entra na loss do TREINO, não na de
    // avaliação. É por isso que o `Test loss` que o main imprime continua
    // comparável com o de antes deste item.
    const semPenalidade = buildModel(4, { l2: 0, dropout: 0 });
    const comPenalidade = buildModel(4, { l2: 10, dropout: 0 });
    comPenalidade.setWeights(semPenalidade.getWeights());

    const x = tf.tensor2d([[0.2, 0.4, 0.6, 0.8], [0.9, 0.1, 0.3, 0.7]]);
    const y = tf.tensor2d([[1], [0]]);

    // When
    const { loss: lossSem } = evaluateModel(semPenalidade, x, y);
    const { loss: lossCom } = evaluateModel(comPenalidade, x, y);

    // Then
    assert.ok(
      Math.abs(lossSem - lossCom) < 1e-6,
      `evaluate não deveria somar λ·Σw², recebido ${lossSem} e ${lossCom}`,
    );

    tf.dispose([x, y]);
    semPenalidade.dispose();
    comPenalidade.dispose();
  });

  it('dado um modelo regularizado salvo em disco, quando é recarregado, então prevê exatamente o mesmo', () => {
    // Given — dropout e L2 precisam sobreviver à serialização; o dropout
    // some da conta na inferência, mas não pode sumir da topologia.
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'reg-'));
    const model = buildModel(4);
    const customer = {
      income: 3500,
      debtRatio: 0.72,
      latePayments: 3,
      creditUtilization: 0.88,
    };

    return saveModel(model, dir)
      .then(() => loadModel(dir))
      .then((recarregado) => {
        // When
        const antes = predictRisk(model, customer);
        const depois = predictRisk(recarregado, customer);
        const densas = recarregado.layers
          .filter((layer) => layer.getClassName() === 'Dense');

        // Then
        assert.equal(antes, depois);
        assert.equal(recarregado.layers.length, 5);
        densas.forEach((layer) => {
          assert.equal(layer.kernelRegularizer.getConfig().l2, L2_LAMBDA);
        });

        model.dispose();
        recarregado.dispose();
        fs.rmSync(dir, { recursive: true, force: true });
      });
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

// ==================================================
// Dataset real: German Credit (UCI / Statlog)
// ==================================================
describe('parseDelimited', () => {
  it('dado um texto com cabeçalho, quando parseado, então cada linha vira um objeto', () => {
    // Given
    const texto = 'a,b\n1,2\n3,4';

    // When
    const linhas = parseDelimited(texto);

    // Then
    assert.deepEqual(linhas, [{ a: '1', b: '2' }, { a: '3', b: '4' }]);
  });

  it('dado um arquivo com CRLF, quando parseado, então o \\r não contamina o último campo', () => {
    // Given — arquivos da UCI já chegaram com quebra de linha do Windows
    const texto = 'a,b\r\n1,2\r\n';

    // When
    const [linha] = parseDelimited(texto);

    // Then
    assert.equal(linha.b, '2');
  });

  it('dado um texto com quebra de linha no fim, quando parseado, então não gera linha vazia', () => {
    // Given / When
    const linhas = parseDelimited('a\n1\n2\n');

    // Then
    assert.equal(linhas.length, 2);
  });
});

describe('toOrdinal', () => {
  it('dado um código conhecido, quando convertido, então vira a posição na lista', () => {
    // Given / When / Then
    assert.equal(toOrdinal(GERMAN_CATEGORICAL.checkingStatus, 'A11'), 0);
    assert.equal(toOrdinal(GERMAN_CATEGORICAL.checkingStatus, 'A14'), 3);
  });

  it('dado A410 e A41, quando convertidos, então não são confundidos um com o outro', () => {
    // Given — 'A41' é prefixo de 'A410'; indexOf compara igualdade, não prefixo
    assert.equal(toOrdinal(GERMAN_CATEGORICAL.purpose, 'A41'), 1);
    assert.equal(toOrdinal(GERMAN_CATEGORICAL.purpose, 'A410'), 9);
  });

  it('dado um código desconhecido, quando convertido, então lança em vez de virar 0', () => {
    // Given — silenciar isso criaria uma feature plausível e errada
    assert.throws(
      () => toOrdinal(GERMAN_CATEGORICAL.checkingStatus, 'A99'),
      /Código desconhecido/,
    );
  });
});

describe('toGermanCustomer', () => {
  // Primeira linha real do arquivo da UCI, com os 20 atributos.
  const linha = {
    Attribute1: 'A11', Attribute2: '6', Attribute3: 'A34', Attribute4: 'A43',
    Attribute5: '1169', Attribute6: 'A65', Attribute7: 'A75', Attribute8: '4',
    Attribute9: 'A93', Attribute10: 'A101', Attribute11: '4', Attribute12: 'A121',
    Attribute13: '67', Attribute14: 'A143', Attribute15: 'A152', Attribute16: '2',
    Attribute17: 'A173', Attribute18: '1', Attribute19: 'A192', Attribute20: 'A201',
    class: '1',
  };

  it('dada uma linha do arquivo, quando convertida, então os numéricos viram Number', () => {
    // Given / When
    const cliente = toGermanCustomer(linha);

    // Then
    assert.equal(cliente.durationMonths, 6);
    assert.equal(cliente.creditAmount, 1169);
    assert.equal(cliente.age, 67);
    assert.equal(cliente.existingCredits, 2);
    assert.equal(cliente.dependents, 1);
  });

  it('dada uma linha do arquivo, quando convertida, então os códigos viram índices', () => {
    // Given / When
    const cliente = toGermanCustomer(linha);

    // Then
    assert.equal(cliente.checkingStatus, 0);   // A11
    assert.equal(cliente.creditHistory, 4);    // A34
    assert.equal(cliente.purpose, 3);          // A43
    assert.equal(cliente.housing, 1);          // A152
    assert.equal(cliente.foreignWorker, 0);    // A201
  });

  it('dada a classe 2 do arquivo, quando convertida, então risk é 1 (mau pagador)', () => {
    // Given — a UCI usa 1 = bom, 2 = mau; aqui risk = 1 é o ALTO RISCO
    assert.equal(toGermanCustomer({ ...linha, class: '2' }).risk, 1);
    assert.equal(toGermanCustomer({ ...linha, class: '1' }).risk, 0);
  });

  it('dada uma linha convertida, quando as colunas são lidas, então cobrem GERMAN_COLUMNS', () => {
    // Given / When
    const cliente = toGermanCustomer(linha);

    // Then — nenhuma coluna declarada pode faltar
    const faltando = GERMAN_COLUMNS.filter((c) => !Number.isFinite(cliente[c]));
    assert.deepEqual(faltando, []);
  });

  it('dada uma linha convertida, quando o atributo 9 é lido, então vira a coluna de auditoria', () => {
    // Given — 'A93' é o terceiro código da lista
    assert.equal(toGermanCustomer(linha)[GERMAN_AUDIT_COLUMN], 2);
  });
});

describe('parseGermanCsv', () => {
  const cabecalho = Array.from({ length: 20 }, (u, i) => `Attribute${i + 1}`)
    .concat('class').join(',');
  const linha = (classe) =>
    `A11,6,A34,A43,1169,A65,A75,4,A93,A101,4,A121,67,A143,A152,2,A173,1,A192,A201,${classe}`;

  it('dado o texto bruto da UCI, quando parseado, então devolve clientes prontos', () => {
    // Given / When
    const clientes = parseGermanCsv([cabecalho, linha(1), linha(2)].join('\n'));

    // Then
    assert.equal(clientes.length, 2);
    assert.equal(clientes[0].risk, 0);
    assert.equal(clientes[1].risk, 1);
    assert.equal(clientes[0].durationMonths, 6);
  });
});

describe('data/german-credit.csv', () => {
  it('dado o arquivo versionado, quando lido, então tem as 1000 solicitações da UCI', async () => {
    // Given / When
    const clientes = await readCustomersCsv(GERMAN_CSV_PATH);

    // Then
    assert.equal(clientes.length, 1000);
  });

  it('dado o arquivo versionado, quando os rótulos são contados, então 300 são maus pagadores', async () => {
    // Given — proporção documentada pela UCI; se mudar, a conversão quebrou
    const clientes = await readCustomersCsv(GERMAN_CSV_PATH);

    // When
    const maus = clientes.filter(({ risk }) => risk === 1).length;

    // Then
    assert.equal(maus, 300);
  });

  it('dado o arquivo versionado, quando as colunas são lidas, então batem com GERMAN_COLUMNS', async () => {
    // Given / When
    const [cliente] = await readCustomersCsv(GERMAN_CSV_PATH);

    // Then
    assert.deepEqual(Object.keys(cliente).sort(), [...GERMAN_COLUMNS].sort());
  });

  it('dado o arquivo versionado, quando os valores são lidos, então são todos finitos', async () => {
    // Given — um NaN aqui viraria loss NaN lá na frente, sem erro nenhum
    const clientes = await readCustomersCsv(GERMAN_CSV_PATH);

    // When
    const invalidos = clientes.filter((cliente) =>
      Object.values(cliente).some((valor) => !Number.isFinite(valor)));

    // Then
    assert.deepEqual(invalidos, []);
  });
});

// ==================================================
// Normalização ajustada no treino
// ==================================================
describe('fitMinMaxScaler / applyMinMaxScaler', () => {
  const clientes = [
    { a: 10, b: 5 },
    { a: 20, b: 5 },
    { a: 30, b: 5 },
  ];

  it('dados clientes, quando o scaler é ajustado, então guarda mínimo e amplitude', () => {
    // Given / When
    const scaler = fitMinMaxScaler(clientes, ['a']);

    // Then
    assert.equal(scaler.min.a, 10);
    assert.equal(scaler.range.a, 20);
  });

  it('dado o scaler ajustado, quando aplicado, então mínimo vira 0 e máximo vira 1', () => {
    // Given
    const scaler = fitMinMaxScaler(clientes, ['a']);

    // When / Then
    assert.deepEqual(applyMinMaxScaler(scaler, { a: 10 }), [0]);
    assert.deepEqual(applyMinMaxScaler(scaler, { a: 30 }), [1]);
    assert.deepEqual(applyMinMaxScaler(scaler, { a: 20 }), [0.5]);
  });

  it('dada uma coluna constante, quando ajustada, então a amplitude vira 1 e não divide por zero', () => {
    // Given — sem isso o resultado seria NaN e contaminaria o treino todo
    const scaler = fitMinMaxScaler(clientes, ['b']);

    // When / Then
    assert.equal(scaler.range.b, 1);
    assert.deepEqual(applyMinMaxScaler(scaler, { b: 5 }), [0]);
  });

  it('dado um valor fora da faixa de treino, quando aplicado, então sai de [0, 1] em vez de ser cortado', () => {
    // Given — cortar esconderia justamente o caso extremo nunca visto
    const scaler = fitMinMaxScaler(clientes, ['a']);

    // When
    const [abaixo] = applyMinMaxScaler(scaler, { a: 0 });
    const [acima] = applyMinMaxScaler(scaler, { a: 50 });

    // Then
    assert.equal(abaixo, -0.5);
    assert.equal(acima, 2);
  });

  it('dado um scaler ajustado só no treino, quando o teste tem extremos, então a escala não muda', () => {
    // Given — este é o teste de VAZAMENTO: o teste não pode influenciar
    // a escala, senão o modelo parece melhor do que é
    const treino = [{ a: 10 }, { a: 20 }];
    const teste = [{ a: 1000 }];

    // When
    const soTreino = fitMinMaxScaler(treino, ['a']);
    const comTeste = fitMinMaxScaler([...treino, ...teste], ['a']);

    // Then
    assert.equal(soTreino.range.a, 10);
    assert.notEqual(comTeste.range.a, soTreino.range.a);
  });

  it('dadas várias features, quando aplicadas, então a ordem do vetor segue featureNames', () => {
    // Given — a ordem precisa casar com a ordem das entradas da rede
    const scaler = fitMinMaxScaler([{ a: 0, b: 0 }, { a: 10, b: 100 }], ['b', 'a']);

    // When / Then
    assert.deepEqual(applyMinMaxScaler(scaler, { a: 10, b: 50 }), [0.5, 1]);
  });
});

// ==================================================
// Embaralhamento reproduzível e separação
// ==================================================
describe('createRandom / shuffle', () => {
  it('dada a mesma semente, quando dois geradores rodam, então produzem a mesma sequência', () => {
    // Given / When
    const a = createRandom(42);
    const b = createRandom(42);

    // Then
    assert.deepEqual([a(), a(), a()], [b(), b(), b()]);
  });

  it('dadas sementes diferentes, quando os geradores rodam, então divergem', () => {
    // Given / When / Then
    assert.notEqual(createRandom(1)(), createRandom(2)());
  });

  it('dado o gerador, quando amostrado, então os valores ficam em [0, 1)', () => {
    // Given
    const random = createRandom(7);

    // When
    const amostras = Array.from({ length: 500 }, random);

    // Then
    assert.ok(amostras.every((valor) => valor >= 0 && valor < 1));
  });

  it('dada uma lista, quando embaralhada, então é uma permutação dos mesmos itens', () => {
    // Given
    const itens = Array.from({ length: 50 }, (ignorado, i) => i);

    // When
    const embaralhado = shuffle(itens, createRandom(SHUFFLE_SEED));

    // Then
    assert.deepEqual([...embaralhado].sort((x, y) => x - y), itens);
  });

  it('dada uma lista, quando embaralhada, então a original não é modificada', () => {
    // Given
    const itens = [1, 2, 3, 4, 5];

    // When
    shuffle(itens, createRandom(1));

    // Then
    assert.deepEqual(itens, [1, 2, 3, 4, 5]);
  });

  it('dada a mesma semente, quando a lista é embaralhada duas vezes, então o resultado é idêntico', () => {
    // Given — é o que torna a execução reproduzível
    const itens = Array.from({ length: 30 }, (ignorado, i) => i);

    // When / Then
    assert.deepEqual(
      shuffle(itens, createRandom(SHUFFLE_SEED)),
      shuffle(itens, createRandom(SHUFFLE_SEED)),
    );
  });
});

describe('splitCustomers', () => {
  const clientes = Array.from({ length: 100 }, (ignorado, i) => ({ id: i, risk: 0 }));

  it('dados clientes, quando divididos em 80/20, então os tamanhos batem', () => {
    // Given / When
    const { trainCustomers, testCustomers } = splitCustomers(clientes);

    // Then
    assert.equal(trainCustomers.length, 80);
    assert.equal(testCustomers.length, 20);
  });

  it('dados clientes divididos, quando as partes são comparadas, então não há sobreposição', () => {
    // Given / When
    const { trainCustomers, testCustomers } = splitCustomers(clientes);
    const treino = new Set(trainCustomers.map(({ id }) => id));

    // Then
    assert.deepEqual(testCustomers.filter(({ id }) => treino.has(id)), []);
  });

  it('dada uma proporção customizada, quando dividido, então respeita a proporção', () => {
    // Given / When
    const { trainCustomers } = splitCustomers(clientes, 0.5);

    // Then
    assert.equal(trainCustomers.length, 50);
  });
});

describe('stratifiedSplitCustomers', () => {
  // 100 clientes, 30 inadimplentes — a proporção do German Credit.
  const clientes = Array.from({ length: 100 }, (ignorado, i) => ({
    id: i,
    risk: i % 10 < 3 ? 1 : 0,
  }));

  it('dados clientes desbalanceados, quando divididos, então as duas partes mantêm a proporção', () => {
    // Given / When
    const { trainCustomers, testCustomers } = stratifiedSplitCustomers(clientes);
    const taxa = (rows) => rows.filter(({ risk }) => risk === 1).length / rows.length;

    // Then — 30% dos dois lados, sem depender de sorte nenhuma
    assert.equal(taxa(trainCustomers), 0.3);
    assert.equal(taxa(testCustomers), 0.3);
  });

  it('dados clientes, quando divididos em 80/20, então os tamanhos batem e nada se perde', () => {
    // Given / When
    const { trainCustomers, testCustomers } = stratifiedSplitCustomers(clientes);

    // Then
    assert.equal(trainCustomers.length, 80);
    assert.equal(testCustomers.length, 20);
    assert.equal(trainCustomers.length + testCustomers.length, clientes.length);
  });

  it('dados clientes divididos, quando as partes são comparadas, então não há sobreposição', () => {
    // Given / When
    const { trainCustomers, testCustomers } = stratifiedSplitCustomers(clientes);
    const treino = new Set(trainCustomers.map(({ id }) => id));

    // Then
    assert.deepEqual(testCustomers.filter(({ id }) => treino.has(id)), []);
  });

  it('dada a ordem embaralhada, quando dividida, então ela é PRESERVADA dentro de cada parte', () => {
    // Given — `fit` reserva os últimos 20% do treino para validação antes
    // de embaralhar. Se a estratificação agrupasse por rótulo, essa fatia
    // sairia quase toda de uma classe só.
    const { trainCustomers, testCustomers } = stratifiedSplitCustomers(clientes);
    const crescente = (rows) => rows.every(({ id }, i) => i === 0 || id > rows[i - 1].id);

    // Then
    assert.ok(crescente(trainCustomers));
    assert.ok(crescente(testCustomers));
  });

  it('dado o mesmo conjunto, quando comparado com o corte cru, então a proporção do teste é mais fiel', () => {
    // Given — um conjunto em que o corte cru cai mal de propósito: os
    // inadimplentes estão concentrados no fim
    const concentrados = Array.from({ length: 100 }, (ignorado, i) => ({
      id: i,
      risk: i >= 70 ? 1 : 0,
    }));

    // When
    const cru = splitCustomers(concentrados).testCustomers;
    const estratificado = stratifiedSplitCustomers(concentrados).testCustomers;
    const taxa = (rows) => rows.filter(({ risk }) => risk === 1).length / rows.length;

    // Then — o corte cru entrega um teste 100% inadimplente; o
    // estratificado entrega os 30% que o conjunto tem
    assert.equal(taxa(cru), 1);
    assert.equal(taxa(estratificado), 0.3);
  });

  it('dada uma proporção customizada, quando dividido, então respeita a proporção', () => {
    // Given / When
    const { trainCustomers } = stratifiedSplitCustomers(clientes, 0.5);

    // Then
    assert.equal(trainCustomers.length, 50);
  });
});

describe('stratifiedFolds', () => {
  const clientes = Array.from({ length: 100 }, (ignorado, i) => ({
    risk: i % 10 < 3 ? 1 : 0,
  }));

  it('dados clientes, quando distribuídos em dobras, então cada dobra recebe a mesma proporção', () => {
    // Given / When
    const dobras = stratifiedFolds(clientes, 5);

    // Then — 20 clientes por dobra, 6 deles inadimplentes
    [0, 1, 2, 3, 4].forEach((dobra) => {
      const rows = clientes.filter((ignorado, i) => dobras[i] === dobra);

      assert.equal(rows.length, 20);
      assert.equal(rows.filter(({ risk }) => risk === 1).length, 6);
    });
  });

  it('dadas as dobras, quando somadas, então cada cliente aparece em exatamente uma', () => {
    // Given / When
    const dobras = stratifiedFolds(clientes, 5);

    // Then
    assert.equal(dobras.length, clientes.length);
    assert.ok(dobras.every((dobra) => Number.isInteger(dobra) && dobra >= 0 && dobra < 5));
  });

  it('dado um número de dobras que não divide a classe, quando distribuído, então a diferença é de no máximo uma linha', () => {
    // Given — 30 inadimplentes em 4 dobras: 8, 8, 7, 7
    const dobras = stratifiedFolds(clientes, 4);
    const positivos = [0, 1, 2, 3].map((dobra) =>
      clientes.filter((cliente, i) => dobras[i] === dobra && cliente.risk === 1).length);

    // Then
    assert.ok(Math.max(...positivos) - Math.min(...positivos) <= 1);
  });
});

describe('summarize', () => {
  it('dados valores, quando resumidos, então trazem média, erro padrão e amplitude', () => {
    // Given / When
    const resumo = summarize([2, 4, 6, 8]);

    // Then
    assert.equal(resumo.mean, 5);
    assert.equal(resumo.lowest, 2);
    assert.equal(resumo.highest, 8);
    // desvio amostral = 2.5820..., erro padrão = desvio / sqrt(4)
    assert.ok(Math.abs(resumo.standardError - 1.2910) < 0.0001);
  });

  it('dado um valor só, quando resumido, então o erro padrão é 0 e não NaN', () => {
    // Given — uma medição não tem do que discordar; a variância amostral
    // seria uma divisão por zero
    const resumo = summarize([0.75]);

    // Then
    assert.equal(resumo.mean, 0.75);
    assert.equal(resumo.standardError, 0);
  });

  it('dados valores idênticos, quando resumidos, então o erro padrão é exatamente 0', () => {
    // Given / When / Then
    assert.equal(summarize([3, 3, 3]).standardError, 0);
  });
});

describe('resolveFolds', () => {
  it('dado nenhum argumento, quando resolvido, então a validação cruzada não roda', () => {
    // Given / When / Then — `null` é "não pedida", diferente de "0 dobras"
    assert.equal(resolveFolds([]), null);
    assert.equal(resolveFolds(['--source=german']), null);
  });

  it('dado --cv sem valor, quando resolvido, então usa o padrão do projeto', () => {
    // Given / When / Then
    assert.equal(resolveFolds(['--cv']), CV_FOLDS);
  });

  it('dado --cv com valor, quando resolvido, então usa o número pedido', () => {
    // Given / When / Then
    assert.equal(resolveFolds(['--cv=10']), 10);
    assert.equal(resolveFolds(['--source=synthetic', '--cv=3']), 3);
  });

  it('dado um k impossível, quando resolvido, então recusa em vez de rodar', () => {
    // Given — uma dobra não é validação cruzada, e 2.5 dobras não existe
    assert.throws(() => resolveFolds(['--cv=1']), /inteiro entre 2 e 20/);
    assert.throws(() => resolveFolds(['--cv=2.5']), /inteiro entre 2 e 20/);
    assert.throws(() => resolveFolds(['--cv=999']), /Valor inválido/);
    assert.throws(() => resolveFolds(['--cv=abc']), /Valor inválido/);
  });
});

describe('rocFromScores', () => {
  it('dados scores e rótulos, quando a curva é traçada, então não precisa de modelo nenhum', () => {
    // Given — separação perfeita, sem tensor e sem rede
    const { auc, positives, negatives } = rocFromScores([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]);

    // Then
    assert.equal(auc, 1);
    assert.equal(positives, 2);
    assert.equal(negatives, 2);
  });

  it('dada uma ordenação invertida, quando a curva é traçada, então a AUC é 0', () => {
    // Given / When / Then
    assert.equal(rocFromScores([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]).auc, 0);
  });

  it('dada uma classe só, quando a curva é traçada, então não há divisão por zero', () => {
    // Given — sem negativos o FPR é indefinido
    const curva = rocFromScores([0.9, 0.1], [1, 1]);

    // Then
    assert.equal(curva.auc, 0);
    assert.equal(curva.negatives, 0);
  });

  it('dado um modelo, quando computeRocCurve roda, então delega para a mesma função', () => {
    // Given — a curva do modelo é a curva dos scores dele
    const tf = require('@tensorflow/tfjs-node');
    const model = buildModel(2, { l2: 0, dropout: 0 });
    const xTest = tf.tensor2d([[0, 0], [1, 1], [0, 1], [1, 0]]);
    const yTest = tf.tensor2d([[0], [1], [1], [0]]);

    const scores = tf.tidy(() =>
      Array.from(model.predict(xTest).reshape([-1]).dataSync()));

    // When / Then
    assert.equal(
      computeRocCurve(model, xTest, yTest).auc,
      rocFromScores(scores, [0, 1, 1, 0]).auc,
    );

    tf.dispose([xTest, yTest]);
    model.dispose();
  });
});

describe('crossValidate', () => {
  // Fonte mínima em memória: o contrato é o mesmo das três reais, mas
  // sem disco, sem escala e com dois números por cliente.
  const fonteEmMemoria = () => {
    const customers = Array.from({ length: 60 }, (ignorado, index) => ({
      a: index % 2,
      b: (index % 5) / 5,
      // 21 inadimplentes em 60 linhas: 3 dobras recebem 7 cada, e 5
      // recebem 20 clientes cada. Divisível de propósito — é o que
      // permite afirmar que o baseline sai IDÊNTICO em todas.
      risk: index % 20 < 7 ? 1 : 0,
    }));

    return {
      id: 'memoria',
      label: 'Fonte de teste',
      csvPath: '(memória)',
      featureNames: ['a', 'b'],
      regularization: { l2: 0, dropout: 0 },
      ensure: () => ({ created: false }),
      read: async () => customers,
      fitScaler: () => null,
      toVector: (customer) => [customer.a, customer.b],
    };
  };

  it('dadas k dobras, quando validadas, então cada cliente é testado exatamente uma vez', async () => {
    // Given / When
    const resultado = await crossValidate(fonteEmMemoria(), { folds: 3 });

    // Then
    assert.equal(resultado.folds.length, 3);
    assert.equal(
      resultado.folds.reduce((total, { testSize }) => total + testSize, 0),
      60,
    );
    assert.equal(resultado.outOfSample.scores.filter(Number.isFinite).length, 60);
  });

  it('dadas dobras estratificadas, quando validadas, então o baseline é o mesmo em todas', async () => {
    // Given — é exatamente isto que a estratificação compra: a régua
    // para de mudar de dobra para dobra
    const { folds } = await crossValidate(fonteEmMemoria(), { folds: 3 });

    // Then
    assert.equal(new Set(folds.map(({ baseline }) => baseline)).size, 1);
  });

  it('dado o resumo, quando lido, então traz média e erro padrão de cada métrica', async () => {
    // Given / When
    const { summary } = await crossValidate(fonteEmMemoria(), { folds: 2 });

    // Then
    ['baseline', 'accuracy', 'auc', 'cost'].forEach((metrica) => {
      assert.ok(Number.isFinite(summary[metrica].mean), metrica);
      assert.ok(Number.isFinite(summary[metrica].standardError), metrica);
    });
  });

  it('dada uma fonte sem atributo protegido, quando validada, então não há auditoria', async () => {
    // Given / When
    const resultado = await crossValidate(fonteEmMemoria(), { folds: 2 });

    // Then
    assert.equal(resultado.audit, null);
    assert.equal(resultado.mitigated, null);
  });

  it('dado o resultado, quando formatado, então a tabela traz uma linha por dobra, a média e o erro', async () => {
    // Given
    const resultado = await crossValidate(fonteEmMemoria(), { folds: 2 });

    // When
    const texto = formatCrossValidation(resultado);

    // Then
    assert.ok(texto.includes('Dobra'));
    assert.ok(texto.includes('Média'));
    assert.ok(texto.includes('Erro'));
    assert.ok(texto.includes('AUC sobre o dataset inteiro'));
  });
});

describe('buildModel — arquitetura', () => {
  const camadas = (model) => model.layers.map((l) => l.getClassName());

  it('dada a arquitetura padrão, quando montada, então é a do laboratório', () => {
    // Given — `HIDDEN_UNITS` é o default, não um número solto no meio do código
    const padrao = buildModel(57);
    const explicito = buildModel(57, { units: HIDDEN_UNITS });

    // Then
    assert.deepStrictEqual(camadas(padrao), camadas(explicito));
    assert.equal(padrao.countParams(), explicito.countParams());
    assert.deepStrictEqual(HIDDEN_UNITS, [16, 8]);

    padrao.dispose();
    explicito.dispose();
  });

  it('dadas unidades diferentes, quando montadas, então a topologia acompanha', () => {
    // Given / When / Then — uma densa por camada oculta, mais a saída,
    // e um dropout depois de cada oculta
    const casos = [
      [[4], 3],
      [[16, 8], 5],
      [[16, 16, 16], 7],
    ];

    casos.forEach(([units, esperado]) => {
      const model = buildModel(57, { units });

      assert.equal(model.layers.length, esperado, JSON.stringify(units));
      model.dispose();
    });
  });

  it('dada a lista vazia, quando montada, então sobra uma REGRESSÃO LOGÍSTICA', () => {
    // Given — sem camada oculta não há não-linearidade nenhuma no meio
    const model = buildModel(57, { units: [] });

    // Then — uma única densa, 57 pesos + 1 viés
    assert.deepStrictEqual(camadas(model), ['Dense']);
    assert.equal(model.countParams(), 58);
    assert.equal(model.layers[0].activation.getClassName().toLowerCase(), 'sigmoid');

    model.dispose();
  });

  it('dada a regressão logística, quando o dropout está ligado, então ela continua sem camadas', () => {
    // Given — não há unidade oculta para desligar; o dropout não tem
    // onde entrar, e a saída nunca o recebe
    const model = buildModel(57, { units: [], dropout: 0.5 });

    // Then
    assert.equal(model.layers.length, 1);

    model.dispose();
  });

  it('dada uma entrada de tamanho diferente, quando a rede não tem oculta, então a saída declara o formato', () => {
    // Given — sem oculta, é a própria saída que precisa saber a entrada
    const model = buildModel(4, { units: [] });

    // Then
    assert.deepStrictEqual(model.inputs[0].shape, [null, 4]);
    assert.equal(model.countParams(), 5);

    model.dispose();
  });

  it('dadas arquiteturas maiores, quando comparadas, então os parâmetros crescem', () => {
    // Given / When
    const contagens = [[], [4], [16, 8], [128, 64]].map((units) => {
      const model = buildModel(57, { units });
      const total = model.countParams();

      model.dispose();

      return total;
    });

    // Then
    assert.deepStrictEqual(contagens, [58, 237, 1073, 15745]);
  });
});

describe('resolveUnits', () => {
  it('dado nenhum argumento, quando resolvido, então a fonte mantém a arquitetura padrão', () => {
    // Given / When / Then — `null` é "não pedida", diferente de "nenhuma camada"
    assert.equal(resolveUnits([]), null);
    assert.equal(resolveUnits(['--source=german']), null);
  });

  it('dada uma lista, quando resolvida, então vira as camadas ocultas', () => {
    // Given / When / Then
    assert.deepStrictEqual(resolveUnits(['--units=64,32']), [64, 32]);
    assert.deepStrictEqual(resolveUnits(['--units=8']), [8]);
    assert.deepStrictEqual(resolveUnits(['--units=16,16,16']), [16, 16, 16]);
  });

  it('dado --units=0, quando resolvido, então pede a regressão logística', () => {
    // Given — é a única grafia aceita para "nenhuma camada oculta"
    assert.deepStrictEqual(resolveUnits(['--units=0']), []);
  });

  it('dado um valor vazio, quando resolvido, então recusa em vez de trocar a arquitetura em silêncio', () => {
    // Given — a mesma lição de `--l2=`: argumento em branco não é intenção
    assert.throws(() => resolveUnits(['--units=']), /Valor inválido para --units/);
    assert.throws(() => resolveUnits(['--units=   ']), /Valor inválido para --units/);
  });

  it('dados valores impossíveis, quando resolvidos, então lançam', () => {
    // Given — zero no meio da lista não é regressão logística, é erro
    assert.throws(() => resolveUnits(['--units=abc']), /Valor inválido/);
    assert.throws(() => resolveUnits(['--units=16,0']), /Valor inválido/);
    assert.throws(() => resolveUnits(['--units=16.5']), /Valor inválido/);
    assert.throws(() => resolveUnits(['--units=-8']), /Valor inválido/);
    assert.throws(() => resolveUnits(['--units=2048']), /Valor inválido/);
    assert.throws(() => resolveUnits(['--units=1,1,1,1,1,1,1,1,1']), /Valor inválido/);
  });
});

describe('resolveArchitectureRun', () => {
  it('dado nenhum argumento, quando resolvido, então a comparação não roda', () => {
    // Given / When / Then
    assert.equal(resolveArchitectureRun([]), null);
    assert.equal(resolveArchitectureRun(['--cv']), null);
  });

  it('dado --arquiteturas, quando resolvido, então usa uma repetição', () => {
    // Given / When / Then
    assert.deepStrictEqual(resolveArchitectureRun(['--arquiteturas']), { repeats: 1 });
  });

  it('dado --repeticoes, quando resolvido, então usa o número pedido', () => {
    // Given / When / Then
    assert.deepStrictEqual(
      resolveArchitectureRun(['--arquiteturas', '--repeticoes=5']),
      { repeats: 5 },
    );
  });

  it('dados valores impossíveis, quando resolvidos, então lançam', () => {
    // Given / When / Then
    assert.throws(() => resolveArchitectureRun(['--arquiteturas=5']), /sem valor/);
    assert.throws(
      () => resolveArchitectureRun(['--arquiteturas', '--repeticoes=0']),
      /inteiro entre 1 e 20/,
    );
    assert.throws(
      () => resolveArchitectureRun(['--arquiteturas', '--repeticoes=2.5']),
      /inteiro entre 1 e 20/,
    );
  });
});

describe('ARCHITECTURES / compareArchitectures', () => {
  const fonteEmMemoria = () => {
    const customers = Array.from({ length: 60 }, (ignorado, index) => ({
      a: index % 2,
      b: (index % 5) / 5,
      risk: index % 20 < 7 ? 1 : 0,
    }));

    return {
      id: 'memoria',
      label: 'Fonte de teste',
      csvPath: '(memória)',
      featureNames: ['a', 'b'],
      regularization: { l2: 0, dropout: 0 },
      ensure: () => ({ created: false }),
      read: async () => customers,
      fitScaler: () => null,
      toVector: (customer) => [customer.a, customer.b],
    };
  };

  it('dada a lista de arquiteturas, quando lida, então o piso é a regressão logística', () => {
    // Given — comparar só redes entre si esconderia a pergunta que
    // importa: as camadas ocultas pagam o próprio custo?
    assert.deepStrictEqual(ARCHITECTURES[0].units, []);
    assert.ok(ARCHITECTURES.some(({ units }) => units === HIDDEN_UNITS));
  });

  it('dadas as arquiteturas, quando conferidas, então nenhum rótulo se repete', () => {
    // Given / When / Then — rótulo repetido tornaria a tabela ilegível
    const rotulos = ARCHITECTURES.map(({ label }) => label);

    assert.equal(new Set(rotulos).size, rotulos.length);
  });

  it('dadas duas arquiteturas, quando comparadas, então cada uma vira uma linha medida', async () => {
    // Given
    const architectures = [
      { label: 'logística', units: [] },
      { label: '4', units: [4] },
    ];

    // When
    const { rows, baseline } = await compareArchitectures(fonteEmMemoria(), {
      architectures, folds: 2,
    });

    // Then
    assert.equal(rows.length, 2);
    assert.equal(rows[0].parameters, 3);   // 2 features + viés
    assert.ok(rows[1].parameters > rows[0].parameters);
    rows.forEach((row) => {
      ['accuracy', 'auc', 'cost', 'epochs'].forEach((metrica) => {
        assert.ok(Number.isFinite(row[metrica].mean), `${row.label} ${metrica}`);
        assert.ok(Number.isFinite(row[metrica].standardError), `${row.label} ${metrica}`);
      });
    });
    assert.ok(Number.isFinite(baseline.mean));
  });

  it('dadas repetições, quando comparadas, então todas as dobras entram no mesmo resumo', async () => {
    // Given — repetir com a MESMA semente mediria só os pesos iniciais
    const architectures = [{ label: 'logística', units: [] }];

    // When
    const uma = await compareArchitectures(fonteEmMemoria(), {
      architectures, folds: 2, repeats: 1,
    });
    const duas = await compareArchitectures(fonteEmMemoria(), {
      architectures, folds: 2, repeats: 2,
    });

    // Then
    assert.equal(uma.repeats, 1);
    assert.equal(duas.repeats, 2);
    assert.ok(duas.baseline.mean > 0);
  });

  it('dado o resultado, quando formatado, então a tabela traz parâmetros, baseline e protocolo', async () => {
    // Given
    const resultado = await compareArchitectures(fonteEmMemoria(), {
      architectures: [{ label: 'logística', units: [] }], folds: 2,
    });

    // When
    const texto = formatArchitectureComparison(resultado);

    // Then
    assert.ok(texto.includes('Arquitetura'));
    assert.ok(texto.includes('Parâmetros'));
    assert.ok(texto.includes('logística'));
    assert.ok(texto.includes('Baseline da classe majoritária'));
    assert.ok(texto.includes('2 dobras × 1'));
  });

  it('dado um orçamento de treino próprio, quando informado, então ele substitui o compartilhado', async () => {
    // Given — comparar arquiteturas com orçamento fixo é justo com as
    // redes e injusto com os modelos pequenos; este argumento é o que
    // permite medir as duas coisas
    const resultado = await crossValidate(fonteEmMemoria(), {
      folds: 2, units: [], training: { epochs: 1 },
    });

    // Then
    resultado.folds.forEach((dobra) => assert.equal(dobra.epochs, 1));
  });

  it('dada a validação cruzada, quando concluída, então informa parâmetros e épocas por dobra', async () => {
    // Given — é o que a comparação de arquiteturas precisa saber
    const resultado = await crossValidate(fonteEmMemoria(), { folds: 2, units: [4] });

    // Then
    assert.equal(resultado.parameters, 17);   // 2*4+4 ocultas + 4+1 saída
    resultado.folds.forEach((dobra) => {
      assert.ok(Number.isInteger(dobra.epochs) && dobra.epochs > 0);
      assert.equal(dobra.parameters, 17);
    });
    assert.ok(Number.isFinite(resultado.summary.epochs.mean));
  });
});

describe('TRAINING', () => {
  it('dada a configuração de treino, quando lida, então é uma só para os dois caminhos', () => {
    // Given — se o fluxo principal e a validação cruzada treinassem
    // diferente, a estimativa cruzada mediria outro modelo
    assert.deepStrictEqual(TRAINING, {
      epochs: 40,
      batchSize: 32,
      validationSplit: 0.2,
      patience: 5,
    });
  });
});

describe('majorityBaseline', () => {
  it('dados rótulos majoritariamente negativos, quando medido, então devolve a fatia da classe majoritária', () => {
    // Given — 7 baixos e 3 altos, como no German Credit
    const labels = [...Array(7).fill([0]), ...Array(3).fill([1])];

    // When / Then
    assert.equal(majorityBaseline(labels), 0.7);
  });

  it('dados rótulos majoritariamente positivos, quando medido, então a maioria continua sendo o piso', () => {
    // Given
    const labels = [...Array(2).fill([0]), ...Array(8).fill([1])];

    // When / Then
    assert.equal(majorityBaseline(labels), 0.8);
  });

  it('dados rótulos equilibrados, quando medido, então o piso é 0.5', () => {
    // Given / When / Then
    assert.equal(majorityBaseline([[0], [1]]), 0.5);
  });
});

// ==================================================
// Fontes de dados
// ==================================================
describe('SOURCES / resolveSourceId', () => {
  it('dado nenhum argumento, quando a fonte é resolvida, então o padrão é o dataset real', () => {
    // Given / When / Then
    assert.equal(resolveSourceId([]), 'german');
    assert.equal(DEFAULT_SOURCE_ID, 'german');
  });

  it('dado --source=synthetic, quando resolvido, então escolhe o dataset sintético', () => {
    // Given / When / Then
    assert.equal(resolveSourceId(['--source=synthetic']), 'synthetic');
  });

  it('dada uma fonte inexistente, quando resolvida, então lança listando as válidas', () => {
    // Given / When / Then
    assert.throws(
      () => resolveSourceId(['--source=xpto']),
      /synthetic, german, german-ordinal/,
    );
  });

  it('dadas as fontes registradas, quando o contrato é conferido, então todas o cumprem', () => {
    // Given — o pipeline depende de todas exporem a mesma superfície
    const obrigatorios = [
      'id', 'label', 'csvPath', 'columns', 'precision', 'featureNames',
      'ensure', 'read', 'fitScaler', 'toVector', 'sampleCustomer',
      'regularization',
    ];

    // When / Then
    Object.values(SOURCES).forEach((source) => {
      const faltando = obrigatorios.filter((chave) => source[chave] === undefined);
      assert.deepEqual(faltando, [], `fonte ${source.id} não cumpre o contrato`);
    });
  });

  it('dado o cliente de exemplo de cada fonte, quando vetorizado, então tem o tamanho das features', () => {
    // Given / When / Then
    Object.values(SOURCES).forEach((source) => {
      const scaler = source.fitScaler([source.sampleCustomer]);
      const vetor = source.toVector(source.sampleCustomer, scaler);

      assert.equal(vetor.length, source.featureNames.length, `fonte ${source.id}`);
    });
  });

  it('dada a fonte sintética, quando o scaler é ajustado, então não depende dos dados', () => {
    // Given — nós geramos os dados, então a escala já era conhecida
    assert.equal(SYNTHETIC_SOURCE.fitScaler([]), null);
  });

  it('dada a fonte sintética, quando vetorizada, então bate com toFeatureVector', () => {
    // Given — as duas precisam ser o MESMO caminho de normalização
    const cliente = SYNTHETIC_SOURCE.sampleCustomer;

    // When / Then
    assert.deepEqual(
      SYNTHETIC_SOURCE.toVector(cliente, null),
      toFeatureVector(cliente),
    );
  });

  it('dada a fonte real, quando vetorizada com um scaler, então usa a escala medida', () => {
    // Given
    const treino = [
      { ...GERMAN_SOURCE.sampleCustomer, age: 20 },
      { ...GERMAN_SOURCE.sampleCustomer, age: 60 },
    ];
    const scaler = GERMAN_SOURCE.fitScaler(treino);
    const posicaoIdade = GERMAN_SOURCE.featureNames.indexOf('age');

    // When
    const vetor = GERMAN_SOURCE.toVector({ ...treino[0], age: 40 }, scaler);

    // Then
    assert.equal(vetor[posicaoIdade], 0.5);
  });

  it('dada a fonte real com o CSV em disco, quando ensure roda, então não lança', () => {
    // Given — o arquivo é versionado, então quem clona já tem os dados
    // When / Then
    assert.equal(GERMAN_SOURCE.ensure().created, false);
  });

  it('dada a fonte real sem o CSV, quando ensure roda, então instrui a rodar o fetch', () => {
    // Given — mensagem acionável em vez de stack trace
    assert.throws(
      () => GERMAN_SOURCE.ensure('/caminho/que/nao/existe.csv'),
      /npm run fetch:german/,
    );
  });
});

describe('toCsv com schema da fonte', () => {
  it('dado o schema do German Credit, quando serializado, então o cabeçalho é o dele', () => {
    // Given
    const cliente = { ...GERMAN_SOURCE.sampleCustomer, risk: 1 };

    // When
    const [cabecalho, linha] = toCsv([cliente], {
      columns: GERMAN_COLUMNS,
      precision: GERMAN_PRECISION,
    }).split('\n');

    // Then
    assert.equal(cabecalho, GERMAN_COLUMNS.join(','));
    assert.equal(linha, GERMAN_COLUMNS.map((c) => cliente[c]).join(','));
  });

  it('dado nenhum schema, quando serializado, então continua usando o sintético', () => {
    // Given — compatibilidade: as chamadas antigas não mudaram de comportamento
    const [cabecalho] = toCsv(createCustomers(1)).split('\n');

    // Then
    assert.equal(cabecalho, CSV_COLUMNS.join(','));
  });
});

// ==================================================
// Codificação one-hot das colunas qualitativas
// ==================================================
describe('oneHotEncode / ordinalEncode', () => {
  it('dado um índice, quando codificado em one-hot, então só aquela posição vale 1', () => {
    // Given / When
    const vetor = oneHotEncode(4, 2);

    // Then
    assert.deepEqual(vetor, [0, 0, 1, 0]);
  });

  it('dado qualquer índice, quando codificado, então a soma do vetor é sempre 1', () => {
    // Given — é o que garante que nenhuma categoria "pesa" mais que outra
    for (let indice = 0; indice < 5; indice += 1) {
      const vetor = oneHotEncode(5, indice);

      assert.equal(vetor.reduce((total, valor) => total + valor, 0), 1);
      assert.equal(vetor.length, 5);
    }
  });

  it('dada uma categoria única, quando codificada, então o vetor tem largura 1', () => {
    // Given / When / Then
    assert.deepEqual(oneHotEncode(1, 0), [1]);
  });

  it('dado um índice, quando codificado como ordinal, então vira um único número em [0, 1]', () => {
    // Given — a codificação que o one-hot substitui
    assert.deepEqual(ordinalEncode(5, 0), [0]);
    assert.deepEqual(ordinalEncode(5, 4), [1]);
    assert.deepEqual(ordinalEncode(5, 2), [0.5]);
  });

  it('dada uma categoria única, quando codificada como ordinal, então não divide por zero', () => {
    // Given / When / Then
    assert.deepEqual(ordinalEncode(1, 0), [0]);
  });
});

describe('germanFeatureNames / toGermanVector', () => {
  const clientes = [
    { ...GERMAN_SOURCE.sampleCustomer, age: 20, creditAmount: 1000 },
    { ...GERMAN_SOURCE.sampleCustomer, age: 60, creditAmount: 5000 },
  ];

  it('dada a codificação one-hot, quando os nomes são gerados, então há 57 features', () => {
    // Given — 7 numéricas + 50 níveis categóricos
    const niveis = Object.values(GERMAN_CATEGORICAL)
      .reduce((total, codes) => total + codes.length, 0);

    // Then
    assert.equal(niveis, 50);
    assert.equal(germanFeatureNames('onehot').length, 57);
  });

  it('dada a codificação ordinal, quando os nomes são gerados, então há 19 features', () => {
    // Given — 7 numéricas + 12 categóricas, uma coluna cada
    assert.equal(germanFeatureNames('ordinal').length, 19);
  });

  it('dados os nomes one-hot, quando lidos, então cada nível vira "campo=código"', () => {
    // Given / When
    const nomes = germanFeatureNames('onehot');

    // Then
    assert.ok(nomes.includes('checkingStatus=A11'));
    assert.ok(nomes.includes('purpose=A410'));
    assert.equal(nomes[0], 'durationMonths');
  });

  it('dado um cliente, quando vetorizado, então o vetor tem o tamanho dos nomes', () => {
    // Given — se estes divergirem, um peso da rede aponta para a coluna errada
    const scaler = GERMAN_SOURCE.fitScaler(clientes);

    // When / Then
    assert.equal(
      toGermanVector(clientes[0], scaler, 'onehot').length,
      germanFeatureNames('onehot').length,
    );
    assert.equal(
      toGermanVector(clientes[0], scaler, 'ordinal').length,
      germanFeatureNames('ordinal').length,
    );
  });

  it('dado um cliente, quando vetorizado, então as numéricas vêm primeiro e escaladas', () => {
    // Given
    const scaler = GERMAN_SOURCE.fitScaler(clientes);

    // When
    const vetor = toGermanVector({ ...clientes[0], age: 40 }, scaler, 'onehot');
    const posicaoIdade = germanFeatureNames('onehot').indexOf('age');

    // Then — 40 no meio de [20, 60]
    assert.equal(vetor[posicaoIdade], 0.5);
  });

  it('dado um cliente, quando vetorizado em one-hot, então o bloco categórico só tem 0 e 1', () => {
    // Given
    const scaler = GERMAN_SOURCE.fitScaler(clientes);

    // When
    const vetor = toGermanVector(clientes[0], scaler, 'onehot');
    const categorico = vetor.slice(GERMAN_NUMERIC.length);

    // Then
    assert.ok(categorico.every((valor) => valor === 0 || valor === 1));
    // Uma coluna qualitativa acesa por campo
    assert.equal(
      categorico.reduce((total, valor) => total + valor, 0),
      Object.keys(GERMAN_CATEGORICAL).length,
    );
  });

  it('dado o mesmo cliente, quando vetorizado nas duas codificações, então os vetores diferem', () => {
    // Given — a troca de codificação precisa realmente mudar a entrada
    const scaler = GERMAN_SOURCE.fitScaler(clientes);

    // When / Then
    assert.notDeepEqual(
      toGermanVector(clientes[0], scaler, 'onehot'),
      toGermanVector(clientes[0], scaler, 'ordinal'),
    );
  });
});

// ==================================================
// Auditoria de disparidade
// ==================================================
describe('isFemale / summarizeGroup / auditByGroup', () => {
  const cliente = (personalStatus, risk) => ({ [GERMAN_AUDIT_COLUMN]: personalStatus, risk });
  const indiceFeminino = GERMAN_AUDIT_CODES.indexOf(FEMALE_CODE);

  it('dado o código A92, quando o grupo é lido, então é o feminino', () => {
    // Given — A92 é o único código feminino presente no arquivo
    assert.equal(isFemale(cliente(indiceFeminino, 0)), true);
    assert.equal(isFemale(cliente(0, 0)), false);
    assert.equal(isFemale(cliente(2, 0)), false);
  });

  it('dado um grupo, quando resumido, então separa taxa real de taxa marcada', () => {
    // Given — 2 de 4 são maus pagadores; o modelo marca 3
    const linhas = [
      { risk: 1, score: 0.9 }, { risk: 1, score: 0.2 },
      { risk: 0, score: 0.8 }, { risk: 0, score: 0.7 },
    ];

    // When
    const resumo = summarizeGroup(linhas, 0.5);

    // Then
    assert.equal(resumo.total, 4);
    assert.equal(resumo.baseRate, 0.5);          // 2 de 4 são realmente ruins
    assert.equal(resumo.flaggedRate, 0.75);      // 3 de 4 foram marcados
    assert.equal(resumo.falseNegativeRate, 0.5); // 1 dos 2 ruins escapou
  });

  it('dado um grupo sem positivos, quando resumido, então o FNR é 0 e não NaN', () => {
    // Given / When
    const resumo = summarizeGroup([{ risk: 0, score: 0.1 }], 0.5);

    // Then
    assert.equal(resumo.falseNegativeRate, 0);
  });

  it('dados dois grupos tratados igual, quando auditados, então a razão é 1', () => {
    // Given — os dois totalmente marcados: 0/0 é tratamento idêntico,
    // não disparidade máxima
    const customers = [cliente(indiceFeminino, 0), cliente(0, 0)];

    // When / Then
    assert.equal(auditByGroup(customers, [0.9, 0.9], 0.5).approvalRatio, 1);
    assert.equal(auditByGroup(customers, [0.1, 0.1], 0.5).approvalRatio, 1);
  });

  it('dado nenhum homem aprovado e alguma mulher sim, quando auditado, então a razão é infinita', () => {
    // Given — disparidade sem limite superior, e é literalmente o caso
    const customers = [cliente(indiceFeminino, 0), cliente(0, 0)];

    // When
    const { approvalRatio } = auditByGroup(customers, [0.1, 0.9], 0.5);

    // Then
    assert.equal(approvalRatio, Infinity);
  });

  it('dado um grupo marcado mais que o outro, quando auditado, então a razão cai abaixo de 1', () => {
    // Given — todas as mulheres marcadas, nenhum homem
    const customers = [
      cliente(indiceFeminino, 0), cliente(indiceFeminino, 0),
      cliente(0, 0), cliente(0, 0),
    ];
    const scores = [0.9, 0.9, 0.1, 0.1];

    // When
    const auditoria = auditByGroup(customers, scores, 0.5);

    // Then
    assert.equal(auditoria.women.flaggedRate, 1);
    assert.equal(auditoria.men.flaggedRate, 0);
    assert.equal(auditoria.approvalRatio, 0);
  });

  it('dada uma auditoria, quando formatada, então mostra os dois grupos e a razão', () => {
    // Given
    const customers = [cliente(indiceFeminino, 1), cliente(0, 0)];

    // When
    const texto = formatAudit(auditByGroup(customers, [0.9, 0.1], 0.5));

    // Then
    assert.ok(texto.includes('Mulheres'));
    assert.ok(texto.includes('Homens'));
    assert.ok(texto.includes('regra dos 4/5'));
  });
});

// ==================================================
// Mitigação da disparidade
// ==================================================
describe('rateThreshold', () => {
  it('dada uma fração, quando o limiar é calibrado, então ele marca essa fração', () => {
    // Given — quatro scores, metade deve ser marcada
    const scores = [0.6, 0.9, 0.7, 0.8];

    // When
    const limiar = rateThreshold(scores, 0.5);

    // Then — o corte cai no MEIO entre 0.8 (marcado) e 0.7 (de fora)
    assert.equal(limiar, 0.75);
    assert.equal(scores.filter((score) => score >= limiar).length, 2);
  });

  it('dada a fração 0, quando calibrada, então o limiar é inalcançável', () => {
    // Given / When — marcar ninguém é uma resposta legítima
    const limiar = rateThreshold([0.1, 0.9], 0);

    // Then — nenhuma probabilidade chega a Infinity
    assert.equal(limiar, Infinity);
    assert.equal([0.1, 0.9].filter((score) => score >= limiar).length, 0);
  });

  it('dada a fração 1, quando calibrada, então o limiar marca todo mundo', () => {
    // Given / When
    const limiar = rateThreshold([0.1, 0.9], 1);

    // Then — 0 é alcançado por qualquer probabilidade
    assert.equal(limiar, 0);
    assert.equal([0.1, 0.9].filter((score) => score >= limiar).length, 2);
  });

  it('dados scores empatados na fronteira, quando calibrado, então a fração vira um piso', () => {
    // Given — o ponto médio de dois empatados é o próprio valor, e o
    // `>=` não tem como separar um do outro
    const scores = [0.9, 0.5, 0.5, 0.1];

    // When
    const limiar = rateThreshold(scores, 0.5);

    // Then — pediu 2, marcou 3, e isso é o comportamento documentado
    assert.equal(limiar, 0.5);
    assert.equal(scores.filter((score) => score >= limiar).length, 3);
  });

  it('dada uma lista vazia, quando calibrada, então o limiar é inalcançável', () => {
    // Given / When / Then — não há como marcar fração de nada
    assert.equal(rateThreshold([], 0.5), Infinity);
  });
});

describe('fitGroupThresholds', () => {
  const cliente = (personalStatus, risk) => ({ [GERMAN_AUDIT_COLUMN]: personalStatus, risk });
  const mulher = GERMAN_AUDIT_CODES.indexOf(FEMALE_CODE);

  // Duas mulheres com score alto, dois homens com score baixo: no limiar
  // único elas são marcadas e eles não.
  const customers = [
    cliente(mulher, 1), cliente(mulher, 0),
    cliente(0, 1), cliente(0, 0),
  ];
  const scores = [0.9, 0.8, 0.4, 0.1];

  it('dado um grupo marcado demais, quando calibrado, então ele recebe o limiar mais alto', () => {
    // Given / When — o limiar único marca 2 de 4, ou seja 50%
    const limiares = fitGroupThresholds(customers, scores, 0.5);

    // Then — cada grupo passa a marcar 1 de 2, e para isso o corte das
    // mulheres tem de subir e o dos homens tem de descer
    assert.ok(limiares.women > 0.5);
    assert.ok(limiares.men < 0.5);
  });

  it('dados os limiares calibrados, quando aplicados ao próprio conjunto, então as taxas se igualam', () => {
    // Given — este é o caso em que a paridade sai por CONSTRUÇÃO: o
    // conjunto calibrado e o auditado são o mesmo. É por isso que o
    // projeto calibra no treino e audita no teste.
    const limiares = fitGroupThresholds(customers, scores, 0.5);

    // When
    const auditoria = auditByGroup(customers, scores, limiares);

    // Then
    assert.equal(auditoria.women.flaggedRate, auditoria.men.flaggedRate);
    assert.equal(auditoria.approvalRatio, 1);
  });

  it('dado um conjunto sem mulheres, quando calibrado, então o limiar delas é inalcançável', () => {
    // Given — grupo vazio não tem quantil; marcar ninguém é a única
    // resposta que não inventa dado
    const somenteHomens = [cliente(0, 1), cliente(0, 0)];

    // When
    const limiares = fitGroupThresholds(somenteHomens, [0.9, 0.1], 0.5);

    // Then
    assert.equal(limiares.women, Infinity);
    assert.ok(Number.isFinite(limiares.men));
  });

  it('dada a mitigação, quando calibrada, então nenhum score muda', () => {
    // Given — a correção é de LIMIAR, não de modelo: os mesmos scores
    // entram nas duas políticas
    const copia = [...scores];

    // When
    fitGroupThresholds(customers, scores, 0.5);

    // Then
    assert.deepStrictEqual(scores, copia);
  });
});

describe('thresholdFor / auditoria com limiar por grupo', () => {
  const cliente = (personalStatus, risk) => ({ [GERMAN_AUDIT_COLUMN]: personalStatus, risk });
  const mulher = GERMAN_AUDIT_CODES.indexOf(FEMALE_CODE);

  it('dado um número, quando consultado, então vale para os dois grupos', () => {
    // Given / When / Then
    assert.equal(thresholdFor(0.5, true), 0.5);
    assert.equal(thresholdFor(0.5, false), 0.5);
  });

  it('dado um par, quando consultado, então cada grupo recebe o seu', () => {
    // Given / When / Then
    assert.equal(thresholdFor({ women: 0.8, men: 0.2 }, true), 0.8);
    assert.equal(thresholdFor({ women: 0.8, men: 0.2 }, false), 0.2);
  });

  it('dados limiares por grupo, quando auditados, então a razão sobe e os dois limiares ficam registrados', () => {
    // Given — todas as mulheres marcadas, nenhum homem: razão 0
    const customers = [
      cliente(mulher, 0), cliente(mulher, 0),
      cliente(0, 0), cliente(0, 0),
    ];
    const scores = [0.9, 0.6, 0.4, 0.1];

    // When
    const antes = auditByGroup(customers, scores, 0.5);
    const depois = auditByGroup(customers, scores, { women: 0.7, men: 0.3 });

    // Then
    assert.equal(antes.approvalRatio, 0);
    assert.equal(depois.approvalRatio, 1);
    assert.deepStrictEqual(depois.thresholds, { women: 0.7, men: 0.3 });
  });

  it('dado um limiar único, quando auditado, então os dois limiares registrados são iguais', () => {
    // Given / When
    const customers = [cliente(mulher, 0), cliente(0, 0)];
    const { thresholds } = auditByGroup(customers, [0.9, 0.1], 0.5);

    // Then
    assert.deepStrictEqual(thresholds, { women: 0.5, men: 0.5 });
  });

  it('dados clientes e scores, quando as linhas de auditoria são montadas, então trazem risco, score e grupo', () => {
    // Given / When
    const linhas = toAuditRows([cliente(mulher, 1), cliente(0, 0)], [0.9, 0.1]);

    // Then
    assert.deepStrictEqual(linhas, [
      { risk: 1, score: 0.9, female: true },
      { risk: 0, score: 0.1, female: false },
    ]);
  });
});

describe('summarizeDecisions', () => {
  const cliente = (personalStatus, risk) => ({ [GERMAN_AUDIT_COLUMN]: personalStatus, risk });
  const mulher = GERMAN_AUDIT_CODES.indexOf(FEMALE_CODE);

  // Uma mulher ruim com score alto, uma boa com score alto, um homem ruim
  // com score baixo e um bom com score baixo.
  const customers = [
    cliente(mulher, 1), cliente(mulher, 0),
    cliente(0, 1), cliente(0, 0),
  ];
  const scores = [0.9, 0.8, 0.2, 0.1];

  it('dado um limiar único, quando as decisões são resumidas, então acurácia e custo saem dos erros', () => {
    // Given / When — marca as duas mulheres: 1 acerto e 1 falso positivo;
    // deixa os dois homens passar: 1 falso negativo e 1 acerto
    const resumo = summarizeDecisions(customers, scores, 0.5);

    // Then
    assert.equal(resumo.falsePositives, 1);
    assert.equal(resumo.falseNegatives, 1);
    assert.equal(resumo.accuracy, 0.5);
    assert.equal(resumo.cost, FALSE_POSITIVE_COST + FALSE_NEGATIVE_COST);
  });

  it('dado o custo assimétrico, quando calculado, então o falso negativo pesa cinco vezes mais', () => {
    // Given — o mesmo conjunto, agora sem nenhuma marcação: os dois maus
    // pagadores escapam
    const resumo = summarizeDecisions(customers, scores, 1);

    // Then
    assert.equal(resumo.falsePositives, 0);
    assert.equal(resumo.falseNegatives, 2);
    assert.equal(resumo.cost, 2 * FALSE_NEGATIVE_COST);
    assert.equal(resumo.cost, 10);
  });

  it('dado um limiar por grupo, quando resumido, então cada grupo é cortado no seu', () => {
    // Given — com um corte alto para elas e baixo para eles, os mesmos
    // scores produzem a decisão CERTA nos quatro casos
    const resumo = summarizeDecisions(customers, scores, { women: 0.85, men: 0.15 });

    // Then
    assert.equal(resumo.falsePositives, 0);
    assert.equal(resumo.falseNegatives, 0);
    assert.equal(resumo.accuracy, 1);
    assert.equal(resumo.cost, 0);
  });

  it('dados custos próprios, quando informados, então substituem os do laboratório', () => {
    // Given / When — invertendo a assimetria, o falso positivo é que dói
    const resumo = summarizeDecisions(customers, scores, 0.5, {
      falsePositive: 10,
      falseNegative: 1,
    });

    // Then
    assert.equal(resumo.cost, 11);
  });
});

describe('formatMitigation', () => {
  const cliente = (personalStatus, risk) => ({ [GERMAN_AUDIT_COLUMN]: personalStatus, risk });
  const mulher = GERMAN_AUDIT_CODES.indexOf(FEMALE_CODE);

  it('dadas duas políticas, quando formatadas, então a tabela mostra limiares, razão e custo', () => {
    // Given
    const customers = [
      cliente(mulher, 0), cliente(mulher, 0),
      cliente(0, 0), cliente(0, 0),
    ];
    const scores = [0.9, 0.6, 0.4, 0.1];
    const politica = (label, threshold) => ({
      label,
      audit: auditByGroup(customers, scores, threshold),
      decisions: summarizeDecisions(customers, scores, threshold),
    });

    // When
    const texto = formatMitigation([
      politica('Limiar único', 0.5),
      politica('Limiar por grupo', { women: 0.7, men: 0.3 }),
    ]);

    // Then
    assert.ok(texto.includes('Limiar único'));
    assert.ok(texto.includes('Limiar por grupo'));
    assert.ok(texto.includes('0.7000'));
    assert.ok(texto.includes('0.3000'));
    assert.ok(texto.includes('1.000'));  // a razão depois da mitigação
  });
});

describe('resolveMitigation', () => {
  it('dado nenhum argumento, quando resolvido, então a mitigação fica desligada', () => {
    // Given / When / Then — o padrão é NÃO usar o atributo protegido
    assert.equal(resolveMitigation([]), false);
    assert.equal(resolveMitigation(['--source=german']), false);
  });

  it('dado --mitigar, quando resolvido, então a mitigação liga', () => {
    // Given / When / Then
    assert.equal(resolveMitigation(['--mitigar']), true);
    assert.equal(resolveMitigation(['--source=german', '--mitigar']), true);
  });

  it('dado --mitigar com valor, quando resolvido, então recusa em vez de adivinhar', () => {
    // Given — `--mitigar=false` LIGARIA a política se a presença bastasse,
    // e é exatamente o oposto do que quem escreveu isso queria
    assert.throws(() => resolveMitigation(['--mitigar=false']), /sem valor/);
    assert.throws(() => resolveMitigation(['--mitigar=0']), /sem valor/);
    assert.throws(() => resolveMitigation(['--mitigar=true']), /sem valor/);
  });
});

describe('atributo protegido fora do modelo', () => {
  it('dadas as features da fonte real, quando procuradas, então nenhuma vem do sexo', () => {
    // Given — este é o teste que impede a coluna de voltar por descuido
    const suspeitas = GERMAN_SOURCE.featureNames.filter((nome) =>
      nome.toLowerCase().includes(GERMAN_AUDIT_COLUMN.toLowerCase())
      || GERMAN_AUDIT_CODES.some((codigo) => nome.endsWith(`=${codigo}`)));

    // Then
    assert.deepEqual(suspeitas, []);
  });

  it('dada a coluna de auditoria, quando o CSV é lido, então ela existe no arquivo', () => {
    // Given — fora do modelo, mas presente para auditar
    assert.ok(GERMAN_COLUMNS.includes(GERMAN_AUDIT_COLUMN));
    assert.ok(!GERMAN_SOURCE.featureNames.includes(GERMAN_AUDIT_COLUMN));
  });

  it('dado o scaler da fonte real, quando ajustado, então não mede a coluna de auditoria', () => {
    // Given / When
    const scaler = GERMAN_SOURCE.fitScaler([GERMAN_SOURCE.sampleCustomer]);

    // Then
    assert.deepEqual(scaler.featureNames, GERMAN_NUMERIC);
    assert.equal(scaler.min[GERMAN_AUDIT_COLUMN], undefined);
  });
});

describe('createGermanSource', () => {
  it('dadas as duas variantes, quando comparadas, então diferem só na codificação', () => {
    // Given / When / Then
    assert.equal(GERMAN_SOURCE.encoding, 'onehot');
    assert.equal(GERMAN_ORDINAL_SOURCE.encoding, 'ordinal');
    assert.equal(GERMAN_SOURCE.csvPath, GERMAN_ORDINAL_SOURCE.csvPath);
    assert.deepEqual(GERMAN_SOURCE.columns, GERMAN_ORDINAL_SOURCE.columns);
    assert.notEqual(
      GERMAN_SOURCE.featureNames.length,
      GERMAN_ORDINAL_SOURCE.featureNames.length,
    );
  });

  it('dada a variante ordinal, quando registrada, então está disponível por --source', () => {
    // Given / When / Then
    assert.equal(resolveSourceId(['--source=german-ordinal']), 'german-ordinal');
  });
});
