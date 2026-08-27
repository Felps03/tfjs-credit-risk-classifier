const fs = require('node:fs');
const path = require('node:path');

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

// Pasta onde o modelo treinado é persistido (ignorada pelo git).
const MODEL_DIR = path.join(__dirname, 'model');

// CSV com os dados brutos. Em um projeto real ele viria de fora
// (export de um banco, entrega de um parceiro); aqui é gerado para o
// laboratório continuar auto-contido.
const CSV_PATH = path.join(__dirname, 'data', 'customers.csv');
const CSV_LABEL_COLUMN = 'risk';
const CSV_COLUMNS = [
  'income',
  'debtRatio',
  'latePayments',
  'creditUtilization',
  CSV_LABEL_COLUMN,
];

// Casas decimais por coluna. CSV é texto: a precisão é uma ESCOLHA, e
// arredondar aqui é o que um export de banco de dados faria.
const CSV_PRECISION = {
  income: 2,
  debtRatio: 6,
  latePayments: 0,
  creditUtilization: 6,
  risk: 0,
};

// --------------------------------------------------
// Dataset REAL: German Credit (UCI / Statlog)
// --------------------------------------------------
// 1000 solicitações de crédito reais, coletadas por Hans Hofmann na
// Universidade de Hamburgo e publicadas em 1994. É o contraponto do
// dataset sintético: aqui ninguém escolheu a regra que separa bom de mau
// pagador — ela precisa ser descoberta, e boa parte dela simplesmente
// não está nas colunas.
const GERMAN_CSV_PATH = path.join(__dirname, 'data', 'german-credit.csv');
const GERMAN_SOURCE_URL = 'https://archive.ics.uci.edu/static/public/144/data.csv';

// Códigos qualitativos do arquivo original, na ordem documentada pela
// UCI. A posição na lista vira o valor numérico (codificação ORDINAL).
const GERMAN_CODES = {
  checkingStatus: ['A11', 'A12', 'A13', 'A14'],
  creditHistory: ['A30', 'A31', 'A32', 'A33', 'A34'],
  savingsStatus: ['A61', 'A62', 'A63', 'A64', 'A65'],
  employmentYears: ['A71', 'A72', 'A73', 'A74', 'A75'],
};

// 8 das 20 colunas originais. O recorte fica com o que tem ordem natural,
// porque codificar como ordinal só faz sentido quando existe um "mais" e
// um "menos" — as demais precisariam de one-hot para não inventar ordem.
//
// Fica DE FORA de propósito o atributo 9 (estado civil e SEXO). Usar sexo
// para negar crédito é discriminação, ilegal em vários países, e é um dos
// motivos pelos quais este dataset virou caso clássico da literatura de
// fairness. A coluna existe no arquivo; não entra no modelo.
const GERMAN_FEATURES = [
  'checkingStatus',
  'durationMonths',
  'creditHistory',
  'creditAmount',
  'savingsStatus',
  'employmentYears',
  'installmentRate',
  'age',
];

const GERMAN_COLUMNS = [...GERMAN_FEATURES, CSV_LABEL_COLUMN];

// Todas as colunas convertidas são inteiras: marcos ordinais, contagens,
// meses, anos e o valor do crédito em marcos alemães.
const GERMAN_PRECISION = Object.fromEntries(
  GERMAN_COLUMNS.map((column) => [column, 0]),
);

// Semente do embaralhamento. Fixa de propósito: sem ela cada execução
// mediria um recorte diferente do dataset e nenhum número deste projeto
// se reproduziria. Trocar a semente é trocar o experimento.
const SHUFFLE_SEED = 42;

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
// Clientes em unidades BRUTAS (reais, contagem, percentual), do jeito
// que sairiam de um banco de dados — é isso que vai para o CSV.
const createCustomers = (total = 1200) => Array.from({ length: total }, () => {
  const customer = {
    income: INCOME_MIN + Math.random() * INCOME_RANGE,
    debtRatio: Math.random(),
    latePayments: Math.floor(Math.random() * (MAX_LATE_PAYMENTS + 1)),
    creditUtilization: Math.random(),
  };

  const [incomeNormalized, debtRatio, latePaymentsNormalized, creditUtilization] =
    toFeatureVector(customer);

  // Regra usada somente para criar rótulos sintéticos.
  // O modelo NÃO recebe esta fórmula: ele precisa aprender o padrão.
  const riskScore =
    1.4 * debtRatio +
    1.2 * latePaymentsNormalized +
    1.0 * creditUtilization -
    0.8 * incomeNormalized;

  return { ...customer, risk: riskScore > RISK_RULE_THRESHOLD ? 1 : 0 };
});

// Clientes brutos → matrizes que o TensorFlow consome. Um só caminho de
// normalização, seja o dado gerado em memória ou lido do CSV.
const toDataset = (customers) => ({
  features: customers.map(toFeatureVector),
  labels: customers.map(({ risk }) => [risk]),
});

const createDataset = (total = 1200) => toDataset(createCustomers(total));

// --------------------------------------------------
// 3. CSV: escrever e ler
// --------------------------------------------------
// O CSV guarda os dados BRUTOS, não as features normalizadas. Normalizar
// antes de salvar congelaria as constantes dentro do arquivo — e qualquer
// ajuste na normalização exigiria reexportar tudo.
const formatCsvValue = (column, value, precision) =>
  value.toFixed(precision[column]);

// `columns` e `precision` são parâmetros porque cada fonte tem o seu
// schema: o sintético e o German Credit não compartilham uma única coluna
// além do rótulo. Os padrões mantêm o dataset sintético como estava.
const toCsv = (customers, options = {}) => {
  const { columns = CSV_COLUMNS, precision = CSV_PRECISION } = options;

  return [
    columns.join(','),
    ...customers.map((customer) => columns
      .map((column) => formatCsvValue(column, customer[column], precision))
      .join(',')),
  ].join('\n');
};

const writeCustomersCsv = (customers, filePath = CSV_PATH, options = {}) => {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
  fs.writeFileSync(filePath, `${toCsv(customers, options)}\n`);

  return filePath;
};

// `tf.data.csv` já faz o parse dos números e separa features de rótulo
// pelo nome da coluna — a ordem das colunas no arquivo não importa.
const readCustomersCsv = async (filePath = CSV_PATH) => {
  const dataset = tf.data.csv(`file://${path.resolve(filePath)}`, {
    columnConfigs: { [CSV_LABEL_COLUMN]: { isLabel: true } },
  });

  const rows = await dataset.toArray();

  return rows.map(({ xs, ys }) => ({ ...xs, risk: ys[CSV_LABEL_COLUMN] }));
};

// Ponto central: o dado do arquivo passa pela MESMA `toFeatureVector` do
// treino. É o que impede o CSV de virar uma segunda fonte de verdade.
const loadDatasetCsv = async (filePath = CSV_PATH) =>
  toDataset(await readCustomersCsv(filePath));

// Gera o CSV apenas se ele ainda não existe. É isso que torna o arquivo
// versionável: `npm start` passa a LER um dataset estável em vez de
// sortear um novo a cada execução (o que sujaria o diff toda vez).
// Para trocar os dados de propósito: `npm run seed`.
const ensureCsv = (filePath = CSV_PATH, total = 1200) => {
  if (fs.existsSync(filePath)) {
    return { path: filePath, created: false };
  }

  writeCustomersCsv(createCustomers(total), filePath);

  return { path: filePath, created: true };
};

// --------------------------------------------------
// 4. Ler o arquivo original do German Credit
// --------------------------------------------------
// O arquivo da UCI não serve direto: as colunas qualitativas são códigos
// ('A11', 'A34') que o `tf.data.csv` leria como NaN. Converter é um passo
// obrigatório — e ele fica aqui, isolado e testável, para que todo o
// resto do laboratório continue enxergando um CSV puramente numérico.

// Parser mínimo: o arquivo da UCI não tem aspas nem vírgula dentro de
// campo, então dividir por vírgula basta. Um CSV arbitrário exigiria
// bem mais do que isto.
const parseDelimited = (text) => {
  const [header, ...lines] = text.trim().split(/\r?\n/);
  const columns = header.split(',');

  return lines.map((line) => Object.fromEntries(
    line.split(',').map((value, index) => [columns[index], value]),
  ));
};

// Código → posição na lista documentada. Código desconhecido vira erro,
// não zero silencioso: dado corrompido precisa aparecer na hora, não
// virar uma feature plausível que ninguém desconfia.
const toOrdinal = (codes, code) => {
  const index = codes.indexOf(code);

  if (index === -1) {
    throw new Error(`Código desconhecido no German Credit: ${code}`);
  }

  return index;
};

// Uma linha do arquivo original vira um cliente no vocabulário do
// laboratório. A coluna `class` vale 1 (bom) ou 2 (mau); nossa convenção
// é risk = 1 para o ALTO RISCO, ou seja, para o mau pagador.
const toGermanCustomer = (row) => ({
  checkingStatus: toOrdinal(GERMAN_CODES.checkingStatus, row.Attribute1),
  durationMonths: Number(row.Attribute2),
  creditHistory: toOrdinal(GERMAN_CODES.creditHistory, row.Attribute3),
  creditAmount: Number(row.Attribute5),
  savingsStatus: toOrdinal(GERMAN_CODES.savingsStatus, row.Attribute6),
  employmentYears: toOrdinal(GERMAN_CODES.employmentYears, row.Attribute7),
  installmentRate: Number(row.Attribute8),
  age: Number(row.Attribute13),
  risk: Number(row.class) === 2 ? 1 : 0,
});

const parseGermanCsv = (text) => parseDelimited(text).map(toGermanCustomer);

// --------------------------------------------------
// 5. Normalização ajustada no treino
// --------------------------------------------------
// O dataset sintético podia usar constantes fixas: nós geramos os dados,
// então conhecíamos as faixas de antemão. Com dado real não existe esse
// luxo — as faixas precisam ser MEDIDAS. E medidas só no treino.
//
// Calcular min/max sobre o dataset inteiro parece inofensivo e não é: o
// maior empréstimo do conjunto de teste passaria a influenciar a escala
// aplicada no treino. Isso é vazamento (data leakage), e o efeito é
// sempre o mesmo — o modelo parece melhor na avaliação do que será
// diante de dados que nunca viu.
const fitMinMaxScaler = (customers, featureNames) => {
  const min = {};
  const range = {};

  featureNames.forEach((feature) => {
    const values = customers.map((customer) => customer[feature]);
    const lowest = Math.min(...values);
    const highest = Math.max(...values);

    min[feature] = lowest;
    // Coluna constante daria divisão por zero; 1 mantém o valor em 0.
    range[feature] = highest - lowest || 1;
  });

  return { featureNames, min, range };
};

// Um valor de teste fora da faixa vista no treino sai de [0, 1] de
// propósito. Cortar em 0 e 1 esconderia justamente o caso extremo que o
// modelo nunca viu — e é sobre ele que se quer saber.
const applyMinMaxScaler = ({ featureNames, min, range }, customer) =>
  featureNames.map((feature) => (customer[feature] - min[feature]) / range[feature]);

// --------------------------------------------------
// 6. Fontes de dados
// --------------------------------------------------
// Cada fonte descreve tudo que muda de um dataset para o outro: onde o
// CSV vive, quais colunas viram features, como normalizar e que cliente
// usar na demonstração de inferência.
//
// Todo o resto do laboratório — treino, matriz de confusão, precision,
// ROC, escolha de limiar — não sabe qual fonte está em uso. Trocar de
// dataset não exigiu tocar em nenhuma dessas partes, e é justamente esse
// o teste de que o pipeline estava bem separado.
const SYNTHETIC_SOURCE = {
  id: 'synthetic',
  label: 'Sintético (gerado por este projeto)',
  csvPath: CSV_PATH,
  columns: CSV_COLUMNS,
  precision: CSV_PRECISION,
  featureNames: ['income', 'debtRatio', 'latePayments', 'creditUtilization'],

  ensure: () => ensureCsv(),
  read: () => readCustomersCsv(CSV_PATH),

  // A escala é CONHECIDA porque nós geramos os dados: "ajustar" aqui é
  // devolver as constantes, e o argumento é ignorado de propósito.
  // É o privilégio que o dataset real não tem.
  fitScaler: () => null,
  toVector: (customer) => toFeatureVector(customer),

  sampleCustomer: {
    income: 3500,
    debtRatio: 0.72,
    latePayments: 3,
    creditUtilization: 0.88,
  },
};

const GERMAN_SOURCE = {
  id: 'german',
  label: 'German Credit — UCI/Statlog (Hofmann, 1994)',
  csvPath: GERMAN_CSV_PATH,
  columns: GERMAN_COLUMNS,
  precision: GERMAN_PRECISION,
  featureNames: GERMAN_FEATURES,

  // Dado real não se "gera": ou já está em disco, ou precisa ser baixado.
  // É a diferença mais concreta entre as duas fontes.
  ensure: (filePath = GERMAN_CSV_PATH) => {
    if (!fs.existsSync(filePath)) {
      throw new Error([
        `Dataset real não encontrado em ${filePath}.`,
        'Rode `npm run fetch:german` para baixá-lo da UCI.',
      ].join('\n'));
    }

    return { path: filePath, created: false };
  },
  read: () => readCustomersCsv(GERMAN_CSV_PATH),

  fitScaler: (customers) => fitMinMaxScaler(customers, GERMAN_FEATURES),
  toVector: (customer, scaler) => applyMinMaxScaler(scaler, customer),

  // Perfil desfavorável em todas as frentes: conta corrente no vermelho,
  // prazo longo, histórico curto, sem poupança e prestação no teto.
  sampleCustomer: {
    checkingStatus: 0,
    durationMonths: 48,
    creditHistory: 1,
    creditAmount: 9000,
    savingsStatus: 0,
    employmentYears: 1,
    installmentRate: 4,
    age: 24,
  },
};

const SOURCES = {
  [SYNTHETIC_SOURCE.id]: SYNTHETIC_SOURCE,
  [GERMAN_SOURCE.id]: GERMAN_SOURCE,
};

// O dataset real é o padrão: é ele que mostra o laboratório sob condições
// honestas. O sintético continua a um argumento de distância, porque
// comparar os dois é metade da lição.
const DEFAULT_SOURCE_ID = GERMAN_SOURCE.id;

const resolveSourceId = (argv = []) => {
  const flag = argv.find((argument) => argument.startsWith('--source='));
  const id = flag ? flag.slice('--source='.length) : DEFAULT_SOURCE_ID;

  if (!SOURCES[id]) {
    throw new Error(
      `Fonte desconhecida: ${id}. Use uma de: ${Object.keys(SOURCES).join(', ')}.`,
    );
  }

  return id;
};

// --------------------------------------------------
// 7. Separar treino e teste
// --------------------------------------------------
// Gerador pseudoaleatório COM SEMENTE (mulberry32). `Math.random()` não
// aceita semente: com ele cada execução embaralharia diferente, e aí
// nenhum resultado deste projeto seria reproduzível.
const createRandom = (seed) => {
  let state = seed;

  return () => {
    state = (state + 0x6D2B79F5) | 0;

    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;

    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
};

// Fisher-Yates sobre uma CÓPIA: a ordem original do arquivo é preservada.
const shuffle = (items, random) => {
  const copy = [...items];

  for (let index = copy.length - 1; index > 0; index -= 1) {
    const target = Math.floor(random() * (index + 1));

    [copy[index], copy[target]] = [copy[target], copy[index]];
  }

  return copy;
};

// Separa os clientes ANTES de normalizar. A ordem importa: normalizar
// primeiro faria as estatísticas do teste vazarem para dentro do treino.
//
// Com dado sintético embaralhar é indiferente — cada linha é sorteada de
// forma independente. Com dado real não: um arquivo pode chegar ordenado
// por data, por agência ou pela própria classe, e um corte cru no meio
// separaria dois conjuntos que não representam a mesma população.
const splitCustomers = (customers, trainRatio = 0.8) => {
  const trainSize = Math.floor(customers.length * trainRatio);

  return {
    trainCustomers: customers.slice(0, trainSize),
    testCustomers: customers.slice(trainSize),
  };
};

// Variante que opera sobre features JÁ normalizadas. Continua servindo o
// caminho sintético e os testes; o fluxo principal usa `splitCustomers`.
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
// 8. Criar a MLP
// --------------------------------------------------
// A compilação fica isolada porque é usada em dois momentos:
// ao montar o modelo do zero e ao recompilar um modelo carregado
// do disco que tenha sido salvo sem o estado do otimizador.
const compileModel = (model) => {
  model.compile({
    optimizer: tf.train.adam(0.001),
    loss: 'binaryCrossentropy',
    metrics: ['accuracy'],
  });

  return model;
};

// O número de entradas vem da fonte de dados: o sintético tem 4 features,
// o German Credit tem 8. Era a única parte da rede que precisava saber
// qual dataset está em uso.
const buildModel = (inputSize = 4) => {
  const model = tf.sequential();

  model.add(tf.layers.dense({
    inputShape: [inputSize],
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

  return compileModel(model);
};

// --------------------------------------------------
// 9. Persistência
// --------------------------------------------------
// O tfjs-node registra o esquema `file://`. Salvar em `./model`
// gera dois arquivos:
//   model.json  → topologia + metadados (e o training config)
//   weights.bin → os pesos em binário
const toFileUrl = (dir) => `file://${path.resolve(dir)}`;

// `includeOptimizer: true` salva também o estado do Adam.
// Sem ele o modelo recarregado vem SEM compilação: dá para prever,
// mas não para avaliar ou continuar o treino sem recompilar.
const saveModel = (model, dir = MODEL_DIR) =>
  model.save(toFileUrl(dir), { includeOptimizer: true });

const loadModel = async (dir = MODEL_DIR) => {
  const model = await tf.loadLayersModel(`${toFileUrl(dir)}/model.json`);

  // Rede de segurança: se o modelo veio sem otimizador, recompila com
  // exatamente a mesma configuração usada no treino.
  if (!model.optimizer) {
    compileModel(model);
  }

  return model;
};

// --------------------------------------------------
// 10. Inferência
// --------------------------------------------------
// Encapsula o ciclo tensor → predição → dispose para que nenhum
// tensor intermediário escape em quem chama.
// `toVector` é injetável porque cada fonte normaliza de um jeito — e o
// dataset real só sabe normalizar depois de ver o conjunto de treino.
const predictRisk = (model, customer, toVector = toFeatureVector) => {
  const input = tf.tensor2d([toVector(customer)]);
  const prediction = model.predict(input);
  const probability = prediction.dataSync()[0];

  tf.dispose([input, prediction]);

  return probability;
};

// --------------------------------------------------
// 11. Matriz de confusão
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

// Tabela de texto com colunas alinhadas: primeira coluna à esquerda,
// as demais à direita. Usada pela matriz e pela comparação de limiares.
const formatTable = (headers, rows) => {
  const widths = headers.map((header, column) => Math.max(
    header.length,
    ...rows.map((row) => row[column].length),
  ));

  const line = (cells) => cells
    .map((cell, column) => (column === 0
      ? cell.padEnd(widths[column])
      : cell.padStart(widths[column])))
    .join(' | ');

  return [
    line(headers),
    widths.map((width) => '-'.repeat(width)).join('-+-'),
    ...rows.map(line),
  ].join('\n');
};

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

// --------------------------------------------------
// 12. Precision, recall e F1-score
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

// --------------------------------------------------
// 13. Curva ROC e AUC
// --------------------------------------------------
// Matriz, precision e recall descrevem UM limiar. A ROC descreve TODOS:
// cada ponto é um corte possível, com sua taxa de acerto (TPR) e seu
// custo em alarmes falsos (FPR).
//
// A AUC — área sob essa curva — resume a curva inteira em um número e,
// por isso, NÃO depende do limiar. Ela mede a capacidade de ORDENAR:
// é a probabilidade de um cliente de alto risco receber score maior que
// um de baixo risco. 0.5 = moeda; 1.0 = separação perfeita.
const computeRocCurve = (model, xTest, yTest) => {
  const { scores, actuals } = tf.tidy(() => ({
    scores: Array.from(model.predict(xTest).reshape([-1]).dataSync()),
    actuals: Array.from(yTest.reshape([-1]).dataSync()),
  }));

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

// --------------------------------------------------
// 14. Escolher o limiar a partir da curva
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

// Acurácia de quem sempre chuta a classe majoritária, sem olhar para
// nenhuma feature. É o PISO: um modelo que não supera este número não
// aprendeu nada aproveitável, por mais alta que a acurácia pareça.
const majorityBaseline = (labels) => {
  const positives = labels.filter(([risk]) => risk === 1).length;

  return Math.max(positives, labels.length - positives) / labels.length;
};

const evaluateModel = (model, xTest, yTest) => {
  const [lossTensor, accuracyTensor] = model.evaluate(xTest, yTest);
  const loss = lossTensor.dataSync()[0];
  const accuracy = accuracyTensor.dataSync()[0];

  tf.dispose([lossTensor, accuracyTensor]);

  return { loss, accuracy };
};

// --------------------------------------------------
// 15. Treinar, avaliar, salvar, recarregar e prever
// --------------------------------------------------
const main = async (sourceId = DEFAULT_SOURCE_ID) => {
  const source = SOURCES[sourceId];

  // O sintético se gera se faltar; o real só confere que está em disco.
  source.ensure();

  console.log('Fonte:', source.label);
  console.log('Arquivo:', source.csvPath);

  // Embaralhar ANTES de separar. O arquivo real chega na ordem em que foi
  // coletado, e essa ordem não é aleatória — pode concentrar um perfil de
  // cliente em um trecho e outro perfil no resto.
  const customers = shuffle(await source.read(), createRandom(SHUFFLE_SEED));

  console.log('Clientes lidos:', customers.length);
  console.log('Features:', source.featureNames.join(', '));

  const { trainCustomers, testCustomers } = splitCustomers(customers);

  // A escala é medida SÓ no treino e aplicada aos dois conjuntos.
  // O teste é tratado como dado que ainda não existia quando o
  // pré-processamento foi definido — porque é assim que será em produção.
  const scaler = source.fitScaler(trainCustomers);
  const toVector = (customer) => source.toVector(customer, scaler);

  const trainLabels = trainCustomers.map(({ risk }) => [risk]);
  const testLabels = testCustomers.map(({ risk }) => [risk]);

  const xTrain = tf.tensor2d(trainCustomers.map(toVector));
  const yTrain = tf.tensor2d(trainLabels);
  const xTest = tf.tensor2d(testCustomers.map(toVector));
  const yTest = tf.tensor2d(testLabels);

  const model = buildModel(source.featureNames.length);
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

  const { loss: testLoss, accuracy: testAccuracy } =
    evaluateModel(model, xTest, yTest);

  // A acurácia sozinha não diz se o modelo presta. Ao lado do piso da
  // classe majoritária ela passa a dizer: a distância entre os dois
  // números é tudo que o treino realmente acrescentou.
  const baseline = majorityBaseline(testLabels);

  console.log('Test loss:', testLoss.toFixed(4));
  console.log('Test accuracy:', testAccuracy.toFixed(4));
  console.log('Baseline (classe majoritária):', baseline.toFixed(4));

  const confusion = computeConfusionMatrix(model, xTest, yTest);

  console.log('\nMatriz de confusão (limiar', `${DECISION_THRESHOLD}):`);
  console.log(formatConfusionMatrix(confusion));

  const metrics = computeMetrics(confusion);

  console.log('');
  console.log(formatMetrics(metrics));

  // O limiar atual é apenas UM ponto da curva; a AUC resume todos.
  const { points, auc, positives, negatives } = computeRocCurve(model, xTest, yTest);
  const operatingPoint = {
    fpr: safeDivide(
      confusion.falsePositives,
      confusion.falsePositives + confusion.trueNegatives,
    ),
    tpr: metrics.recall,
  };

  console.log('\nCurva ROC (O = limiar', `${DECISION_THRESHOLD}, . = aleatório):`);
  console.log(formatRocCurve(points, { mark: operatingPoint }));
  console.log('AUC:', auc.toFixed(4));

  // ------------------------------------------------
  // Escolher o corte em vez de herdá-lo
  // ------------------------------------------------
  const costs = {
    falsePositive: FALSE_POSITIVE_COST,
    falseNegative: FALSE_NEGATIVE_COST,
  };
  const roc = { points, auc, positives, negatives };

  const candidates = [
    {
      label: 'Padrão (0.5)',
      point: scorePoint(
        { ...operatingPoint, threshold: DECISION_THRESHOLD },
        roc,
        costs,
      ),
    },
    {
      label: 'Youden (max J)',
      point: scorePoint(chooseThresholdByYouden(roc), roc, costs),
    },
    {
      label: 'Menor custo',
      point: chooseThresholdByCost(roc, costs),
    },
  ];

  console.log(
    `\nAjuste do limiar (FP custa ${FALSE_POSITIVE_COST}, FN custa ${FALSE_NEGATIVE_COST}):`,
  );
  console.log(formatThresholdComparison(candidates));

  const chosen = candidates[2].point;

  console.log('\nMatriz no limiar escolhido', `(${formatThreshold(chosen.threshold)}):`);
  console.log(formatConfusionMatrix(
    computeConfusionMatrix(model, xTest, yTest, chosen.threshold),
  ));
  console.log('');

  // ------------------------------------------------
  // Novo cliente
  // ------------------------------------------------
  // O cliente de exemplo vem da fonte: as features de um dataset não
  // fazem sentido nenhum no outro.
  const probability = predictRisk(model, source.sampleCustomer, toVector);

  console.log('Probabilidade de alto risco:', probability.toFixed(4));
  console.log('Classificação:', classify(probability));

  // ------------------------------------------------
  // Salvar em disco e descartar o modelo da memória
  // ------------------------------------------------
  await saveModel(model);
  console.log('Modelo salvo em:', MODEL_DIR);

  model.dispose();

  // ------------------------------------------------
  // Recarregar e conferir que nada se perdeu
  // ------------------------------------------------
  const loadedModel = await loadModel();

  const { loss: loadedLoss, accuracy: loadedAccuracy } =
    evaluateModel(loadedModel, xTest, yTest);
  const loadedProbability = predictRisk(loadedModel, source.sampleCustomer, toVector);

  console.log('Modelo recarregado — test loss:', loadedLoss.toFixed(4));
  console.log('Modelo recarregado — test accuracy:', loadedAccuracy.toFixed(4));
  console.log(
    'Modelo recarregado — probabilidade:',
    loadedProbability.toFixed(4),
  );
  console.log(
    'Mesma predição do modelo original?',
    loadedProbability === probability ? 'sim' : 'não',
  );

  // ------------------------------------------------
  // Limpeza de memória
  // ------------------------------------------------
  tf.dispose([xTrain, yTrain, xTest, yTest]);
  loadedModel.dispose();
};

// Só executa quando chamado direto (node index.js).
// Ao ser importado pelos testes, apenas expõe as funções.
if (require.main === module) {
  // Envolver em uma função async faz o erro SÍNCRONO de `resolveSourceId`
  // virar rejeição e cair no mesmo `.catch` dos erros assíncronos. Sem
  // isso, um argumento inválido imprimiria stack trace no lugar da
  // mensagem que diz quais fontes existem.
  const run = async () => main(resolveSourceId(process.argv.slice(2)));

  run().catch((error) => {
    console.error(`\n${error.message}\n`);
    process.exitCode = 1;
  });
}


module.exports = {
  INCOME_MIN,
  INCOME_RANGE,
  MAX_LATE_PAYMENTS,
  RISK_RULE_THRESHOLD,
  DECISION_THRESHOLD,
  MODEL_DIR,
  CSV_PATH,
  CSV_COLUMNS,
  CSV_LABEL_COLUMN,
  CSV_PRECISION,
  GERMAN_CSV_PATH,
  GERMAN_SOURCE_URL,
  GERMAN_CODES,
  GERMAN_FEATURES,
  GERMAN_COLUMNS,
  GERMAN_PRECISION,
  SHUFFLE_SEED,
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
  majorityBaseline,
  readCustomersCsv,
  loadDatasetCsv,
  splitDataset,
  compileModel,
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
  main,
};
