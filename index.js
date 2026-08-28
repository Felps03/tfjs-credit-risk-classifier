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

// Configuração do gerador sintético. Um dataset de brinquedo limpo e
// balanceado ensina o pipeline, mas esconde os dois problemas que
// aparecem em quase todo projeto real — e que só se enfrenta medindo:
//
//   ruído          o banco não observa a verdade, observa uma MEDIDA dela.
//                  Renda declarada, atraso registrado com erro, utilização
//                  medida no dia errado. Isso cria um teto: acima dele
//                  nenhum modelo chega, porque a informação não existe.
//   desbalanceamento  inadimplente é minoria. Quando 85% dos clientes são
//                  bons, "chutar bom para todo mundo" já acerta 85% — e a
//                  acurácia deixa de significar qualquer coisa.
const SYNTHETIC_SEED = 7;
const SYNTHETIC_TOTAL = 1200;

// Fração de inadimplentes ALVO. O limiar da regra deixa de ser um número
// fixo e passa a ser o quantil que produz exatamente esta taxa.
const SYNTHETIC_POSITIVE_RATE = 0.15;

// Desvio padrão do ruído de medição, como fração da faixa de cada coluna.
// 0.05 sobre a renda = R$ 650 de erro típico.
const SYNTHETIC_FEATURE_NOISE = 0.05;

// Fração de rótulos trocados: o desfecho registrado está errado. Um bom
// pagador que ficou desempregado, uma baixa lançada na conta errada.
const SYNTHETIC_LABEL_NOISE = 0.02;

// Limiar de decisão do classificador: escolha de negócio, não do modelo.
const DECISION_THRESHOLD = 0.5;

// Regularização: os dois freios contra decorar o treino.
//
// O modelo real tem 1.073 parâmetros para 640 linhas de treino efetivo.
// Com mais parâmetros do que dados, decorar é o caminho mais barato para
// baixar o erro — e a conta chega no teste, não no treino. Os dois freios
// atacam o mesmo problema por caminhos diferentes:
//
//   L2       soma λ·Σw² à loss do treino. Peso grande passa a custar caro,
//            então a rede só paga por um se ele render redução de erro
//            suficiente. Achata os pesos; não zera nenhum.
//   dropout  desliga uma fração das unidades a cada passo do treino.
//            Nenhuma unidade pode contar com uma vizinha específica, então
//            a informação precisa ficar distribuída em vez de concentrada.
//
// Os valores não vieram de convenção: saíram de uma varredura de 6 × 5
// combinações sobre 15 divisões, medindo a diferença treino−teste. O
// README registra a grade inteira.
const L2_LAMBDA = 0.003;
const DROPOUT_RATE = 0.2;

// A arquitetura padrão: duas camadas ocultas, afunilando.
//
// Ela ficou onde estava por uma razão medida, não por convenção. Oito
// topologias foram comparadas por validação cruzada, da regressão
// logística sem camada oculta nenhuma até uma rede de 15.745
// parâmetros: entre as SETE redes, nenhuma se distingue de nenhuma. A
// documentação traz a tabela.
//
// Trocar o padrão por outra dessas sete seria trocar por ruído — que é
// exatamente o erro contra o qual o resto do projeto argumenta.
const HIDDEN_UNITS = [16, 8];

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

// Colunas com magnitude de verdade: um valor maior significa "mais".
// Só estas passam pelo min-max — as outras não têm o que escalar.
const GERMAN_NUMERIC = [
  'durationMonths',
  'creditAmount',
  'installmentRate',
  'residenceSince',
  'age',
  'existingCredits',
  'dependents',
];

// Colunas qualitativas, com os códigos na ordem documentada pela UCI.
//
// ATENÇÃO ao que o inteiro significa daqui em diante: ele é um CÓDIGO,
// não uma quantidade. `purpose = 3` não é o triplo de `purpose = 1`; são
// "rádio/TV" e "carro usado". É exatamente por isso que estas colunas
// viram one-hot antes de entrar na rede.
const GERMAN_CATEGORICAL = {
  checkingStatus: ['A11', 'A12', 'A13', 'A14'],
  creditHistory: ['A30', 'A31', 'A32', 'A33', 'A34'],
  purpose: ['A40', 'A41', 'A42', 'A43', 'A44', 'A45', 'A46', 'A48', 'A49', 'A410'],
  savingsStatus: ['A61', 'A62', 'A63', 'A64', 'A65'],
  employmentYears: ['A71', 'A72', 'A73', 'A74', 'A75'],
  otherDebtors: ['A101', 'A102', 'A103'],
  property: ['A121', 'A122', 'A123', 'A124'],
  otherInstallments: ['A141', 'A142', 'A143'],
  housing: ['A151', 'A152', 'A153'],
  job: ['A171', 'A172', 'A173', 'A174'],
  telephone: ['A191', 'A192'],
  foreignWorker: ['A201', 'A202'],
};

// Atributo 9 do arquivo original: estado civil e SEXO.
//
// Ele é lido e vai para o CSV, mas NÃO entra no modelo. Usar sexo para
// negar crédito é discriminação e é ilegal em vários países. A coluna
// serve a um propósito diferente: AUDITAR as decisões do modelo depois
// que ele já decidiu sem ela. Veja `auditByGroup`.
//
// A95 (mulher solteira) não aparece nos 1000 registros; só A91–A94.
const GERMAN_AUDIT_COLUMN = 'personalStatus';
const GERMAN_AUDIT_CODES = ['A91', 'A92', 'A93', 'A94'];

// A92 é o único código feminino presente no arquivo. Os outros três são
// recortes de estado civil masculino — ou seja, dá para recuperar o sexo
// desta coluna, e é isso que a auditoria usa.
const FEMALE_CODE = 'A92';

// Uma linha do CSV: as numéricas, os códigos das qualitativas, a coluna
// de auditoria e o rótulo. 21 colunas — 19 delas viram features.
const GERMAN_COLUMNS = [
  ...GERMAN_NUMERIC,
  ...Object.keys(GERMAN_CATEGORICAL),
  GERMAN_AUDIT_COLUMN,
  CSV_LABEL_COLUMN,
];

// Todos os valores gravados são inteiros: meses, marcos, contagens,
// idades e códigos de categoria.
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

// Gerador pseudoaleatório COM SEMENTE (mulberry32). `Math.random()` não
// aceita semente: com ele cada execução geraria e embaralharia diferente,
// e aí nenhum resultado deste projeto seria reproduzível.
const createRandom = (seed) => {
  let state = seed;

  return () => {
    state = (state + 0x6D2B79F5) | 0;

    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;

    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
};

// Box-Muller: transforma dois sorteios uniformes em um valor normal de
// média 0 e desvio 1. Ruído de medição é normal por um bom motivo — ele
// costuma ser a soma de muitos erros pequenos e independentes, e o
// Teorema Central do Limite diz que essa soma tende à normal.
const createGaussian = (random) => () => {
  const uniform = 1 - random();          // (0, 1]: evita log(0) = -Infinity
  const angle = 2 * Math.PI * random();

  return Math.sqrt(-2 * Math.log(uniform)) * Math.cos(angle);
};

const clamp = (value, lowest, highest) =>
  Math.min(Math.max(value, lowest), highest);

// Valor abaixo do qual está `fraction` dos dados. É o que permite pedir
// "15% de inadimplentes" em vez de chutar um limiar até a conta fechar.
const quantile = (values, fraction) => {
  const sorted = [...values].sort((a, b) => a - b);
  const index = clamp(Math.floor(fraction * sorted.length), 0, sorted.length - 1);

  return sorted[index];
};

// Faixa válida de cada coluna. Serve para duas coisas: dar escala ao
// ruído (5% da faixa) e impedir que ele produza absurdos — renda
// negativa, utilização de 130%.
const SYNTHETIC_BOUNDS = {
  income: [INCOME_MIN, INCOME_MIN + INCOME_RANGE],
  debtRatio: [0, 1],
  latePayments: [0, MAX_LATE_PAYMENTS],
  creditUtilization: [0, 1],
};

// Regra que gera os rótulos. O modelo NÃO recebe esta fórmula: ele
// precisa reconstruí-la a partir dos exemplos.
//
// Importante: ela é aplicada ao cliente VERDADEIRO, não ao medido. É
// isso que torna o ruído irredutível — a resposta certa depende de um
// valor que nunca chega ao arquivo.
const riskScore = (customer) => {
  const [income, debtRatio, latePayments, creditUtilization] =
    toFeatureVector(customer);

  return (
    1.4 * debtRatio +
    1.2 * latePayments +
    1.0 * creditUtilization -
    0.8 * income
  );
};

// Verdade → medida. Cada coluna ganha um desvio normal proporcional à
// sua faixa, e volta para dentro dos limites. `latePayments` é contagem:
// arredonda, porque "2,3 atrasos" não existe no sistema do banco.
const measureCustomer = (customer, noise, gaussian) => {
  const measured = Object.fromEntries(
    Object.entries(SYNTHETIC_BOUNDS).map(([column, [lowest, highest]]) => {
      const drift = gaussian() * noise * (highest - lowest);

      return [column, clamp(customer[column] + drift, lowest, highest)];
    }),
  );

  return { ...measured, latePayments: Math.round(measured.latePayments) };
};

// Clientes em unidades BRUTAS (reais, contagem, percentual), do jeito
// que sairiam de um banco de dados — é isso que vai para o CSV.
//
// Os três parâmetros de ruído e desbalanceamento são argumentos, não
// constantes escondidas: é o que permite ao teste gerar uma versão limpa
// e comparar, e a você mexer em um de cada vez para ver o efeito.
const createCustomers = (total = SYNTHETIC_TOTAL, options = {}) => {
  const {
    seed = SYNTHETIC_SEED,
    positiveRate = SYNTHETIC_POSITIVE_RATE,
    featureNoise = SYNTHETIC_FEATURE_NOISE,
    labelNoise = SYNTHETIC_LABEL_NOISE,
  } = options;

  const random = createRandom(seed);
  const gaussian = createGaussian(random);

  // 1. O estado VERDADEIRO do cliente. No mundo real ele existe e ninguém
  //    o enxerga; aqui ele existe, é usado para rotular, e é descartado.
  const truths = Array.from({ length: total }, () => ({
    income: INCOME_MIN + random() * INCOME_RANGE,
    debtRatio: random(),
    latePayments: Math.floor(random() * (MAX_LATE_PAYMENTS + 1)),
    creditUtilization: random(),
  }));

  // 2. O corte que produz a taxa de inadimplência pedida. Desbalancear é
  //    só empurrar este limiar para cima: quanto mais alto, mais raro o
  //    positivo — e mais inútil a acurácia como métrica.
  const scores = truths.map(riskScore);
  const cut = quantile(scores, 1 - positiveRate);

  // 3. O que vai para o arquivo é a MEDIDA e o desfecho REGISTRADO.
  return truths.map((truth, index) => {
    const label = scores[index] > cut ? 1 : 0;
    const mistaken = random() < labelNoise;

    return {
      ...measureCustomer(truth, featureNoise, gaussian),
      risk: mistaken ? 1 - label : label,
    };
  });
};

// Clientes brutos → matrizes que o TensorFlow consome. Um só caminho de
// normalização, seja o dado gerado em memória ou lido do CSV.
const toDataset = (customers) => ({
  features: customers.map(toFeatureVector),
  labels: customers.map(({ risk }) => [risk]),
});

const createDataset = (total = SYNTHETIC_TOTAL, options) =>
  toDataset(createCustomers(total, options));

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
const ensureCsv = (filePath = CSV_PATH, total = SYNTHETIC_TOTAL) => {
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
//
// As qualitativas viram o ÍNDICE do código, não uma nota: quem transforma
// isso em features é `toGermanVector`, com one-hot.
const GERMAN_SOURCE_ATTRIBUTES = {
  durationMonths: 'Attribute2',
  creditAmount: 'Attribute5',
  installmentRate: 'Attribute8',
  residenceSince: 'Attribute11',
  age: 'Attribute13',
  existingCredits: 'Attribute16',
  dependents: 'Attribute18',

  checkingStatus: 'Attribute1',
  creditHistory: 'Attribute3',
  purpose: 'Attribute4',
  savingsStatus: 'Attribute6',
  employmentYears: 'Attribute7',
  otherDebtors: 'Attribute10',
  property: 'Attribute12',
  otherInstallments: 'Attribute14',
  housing: 'Attribute15',
  job: 'Attribute17',
  telephone: 'Attribute19',
  foreignWorker: 'Attribute20',

  [GERMAN_AUDIT_COLUMN]: 'Attribute9',
};

const toGermanCustomer = (row) => {
  const customer = {};

  GERMAN_NUMERIC.forEach((field) => {
    customer[field] = Number(row[GERMAN_SOURCE_ATTRIBUTES[field]]);
  });

  Object.entries(GERMAN_CATEGORICAL).forEach(([field, codes]) => {
    customer[field] = toOrdinal(codes, row[GERMAN_SOURCE_ATTRIBUTES[field]]);
  });

  customer[GERMAN_AUDIT_COLUMN] = toOrdinal(
    GERMAN_AUDIT_CODES,
    row[GERMAN_SOURCE_ATTRIBUTES[GERMAN_AUDIT_COLUMN]],
  );

  customer[CSV_LABEL_COLUMN] = Number(row.class) === 2 ? 1 : 0;

  return customer;
};

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
// 6. Codificação das colunas qualitativas
// --------------------------------------------------
// Codificar categoria como número inteiro (ordinal) é conveniente e, na
// maioria das colunas, é uma MENTIRA: diz à rede que `purpose = 3` fica
// entre `2` e `4`, quando "rádio/TV", "eletrodoméstico" e "reparos" não
// têm ordem nenhuma entre si.
//
// One-hot desfaz essa suposição. Cada categoria vira uma coluna própria,
// que vale 1 quando é aquela e 0 nas outras — nenhuma fica "maior" que
// as demais, e a rede aprende um peso independente para cada uma.
//
//   purpose = 3  →  ordinal: [0.333]
//                   one-hot: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
const oneHotEncode = (size, index) =>
  Array.from({ length: size }, (unused, position) => (position === index ? 1 : 0));

// Codificação ordinal normalizada, mantida para comparação. Divide pelo
// maior índice para cair em [0, 1], igual às numéricas.
const ordinalEncode = (size, index) => [size > 1 ? index / (size - 1) : 0];

const ENCODERS = {
  onehot: oneHotEncode,
  ordinal: ordinalEncode,
};

// Nome de cada posição do vetor final. Com 57 entradas, saber qual coluna
// é qual deixa de ser óbvio — e o nome é o que liga um peso da rede de
// volta a um fato sobre o cliente.
const germanFeatureNames = (encoding) => [
  ...GERMAN_NUMERIC,
  ...Object.entries(GERMAN_CATEGORICAL).flatMap(([field, codes]) => (
    encoding === 'onehot'
      ? codes.map((code) => `${field}=${code}`)
      : [field]
  )),
];

// Numéricas escaladas + qualitativas codificadas, sempre nessa ordem.
// A ordem precisa ser estável: é ela que casa cada valor com a entrada
// correspondente da rede, no treino e na inferência.
const toGermanVector = (customer, scaler, encoding) => {
  const encode = ENCODERS[encoding];

  return [
    ...applyMinMaxScaler(scaler, customer),
    ...Object.entries(GERMAN_CATEGORICAL).flatMap(([field, codes]) =>
      encode(codes.length, customer[field])),
  ];
};

// --------------------------------------------------
// 7. Auditoria de disparidade
// --------------------------------------------------
// O modelo nunca recebe a coluna de sexo. Isso NÃO garante que ele decida
// igual para os dois grupos: as outras colunas carregam o sinal por
// tabela, e o modelo o reconstrói sem nunca ver o atributo protegido.
//
// Por isso a coluna fica no CSV mesmo fora do modelo — para conferir,
// depois da decisão tomada, se ela caiu diferente entre os grupos.
const isFemale = (customer) =>
  customer[GERMAN_AUDIT_COLUMN] === GERMAN_AUDIT_CODES.indexOf(FEMALE_CODE);

// Para cada grupo: quantos são, qual a taxa REAL de inadimplência e o que
// o modelo decidiu. Separar as duas últimas é o ponto — diferença na
// decisão só é injustiça se não vier de diferença nos dados.
const summarizeGroup = (rows, threshold) => {
  const flagged = rows.filter(({ score }) => score >= threshold).length;
  const positives = rows.filter(({ risk }) => risk === 1);
  const missed = positives.filter(({ score }) => score < threshold).length;

  return {
    total: rows.length,
    baseRate: safeDivide(positives.length, rows.length),
    flaggedRate: safeDivide(flagged, rows.length),
    falseNegativeRate: safeDivide(missed, positives.length),
  };
};

// Regra dos quatro quintos (EEOC): a razão entre as taxas de APROVAÇÃO dos
// dois grupos. Abaixo de 0.8 é o patamar que, nos EUA, liga o alerta de
// impacto desigual. Não é lei brasileira nem prova de discriminação — é um
// termômetro consagrado, e é assim que entra aqui.
//
// Os dois casos degenerados precisam de cuidado. Taxas IGUAIS são paridade,
// inclusive quando as duas são zero: reprovar todo mundo nos dois grupos é
// tratamento idêntico, e um `0 / 0` virando 0 acusaria disparidade máxima
// onde não há nenhuma. Já aprovar em um grupo e em nenhum do outro é
// disparidade sem limite — daí o `Infinity`, que é literalmente o caso.
const approvalRatio = (women, men) => {
  const approvedWomen = 1 - women.flaggedRate;
  const approvedMen = 1 - men.flaggedRate;

  if (approvedWomen === approvedMen) {
    return 1;
  }

  return approvedMen === 0 ? Infinity : approvedWomen / approvedMen;
};

// Uma linha por cliente com as três coisas que a auditoria precisa: o que
// aconteceu de verdade, o que o modelo achou e a qual grupo a pessoa
// pertence. A mitigação usa exatamente as mesmas linhas.
const toAuditRows = (customers, scores) => customers.map((customer, index) => ({
  risk: customer[CSV_LABEL_COLUMN],
  score: scores[index],
  female: isFemale(customer),
}));

// O limiar pode chegar de duas formas: um número, que vale para todo
// mundo, ou um par `{ women, men }`, que vale um para cada grupo. A
// segunda forma é a mitigação da seção 7.1 — e é justamente por isso que
// a auditoria aceita as duas: para medir a decisão mitigada com a mesma
// régua que mediu a original.
const thresholdFor = (threshold, female) => (
  typeof threshold === 'number'
    ? threshold
    : threshold[female ? 'women' : 'men']
);

const auditByGroup = (customers, scores, threshold = DECISION_THRESHOLD) => {
  const rows = toAuditRows(customers, scores);

  const women = summarizeGroup(
    rows.filter(({ female }) => female),
    thresholdFor(threshold, true),
  );
  const men = summarizeGroup(
    rows.filter(({ female }) => !female),
    thresholdFor(threshold, false),
  );

  return {
    women,
    men,
    approvalRatio: approvalRatio(women, men),
    thresholds: {
      women: thresholdFor(threshold, true),
      men: thresholdFor(threshold, false),
    },
  };
};

const formatAudit = ({ women, men, approvalRatio }) => [
  formatTable(
    ['Grupo', 'N', 'Inadimp. real', 'Marcados ALTO', 'FN não pegos'],
    [
      ['Mulheres', String(women.total), `${(100 * women.baseRate).toFixed(1)}%`,
        `${(100 * women.flaggedRate).toFixed(1)}%`, `${(100 * women.falseNegativeRate).toFixed(1)}%`],
      ['Homens', String(men.total), `${(100 * men.baseRate).toFixed(1)}%`,
        `${(100 * men.flaggedRate).toFixed(1)}%`, `${(100 * men.falseNegativeRate).toFixed(1)}%`],
    ],
  ),
  '',
  `Razão de aprovação (regra dos 4/5): `
    + `${Number.isFinite(approvalRatio) ? approvalRatio.toFixed(3) : 'infinita'}`
    + `${approvalRatio < 0.8 ? '  <- abaixo de 0.80' : ''}`,
].join('\n');

// --------------------------------------------------
// 7.1 Mitigação: mexer na decisão, não só no relatório
// --------------------------------------------------
// Medir a disparidade e parar aí deixa o problema documentado e intacto.
// O passo seguinte é agir sobre ele, e o lugar mais barato de agir é
// DEPOIS do modelo: no limiar. Nenhum peso muda, nenhum treino é
// refeito — os scores continuam exatamente os mesmos, e o que se move é
// onde cada grupo é cortado.
//
// O preço é explícito e desconfortável: para igualar as taxas de
// aprovação é preciso LER o atributo protegido na hora de decidir — a
// mesma coluna que o modelo foi proibido de ver. Não há como fugir
// disso: corrigir uma diferença entre grupos exige saber o grupo. Por
// isso a mitigação aqui é uma opção de linha de comando e não o padrão.
// Ligá-la é uma decisão de política, não um detalhe de engenharia: em
// vários países usar sexo na decisão é ilegal mesmo quando a intenção
// declarada é reduzir a diferença.

// Limiar que marca a fração `rate` dos scores recebidos.
//
// Ordena do maior para o menor e corta no ponto médio entre a última
// linha marcada e a primeira que fica de fora. O meio do intervalo é
// preferível a qualquer uma das pontas: um score novo que caia
// exatamente sobre a fronteira não decide a própria sorte por um empate.
//
// Empate de verdade não tem saída: se os dois vizinhos têm o mesmo
// score, o ponto médio é esse score, e o `>=` marca os dois. A fração
// pedida passa a ser um piso, não uma promessa.
const rateThreshold = (scores, rate) => {
  const sorted = [...scores].sort((first, second) => second - first);
  const marked = Math.round(rate * sorted.length);

  // Marcar ninguém e marcar todo mundo são respostas legítimas, e as
  // duas precisam de um limiar que exista de fato. `Infinity` não é
  // alcançado por nenhuma probabilidade; `0` é alcançado por todas.
  if (marked <= 0) {
    return Infinity;
  }

  if (marked >= sorted.length) {
    return 0;
  }

  return (sorted[marked - 1] + sorted[marked]) / 2;
};

// Calibra um limiar por grupo de forma que os dois marquem a MESMA
// fração — a taxa que o limiar único produz no conjunto inteiro.
//
// Os clientes passados aqui precisam ser os de TREINO. Calibrar no mesmo
// conjunto que será auditado devolve paridade por construção: a razão
// sai exatamente 1 — há um teste que fixa isso — e não significa nada,
// porque a régua foi ajustada às respostas da prova. Calibrando fora, e
// medindo em 1000 clientes por validação cruzada, o número honesto é
// 1.0140 ± 0.0129 contra 0.8258 ± 0.0172 sem mitigação nenhuma.
const fitGroupThresholds = (customers, scores, threshold = DECISION_THRESHOLD) => {
  const rows = toAuditRows(customers, scores);
  const rate = safeDivide(
    rows.filter(({ score }) => score >= threshold).length,
    rows.length,
  );

  const thresholdOf = (female) => rateThreshold(
    rows.filter((row) => row.female === female).map(({ score }) => score),
    rate,
  );

  return { women: thresholdOf(true), men: thresholdOf(false) };
};

// O que uma política de limiares custa em decisão. A auditoria diz se a
// diferença encolheu; isto diz quanto se pagou por isso — e as duas
// respostas precisam aparecer juntas, senão a mitigação parece de graça.
const summarizeDecisions = (customers, scores, threshold, costs = {
  falsePositive: FALSE_POSITIVE_COST,
  falseNegative: FALSE_NEGATIVE_COST,
}) => {
  const decided = toAuditRows(customers, scores).map((row) => ({
    ...row,
    flagged: row.score >= thresholdFor(threshold, row.female),
  }));

  const falsePositives = decided
    .filter(({ risk, flagged }) => flagged && risk === 0).length;
  const falseNegatives = decided
    .filter(({ risk, flagged }) => !flagged && risk === 1).length;

  return {
    falsePositives,
    falseNegatives,
    accuracy: safeDivide(
      decided.length - falsePositives - falseNegatives,
      decided.length,
    ),
    cost: falsePositives * costs.falsePositive
      + falseNegatives * costs.falseNegative,
  };
};

// As duas políticas lado a lado. Sem a coluna de custo a tabela contaria
// meia história: a razão de aprovação sempre melhora quando se ajusta o
// limiar para ela — a pergunta é o que se perdeu no caminho.
const formatMitigation = (policies) => formatTable(
  ['Política', 'Limiar M', 'Limiar H', 'Razão aprov.', 'Acurácia', 'Custo'],
  policies.map(({ label, audit, decisions }) => [
    label,
    formatThreshold(audit.thresholds.women),
    formatThreshold(audit.thresholds.men),
    Number.isFinite(audit.approvalRatio)
      ? audit.approvalRatio.toFixed(3)
      : 'infinita',
    decisions.accuracy.toFixed(4),
    String(decisions.cost),
  ]),
);

// --------------------------------------------------
// 8. Fontes de dados
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

  // Sem regularização, e isso foi MEDIDO, não herdado. São 225 parâmetros
  // para 768 linhas de treino efetivo: a capacidade já cabe no dado, e a
  // diferença treino−teste é 0.0067 ± 0.0058 — indistinguível de zero.
  // Não há o que frear aqui, e frear cobra: com os valores do dataset
  // real, a AUC cai de 0.9635 para 0.9567 e o custo mínimo sobe 21%.
  regularization: { l2: 0, dropout: 0 },

  sampleCustomer: {
    income: 3500,
    debtRatio: 0.72,
    latePayments: 3,
    creditUtilization: 0.88,
  },
};

// As duas variantes do dataset real diferem em UMA coisa: como as colunas
// qualitativas viram números. Tudo o mais — arquivo, leitura, escala,
// cliente de exemplo — é idêntico, então a fábrica evita duplicar.
// As duas variantes compartilham os freios porque compartilham o problema:
// é o mesmo dataset, com 1.073 ou 465 parâmetros para as mesmas 640 linhas.
// A grade medida separadamente para cada uma dá o mesmo veredito.
const createGermanSource = ({
  id,
  label,
  encoding,
  regularization = { l2: L2_LAMBDA, dropout: DROPOUT_RATE },
}) => ({
  id,
  label,
  encoding,
  regularization,
  csvPath: GERMAN_CSV_PATH,
  columns: GERMAN_COLUMNS,
  precision: GERMAN_PRECISION,
  featureNames: germanFeatureNames(encoding),

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

  // Só as numéricas passam pelo min-max: escalar um código de categoria
  // seria escalar um rótulo.
  fitScaler: (customers) => fitMinMaxScaler(customers, GERMAN_NUMERIC),
  toVector: (customer, scaler) => toGermanVector(customer, scaler, encoding),

  // A auditoria só existe nesta fonte, porque só ela tem um atributo
  // protegido para auditar.
  audit: auditByGroup,

  // Perfil desfavorável em todas as frentes: conta corrente no vermelho,
  // prazo longo, histórico curto, sem poupança e prestação no teto.
  sampleCustomer: {
    durationMonths: 48,
    creditAmount: 9000,
    installmentRate: 4,
    residenceSince: 2,
    age: 24,
    existingCredits: 2,
    dependents: 1,

    checkingStatus: 0,
    creditHistory: 1,
    purpose: 0,
    savingsStatus: 0,
    employmentYears: 1,
    otherDebtors: 0,
    property: 3,
    otherInstallments: 0,
    housing: 0,
    job: 1,
    telephone: 0,
    foreignWorker: 0,

    [GERMAN_AUDIT_COLUMN]: 2,
  },
});

// One-hot é o padrão: é a codificação correta para colunas sem ordem.
const GERMAN_SOURCE = createGermanSource({
  id: 'german',
  label: 'German Credit — UCI/Statlog (Hofmann, 1994), one-hot',
  encoding: 'onehot',
});

// A variante ordinal fica disponível para comparação. Ela não está aqui
// por ser recomendável, e sim para que a afirmação "one-hot é melhor"
// possa ser MEDIDA em vez de repetida.
const GERMAN_ORDINAL_SOURCE = createGermanSource({
  id: 'german-ordinal',
  label: 'German Credit — UCI/Statlog (Hofmann, 1994), ordinal',
  encoding: 'ordinal',
});

const SOURCES = {
  [SYNTHETIC_SOURCE.id]: SYNTHETIC_SOURCE,
  [GERMAN_SOURCE.id]: GERMAN_SOURCE,
  [GERMAN_ORDINAL_SOURCE.id]: GERMAN_ORDINAL_SOURCE,
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

// Os dois freios também são ajustáveis pela linha de comando, para que o
// efeito possa ser VISTO em vez de lido:
//
//   node index.js --l2=0 --dropout=0
//
// devolve exatamente a rede de antes deste item, e a diferença
// treino−teste que o `main` imprime volta a abrir.
const parseNumericFlag = (argv, name, fallback, highest) => {
  const prefix = `--${name}=`;
  const flag = argv.find((argument) => argument.startsWith(prefix));

  if (!flag) {
    return fallback;
  }

  const raw = flag.slice(prefix.length);
  const value = Number(raw);

  // `Number('')` é 0, e `Number(' ')` também. Sem a primeira condição,
  // `--l2=` desligaria a penalidade em silêncio — o pior tipo de erro de
  // configuração, porque o programa roda e o resultado parece legítimo.
  if (raw.trim() === '' || !Number.isFinite(value) || value < 0 || value > highest) {
    throw new Error(
      `Valor inválido para --${name}: ${raw}. Use um número entre 0 e ${highest}.`,
    );
  }

  return value;
};

// Devolve APENAS o que foi pedido explicitamente. O que não vier daqui
// fica com o valor que a FONTE declara, porque a intensidade certa
// depende da razão entre parâmetros e linhas — e essa razão é
// propriedade do dataset, não do laboratório.
//
// O teto do dropout é 0.9 de propósito: com taxa 1 a camada zeraria tudo
// que recebe e o treino não teria sinal nenhum para seguir.
const resolveRegularization = (argv = []) => {
  const pedido = {
    l2: parseNumericFlag(argv, 'l2', null, 1),
    dropout: parseNumericFlag(argv, 'dropout', null, 0.9),
  };

  return Object.fromEntries(
    Object.entries(pedido).filter(([, value]) => value !== null),
  );
};

// `--cv` sozinho usa o padrão de dobras; `--cv=10` escolhe. Duas dobras
// é o mínimo que ainda é validação cruzada, e um k não inteiro não
// significa nada — os dois casos param aqui em vez de virar um resultado
// estranho lá adiante.
const resolveFolds = (argv = []) => {
  const flag = argv.find((argument) => argument.startsWith('--cv'));

  if (!flag) {
    return null;
  }

  if (flag === '--cv') {
    return CV_FOLDS;
  }

  const folds = parseNumericFlag(argv, 'cv', CV_FOLDS, 20);

  if (!Number.isInteger(folds) || folds < 2) {
    throw new Error(
      `Valor inválido para --cv: ${folds}. Use um inteiro entre 2 e 20.`,
    );
  }

  return folds;
};

// `--units=64,32` monta duas camadas ocultas; `--units=0` monta NENHUMA,
// que é a regressão logística. O zero é a única forma aceita de pedir a
// rede sem camada oculta — string vazia continua sendo erro, pela mesma
// razão que em `--l2=`: um argumento em branco quase nunca é intenção, e
// silenciosamente trocar a arquitetura seria o pior desfecho possível.
const resolveUnits = (argv = []) => {
  const prefix = '--units=';
  const flag = argv.find((argument) => argument.startsWith(prefix));

  if (!flag) {
    return null;
  }

  const raw = flag.slice(prefix.length).trim();

  if (raw === '0') {
    return [];
  }

  const units = raw === '' ? [NaN] : raw.split(',').map((part) => Number(part.trim()));
  const invalido = units.some((count) =>
    !Number.isInteger(count) || count < 1 || count > 1024);

  if (invalido || units.length > 8) {
    throw new Error(
      `Valor inválido para --units: ${raw}. Use até 8 inteiros entre 1 e 1024 `
      + 'separados por vírgula, ou 0 para a regressão logística.',
    );
  }

  return units;
};

// `--arquiteturas` liga a comparação; `--repeticoes=k` diz quantas vezes
// a validação cruzada inteira se repete, com sementes diferentes.
const resolveArchitectureRun = (argv = []) => {
  const flag = argv.find((argument) => argument.startsWith('--arquiteturas'));

  if (!flag) {
    return null;
  }

  if (flag !== '--arquiteturas') {
    throw new Error(
      `Use --arquiteturas sem valor. Recebido: ${flag}.`,
    );
  }

  const repeats = parseNumericFlag(argv, 'repeticoes', 1, 20);

  if (!Number.isInteger(repeats) || repeats < 1) {
    throw new Error(
      `Valor inválido para --repeticoes: ${repeats}. Use um inteiro entre 1 e 20.`,
    );
  }

  return { repeats };
};

// A mitigação é um interruptor, não um número: ou a decisão olha o grupo
// ou não olha. Um valor depois do sinal de igual é recusado de propósito
// — aceitar `--mitigar=false` daria a impressão de que existe um terceiro
// estado, e `--mitigar=0` ligaria a política que o usuário quis desligar.
const resolveMitigation = (argv = []) => {
  const flag = argv.find((argument) => argument.startsWith('--mitigar'));

  if (!flag) {
    return false;
  }

  if (flag !== '--mitigar') {
    throw new Error(
      `Use --mitigar sem valor. Recebido: ${flag}.`,
    );
  }

  return true;
};

// --------------------------------------------------
// 9. Separar treino e teste
// --------------------------------------------------
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

// Estratificar é preservar a proporção de classes nos dois lados.
//
// O corte cru já embaralhado acerta a proporção EM MÉDIA, e erra em
// qualquer execução específica: com 30% de inadimplentes e 200 linhas de
// teste, o sorteio entrega entre 24% e 36% conforme a semente. Isso move
// o piso da classe majoritária, e o piso é a régua contra a qual toda a
// acurácia deste projeto é lida — ou seja, o ruído do sorteio entra
// direto na conclusão.
//
// A implementação separa os índices por classe, corta cada classe na
// mesma proporção e devolve os clientes na ORDEM original. A ordem
// importa: `fit` reserva os últimos 20% do treino para validação antes
// de embaralhar, então um conjunto agrupado por rótulo daria uma fatia
// de validação quase toda de uma classe só.
const stratifiedSplitCustomers = (customers, trainRatio = 0.8) => {
  const training = new Set();

  [...new Set(customers.map(({ risk }) => risk))].forEach((label) => {
    const indexes = customers
      .map((customer, index) => ({ risk: customer.risk, index }))
      .filter((row) => row.risk === label);

    indexes
      .slice(0, Math.floor(indexes.length * trainRatio))
      .forEach(({ index }) => training.add(index));
  });

  return {
    trainCustomers: customers.filter((ignored, index) => training.has(index)),
    testCustomers: customers.filter((ignored, index) => !training.has(index)),
  };
};

// Atribui uma dobra a cada cliente, também mantendo a proporção de
// classes. Distribuir em rodízio DENTRO de cada classe é o que garante
// que nenhuma dobra fique com inadimplentes de menos — com 5 dobras e
// 300 positivos, cada uma recebe 60.
const stratifiedFolds = (customers, folds) => {
  const seen = new Map();

  return customers.map(({ risk }) => {
    const position = seen.get(risk) ?? 0;

    seen.set(risk, position + 1);

    return position % folds;
  });
};

// Variante que opera sobre features JÁ normalizadas. Continua servindo o
// caminho sintético e os testes; o fluxo principal usa o split
// estratificado.
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
// 10. Criar a MLP
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

// A penalidade L2 precisa de uma instância por camada, mas a intensidade
// é a mesma em todas. `null` é o valor que o tfjs entende como "sem
// regularizador", então desligar o freio não exige um segundo caminho de
// código — só um argumento diferente.
const createRegularizer = (l2 = L2_LAMBDA) =>
  (l2 > 0 ? tf.regularizers.l2({ l2 }) : null);

// O número de entradas vem da fonte de dados: o sintético tem 4 features,
// o German Credit tem 19 ou 57 conforme a codificação. Era a única parte
// da rede que precisava saber qual dataset está em uso — os dois freios
// entraram como argumento pelo mesmo motivo: para poderem ser desligados
// e medidos, em vez de aceitos.
const buildModel = (inputSize = 4, options = {}) => {
  const {
    units = HIDDEN_UNITS,
    l2 = L2_LAMBDA,
    dropout = DROPOUT_RATE,
  } = options;
  const model = tf.sequential();

  // Com `dropout: 0` a camada é OMITIDA, não adicionada com taxa zero.
  // Uma camada inerte apareceria no `model.summary()` e no `model.json`
  // salvo em disco sugerindo um freio que não existe.
  const addHidden = (units, shape = {}) => {
    model.add(tf.layers.dense({
      ...shape,
      units,
      activation: 'relu',
      kernelRegularizer: createRegularizer(l2),
    }));

    if (dropout > 0) {
      model.add(tf.layers.dropout({ rate: dropout }));
    }
  };

  // A primeira camada é a única que declara o formato da entrada; as
  // demais o inferem da anterior.
  units.forEach((count, index) => addHidden(
    count,
    index === 0 ? { inputShape: [inputSize] } : {},
  ));

  // A saída não leva dropout. Descartar a única unidade que produz a
  // resposta não removeria um caminho redundante — apagaria a predição.
  //
  // Com `units: []` não há camada oculta nenhuma, e é a saída que passa a
  // declarar a entrada. O que sobra é uma REGRESSÃO LOGÍSTICA: uma soma
  // ponderada das features passando por uma sigmoide, sem não-linearidade
  // no meio. É o piso de arquitetura, e existe para ser medido — se a
  // rede não bater a linha reta, as camadas ocultas não estão pagando o
  // próprio custo.
  model.add(tf.layers.dense({
    ...(units.length === 0 ? { inputShape: [inputSize] } : {}),
    units: 1,
    activation: 'sigmoid',
    kernelRegularizer: createRegularizer(l2),
  }));

  return compileModel(model);
};

// A configuração de treino fica em um lugar só porque DOIS caminhos a
// usam: o fluxo principal e a validação cruzada. Se elas divergissem, a
// estimativa cruzada estaria medindo um modelo que o projeto não entrega.
const TRAINING = {
  epochs: 40,
  batchSize: 32,
  validationSplit: 0.2,
  patience: 5,
};

const fitModel = (model, xTrain, yTrain, options = {}) => model.fit(xTrain, yTrain, {
  epochs: TRAINING.epochs,
  batchSize: TRAINING.batchSize,
  validationSplit: TRAINING.validationSplit,
  shuffle: true,
  callbacks: [
    tf.callbacks.earlyStopping({
      monitor: 'val_loss',
      patience: TRAINING.patience,
    }),
  ],
  ...options,
});

// --------------------------------------------------
// 11. Persistência
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
// 12. Inferência
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

    const xTrain = tf.tensor2d(trainCustomers.map(toVector));
    const yTrain = tf.tensor2d(trainCustomers.map(({ risk }) => [risk]));
    const xTest = tf.tensor2d(testCustomers.map(toVector));
    const yTest = tf.tensor2d(testCustomers.map(({ risk }) => [risk]));

    const model = buildModel(source.featureNames.length, { units, l2, dropout });

    // eslint-disable-next-line no-await-in-loop
    const history = await fitModel(model, xTrain, yTrain, { verbose, ...training });

    const { loss, accuracy } = evaluateModel(model, xTest, yTest);
    const roc = computeRocCurve(model, xTest, yTest);
    const chosen = chooseThresholdByCost(roc, costs);

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
      threshold: chosen.threshold,
      cost: chosen.cost,
    });

    tf.dispose([xTrain, yTrain, xTest, yTest]);
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

// --------------------------------------------------
// 17.1 Comparar arquiteturas
// --------------------------------------------------
// "Qual arquitetura usar?" é a pergunta que mais se responde por hábito
// em projetos de rede neural: duas camadas ocultas, umas dezenas de
// unidades, afunilando. Aqui ela é respondida do mesmo jeito que as
// outras — medindo, com a mesma validação cruzada que mede o resto.
//
// A lista começa DELIBERADAMENTE no piso. Uma regressão logística não é
// rede neural nenhuma: é uma soma ponderada das features passando por
// uma sigmoide. Se as camadas ocultas não baterem essa linha reta, elas
// não estão pagando o próprio custo — e essa é uma resposta possível.
const ARCHITECTURES = [
  { label: 'regressão logística', units: [] },
  { label: '4', units: [4] },
  { label: '16', units: [16] },
  { label: '16 → 8 (padrão)', units: HIDDEN_UNITS },
  { label: '32 → 16', units: [32, 16] },
  { label: '64 → 32', units: [64, 32] },
  { label: '128 → 64', units: [128, 64] },
  { label: '16 → 16 → 16', units: [16, 16, 16] },
];

// Cada arquitetura roda a validação cruzada inteira, e todas as dobras de
// todas as repetições entram no mesmo resumo. Com `repeats > 1` a semente
// do embaralhamento muda a cada repetição: repetir com a MESMA semente
// mediria só a variação dos pesos iniciais, que já se sabe ser pequena
// perto da variação do sorteio.
const compareArchitectures = async (source, options = {}) => {
  const {
    architectures = ARCHITECTURES,
    folds = CV_FOLDS,
    repeats = 1,
    seed = SHUFFLE_SEED,
    l2,
    dropout,
    training = {},
    verbose = 0,
  } = options;

  const rows = [];
  const baselines = [];

  for (const { label, units } of architectures) {
    const measurements = { accuracy: [], auc: [], cost: [], epochs: [] };
    let parameters = 0;

    for (let repeat = 0; repeat < repeats; repeat += 1) {
      // eslint-disable-next-line no-await-in-loop
      const result = await crossValidate(source, {
        folds, units, l2, dropout, training, verbose, seed: seed + repeat,
      });

      parameters = result.parameters;
      result.folds.forEach((entry) => {
        measurements.accuracy.push(entry.accuracy);
        measurements.auc.push(entry.auc);
        measurements.cost.push(entry.cost);
        measurements.epochs.push(entry.epochs);
        baselines.push(entry.baseline);
      });
    }

    rows.push({
      label,
      units,
      parameters,
      accuracy: summarize(measurements.accuracy),
      auc: summarize(measurements.auc),
      cost: summarize(measurements.cost),
      epochs: summarize(measurements.epochs),
    });
  }

  return { rows, baseline: summarize(baselines), folds, repeats };
};

const formatArchitectureComparison = ({ rows, baseline, folds, repeats }) => [
  formatTable(
    ['Arquitetura', 'Parâmetros', 'Épocas', 'Acurácia', 'AUC', 'Custo'],
    rows.map((row) => [
      row.label,
      String(row.parameters),
      row.epochs.mean.toFixed(1),
      `${row.accuracy.mean.toFixed(4)} ± ${row.accuracy.standardError.toFixed(4)}`,
      `${row.auc.mean.toFixed(4)} ± ${row.auc.standardError.toFixed(4)}`,
      `${row.cost.mean.toFixed(1)} ± ${row.cost.standardError.toFixed(1)}`,
    ]),
  ),
  '',
  `Baseline da classe majoritária: ${baseline.mean.toFixed(4)}`,
  `Protocolo: ${folds} dobras × ${repeats} repetição(ões) = `
    + `${folds * repeats} medidas por arquitetura.`,
  '',
  // O leitor precisa saber a régua ANTES de comparar as linhas: com erros
  // padrão desta ordem, diferenças na terceira casa não são diferenças.
  'Duas arquiteturas só se distinguem se a distância entre elas superar a',
  'soma dos erros padrão. Compare as colunas com isso em mente.',
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

const reportArchitectures = async (sourceId = DEFAULT_SOURCE_ID, options = {}) => {
  const source = SOURCES[sourceId];
  const { folds = CV_FOLDS, repeats = 1 } = options;

  console.log('Fonte:', source.label);
  console.log('Arquivo:', source.csvPath);
  console.log('Entradas da rede:', source.featureNames.length);
  console.log(
    `Comparando ${ARCHITECTURES.length} arquiteturas por validação cruzada `
    + `(${folds} dobras × ${repeats}); isso treina `
    + `${ARCHITECTURES.length * folds * repeats} modelos.\n`,
  );

  const result = await compareArchitectures(source, options);

  console.log(formatArchitectureComparison(result));

  return result;
};

// --------------------------------------------------
// 18. Treinar, avaliar, salvar, recarregar e prever
// --------------------------------------------------
const main = async (sourceId = DEFAULT_SOURCE_ID, overrides = {}) => {
  const source = SOURCES[sourceId];
  const { mitigate = false, units = HIDDEN_UNITS, ...regularization } = overrides;

  // A fonte decide; a linha de comando tem a última palavra.
  const { l2, dropout } = { ...source.regularization, ...regularization };

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

  // Estratificado: o teste recebe a MESMA proporção de inadimplentes do
  // arquivo, então o piso da classe majoritária deixa de depender do
  // sorteio. Sem isso, o número contra o qual toda a acurácia é lida
  // muda de execução para execução por motivo nenhum.
  const { trainCustomers, testCustomers } = stratifiedSplitCustomers(customers);

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

  const model = buildModel(source.featureNames.length, { units, l2, dropout });

  console.log('Arquitetura:', units.length === 0
    ? 'regressão logística (sem camada oculta)'
    : units.join(' → '));
  console.log(`Regularização: L2 = ${l2}, dropout = ${dropout}`);
  model.summary();

  await fitModel(model, xTrain, yTrain);

  // O mesmo modelo avaliado nos dois conjuntos. `evaluate` roda em modo de
  // inferência, então o dropout está DESLIGADO nas duas medidas — é a
  // comparação justa, e não a acurácia pessimista que o `fit` imprime.
  const { accuracy: trainAccuracy } = evaluateModel(model, xTrain, yTrain);
  const { loss: testLoss, accuracy: testAccuracy } =
    evaluateModel(model, xTest, yTest);

  // A acurácia sozinha não diz se o modelo presta. Ao lado do piso da
  // classe majoritária ela passa a dizer: a distância entre os dois
  // números é tudo que o treino realmente acrescentou.
  const baseline = majorityBaseline(testLabels);

  console.log('Test loss:', testLoss.toFixed(4));
  console.log('Train accuracy:', trainAccuracy.toFixed(4));
  console.log('Test accuracy:', testAccuracy.toFixed(4));
  console.log('Baseline (classe majoritária):', baseline.toFixed(4));

  // O termômetro do overfitting. Quanto o modelo vai melhor no que já viu
  // do que no que não viu é exatamente o que os dois freios existem para
  // encolher — e `--l2=0 --dropout=0` mostra o número sem eles.
  console.log(
    'Diferença treino − teste:',
    (trainAccuracy - testAccuracy).toFixed(4),
  );

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
  // Auditoria: a coluna que o modelo NÃO recebeu
  // ------------------------------------------------
  if (source.audit) {
    const scores = tf.tidy(() =>
      Array.from(model.predict(xTest).reshape([-1]).dataSync()));

    // Os scores do treino existem aqui por um motivo só: calibrar os
    // limiares por grupo. Eles não entram em nenhuma métrica publicada —
    // quem é auditado continua sendo o conjunto de teste.
    const trainScores = tf.tidy(() =>
      Array.from(model.predict(xTrain).reshape([-1]).dataSync()));

    const policies = [
      { label: 'Limiar único', threshold: chosen.threshold },
      {
        label: 'Limiar por grupo',
        threshold: fitGroupThresholds(trainCustomers, trainScores, chosen.threshold),
      },
    ].map(({ label, threshold }) => ({
      label,
      audit: source.audit(testCustomers, scores, threshold),
      decisions: summarizeDecisions(testCustomers, scores, threshold, costs),
    }));

    const active = policies[mitigate ? 1 : 0];

    console.log(
      `Auditoria por sexo — ${active.label.toLowerCase()}`,
      '(o modelo nunca recebeu esta coluna):',
    );
    console.log(formatAudit(active.audit));
    console.log('');

    console.log('Mitigação (limiares calibrados no TREINO, auditados no teste):');
    console.log(formatMitigation(policies));
    console.log(mitigate
      ? 'Política ativa: limiar por grupo — a decisão lê o sexo do cliente.'
      : 'Política ativa: limiar único. Use --mitigar para decidir por grupo.');
    console.log('');
  }

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
  const argv = process.argv.slice(2);
  const run = async () => {
    // Tudo que pode lançar fica DENTRO da função async, para que um
    // argumento inválido vire rejeição e caia no mesmo `.catch`.
    const folds = resolveFolds(argv);
    const arquiteturas = resolveArchitectureRun(argv);
    const sourceId = resolveSourceId(argv);
    const regularization = resolveRegularization(argv);
    const units = resolveUnits(argv);

    if (arquiteturas) {
      return reportArchitectures(sourceId, {
        ...regularization,
        ...arquiteturas,
        ...(folds === null ? {} : { folds }),
      });
    }

    return folds === null
      ? main(sourceId, {
        ...regularization,
        ...(units === null ? {} : { units }),
        mitigate: resolveMitigation(argv),
      })
      : reportCrossValidation(sourceId, {
        ...regularization,
        ...(units === null ? {} : { units }),
        folds,
      });
  };

  run().catch((error) => {
    console.error(`\n${error.message}\n`);
    process.exitCode = 1;
  });
}


module.exports = {
  INCOME_MIN,
  INCOME_RANGE,
  MAX_LATE_PAYMENTS,
  SYNTHETIC_SEED,
  SYNTHETIC_TOTAL,
  SYNTHETIC_POSITIVE_RATE,
  SYNTHETIC_FEATURE_NOISE,
  SYNTHETIC_LABEL_NOISE,
  SYNTHETIC_BOUNDS,
  DECISION_THRESHOLD,
  L2_LAMBDA,
  DROPOUT_RATE,
  MODEL_DIR,
  CSV_PATH,
  CSV_COLUMNS,
  CSV_LABEL_COLUMN,
  CSV_PRECISION,
  GERMAN_CSV_PATH,
  GERMAN_SOURCE_URL,
  GERMAN_NUMERIC,
  GERMAN_CATEGORICAL,
  GERMAN_AUDIT_COLUMN,
  GERMAN_AUDIT_CODES,
  GERMAN_SOURCE_ATTRIBUTES,
  FEMALE_CODE,
  GERMAN_COLUMNS,
  GERMAN_PRECISION,
  SHUFFLE_SEED,
  normalizeIncome,
  normalizeLatePayments,
  toFeatureVector,
  classify,
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
  parseDelimited,
  toOrdinal,
  toGermanCustomer,
  parseGermanCsv,
  fitMinMaxScaler,
  applyMinMaxScaler,
  oneHotEncode,
  ordinalEncode,
  germanFeatureNames,
  toGermanVector,
  isFemale,
  toAuditRows,
  summarizeGroup,
  approvalRatio,
  thresholdFor,
  auditByGroup,
  formatAudit,
  rateThreshold,
  fitGroupThresholds,
  summarizeDecisions,
  formatMitigation,
  createGermanSource,
  SYNTHETIC_SOURCE,
  GERMAN_SOURCE,
  GERMAN_ORDINAL_SOURCE,
  SOURCES,
  DEFAULT_SOURCE_ID,
  resolveSourceId,
  parseNumericFlag,
  resolveRegularization,
  resolveMitigation,
  resolveFolds,
  resolveUnits,
  resolveArchitectureRun,
  createRandom,
  shuffle,
  splitCustomers,
  stratifiedSplitCustomers,
  stratifiedFolds,
  majorityBaseline,
  readCustomersCsv,
  loadDatasetCsv,
  splitDataset,
  compileModel,
  createRegularizer,
  HIDDEN_UNITS,
  buildModel,
  saveModel,
  loadModel,
  predictRisk,
  computeConfusionMatrix,
  formatConfusionMatrix,
  computeMetrics,
  formatMetrics,
  rocFromScores,
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
  TRAINING,
  fitModel,
  CV_FOLDS,
  summarize,
  crossValidate,
  formatCrossValidation,
  reportCrossValidation,
  ARCHITECTURES,
  compareArchitectures,
  formatArchitectureComparison,
  reportArchitectures,
  main,
};
