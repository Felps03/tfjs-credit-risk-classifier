const {
  INCOME_MIN,
  INCOME_RANGE,
  MAX_LATE_PAYMENTS,
  SYNTHETIC_FEATURE_NOISE,
  SYNTHETIC_LABEL_NOISE,
  SYNTHETIC_POSITIVE_RATE,
  SYNTHETIC_SEED,
  SYNTHETIC_TOTAL,
} = require('./00-constants');
const { toFeatureVector } = require('./01-preprocess');

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

module.exports = {
  createRandom,
  createGaussian,
  clamp,
  quantile,
  SYNTHETIC_BOUNDS,
  riskScore,
  measureCustomer,
  createCustomers,
  toDataset,
  createDataset,
};
