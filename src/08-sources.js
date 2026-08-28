const fs = require('node:fs');

const {
  CSV_COLUMNS,
  CSV_PATH,
  CSV_PRECISION,
  DROPOUT_RATE,
  GERMAN_AUDIT_COLUMN,
  GERMAN_CATEGORICAL,
  GERMAN_COLUMNS,
  GERMAN_CSV_PATH,
  GERMAN_NUMERIC,
  GERMAN_PRECISION,
  L2_LAMBDA,
} = require('./00-constants');
const { auditByGroup } = require('./07-audit');
const { ensureCsv, readCustomersCsv } = require('./03-csv');
const { germanFeatureNames, toGermanVector } = require('./06-encoding');
const { toFeatureVector } = require('./01-preprocess');
const { fitMinMaxScaler } = require('./05-scaler');

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

  // O que um cliente novo precisa trazer para ser pontuado. Aqui as
  // quatro colunas são todas numéricas e nenhuma é proibida — a fonte
  // sintética não tem atributo protegido para proteger.
  requestSchema: { numeric: CSV_COLUMNS.slice(0, -1), categorical: {}, rejected: [] },

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

  // O contrato de entrada de um cliente novo: as sete numéricas em
  // unidades BRUTAS (meses, reais, anos) e as doze qualitativas como
  // índice do código na lista da UCI.
  //
  // `personalStatus` aparece em `rejected`, não em `categorical`, e a
  // diferença importa: ele não é uma coluna que o serviço esqueceu de
  // aceitar, é uma coluna que o serviço RECUSA. Ignorá-lo em silêncio
  // deixaria quem chama achando que mandou algo que foi usado.
  requestSchema: {
    numeric: GERMAN_NUMERIC,
    categorical: GERMAN_CATEGORICAL,
    rejected: [GERMAN_AUDIT_COLUMN],
  },

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

module.exports = {
  SYNTHETIC_SOURCE,
  createGermanSource,
  GERMAN_SOURCE,
  GERMAN_ORDINAL_SOURCE,
  SOURCES,
  DEFAULT_SOURCE_ID,
};
