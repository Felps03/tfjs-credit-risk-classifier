const {
  CSV_LABEL_COLUMN,
  GERMAN_AUDIT_CODES,
  GERMAN_AUDIT_COLUMN,
  GERMAN_CATEGORICAL,
  GERMAN_NUMERIC,
} = require('./00-constants');

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


module.exports = {
  parseDelimited,
  toOrdinal,
  GERMAN_SOURCE_ATTRIBUTES,
  toGermanCustomer,
  parseGermanCsv,
};
