const fs = require('node:fs');
const path = require('node:path');

const tf = require('@tensorflow/tfjs-node');

const {
  CSV_COLUMNS,
  CSV_LABEL_COLUMN,
  CSV_PATH,
  CSV_PRECISION,
  SYNTHETIC_TOTAL,
} = require('./00-constants');
const { createCustomers, toDataset } = require('./02-synthetic');

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

module.exports = {
  formatCsvValue,
  toCsv,
  writeCustomersCsv,
  readCustomersCsv,
  loadDatasetCsv,
  ensureCsv,
};
