// Ponto de entrada do laboratório. O código vive em `src/`, dividido por
// seção; aqui ficam apenas duas coisas: a leitura dos argumentos quando o
// arquivo é executado direto, e a reexportação das funções para os testes
// e os scripts, que continuam importando deste módulo.
const {
  INCOME_MIN,
  INCOME_RANGE,
  MAX_LATE_PAYMENTS,
  SYNTHETIC_SEED,
  SYNTHETIC_TOTAL,
  SYNTHETIC_POSITIVE_RATE,
  SYNTHETIC_FEATURE_NOISE,
  SYNTHETIC_LABEL_NOISE,
  DECISION_THRESHOLD,
  CALIBRATION_SPLIT,
  L2_LAMBDA,
  DROPOUT_RATE,
  HIDDEN_UNITS,
  MODEL_DIR,
  CSV_PATH,
  CSV_LABEL_COLUMN,
  CSV_COLUMNS,
  CSV_PRECISION,
  GERMAN_CSV_PATH,
  GERMAN_SOURCE_URL,
  GERMAN_NUMERIC,
  GERMAN_CATEGORICAL,
  GERMAN_AUDIT_COLUMN,
  GERMAN_AUDIT_CODES,
  FEMALE_CODE,
  GERMAN_COLUMNS,
  GERMAN_PRECISION,
  SHUFFLE_SEED,
  API_PORT,
  API_BODY_LIMIT,
} = require('./src/00-constants');

const {
  normalizeIncome,
  normalizeLatePayments,
  toFeatureVector,
  classify,
} = require('./src/01-preprocess');

const {
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
} = require('./src/02-synthetic');

const {
  toCsv,
  writeCustomersCsv,
  readCustomersCsv,
  loadDatasetCsv,
  ensureCsv,
} = require('./src/03-csv');

const {
  parseDelimited,
  toOrdinal,
  GERMAN_SOURCE_ATTRIBUTES,
  toGermanCustomer,
  parseGermanCsv,
} = require('./src/04-german');

const { fitMinMaxScaler, applyMinMaxScaler } = require('./src/05-scaler');

const {
  oneHotEncode,
  ordinalEncode,
  germanFeatureNames,
  toGermanVector,
} = require('./src/06-encoding');

const {
  isFemale,
  summarizeGroup,
  approvalRatio,
  toAuditRows,
  thresholdFor,
  auditByGroup,
  formatAudit,
  rateThreshold,
  fitGroupThresholds,
  summarizeDecisions,
  formatMitigation,
} = require('./src/07-audit');

const {
  SYNTHETIC_SOURCE,
  createGermanSource,
  GERMAN_SOURCE,
  GERMAN_ORDINAL_SOURCE,
  SOURCES,
  DEFAULT_SOURCE_ID,
} = require('./src/08-sources');

const {
  resolveSourceId,
  resolvePort,
  parseNumericFlag,
  resolveRegularization,
  resolveFolds,
  resolveUnits,
  resolveArchitectureRun,
  resolveMitigation,
  resolveBalance,
} = require('./src/08a-cli');

const {
  shuffle,
  splitCustomers,
  stratifiedSplitCustomers,
  splitCalibration,
  stratifiedFolds,
  splitDataset,
} = require('./src/09-split');

const {
  compileModel,
  createRegularizer,
  buildModel,
  TRAINING,
  balancedClassWeight,
  fitModel,
} = require('./src/10-model');

const { saveModel, loadModel } = require('./src/11-persistence');

const { predictRisk } = require('./src/12-inference');

const {
  computeConfusionMatrix,
  formatConfusionMatrix,
} = require('./src/13-confusion');

const { formatTable } = require('./src/13a-format');

const { computeMetrics, formatMetrics } = require('./src/14-metrics');

const {
  rocFromScores,
  computeRocCurve,
  formatRocCurve,
} = require('./src/15-roc');

const {
  FALSE_POSITIVE_COST,
  FALSE_NEGATIVE_COST,
  scorePoint,
  chooseThresholdByYouden,
  chooseThresholdByCost,
  formatThresholdComparison,
} = require('./src/16-threshold');

const { majorityBaseline, evaluateModel } = require('./src/16a-evaluate');

const {
  CV_FOLDS,
  summarize,
  crossValidate,
  formatCrossValidation,
  reportCrossValidation,
} = require('./src/17-cross-validation');

const {
  ARCHITECTURES,
  compareArchitectures,
  formatArchitectureComparison,
  reportArchitectures,
} = require('./src/17a-architectures');

const { main } = require('./src/18-main');

const {
  METADATA_FILE,
  ARTIFACTS_VERSION,
  PROBABILITY_DECIMALS,
  round,
  metadataPath,
  assertServable,
  saveArtifacts,
  readMetadata,
  assertConsistent,
  loadArtifacts,
} = require('./src/19-artifacts');

const {
  isNumber,
  validateCategorical,
  validateCustomer,
  describeSchema,
} = require('./src/20-contract');

const {
  scoreCustomer,
  exampleCustomer,
  observedRange,
  sendJson,
  readJsonBody,
  isJsonRequest,
  createRequestListener,
  createRoutes,
  createApi,
  listen,
} = require('./src/21-api');

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

    const balance = resolveBalance(argv);

    if (arquiteturas) {
      return reportArchitectures(sourceId, {
        ...regularization,
        ...arquiteturas,
        balance,
        ...(folds === null ? {} : { folds }),
      });
    }

    return folds === null
      ? main(sourceId, {
        ...regularization,
        ...(units === null ? {} : { units }),
        mitigate: resolveMitigation(argv),
        balance,
      })
      : reportCrossValidation(sourceId, {
        ...regularization,
        ...(units === null ? {} : { units }),
        balance,
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
  CALIBRATION_SPLIT,
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
  resolvePort,
  parseNumericFlag,
  resolveRegularization,
  resolveMitigation,
  resolveBalance,
  resolveFolds,
  resolveUnits,
  resolveArchitectureRun,
  createRandom,
  shuffle,
  splitCustomers,
  stratifiedSplitCustomers,
  splitCalibration,
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
  balancedClassWeight,
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
  API_PORT,
  API_BODY_LIMIT,
  METADATA_FILE,
  ARTIFACTS_VERSION,
  PROBABILITY_DECIMALS,
  round,
  metadataPath,
  assertServable,
  saveArtifacts,
  readMetadata,
  assertConsistent,
  loadArtifacts,
  isNumber,
  validateCategorical,
  validateCustomer,
  describeSchema,
  round,
  scoreCustomer,
  exampleCustomer,
  observedRange,
  sendJson,
  readJsonBody,
  isJsonRequest,
  createRequestListener,
  createRoutes,
  createApi,
  listen,
};
