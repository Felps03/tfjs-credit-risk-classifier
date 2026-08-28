const {
  DECISION_THRESHOLD,
  INCOME_MIN,
  INCOME_RANGE,
  MAX_LATE_PAYMENTS,
} = require('./00-constants');

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

module.exports = {
  normalizeIncome,
  normalizeLatePayments,
  toFeatureVector,
  classify,
};
