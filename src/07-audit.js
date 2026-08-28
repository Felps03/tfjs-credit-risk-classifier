const {
  CSV_LABEL_COLUMN,
  DECISION_THRESHOLD,
  FEMALE_CODE,
  GERMAN_AUDIT_CODES,
  GERMAN_AUDIT_COLUMN,
} = require('./00-constants');
const { formatTable } = require('./13a-format');
const { safeDivide } = require('./14-metrics');
const {
  FALSE_NEGATIVE_COST,
  FALSE_POSITIVE_COST,
  formatThreshold,
} = require('./16-threshold');

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

module.exports = {
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
};
