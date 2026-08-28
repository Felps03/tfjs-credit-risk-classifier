const { HIDDEN_UNITS, SHUFFLE_SEED } = require('./00-constants');
const {
  CV_FOLDS,
  crossValidate,
  summarize,
} = require('./17-cross-validation');
const { formatTable } = require('./13a-format');
const { DEFAULT_SOURCE_ID, SOURCES } = require('./08-sources');

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
    balance = false,
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
        folds, units, l2, dropout, training, balance, verbose, seed: seed + repeat,
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

module.exports = {
  ARCHITECTURES,
  compareArchitectures,
  formatArchitectureComparison,
  reportArchitectures,
};
