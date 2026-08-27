// Regenera o dataset sintético em `data/customers.csv`.
//
// Diferente do que este script era antes, agora ele é DETERMINÍSTICO: o
// gerador tem semente, então rodar duas vezes produz um arquivo idêntico
// byte a byte. É o que torna o CSV versionado auditável — quem clona pode
// reconstruí-lo a partir do código e conferir que ninguém o editou à mão.
//
// Para produzir dados diferentes de propósito, mude a semente. Para ver o
// efeito do ruído e do desbalanceamento, mexa em um parâmetro de cada vez:
//
//   node scripts/seed.js --seed=99
//   node scripts/seed.js --feature-noise=0 --label-noise=0   # dataset limpo
//   node scripts/seed.js --positive-rate=0.5                 # balanceado
const {
  createCustomers,
  writeCustomersCsv,
  CSV_PATH,
  SYNTHETIC_SEED,
  SYNTHETIC_TOTAL,
  SYNTHETIC_POSITIVE_RATE,
  SYNTHETIC_FEATURE_NOISE,
  SYNTHETIC_LABEL_NOISE,
} = require('../index');

const FLAGS = {
  seed: SYNTHETIC_SEED,
  total: SYNTHETIC_TOTAL,
  'positive-rate': SYNTHETIC_POSITIVE_RATE,
  'feature-noise': SYNTHETIC_FEATURE_NOISE,
  'label-noise': SYNTHETIC_LABEL_NOISE,
};

// `--chave=valor` → número. Uma flag desconhecida é erro, não silêncio:
// `--featureNoise=0` passaria batido e o usuário acharia que funcionou.
const parseFlags = (argv) => argv.reduce((flags, argument) => {
  const [key, value] = argument.replace(/^--/, '').split('=');

  if (!(key in FLAGS)) {
    throw new Error(
      `Flag desconhecida: --${key}\nDisponíveis: ${Object.keys(FLAGS).map((f) => `--${f}`).join(', ')}`,
    );
  }

  if (!Number.isFinite(Number(value))) {
    throw new Error(`--${key} espera um número, recebi: ${value}`);
  }

  return { ...flags, [key]: Number(value) };
}, FLAGS);

const main = () => {
  const flags = parseFlags(process.argv.slice(2));

  const customers = createCustomers(flags.total, {
    seed: flags.seed,
    positiveRate: flags['positive-rate'],
    featureNoise: flags['feature-noise'],
    labelNoise: flags['label-noise'],
  });

  writeCustomersCsv(customers, CSV_PATH);

  const positives = customers.filter(({ risk }) => risk === 1).length;
  const share = (positives / customers.length) * 100;

  console.log(`Dataset regerado em: ${CSV_PATH}`);
  console.log(`  ${customers.length} clientes, semente ${flags.seed}`);
  console.log(`  ${positives} inadimplentes (${share.toFixed(1)}%), `
    + `ruído de medição ${flags['feature-noise']}, rótulos trocados ${flags['label-noise']}`);
};

try {
  main();
} catch (error) {
  console.error(`\n${error.message}\n`);
  process.exitCode = 1;
}
