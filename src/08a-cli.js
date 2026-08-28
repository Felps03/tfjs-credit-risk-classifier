// --------------------------------------------------
// 8.1 Argumentos de linha de comando
// --------------------------------------------------
const { CV_FOLDS } = require('./17-cross-validation');
const { DEFAULT_SOURCE_ID, SOURCES } = require('./08-sources');

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

module.exports = {
  resolveSourceId,
  parseNumericFlag,
  resolveRegularization,
  resolveFolds,
  resolveUnits,
  resolveArchitectureRun,
  resolveMitigation,
};
