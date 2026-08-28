const fs = require('node:fs');
const path = require('node:path');

const { MODEL_DIR } = require('./00-constants');
const { SOURCES } = require('./08-sources');
const { loadModel, saveModel } = require('./11-persistence');
const { predictRisk } = require('./12-inference');

// --------------------------------------------------
// 19. O pacote servido: os pesos e tudo que eles pressupõem
// --------------------------------------------------
// Até aqui `saveModel` salvava o modelo e mais nada — e isso bastava,
// porque quem recarregava era o mesmo processo que tinha acabado de
// treinar: o `scaler` ainda estava na memória, o limiar escolhido também.
//
// Um serviço não tem esse luxo. Ele sobe horas depois, em outra máquina,
// e recebe um cliente em unidades brutas: 48 meses, R$ 9.000, 24 anos.
// Para virar o vetor que a rede espera, ele precisa da MESMA escala que
// foi medida no treino. Se ele remedir a escala com os dados que estiver
// vendo, ou pior, chutar constantes, a rede recebe números que não
// significam o que os pesos aprenderam — e a resposta sai errada sem
// nenhum erro aparecer. É o *training-serving skew*, e ele não quebra:
// ele degrada em silêncio.
//
// A correção é tratar o modelo como um PACOTE, não como um arquivo de
// pesos. Junto de `model.json` e `weights.bin` passa a viver:
//
//   metadata.json  → a fonte, a codificação, a lista de features na
//                    ordem exata, o scaler medido no treino e o limiar
//                    que foi escolhido pela matriz de custo.
//
// Sem esses quatro, os pesos são um monte de números sem contrato.
const METADATA_FILE = 'metadata.json';

// Casas decimais do limiar e das probabilidades servidas.
//
// Arredondar UMA vez, aqui, resolve um detalhe que parece cosmético e não
// é: o limiar sai da curva ROC como um float32 (`0.14885064959526062`) e
// o serviço precisa exibir esse número na resposta. Exibir arredondado e
// comparar com o cheio produziria, na fronteira, um JSON que se
// contradiz — `riskProbability` igual ao `threshold` e classificação
// LOW_RISK. Gravando já arredondado, o número do pacote, o número que
// decide e o número que aparece na resposta são o MESMO número.
const PROBABILITY_DECIMALS = 6;

const round = (value, decimals = PROBABILITY_DECIMALS) => Number(value.toFixed(decimals));

// A versão existe para que um pacote antigo seja RECUSADO em vez de
// interpretado errado por um código novo. É barato agora e impagável na
// primeira vez que o formato mudar.
const ARTIFACTS_VERSION = 1;

const metadataPath = (dir = MODEL_DIR) => path.join(dir, METADATA_FILE);

// O primeiro ponto da curva ROC tem `threshold: Infinity` — é o corte que
// não aprova ninguém. Ele quase nunca ganha a comparação por custo, mas
// "quase nunca" não é "nunca", e `JSON.stringify(Infinity)` é `null`.
// Um limiar `null` no pacote faria `probabilidade >= null` virar
// `probabilidade >= 0`: TODO cliente sairia como alto risco, sem erro
// nenhum aparecer. Recusar na hora de gravar é mais barato do que
// descobrir isso pela taxa de aprovação do serviço.
const assertServable = ({ source, featureNames, threshold }) => {
  if (typeof source !== 'string' || !Array.isArray(featureNames)
    || featureNames.length === 0) {
    throw new Error(
      'Pacote sem fonte ou sem lista de features: os pesos ficariam sem contrato.',
    );
  }

  if (!Number.isFinite(threshold) || threshold < 0 || threshold > 1) {
    throw new Error(
      `Limiar impróprio para servir: ${threshold}. Esperado um número `
      + 'entre 0 e 1 — o corte precisa ser alcançável por uma probabilidade.',
    );
  }
};

// Salva os pesos e o contrato na mesma operação. Chamar `saveModel`
// sozinho continua possível — os testes fazem isso —, mas o fluxo
// principal usa esta função justamente para que os dois nunca saiam
// de sincronia.
const saveArtifacts = async (model, metadata, dir = MODEL_DIR) => {
  assertServable(metadata);

  await saveModel(model, dir);

  const record = {
    version: ARTIFACTS_VERSION,
    savedAt: new Date().toISOString(),
    ...metadata,
    threshold: round(metadata.threshold),
  };

  fs.writeFileSync(metadataPath(dir), `${JSON.stringify(record, null, 2)}\n`);

  return record;
};

const readMetadata = (dir = MODEL_DIR) => {
  const file = metadataPath(dir);

  if (!fs.existsSync(file)) {
    throw new Error([
      `Pacote do modelo incompleto: ${file} não existe.`,
      'Rode `npm start` para treinar e salvar o modelo junto do scaler.',
    ].join('\n'));
  }

  return JSON.parse(fs.readFileSync(file, 'utf8'));
};

// As três perguntas que precisam ser respondidas ANTES da primeira
// predição, porque depois dela ninguém mais percebe.
//
// A terceira é a menos óbvia e a mais útil: o `metadata.json` guarda a
// lista de features como ela era no dia do treino. Se alguém acrescentar
// uma coluna ao dataset, mudar a ordem de `GERMAN_CATEGORICAL` ou trocar
// a codificação, a lista que o código gera hoje deixa de bater com a
// lista gravada — e os pesos passam a receber outra coisa em cada
// entrada. O tamanho do vetor pode até continuar o mesmo; o significado
// de cada posição, não.
const assertConsistent = (model, metadata, source) => {
  if (metadata.version !== ARTIFACTS_VERSION) {
    throw new Error(
      `Pacote na versão ${metadata.version}, mas este código lê a `
      + `versão ${ARTIFACTS_VERSION}. Retreine com \`npm start\`.`,
    );
  }

  const inputSize = model.inputs[0].shape[1];

  if (inputSize !== metadata.featureNames.length) {
    throw new Error(
      `A rede espera ${inputSize} entradas, mas o pacote declara `
      + `${metadata.featureNames.length} features. Retreine com \`npm start\`.`,
    );
  }

  if (source.featureNames.join('|') !== metadata.featureNames.join('|')) {
    throw new Error([
      'As features mudaram desde que o modelo foi salvo: o código gera',
      `${source.featureNames.length} entradas em uma ordem diferente da`,
      'que está gravada no pacote. Servir assim daria previsão errada sem',
      'erro nenhum. Retreine com `npm start`.',
    ].join(' '));
  }
};

// Devolve tudo pronto para pontuar: o modelo, o contrato e um `predict`
// que já embute a normalização certa. Quem serve não precisa saber que
// existe scaler, codificação ou ordem de features — e é exatamente por
// não precisar saber que ele não tem como errar.
const loadArtifacts = async (dir = MODEL_DIR) => {
  const metadata = readMetadata(dir);
  const source = SOURCES[metadata.source];

  if (!source) {
    throw new Error(
      `O pacote foi salvo com a fonte "${metadata.source}", que não existe `
      + `mais. Fontes atuais: ${Object.keys(SOURCES).join(', ')}.`,
    );
  }

  const model = await loadModel(dir);

  try {
    assertConsistent(model, metadata, source);
  } catch (error) {
    // Um modelo que não vai ser servido não deve ficar ocupando memória
    // só porque a checagem falhou.
    model.dispose();
    throw error;
  }

  const toVector = (customer) => source.toVector(customer, metadata.scaler);

  return {
    model,
    metadata,
    source,
    toVector,
    predict: (customer) => predictRisk(model, customer, toVector),
    dispose: () => model.dispose(),
  };
};

module.exports = {
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
};
