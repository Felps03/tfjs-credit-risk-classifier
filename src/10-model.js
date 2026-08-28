const tf = require('@tensorflow/tfjs-node');

const { DROPOUT_RATE, HIDDEN_UNITS, L2_LAMBDA } = require('./00-constants');

// --------------------------------------------------
// 10. Criar a MLP
// --------------------------------------------------
// A compilação fica isolada porque é usada em dois momentos:
// ao montar o modelo do zero e ao recompilar um modelo carregado
// do disco que tenha sido salvo sem o estado do otimizador.
const compileModel = (model) => {
  model.compile({
    optimizer: tf.train.adam(0.001),
    loss: 'binaryCrossentropy',
    metrics: ['accuracy'],
  });

  return model;
};

// A penalidade L2 precisa de uma instância por camada, mas a intensidade
// é a mesma em todas. `null` é o valor que o tfjs entende como "sem
// regularizador", então desligar o freio não exige um segundo caminho de
// código — só um argumento diferente.
const createRegularizer = (l2 = L2_LAMBDA) =>
  (l2 > 0 ? tf.regularizers.l2({ l2 }) : null);

// O número de entradas vem da fonte de dados: o sintético tem 4 features,
// o German Credit tem 19 ou 57 conforme a codificação. Era a única parte
// da rede que precisava saber qual dataset está em uso — os dois freios
// entraram como argumento pelo mesmo motivo: para poderem ser desligados
// e medidos, em vez de aceitos.
const buildModel = (inputSize = 4, options = {}) => {
  const {
    units = HIDDEN_UNITS,
    l2 = L2_LAMBDA,
    dropout = DROPOUT_RATE,
  } = options;
  const model = tf.sequential();

  // Com `dropout: 0` a camada é OMITIDA, não adicionada com taxa zero.
  // Uma camada inerte apareceria no `model.summary()` e no `model.json`
  // salvo em disco sugerindo um freio que não existe.
  const addHidden = (units, shape = {}) => {
    model.add(tf.layers.dense({
      ...shape,
      units,
      activation: 'relu',
      kernelRegularizer: createRegularizer(l2),
    }));

    if (dropout > 0) {
      model.add(tf.layers.dropout({ rate: dropout }));
    }
  };

  // A primeira camada é a única que declara o formato da entrada; as
  // demais o inferem da anterior.
  units.forEach((count, index) => addHidden(
    count,
    index === 0 ? { inputShape: [inputSize] } : {},
  ));

  // A saída não leva dropout. Descartar a única unidade que produz a
  // resposta não removeria um caminho redundante — apagaria a predição.
  //
  // Com `units: []` não há camada oculta nenhuma, e é a saída que passa a
  // declarar a entrada. O que sobra é uma REGRESSÃO LOGÍSTICA: uma soma
  // ponderada das features passando por uma sigmoide, sem não-linearidade
  // no meio. É o piso de arquitetura, e existe para ser medido — se a
  // rede não bater a linha reta, as camadas ocultas não estão pagando o
  // próprio custo.
  model.add(tf.layers.dense({
    ...(units.length === 0 ? { inputShape: [inputSize] } : {}),
    units: 1,
    activation: 'sigmoid',
    kernelRegularizer: createRegularizer(l2),
  }));

  return compileModel(model);
};

// A configuração de treino fica em um lugar só porque DOIS caminhos a
// usam: o fluxo principal e a validação cruzada. Se elas divergissem, a
// estimativa cruzada estaria medindo um modelo que o projeto não entrega.
const TRAINING = {
  epochs: 40,
  batchSize: 32,
  validationSplit: 0.2,
  patience: 5,
};

// Peso por classe no estilo "balanced": cada classe recebe n / (k · n_c).
// Com 30% de inadimplentes, o positivo passa a valer ~1,67 e o negativo
// ~0,71 — as duas classes somam o mesmo peso total, e errar um positivo
// deixa de ser barato só por haver menos deles.
//
// É o único lugar do projeto em que o desbalanceamento é atacado DURANTE
// o treino. O ajuste do limiar age depois, sobre uma rede que já
// aprendeu de uma população torta; o peso muda o que ela aprende. São
// correções diferentes para o mesmo problema, e por isso comparáveis.
//
// Devolve `null` quando uma das classes não aparece: um peso infinito
// para a classe ausente não corrigiria nada e quebraria o treino.
const balancedClassWeight = (labels) => {
  const flat = labels.map((label) => (Array.isArray(label) ? label[0] : label));
  const positives = flat.filter((risk) => risk === 1).length;
  const negatives = flat.length - positives;

  if (positives === 0 || negatives === 0) {
    return null;
  }

  return {
    0: flat.length / (2 * negatives),
    1: flat.length / (2 * positives),
  };
};

const fitModel = (model, xTrain, yTrain, options = {}) => {
  // `validationSplit` e `validationData` são mutuamente exclusivos no
  // tfjs. Quando quem chama traz o próprio conjunto de validação — e
  // desde a fatia de calibração ele traz —, é ele que manda, e o corte
  // automático dos últimos 20% sai de cena.
  const { validationData, validationSplit, ...rest } = options;
  const validacao = validationData
    ? { validationData }
    : { validationSplit: validationSplit ?? TRAINING.validationSplit };

  return model.fit(xTrain, yTrain, {
    epochs: TRAINING.epochs,
    batchSize: TRAINING.batchSize,
    ...validacao,
    shuffle: true,
    callbacks: [
      tf.callbacks.earlyStopping({
        monitor: 'val_loss',
        patience: TRAINING.patience,
      }),
    ],
    ...rest,
  });
};

module.exports = {
  compileModel,
  createRegularizer,
  buildModel,
  TRAINING,
  balancedClassWeight,
  fitModel,
};
