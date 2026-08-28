const { CALIBRATION_SPLIT } = require('./00-constants');

// --------------------------------------------------
// 9. Separar treino e teste
// --------------------------------------------------
// Fisher-Yates sobre uma CÓPIA: a ordem original do arquivo é preservada.
const shuffle = (items, random) => {
  const copy = [...items];

  for (let index = copy.length - 1; index > 0; index -= 1) {
    const target = Math.floor(random() * (index + 1));

    [copy[index], copy[target]] = [copy[target], copy[index]];
  }

  return copy;
};

// Separa os clientes ANTES de normalizar. A ordem importa: normalizar
// primeiro faria as estatísticas do teste vazarem para dentro do treino.
//
// Com dado sintético embaralhar é indiferente — cada linha é sorteada de
// forma independente. Com dado real não: um arquivo pode chegar ordenado
// por data, por agência ou pela própria classe, e um corte cru no meio
// separaria dois conjuntos que não representam a mesma população.
const splitCustomers = (customers, trainRatio = 0.8) => {
  const trainSize = Math.floor(customers.length * trainRatio);

  return {
    trainCustomers: customers.slice(0, trainSize),
    testCustomers: customers.slice(trainSize),
  };
};

// Estratificar é preservar a proporção de classes nos dois lados.
//
// O corte cru já embaralhado acerta a proporção EM MÉDIA, e erra em
// qualquer execução específica: com 30% de inadimplentes e 200 linhas de
// teste, o sorteio entrega entre 24% e 36% conforme a semente. Isso move
// o piso da classe majoritária, e o piso é a régua contra a qual toda a
// acurácia deste projeto é lida — ou seja, o ruído do sorteio entra
// direto na conclusão.
//
// A implementação separa os índices por classe, corta cada classe na
// mesma proporção e devolve os clientes na ORDEM original. A ordem
// importa: `fit` reserva os últimos 20% do treino para validação antes
// de embaralhar, então um conjunto agrupado por rótulo daria uma fatia
// de validação quase toda de uma classe só.
const stratifiedSplitCustomers = (customers, trainRatio = 0.8) => {
  const training = new Set();

  [...new Set(customers.map(({ risk }) => risk))].forEach((label) => {
    const indexes = customers
      .map((customer, index) => ({ risk: customer.risk, index }))
      .filter((row) => row.risk === label);

    indexes
      .slice(0, Math.floor(indexes.length * trainRatio))
      .forEach(({ index }) => training.add(index));
  });

  return {
    trainCustomers: customers.filter((ignored, index) => training.has(index)),
    testCustomers: customers.filter((ignored, index) => !training.has(index)),
  };
};

// Separa do TREINO a fatia que vai calibrar o limiar — e que também é a
// validação do early stopping. Não toca no teste: é justamente por não
// tocar nele que o número publicado deixa de ser otimista.
//
// Estratificada pelo mesmo motivo que o corte de fora: uma fatia de
// calibração com metade dos inadimplentes do arquivo escolheria o corte
// para uma população que não existe.
const splitCalibration = (trainCustomers, calibrationRatio = CALIBRATION_SPLIT) => {
  const { trainCustomers: fitCustomers, testCustomers: calibrationCustomers } =
    stratifiedSplitCustomers(trainCustomers, 1 - calibrationRatio);

  return { fitCustomers, calibrationCustomers };
};

// Atribui uma dobra a cada cliente, também mantendo a proporção de
// classes. Distribuir em rodízio DENTRO de cada classe é o que garante
// que nenhuma dobra fique com inadimplentes de menos — com 5 dobras e
// 300 positivos, cada uma recebe 60.
const stratifiedFolds = (customers, folds) => {
  const seen = new Map();

  return customers.map(({ risk }) => {
    const position = seen.get(risk) ?? 0;

    seen.set(risk, position + 1);

    return position % folds;
  });
};

// Variante que opera sobre features JÁ normalizadas. Continua servindo o
// caminho sintético e os testes; o fluxo principal usa o split
// estratificado.
const splitDataset = ({ features, labels }, trainRatio = 0.8) => {
  const trainSize = Math.floor(features.length * trainRatio);

  return {
    trainFeatures: features.slice(0, trainSize),
    trainLabels: labels.slice(0, trainSize),
    testFeatures: features.slice(trainSize),
    testLabels: labels.slice(trainSize),
  };
};

module.exports = {
  shuffle,
  splitCustomers,
  stratifiedSplitCustomers,
  splitCalibration,
  stratifiedFolds,
  splitDataset,
};
