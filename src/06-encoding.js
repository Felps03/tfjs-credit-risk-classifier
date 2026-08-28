const { GERMAN_CATEGORICAL, GERMAN_NUMERIC } = require('./00-constants');
const { applyMinMaxScaler } = require('./05-scaler');

// --------------------------------------------------
// 6. Codificação das colunas qualitativas
// --------------------------------------------------
// Codificar categoria como número inteiro (ordinal) é conveniente e, na
// maioria das colunas, é uma MENTIRA: diz à rede que `purpose = 3` fica
// entre `2` e `4`, quando "rádio/TV", "eletrodoméstico" e "reparos" não
// têm ordem nenhuma entre si.
//
// One-hot desfaz essa suposição. Cada categoria vira uma coluna própria,
// que vale 1 quando é aquela e 0 nas outras — nenhuma fica "maior" que
// as demais, e a rede aprende um peso independente para cada uma.
//
//   purpose = 3  →  ordinal: [0.333]
//                   one-hot: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
const oneHotEncode = (size, index) =>
  Array.from({ length: size }, (unused, position) => (position === index ? 1 : 0));

// Codificação ordinal normalizada, mantida para comparação. Divide pelo
// maior índice para cair em [0, 1], igual às numéricas.
const ordinalEncode = (size, index) => [size > 1 ? index / (size - 1) : 0];

const ENCODERS = {
  onehot: oneHotEncode,
  ordinal: ordinalEncode,
};

// Nome de cada posição do vetor final. Com 57 entradas, saber qual coluna
// é qual deixa de ser óbvio — e o nome é o que liga um peso da rede de
// volta a um fato sobre o cliente.
const germanFeatureNames = (encoding) => [
  ...GERMAN_NUMERIC,
  ...Object.entries(GERMAN_CATEGORICAL).flatMap(([field, codes]) => (
    encoding === 'onehot'
      ? codes.map((code) => `${field}=${code}`)
      : [field]
  )),
];

// Numéricas escaladas + qualitativas codificadas, sempre nessa ordem.
// A ordem precisa ser estável: é ela que casa cada valor com a entrada
// correspondente da rede, no treino e na inferência.
const toGermanVector = (customer, scaler, encoding) => {
  const encode = ENCODERS[encoding];

  return [
    ...applyMinMaxScaler(scaler, customer),
    ...Object.entries(GERMAN_CATEGORICAL).flatMap(([field, codes]) =>
      encode(codes.length, customer[field])),
  ];
};

module.exports = {
  oneHotEncode,
  ordinalEncode,
  ENCODERS,
  germanFeatureNames,
  toGermanVector,
};
