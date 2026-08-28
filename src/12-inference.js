const tf = require('@tensorflow/tfjs-node');

const { toFeatureVector } = require('./01-preprocess');

// --------------------------------------------------
// 12. Inferência
// --------------------------------------------------
// Encapsula o ciclo tensor → predição → dispose para que nenhum
// tensor intermediário escape em quem chama.
// `toVector` é injetável porque cada fonte normaliza de um jeito — e o
// dataset real só sabe normalizar depois de ver o conjunto de treino.
const predictRisk = (model, customer, toVector = toFeatureVector) => {
  const input = tf.tensor2d([toVector(customer)]);
  const prediction = model.predict(input);
  const probability = prediction.dataSync()[0];

  tf.dispose([input, prediction]);

  return probability;
};

module.exports = { predictRisk };
