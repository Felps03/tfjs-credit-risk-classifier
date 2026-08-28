const tf = require('@tensorflow/tfjs-node');

// Acurácia de quem sempre chuta a classe majoritária, sem olhar para
// nenhuma feature. É o PISO: um modelo que não supera este número não
// aprendeu nada aproveitável, por mais alta que a acurácia pareça.
const majorityBaseline = (labels) => {
  const positives = labels.filter(([risk]) => risk === 1).length;

  return Math.max(positives, labels.length - positives) / labels.length;
};

const evaluateModel = (model, xTest, yTest) => {
  const [lossTensor, accuracyTensor] = model.evaluate(xTest, yTest);
  const loss = lossTensor.dataSync()[0];
  const accuracy = accuracyTensor.dataSync()[0];

  tf.dispose([lossTensor, accuracyTensor]);

  return { loss, accuracy };
};

module.exports = {
  majorityBaseline,
  evaluateModel,
};
