const path = require('node:path');

const tf = require('@tensorflow/tfjs-node');

const { MODEL_DIR } = require('./00-constants');
const { compileModel } = require('./10-model');

// --------------------------------------------------
// 11. Persistência
// --------------------------------------------------
// O tfjs-node registra o esquema `file://`. Salvar em `./model`
// gera dois arquivos:
//   model.json  → topologia + metadados (e o training config)
//   weights.bin → os pesos em binário
const toFileUrl = (dir) => `file://${path.resolve(dir)}`;

// `includeOptimizer: true` salva também o estado do Adam.
// Sem ele o modelo recarregado vem SEM compilação: dá para prever,
// mas não para avaliar ou continuar o treino sem recompilar.
const saveModel = (model, dir = MODEL_DIR) =>
  model.save(toFileUrl(dir), { includeOptimizer: true });

const loadModel = async (dir = MODEL_DIR) => {
  const model = await tf.loadLayersModel(`${toFileUrl(dir)}/model.json`);

  // Rede de segurança: se o modelo veio sem otimizador, recompila com
  // exatamente a mesma configuração usada no treino.
  if (!model.optimizer) {
    compileModel(model);
  }

  return model;
};

module.exports = {
  toFileUrl,
  saveModel,
  loadModel,
};
