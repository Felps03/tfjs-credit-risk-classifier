// --------------------------------------------------
// 5. Normalização ajustada no treino
// --------------------------------------------------
// O dataset sintético podia usar constantes fixas: nós geramos os dados,
// então conhecíamos as faixas de antemão. Com dado real não existe esse
// luxo — as faixas precisam ser MEDIDAS. E medidas só no treino.
//
// Calcular min/max sobre o dataset inteiro parece inofensivo e não é: o
// maior empréstimo do conjunto de teste passaria a influenciar a escala
// aplicada no treino. Isso é vazamento (data leakage), e o efeito é
// sempre o mesmo — o modelo parece melhor na avaliação do que será
// diante de dados que nunca viu.
const fitMinMaxScaler = (customers, featureNames) => {
  const min = {};
  const range = {};

  featureNames.forEach((feature) => {
    const values = customers.map((customer) => customer[feature]);
    const lowest = Math.min(...values);
    const highest = Math.max(...values);

    min[feature] = lowest;
    // Coluna constante daria divisão por zero; 1 mantém o valor em 0.
    range[feature] = highest - lowest || 1;
  });

  return { featureNames, min, range };
};

// Um valor de teste fora da faixa vista no treino sai de [0, 1] de
// propósito. Cortar em 0 e 1 esconderia justamente o caso extremo que o
// modelo nunca viu — e é sobre ele que se quer saber.
const applyMinMaxScaler = ({ featureNames, min, range }, customer) =>
  featureNames.map((feature) => (customer[feature] - min[feature]) / range[feature]);

module.exports = {
  fitMinMaxScaler,
  applyMinMaxScaler,
};
