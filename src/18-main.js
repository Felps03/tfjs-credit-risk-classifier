const tf = require('@tensorflow/tfjs-node');

const {
  CALIBRATION_SPLIT,
  DECISION_THRESHOLD,
  HIDDEN_UNITS,
  MODEL_DIR,
  SHUFFLE_SEED,
} = require('./00-constants');
const {
  fitGroupThresholds,
  formatAudit,
  formatMitigation,
  summarizeDecisions,
} = require('./07-audit');
const { computeConfusionMatrix, formatConfusionMatrix } = require('./13-confusion');
const { evaluateModel, majorityBaseline } = require('./16a-evaluate');
const { predictRisk } = require('./12-inference');
const { computeMetrics, formatMetrics, safeDivide } = require('./14-metrics');
const { TRAINING, balancedClassWeight, buildModel, fitModel } = require('./10-model');
const { loadModel } = require('./11-persistence');
const { round, saveArtifacts } = require('./19-artifacts');
const { classify } = require('./01-preprocess');
const { computeRocCurve, formatRocCurve } = require('./15-roc');
const { shuffle, splitCalibration, stratifiedSplitCustomers } = require('./09-split');
const { DEFAULT_SOURCE_ID, SOURCES } = require('./08-sources');
const { createRandom } = require('./02-synthetic');
const {
  chooseThresholdByCost,
  chooseThresholdByYouden,
  FALSE_NEGATIVE_COST,
  FALSE_POSITIVE_COST,
  formatThreshold,
  formatThresholdComparison,
  scorePoint,
} = require('./16-threshold');

// --------------------------------------------------
// 18. Treinar, avaliar, salvar, recarregar e prever
// --------------------------------------------------
const main = async (sourceId = DEFAULT_SOURCE_ID, overrides = {}) => {
  const source = SOURCES[sourceId];
  const {
    mitigate = false,
    balance = false,
    units = HIDDEN_UNITS,
    ...regularization
  } = overrides;

  // A fonte decide; a linha de comando tem a última palavra.
  const { l2, dropout } = { ...source.regularization, ...regularization };

  // O sintético se gera se faltar; o real só confere que está em disco.
  source.ensure();

  console.log('Fonte:', source.label);
  console.log('Arquivo:', source.csvPath);

  // Embaralhar ANTES de separar. O arquivo real chega na ordem em que foi
  // coletado, e essa ordem não é aleatória — pode concentrar um perfil de
  // cliente em um trecho e outro perfil no resto.
  const customers = shuffle(await source.read(), createRandom(SHUFFLE_SEED));

  console.log('Clientes lidos:', customers.length);
  console.log('Features:', source.featureNames.join(', '));

  // Estratificado: o teste recebe a MESMA proporção de inadimplentes do
  // arquivo, então o piso da classe majoritária deixa de depender do
  // sorteio. Sem isso, o número contra o qual toda a acurácia é lida
  // muda de execução para execução por motivo nenhum.
  const { trainCustomers, testCustomers } = stratifiedSplitCustomers(customers);

  // O treino se parte de novo. A fatia de calibração faz duas coisas —
  // valida o early stopping e escolhe o limiar —, e nenhuma delas pode
  // acontecer no teste: escolher o corte onde ele é medido otimiza para
  // aquele sorteio e publica um número que não se sustenta em produção.
  const { fitCustomers, calibrationCustomers } = splitCalibration(trainCustomers);

  // A escala continua sendo medida no TREINO INTEIRO — calibração
  // inclusive. Ela não é escolhida nem ajustada, é medida; e o pacote que
  // vai para o serviço precisa da escala do maior conjunto disponível
  // antes do teste.
  const scaler = source.fitScaler(trainCustomers);
  const toVector = (customer) => source.toVector(customer, scaler);

  const fitLabels = fitCustomers.map(({ risk }) => [risk]);
  const trainLabels = trainCustomers.map(({ risk }) => [risk]);
  const testLabels = testCustomers.map(({ risk }) => [risk]);

  const xFit = tf.tensor2d(fitCustomers.map(toVector));
  const yFit = tf.tensor2d(fitLabels);
  const xCal = tf.tensor2d(calibrationCustomers.map(toVector));
  const yCal = tf.tensor2d(calibrationCustomers.map(({ risk }) => [risk]));
  const xTrain = tf.tensor2d(trainCustomers.map(toVector));
  const yTrain = tf.tensor2d(trainLabels);
  const xTest = tf.tensor2d(testCustomers.map(toVector));
  const yTest = tf.tensor2d(testLabels);

  console.log(
    `Divisão: ${fitCustomers.length} para ajustar os pesos, `
    + `${calibrationCustomers.length} para calibrar (validação + limiar), `
    + `${testCustomers.length} para testar.`,
  );

  const model = buildModel(source.featureNames.length, { units, l2, dropout });

  console.log('Arquitetura:', units.length === 0
    ? 'regressão logística (sem camada oculta)'
    : units.join(' → '));
  console.log(`Regularização: L2 = ${l2}, dropout = ${dropout}`);
  model.summary();

  // O `fit` devolve o histórico por época e ele SEMPRE foi descartado
  // aqui. Guardá-lo custa alguns kilobytes no pacote e paga por si: é a
  // única evidência do que aconteceu durante o treino, e sem ela o laço
  // entre treinamento e validação — o coração do processo — só existe
  // como afirmação. Com ela, dá para VER a val_loss parar de melhorar
  // enquanto a loss de treino continua caindo, que é exatamente o que o
  // early stopping está cortando.
  // O peso por classe é o único tratamento do desbalanceamento que age
  // DURANTE o treino. Sem ele, errar um inadimplente é barato só porque
  // há menos deles — e a rede aprende exatamente isso.
  const classWeight = balance ? balancedClassWeight(fitLabels) : null;

  console.log('Peso por classe:', classWeight
    ? `balanceado (0 → ${classWeight[0].toFixed(3)}, 1 → ${classWeight[1].toFixed(3)})`
    : 'nenhum (use --balancear)');

  // A validação do early stopping agora é a fatia de calibração, passada
  // explicitamente. Antes ela era os últimos 20% recortados por dentro do
  // `validationSplit` — os mesmos clientes, mas invisíveis e sem nome.
  const { history } = await fitModel(model, xFit, yFit, {
    validationData: [xCal, yCal],
    ...(classWeight === null ? {} : { classWeight }),
  });

  // O mesmo modelo avaliado nos dois conjuntos. `evaluate` roda em modo de
  // inferência, então o dropout está DESLIGADO nas duas medidas — é a
  // comparação justa, e não a acurácia pessimista que o `fit` imprime.
  const { accuracy: trainAccuracy } = evaluateModel(model, xTrain, yTrain);
  const { loss: testLoss, accuracy: testAccuracy } =
    evaluateModel(model, xTest, yTest);

  // A acurácia sozinha não diz se o modelo presta. Ao lado do piso da
  // classe majoritária ela passa a dizer: a distância entre os dois
  // números é tudo que o treino realmente acrescentou.
  const baseline = majorityBaseline(testLabels);

  console.log('Test loss:', testLoss.toFixed(4));
  console.log('Train accuracy:', trainAccuracy.toFixed(4));
  console.log('Test accuracy:', testAccuracy.toFixed(4));
  console.log('Baseline (classe majoritária):', baseline.toFixed(4));

  // O termômetro do overfitting. Quanto o modelo vai melhor no que já viu
  // do que no que não viu é exatamente o que os dois freios existem para
  // encolher — e `--l2=0 --dropout=0` mostra o número sem eles.
  console.log(
    'Diferença treino − teste:',
    (trainAccuracy - testAccuracy).toFixed(4),
  );

  const confusion = computeConfusionMatrix(model, xTest, yTest);

  console.log('\nMatriz de confusão (limiar', `${DECISION_THRESHOLD}):`);
  console.log(formatConfusionMatrix(confusion));

  const metrics = computeMetrics(confusion);

  console.log('');
  console.log(formatMetrics(metrics));

  // Duas curvas, e a diferença entre elas é a correção inteira deste
  // fluxo. A do TESTE é a que se reporta: a AUC não depende de limiar
  // nenhum, então medi-la no teste é legítimo e é o que o pacote publica.
  // A da CALIBRAÇÃO é a que ESCOLHE o corte — e ela nunca viu o teste.
  const { points, auc, positives, negatives } = computeRocCurve(model, xTest, yTest);
  const calibrationRoc = computeRocCurve(model, xCal, yCal);
  const operatingPoint = {
    fpr: safeDivide(
      confusion.falsePositives,
      confusion.falsePositives + confusion.trueNegatives,
    ),
    tpr: metrics.recall,
  };

  console.log('\nCurva ROC (O = limiar', `${DECISION_THRESHOLD}, . = aleatório):`);
  console.log(formatRocCurve(points, { mark: operatingPoint }));
  console.log('AUC:', auc.toFixed(4));

  // ------------------------------------------------
  // Escolher o corte em vez de herdá-lo
  // ------------------------------------------------
  const costs = {
    falsePositive: FALSE_POSITIVE_COST,
    falseNegative: FALSE_NEGATIVE_COST,
  };
  const roc = { points, auc, positives, negatives };

  // A comparação dos três cortes roda sobre a CALIBRAÇÃO. Os FP, FN e
  // custos desta tabela são, portanto, os da calibração — e é por isso
  // que eles não batem com a matriz do teste logo abaixo. Não é
  // inconsistência: é a separação funcionando. Se batessem, seria porque
  // o corte foi escolhido onde é medido.
  // O ponto de operação do corte padrão, medido na calibração. Sem ele o
  // `scorePoint` receberia um ponto sem `fpr` nem `tpr` e devolveria um
  // custo `NaN` — que apareceria na tabela como um traço, e não como erro.
  const calibrationConfusion = computeConfusionMatrix(model, xCal, yCal, DECISION_THRESHOLD);
  const calibrationPoint = {
    threshold: DECISION_THRESHOLD,
    fpr: safeDivide(
      calibrationConfusion.falsePositives,
      calibrationConfusion.falsePositives + calibrationConfusion.trueNegatives,
    ),
    tpr: computeMetrics(calibrationConfusion).recall,
  };

  const candidates = [
    {
      label: 'Padrão (0.5)',
      point: scorePoint(calibrationPoint, calibrationRoc, costs),
    },
    {
      label: 'Youden (max J)',
      point: scorePoint(chooseThresholdByYouden(calibrationRoc), calibrationRoc, costs),
    },
    {
      label: 'Menor custo',
      point: chooseThresholdByCost(calibrationRoc, costs),
    },
  ];

  console.log(
    `\nAjuste do limiar em ${calibrationCustomers.length} clientes de CALIBRAÇÃO`,
    `(FP custa ${FALSE_POSITIVE_COST}, FN custa ${FALSE_NEGATIVE_COST}):`,
  );
  console.log(formatThresholdComparison(candidates));

  const chosen = candidates[2].point;

  // A matriz que vale é a do limiar ESCOLHIDO, não a do 0.5 herdado — e
  // ela existia só dentro do `console.log`. Vira variável para poder ser
  // gravada no pacote junto do resto.
  const confusionEscolhida = computeConfusionMatrix(model, xTest, yTest, chosen.threshold);
  const metricasEscolhidas = computeMetrics(confusionEscolhida);

  console.log(
    `\nMatriz no limiar escolhido (${formatThreshold(chosen.threshold)})`,
    `— ${testCustomers.length} clientes de TESTE:`,
  );
  console.log(formatConfusionMatrix(confusionEscolhida));

  // O número que a separação existe para tornar visível. O corte foi
  // escolhido onde custa `chosen.cost`; ele é aplicado onde custa isto.
  // A distância entre os dois, por cliente, é exatamente o otimismo que
  // escolher o corte no próprio teste escondia — e que este projeto
  // publicava sem saber.
  const custoNoTeste = confusionEscolhida.falsePositives * FALSE_POSITIVE_COST
    + confusionEscolhida.falseNegatives * FALSE_NEGATIVE_COST;
  const porCliente = (custo, n) => (custo / n).toFixed(3);

  console.log(
    `Custo por cliente: ${porCliente(chosen.cost, calibrationCustomers.length)}`,
    `na calibração (onde o corte foi escolhido) ·`,
    `${porCliente(custoNoTeste, testCustomers.length)} no teste (onde ele vale).`,
  );
  console.log(
    'A distância entre os dois é o otimismo de escolher o corte no mesmo',
    'conjunto em que ele é medido — que é o que este fluxo deixou de fazer.',
  );
  console.log('');

  // ------------------------------------------------
  // Auditoria: a coluna que o modelo NÃO recebeu
  // ------------------------------------------------
  // A auditoria nasce dentro do `if` e morria lá. Ela é metade do
  // argumento deste projeto — o modelo não recebe a coluna de sexo, e é
  // JUSTAMENTE por isso que as decisões dele precisam ser medidas por
  // grupo depois. Guardá-la no pacote é o que permite mostrá-la sem
  // reexecutar o treino.
  let auditoria = null;

  if (source.audit) {
    const scores = tf.tidy(() =>
      Array.from(model.predict(xTest).reshape([-1]).dataSync()));

    // Os scores do treino existem aqui por um motivo só: calibrar os
    // limiares por grupo. Eles não entram em nenhuma métrica publicada —
    // quem é auditado continua sendo o conjunto de teste.
    const trainScores = tf.tidy(() =>
      Array.from(model.predict(xTrain).reshape([-1]).dataSync()));

    const policies = [
      { label: 'Limiar único', threshold: chosen.threshold },
      {
        label: 'Limiar por grupo',
        threshold: fitGroupThresholds(trainCustomers, trainScores, chosen.threshold),
      },
    ].map(({ label, threshold }) => ({
      label,
      audit: source.audit(testCustomers, scores, threshold),
      decisions: summarizeDecisions(testCustomers, scores, threshold, costs),
    }));

    const active = policies[mitigate ? 1 : 0];

    auditoria = { politica: active.label, ...active.audit };

    console.log(
      `Auditoria por sexo — ${active.label.toLowerCase()}`,
      '(o modelo nunca recebeu esta coluna):',
    );
    console.log(formatAudit(active.audit));
    console.log('');

    console.log('Mitigação (limiares calibrados no TREINO, auditados no teste):');
    console.log(formatMitigation(policies));
    console.log(mitigate
      ? 'Política ativa: limiar por grupo — a decisão lê o sexo do cliente.'
      : 'Política ativa: limiar único. Use --mitigar para decidir por grupo.');
    console.log('');
  }

  // ------------------------------------------------
  // Novo cliente
  // ------------------------------------------------
  // O cliente de exemplo vem da fonte: as features de um dataset não
  // fazem sentido nenhum no outro.
  const probability = predictRisk(model, source.sampleCustomer, toVector);

  console.log('Probabilidade de alto risco:', probability.toFixed(4));
  console.log('Classificação:', classify(probability));

  // ------------------------------------------------
  // Salvar em disco e descartar o modelo da memória
  // ------------------------------------------------
  // Não são só os pesos: junto vai o contrato que os pesos pressupõem —
  // a escala medida NESTE treino, a ordem das features e o limiar
  // escolhido pela matriz de custo. É o que permite que outro processo
  // (o serviço HTTP, horas depois) normalize um cliente novo do mesmo
  // jeito que a rede aprendeu, em vez de remedir a escala e errar em
  // silêncio.
  await saveArtifacts(model, {
    source: source.id,
    encoding: source.encoding ?? null,
    featureNames: source.featureNames,
    scaler,
    threshold: chosen.threshold,
    thresholdStrategy:
      `menor custo (FP=${FALSE_POSITIVE_COST}, FN=${FALSE_NEGATIVE_COST})`
      + ` em ${calibrationCustomers.length} clientes de calibração`,
    training: {
      customers: trainCustomers.length,

      // As duas fatias do treino, gravadas separadas porque fazem coisas
      // diferentes: uma ajusta os pesos, a outra valida a parada e
      // escolhe o corte. Publicá-las como um número só esconderia
      // exatamente o que este pacote passou a fazer certo.
      fitCustomers: fitCustomers.length,
      calibrationCustomers: calibrationCustomers.length,
      balanced: classWeight !== null,
      units,
      l2,
      dropout,

      // Os hiperparâmetros que governaram ESTE treino. Eles vivem em
      // `TRAINING` e nunca saíam de lá; gravá-los no pacote é o que
      // permite descrever o treino sem abrir o código — e o que denuncia
      // um pacote treinado com outra configuração.
      epochs: TRAINING.epochs,
      batchSize: TRAINING.batchSize,

      // A validação deixou de ser um corte automático e virou um
      // conjunto com nome. A fração continua sendo publicada porque é ela
      // que descreve o tamanho da fatia; o que mudou é que agora ela é
      // aplicada antes do `fit`, e não dentro dele.
      validationSplit: CALIBRATION_SPLIT,
      patience: TRAINING.patience,

      // Quatro casas bastam para desenhar uma curva; a precisão cheia do
      // float só engordaria o arquivo.
      history: {
        loss: (history.loss ?? []).map((valor) => round(Number(valor), 4)),
        valLoss: (history.val_loss ?? []).map((valor) => round(Number(valor), 4)),
      },
    },

    // ------------------------------------------------
    // O que o modelo VALE
    // ------------------------------------------------
    // Tudo isto era calculado, impresso e descartado — a mesma sorte que
    // o histórico de treino teve até pouco tempo atrás. O pacote sabia
    // decidir e não sabia dizer se decide bem.
    //
    // O número que mais importa é o `baseline`: a acurácia de quem chuta
    // a classe majoritária sem olhar para nada. Uma acurácia de 70,5%
    // parece boa até aparecer ao lado de um piso de 70,0%, e a distância
    // entre os dois é tudo que o treino acrescentou. Publicar um sem o
    // outro é publicar meia verdade.
    evaluation: {
      baseline: round(baseline, 4),
      trainAccuracy: round(trainAccuracy, 4),
      testAccuracy: round(testAccuracy, 4),
      testLoss: round(testLoss, 4),
      testCustomers: testCustomers.length,

      // A AUC resume a curva inteira; a acurácia resume UM ponto dela.
      auc: round(auc, 4),

      // A matriz e as métricas no limiar que realmente decide.
      confusion: {
        truePositives: confusionEscolhida.truePositives,
        trueNegatives: confusionEscolhida.trueNegatives,
        falsePositives: confusionEscolhida.falsePositives,
        falseNegatives: confusionEscolhida.falseNegatives,
      },
      metrics: {
        precision: round(metricasEscolhidas.precision, 4),
        recall: round(metricasEscolhidas.recall, 4),
        f1Score: round(metricasEscolhidas.f1Score, 4),
      },

      costs: { falsePositive: FALSE_POSITIVE_COST, falseNegative: FALSE_NEGATIVE_COST },

      // Os três cortes comparados. `Infinity` é um limiar legítimo da
      // curva ROC (o corte que não aprova ninguém) e vira `null` no JSON;
      // gravar `null` de propósito é melhor que gravar um número falso.
      thresholds: candidates.map(({ label, point }) => ({
        label,
        threshold: Number.isFinite(point.threshold) ? round(point.threshold, 4) : null,
        cost: point.cost,
        falsePositives: point.falsePositives,
        falseNegatives: point.falseNegatives,
      })),

      audit: auditoria,
    },
  });
  console.log('Modelo salvo em:', MODEL_DIR);
  console.log('Pacote inclui o scaler e o limiar', chosen.threshold.toFixed(4),
    '— `npm run serve` sobe a API com eles.');

  model.dispose();

  // ------------------------------------------------
  // Recarregar e conferir que nada se perdeu
  // ------------------------------------------------
  const loadedModel = await loadModel();

  const { loss: loadedLoss, accuracy: loadedAccuracy } =
    evaluateModel(loadedModel, xTest, yTest);
  const loadedProbability = predictRisk(loadedModel, source.sampleCustomer, toVector);

  console.log('Modelo recarregado — test loss:', loadedLoss.toFixed(4));
  console.log('Modelo recarregado — test accuracy:', loadedAccuracy.toFixed(4));
  console.log(
    'Modelo recarregado — probabilidade:',
    loadedProbability.toFixed(4),
  );
  console.log(
    'Mesma predição do modelo original?',
    loadedProbability === probability ? 'sim' : 'não',
  );

  // ------------------------------------------------
  // Limpeza de memória
  // ------------------------------------------------
  tf.dispose([xFit, yFit, xCal, yCal, xTrain, yTrain, xTest, yTest]);
  loadedModel.dispose();
};

module.exports = { main };
