import { describeControl, describeField, formatValue } from './domain.js';

// --------------------------------------------------
// DTO → dados do componente
// --------------------------------------------------
// O componente recebe `inputs`, `layers`, `connections` e `results`, e
// nunca um corpo de resposta HTTP. Toda a tradução acontece aqui, em
// funções puras — o que significa que a visualização pode ser montada a
// partir do mock, de um teste ou da API sem saber a diferença.
//
// A topologia também sai daqui, e não do componente, porque ela é um
// FATO do pipeline e não uma decisão de desenho: as colunas numéricas
// realmente passam pelo min-max, as qualitativas realmente passam pelo
// one-hot, e as duas coisas realmente se encontram só na primeira camada
// oculta. Desenhar uma ligação que não existe seria ilustrar outra coisa.

// Quantos círculos representam uma camada. Uma camada de 16 unidades não
// vira 16 bolinhas: viraria uma parede ilegível. O número real fica na
// legenda da coluna, e a legenda diz quantos estão desenhados — a tela
// não finge que a camada tem o tamanho que aparece.
const representar = (units) => (units >= 12 ? 4 : units >= 6 ? 3 : 2);

const ENCODING_LABEL = {
  onehot: { label: 'One-hot', detail: 'uma coluna por categoria' },
  ordinal: { label: 'Codificação ordinal', detail: 'índice normalizado' },
};

// Cada campo vira o descritor de um CONTROLE, não de um texto: o valor
// cru (que é o que volta para o payload), o texto formatado (que é o que
// se lê) e o que o controle precisa para existir — passo e unidade nas
// numéricas, a lista de opções nas qualitativas.
//
// A ordem das opções é a do contrato, e o índice É o valor enviado. Por
// isso as opções carregam `value: index` e não o código: `purpose = 3` é
// o que o serviço espera, "rádio/TV" é o que a pessoa escolhe.
export const toInputs = (customer, request, observedRange = {}) => {
  const ordem = [...request.numeric, ...Object.keys(request.categorical)];

  return ordem
    .filter((field) => field in customer)
    .map((field) => {
      const { label, icon, group, codes } = describeField(field);
      const base = {
        id: field,
        label,
        icon,
        group,
        value: customer[field],
        text: formatValue(field, customer[field]),
      };

      if (group !== 'categorical') {
        const { step, suffix, min, max } = describeControl(field);

        return { ...base, step, suffix, min, max, observed: observedRange[field] ?? null };
      }

      return {
        ...base,
        options: (request.categorical[field]?.codes ?? []).map((code, index) => ({
          value: index,
          code,
          label: codes?.[index] ?? code,
        })),
      };
    });
};

// Os campos que o serviço RECUSA. Eles não entram na rede e não deveriam
// sumir da tela em silêncio: a auditoria de viés do projeto depende
// justamente de o modelo não ter visto esta coluna, e isso é uma decisão
// que vale mostrar.
export const toRejected = (request) => (request.rejected ?? []).map((field) => {
  const { label, icon } = describeField(field);

  return { id: field, label, icon };
});

export const toLayers = ({ request, model, encoding }) => {
  const numericas = request.numeric.length;
  const categoricas = Object.entries(request.categorical);
  const colunasOneHot = categoricas.reduce((total, [, spec]) => total + spec.codes.length, 0);
  const codificacao = ENCODING_LABEL[encoding] ?? ENCODING_LABEL.onehot;

  const preparo = {
    id: 'preparo',
    title: 'Preparo',
    caption: `${model.features} entradas`,
    nodes: [
      numericas > 0 && {
        id: 'escala',
        label: 'Escala min–max',
        detail: `${numericas} colunas numéricas, na escala medida no treino`,
      },
      categoricas.length > 0 && {
        id: 'codificacao',
        label: codificacao.label,
        detail: `${categoricas.length} qualitativas → ${encoding === 'onehot' ? colunasOneHot : categoricas.length} colunas (${codificacao.detail})`,
      },
    ].filter(Boolean),
  };

  const ocultas = (model.units ?? []).map((units, index) => {
    const desenhados = representar(units);

    return {
      id: `oculta-${index + 1}`,
      title: `Camada oculta ${index + 1}`,
      caption: `${units} unidades · ${desenhados} representadas`,
      nodes: Array.from({ length: desenhados }, (unused, position) => ({
        id: `u${position + 1}`,
        label: `Unidade ${position + 1} de ${units}`,
        detail: 'combina todos os sinais da camada anterior (ReLU)',
      })),
    };
  });

  const saida = {
    id: 'saida',
    title: 'Saída',
    caption: 'sigmoide',
    nodes: [{
      id: 'sigmoide',
      label: 'Sigmoide',
      detail: 'comprime a combinação final em uma probabilidade entre 0 e 1',
    }],
  };

  return [preparo, ...ocultas, saida];
};

// Âncoras. São só strings, mas são o contrato entre o que os painéis
// renderizam e o que a camada de SVG procura no DOM — por isso vivem
// numa função só, e não espalhadas por três componentes.
export const anchorId = {
  input: (field) => `in:${field}`,
  node: (layerId, nodeId) => `node:${layerId}:${nodeId}`,
  result: (id) => `out:${id}`,
};

// Uma a cada `PASSO` ligações ganha o ponto que percorre o traço. Todas
// animadas viraria um formigueiro; nenhuma não diria que há algo
// acontecendo. O critério é o índice, não sorteio: a mesma entrada
// desenha sempre a mesma cena, o que importa quando alguém compara duas
// execuções lado a lado.
const PASSO_ANIMADO = 5;

export const toConnections = ({ inputs, layers, results }) => {
  const links = [];

  const preparo = layers[0];
  const escala = preparo.nodes.find((node) => node.id === 'escala');
  const codificacao = preparo.nodes.find((node) => node.id === 'codificacao');

  // Entrada → preparo: cada coluna vai para o tratamento que REALMENTE
  // recebe ela. É onde o desenho para de ser decorativo.
  inputs.forEach((input) => {
    const destino = input.group === 'numeric' ? escala : codificacao;

    if (destino) {
      links.push({
        from: anchorId.input(input.id),
        to: anchorId.node(preparo.id, destino.id),
      });
    }
  });

  // Preparo → oculta → oculta → saída: densas, todas com todas, porque a
  // rede é densa. É aqui que os cruzamentos aparecem, e eles não são
  // enfeite: é literalmente cada entrada influenciando cada unidade.
  layers.slice(0, -1).forEach((layer, index) => {
    const proxima = layers[index + 1];

    layer.nodes.forEach((origem) => {
      proxima.nodes.forEach((destino) => {
        links.push({
          from: anchorId.node(layer.id, origem.id),
          to: anchorId.node(proxima.id, destino.id),
        });
      });
    });
  });

  // Saída → resultados.
  const saida = layers[layers.length - 1];

  results.forEach((result) => {
    saida.nodes.forEach((node) => {
      links.push({
        from: anchorId.node(saida.id, node.id),
        to: anchorId.result(result.id),
        emphasis: true,
      });
    });
  });

  return links.map((link, index) => ({
    ...link,
    id: `${link.from}->${link.to}`,
    animated: index % PASSO_ANIMADO === 0,
    delay: (index % 7) * 0.42,
  }));
};

// Os dois lados da mesma probabilidade. Não são três classes: o modelo
// tem UMA saída, e inventar uma terceira faixa para preencher a tela
// seria apresentar uma decisão de layout como se fosse do modelo.
export const toResults = ({ riskProbability, threshold }) => [
  {
    id: 'risco',
    label: 'Risco de inadimplência',
    value: riskProbability,
    tone: 'warn',
    description: 'Probabilidade de o contrato não ser honrado, estimada pela rede.',

    // O limiar só faz sentido sobre ESTA barra: é ela que ele corta.
    marker: { value: threshold, label: 'limiar de decisão' },
  },
  {
    id: 'adimplencia',
    label: 'Probabilidade de adimplência',
    value: Math.max(0, 1 - riskProbability),
    tone: 'success',
    description: 'O complemento da primeira — a mesma saída, vista pelo outro lado.',
  },
];

export const toDecision = ({ classification, threshold }, schema) => ({
  classification,
  alto: classification === 'HIGH_RISK',
  label: classification === 'HIGH_RISK' ? 'Alto risco' : 'Baixo risco',
  threshold,
  strategy: schema.thresholdStrategy ?? '',
  explanation: classification === 'HIGH_RISK'
    ? 'A probabilidade ficou acima do limiar, então o contrato é sinalizado.'
    : 'A probabilidade ficou abaixo do limiar, então o contrato segue.',
});

// --------------------------------------------------
// Modo treinamento
// --------------------------------------------------
// Tudo aqui sai do pacote gravado por `npm start`. A curva é a curva
// daquele treino, não uma curva bonita desenhada para ilustrar: os
// números vêm do `history` que o `model.fit` devolveu, época por época.
//
// A ANIMAÇÃO da rede é outra coisa, e a tela diz isso: ela é um esquema
// do algoritmo — passo à frente, erro, passo atrás, ajuste. Não há
// registro dos pesos época a época, então fingir que os círculos mostram
// os pesos mudando seria inventar. O que se anima é o PROCEDIMENTO; o
// que se mede é a curva ao lado.

// Parâmetros de uma MLP densa: cada camada tem pesos (entrada × saída)
// mais um viés por unidade. Calculado aqui em vez de publicado porque é
// determinado pela topologia, e uma conta que se refaz não desatualiza.
export const countParams = (features, units = []) => {
  const camadas = [features, ...units, 1];

  return camadas
    .slice(1)
    .reduce((total, saida, index) => total + camadas[index] * saida + saida, 0);
};

const inteiro = new Intl.NumberFormat('pt-BR');

export const toTraining = (schema) => {
  const training = schema.training ?? null;
  const loss = training?.history?.loss ?? [];
  const valLoss = training?.history?.valLoss ?? [];

  // Pacote salvo antes de o histórico existir. Não é erro: é um pacote
  // antigo, e a tela precisa dizer isso em vez de desenhar uma curva
  // vazia como se fosse uma curva plana.
  if (loss.length === 0) {
    return null;
  }

  const epochs = loss.length;
  const melhor = valLoss.indexOf(Math.min(...valLoss));
  const validacao = Math.round(training.customers * (training.validationSplit ?? 0));
  const ajuste = training.customers - validacao;

  return {
    epochs,
    loss,
    valLoss,

    // A época em que a validação foi melhor, e a última que rodou. A
    // distância entre as duas é exatamente a paciência do early stopping
    // — é ali que dá para VER por que o treino parou.
    best: melhor,
    limit: training.epochs,
    patience: training.patience,
    interrompido: epochs < training.epochs,

    facts: [
      {
        id: 'clientes',
        icon: 'users',
        label: `${inteiro.format(training.customers)} clientes de treino`,
        detail: `${inteiro.format(ajuste)} para ajustar os pesos, `
          + `${inteiro.format(validacao)} separados para validar`,
      },
      {
        id: 'topologia',
        icon: 'layers',
        label: `${schema.model.features} → ${(training.units ?? []).join(' → ')} → 1`,
        detail: `${inteiro.format(countParams(schema.model.features, training.units))} `
          + 'parâmetros para ajustar',
      },
      {
        id: 'lote',
        icon: 'gauge',
        label: `Lotes de ${training.batchSize}`,
        detail: 'os pesos são corrigidos a cada lote, não a cada cliente',
      },
      {
        id: 'freios',
        icon: 'shield',
        label: `L2 ${training.l2} · dropout ${training.dropout}`,
        detail: 'os dois freios contra decorar o treino',
      },
      {
        id: 'parada',
        icon: 'clock',
        label: `Parou na época ${epochs} de ${training.limit ?? training.epochs}`,
        detail: `a validação não melhorava havia ${training.patience} épocas`,
      },
    ],
  };
};

// Os quatro números que resumem o pacote servido. Todos saem do
// contrato: nenhum é digitado, e um retreino que mude a topologia muda a
// faixa sozinho.
export const toStats = (schema) => {
  const training = schema.training ?? {};
  const epocas = training.history?.loss?.length ?? null;

  return [
    { id: 'entradas', valor: String(schema.model.features), rotulo: 'entradas na rede' },
    {
      id: 'parametros',
      valor: inteiro.format(countParams(schema.model.features, schema.model.units)),
      rotulo: 'parâmetros ajustados',
    },
    epocas
      ? { id: 'epocas', valor: String(epocas), rotulo: 'épocas até parar' }
      : null,
    {
      id: 'limiar',
      valor: `${(schema.threshold * 100).toFixed(1).replace('.', ',')}%`,
      rotulo: 'limiar de decisão',
    },
  ].filter(Boolean);
};

// --------------------------------------------------
// Modo avaliação
// --------------------------------------------------
// O que o modelo VALE, medido no conjunto de teste. Tudo vem do pacote:
// nenhum número aqui é recalculado no navegador, porque recalcular
// exigiria os dados de teste — e eles não saem do servidor.
export const toEvaluation = (schema) => {
  const avaliacao = schema.evaluation ?? null;

  if (!avaliacao) {
    return null;
  }

  const { confusion, metrics, audit } = avaliacao;

  // Qual dos três cortes é o que está decidindo. A comparação é por
  // tolerância porque o limiar do pacote tem seis casas e os candidatos
  // têm quatro — comparar por igualdade não acharia nenhum.
  const thresholds = (avaliacao.thresholds ?? []).map((corte) => ({
    ...corte,
    ativo: corte.threshold !== null
      && Math.abs(corte.threshold - schema.threshold) < 5e-4,
  }));

  return {
    accuracy: avaliacao.testAccuracy,
    baseline: avaliacao.baseline,
    auc: avaliacao.auc,

    // O termômetro do overfitting: quanto ele vai melhor no que já viu.
    gap: avaliacao.trainAccuracy - avaliacao.testAccuracy,
    testCustomers: avaliacao.testCustomers,

    // A acurácia sozinha não diz nada. `ganho` é o que o treino
    // acrescentou sobre chutar a classe majoritária — e é ele, não a
    // acurácia, que responde "o modelo aprendeu alguma coisa?".
    ganho: avaliacao.testAccuracy - avaliacao.baseline,

    metrics,
    costs: avaliacao.costs,

    // Onde os três cortes foram medidos. Sem este número a faixa do topo
    // diria "custo 65" ao lado de uma matriz de teste que custa 104, e
    // pareceria contradição — quando é justamente a separação entre
    // escolher e medir funcionando.
    calibrationCustomers: schema.training?.calibrationCustomers ?? null,
    thresholds,

    confusion: [
      {
        id: 'tn',
        valor: confusion.trueNegatives,
        sigla: 'TN',
        titulo: 'Acertou o bom pagador',
        real: 'Real BAIXO', predito: 'Predito BAIXO',
        tom: 'success',
      },
      {
        id: 'fp',
        valor: confusion.falsePositives,
        sigla: 'FP',
        titulo: 'Recusou quem pagaria',
        real: 'Real BAIXO', predito: 'Predito ALTO',
        tom: 'warn',
      },
      {
        id: 'fn',
        valor: confusion.falseNegatives,
        sigla: 'FN',
        titulo: 'Deixou passar quem não paga',
        real: 'Real ALTO', predito: 'Predito BAIXO',
        tom: 'danger',
      },
      {
        id: 'tp',
        valor: confusion.truePositives,
        sigla: 'TP',
        titulo: 'Pegou o inadimplente',
        real: 'Real ALTO', predito: 'Predito ALTO',
        tom: 'success',
      },
    ],

    audit: audit ? {
      politica: audit.politica,
      approvalRatio: audit.approvalRatio,

      // A regra dos quatro quintos (EEOC): abaixo de 0,80 liga o alerta
      // de impacto desigual. Não é lei brasileira nem prova de
      // discriminação — é um termômetro consagrado.
      dentroDaRegra: audit.approvalRatio >= 0.8,
      grupos: [
        { id: 'women', rotulo: 'Mulheres', ...audit.women },
        { id: 'men', rotulo: 'Homens', ...audit.men },
      ],
    } : null,
  };
};

// No modo treinamento não há dezenove entradas: o card da esquerda passa
// a descrever o treino, e o feixe sai dele inteiro. Os pontos das pontas
// mudam; o miolo — camada a camada — é exatamente o mesmo.
export const toTrainingConnections = (layers) => {
  const links = [];
  const preparo = layers[0];
  const saida = layers[layers.length - 1];

  preparo.nodes.forEach((node) => {
    links.push({ from: 'stage:left', to: anchorId.node(preparo.id, node.id) });
  });

  layers.slice(0, -1).forEach((layer, index) => {
    layer.nodes.forEach((origem) => {
      layers[index + 1].nodes.forEach((destino) => {
        links.push({
          from: anchorId.node(layer.id, origem.id),
          to: anchorId.node(layers[index + 1].id, destino.id),
        });
      });
    });
  });

  saida.nodes.forEach((node) => {
    links.push({ from: anchorId.node(saida.id, node.id), to: 'stage:right', emphasis: true });
  });

  return links.map((link, index) => ({
    ...link,
    id: `${link.from}->${link.to}`,
    animated: true,
    delay: (index % 6) * 0.14,
  }));
};

// A função que a página inteira usa: DTOs entram, os dados do componente
// saem. Nada aqui toca no DOM, e nada aqui faz requisição.
export const toFlowData = ({ schema, score, customer }) => {
  const inputs = toInputs(customer, schema.request, schema.observedRange);
  const layers = toLayers(schema);
  const results = toResults(score);

  return {
    inputs,
    rejected: toRejected(schema.request),
    layers,
    results,
    connections: toConnections({ inputs, layers, results }),
    decision: toDecision(score, schema),

    // O cliente cru viaja junto: é ele que o formulário edita e reenvia,
    // e é o `example` que o botão de restaurar devolve.
    customer,
    example: schema.example,
    meta: {
      source: schema.label ?? schema.source,
      features: schema.model.features,
      units: schema.model.units ?? [],
      encoding: schema.encoding,
      savedAt: schema.model.savedAt,
    },
  };
};
