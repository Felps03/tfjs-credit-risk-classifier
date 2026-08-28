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
        const { step, suffix } = describeControl(field);

        return { ...base, step, suffix, observed: observedRange[field] ?? null };
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
