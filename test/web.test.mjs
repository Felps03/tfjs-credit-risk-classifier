import assert from 'node:assert/strict';
import { describe, it } from 'node:test';

import {
  countParams,
  toConnections,
  toDecision,
  toEvaluation,
  toInputs,
  toLayers,
  toRejected,
  toResults,
  toStats,
  toTraining,
  toTrainingConnections,
  anchorId,
} from '../web/js/mappers.js';
import { describeControl, describeField, formatPercent, formatValue } from '../web/js/domain.js';

// --------------------------------------------------
// A camada de tradução da página
// --------------------------------------------------
// `mappers.js` e `domain.js` são funções puras de propósito: elas
// traduzem o DTO do serviço nos dados que o componente desenha, sem
// tocar no DOM e sem fazer requisição. Era a única camada do projeto sem
// teste nenhum, e o que impedia não era o desenho — era o empacotamento:
// o `package.json` da raiz declara `commonjs`, e sem o marcador em
// `web/package.json` o Node recusa os `export` destes arquivos.
//
// O que se testa aqui é o que a tela AFIRMA: que a topologia desenhada é
// a do pipeline, que os números vêm do pacote e que nada é inventado
// para preencher espaço.

// Um contrato mínimo, no formato exato que `GET /schema` publica.
const schema = () => ({
  threshold: 0.1587,
  thresholdStrategy: 'menor custo (FP=1, FN=5) em 160 clientes de calibração',
  encoding: 'onehot',
  label: 'German Credit — UCI/Statlog',
  model: { features: 57, units: [16, 8], savedAt: '2026-08-28T04:07:00.306Z' },
  request: {
    numeric: ['durationMonths', 'age'],
    categorical: {
      checkingStatus: { range: [0, 3], codes: ['A11', 'A12', 'A13', 'A14'] },
      housing: { range: [0, 2], codes: ['A151', 'A152', 'A153'] },
    },
    rejected: ['personalStatus'],
  },
  observedRange: {
    durationMonths: { min: 4, max: 72 },
    age: { min: 19, max: 75 },
  },
  example: { durationMonths: 48, age: 24, checkingStatus: 0, housing: 1 },
});

describe('countParams', () => {
  it('dada a topologia servida, quando contada, então bate com o `model.summary()`', () => {
    // Given — 57 → 16 → 8 → 1 é a rede que o pacote descreve; o
    // `model.summary()` do treino imprime 1073 para ela
    // When / Then
    assert.equal(countParams(57, [16, 8]), 1073);
  });

  it('dada uma rede sem camada oculta, quando contada, então é a regressão logística', () => {
    // Given — 57 pesos mais um viés
    assert.equal(countParams(57, []), 58);
  });

  it('dada uma camada, quando contada, então soma pesos e viés de cada uma', () => {
    // Given — (4×3 + 3) + (3×1 + 1)
    assert.equal(countParams(4, [3]), 19);
  });
});

describe('toResults', () => {
  it('dada uma probabilidade, quando mapeada, então são dois lados da MESMA saída', () => {
    // Given — o modelo tem uma saída; a segunda barra é o complemento
    const [risco, adimplencia] = toResults({ riskProbability: 0.8, threshold: 0.1587 });

    // When / Then
    assert.equal(risco.value, 0.8);
    assert.ok(Math.abs(adimplencia.value - 0.2) < 1e-9);
    assert.equal(risco.value + adimplencia.value, 1);
  });

  it('dado o limiar, quando mapeado, então marca SÓ a barra que ele corta', () => {
    // Given — o limiar corta o risco, não o complemento dele
    const [risco, adimplencia] = toResults({ riskProbability: 0.8, threshold: 0.1587 });

    assert.equal(risco.marker.value, 0.1587);
    assert.equal(adimplencia.marker, undefined);
  });

  it('dada uma probabilidade acima de 1, quando mapeada, então o complemento não fica negativo', () => {
    // Given — barra de largura negativa não existe; o piso é zero
    const [, adimplencia] = toResults({ riskProbability: 1.4, threshold: 0.5 });

    assert.equal(adimplencia.value, 0);
  });
});

describe('toDecision', () => {
  it('dada uma classificação de alto risco, quando mapeada, então carrega o limiar e a estratégia', () => {
    // Given
    const decisao = toDecision(
      { classification: 'HIGH_RISK', threshold: 0.1587 },
      schema(),
    );

    // When / Then
    assert.equal(decisao.alto, true);
    assert.equal(decisao.label, 'Alto risco');
    assert.equal(decisao.threshold, 0.1587);
    assert.match(decisao.strategy, /calibração/);
  });

  it('dado um pacote sem estratégia, quando mapeado, então não vaza `undefined` para a tela', () => {
    // Given — pacote antigo, sem o campo
    const decisao = toDecision({ classification: 'LOW_RISK', threshold: 0.5 }, { });

    assert.equal(decisao.alto, false);
    assert.equal(decisao.strategy, '');
  });
});

describe('toLayers', () => {
  it('dado o contrato, quando mapeado, então a topologia é a do pipeline', () => {
    // Given / When
    const layers = toLayers(schema());

    // Then — preparo, as duas ocultas e a saída, nessa ordem
    assert.deepEqual(layers.map(({ id }) => id), ['preparo', 'oculta-1', 'oculta-2', 'saida']);
  });

  it('dada uma camada de 16 unidades, quando mapeada, então a legenda diz quantas estão desenhadas', () => {
    // Given — desenhar 4 bolinhas e chamá-las de 16 seria mentir; a
    // legenda é o que impede a mentira
    const [, oculta] = toLayers(schema());

    assert.equal(oculta.nodes.length, 4);
    assert.match(oculta.caption, /16 unidades/);
    assert.match(oculta.caption, /4 representadas/);
  });

  it('dado o preparo, quando mapeado, então as qualitativas viram as colunas one-hot que existem', () => {
    // Given — 4 códigos de checkingStatus + 3 de housing = 7 colunas
    const [preparo] = toLayers(schema());
    const codificacao = preparo.nodes.find(({ id }) => id === 'codificacao');

    assert.match(codificacao.detail, /2 qualitativas → 7 colunas/);
  });
});

describe('toConnections', () => {
  it('dado um campo numérico, quando ligado, então ele chega na escala e NÃO no one-hot', () => {
    // Given — é o fato do pipeline que o desenho afirma
    const dados = schema();
    const inputs = toInputs(dados.example, dados.request, dados.observedRange);
    const layers = toLayers(dados);
    const results = toResults({ riskProbability: 0.8, threshold: 0.1587 });
    const links = toConnections({ inputs, layers, results });

    // When
    const daIdade = links.filter(({ from }) => from === anchorId.input('age'));

    // Then
    assert.equal(daIdade.length, 1);
    assert.equal(daIdade[0].to, anchorId.node('preparo', 'escala'));
  });

  it('dado um campo qualitativo, quando ligado, então ele chega no one-hot e NÃO na escala', () => {
    // Given
    const dados = schema();
    const inputs = toInputs(dados.example, dados.request, dados.observedRange);
    const layers = toLayers(dados);
    const results = toResults({ riskProbability: 0.8, threshold: 0.1587 });

    // When
    const daConta = toConnections({ inputs, layers, results })
      .filter(({ from }) => from === anchorId.input('checkingStatus'));

    // Then
    assert.equal(daConta.length, 1);
    assert.equal(daConta[0].to, anchorId.node('preparo', 'codificacao'));
  });

  it('dadas duas camadas, quando ligadas, então são densas — todas com todas', () => {
    // Given — a rede É densa; desenhar menos ligações ilustraria outra coisa
    const dados = schema();
    const inputs = toInputs(dados.example, dados.request, dados.observedRange);
    const layers = toLayers(dados);
    const results = toResults({ riskProbability: 0.8, threshold: 0.1587 });
    const links = toConnections({ inputs, layers, results });

    // When — oculta 1 (4 nós) → oculta 2 (3 nós)
    const entreOcultas = links.filter(({ from, to }) =>
      from.startsWith('node:oculta-1:') && to.startsWith('node:oculta-2:'));

    // Then
    assert.equal(entreOcultas.length, 4 * 3);
  });

  it('dadas as ligações, quando geradas duas vezes, então a cena é a MESMA', () => {
    // Given — o critério de animação é o índice, não sorteio: comparar
    // duas execuções lado a lado exige que a mesma entrada desenhe o
    // mesmo desenho
    const dados = schema();
    const montar = () => {
      const inputs = toInputs(dados.example, dados.request, dados.observedRange);
      const layers = toLayers(dados);
      const results = toResults({ riskProbability: 0.8, threshold: 0.1587 });

      return toConnections({ inputs, layers, results });
    };

    assert.deepEqual(montar(), montar());
  });
});

describe('toInputs', () => {
  it('dado o contrato, quando mapeado, então numéricas vêm antes das qualitativas', () => {
    // Given / When
    const dados = schema();
    const inputs = toInputs(dados.example, dados.request, dados.observedRange);

    // Then — é a ordem do contrato, e é ela que a rede espera
    assert.deepEqual(inputs.map(({ id }) => id),
      ['durationMonths', 'age', 'checkingStatus', 'housing']);
  });

  it('dada uma qualitativa, quando mapeada, então o valor da opção é o ÍNDICE', () => {
    // Given — o serviço recebe o índice, não o código: `housing = 1` é
    // "própria", e mandar "A152" renderia 400
    const dados = schema();
    const [,, , moradia] = toInputs(dados.example, dados.request, dados.observedRange);

    assert.deepEqual(moradia.options.map(({ value }) => value), [0, 1, 2]);
    assert.deepEqual(moradia.options.map(({ code }) => code), ['A151', 'A152', 'A153']);
    assert.equal(moradia.options[1].label, 'própria');
  });

  it('dada uma faixa ordenada, quando mapeada, então o controle carrega min e max', () => {
    // Given — 1 a 4 é a definição do dataset; 9 ali não é extrapolação,
    // é um código que não existe
    assert.deepEqual(describeControl('installmentRate'), { step: 1, suffix: 'de 4', min: 1, max: 4 });
  });

  it('dada uma medida, quando mapeada, então NÃO carrega min nem max', () => {
    // Given — idade 90 está fora do que a rede viu, e ver a extrapolação
    // acontecer é metade do que o formulário existe para mostrar
    const { min, max } = describeControl('age');

    assert.equal(min, undefined);
    assert.equal(max, undefined);
  });

  it('dada a faixa vista no treino, quando mapeada, então acompanha o campo', () => {
    // Given / When
    const dados = schema();
    const [prazo] = toInputs(dados.example, dados.request, dados.observedRange);

    // Then
    assert.deepEqual(prazo.observed, { min: 4, max: 72 });
  });

  it('dado um campo ausente do cliente, quando mapeado, então não vira controle fantasma', () => {
    // Given — um cliente sem `age` não deve render um campo vazio
    const dados = schema();
    const { age, ...semIdade } = dados.example;

    const ids = toInputs(semIdade, dados.request, dados.observedRange).map((i) => i.id);

    assert.ok(!ids.includes('age'));
  });
});

describe('toRejected', () => {
  it('dado o campo recusado, quando mapeado, então continua NOMEADO na tela', () => {
    // Given — um campo que some sem explicação parece esquecimento; a
    // auditoria de viés depende justamente de o modelo não tê-lo visto
    const [recusado] = toRejected(schema().request);

    assert.equal(recusado.id, 'personalStatus');
    assert.equal(recusado.label, 'Estado civil e sexo');
  });

  it('dado um contrato sem recusados, quando mapeado, então devolve lista vazia', () => {
    assert.deepEqual(toRejected({ numeric: [], categorical: {} }), []);
  });
});

describe('toStats', () => {
  it('dado o pacote, quando resumido, então todo número sai dele', () => {
    // Given
    const dados = schema();

    dados.training = { history: { loss: [0.7, 0.6, 0.5], valLoss: [0.6, 0.5, 0.5] } };

    // When
    const stats = toStats(dados);

    // Then — nenhum deles é digitado; um retreino muda os quatro
    assert.deepEqual(stats.map(({ id }) => id),
      ['entradas', 'parametros', 'epocas', 'limiar']);
    assert.equal(stats[0].valor, '57');
    assert.equal(stats[1].valor, '1.073');
    assert.equal(stats[2].valor, '3');
    assert.equal(stats[3].valor, '15,9%');
  });

  it('dado um pacote sem histórico, quando resumido, então a estatística some em vez de mentir', () => {
    // Given — pacote salvo antes de o histórico existir
    const ids = toStats(schema()).map(({ id }) => id);

    assert.ok(!ids.includes('epocas'));
  });
});

describe('toTraining', () => {
  it('dado o histórico, quando mapeado, então a melhor época é a de menor val_loss', () => {
    // Given — a distância entre a melhor e a última É a paciência do
    // early stopping, e é ali que dá para VER por que o treino parou
    const dados = schema();

    dados.training = {
      customers: 800,
      validationSplit: 0.2,
      epochs: 40,
      patience: 5,
      batchSize: 32,
      l2: 0.003,
      dropout: 0.2,
      units: [16, 8],
      history: {
        loss: [0.7, 0.65, 0.6, 0.58, 0.57],
        valLoss: [0.6, 0.55, 0.5, 0.52, 0.53],
      },
    };

    // When
    const treino = toTraining(dados);

    // Then
    assert.equal(treino.epochs, 5);
    assert.equal(treino.best, 2);
    assert.equal(treino.interrompido, true);
  });

  it('dado um pacote sem histórico, quando mapeado, então devolve null em vez de curva vazia', () => {
    // Given — um pacote antigo não é erro; desenhar uma curva plana como
    // se fosse medida seria
    assert.equal(toTraining(schema()), null);
    assert.equal(toTraining({ ...schema(), training: { history: { loss: [] } } }), null);
  });
});

describe('toEvaluation', () => {
  const comAvaliacao = () => ({
    ...schema(),
    evaluation: {
      baseline: 0.7,
      testAccuracy: 0.725,
      trainAccuracy: 0.795,
      testCustomers: 200,
      auc: 0.748,
      confusion: {
        truePositives: 55, trueNegatives: 61, falsePositives: 79, falseNegatives: 5,
      },
      metrics: { precision: 0.41, recall: 0.9167, f1Score: 0.5668 },
      costs: { falsePositive: 1, falseNegative: 5 },
      thresholds: [
        { label: 'Padrão (0.5)', threshold: 0.5, cost: 165, falsePositives: 10, falseNegatives: 31 },
        { label: 'Menor custo', threshold: 0.1587, cost: 65, falsePositives: 60, falseNegatives: 1 },
      ],
      audit: {
        politica: 'Limiar único',
        approvalRatio: 0.761,
        women: { total: 66 },
        men: { total: 134 },
      },
    },
  });

  it('dada a avaliação, quando mapeada, então o ganho é a distância até o baseline', () => {
    // Given — a acurácia sozinha não responde "o modelo aprendeu algo?"
    const avaliacao = toEvaluation(comAvaliacao());

    assert.ok(Math.abs(avaliacao.ganho - 0.025) < 1e-9);
    assert.ok(Math.abs(avaliacao.gap - 0.07) < 1e-9);
  });

  it('dado o limiar do pacote, quando comparado, então o corte ativo é encontrado por tolerância', () => {
    // Given — o limiar do pacote tem mais casas que os candidatos;
    // comparar por igualdade não acharia nenhum
    const ativos = toEvaluation(comAvaliacao()).thresholds.filter(({ ativo }) => ativo);

    assert.equal(ativos.length, 1);
    assert.equal(ativos[0].label, 'Menor custo');
  });

  it('dada a matriz, quando mapeada, então cada célula diz o que ela significa', () => {
    // Given — "FN" não diz nada a quem chegou agora
    const { confusion } = toEvaluation(comAvaliacao());

    assert.deepEqual(confusion.map(({ sigla }) => sigla), ['TN', 'FP', 'FN', 'TP']);
    assert.equal(confusion.find(({ sigla }) => sigla === 'FN').valor, 5);
    assert.equal(confusion.find(({ sigla }) => sigla === 'FN').tom, 'danger');
  });

  it('dada uma razão abaixo de 0,80, quando auditada, então cai fora da regra dos quatro quintos', () => {
    // Given — 0.761 é o que o pacote atual traz
    assert.equal(toEvaluation(comAvaliacao()).audit.dentroDaRegra, false);
  });

  it('dada uma razão de 0,80 exatos, quando auditada, então a fronteira PERTENCE à regra', () => {
    // Given — a regra é "pelo menos 80%", e a fronteira é o único ponto
    // em que um `>` no lugar de `>=` erra
    const dados = comAvaliacao();

    dados.evaluation.audit.approvalRatio = 0.8;

    assert.equal(toEvaluation(dados).audit.dentroDaRegra, true);
  });

  it('dada a calibração, quando mapeada, então a tela sabe ONDE os cortes foram medidos', () => {
    // Given — sem isto a faixa diria "custo 65" ao lado de uma matriz de
    // teste que custa 104, e pareceria contradição
    const dados = comAvaliacao();

    dados.training = { calibrationCustomers: 160 };

    assert.equal(toEvaluation(dados).calibrationCustomers, 160);
  });

  it('dado um pacote antigo, quando mapeado, então a origem dos cortes é null em vez de inventada', () => {
    assert.equal(toEvaluation(comAvaliacao()).calibrationCustomers, null);
  });

  it('dado um pacote sem avaliação, quando mapeado, então devolve null', () => {
    assert.equal(toEvaluation(schema()), null);
  });
});

describe('toTrainingConnections', () => {
  it('dado o modo treinamento, quando ligado, então o feixe sai do card inteiro', () => {
    // Given — não há dezenove entradas no treino; o card da esquerda vira
    // uma ponta só, e o miolo continua sendo a mesma rede
    const layers = toLayers(schema());
    const links = toTrainingConnections(layers);

    assert.equal(links.filter(({ from }) => from === 'stage:left').length, 2);
    assert.equal(links.filter(({ to }) => to === 'stage:right').length, 1);
    assert.ok(links.every(({ animated }) => animated === true));
  });
});

describe('o vocabulário do domínio', () => {
  it('dado um código qualitativo, quando formatado, então vira a frase que uma pessoa lê', () => {
    // Given — o serviço fala `checkingStatus = 0`; uma pessoa não
    assert.equal(formatValue('checkingStatus', 0), 'saldo negativo');
    assert.equal(formatValue('savingsStatus', 0), 'menos de 100 DM');
  });

  it('dada uma numérica, quando formatada, então carrega a unidade em que foi coletada', () => {
    // Given — DM, porque é a moeda dos dados; "R$" converteria uma
    // unidade que ninguém converteu
    assert.equal(formatValue('creditAmount', 9000), '9.000 DM');
    assert.equal(formatValue('durationMonths', 48), '48 meses');
    assert.equal(formatValue('age', 24), '24 anos');
  });

  it('dada uma faixa ordenada, quando formatada, então não se disfarça de quantidade', () => {
    // Given — "4" sozinho sugeriria reais ou por cento
    assert.equal(formatValue('installmentRate', 4), 'faixa 4 de 4');
  });

  it('dada uma contagem, quando formatada, então concorda em número', () => {
    assert.equal(formatValue('dependents', 1), '1 dependente');
    assert.equal(formatValue('dependents', 2), '2 dependentes');
    assert.equal(formatValue('existingCredits', 1), '1 crédito');
  });

  it('dado um código fora da lista, quando formatado, então diz o que recebeu em vez de quebrar', () => {
    // Given — trocar de fonte de dados não pode render uma tela em branco
    assert.equal(formatValue('housing', 99), 'código 99');
  });

  it('dado um campo desconhecido, quando descrito, então vira um item genérico legível', () => {
    // Given — o dicionário é de UMA fonte; a tela precisa sobreviver a outra
    const campo = describeField('rendaMensal');

    assert.equal(campo.label, 'rendaMensal');
    assert.equal(campo.group, 'numeric');
    assert.equal(campo.icon, 'field');
  });

  it('dado um percentual, quando formatado, então usa a vírgula decimal do pt-BR', () => {
    assert.equal(formatPercent(0.1587), '15,9%');
    assert.equal(formatPercent(0.8046, 2), '80,46%');
  });
});
