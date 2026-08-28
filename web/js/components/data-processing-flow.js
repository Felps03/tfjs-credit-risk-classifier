import { clear, el, icon, reduzMovimento } from '../dom.js';
import { createPlayer, FASES } from '../training-player.js';
import './input-panel.js';
import './processing-network.js';
import './result-panel.js';
import './connection-layer.js';
import './training-facts.js';
import './loss-chart.js';
import './accuracy-panel.js';
import './confusion-panel.js';
import './audit-panel.js';

// --------------------------------------------------
// <data-processing-flow> — a composição
// --------------------------------------------------
// O componente reutilizável. Ele não busca nada, não conhece rota e não
// sabe o que é `riskProbability`: recebe `inputs`, `layers`,
// `connections`, `results` e `decision` já traduzidos por `mappers.js`.
// Trocar a API por outra é trocar o mapper, não isto aqui.
//
//   flow.state = 'loading' | 'error' | 'empty' | 'ready'
//   flow.data  = { inputs, layers, connections, results, decision, meta }
//   flow.score = { results, decision }   repontuação: só os números
//   flow.busy  = true | false            há uma requisição em curso
//
// `data` e `score` são separados porque o formulário repontua a cada
// alteração. Se cada resposta passasse por `data`, o painel de entrada
// seria reconstruído junto — e quem estivesse digitando perderia o foco
// no meio do número. `score` não toca em nada além dos resultados.
//
// Os quatro estados existem porque os quatro acontecem. O de carregamento
// mostra esqueleto e NENHUM número: um "78%" cinza que depois vira "31%"
// é pior do que um retângulo vazio, porque o primeiro alguém lê.

const ESTADOS = ['loading', 'error', 'empty', 'ready'];

const skeletonLinhas = (quantidade, classe) =>
  Array.from({ length: quantidade }, () => el('div', { class: classe }));

const skeleton = () => el('div', { class: 'flow flow--skeleton', attrs: { 'aria-busy': 'true' } }, [
  el('div', { class: 'flow__stage' }, [
    el('div', { class: 'panel' }, [
      el('div', { class: 'skeleton skeleton--title' }),
      ...skeletonLinhas(6, 'skeleton skeleton--row'),
    ]),
  ]),
  el('div', { class: 'flow__stage' }, [
    el('div', { class: 'network-panel' }, [
      el('div', { class: 'skeleton skeleton--title' }),
      el('div', { class: 'skeleton skeleton--network' }),
    ]),
  ]),
  el('div', { class: 'flow__stage' }, [
    el('div', { class: 'panel' }, [
      el('div', { class: 'skeleton skeleton--title' }),
      ...skeletonLinhas(2, 'skeleton skeleton--bar'),
    ]),
  ]),
  el('p', { class: 'sr-only', text: 'Carregando a análise.' }),
]);

const aviso = ({ tone, titulo, texto, acao }) => el('div', {
  class: 'state',
  dataset: { tone },
  attrs: { role: tone === 'warn' ? 'alert' : 'status' },
}, [
  el('span', { class: 'state__icon' }, [icon(tone === 'warn' ? 'alert' : 'field')]),
  el('h2', { class: 'state__title', text: titulo }),
  el('p', { class: 'state__text', text: texto }),
  acao ?? null,
]);

export class DataProcessingFlow extends HTMLElement {
  constructor() {
    super();
    this.estado = 'loading';
    this.dados = null;
    this.erro = null;
  }

  connectedCallback() {
    this.render();
  }

  set state(valor) {
    this.estado = ESTADOS.includes(valor) ? valor : 'loading';
    this.render();
  }

  get state() {
    return this.estado;
  }

  set data(valor) {
    this.dados = valor;
    this.render();
  }

  set error(valor) {
    this.erro = valor;
    this.render();
  }

  // Repontuação. Não passa por `render()` de propósito: o formulário, a
  // rede e as ligações continuam exatamente como estão.
  set score({ results, decision }) {
    this.dados = { ...this.dados, results, decision };
    this.saidas?.update({ results, decision });
  }

  // 'analise' | 'treinamento'. A rede do meio é a MESMA nos dois: só os
  // painéis das pontas trocam, e é essa continuidade que faz o modo
  // treinamento parecer a mesma rede vista por outro ângulo, e não uma
  // segunda tela.
  set mode(valor) {
    this.modo = ['treinamento', 'avaliacao'].includes(valor) ? valor : 'analise';
    this.render();
  }

  get mode() {
    return this.modo ?? 'analise';
  }

  set training(valor) {
    this.treino = valor;
  }

  set evaluation(valor) {
    this.avaliacao = valor;
  }

  set scoreError(erro) {
    if (this.saidas) {
      this.saidas.problem = erro;
    }
  }

  // Enquanto a resposta não volta, os pontos que percorrem as ligações
  // aceleram. É o único momento em que a animação significa alguma coisa
  // literal: há mesmo um cálculo acontecendo.
  set busy(valor) {
    this.querySelector('.flow')?.setAttribute('data-busy', String(Boolean(valor)));
    this.saidas?.setAttribute('aria-busy', String(Boolean(valor)));
  }

  // "Sem dados suficientes" não é um erro: é uma resposta. Um pacote
  // recém-treinado sobre uma fonte sem colunas, ou uma análise que ainda
  // não rodou, chegam aqui — e chegam sem nada para desenhar.
  get vazio() {
    return !this.dados?.inputs?.length || !this.dados?.results?.length;
  }

  botaoTentarDeNovo() {
    const botao = el('button', {
      class: 'button button--primary',
      type: 'button',
      text: 'Tentar de novo',
    });

    botao.addEventListener('click', () => this.dispatchEvent(
      new CustomEvent('flow-retry', { bubbles: true }),
    ));

    return botao;
  }

  conteudo() {
    if (this.estado === 'loading') {
      return skeleton();
    }

    if (this.estado === 'error') {
      return aviso({
        tone: 'warn',
        titulo: 'Não foi possível carregar a análise',
        texto: this.erro?.message ?? 'O serviço não respondeu como esperado.',
        acao: this.botaoTentarDeNovo(),
      });
    }

    if (this.mode === 'treinamento' && this.estado === 'ready') {
      return this.fluxoTreino();
    }

    if (this.mode === 'avaliacao' && this.estado === 'ready') {
      return this.fluxoAvaliacao();
    }

    if (this.estado === 'empty' || this.vazio) {
      return aviso({
        tone: 'neutral',
        titulo: 'Sem análise para mostrar',
        texto: 'Ainda não há dados suficientes para gerar esta análise.',
      });
    }

    return this.fluxo();
  }

  fluxo() {
    const {
      inputs, rejected, layers, results, decision, connections, meta, example,
    } = this.dados;

    const entradas = el('flow-input-panel');
    const rede = el('flow-network');
    const saidas = el('flow-results');
    const ligacoes = el('flow-connections');

    // Guardadas para que `score` e `busy` alcancem os filhos certos sem
    // varrer o DOM a cada resposta.
    this.saidas = saidas;
    this.entradas = entradas;

    const flow = el('div', {
      class: 'flow',
      attrs: {
        role: 'group',
        'aria-label': 'Do dado de entrada ao resultado da análise',
      },
    }, [
      el('div', { class: 'flow__stage flow__stage--inputs' }, [entradas]),
      el('div', { class: 'flow__stage flow__stage--network' }, [rede]),
      el('div', { class: 'flow__stage flow__stage--results' }, [saidas]),
      ligacoes,
    ]);

    // Os painéis são preenchidos DEPOIS de entrarem no documento: a
    // camada de conexões mede posições reais, e não há posição real
    // antes do primeiro layout.
    clear(this);
    this.append(flow, this.rodape(meta));

    // O rastro precisa saber para qual NÓ o campo vai. Numérica termina
    // na escala min–max, qualitativa na codificação — é a mesma regra que
    // o mapper usou para desenhar as ligações, e ela vive num lugar só.
    const preparo = layers[0];
    const destino = {
      numeric: preparo.nodes.find((no) => no.id === 'escala'),
      categorical: preparo.nodes.find((no) => no.id === 'codificacao'),
    };

    flow.addEventListener('flow-trace', (evento) => {
      const { campo, grupo } = evento.detail;
      const no = destino[grupo === 'categorical' ? 'categorical' : 'numeric'];

      ligacoes.trace = campo;
      rede.trace = campo && no ? `${preparo.id}:${no.id}` : null;
    });

    entradas.data = { inputs, rejected, example };
    rede.data = { layers, meta };
    saidas.data = { results, decision };
    ligacoes.data = connections;

    return null;
  }

  // --------------------------------------------------
  // Modo avaliação
  // --------------------------------------------------
  // Sem rede no meio: aqui o assunto não é o caminho do dado, é o
  // resultado dele. As ligações ainda saem do card da esquerda e chegam
  // no da direita — acurácia e auditoria são as duas pontas da mesma
  // medição, e o feixe entre elas passa por onde os erros aparecem.
  fluxoAvaliacao() {
    if (!this.avaliacao) {
      clear(this);
      this.append(aviso({
        tone: 'neutral',
        titulo: 'Sem avaliação no pacote',
        texto: 'Este pacote foi salvo antes de as métricas passarem a ser gravadas. '
          + 'Rode npm start para treinar de novo — os números aparecem sozinhos.',
      }));

      return null;
    }

    const acuracia = el('flow-accuracy');
    const matriz = el('flow-confusion');
    const auditoria = el('flow-audit');

    const flow = el('div', {
      class: 'flow',
      dataset: { mode: 'avaliacao' },
      attrs: { role: 'group', 'aria-label': 'Quanto o modelo acerta e quem ele penaliza' },
    }, [
      el('div', { class: 'flow__stage flow__stage--inputs' }, [acuracia]),
      el('div', { class: 'flow__stage flow__stage--network' }, [matriz]),
      el('div', { class: 'flow__stage flow__stage--results' }, [auditoria]),
    ]);

    clear(this);
    this.append(this.faixaDeLimiares(), flow, this.rodape(this.dados?.meta));

    acuracia.data = this.avaliacao;
    matriz.data = this.avaliacao;
    auditoria.data = this.avaliacao;

    return null;
  }

  // Os três cortes comparados, com o que está decidindo em destaque. É a
  // única parte da tela em que o limiar deixa de ser um número dado e
  // passa a ser uma ESCOLHA com alternativas e preço.
  faixaDeLimiares() {
    const { thresholds, costs, calibrationCustomers } = this.avaliacao;
    const onde = calibrationCustomers
      ? `${calibrationCustomers} clientes de calibração`
      : 'a calibração';

    // Sem os controles de reprodução ao lado, os três cortes ficavam
    // espremidos na metade esquerda de uma barra vazia. Aqui eles são o
    // conteúdo inteiro — o modificador diz isso à folha de estilo.
    return el('section', { class: 'training-bar training-bar--thresholds' }, [
      el('ol', { class: 'phase-list' }, thresholds.map((corte) => el('li', {
        class: 'phase',
        dataset: { active: String(corte.ativo) },
      }, [
        el('span', { class: 'phase__label', text: corte.label }),
        el('span', {
          class: 'phase__detail',
          text: corte.threshold === null
            ? 'não aprova ninguém'
            : `corte ${corte.threshold.toFixed(3).replace('.', ',')} · `
              + `custo ${corte.cost} (${corte.falsePositives} FP, ${corte.falseNegatives} FN)`,
        }),
      ]))),
      el('p', {
        class: 'training-note',
        text: `O limiar não é 0,5 herdado: com FN custando ${costs.falseNegative}× o FP, `
          + 'o corte de menor custo desce e a rede passa a sinalizar mais. '
          + `Os três custos acima foram medidos em ${onde} — nunca no teste, `
          + 'porque escolher o corte onde ele é medido publica um número que '
          + 'não se sustenta. A matriz ao lado é a do teste.',
      }),
    ]);
  }

  // --------------------------------------------------
  // Modo treinamento
  // --------------------------------------------------
  fluxoTreino() {
    const { layers, meta, trainingConnections } = this.dados;

    const fatos = el('flow-training-facts');
    const rede = el('flow-network');
    const grafico = el('flow-loss-chart');
    const ligacoes = el('flow-connections');

    this.redeTreino = rede;
    this.grafico = grafico;
    this.ligacoesTreino = ligacoes;

    const flow = el('div', {
      class: 'flow',
      dataset: { mode: 'treinamento' },
      attrs: { role: 'group', 'aria-label': 'Como o modelo foi treinado' },
    }, [
      el('div', { class: 'flow__stage flow__stage--inputs' }, [fatos]),
      el('div', { class: 'flow__stage flow__stage--network' }, [rede]),
      el('div', { class: 'flow__stage flow__stage--results' }, [grafico]),
      ligacoes,
    ]);

    clear(this);
    this.append(this.faixaDeFases(), flow, this.rodape(meta));

    fatos.data = { facts: this.treino?.facts ?? [] };
    rede.data = {
      layers,
      meta,
      titulo: 'O laço do treino',
      legenda: 'esquema do algoritmo — os números medidos estão na curva ao lado',
    };
    grafico.data = this.treino;
    ligacoes.data = trainingConnections;
    ligacoes.animateAll = true;

    this.iniciarPlayer();

    return null;
  }

  // Os quatro passos, sempre visíveis, com o atual em destaque. Sem eles
  // a animação seria bonita e muda: pontinhos correndo não dizem
  // "retropropagação" a ninguém que ainda não saiba o que é.
  faixaDeFases() {
    this.chips = FASES.map((fase) => el('li', {
      class: 'phase',
      dataset: { phase: fase.id },
    }, [
      el('span', { class: 'phase__label', text: fase.label }),
      el('span', { class: 'phase__detail', text: fase.detail }),
    ]));

    this.botaoPlay = el('button', {
      class: 'button button--primary',
      type: 'button',
      text: 'Pausar',
    });
    this.botaoPlay.addEventListener('click', () => this.player?.toggle());

    this.barra = el('input', {
      class: 'epoch-range',
      type: 'range',
      min: '0',
      max: String((this.treino?.epochs ?? 1) - 1),
      value: '0',
      attrs: { 'aria-label': 'Época do treino' },
    });
    this.barra.addEventListener('input', () => this.player?.seek(Number(this.barra.value)));

    this.rotuloEpoca = el('span', { class: 'epoch-readout' });

    return el('section', { class: 'training-bar' }, [
      el('ol', { class: 'phase-list' }, this.chips),
      el('div', { class: 'training-controls' }, [
        this.botaoPlay,
        this.barra,
        this.rotuloEpoca,
      ]),
    ]);
  }

  iniciarPlayer() {
    this.player?.destroy();

    if (!this.treino) {
      return;
    }

    // Com movimento reduzido o laço não roda sozinho: a curva aparece
    // inteira e a barra continua funcionando. A informação é a mesma; o
    // que some é o movimento automático.
    this.player = createPlayer({
      epochs: this.treino.epochs,
      autoplay: !reduzMovimento(),
      onChange: (estado) => this.aplicarEstado(estado),
    });
  }

  aplicarEstado({ epoca, fase, rodando }) {
    this.redeTreino?.setAttribute('data-phase', fase.id);

    // O passo atrás é o único momento em que o fluxo inverte — e é a
    // coisa mais importante que esta tela tem a mostrar.
    if (this.ligacoesTreino) {
      this.ligacoesTreino.direction = fase.id === 'atras' ? 'backward' : 'forward';
    }

    if (this.grafico) {
      this.grafico.epoch = epoca;
    }

    this.chips?.forEach((chip) => {
      chip.dataset.active = String(chip.dataset.phase === fase.id);
    });

    if (this.barra && Number(this.barra.value) !== epoca) {
      this.barra.value = String(epoca);
    }

    if (this.botaoPlay) {
      this.botaoPlay.textContent = rodando ? 'Pausar' : 'Reproduzir';
    }

    if (this.rotuloEpoca) {
      this.rotuloEpoca.textContent = `época ${epoca + 1} de ${this.treino.epochs}`;
    }
  }

  disconnectedCallback() {
    this.player?.destroy();
  }

  rodape(meta) {
    if (!meta) {
      return null;
    }

    const treinado = meta.savedAt
      ? new Date(meta.savedAt).toLocaleString('pt-BR', { dateStyle: 'short', timeStyle: 'short' })
      : null;

    return el('footer', { class: 'flow-meta' }, [
      el('span', { class: 'flow-meta__item', text: meta.source }),
      el('span', { class: 'flow-meta__item', text: `${meta.features} entradas na rede` }),
      meta.units?.length
        ? el('span', { class: 'flow-meta__item', text: `camadas ocultas ${meta.units.join(' → ')}` })
        : null,
      treinado ? el('span', { class: 'flow-meta__item', text: `treinado em ${treinado}` }) : null,
    ]);
  }

  render() {
    // Trocar de modo destrói a árvore inteira; o relógio do treino
    // precisa parar junto, ou continuaria disparando contra nós que já
    // não existem.
    this.player?.destroy();
    this.player = null;

    const conteudo = this.conteudo();

    // `fluxo()` já montou tudo por conta própria — ele precisa do DOM
    // vivo antes de entregar os dados aos filhos.
    if (conteudo === null) {
      return;
    }

    clear(this);
    this.append(conteudo);
  }
}

customElements.define('data-processing-flow', DataProcessingFlow);
