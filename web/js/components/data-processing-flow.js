import { clear, el, icon } from '../dom.js';
import './input-panel.js';
import './processing-network.js';
import './result-panel.js';
import './connection-layer.js';

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

    entradas.data = { inputs, rejected, example };
    rede.data = { layers, meta };
    saidas.data = { results, decision };
    ligacoes.data = connections;

    return null;
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
