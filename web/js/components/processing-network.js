import { clear, el } from '../dom.js';
import { anchorId } from '../mappers.js';

// --------------------------------------------------
// <flow-network> — onde os dados são relacionados
// --------------------------------------------------
// Os círculos NÃO são enfeite e não são aleatórios: cada coluna é uma
// etapa que existe no pipeline, com o nome que ela tem no código —
// escala min–max, one-hot, as camadas ocultas e a sigmoide. A legenda de
// cada coluna diz o tamanho REAL da etapa e quantos círculos a
// representam, porque desenhar quatro bolinhas e chamá-las de "16
// unidades" seria mentir em silêncio.
//
// A marcação é uma lista ordenada de verdade. Sem o SVG por cima, sem
// CSS, ou lido por voz, o conteúdo continua sendo "Preparo → Camada
// oculta 1 → Camada oculta 2 → Saída", na ordem certa.

const node = (layer, item) => el('li', {
  class: 'node-slot',
  dataset: { named: String(Boolean(item.named)) },
}, [
  el('span', {
    class: 'node',
    tabIndex: 0,
    dataset: { anchor: anchorId.node(layer.id, item.id) },
    attrs: { role: 'img', 'aria-label': `${item.label}: ${item.detail}` },
  }),
  el('span', { class: 'node__caption', text: item.label }),
  el('span', { class: 'node__tip', attrs: { 'aria-hidden': 'true' } }, [
    el('strong', { text: item.label }),
    el('span', { text: item.detail }),
  ]),
]);

// Só as etapas nomeadas mostram o rótulo embaixo do círculo. Uma unidade
// oculta não tem nome próprio — ela é uma das dezesseis —, então o
// rótulo dela vive no `aria-label` e no balão de hover, e não na tela.
const nomeada = (layerId) => layerId === 'preparo' || layerId === 'saida';

const layerBlock = (layer) => el('li', { class: 'network__layer' }, [
  el('h3', { class: 'network__layer-title', text: layer.title }),
  el('p', { class: 'network__layer-caption', text: layer.caption }),
  el('ul', { class: 'network__nodes' }, layer.nodes.map((item) =>
    node(layer, { ...item, named: nomeada(layer.id) }))),
]);

export class FlowNetwork extends HTMLElement {
  set data({ layers = [], meta = {} } = {}) {
    this.layers = layers;
    this.meta = meta;
    this.render();
  }

  render() {
    const layers = this.layers ?? [];
    const caminho = layers.map((layer) => layer.title).join(' → ');

    clear(this);
    this.append(el('section', {
      class: 'network-panel',
      attrs: { 'aria-labelledby': 'titulo-processamento' },
    }, [
      el('header', { class: 'panel__header panel__header--center' }, [
        el('h2', {
          class: 'panel__title',
          text: 'Processamento',
          id: 'titulo-processamento',
        }),
        el('p', {
          class: 'panel__subtitle',
          text: 'cada entrada influencia cada unidade — as ligações são a rede densa',
        }),
      ]),

      // Este parágrafo é a versão em texto do desenho. Ele não é um
      // resumo aproximado: é a mesma sequência que os círculos mostram.
      el('p', { class: 'sr-only', text: `Caminho do dado: ${caminho}.` }),

      el('ol', { class: 'network' }, layers.map(layerBlock)),
    ]));
  }
}

customElements.define('flow-network', FlowNetwork);
