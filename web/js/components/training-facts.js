import { clear, el, icon } from '../dom.js';

// --------------------------------------------------
// <flow-training-facts> — como ele foi treinado
// --------------------------------------------------
// O que substitui "Dados utilizados" no modo treinamento. Cada linha é um
// número que governou ESTE treino e que estava gravado no pacote —
// nenhum deles é uma convenção repetida de memória.

const fato = (item) => el('li', { class: 'fact' }, [
  el('span', { class: 'fact__icon' }, [icon(item.icon)]),
  el('div', { class: 'fact__body' }, [
    el('p', { class: 'fact__label', text: item.label }),
    el('p', { class: 'fact__detail', text: item.detail }),
  ]),
]);

export class FlowTrainingFacts extends HTMLElement {
  set data({ facts = [] } = {}) {
    this.facts = facts;
    this.render();
  }

  render() {
    clear(this);
    this.append(el('section', { class: 'panel', attrs: { 'aria-labelledby': 'titulo-treino' } }, [
      el('header', { class: 'panel__header' }, [
        el('h2', { class: 'panel__title', text: 'Como foi treinado', id: 'titulo-treino' }),
        // O comando vai num `<code>`, não entre crases: crase em texto
        // de interface aparece como crase na tela.
        el('p', { class: 'panel__subtitle' }, [
          document.createTextNode('os números gravados no pacote por '),
          el('code', { text: 'npm start' }),
        ]),
      ]),
      el('ul', { class: 'fact-list' }, (this.facts ?? []).map(fato)),
      el('span', {
        class: 'anchor anchor--stage anchor--stage-out',
        dataset: { anchor: 'stage:left' },
        attrs: { 'aria-hidden': 'true' },
      }),
    ]));
  }
}

customElements.define('flow-training-facts', FlowTrainingFacts);
