import { clear, el } from '../dom.js';
import { formatPercent } from '../domain.js';

// --------------------------------------------------
// <flow-accuracy> — quanto ele acerta
// --------------------------------------------------
// A acurácia NUNCA aparece sozinha aqui. Ela aparece encostada no piso da
// classe majoritária — a taxa de quem chuta "bom pagador" para todo mundo
// sem olhar para nenhuma feature. Setenta e dois por cento parece ótimo
// até estar ao lado de um piso de setenta, e a distância entre os dois é
// tudo que o treino realmente acrescentou.

const barra = (valor, tom) => el('div', { class: 'gauge', dataset: { tone: tom } }, [
  el('div', { class: 'gauge__fill', attrs: { style: `--fill: ${valor * 100}%` } }),
]);

const numero = (item) => el('li', { class: 'measure' }, [
  el('p', { class: 'measure__label', text: item.rotulo }),
  el('p', { class: 'measure__value', text: item.valor }),
  item.detalhe ? el('p', { class: 'measure__detail', text: item.detalhe }) : null,
]);

export class FlowAccuracy extends HTMLElement {
  set data(avaliacao) {
    this.avaliacao = avaliacao;
    this.render();
  }

  render() {
    clear(this);

    if (!this.avaliacao) {
      return;
    }

    const { accuracy, baseline, ganho, auc, gap, testCustomers } = this.avaliacao;

    this.append(el('section', {
      class: 'panel',
      attrs: { 'aria-labelledby': 'titulo-acuracia' },
    }, [
      el('header', { class: 'panel__header' }, [
        el('h2', { class: 'panel__title', text: 'Quanto ele acerta', id: 'titulo-acuracia' }),
        el('p', {
          class: 'panel__subtitle',
          text: `medido em ${testCustomers} clientes que o treino nunca viu`,
        }),
      ]),

      // As duas barras compartilham a mesma escala de propósito: é a
      // sobreposição delas que mostra o tamanho real do ganho.
      el('div', { class: 'compare' }, [
        el('div', { class: 'compare__row' }, [
          el('p', { class: 'compare__label', text: 'A rede' }),
          el('p', { class: 'compare__value', text: formatPercent(accuracy) }),
        ]),
        barra(accuracy, 'accent'),
        el('div', { class: 'compare__row compare__row--muted' }, [
          el('p', { class: 'compare__label', text: 'Chutar a classe majoritária' }),
          el('p', { class: 'compare__value', text: formatPercent(baseline) }),
        ]),
        barra(baseline, 'muted'),
        el('p', {
          class: 'compare__note',
          text: `O treino acrescentou ${formatPercent(ganho)} sobre não olhar `
            + 'para feature nenhuma. É esse número, e não a acurácia, que diz '
            + 'se o modelo aprendeu algo.',
        }),
      ]),

      el('ul', { class: 'measure-list' }, [
        numero({
          rotulo: 'AUC',
          valor: auc.toFixed(4).replace('.', ','),
          detalhe: 'a curva inteira, não um corte só',
        }),
        numero({
          rotulo: 'Treino − teste',
          valor: formatPercent(gap),
          detalhe: 'quanto ele vai melhor no que já viu',
        }),
      ]),

      el('span', {
        class: 'anchor anchor--stage anchor--stage-out',
        dataset: { anchor: 'stage:left' },
        attrs: { 'aria-hidden': 'true' },
      }),
    ]));
  }
}

customElements.define('flow-accuracy', FlowAccuracy);
