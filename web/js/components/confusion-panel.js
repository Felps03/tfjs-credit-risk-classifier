import { clear, el } from '../dom.js';
import { formatPercent } from '../domain.js';

// --------------------------------------------------
// <flow-confusion> — onde ele erra
// --------------------------------------------------
// A matriz de confusão como grade, não como tabela de texto. Os dois
// erros NÃO são equivalentes e a tela precisa dizer isso: recusar quem
// pagaria custa 1, deixar passar quem não paga custa 5. É essa assimetria
// que puxou o limiar para longe do 0,5 herdado.

const celula = (item, total) => {
  const fracao = total > 0 ? item.valor / total : 0;

  return el('div', {
    class: 'cell',
    dataset: { tone: item.tom },

    // A intensidade do fundo acompanha a fração: a matriz vira um mapa de
    // calor sem deixar de ser legível como números.
    attrs: { style: `--peso: ${(0.1 + fracao * 0.9).toFixed(3)}` },
  }, [
    el('p', { class: 'cell__value', text: String(item.valor) }),
    el('p', { class: 'cell__code', text: item.sigla }),
    el('p', { class: 'cell__title', text: item.titulo }),
  ]);
};

const metrica = (rotulo, valor, explicacao) => el('li', { class: 'metric' }, [
  el('div', { class: 'metric__head' }, [
    el('span', { class: 'metric__label', text: rotulo }),
    el('span', { class: 'metric__value', text: formatPercent(valor) }),
  ]),
  el('p', { class: 'metric__detail', text: explicacao }),
]);

export class FlowConfusion extends HTMLElement {
  set data(avaliacao) {
    this.avaliacao = avaliacao;
    this.render();
  }

  render() {
    clear(this);

    if (!this.avaliacao) {
      return;
    }

    const { confusion, metrics, costs } = this.avaliacao;
    const total = confusion.reduce((soma, item) => soma + item.valor, 0);

    this.append(el('section', {
      class: 'confusion-panel',
      attrs: { 'aria-labelledby': 'titulo-matriz' },
    }, [
      el('header', { class: 'panel__header panel__header--center' }, [
        el('h2', { class: 'panel__title', text: 'Onde ele erra', id: 'titulo-matriz' }),
        el('p', {
          class: 'panel__subtitle',
          text: `os dois erros não custam igual — FP vale ${costs.falsePositive}, `
            + `FN vale ${costs.falseNegative}`,
        }),
      ]),

      el('div', { class: 'matrix' }, [
        el('span', { class: 'matrix__corner' }),
        el('span', { class: 'matrix__head', text: 'Predito BAIXO' }),
        el('span', { class: 'matrix__head', text: 'Predito ALTO' }),
        el('span', { class: 'matrix__side', text: 'Real BAIXO' }),
        celula(confusion[0], total),
        celula(confusion[1], total),
        el('span', { class: 'matrix__side', text: 'Real ALTO' }),
        celula(confusion[2], total),
        celula(confusion[3], total),
      ]),

      el('ul', { class: 'metric-list' }, [
        metrica('Precision', metrics.precision, 'dos marcados como alto risco, quantos eram'),
        metrica('Recall', metrics.recall, 'dos que eram alto risco, quantos foram pegos'),
        metrica('F1', metrics.f1Score, 'média harmônica entre as duas'),
      ]),
    ]));
  }
}

customElements.define('flow-confusion', FlowConfusion);
