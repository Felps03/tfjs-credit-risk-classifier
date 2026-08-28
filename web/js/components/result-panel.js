import { clear, el, icon, reduzMovimento } from '../dom.js';
import { formatPercent } from '../domain.js';
import { anchorId } from '../mappers.js';

// --------------------------------------------------
// <flow-results> — o que sai
// --------------------------------------------------
// Duas barras, não três. O modelo tem UMA saída — uma probabilidade — e a
// segunda barra é o complemento dela, não uma segunda classe.
//
// A distinção que importa aqui é entre `data` e `update`. `data` MONTA a
// coluna; `update` só troca os números de uma coluna que já existe. Elas
// existem separadas porque o formulário repontua a cada alteração: se
// cada resposta recriasse o DOM, a barra recomeçaria do zero a cada
// tecla, piscando em vez de se mover. Trocando só `--fill`, a mesma
// transição de CSS leva a barra do valor antigo ao novo — e o que se vê é
// a probabilidade REAGINDO ao campo que acabou de mudar.

const trilha = (item) => el('div', { class: 'result__track', attrs: { 'aria-hidden': 'true' } }, [
  el('div', { class: 'result__bar', attrs: { style: `--fill: ${item.value * 100}%` } }),
  item.marker
    ? el('div', {
      class: 'result__marker',
      attrs: { style: `--at: ${item.marker.value * 100}%` },
    }, [el('span', { class: 'result__marker-label', text: item.marker.label })])
    : null,
]);

const result = (item) => el('li', {
  class: 'result',
  dataset: { tone: item.tone ?? 'accent', result: item.id },
}, [
  el('div', { class: 'result__head' }, [
    el('span', { class: 'result__label', text: item.label }),
    el('span', { class: 'result__value', text: formatPercent(item.value) }),
  ]),
  trilha(item),
  item.marker
    ? el('p', {
      class: 'result__meta',
      text: `${item.marker.label}: ${formatPercent(item.marker.value)}`,
    })
    : null,
  item.description ? el('p', { class: 'result__description', text: item.description }) : null,
  el('span', {
    class: 'anchor anchor--in',
    dataset: { anchor: anchorId.result(item.id), fallback: 'panel:results' },
    attrs: { 'aria-hidden': 'true' },
  }),
]);

const verdict = (decision) => el('div', {
  class: 'verdict',
  dataset: { tone: decision.alto ? 'warn' : 'success' },
}, [
  el('span', { class: 'verdict__icon' }, [icon(decision.alto ? 'alert' : 'check')]),
  el('div', { class: 'verdict__body' }, [
    el('p', { class: 'verdict__label', text: decision.label }),
    el('p', {
      class: 'verdict__text',
      text: `${decision.explanation} O corte não é 0,5 herdado: saiu da matriz de `
        + `custo — ${decision.strategy}.`,
    }),
  ]),
]);

export class FlowResults extends HTMLElement {
  disconnectedCallback() {
    this.observer?.disconnect();
  }

  set data({ results = [], decision = null } = {}) {
    this.results = results;
    this.decision = decision;
    this.render();
  }

  // O caminho da repontuação. Nada é recriado: os mesmos nós recebem
  // valores novos, e o CSS faz o resto.
  update({ results = [], decision = null } = {}) {
    if (!this.querySelector('.result-list')) {
      this.data = { results, decision };

      return;
    }

    this.results = results;
    this.decision = decision;

    results.forEach((item) => {
      const linha = this.querySelector(`[data-result="${CSS.escape(item.id)}"]`);

      if (!linha) {
        return;
      }

      linha.querySelector('.result__value').textContent = formatPercent(item.value);
      linha.querySelector('.result__bar').style.setProperty('--fill', `${item.value * 100}%`);
    });

    // A partir da primeira atualização a transição encurta: 900ms é o
    // tempo certo para a barra APARECER, e longo demais para ela
    // acompanhar alguém mexendo num campo.
    this.querySelector('.result-list')?.classList.add('is-live');

    if (decision) {
      this.trocarVeredito(decision);
    }

    this.anunciar();
  }

  trocarVeredito(decision) {
    const atual = this.querySelector('.verdict');

    if (!atual) {
      return;
    }

    atual.replaceWith(verdict(decision));
  }

  // Uma repontuação que falha NÃO derruba a tela. O formulário continua
  // preenchido, os números anteriores continuam à vista, e o problema
  // aparece ao lado deles — porque apagar tudo por causa de um `fetch`
  // que não voltou custaria a quem estava no meio de uma simulação.
  set problem(erro) {
    const alvo = this.querySelector('.result__problema');

    if (!alvo) {
      return;
    }

    clear(alvo);
    alvo.toggleAttribute('hidden', !erro);

    if (!erro) {
      return;
    }

    alvo.append(
      el('span', { class: 'result__problema-icon' }, [icon('alert')]),
      el('div', {}, [
        el('p', { text: erro.message }),
        ...(erro.details ?? []).map((detalhe) => el('p', {
          class: 'result__problema-detalhe',
          text: detalhe,
        })),
      ]),
    );
  }

  // Um resumo em texto, atualizado a cada resposta. Sem ele, quem navega
  // por voz ouviria o formulário mudar e nunca saberia o resultado — as
  // barras são `aria-hidden`, e o número muda sem avisar ninguém.
  anunciar() {
    const resumo = this.querySelector('.result-live');

    if (!resumo) {
      return;
    }

    resumo.textContent = [
      ...this.results.map((item) => `${item.label}: ${formatPercent(item.value)}.`),
      this.decision?.label ?? '',
    ].join(' ');
  }

  observarEntrada(alvo) {
    this.observer?.disconnect();

    if (reduzMovimento() || typeof IntersectionObserver !== 'function') {
      alvo.classList.add('is-visible');

      return;
    }

    this.observer = new IntersectionObserver((entries, observer) => {
      entries.filter((entry) => entry.isIntersecting).forEach((entry) => {
        entry.target.classList.add('is-visible');
        observer.unobserve(entry.target);
      });
    }, { threshold: 0.25 });

    this.observer.observe(alvo);
  }

  render() {
    const results = this.results ?? [];
    const lista = el('ul', { class: 'result-list' }, results.map(result));

    clear(this);
    this.append(el('section', {
      class: 'panel panel--results',
      attrs: { 'aria-labelledby': 'titulo-resultados' },
    }, [
      el('header', { class: 'panel__header' }, [
        el('h2', {
          class: 'panel__title',
          text: 'Resultados da análise',
          id: 'titulo-resultados',
        }),
        el('p', { class: 'panel__subtitle', text: 'uma saída, vista pelos dois lados' }),
      ]),
      lista,
      this.decision ? verdict(this.decision) : null,
      el('div', { class: 'result__problema', hidden: true, attrs: { role: 'alert' } }),
      el('p', { class: 'sr-only result-live', attrs: { 'aria-live': 'polite' } }),
      el('span', {
        class: 'anchor anchor--panel anchor--panel-in',
        dataset: { anchor: 'panel:results' },
        attrs: { 'aria-hidden': 'true' },
      }),
    ]));

    this.observarEntrada(lista);
    this.anunciar();
  }
}

customElements.define('flow-results', FlowResults);
