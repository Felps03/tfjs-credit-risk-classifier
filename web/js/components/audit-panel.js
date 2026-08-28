import { clear, el, icon } from '../dom.js';
import { formatPercent } from '../domain.js';

// --------------------------------------------------
// <flow-audit> — quem ele penaliza
// --------------------------------------------------
// O modelo NUNCA recebeu a coluna de sexo. É exatamente por isso que as
// decisões dele precisam ser medidas por grupo depois: não receber o
// atributo não impede de reproduzir a disparidade — impede só de ser
// acusado disso pela via mais óbvia.
//
// A tela já dizia que `personalStatus` é recusado. Faltava mostrar o que
// se faz com essa recusa.

const grupo = (item) => el('li', { class: 'group' }, [
  el('div', { class: 'group__head' }, [
    el('span', { class: 'group__label', text: item.rotulo }),
    el('span', { class: 'group__n', text: `n = ${item.total}` }),
  ]),

  // Duas barras na mesma escala: quanto o modelo MARCA como alto risco, e
  // quanta inadimplência o grupo realmente tem. A distância entre elas é
  // o que a auditoria procura.
  el('div', { class: 'group__bars' }, [
    el('div', { class: 'group__bar' }, [
      el('span', { class: 'group__bar-label', text: 'marcados alto risco' }),
      el('div', { class: 'gauge', dataset: { tone: 'warn' } }, [
        el('div', { class: 'gauge__fill', attrs: { style: `--fill: ${item.flaggedRate * 100}%` } }),
      ]),
      el('span', { class: 'group__bar-value', text: formatPercent(item.flaggedRate) }),
    ]),
    el('div', { class: 'group__bar' }, [
      el('span', { class: 'group__bar-label', text: 'inadimplência real' }),
      el('div', { class: 'gauge', dataset: { tone: 'muted' } }, [
        el('div', { class: 'gauge__fill', attrs: { style: `--fill: ${item.baseRate * 100}%` } }),
      ]),
      el('span', { class: 'group__bar-value', text: formatPercent(item.baseRate) }),
    ]),
  ]),

  el('p', {
    class: 'group__note',
    text: `${formatPercent(item.falseNegativeRate)} dos inadimplentes do grupo passaram batido.`,
  }),
]);

export class FlowAudit extends HTMLElement {
  set data(avaliacao) {
    this.avaliacao = avaliacao;
    this.render();
  }

  render() {
    clear(this);

    const auditoria = this.avaliacao?.audit ?? null;

    if (!auditoria) {
      this.append(el('section', { class: 'panel' }, [
        el('header', { class: 'panel__header' }, [
          el('h2', { class: 'panel__title', text: 'Quem ele penaliza' }),
        ]),
        el('p', {
          class: 'panel__empty',
          text: 'Esta fonte não tem atributo protegido para auditar. A auditoria por '
            + 'sexo existe no dataset real (German Credit), não no sintético.',
        }),
      ]));

      return;
    }

    const { approvalRatio, dentroDaRegra, grupos, politica } = auditoria;
    const razao = Number.isFinite(approvalRatio) ? approvalRatio.toFixed(3).replace('.', ',') : '∞';

    this.append(el('section', {
      class: 'panel',
      attrs: { 'aria-labelledby': 'titulo-auditoria' },
    }, [
      el('header', { class: 'panel__header' }, [
        el('h2', { class: 'panel__title', text: 'Quem ele penaliza', id: 'titulo-auditoria' }),
        el('p', {
          class: 'panel__subtitle',
          text: `${politica.toLowerCase()} — o modelo nunca recebeu esta coluna`,
        }),
      ]),

      el('ul', { class: 'group-list' }, grupos.map(grupo)),

      el('div', { class: 'verdict', dataset: { tone: dentroDaRegra ? 'success' : 'warn' } }, [
        el('span', { class: 'verdict__icon' }, [icon(dentroDaRegra ? 'check' : 'alert')]),
        el('div', { class: 'verdict__body' }, [
          el('p', { class: 'verdict__label', text: `Razão de aprovação ${razao}` }),
          el('p', {
            class: 'verdict__text',
            text: dentroDaRegra
              ? 'Acima de 0,80 — dentro da regra dos quatro quintos.'
              : 'Abaixo de 0,80 — a regra dos quatro quintos (EEOC) liga o alerta de '
                + 'impacto desigual aqui. Não é lei brasileira nem prova de '
                + 'discriminação: é um termômetro, e ele está apitando.',
          }),
        ]),
      ]),

      el('span', {
        class: 'anchor anchor--stage anchor--stage-in',
        dataset: { anchor: 'stage:right' },
        attrs: { 'aria-hidden': 'true' },
      }),
    ]));
  }
}

customElements.define('flow-audit', FlowAudit);
