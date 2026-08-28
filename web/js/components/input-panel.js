import { clear, el, icon } from '../dom.js';
import { anchorId } from '../mappers.js';

// --------------------------------------------------
// <flow-input-panel> — o que entra, e agora editável
// --------------------------------------------------
// Deixou de ser uma lista de leitura e virou o formulário que simula um
// cliente. Nenhum campo é escrito à mão aqui: o tipo de cada controle, as
// opções de cada qualitativa e a faixa de cada numérica vêm todos do
// `GET /schema`. Um retreino que mude as colunas muda o formulário sem
// ninguém tocar neste arquivo.
//
// Continua sendo marcação semântica: `<label for>` ligado ao controle,
// `<ul>` de verdade, e o texto formatado do valor preservado para quem lê
// por voz. Se o SVG das ligações não renderizar, isto aqui continua sendo
// um formulário que funciona.

const campoId = (field) => `campo-${field}`;

// Fora da faixa vista no treino não é erro: o serviço aceita, e a rede
// pontua. É EXTRAPOLAÇÃO — a rede nunca viu nada parecido e responde
// assim mesmo. Avisar é a diferença entre um número e um número que
// alguém sabe interpretar.
const foraDaFaixa = (valor, observed) =>
  observed !== null && observed !== undefined
  && (valor < observed.min || valor > observed.max);

const numero = (input, aoMudar) => {
  const campo = el('input', {
    class: 'field field--number',
    id: campoId(input.id),
    type: 'number',
    value: String(input.value),
    step: String(input.step ?? 1),
    inputMode: 'numeric',
    attrs: {
      // Só os campos que carregam `min`/`max` no dicionário ganham a
      // trava — e são os dois que são faixa ordenada, não medida.
      ...(input.min === undefined ? {} : { min: String(input.min) }),
      ...(input.max === undefined ? {} : { max: String(input.max) }),
      'aria-describedby': input.observed ? `faixa-${input.id}` : '',

      // Sem isto o Chrome RESTAURA o valor digitado antes do reload,
      // por cima do que o `GET /schema` acabou de trazer — e dispara
      // `change` ao fazê-lo. O resultado é a tela abrir pontuando um
      // cliente que ninguém pediu, com um número que não corresponde
      // aos campos que a pessoa está vendo. O campo tem que refletir o
      // exemplo do serviço, não a sessão anterior.
      autocomplete: 'off',
    },
  });

  campo.addEventListener('input', () => aoMudar(campo.value === '' ? null : Number(campo.value)));

  return campo;
};

const selecao = (input, aoMudar) => {
  const campo = el('select', {
    class: 'field field--select',
    id: campoId(input.id),
    attrs: { autocomplete: 'off' },
  },
  input.options.map((opcao) => el('option', {
      value: String(opcao.value),
      text: opcao.label,
    selected: opcao.value === input.value,
  })));

  campo.addEventListener('change', () => aoMudar(Number(campo.value)));

  return campo;
};

const rejeitados = (campos) => el('div', { class: 'panel__note' }, [
  el('span', { class: 'panel__note-icon' }, [icon(campos[0].icon)]),
  el('p', {
    class: 'panel__note-text',
    text: `${campos.map((campo) => campo.label).join(', ')} — não é editável porque `
      + 'não é enviado. A auditoria de viés usa essa coluna; a decisão, não.',
  }),
]);

export class FlowInputPanel extends HTMLElement {
  set data({ inputs = [], rejected = [], example = null } = {}) {
    this.inputs = inputs;
    this.rejected = rejected;
    this.example = example;
    this.customer = Object.fromEntries(inputs.map((input) => [input.id, input.value]));
    this.render();
  }

  get alterado() {
    return this.example !== null
      && Object.entries(this.customer).some(([campo, valor]) => this.example[campo] !== valor);
  }

  // Um campo vazio ou não numérico NÃO é enviado. Mandar `null` faria o
  // serviço responder 400 com razão, mas o efeito na tela seria pior:
  // a probabilidade sumiria enquanto alguém apaga um número para digitar
  // outro. O estado inválido fica local até virar um número de novo.
  aplicar(campo, valor, item) {
    const invalido = valor === null || !Number.isFinite(valor);

    item.dataset.invalid = String(invalido);
    item.querySelector('.input-item__erro')?.replaceChildren(
      invalido ? 'informe um número' : '',
    );

    if (invalido) {
      return;
    }

    // Valor igual ao que já está não repontua. Sem esta linha, qualquer
    // evento que não mude nada — uma restauração do navegador, um
    // `change` disparado por script, um blur — custa uma requisição e
    // uma animação de barra que não vai a lugar nenhum.
    if (this.customer[campo] === valor) {
      return;
    }

    this.customer[campo] = valor;
    this.marcarFaixa(campo, valor, item);
    this.atualizarRestaurar();
    this.dispatchEvent(new CustomEvent('flow-customer-change', {
      bubbles: true,
      detail: { customer: { ...this.customer } },
    }));
  }

  marcarFaixa(campo, valor, item) {
    const input = this.inputs.find((candidato) => candidato.id === campo);

    item.dataset.foraDaFaixa = String(foraDaFaixa(valor, input?.observed));
  }

  atualizarRestaurar() {
    this.botaoRestaurar?.toggleAttribute('hidden', !this.alterado);
  }

  restaurar() {
    this.customer = { ...this.example };
    this.inputs = this.inputs.map((input) => ({ ...input, value: this.example[input.id] }));
    this.render();
    this.dispatchEvent(new CustomEvent('flow-customer-change', {
      bubbles: true,
      detail: { customer: { ...this.customer } },
    }));
  }

  item(input) {
    const linha = el('li', {
      class: 'input-item',
      dataset: {
        field: input.id,
        group: input.group,
        invalid: 'false',
        foraDaFaixa: String(foraDaFaixa(input.value, input.observed)),
      },
    });

    const controle = input.group === 'categorical'
      ? selecao(input, (valor) => this.aplicar(input.id, valor, linha))
      : numero(input, (valor) => this.aplicar(input.id, valor, linha));

    // `Node.append` NÃO ignora `null` como o `el()` faz: ele insere o
    // texto "null" na tela. Os filhos opcionais são filtrados antes.
    const filhos = [
      el('span', { class: 'input-item__icon' }, [icon(input.icon)]),
      el('label', {
        class: 'input-item__label',
        text: input.label,
        attrs: { for: campoId(input.id) },
      }),
      el('span', { class: 'input-item__control' }, [
        controle,

        // A coluna da unidade existe mesmo vazia nas NUMÉRICAS. Sem ela,
        // "Créditos neste banco" e "Dependentes" — os dois campos sem
        // unidade — encostavam na borda enquanto os outros paravam 38px
        // antes, e dezenove controles desalinhados leem como dezenove
        // erros.
        //
        // Nas qualitativas ela não existe: não há unidade nenhuma para
        // escrever, e os 33px que ela custaria saem justamente do único
        // controle desta tela cujo texto precisa caber por inteiro.
        input.group === 'categorical'
          ? null
          : el('span', { class: 'field__suffix', text: input.suffix ?? '' }),
      ]),

      // Duas mensagens, ambas ligadas ao campo por `aria-describedby` ou
      // vizinhança: a faixa do treino (constante) e o erro (eventual).
      input.observed
        ? el('span', {
          class: 'input-item__faixa',
          id: `faixa-${input.id}`,
          text: `treino: ${input.observed.min} a ${input.observed.max}`,
        })
        : null,
      el('span', { class: 'input-item__erro', attrs: { role: 'status' } }),

      el('span', {
        class: 'anchor anchor--out',
        dataset: { anchor: anchorId.input(input.id), fallback: 'panel:inputs' },
        attrs: { 'aria-hidden': 'true' },
      }),
    ];

    linha.append(...filhos.filter(Boolean));

    // Passar o mouse ou o foco por uma linha ilumina o caminho DAQUELE
    // campo pela rede. É a resposta visual a "para onde vai a idade?" —
    // e o caminho é o real: numérica vai ao min–max, qualitativa ao
    // one-hot. O evento sobe; quem sabe pintar é a camada de ligações.
    const rastrear = (campo) => this.dispatchEvent(new CustomEvent('flow-trace', {
      bubbles: true,
      detail: { campo, grupo: input.group },
    }));

    linha.addEventListener('pointerenter', () => rastrear(input.id));
    linha.addEventListener('pointerleave', () => rastrear(null));

    // No toque o `pointerleave` pode nunca chegar: o dedo sai da TELA, e
    // não do elemento. Sem isto o rastro fica preso no último campo
    // tocado, e a rede segue apagada em volta de um caminho que ninguém
    // está mais consultando.
    linha.addEventListener('pointercancel', () => rastrear(null));
    linha.addEventListener('focusin', () => rastrear(input.id));
    linha.addEventListener('focusout', () => rastrear(null));

    return linha;
  }

  render() {
    const inputs = this.inputs ?? [];

    this.botaoRestaurar = el('button', {
      class: 'button button--ghost',
      type: 'button',
      text: 'Restaurar exemplo',
      hidden: !this.alterado,
    });
    this.botaoRestaurar.addEventListener('click', () => this.restaurar());

    clear(this);
    this.append(el('section', { class: 'panel', attrs: { 'aria-labelledby': 'titulo-entradas' } }, [
      el('header', { class: 'panel__header panel__header--row' }, [
        el('div', {}, [
          el('h2', { class: 'panel__title', text: 'Dados utilizados', id: 'titulo-entradas' }),
          el('p', {
            class: 'panel__subtitle',
            text: `${inputs.length} campos — edite qualquer um para repontuar`,
          }),
        ]),
        this.botaoRestaurar,
      ]),

      // `<ul>` e não `<form>`: não há submit. Cada alteração já é a
      // submissão, e um botão "enviar" só adiaria o que a tela existe
      // para mostrar.
      el('ul', { class: 'input-list input-list--form' }, inputs.map((input) => this.item(input))),

      this.rejected?.length ? rejeitados(this.rejected) : null,
      el('span', {
        class: 'anchor anchor--panel anchor--panel-out',
        dataset: { anchor: 'panel:inputs' },
        attrs: { 'aria-hidden': 'true' },
      }),
    ]));
  }
}

customElements.define('flow-input-panel', FlowInputPanel);
