import { ICONS } from './domain.js';

// --------------------------------------------------
// Utilitários de DOM
// --------------------------------------------------
// Três funções, nenhuma biblioteca. Existem para que os componentes
// montem elementos em vez de concatenar HTML: `innerHTML` com valor
// vindo da API é o caminho mais curto para injeção, e aqui todo texto
// entra por `textContent`.

export const el = (tag, props = {}, children = []) => {
  const node = document.createElement(tag);
  const { class: className, text, dataset = {}, attrs = {}, ...rest } = props;

  if (className) {
    node.className = className;
  }

  if (text !== undefined) {
    node.textContent = text;
  }

  Object.assign(node, rest);
  Object.entries(dataset).forEach(([key, value]) => { node.dataset[key] = value; });
  Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, value));

  children.filter(Boolean).forEach((child) => node.append(child));

  return node;
};

const SVG_NS = 'http://www.w3.org/2000/svg';

export const svg = (tag, attrs = {}) => {
  const node = document.createElementNS(SVG_NS, tag);

  Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, value));

  return node;
};

// O ícone é decorativo: quem carrega o significado é o rótulo ao lado.
// `aria-hidden` evita que um leitor de tela anuncie "imagem" antes de
// cada linha da lista.
export const icon = (name, className = 'icon') => {
  const node = svg('svg', {
    class: className,
    viewBox: '0 0 24 24',
    fill: 'none',
    stroke: 'currentColor',
    'stroke-width': '1.6',
    'stroke-linecap': 'round',
    'stroke-linejoin': 'round',
    'aria-hidden': 'true',
    focusable: 'false',
  });

  (ICONS[name] ?? ICONS.field).forEach((d) => node.append(svg('path', { d })));

  return node;
};

export const clear = (node) => {
  node.replaceChildren();

  return node;
};

// Uma consulta única, respondida uma vez por carga: a preferência não
// muda no meio da sessão, e reler a media query a cada quadro custaria
// mais do que ela vale.
export const reduzMovimento = () =>
  window.matchMedia('(prefers-reduced-motion: reduce)').matches;
