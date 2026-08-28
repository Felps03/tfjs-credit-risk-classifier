import { reduzMovimento, svg } from '../dom.js';

// --------------------------------------------------
// <flow-connections> — as ligações
// --------------------------------------------------
// Um único SVG absoluto por cima do container inteiro. Nenhuma linha é
// uma `div` girada: as ligações precisam sair de um ponto e chegar a
// outro com precisão, e curvas são curvas.
//
// As coordenadas NÃO são escritas à mão. Cada ponta procura no DOM o
// elemento com o `data-anchor` correspondente, mede onde ele está com
// `getBoundingClientRect()` e converte para o sistema do container. É o
// que faz o desenho continuar certo depois de um resize, de uma troca de
// orientação ou de a fonte terminar de carregar — três coisas que
// acontecem depois do primeiro render e que uma coordenada fixa não
// sobreviveria.
//
// O redesenho é agendado em `requestAnimationFrame`: o `ResizeObserver`
// dispara muitas vezes durante um arrasto de janela, e medir a cada
// disparo custaria um reflow por evento.

// Profundidade máxima ao seguir `data-fallback`. Um ponto escondido
// (o do item, no mobile) delega para o ponto do card; se ESSE também
// estiver escondido, algo está errado na folha de estilo e é melhor não
// desenhar do que entrar em laço.
const SALTOS = 2;

const centro = (rect, hostRect) => ({
  x: rect.left - hostRect.left + rect.width / 2,
  y: rect.top - hostRect.top + rect.height / 2,
});

const invisivel = (rect) => rect.width === 0 && rect.height === 0;

// A curva. Ela dobra no eixo em que a distância é maior, e é só isso que
// o layout vertical do mobile precisa: as mesmas ligações, medidas nos
// mesmos pontos, passam a sair por baixo em vez de pela direita porque
// os pontos mudaram de lugar — não porque exista um segundo código.
const curva = ({ x: x1, y: y1 }, { x: x2, y: y2 }) => {
  const dx = x2 - x1;
  const dy = y2 - y1;

  if (Math.abs(dx) >= Math.abs(dy)) {
    const c = Math.max(24, Math.abs(dx) * 0.42);

    return `M ${x1} ${y1} C ${x1 + c} ${y1}, ${x2 - c} ${y2}, ${x2} ${y2}`;
  }

  const c = Math.max(24, Math.abs(dy) * 0.42);

  return `M ${x1} ${y1} C ${x1} ${y1 + c}, ${x2} ${y2 - c}, ${x2} ${y2}`;
};

export class FlowConnections extends HTMLElement {
  constructor() {
    super();
    this.connections = [];
    this.pendente = null;
  }

  connectedCallback() {
    this.setAttribute('aria-hidden', 'true');

    this.canvas = svg('svg', { class: 'connections', focusable: 'false' });
    this.append(this.canvas);

    this.host = this.parentElement;
    this.observer = new ResizeObserver(() => this.agendar());
    this.observar();

    // A fonte troca depois do primeiro layout e move tudo alguns pixels.
    document.fonts?.ready.then(() => this.agendar());
    window.addEventListener('resize', this.agendar);
  }

  disconnectedCallback() {
    this.observer?.disconnect();
    window.removeEventListener('resize', this.agendar);
    cancelAnimationFrame(this.pendente);
  }

  // O container inteiro E cada coluna. Só o container não basta: uma
  // coluna pode reorganizar por dentro — a lista de entradas crescendo,
  // por exemplo — sem que a largura de fora mude um pixel.
  observar() {
    if (!this.observer || !this.host) {
      return;
    }

    this.observer.disconnect();
    this.observer.observe(this.host);
    this.host.querySelectorAll('.flow__stage').forEach((stage) => this.observer.observe(stage));
  }

  set data(connections) {
    this.connections = connections ?? [];
    this.observar();
    this.agendar();
  }

  agendar = () => {
    cancelAnimationFrame(this.pendente);
    this.pendente = requestAnimationFrame(() => this.draw());
  };

  ponto(id, hostRect, saltos = SALTOS) {
    const node = this.host?.querySelector(`[data-anchor="${CSS.escape(id)}"]`);

    if (!node) {
      return null;
    }

    const rect = node.getBoundingClientRect();

    if (!invisivel(rect)) {
      return centro(rect, hostRect);
    }

    return saltos > 0 && node.dataset.fallback
      ? this.ponto(node.dataset.fallback, hostRect, saltos - 1)
      : null;
  }

  draw() {
    if (!this.host || !this.canvas) {
      return;
    }

    const hostRect = this.host.getBoundingClientRect();

    if (hostRect.width === 0) {
      return;
    }

    this.canvas.setAttribute('viewBox', `0 0 ${hostRect.width} ${hostRect.height}`);
    this.canvas.setAttribute('width', hostRect.width);
    this.canvas.setAttribute('height', hostRect.height);

    const anima = !reduzMovimento();
    const desenhadas = new Set();
    const paths = [];

    this.connections.forEach((link) => {
      const origem = this.ponto(link.from, hostRect);
      const destino = this.ponto(link.to, hostRect);

      if (!origem || !destino) {
        return;
      }

      // No mobile dezenove ligações colapsam no mesmo par de pontos.
      // Desenhar as dezenove empilharia traços idênticos e deixaria a
      // linha escura demais — o feixe vira um traço só.
      const chave = `${Math.round(origem.x)},${Math.round(origem.y)}-${Math.round(destino.x)},${Math.round(destino.y)}`;

      if (desenhadas.has(chave)) {
        return;
      }

      desenhadas.add(chave);

      const d = curva(origem, destino);
      const path = svg('path', { class: 'link', d });

      if (link.emphasis) {
        path.classList.add('link--emphasis');
      }

      paths.push(path);

      if (anima && link.animated) {
        paths.push(svg('path', { class: 'link-flow', d, 'data-delay': link.delay ?? 0 }));
      }
    });

    this.canvas.replaceChildren(...paths);

    // O comprimento só existe depois de o `path` estar no documento, e
    // ele é o que define o passo do tracejado — sem isso o ponto que
    // corre pela ligação teria velocidade diferente em cada curva.
    this.canvas.querySelectorAll('.link-flow').forEach((path) => {
      const comprimento = path.getTotalLength();

      path.style.setProperty('--len', `${comprimento}px`);
      path.style.setProperty('--delay', `${path.dataset.delay}s`);
    });
  }
}

customElements.define('flow-connections', FlowConnections);
