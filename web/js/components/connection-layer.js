import { reduzMovimento, svg } from '../dom.js';
import { anchorId } from '../mappers.js';

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
    clearTimeout(this.fimDaEntrada);
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

  // A retropropagação é o mesmo caminho, percorrido ao contrário. Não há
  // um segundo conjunto de curvas: só a animação inverte, porque é
  // literalmente isso que acontece — o erro volta pelas MESMAS ligações
  // por onde o dado veio.
  set direction(valor) {
    this.canvas?.setAttribute('data-direction', valor === 'backward' ? 'backward' : 'forward');
  }

  // No modo treinamento todas as ligações pulsam. Em uma a cada cinco, o
  // que se vê é um gotejamento; o treino é o momento em que a rede toda
  // está sendo percorrida de uma vez, e a imagem tem que dizer isso.
  set animateAll(valor) {
    this.todas = Boolean(valor);
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
      const path = svg('path', {
        class: 'link',
        d,

        // A ponta de origem fica no atributo para que iluminar o caminho
        // de UM campo seja uma consulta no DOM, e não um segundo desenho.
        'data-from': link.from,

        // O atraso do desenho de entrada cresce com a posição horizontal:
        // as ligações aparecem varrendo da entrada para a saída, na mesma
        // direção em que o dado anda. Espalhar por sorteio faria pipoca.
        'data-enter': (origem.x / hostRect.width).toFixed(3),
      });

      if (link.emphasis) {
        path.classList.add('link--emphasis');
      }

      paths.push(path);

      if (anima && (this.todas || link.animated)) {
        paths.push(svg('path', {
          class: 'link-flow',
          d,
          'data-from': link.from,
          'data-delay': link.delay ?? 0,
        }));
      }
    });

    // Um gradiente só, em coordenadas da tela, atravessando o container
    // da esquerda para a direita: a ligação nasce apagada perto da entrada
    // e chega acesa na saída. Em coordenadas do próprio traço isto
    // falharia — uma ligação horizontal tem caixa de altura zero, e o
    // gradiente relativo a ela simplesmente não pinta.
    const defs = svg('defs');
    const gradiente = svg('linearGradient', {
      id: 'fluxo-ligacao',
      gradientUnits: 'userSpaceOnUse',
      x1: 0, y1: 0, x2: hostRect.width, y2: 0,
    });

    gradiente.append(
      svg('stop', { offset: '0%', 'stop-color': 'var(--color-link)' }),
      svg('stop', { offset: '55%', 'stop-color': 'var(--color-link-mid)' }),
      svg('stop', { offset: '100%', 'stop-color': 'var(--color-link-active)' }),
    );
    defs.append(gradiente);

    this.canvas.replaceChildren(defs, ...paths);

    // O comprimento só existe depois de o `path` estar no documento, e
    // ele é o que define o passo do tracejado — sem isso o ponto que
    // corre pela ligação teria velocidade diferente em cada curva.
    this.canvas.querySelectorAll('.link-flow').forEach((path) => {
      const comprimento = path.getTotalLength();

      path.style.setProperty('--len', `${comprimento}px`);
      path.style.setProperty('--delay', `${path.dataset.delay}s`);
    });

    this.canvas.querySelectorAll('.link').forEach((path) => {
      path.style.setProperty('--len', `${path.getTotalLength()}px`);
      path.style.setProperty('--enter', `${Number(path.dataset.enter) * 420}ms`);
    });

    // O traçado de entrada roda UMA vez. Sem esta trava, cada resize
    // redesenharia a rede do zero na frente de quem só arrastou a janela.
    if (anima && !this.jaEntrou) {
      this.jaEntrou = true;
      this.canvas.classList.add('is-entering');

      // A classe precisa SAIR quando a animação termina. Ela fica no
      // `<svg>`, que sobrevive aos redesenhos; deixada ali, os traços
      // recriados a cada resize herdariam o `stroke-dasharray` e a rede
      // inteira se redesenharia na frente de quem só arrastou a janela.
      clearTimeout(this.fimDaEntrada);
      this.fimDaEntrada = setTimeout(() => {
        this.canvas?.classList.remove('is-entering');
      }, 1400);
    }

    this.aplicarRastro();
  }

  // Iluminar o caminho de um campo: as ligações que saem dele acendem, o
  // resto recua. É a resposta visual à pergunta "para onde vai a idade?",
  // e ela não precisa de um segundo desenho — só de uma classe.
  set trace(campo) {
    this.rastro = campo ? anchorId.input(campo) : null;
    this.aplicarRastro();
  }

  aplicarRastro() {
    if (!this.canvas) {
      return;
    }

    this.canvas.classList.toggle('is-tracing', Boolean(this.rastro));
    this.canvas.querySelectorAll('.link, .link-flow').forEach((path) => {
      path.classList.toggle('is-traced', path.dataset.from === this.rastro);
    });
  }
}

customElements.define('flow-connections', FlowConnections);
