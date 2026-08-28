import { clear, el, svg } from '../dom.js';

// --------------------------------------------------
// <flow-loss-chart> — a curva do treino
// --------------------------------------------------
// As duas linhas são o argumento inteiro do projeto em uma imagem: a loss
// de TREINO continua caindo, a de VALIDAÇÃO para de cair. O ponto em que
// elas se separam é o momento em que o modelo deixa de aprender o
// problema e começa a decorar o conjunto — e é exatamente ali que o early
// stopping corta.
//
// Os números são os do `history` que o `model.fit` devolveu. Nenhum ponto
// é suavizado, interpolado ou escolhido: se a curva sobe e desce, é
// porque subiu e desceu.

const W = 340;
const H = 190;
const PAD = { top: 14, right: 12, bottom: 26, left: 38 };

const escala = (valores) => {
  const min = Math.min(...valores);
  const max = Math.max(...valores);

  // Uma folga de 8% impede que o ponto mais baixo encoste no eixo e o
  // mais alto seja cortado pela borda.
  const folga = (max - min) * 0.08 || 0.01;

  return { min: min - folga, max: max + folga };
};

export class FlowLossChart extends HTMLElement {
  set data(training) {
    this.training = training;
    this.epoca = training ? training.epochs - 1 : 0;
    this.render();
  }

  // Só move o marcador. Redesenhar o gráfico inteiro a cada época faria
  // o navegador refazer duas polylines de 25 pontos vinte e cinco vezes
  // por ciclo, para mudar duas coordenadas.
  set epoch(indice) {
    this.epoca = indice;
    this.moverMarcador();
  }

  ponto(indice, valor) {
    const { epochs } = this.training;
    const largura = W - PAD.left - PAD.right;
    const altura = H - PAD.top - PAD.bottom;
    const x = PAD.left + (epochs === 1 ? largura / 2 : (indice / (epochs - 1)) * largura);
    const y = PAD.top + (1 - (valor - this.y.min) / (this.y.max - this.y.min)) * altura;

    return { x, y };
  }

  linha(valores, classe) {
    return svg('polyline', {
      class: classe,
      points: valores.map((valor, indice) => {
        const { x, y } = this.ponto(indice, valor);

        return `${x.toFixed(1)},${y.toFixed(1)}`;
      }).join(' '),
      'vector-effect': 'non-scaling-stroke',
    });
  }

  moverMarcador() {
    if (!this.training || !this.marcador) {
      return;
    }

    const indice = Math.max(0, Math.min(this.training.epochs - 1, this.epoca));
    const treino = this.ponto(indice, this.training.loss[indice]);
    const validacao = this.ponto(indice, this.training.valLoss[indice]);

    this.marcador.setAttribute('transform', `translate(${treino.x - PAD.left} 0)`);
    this.pontoTreino.setAttribute('cy', treino.y);
    this.pontoValidacao.setAttribute('cy', validacao.y);

    this.leitura.replaceChildren(
      el('span', { class: 'chart__reading-item', text: `época ${indice + 1}` }),
      el('span', {
        class: 'chart__reading-item chart__reading-item--treino',
        text: `treino ${this.training.loss[indice].toFixed(4)}`,
      }),
      el('span', {
        class: 'chart__reading-item chart__reading-item--validacao',
        text: `validação ${this.training.valLoss[indice].toFixed(4)}`,
      }),
    );
  }

  grafico() {
    const { loss, valLoss, best, epochs } = this.training;

    this.y = escala([...loss, ...valLoss]);

    const canvas = svg('svg', {
      class: 'chart',
      viewBox: `0 0 ${W} ${H}`,
      preserveAspectRatio: 'xMidYMid meet',
      role: 'img',
      'aria-label': `Curva de perda ao longo de ${epochs} épocas. A perda de treino cai `
        + `de ${loss[0].toFixed(3)} a ${loss[epochs - 1].toFixed(3)}. A de validação `
        + `chega ao melhor valor, ${Math.min(...valLoss).toFixed(3)}, na época ${best + 1}, `
        + 'e não melhora mais depois disso.',
    });

    // Eixos: só as duas linhas que dizem onde é o chão e a esquerda.
    canvas.append(
      svg('line', {
        class: 'chart__axis',
        x1: PAD.left, y1: H - PAD.bottom, x2: W - PAD.right, y2: H - PAD.bottom,
      }),
      svg('line', {
        class: 'chart__axis', x1: PAD.left, y1: PAD.top, x2: PAD.left, y2: H - PAD.bottom,
      }),
    );

    // A época da melhor validação. É a linha que explica a parada.
    const melhor = this.ponto(best, valLoss[best]);

    // O rótulo troca de lado perto da borda direita — e o deslocamento
    // troca de sinal junto. Com `text-anchor: end` e `+4`, o texto saía
    // pela borda em vez de recuar dela.
    const aDireita = best > epochs * 0.7;

    canvas.append(
      svg('line', {
        class: 'chart__best',
        x1: melhor.x, y1: PAD.top, x2: melhor.x, y2: H - PAD.bottom,
      }),
      svg('text', {
        class: 'chart__best-label',
        x: melhor.x + (aDireita ? -4 : 4),
        y: PAD.top + 8,
        'text-anchor': aDireita ? 'end' : 'start',
      }),
    );
    canvas.querySelector('.chart__best-label').textContent = `melhor: ${best + 1}`;

    canvas.append(
      this.linha(loss, 'chart__line chart__line--treino'),
      this.linha(valLoss, 'chart__line chart__line--validacao'),
    );

    // O marcador móvel: uma vertical e dois pontos, agrupados para que
    // mover a época seja UM `translate` em vez de três atributos.
    this.pontoTreino = svg('circle', { class: 'chart__dot chart__dot--treino', cx: PAD.left, r: 3.4 });
    this.pontoValidacao = svg('circle', {
      class: 'chart__dot chart__dot--validacao', cx: PAD.left, r: 3.4,
    });
    this.marcador = svg('g', { class: 'chart__marker' });
    this.marcador.append(
      svg('line', { x1: PAD.left, y1: PAD.top, x2: PAD.left, y2: H - PAD.bottom }),
      this.pontoTreino,
      this.pontoValidacao,
    );
    canvas.append(this.marcador);

    // Rótulos dos eixos: o mínimo e o máximo em y, a primeira e a última
    // época em x. Um gráfico pequeno não comporta mais do que isso.
    const eixo = (x, y, texto, anchor = 'middle') => {
      const node = svg('text', { class: 'chart__tick', x, y, 'text-anchor': anchor });

      node.textContent = texto;

      return node;
    };

    canvas.append(
      eixo(PAD.left - 5, PAD.top + 4, this.y.max.toFixed(2), 'end'),
      eixo(PAD.left - 5, H - PAD.bottom + 3, this.y.min.toFixed(2), 'end'),
      eixo(PAD.left, H - PAD.bottom + 15, '1'),
      eixo(W - PAD.right, H - PAD.bottom + 15, String(epochs), 'end'),
    );

    return canvas;
  }

  render() {
    clear(this);

    if (!this.training) {
      this.append(el('section', { class: 'panel' }, [
        el('header', { class: 'panel__header' }, [
          el('h2', { class: 'panel__title', text: 'A curva do treino' }),
        ]),
        el('p', {
          class: 'panel__empty',
          text: 'Este pacote foi salvo antes de o histórico passar a ser gravado. '
            + 'Rode `npm start` para treinar de novo — a curva aparece sozinha.',
        }),
      ]));

      return;
    }

    this.leitura = el('div', { class: 'chart__reading', attrs: { 'aria-live': 'polite' } });

    this.append(el('section', {
      class: 'panel panel--chart',
      attrs: { 'aria-labelledby': 'titulo-curva' },
    }, [
      el('header', { class: 'panel__header' }, [
        el('h2', { class: 'panel__title', text: 'A curva do treino', id: 'titulo-curva' }),
        el('p', {
          class: 'panel__subtitle',
          text: 'perda por época — medida, não ilustrada',
        }),
      ]),
      el('div', { class: 'chart__legend' }, [
        el('span', { class: 'chart__key chart__key--treino', text: 'treino' }),
        el('span', { class: 'chart__key chart__key--validacao', text: 'validação' }),
      ]),
      this.grafico(),
      this.leitura,
      el('p', {
        class: 'chart__note',
        text: 'A de treino continua caindo; a de validação para. A distância entre as '
          + 'duas é o modelo decorando em vez de aprender — e é ali que o treino foi cortado.',
      }),
      el('span', {
        class: 'anchor anchor--stage anchor--stage-in',
        dataset: { anchor: 'stage:right' },
        attrs: { 'aria-hidden': 'true' },
      }),
    ]));

    this.moverMarcador();
  }
}

customElements.define('flow-loss-chart', FlowLossChart);
