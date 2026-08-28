// --------------------------------------------------
// O reprodutor do treino
// --------------------------------------------------
// Um treino não é uma foto: é um laço que roda dezenas de vezes. Este
// módulo é só o relógio desse laço — ele não sabe desenhar nada, e é por
// isso que a rede e a curva conseguem ficar em sincronia sem uma saber
// da outra: as duas ouvem o mesmo `onChange`.
//
// Cada época passa pelas quatro fases do algoritmo, na ordem em que elas
// realmente acontecem. Os rótulos não são enfeite — são o que transforma
// pontinhos correndo em "isto é retropropagação".

export const FASES = [
  {
    id: 'frente',
    label: 'Passo à frente',
    detail: 'o lote atravessa a rede e vira uma probabilidade',
    ms: 640,
  },
  {
    id: 'erro',
    label: 'Erro medido',
    detail: 'a probabilidade é comparada com o rótulo verdadeiro',
    ms: 420,
  },
  {
    id: 'atras',
    label: 'Passo atrás',
    detail: 'o erro volta pela rede, repartido entre os pesos',
    ms: 640,
  },
  {
    id: 'ajuste',
    label: 'Pesos ajustados',
    detail: 'cada peso anda um passo na direção que reduz o erro',
    ms: 360,
  },
];

export const createPlayer = ({ epochs, onChange, autoplay = true }) => {
  let epoca = 0;
  let fase = 0;
  let timer = null;
  let rodando = false;

  const publicar = () => onChange({
    epoca,
    fase: FASES[fase],
    indiceFase: fase,
    rodando,
    epochs,
  });

  const parar = () => {
    clearTimeout(timer);
    timer = null;
  };

  const passo = () => {
    fase += 1;

    // Fim das quatro fases: a época terminou. Voltar ao início em vez de
    // parar é o certo — um treino de 25 épocas visto uma vez só some
    // antes de alguém entender o que viu.
    if (fase >= FASES.length) {
      fase = 0;
      epoca = (epoca + 1) % epochs;
    }

    publicar();
    agendar();
  };

  const agendar = () => {
    parar();

    if (rodando) {
      timer = setTimeout(passo, FASES[fase].ms);
    }
  };

  const play = () => {
    rodando = true;
    publicar();
    agendar();
  };

  const pause = () => {
    rodando = false;
    parar();
    publicar();
  };

  // Arrastar a barra pausa: quem está procurando uma época específica não
  // quer que ela fuja. A fase vai para "ajuste" porque é o estado de
  // repouso — a época já aconteceu.
  const seek = (valor) => {
    rodando = false;
    parar();
    epoca = Math.max(0, Math.min(epochs - 1, Math.round(valor)));
    fase = FASES.length - 1;
    publicar();
  };

  const toggle = () => (rodando ? pause() : play());

  if (autoplay) {
    play();
  } else {
    publicar();
  }

  return { play, pause, toggle, seek, destroy: parar };
};
