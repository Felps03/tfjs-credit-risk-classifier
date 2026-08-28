import './components/data-processing-flow.js';
import { fetchAnalysis, fetchScore } from './api.js';
import {
  toDecision,
  toFlowData,
  toResults,
  toEvaluation,
  toStats,
  toTraining,
  toTrainingConnections,
} from './mappers.js';
import { fetchAnalysisMock, scoreMock } from './mock.js';

// --------------------------------------------------
// Bootstrap
// --------------------------------------------------
// A única camada que costura tudo: busca, mapeia, entrega ao componente e
// cuida dos estados. Nenhuma das quatro coisas mora dentro da
// visualização, e é o que permite reaproveitá-la em outra tela.

const flow = document.querySelector('data-processing-flow');
const painelStats = document.querySelector('[data-stats]');

// `<dl>` de verdade: cada número é um valor com um termo. Lido por voz
// sai "entradas na rede, 57", que é a mesma frase que a tela mostra.
const renderizarStats = (stats) => {
  if (!painelStats) {
    return;
  }

  painelStats.replaceChildren(...stats.flatMap((stat, indice) => {
    const grupo = document.createElement('div');

    grupo.className = 'stat';
    grupo.style.setProperty('--ordem', String(indice));

    const valor = document.createElement('dd');
    const rotulo = document.createElement('dt');

    valor.className = 'stat__value';
    valor.textContent = stat.valor;
    rotulo.className = 'stat__label';
    rotulo.textContent = stat.rotulo;

    grupo.append(rotulo, valor);

    return [grupo];
  }));
};
const botaoTema = document.querySelector('[data-action="tema"]');
const botaoRecarregar = document.querySelector('[data-action="recarregar"]');

// `?mock=1` desenha a tela sem serviço nenhum no ar. Fora desse ramo, o
// mock não é sequer consultado.
const usarMock = new URLSearchParams(window.location.search).has('mock');

let schemaAtual = null;
let clienteAtual = null;
let scoreAtual = null;

// `preservarCliente` é o que separa os dois botões que recarregam. O
// "Reprocessar" do topo recarrega o PACOTE e repontua quem está no
// formulário: quem editou seis campos e clicou ali queria pontuar de
// novo, não voltar ao exemplo — voltar ao exemplo é o que o "Restaurar
// exemplo" faz, e ele diz isso no rótulo. O "Tentar de novo" do estado de
// erro não preserva nada, porque ali não há nada preservado.
const carregar = async ({ preservarCliente = false } = {}) => {
  const anterior = preservarCliente ? clienteAtual : null;

  flow.state = 'loading';

  try {
    const dtos = usarMock ? await fetchAnalysisMock() : await fetchAnalysis();

    // A simulação só é reaproveitada se ainda couber no contrato que
    // acabou de chegar. Um retreino que mude as colunas invalida o
    // cliente antigo, e insistir nele renderia um 400 — nesse caso o
    // exemplo novo é a resposta certa.
    const cabeNoContrato = anterior !== null
      && Object.keys(dtos.customer).every((campo) => campo in anterior);

    const customer = cabeNoContrato ? anterior : dtos.customer;
    const score = cabeNoContrato
      ? await (usarMock ? scoreMock(customer) : fetchScore(customer))
      : dtos.score;

    const dados = toFlowData({ schema: dtos.schema, score, customer });

    schemaAtual = dtos.schema;
    clienteAtual = customer;
    scoreAtual = score;
    renderizarStats(toStats(dtos.schema));

    // As ligações do modo treinamento saem daqui e não do componente: são
    // a mesma rede com outras pontas, e quem decide topologia é o mapper.
    const treino = toTraining(dtos.schema);
    const avaliacao = toEvaluation(dtos.schema);

    flow.training = treino;
    flow.evaluation = avaliacao;
    ajustarAberturaDoTreino(treino);
    ajustarAberturaDaAvaliacao(avaliacao);
    flow.data = { ...dados, trainingConnections: toTrainingConnections(dados.layers) };
    flow.state = dados.inputs.length === 0 ? 'empty' : 'ready';
  } catch (erro) {
    flow.error = erro;
    flow.state = 'error';
  }
};

// --------------------------------------------------
// Simulação: o formulário repontua a cada alteração
// --------------------------------------------------
// Duas proteções, e as duas importam quando alguém segura uma tecla:
//
//   debounce   sem ele, cada dígito de "9000" vira uma requisição, e as
//              três primeiras pontuam clientes que ninguém quis simular.
//   sequência  o debounce reduz as requisições, não as ordena. Duas em
//              voo podem voltar trocadas, e a resposta ANTIGA sobrescrever
//              a nova — a tela mostraria o resultado de um cliente que já
//              não está no formulário. Só a última pedida escreve.
const ESPERA = 260;

let sequencia = 0;
let agendada = null;

const repontuar = async (customer) => {
  const minha = ++sequencia;

  flow.busy = true;

  try {
    const score = usarMock ? await scoreMock(customer) : await fetchScore(customer);

    if (minha !== sequencia) {
      return;
    }

    scoreAtual = score;
    clienteAtual = customer;
    flow.scoreError = null;
    flow.score = {
      results: toResults(score),
      decision: toDecision(score, schemaAtual),
    };
  } catch (erro) {
    if (minha === sequencia) {
      flow.scoreError = erro;
    }
  } finally {
    if (minha === sequencia) {
      flow.busy = false;
    }
  }
};

flow.addEventListener('flow-customer-change', (evento) => {
  clearTimeout(agendada);
  agendada = setTimeout(() => repontuar(evento.detail.customer), ESPERA);
});

flow.addEventListener('flow-retry', () => carregar());
botaoRecarregar?.addEventListener('click', () => carregar({ preservarCliente: true }));

// --------------------------------------------------
// Troca de modo
// --------------------------------------------------
// Voltar para a análise REMONTA os dados a partir do cliente que estava
// no formulário, não do exemplo. Sem isso, alguém que simulou um cliente,
// foi ver o treino e voltou perderia a simulação — e não teria como saber
// que perdeu, porque a tela pareceria correta.
const botoesModo = document.querySelectorAll('[data-mode]');
const abertura = document.querySelector('.page__lead');
const etiqueta = document.querySelector('[data-kicker]');
const manchete = document.querySelector('[data-headline]');

const ABERTURA = {
  analise: {
    kicker: 'Análise de risco',
    headline: 'Dezenove campos entram. Um número sai.',
    lead: abertura?.textContent.trim() ?? '',
  },
  treinamento: {
    kicker: 'Como a rede aprendeu',

    // Também substituída pelos números do pacote. Ver o comentário abaixo.
    headline: 'Ela errou até parar de melhorar.',
    lead: 'A rede não nasceu sabendo. Ela viu 800 clientes cujo desfecho já era '
      + 'conhecido, chutou, mediu o erro e corrigiu os pesos — época após época. '
      + 'O laço abaixo é o esquema desse procedimento; a curva à direita é o que foi '
      + 'medido enquanto ele rodava.',
  },
  avaliacao: {
    kicker: 'O que ele vale',

    // Substituída pelos números REAIS do pacote assim que ele carrega.
    // Uma manchete que cita percentuais e é escrita à mão vira mentira no
    // primeiro retreino — e este projeto inteiro existe para não fazer
    // isso.
    headline: 'O que a rede acrescentou sobre chutar.',
    lead: 'Toda medida aqui vem de clientes que o treino nunca viu. A acurácia '
      + 'aparece encostada no piso da classe majoritária de propósito: é a distância '
      + 'entre as duas que diz o que o modelo acrescentou. À direita, a auditoria por '
      + 'sexo — a coluna que a rede nunca recebeu, e que por isso mesmo precisa ser '
      + 'medida depois.',
  },
};

const percentual = (valor) => `${(valor * 100).toFixed(0)}%`;

// A manchete e a abertura da avaliação passam a citar o pacote carregado.
// Sem isto elas envelheceriam em silêncio: o texto continuaria dizendo
// "72% contra 70%" enquanto a tela ao lado mostrasse outra coisa.
const ajustarAberturaDoTreino = (treino) => {
  if (!treino) {
    return;
  }

  ABERTURA.treinamento.headline = `Ela errou ${treino.epochs} vezes até parar de melhorar.`;
};

const ajustarAberturaDaAvaliacao = (avaliacao) => {
  if (!avaliacao) {
    return;
  }

  ABERTURA.avaliacao.headline = `Acertar ${percentual(avaliacao.accuracy)} parece bom. `
    + `Chutar acerta ${percentual(avaliacao.baseline)}.`;
  ABERTURA.avaliacao.lead = `Toda medida aqui vem de ${avaliacao.testCustomers} clientes `
    + 'que o treino nunca viu. A acurácia aparece encostada no piso da classe '
    + 'majoritária de propósito: é a distância entre as duas que diz o que o modelo '
    + 'acrescentou. À direita, a auditoria por sexo — a coluna que a rede nunca '
    + 'recebeu, e que por isso mesmo precisa ser medida depois.';
};

const trocarModo = (modo) => {
  const texto = ABERTURA[modo];

  if (texto) {
    if (etiqueta) {
      etiqueta.textContent = texto.kicker;
    }

    if (manchete) {
      manchete.textContent = texto.headline;
    }

    if (abertura) {
      abertura.textContent = texto.lead;
    }
  }

  botoesModo.forEach((botao) => {
    const ativo = botao.dataset.mode === modo;

    botao.setAttribute('aria-checked', String(ativo));

    // O `tabindex` acompanha a marcação: o grupo inteiro é UMA parada de
    // tabulação, e ela cai sempre na opção que está valendo.
    botao.tabIndex = ativo ? 0 : -1;
  });

  if (modo === 'analise' && schemaAtual && clienteAtual && scoreAtual) {
    const dados = toFlowData({
      schema: schemaAtual,
      score: scoreAtual,
      customer: clienteAtual,
    });

    flow.data = { ...dados, trainingConnections: toTrainingConnections(dados.layers) };
  }

  flow.mode = modo;
};

// Num `radiogroup`, escolher É navegar: a seta move o foco e marca a
// opção no mesmo gesto — não existe "focado mas não marcado". Home e End
// vão às pontas, e o `preventDefault` impede que a seta role a página
// enquanto alguém percorre os modos.
const PASSO_DA_SETA = {
  ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1,
};

const irParaModo = (indice) => {
  const alvo = botoesModo[(indice + botoesModo.length) % botoesModo.length];

  trocarModo(alvo.dataset.mode);
  alvo.focus();
};

botoesModo.forEach((botao, indice) => {
  botao.addEventListener('click', () => trocarModo(botao.dataset.mode));

  botao.addEventListener('keydown', (evento) => {
    if (evento.key === 'Home' || evento.key === 'End') {
      evento.preventDefault();
      irParaModo(evento.key === 'Home' ? 0 : botoesModo.length - 1);

      return;
    }

    const passo = PASSO_DA_SETA[evento.key];

    if (passo === undefined) {
      return;
    }

    evento.preventDefault();
    irParaModo(indice + passo);
  });
});

// --------------------------------------------------
// Tema
// --------------------------------------------------
// Sem atributo, o CSS segue o sistema. O botão só grava algo quando
// alguém escolhe — e a escolha sobrevive ao reload porque um tema que se
// perde a cada F5 é pior do que não ter botão.
const CHAVE_TEMA = 'fluxo:tema';

const temaAtual = () => document.documentElement.dataset.theme
  ?? (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');

const aplicarTema = (tema) => {
  document.documentElement.dataset.theme = tema;
  botaoTema?.setAttribute('aria-label', tema === 'dark' ? 'Usar tema claro' : 'Usar tema escuro');
  botaoTema?.setAttribute('aria-pressed', String(tema === 'dark'));
};

const salvo = (() => {
  try {
    return window.localStorage.getItem(CHAVE_TEMA);
  } catch {
    // Janela privada ou armazenamento bloqueado. A página funciona sem.
    return null;
  }
})();

aplicarTema(salvo === 'dark' || salvo === 'light' ? salvo : temaAtual());

botaoTema?.addEventListener('click', () => {
  const proximo = temaAtual() === 'dark' ? 'light' : 'dark';

  aplicarTema(proximo);

  try {
    window.localStorage.setItem(CHAVE_TEMA, proximo);
  } catch {
    // Idem: não conseguir lembrar não impede de trocar.
  }
});

carregar();
