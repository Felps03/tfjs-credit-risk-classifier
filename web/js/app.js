import './components/data-processing-flow.js';
import { fetchAnalysis, fetchScore } from './api.js';
import { toDecision, toFlowData, toResults } from './mappers.js';
import { fetchAnalysisMock, scoreMock } from './mock.js';

// --------------------------------------------------
// Bootstrap
// --------------------------------------------------
// A única camada que costura tudo: busca, mapeia, entrega ao componente e
// cuida dos estados. Nenhuma das quatro coisas mora dentro da
// visualização, e é o que permite reaproveitá-la em outra tela.

const flow = document.querySelector('data-processing-flow');
const botaoTema = document.querySelector('[data-action="tema"]');
const botaoRecarregar = document.querySelector('[data-action="recarregar"]');

// `?mock=1` desenha a tela sem serviço nenhum no ar. Fora desse ramo, o
// mock não é sequer consultado.
const usarMock = new URLSearchParams(window.location.search).has('mock');

let schemaAtual = null;

const carregar = async () => {
  flow.state = 'loading';

  try {
    const dtos = usarMock ? await fetchAnalysisMock() : await fetchAnalysis();
    const dados = toFlowData(dtos);

    schemaAtual = dtos.schema;
    flow.data = dados;
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

flow.addEventListener('flow-retry', carregar);
botaoRecarregar?.addEventListener('click', carregar);

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
