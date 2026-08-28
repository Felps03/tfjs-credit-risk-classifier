// --------------------------------------------------
// Camada HTTP
// --------------------------------------------------
// A única parte da tela que sabe que existe uma API. Ela devolve os DTOs
// exatamente como o serviço os escreve — sem renomear campo, sem calcular
// nada — porque quem adapta é `mappers.js`. É essa fronteira que permite
// mudar a resposta do `POST /risk-score` sem tocar em componente nenhum.

export class ApiError extends Error {
  constructor(message, { status = 0, details = [] } = {}) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.details = details;
  }
}

const request = async (path, options = {}) => {
  let response;

  try {
    response = await fetch(path, options);
  } catch (cause) {
    // Rede fora, serviço no ar sem o modelo, porta errada: o `fetch` só
    // rejeita, sem status. Sem esta mensagem, a tela mostraria
    // "Failed to fetch", que não diz a ninguém o que fazer a seguir.
    throw new ApiError('Não foi possível falar com o serviço. Ele está no ar?', { status: 0 });
  }

  const body = await response.json().catch(() => null);

  if (!response.ok) {
    throw new ApiError(body?.error ?? `O serviço respondeu ${response.status}.`, {
      status: response.status,
      details: body?.details ?? [],
    });
  }

  return body;
};

export const fetchSchema = () => request('/schema');

export const fetchScore = (customer) => request('/risk-score', {
  method: 'POST',
  headers: { 'content-type': 'application/json' },
  body: JSON.stringify(customer),
});

// O contrato traz um cliente de exemplo que o próprio serviço aceita, e
// é ele que é pontuado. Buscar o esquema primeiro não é um passo a mais:
// é o que faz a tela funcionar depois de um retreino que mudou as
// colunas, sem ninguém precisar editar um payload aqui dentro.
export const fetchAnalysis = async () => {
  const schema = await fetchSchema();
  const score = await fetchScore(schema.example);

  return { schema, score, customer: schema.example };
};
